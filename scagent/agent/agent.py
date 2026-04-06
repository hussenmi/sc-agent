"""
Main agent class for autonomous single-cell analysis.

Uses Claude or OpenAI API with tools to perform single-cell analysis tasks.
Returns structured JSON from tools for reliable LLM reasoning.
Creates run directories with manifests for reproducibility.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
import logging
from datetime import datetime
import re

from .codex_bridge import CODEX_DECISION_SCHEMA, CodexCLIClient, CodexCLIError
from .tools import get_tools, get_openai_tools, process_tool_call
from .prompts import SYSTEM_PROMPT
from .run_manager import RunManager, create_run
from .decision_policy import (
    decision_for_clustering_selection,
)
from .world_state import AgentWorldState, artifact_id_from_path

logger = logging.getLogger(__name__)

Provider = Literal["anthropic", "openai", "groq", "codex"]

ACTION_TOOL_NAMES = {
    "run_qc",
    "normalize_and_hvg",
    "run_dimred",
    "run_clustering",
    "compare_clusterings",
    "run_celltypist",
    "run_scimilarity",
    "run_batch_correction",
    "run_deg",
    "run_gsea",
    "save_data",
    "run_code",
    "install_package",
}

INSPECTION_TOOL_NAMES = {
    "inspect_data",
    "inspect_session",
    "list_artifacts",
    "get_cluster_sizes",
    "get_top_markers",
    "summarize_qc_metrics",
    "get_celltypes",
    "list_obs_columns",
    "review_figure",
    "review_artifact",
    "inspect_run_state",
    "inspect_workspace",
    "web_search_docs",
    "search_papers",
    "fetch_url",
    "web_search",
    "research_findings",
}

# Load .env file if present
def _load_dotenv():
    """Load .env file from current directory or package root."""
    try:
        from dotenv import load_dotenv
        # Try multiple locations
        search_paths = [
            Path.cwd() / ".env",
            Path(__file__).parent.parent.parent / ".env",  # scagent/agent -> scagent -> project root
            Path(os.environ.get("SCAGENT_HOME", "")) / ".env",
        ]
        for path in search_paths:
            if path.exists():
                load_dotenv(path, override=True)
                logger.info(f"Loaded config from {path}")
                return True
    except ImportError:
        pass  # python-dotenv not installed
    return False

_load_dotenv()


class SCAgent:
    """
    Autonomous single-cell RNA-seq analysis agent.

    Uses Claude, OpenAI-compatible APIs, or Codex CLI to analyze single-cell data
    following lab best practices.
    All tool calls return structured JSON for reliable LLM reasoning.
    Optionally creates run directories with manifests for reproducibility.

    Parameters
    ----------
    provider : str, default "anthropic"
        LLM provider: "anthropic", "openai", "groq", or "codex".
    api_key : str, optional
        API key. If not provided, reads from ANTHROPIC_API_KEY or OPENAI_API_KEY.
    model : str, optional
        Model to use. Defaults depend on provider.
    verbose : bool, default True
        Print agent outputs.
    collaborative : bool, default True
        Pause at major checkpoints, summarize findings, and ask before consequential steps.
    create_run_dir : bool, default True
        Create structured run directory with manifest.
    output_dir : str, default "."
        Base directory for run outputs.
    save_checkpoints : bool, default False
        Save intermediate checkpoint h5ad files. Disabled by default.

    Examples
    --------
    >>> # Using Anthropic (default)
    >>> agent = SCAgent()
    >>> result = agent.analyze("QC and cluster this PBMC data", data_path="pbmc.h5")

    >>> # Using OpenAI
    >>> agent = SCAgent(provider="openai")
    >>> result = agent.analyze("QC and cluster this PBMC data", data_path="pbmc.h5")
    """

    def __init__(
        self,
        provider: Optional[Provider] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = True,
        collaborative: bool = True,
        create_run_dir: bool = True,
        output_dir: str = ".",
        save_checkpoints: bool = False,
    ):
        # Use environment defaults if not specified
        if provider is None:
            provider = os.environ.get("SCAGENT_PROVIDER", "anthropic")
        if model is None:
            if provider == "codex":
                # Prefer a Codex-specific override, but let SCAGENT_MODEL keep
                # working for users who already configure one model in .env.
                model = os.environ.get("SCAGENT_CODEX_MODEL") or os.environ.get("SCAGENT_MODEL")
            else:
                model = os.environ.get("SCAGENT_MODEL")  # None = use provider default
        if base_url is None:
            base_url = os.environ.get("SCAGENT_BASE_URL")  # For OpenAI-compatible APIs

        self.provider = provider
        self.verbose = verbose
        self.collaborative = collaborative
        self.create_run_dir = create_run_dir
        self.output_dir = output_dir
        self.save_checkpoints = save_checkpoints
        self.adata = None
        self.run_manager: Optional[RunManager] = None
        self.world_state = AgentWorldState()
        self.biological_context: Optional[Dict[str, Any]] = None
        self._pending_image: Optional[Dict[str, str]] = None  # For vision support
        self._conversation_history: List[Dict[str, Any]] = []  # For interactive mode
        self._active_request: str = ""
        self._active_request_is_followup: bool = False
        self._interaction_state: Dict[str, List[Dict[str, Any]]] = {
            "shown_figures": [],
            "reviewed_figures": [],
            "asked_questions": [],
        }
        self._pending_checkpoint: Optional[Dict[str, Any]] = None

        if provider == "anthropic":
            self._init_anthropic(api_key, model)
        elif provider == "openai":
            self._init_openai(api_key, model, base_url)
        elif provider == "codex":
            self._init_codex(model)
        elif provider == "groq":
            # Groq uses OpenAI-compatible API
            self._init_openai(
                api_key or os.environ.get("GROQ_API_KEY"),
                model or "llama-3.3-70b-versatile",
                base_url or "https://api.groq.com/openai/v1"
            )
            self.provider = "groq"  # Keep track of actual provider
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'anthropic', 'openai', 'groq', or 'codex'."
            )

    def _init_anthropic(self, api_key: Optional[str], model: Optional[str]):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Install with: pip install anthropic")

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        self.client = Anthropic(api_key=api_key)
        self.model = model or "claude-sonnet-4-20250514"
        self.tools = get_tools()

    def _init_openai(self, api_key: Optional[str], model: Optional[str], base_url: Optional[str] = None):
        """Initialize OpenAI-compatible client (works with OpenAI, Groq, Together, etc.)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No API key provided. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        # Support custom base URLs for OpenAI-compatible APIs (Groq, Together, etc.)
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model or "gpt-4o"
        self.tools = get_openai_tools()

    def _init_codex(self, model: Optional[str]):
        """Initialize Codex CLI bridge for ChatGPT-login-backed runs."""
        codex_model = model or os.environ.get("SCAGENT_CODEX_MODEL")
        self.client = CodexCLIClient(model=codex_model, cwd=os.getcwd())
        self.model = codex_model or "codex-default"
        self.tools = get_openai_tools()

    def _print(self, message: str, style: str = None, markdown: bool = False):
        """Print message if verbose using rich formatting.

        If markdown=True or message contains markdown patterns, renders as markdown.
        """
        if self.verbose:
            from rich.console import Console
            from rich.markdown import Markdown
            console = Console()

            # Auto-detect markdown if not explicitly set
            if not markdown and message:
                # Check for common markdown patterns
                md_patterns = ['**', '##', '- **', '```', '1. ', '2. ', '3. ']
                if any(p in message for p in md_patterns):
                    markdown = True

            if markdown and message:
                console.print(Markdown(message))
            elif style:
                console.print(message, style=style)
            else:
                console.print(message)

    def _print_thinking(self, message: str):
        """Print agent thinking/reasoning."""
        if self.verbose:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(f"[dim]💭[/dim] [italic]{message}[/italic]")

    def _print_error(self, message: str):
        """Print error message."""
        if self.verbose:
            from rich.console import Console
            console = Console()
            console.print(f"[red]✗ Error:[/red] {message}")

    def _print_success(self, message: str):
        """Print success message."""
        if self.verbose:
            from rich.console import Console
            console = Console()
            console.print(f"[green]✓[/green] {message}")

    def _runtime_guidance(self) -> str:
        """Build compact runtime guidance so the model can avoid repeating itself."""
        payload = self.world_state.snapshot()
        payload["is_followup"] = self._active_request_is_followup
        payload["shown_figures"] = self._interaction_state["shown_figures"][-5:]
        payload["reviewed_figures"] = self._interaction_state["reviewed_figures"][-5:]
        payload["recent_questions"] = self._interaction_state["asked_questions"][-3:]
        payload["pending_checkpoint"] = self._pending_checkpoint
        return json.dumps(payload, indent=2)

    def _is_action_tool(self, tool_name: str) -> bool:
        return tool_name in ACTION_TOOL_NAMES

    def _is_inspection_tool(self, tool_name: str) -> bool:
        return tool_name in INSPECTION_TOOL_NAMES

    def _checkpoint_artifact_paths(self, result_data: Dict[str, Any]) -> List[str]:
        paths: List[str] = []
        for artifact in result_data.get("artifacts_created", []) or []:
            path = artifact.get("path")
            if path:
                paths.append(path)
        for key in ("output_path", "figure_path"):
            path = result_data.get(key)
            if path:
                paths.append(path)
        for figure_path in result_data.get("figures", []) or []:
            if figure_path:
                paths.append(figure_path)
        deduped: List[str] = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped[:5]

    def _set_pending_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._pending_checkpoint = checkpoint
        if self.run_manager:
            self.run_manager.append_event("checkpoint_pending", checkpoint)

    def _checkpoint_options(
        self,
        entries: List[tuple[str, str]],
    ) -> tuple[List[str], List[str]]:
        options = [label for label, _ in entries]
        actions = [action for _, action in entries]
        return options, actions

    def _clear_pending_checkpoint(self, user_response: Optional[str] = None) -> None:
        if self._pending_checkpoint and self.run_manager:
            payload = dict(self._pending_checkpoint)
            if user_response is not None:
                payload["user_response"] = user_response
            self.run_manager.append_event("checkpoint_resolved", payload)
        self._pending_checkpoint = None

    def _run_nested_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        result_json = self._execute_tool(tool_name, tool_input)
        try:
            return json.loads(result_json)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "tool": tool_name,
                "message": "Nested tool execution returned invalid JSON.",
            }

    def _execute_checkpoint_action(
        self,
        selected_action: str,
        checkpoint: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not selected_action or selected_action == "custom":
            return None

        action_inputs = checkpoint.get("action_inputs", {}) or {}
        steps: List[Dict[str, Any]] = []

        def run_step(tool_name: str, tool_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            payload = self._run_nested_tool(tool_name, tool_input or {})
            steps.append(
                {
                    "tool": tool_name,
                    "status": payload.get("status"),
                    "summary": (payload.get("state_delta") or {}).get("summary", payload.get("message", "")),
                    "checkpoint_required": payload.get("checkpoint_required", False),
                }
            )
            return payload

        if selected_action == "run_qc_apply":
            qc_input = dict(action_inputs.get("run_qc_apply", {}))
            qc_input["preview_only"] = False
            run_step("run_qc", qc_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "run_normalize_and_hvg":
            normalize_input = dict(action_inputs.get("run_normalize_and_hvg", {}))
            run_step("normalize_and_hvg", normalize_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "run_dimred":
            dimred_input = dict(action_inputs.get("run_dimred", {}))
            run_step("run_dimred", dimred_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "run_annotation":
            annotation_input = dict(action_inputs.get("run_annotation", {}))
            run_step("run_celltypist", annotation_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "run_deg":
            deg_input = dict(action_inputs.get("run_deg", {}))
            run_step("run_deg", deg_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "save_data":
            save_input = dict(action_inputs.get("save_data", {}))
            if not save_input.get("output_path"):
                if self.run_manager:
                    save_input["output_path"] = str(self.run_manager.run_dir / "final_analyzed.h5ad")
                else:
                    save_input["output_path"] = "final_analyzed.h5ad"
            run_step("save_data", save_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "compare_clusterings":
            compare_input = dict(action_inputs.get("compare_clusterings", {}))
            if "resolutions" not in compare_input:
                compare_input["resolutions"] = [0.5, 1.0, 1.5]
            compare_input.setdefault("generate_figures", True)
            run_step("compare_clusterings", compare_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "run_clustering":
            clustering_input = dict(action_inputs.get("run_clustering", {}))
            run_step("run_clustering", clustering_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "render_plain_umap":
            figure_input = dict(action_inputs.get("render_plain_umap", {}))
            figure_input.setdefault("plot_type", "umap")
            figure_input.setdefault("color_by", None)
            figure_input.setdefault("include_image", True)
            run_step("generate_figure", figure_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "list_plot_colors":
            run_step("list_obs_columns", {})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "inspect_existing_state":
            run_step("inspect_session", {"include_history": True})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "review_post_qc_state":
            run_step("inspect_session", {"include_history": True})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "review_qc_artifacts":
            artifact_paths = checkpoint.get("artifacts", [])
            if artifact_paths:
                run_step(
                    "review_artifact",
                    {
                        "artifact_path": artifact_paths[0],
                        "question": "Summarize the key QC issues in this artifact before filtering.",
                    },
                )
            else:
                run_step("inspect_session", {"include_history": True})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "review_corrected_embedding":
            artifact_paths = checkpoint.get("artifacts", [])
            if artifact_paths:
                run_step(
                    "review_artifact",
                    {
                        "artifact_path": artifact_paths[0],
                        "question": "Review this corrected embedding artifact and summarize whether batch structure still dominates.",
                    },
                )
            else:
                run_step("inspect_session", {"include_history": True})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "review_cluster_markers":
            cluster_key = self.world_state.data_summary.get("cluster_key") or "leiden"
            run_step("get_cluster_sizes", {"cluster_key": cluster_key})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "review_annotation_quality":
            annotation_key = (self._current_capabilities().get("annotation_keys") or [None])[0]
            run_step("get_celltypes", {"annotation_key": annotation_key} if annotation_key else {})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "review_deg_results":
            run_step("inspect_session", {"include_history": True})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "inspect_batch_mixing":
            run_step("inspect_session", {"include_history": True})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "restart_from_raw":
            run_step("inspect_session", {"include_history": True})
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "promote_primary_clustering":
            run_step("inspect_session", {"include_history": True})
            return {"selected_action": selected_action, "steps": steps}

        return None

    # Tools that should NOT be blocked by pending checkpoints - they're orthogonal or flexible
    CHECKPOINT_EXEMPT_TOOLS = {
        "run_code",  # Flexible fallback - always allow
        "inspect_data",
        "inspect_session",
        "list_artifacts",
        "get_cluster_sizes",
        "get_top_markers",
        "summarize_qc_metrics",
        "get_celltypes",
        "list_obs_columns",
        "review_figure",
        "review_artifact",
        "generate_figure",  # Visualization doesn't change state
        "save_data",  # Saving is always ok
        "web_search_docs",
        "search_papers",
        "fetch_url",
        "web_search",
        "research_findings",
    }

    def _checkpoint_context_for_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Return checkpoint context without blocking the tool call."""
        if not self._pending_checkpoint:
            return None
        return {
            "pending_decision": self._pending_checkpoint.get("kind", "unknown"),
            "question": self._pending_checkpoint.get("question", ""),
            "note": "A decision point exists. You may proceed if this action addresses it or is orthogonal.",
        }

    def _blocked_by_checkpoint_result(self, tool_name: str) -> str:
        checkpoint = self._pending_checkpoint or {}
        return json.dumps(
            {
                "status": "error",
                "tool": tool_name,
                "message": (
                    "A collaborative checkpoint is pending. Resolve it before running another "
                    "state-changing step."
                ),
                "pending_checkpoint": checkpoint,
                "required_next_action": "ask_user",
            },
            indent=2,
        )

    def _build_checkpoint_payload(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.collaborative or result_data.get("status") != "ok":
            return None

        checkpoint: Optional[Dict[str, Any]] = None
        artifacts = self._checkpoint_artifact_paths(result_data)

        if tool_name == "run_qc":
            if tool_input.get("preview_only", False):
                mt_threshold = tool_input.get("mt_threshold", "auto")
                # Get doublet count from the correct location
                qc_decisions = result_data.get("qc_decisions", {})
                doublet_count = int(qc_decisions.get("doublet_detection", {}).get("cells_flagged", 0))
                high_mt_cells = int(qc_decisions.get("mt_threshold", {}).get("cells_flagged", 0))
                before_cells = result_data.get("before", {}).get("n_cells", 0)
                # Projected = before - high_mt (doublets are flagged but not auto-removed)
                projected_cells = before_cells - high_mt_cells if before_cells else "?"
                question = (
                    "QC preview is complete. I summarized the proposed removals and saved the QC figures. "
                    "What should I do next?"
                )
                options, option_actions = self._checkpoint_options([
                    (
                        f"Apply the proposed QC filters ({projected_cells} projected cells retained)",
                        "run_qc_apply",
                    ),
                    (
                        f"Adjust the mitochondrial threshold from {mt_threshold} before applying QC",
                        "adjust_qc_thresholds",
                    ),
                    (
                        "Inspect the QC figures and flagged doublet/high-MT cells in more detail first",
                        "review_qc_artifacts",
                    ),
                    ("Something else", "custom"),
                ])
                summary = (
                    f"QC preview: {high_mt_cells} high-MT cells to remove, {doublet_count} doublets flagged. "
                    f"Estimated {projected_cells} cells retained after MT filtering."
                )
                checkpoint = {
                    "kind": "qc_preview",
                    "question": question,
                    "options": options,
                    "default": options[0],
                    "decision_key": "qc_next_step",
                    "summary": summary,
                    "recommendation": options[0],
                    "option_actions": option_actions,
                    "action_inputs": {
                        "run_qc_apply": {
                            "mt_threshold": tool_input.get("mt_threshold"),
                            "min_cells": tool_input.get("min_cells"),
                            "remove_ribo": tool_input.get("remove_ribo", True),
                            "remove_mt": tool_input.get("remove_mt", False),
                            "detect_doublets_flag": tool_input.get("detect_doublets_flag", True),
                            "batch_key": tool_input.get("batch_key"),
                        }
                    },
                    "artifacts": artifacts,
                }

            else:
                retained = result_data.get("after", {}).get("n_cells", "?")
                summary = f"QC filtering is complete and retained {retained} cells."
                options, option_actions = self._checkpoint_options([
                    ("Normalize and select HVGs", "run_normalize_and_hvg"),
                    ("Review the post-QC state and saved artifacts before continuing", "review_post_qc_state"),
                    ("Something else", "custom"),
                ])
                checkpoint = {
                    "kind": "qc_applied",
                    "question": "QC filtering is complete. What should I do next?",
                    "options": options,
                    "default": options[0],
                    "decision_key": "post_qc_next_step",
                    "summary": summary,
                    "recommendation": options[0],
                    "option_actions": option_actions,
                    "action_inputs": {
                        "run_normalize_and_hvg": {},
                    },
                    "artifacts": artifacts,
                }

        elif tool_name == "normalize_and_hvg":
            summary = "Normalization and HVG selection are complete."
            if result_data.get("n_hvg") is not None:
                summary = f"Normalization and HVG selection are complete ({result_data['n_hvg']} HVGs selected)."
            options, option_actions = self._checkpoint_options([
                ("Compute PCA, neighbors, and UMAP", "run_dimred"),
                ("Inspect the normalized dataset state before computing embeddings", "inspect_existing_state"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "normalized",
                "question": "Normalization and HVG selection are complete. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "normalized_next_step",
                "summary": summary,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {
                    "run_dimred": {},
                },
                "artifacts": artifacts,
            }

        elif tool_name == "run_dimred":
            summary = "PCA, neighbors, and UMAP are complete."
            options, option_actions = self._checkpoint_options([
                ("Run clustering on the current embedding", "run_clustering"),
                ("Review the embedding outputs before clustering", "review_corrected_embedding"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "dimred",
                "question": "Dimensionality reduction is complete. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "dimred_next_step",
                "summary": summary,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {
                    "run_clustering": {},
                },
                "artifacts": artifacts,
            }

        elif tool_name in {"run_clustering", "compare_clusterings"}:
            comparisons = result_data.get("comparisons", []) or []
            if comparisons:
                comparison_bits = []
                for comparison in comparisons[:3]:
                    comparison_bits.append(
                        f"{comparison.get('cluster_key')} ({comparison.get('n_clusters')} clusters)"
                    )
                summary = "Compared clustering resolutions: " + ", ".join(comparison_bits) + "."
                options, option_actions = self._checkpoint_options([
                    ("Promote the recommended clustering resolution as the primary clustering", "promote_primary_clustering"),
                    ("Review cluster sizes or marker genes before choosing a primary clustering", "review_clustering_quality"),
                    ("Proceed with annotation using the recommended clustering", "run_annotation"),
                    ("Something else", "custom"),
                ])
                checkpoint = {
                    "kind": "clustering_comparison",
                    "question": "I generated multiple clustering resolutions. What should I do next?",
                    "options": options,
                    "default": options[0],
                    "decision_key": "primary_clustering_next_step",
                    "summary": summary,
                    "recommendation": options[0],
                    "option_actions": option_actions,
                    "action_inputs": {
                        "compare_clusterings": {
                            "resolutions": [
                                comparison.get("resolution")
                                for comparison in comparisons
                                if comparison.get("resolution") is not None
                            ]
                        }
                    },
                    "artifacts": artifacts,
                }
            else:
                cluster_key = result_data.get("cluster_key", "clustering")
                n_clusters = result_data.get("n_clusters", "?")
                summary = f"Clustering produced {n_clusters} clusters in '{cluster_key}'."
                options, option_actions = self._checkpoint_options([
                    ("Proceed to cell type annotation", "run_annotation"),
                    ("Compare alternative clustering resolutions before annotating", "compare_clusterings"),
                    ("Run marker analysis or cluster-size review before annotating", "review_cluster_markers"),
                    ("Something else", "custom"),
                ])
                checkpoint = {
                    "kind": "clustering",
                    "question": "Clustering is complete. What should I do next?",
                    "options": options,
                    "default": options[0],
                    "decision_key": "clustering_next_step",
                    "summary": summary,
                    "recommendation": options[0],
                    "option_actions": option_actions,
                    "action_inputs": {
                        "run_annotation": {
                            "majority_voting": True,
                            "cluster_key": result_data.get("primary_cluster_key") or result_data.get("cluster_key", "leiden"),
                        },
                        "compare_clusterings": {
                            "method": result_data.get("method", "leiden"),
                            "resolutions": [0.5, 1.0, 1.5],
                            "generate_figures": True,
                        },
                    },
                    "artifacts": artifacts,
                }

        elif tool_name in {"run_celltypist", "run_scimilarity"}:
            n_types = result_data.get("n_types", "?")
            annotation_key = result_data.get("annotation_key", tool_name)
            summary = f"Annotation is complete with {n_types} predicted cell types in '{annotation_key}'."
            options, option_actions = self._checkpoint_options([
                ("Review annotation quality and dominant labels per cluster", "review_annotation_quality"),
                ("Run DEG on the current clustering to validate cluster identities", "run_deg"),
                ("Save the current annotated dataset", "save_data"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "annotation",
                "question": "Annotation is complete. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "annotation_next_step",
                "summary": summary,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {
                    "run_deg": {
                        "groupby": self.world_state.data_summary.get("cluster_key") or "leiden",
                    },
                    "save_data": {},
                },
                "artifacts": artifacts,
            }

        elif tool_name == "run_batch_correction":
            method = result_data.get("method", "batch correction")
            batch_key = result_data.get("batch_key", "batch")
            summary = f"Applied {method} batch correction using '{batch_key}' and recomputed the embedding."
            options, option_actions = self._checkpoint_options([
                ("Run clustering on the corrected representation", "run_clustering"),
                ("Generate or review corrected UMAP figures before clustering", "review_corrected_embedding"),
                ("Inspect batch mixing quality before continuing", "inspect_batch_mixing"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "batch_correction",
                "question": "Batch correction is complete. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "batch_correction_next_step",
                "summary": summary,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {
                    "run_clustering": {},
                },
                "artifacts": artifacts,
            }

        elif tool_name == "run_deg":
            groupby = result_data.get("groupby", "clusters")
            n_groups = result_data.get("n_groups", "?")
            summary = f"DEG is complete across {n_groups} groups using '{groupby}'."
            options, option_actions = self._checkpoint_options([
                ("Review top markers and caveats before any pathway analysis", "review_deg_results"),
                ("Run GSEA for one or more interesting groups", "run_gsea"),
                ("Save the current dataset with DEG results", "save_data"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "deg",
                "question": "Differential expression analysis is complete. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "deg_next_step",
                "summary": summary,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {
                    "save_data": {},
                },
                "artifacts": artifacts,
            }

        elif tool_name == "inspect_data" and not self._active_request_is_followup:
            processing = result_data.get("processing", {}) or {}
            existing_stage = (
                processing.get("has_clusters")
                or result_data.get("clustering", {}).get("has_clusters")
                or processing.get("has_umap")
                or processing.get("is_normalized")
            )
            if existing_stage:
                n_obs = result_data.get("shape", {}).get("n_obs", "?")
                summary = (
                    f"The loaded dataset already has existing analysis state "
                    f"({n_obs} cells; processing includes normalization/embedding/clustering)."
                )
                options, option_actions = self._checkpoint_options([
                    ("Continue from the existing processed state", "continue_existing_state"),
                    ("Inspect the current clustering/annotation state before deciding", "inspect_existing_state"),
                    ("Start over from the original data state if raw counts are available", "restart_from_raw"),
                    ("Something else", "custom"),
                ])
                checkpoint = {
                    "kind": "continue_existing_run",
                    "question": "I found an already processed dataset state. What should I do next?",
                    "options": options,
                    "default": options[0],
                    "decision_key": "existing_state_next_step",
                    "summary": summary,
                    "recommendation": options[0],
                    "option_actions": option_actions,
                    "action_inputs": {},
                    "artifacts": artifacts,
                }

        return checkpoint

    def _build_recovery_checkpoint(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.collaborative:
            return None
        if result_data.get("status") not in {"warning", "error"}:
            return None

        missing = result_data.get("missing_prerequisites") or []
        if not missing:
            return None

        artifacts = self._checkpoint_artifact_paths(result_data)
        message = result_data.get("message", f"{tool_name} requires additional state before it can run.")
        checkpoint: Optional[Dict[str, Any]] = None

        if tool_name == "generate_figure" and "embedding" in missing:
            options, option_actions = self._checkpoint_options([
                ("Compute PCA, neighbors, and UMAP now", "run_dimred"),
                ("Inspect the current dataset state before computing embeddings", "inspect_existing_state"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "missing_embedding",
                "question": "This plot needs a UMAP embedding that is not available yet. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "missing_embedding_next_step",
                "summary": message,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {"run_dimred": {}},
                "artifacts": artifacts,
            }
        elif tool_name == "generate_figure" and "valid_color_key" in missing:
            options, option_actions = self._checkpoint_options([
                ("Render a plain UMAP without coloring", "render_plain_umap"),
                ("List the available obs columns and plot color choices", "list_plot_colors"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "invalid_plot_color",
                "question": "The requested UMAP coloring key is not available. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "invalid_plot_color_next_step",
                "summary": message,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {
                    "render_plain_umap": {"plot_type": "umap", "color_by": None, "include_image": True},
                },
                "artifacts": artifacts,
            }
        elif tool_name == "run_celltypist" and "clustering" in missing:
            options, option_actions = self._checkpoint_options([
                ("Run clustering now, then return to annotation", "run_clustering"),
                ("Inspect available clustering state before deciding", "inspect_existing_state"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "annotation_missing_clustering",
                "question": "Annotation needs a valid clustering column that is not available yet. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "annotation_missing_clustering_next_step",
                "summary": message,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {"run_clustering": {}},
                "artifacts": artifacts,
            }
        elif tool_name == "run_deg" and "grouping" in missing:
            options, option_actions = self._checkpoint_options([
                ("Run clustering now so DEG has a grouping column", "run_clustering"),
                ("List available obs columns to choose a DEG grouping", "list_plot_colors"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "deg_missing_grouping",
                "question": "Differential expression needs a valid grouping column. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "deg_missing_grouping_next_step",
                "summary": message,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {"run_clustering": {}},
                "artifacts": artifacts,
            }
        elif tool_name == "get_top_markers" and "deg" in missing:
            groupby = self.world_state.data_summary.get("cluster_key") or "leiden"
            options, option_actions = self._checkpoint_options([
                (f"Run DEG now using `{groupby}`", "run_deg"),
                ("Inspect current clustering state before running DEG", "inspect_existing_state"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "markers_missing_deg",
                "question": "Top markers are not available because DEG has not been run yet. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "markers_missing_deg_next_step",
                "summary": message,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {"run_deg": {"groupby": groupby}},
                "artifacts": artifacts,
            }
        elif tool_name == "get_celltypes" and "annotation" in missing:
            cluster_key = self.world_state.data_summary.get("cluster_key") or "leiden"
            options, option_actions = self._checkpoint_options([
                ("Run cell type annotation now", "run_annotation"),
                ("Inspect current clustering state before annotating", "inspect_existing_state"),
                ("Something else", "custom"),
            ])
            checkpoint = {
                "kind": "celltypes_missing_annotation",
                "question": "Cell type summaries are not available because annotation has not been run yet. What should I do next?",
                "options": options,
                "default": options[0],
                "decision_key": "celltypes_missing_annotation_next_step",
                "summary": message,
                "recommendation": options[0],
                "option_actions": option_actions,
                "action_inputs": {
                    "run_annotation": {"majority_voting": True, "cluster_key": cluster_key},
                },
                "artifacts": artifacts,
            }

        return checkpoint

    def _build_system_prompt(self) -> str:
        """Attach runtime state to the static system prompt."""
        return f"{SYSTEM_PROMPT}\n\n## Runtime Interaction State\n{self._runtime_guidance()}"

    def _current_capabilities(self) -> Dict[str, Any]:
        return (self.world_state.data_summary or {}).get("capabilities", {})

    def _is_yes_response(self, value: str) -> bool:
        text = (value or "").strip().lower()
        return text in {"y", "yes", "1", "ok", "okay", "sure", "continue", "do it", "run it", "compute it"}

    def _run_reconciled_action(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return parsed result. Used for checkpoint option handling."""
        result_json = self._execute_tool(tool_name, tool_input)
        try:
            return json.loads(result_json)
        except json.JSONDecodeError:
            return {"status": "error", "tool": tool_name, "message": "Invalid JSON result"}

    def _format_reconciled_response(self, intro: str, steps: List[Dict[str, Any]], final_result: Optional[Dict[str, Any]] = None) -> str:
        """Format a response from executed steps. Used for checkpoint option handling."""
        lines = [intro]
        for step in steps:
            summary = step.get("summary") or step.get("message") or ""
            if summary:
                lines.append(f"  {summary}")
        if final_result and final_result.get("output_path"):
            lines.append(f"Output: {final_result['output_path']}")
        return "\n".join(lines)

    def _sync_world_state(self, extra_text: Optional[str] = None) -> None:
        """Refresh the unified world state from the active AnnData and request context."""
        self.world_state.set_active_request(self._active_request)
        self.world_state.sync_from_adata(
            self.adata,
            request_text=extra_text or self._active_request,
        )

    def _record_world_state_snapshot(self) -> None:
        """Persist a compact world-state snapshot into the run ledger."""
        if self.run_manager:
            self.run_manager.append_world_state_snapshot(self.world_state.snapshot())

    def _remember_user_preferences(self, message: str) -> None:
        """Persist explicit user corrections so later tools can reuse them mechanically."""
        if not message:
            return

        batch_patterns = [
            r"\buse\s+([A-Za-z_][A-Za-z0-9_]*)\s+as\s+(?:the\s+)?batch(?:\s+key|\s+column)?\b",
            r"\bbatch(?:\s+key|\s+column)?\s+(?:is|=)\s*([A-Za-z_][A-Za-z0-9_]*)\b",
            r"\bsample(?:\s+key|\s+column)?\s+(?:is|=)\s*([A-Za-z_][A-Za-z0-9_]*)\b",
        ]

        for pattern in batch_patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1)
            if self.adata is not None and candidate not in self.adata.obs.columns:
                continue
            self.world_state.resolve_decision(
                "batch_key",
                candidate,
                source="user",
                message=message,
            )
            if self.run_manager:
                self.run_manager.add_user_decision(
                    {
                        "key": "batch_key",
                        "policy_action": "recommend_and_confirm",
                        "status": "user_corrected",
                        "applied_value": candidate,
                        "user_message": message,
                    }
                )
            break

    def _apply_world_state_overrides(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Apply confirmed decisions to tool inputs when the user did not restate them."""
        if tool_name in {"run_qc", "run_batch_correction"} and not tool_input.get("batch_key"):
            batch_key = self.world_state.get_confirmed_value("batch_key")
            if batch_key and (self.adata is None or batch_key in self.adata.obs.columns):
                tool_input["batch_key"] = batch_key

    def _artifact_kind_from_path(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return "figure"
        if suffix in {".h5ad", ".h5", ".loom"}:
            return "data"
        if suffix in {".json"}:
            return "json"
        if suffix in {".md", ".txt", ".csv", ".tsv"}:
            return "report"
        if suffix in {".log"}:
            return "log"
        return "artifact"

    def _generic_artifacts_from_result(self, tool_name: str, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        paths: List[tuple[str, Dict[str, Any]]] = []
        for key in ["output_path", "figure_path", "gsea_evidence_report", "gsea_evidence_json"]:
            value = result_data.get(key)
            if value:
                paths.append((value, {}))
        for figure_path in result_data.get("figures", []) or []:
            paths.append((figure_path, {"mode": result_data.get("mode", "")}))
        for comparison in result_data.get("comparisons", []) or []:
            if comparison.get("figure_path"):
                paths.append(
                    (
                        comparison["figure_path"],
                        {"cluster_key": comparison.get("cluster_key")},
                    )
                )

        artifacts = []
        for path, metadata in paths:
            normalized = os.path.abspath(path)
            artifacts.append(
                {
                    "artifact_id": artifact_id_from_path(normalized),
                    "path": normalized,
                    "kind": self._artifact_kind_from_path(normalized),
                    "role": "artifact",
                    "source_tool": tool_name,
                    "created_at": datetime.now().isoformat(),
                    "exists": os.path.exists(normalized),
                    "metadata": metadata,
                    "review_count": 0,
                    "last_reviewed_at": None,
                    "last_review_question": "",
                }
            )
        # Deduplicate while preserving order
        deduped = []
        seen_paths = set()
        for artifact in artifacts:
            if artifact["path"] in seen_paths:
                continue
            seen_paths.add(artifact["path"])
            deduped.append(artifact)
        return deduped

    def _generic_decisions_from_result(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        decisions = list(result_data.get("decisions_raised", []) or [])
        if decisions:
            return decisions

        if tool_name == "compare_clusterings" and result_data.get("comparisons") and not tool_input.get("promote_resolution"):
            clustering_decision = decision_for_clustering_selection(
                result_data.get("comparisons", []),
                source_tool=tool_name,
            )
            if clustering_decision is not None:
                decisions.append(clustering_decision)

        return decisions

    def _generic_verification(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "verification" in result_data:
            return result_data["verification"]

        status = result_data.get("status", "ok")
        if status == "error":
            return {
                "status": "failed",
                "summary": result_data.get("message", f"{tool_name} failed."),
                "checks": [],
                "recovery_options": ["Inspect the error message and choose a corrective next step."],
            }

        checks = []
        for artifact in self._generic_artifacts_from_result(tool_name, result_data):
            checks.append(
                {
                    "name": f"artifact_exists:{Path(artifact['path']).name}",
                    "status": "passed" if artifact["exists"] else "failed",
                    "details": f"Artifact path: {artifact['path']}",
                }
            )

        if tool_name == "run_clustering" and self.adata is not None:
            cluster_key = result_data.get("cluster_key")
            if cluster_key:
                checks.append(
                    {
                        "name": "cluster_key_present",
                        "status": "passed" if cluster_key in self.adata.obs.columns else "failed",
                        "details": f"Clustering key '{cluster_key}' should exist in adata.obs.",
                    }
                )
        if tool_name == "generate_figure":
            output_path = result_data.get("output_path")
            if output_path:
                checks.append(
                    {
                        "name": "figure_exists",
                        "status": "passed" if os.path.exists(output_path) else "failed",
                        "details": f"Figure output path: {output_path}",
                    }
                )

        verification_status = "passed" if all(check["status"] == "passed" for check in checks) else "warning"
        return {
            "status": verification_status,
            "summary": f"{tool_name} completed with {'no' if verification_status == 'passed' else 'some'} verification issues.",
            "checks": checks,
            "recovery_options": [] if verification_status == "passed" else ["Review the failed checks before continuing."],
        }

    def _ensure_standard_tool_result(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result_data: Dict[str, Any],
        before_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        after_snapshot = self.world_state.snapshot()

        if "state_delta" not in result_data:
            before_stage = before_snapshot.get("analysis_stage", "uninitialized")
            after_stage = after_snapshot.get("analysis_stage", before_stage)
            before_processing = (before_snapshot.get("data_summary") or {}).get("processing", {})
            after_processing = (after_snapshot.get("data_summary") or {}).get("processing", {})
            changed_flags = {}
            for key in sorted(set(before_processing.keys()) | set(after_processing.keys())):
                if before_processing.get(key) != after_processing.get(key):
                    changed_flags[key] = {
                        "before": before_processing.get(key),
                        "after": after_processing.get(key),
                    }
            result_data["state_delta"] = {
                "tool": tool_name,
                "summary": result_data.get("message") or f"{tool_name} completed.",
                "dataset_changed": tool_name in {
                    "run_qc",
                    "normalize_and_hvg",
                    "run_dimred",
                    "run_clustering",
                    "compare_clusterings",
                    "run_celltypist",
                    "run_scimilarity",
                    "run_batch_correction",
                    "run_deg",
                },
                "stage_before": before_stage,
                "stage_after": after_stage,
                "changed_flags": changed_flags,
                "notes": [],
            }

        existing_artifacts = result_data.get("artifacts_created", []) or []
        if existing_artifacts:
            result_data["artifacts_created"] = existing_artifacts
        else:
            result_data["artifacts_created"] = self._generic_artifacts_from_result(tool_name, result_data)
        existing_decisions = result_data.get("decisions_raised", []) or []
        if existing_decisions:
            result_data["decisions_raised"] = existing_decisions
        else:
            result_data["decisions_raised"] = self._generic_decisions_from_result(tool_name, tool_input, result_data)
        if (
            tool_name == "ask_user"
            and result_data.get("decision_key")
            and result_data.get("user_response")
            and result_data.get("user_response") not in {"proceed", "no response"}
        ):
            self.world_state.resolve_decision(
                result_data["decision_key"],
                result_data["user_response"],
                source="user",
                message=result_data.get("question", ""),
            )
            result_data["decisions_raised"] = [
                decision
                for decision in self.world_state.resolved_decisions[-1:]
            ]
            result_data["decisions_raised"] = [
                decision.to_dict() if hasattr(decision, "to_dict") else decision
                for decision in result_data["decisions_raised"]
            ]
        result_data["verification"] = self._generic_verification(tool_name, tool_input, result_data)
        return result_data

    def _ask_continue(self, error_msg: str, suggestions: list = None) -> str:
        """Ask user how to proceed after an error."""
        from rich.console import Console
        from rich.panel import Panel
        console = Console()

        console.print()
        console.print(Panel(
            f"{error_msg}",
            title="⚠️  Issue Detected",
            border_style="yellow"
        ))

        if suggestions:
            console.print("\n[bold]Suggestions:[/bold]")
            for i, s in enumerate(suggestions, 1):
                console.print(f"  {i}. {s}")

        console.print("\n[bold]What would you like to do?[/bold]")
        console.print("  • Type a new instruction")
        console.print("  • Press Enter to let the agent try to recover")
        console.print("  • Type 'quit' to stop")

        try:
            response = input("\n> ").strip()
            return response if response else "try to recover from the error"
        except (EOFError, KeyboardInterrupt):
            return "quit"

    def analyze(
        self,
        request: str,
        data_path: Optional[str] = None,
        run_name: Optional[str] = None,
        max_iterations: int = 20,
        continue_conversation: bool = False,
    ) -> str:
        """
        Analyze single-cell data based on a natural language request.

        The agent will:
        1. Create a run directory (if enabled)
        2. Inspect the data state
        3. Plan necessary analysis steps
        4. Execute each step using tools (returning structured JSON)
        5. Report results and save manifest

        Parameters
        ----------
        request : str
            Natural language description of the analysis to perform.
        data_path : str, optional
            Path to the input data file (h5ad or 10X h5).
            If None and self.adata exists, uses already-loaded data.
        run_name : str, optional
            Name for the run directory.
        max_iterations : int, default 20
            Maximum number of tool calls.
        continue_conversation : bool, default False
            If True, continue from previous conversation history.
            Useful for interactive follow-up questions.

        Returns
        -------
        str
            Summary of the analysis performed.
        """
        # Determine if this is a follow-up (data already loaded)
        is_followup = data_path is None and self.adata is not None
        self._active_request = request
        self._active_request_is_followup = is_followup
        self.world_state.set_active_request(request)
        self._remember_user_preferences(request)
        self._sync_world_state(extra_text=request)

        # Create run directory only for first analysis
        if self.create_run_dir and self.run_manager is None:
            self.run_manager = create_run(
                base_dir=self.output_dir,
                run_name=run_name,
                mode="agent",
                keep_intermediate=self.save_checkpoints,
            )
            self.run_manager.set_request(request)
            self.run_manager.set_model(f"{self.provider}:{self.model}")

            try:
                from scagent import __version__
                self.run_manager.set_version(__version__)
            except:
                pass

            self._print(f"[dim]📁 Output: {self.run_manager.run_dir}[/dim]")

            if data_path:
                self.run_manager.add_input(data_path)
            self._record_world_state_snapshot()
        elif is_followup and self.run_manager:
            # Log follow-up request in existing manifest
            self.run_manager.log_step(
                tool="follow_up",
                parameters={"request": request},
                result={"status": "starting"}
            )
            self.run_manager.append_event("follow_up_request", {"request": request})

        # Clear any stale checkpoint - the LLM's response options take precedence
        # The LLM knows what options it presented and will interpret numbered inputs correctly
        if self._pending_checkpoint:
            self._clear_pending_checkpoint("superseded by new response")

        # Build initial message
        user_message = request
        if data_path:
            user_message += f"\n\nData file: {data_path}"
        elif is_followup:
            # Inform agent that data is already loaded
            user_message += f"\n\n[Data already loaded in memory: {self.adata.n_obs} cells x {self.adata.n_vars} genes]"
        if self.run_manager:
            user_message += f"\nOutput directory: {self.run_manager.run_dir}"

        if self.verbose:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print()
            console.print(Panel(request, title="🔬 Analyzing", border_style="cyan"))

        if self.collaborative and not sys.stdin.isatty():
            message = (
                "Collaborative agent mode requires an interactive terminal because the agent must "
                "pause and ask for decisions at checkpoints."
            )
            if self.run_manager:
                self.run_manager.fail(message)
            raise RuntimeError(message)

        # Route to provider-specific implementation
        if self.provider == "anthropic":
            return self._analyze_anthropic(user_message, max_iterations, continue_conversation)
        elif self.provider in {"openai", "groq"}:
            return self._analyze_openai(user_message, max_iterations, continue_conversation)
        elif self.provider == "codex":
            return self._analyze_codex(user_message, max_iterations, continue_conversation)
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def _codex_tool_specs(self) -> List[Dict[str, Any]]:
        """Return compact tool specs for the Codex decision prompt."""
        specs: List[Dict[str, Any]] = []
        for tool in self.tools:
            function = tool.get("function", {})
            specs.append({
                "name": function.get("name"),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {}),
            })
        return specs

    def _build_codex_decision_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Build a one-step planner prompt for the Codex CLI bridge."""
        tool_result_limit = int(os.environ.get("SCAGENT_CODEX_TOOL_RESULT_LIMIT", "50000"))

        def compact_message(message: Dict[str, Any]) -> Dict[str, Any]:
            compact = dict(message)
            content = compact.get("content")
            if isinstance(content, str) and len(content) > tool_result_limit:
                compact["content"] = (
                    content[:tool_result_limit]
                    + f"\n\n[truncated to {tool_result_limit} characters]"
                )
            return compact

        payload = {
            "runtime_state": json.loads(self._runtime_guidance()),
            "conversation": [compact_message(message) for message in messages],
            "available_tools": self._codex_tool_specs(),
        }
        return (
            f"{self._build_system_prompt()}\n\n"
            "## Codex CLI Bridge Instructions\n"
            "You are selecting the next SCAgent action. Do not run shell commands or edit files. "
            "SCAgent will execute exactly one returned tool call, then send you the JSON result "
            "on the next iteration.\n\n"
            "Return JSON matching the required schema only:\n"
            "- Use kind='tool_call' when another SCAgent tool should run. Set tool_name to one "
            "available tool and tool_input_json to a string containing a JSON object.\n"
            "- Use kind='final' when the analysis response is complete. Set content to the final "
            "user-facing answer.\n"
            "- For final responses, set tool_name and tool_input_json to null. For tool calls, set "
            "content to null.\n\n"
            "## Current Request, History, Runtime State, and Tools\n"
            f"{json.dumps(payload, indent=2, default=str)}"
        )

    def _analyze_codex(
        self,
        user_message: str,
        max_iterations: int,
        continue_conversation: bool = False,
    ) -> str:
        """Run analysis loop using Codex CLI ChatGPT login."""
        if continue_conversation and self._conversation_history:
            messages = self._conversation_history.copy()
            messages.append({"role": "user", "content": user_message})
        else:
            messages = [{"role": "user", "content": user_message}]
        final_result = ""
        tool_names = {tool.get("name") for tool in self._codex_tool_specs()}

        try:
            for _iteration in range(max_iterations):
                prompt = self._build_codex_decision_prompt(messages)
                decision = self.client.complete_json(prompt, CODEX_DECISION_SCHEMA)

                thought = decision.get("thought")
                if thought:
                    self._print_thinking(str(thought))

                kind = decision.get("kind")
                if kind == "tool_call":
                    tool_name = decision.get("tool_name")
                    tool_input_text = decision.get("tool_input_json") or "{}"
                    if tool_name not in tool_names:
                        raise CodexCLIError(f"Codex requested unknown SCAgent tool: {tool_name}")
                    try:
                        tool_input = json.loads(tool_input_text)
                    except json.JSONDecodeError as exc:
                        raise CodexCLIError(
                            f"Codex returned invalid JSON tool_input_json for {tool_name}."
                        ) from exc
                    if not isinstance(tool_input, dict):
                        raise CodexCLIError(f"Codex returned non-object tool_input_json for {tool_name}.")

                    result_json = self._execute_tool(str(tool_name), tool_input)
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_call": {
                            "name": tool_name,
                            "arguments": tool_input,
                        },
                    })
                    messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": result_json,
                    })

                    if self._pending_image:
                        messages.append({
                            "role": "user",
                            "content": (
                                "A figure was generated at "
                                f"{self._pending_image['path']}. The Codex CLI bridge did not "
                                "send inline image bytes; call review_figure if visual review is needed."
                            ),
                        })
                        self._pending_image = None
                    continue

                if kind == "final":
                    final_result = decision.get("content") or ""
                    messages.append({"role": "assistant", "content": final_result})
                    self._conversation_history = messages
                    self._print("\n" + "-" * 50)
                    self._print(final_result)
                    if self.run_manager:
                        self.run_manager.complete(summary=final_result)
                        self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")
                    return final_result

                raise CodexCLIError(f"Codex returned unknown decision kind: {kind}")

            final_result = "Analysis stopped: max iterations reached"
            self._conversation_history = messages
        except Exception as e:
            if self.run_manager:
                self.run_manager.fail(str(e))
            raise

        return final_result

    def _analyze_anthropic(self, user_message: str, max_iterations: int, continue_conversation: bool = False) -> str:
        """Run analysis loop using Anthropic API."""
        if continue_conversation and self._conversation_history:
            # Continue from previous conversation
            messages = self._conversation_history.copy()
            messages.append({"role": "user", "content": user_message})
        else:
            # Start fresh
            messages = [{"role": "user", "content": user_message}]
        final_result = ""

        try:
            for iteration in range(max_iterations):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self._build_system_prompt(),
                    tools=self.tools,
                    messages=messages,
                )

                if response.stop_reason == "tool_use":
                    tool_results = []
                    assistant_content = []

                    for content in response.content:
                        if content.type == "text":
                            assistant_content.append(content)
                            if self.verbose:
                                self._print_thinking(content.text)

                        elif content.type == "tool_use":
                            assistant_content.append(content)
                            result_json = self._execute_tool(content.name, content.input)

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result_json,
                            })

                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({"role": "user", "content": tool_results})

                    # If there's a pending image, add it as a user message with vision
                    if self._pending_image:
                        image_msg = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Here is the generated figure ({self._pending_image['path']}). Please analyze it:"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": self._pending_image['mime'],
                                        "data": self._pending_image['base64']
                                    }
                                }
                            ]
                        }
                        messages.append(image_msg)
                        self._print(f"    [Sending image to LLM for analysis]")
                        self._pending_image = None

                elif response.stop_reason == "end_turn":
                    # Add final assistant message to history
                    messages.append({"role": "assistant", "content": response.content})

                    for content in response.content:
                        if hasattr(content, "text"):
                            final_result = content.text
                            self._print("\n" + "-" * 50)
                            self._print(final_result)

                    # Check if the response indicates failure/incomplete
                    failure_indicators = ["error", "failed", "couldn't", "unable to", "not installed",
                                         "missing", "cannot", "exception", "try again"]
                    seems_like_failure = any(ind in final_result.lower() for ind in failure_indicators)

                    # If it looks like a failure, offer interactive recovery
                    if seems_like_failure and self.verbose:
                        user_input = self._ask_continue(
                            final_result,  # Pass the actual response - it contains the specific issue
                            suggestions=["Provide additional instructions", "Try a different approach"]
                        )
                        if user_input.lower() not in ["quit", "exit", "q"]:
                            # Continue the conversation with user input
                            messages.append({"role": "user", "content": user_input})
                            self._conversation_history = messages
                            continue  # Go to next iteration

                    # Save conversation history for potential follow-ups
                    self._conversation_history = messages

                    if self.run_manager:
                        self.run_manager.complete(summary=final_result)
                        self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")

                    return final_result

                else:
                    logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                    break

            final_result = "Analysis stopped: max iterations reached"

        except Exception as e:
            if self.run_manager:
                self.run_manager.fail(str(e))
            raise

        return final_result

    def _analyze_openai(self, user_message: str, max_iterations: int, continue_conversation: bool = False) -> str:
        """Run analysis loop using OpenAI API."""
        if continue_conversation and self._conversation_history:
            # Continue from previous conversation
            messages = self._conversation_history.copy()
            messages.append({"role": "user", "content": user_message})
        else:
            # Start fresh
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_message},
            ]
        final_result = ""

        try:
            for iteration in range(max_iterations):
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = self._build_system_prompt()
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=4096,
                    tools=self.tools,
                    messages=messages,
                )

                choice = response.choices[0]
                message = choice.message

                if choice.finish_reason == "tool_calls" and message.tool_calls:
                    # Add assistant message with tool calls
                    messages.append(message)

                    # Print any reasoning/text content from the agent
                    if message.content:
                        self._print_thinking(message.content)

                    # Process each tool call
                    for tool_call in message.tool_calls:
                        tool_input = json.loads(tool_call.function.arguments)
                        result_json = self._execute_tool(tool_call.function.name, tool_input)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_json,
                        })

                    # If there's a pending image, add it as a user message with vision
                    if self._pending_image:
                        image_msg = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Here is the generated figure ({self._pending_image['path']}). Please analyze it:"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{self._pending_image['mime']};base64,{self._pending_image['base64']}"
                                    }
                                }
                            ]
                        }
                        messages.append(image_msg)
                        self._print(f"    [Sending image to LLM for analysis]")
                        self._pending_image = None

                elif choice.finish_reason == "stop":
                    # Add final assistant message to history
                    messages.append({"role": "assistant", "content": message.content})

                    final_result = message.content or ""

                    # Check if the response indicates failure/incomplete
                    failure_indicators = ["error", "failed", "couldn't", "unable to", "not installed",
                                         "missing", "cannot", "exception", "try again"]
                    seems_like_failure = any(ind in final_result.lower() for ind in failure_indicators)

                    self._print("\n" + "-" * 50)
                    self._print(final_result)

                    # If it looks like a failure, offer interactive recovery
                    if seems_like_failure and self.verbose:
                        user_input = self._ask_continue(
                            final_result,  # Pass the actual response - it contains the specific issue
                            suggestions=["Provide additional instructions", "Try a different approach"]
                        )
                        if user_input.lower() not in ["quit", "exit", "q"]:
                            # Continue the conversation with user input
                            messages.append({"role": "user", "content": user_input})
                            self._conversation_history = messages
                            continue  # Go to next iteration

                    # Save conversation history for potential follow-ups
                    self._conversation_history = messages

                    if self.run_manager:
                        self.run_manager.complete(summary=final_result)
                        self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")

                    return final_result

                elif choice.finish_reason == "length":
                    # Response was truncated due to length
                    self._print("\n[Warning: Response truncated due to length]")
                    final_result = message.content or ""
                    self._print(final_result)
                    if self.run_manager:
                        self.run_manager.complete(summary=final_result)
                    return final_result

                else:
                    self._print(f"\n[Debug: finish_reason={choice.finish_reason}]")
                    logger.warning(f"Unexpected finish reason: {choice.finish_reason}")
                    break

            final_result = "Analysis stopped: max iterations reached"

        except Exception as e:
            if self.run_manager:
                self.run_manager.fail(str(e))
            raise

        return final_result

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool and return JSON result."""
        from rich.console import Console
        from rich.status import Status
        from pathlib import Path

        console = Console()
        logger.info(f"Tool call: {tool_name} with {tool_input}")

        # Only block truly pipeline-progressing tools when checkpoint pending
        # Allow flexible tools (run_code, inspection, visualization) to proceed
        if self._pending_checkpoint and self._is_action_tool(tool_name) and tool_name != "ask_user":
            if tool_name not in self.CHECKPOINT_EXEMPT_TOOLS:
                return self._blocked_by_checkpoint_result(tool_name)
            # For exempt tools, we'll include checkpoint context in the result later

        def _sanitize_name(value: str) -> str:
            value = value or tool_name
            cleaned = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in value)
            return cleaned.strip("_")[:80] or tool_name

        def _prepare_tool_paths() -> None:
            """Route artifacts into the structured run directories."""
            if not self.run_manager:
                return

            figure_tools = {"generate_figure"}
            checkpoint_tools = {
                "run_qc",
                "normalize_and_hvg",
                "run_dimred",
                "run_clustering",
                "run_celltypist",
                "run_scimilarity",
                "run_batch_correction",
                "run_deg",
            }

            run_root = self.run_manager.run_dir

            if tool_name in figure_tools:
                requested = tool_input.get("output_path")
                if not requested:
                    stem = f"{tool_input.get('plot_type', 'figure')}_{tool_input.get('color_by', 'plot')}"
                    tool_input["output_path"] = self.run_manager.get_figure_path(_sanitize_name(stem))
                else:
                    requested_path = Path(requested)
                    if not requested_path.is_absolute():
                        if requested_path.parent == Path(".") or (
                            requested_path.parts and requested_path.parts[0] == run_root.name
                        ):
                            tool_input["output_path"] = self.run_manager.get_figure_path(
                                _sanitize_name(requested_path.stem),
                                ext=requested_path.suffix.lstrip(".") or "png",
                            )

            if tool_name == "run_qc" and not tool_input.get("figure_dir"):
                tool_input["figure_dir"] = str(self.run_manager.dirs["figures"])

            if tool_name == "compare_clusterings" and tool_input.get("generate_figures") and not tool_input.get("figure_dir"):
                tool_input["figure_dir"] = str(self.run_manager.dirs["figures"])

            if tool_name == "run_gsea":
                requested_dir = tool_input.get("output_dir")
                if not requested_dir:
                    tool_input["output_dir"] = str(self.run_manager.dirs["gsea"])
                else:
                    requested_path = Path(requested_dir)
                    if not requested_path.is_absolute():
                        if requested_path == Path(".") or requested_path.name == run_root.name:
                            tool_input["output_dir"] = str(self.run_manager.dirs["gsea"])

            # When save_checkpoints is False, NEVER save intermediate h5ad files
            # Only save when save_checkpoints is True OR when it's save_data tool
            if tool_name in checkpoint_tools and not self.save_checkpoints:
                # Always remove output_path for checkpoint tools when not saving intermediates
                tool_input.pop("output_path", None)

            if self.save_checkpoints and tool_name in checkpoint_tools and not tool_input.get("output_path"):
                tool_input["output_path"] = self.run_manager.get_intermediate_path(_sanitize_name(tool_name))

        _prepare_tool_paths()
        self._apply_world_state_overrides(tool_name, tool_input)
        self._sync_world_state()
        before_snapshot = self.world_state.snapshot()
        if self.run_manager:
            self.run_manager.append_log(f"START {tool_name} {json.dumps(tool_input, default=str)}")

        # Special handling for ask_user - get input from user but still track it like other tools
        if tool_name == "ask_user":
            if self._pending_checkpoint:
                tool_input.setdefault("question", self._pending_checkpoint.get("question", "What would you like to do next?"))
                tool_input.setdefault("options", self._pending_checkpoint.get("options", []))
                tool_input.setdefault("option_actions", self._pending_checkpoint.get("option_actions", []))
                tool_input.setdefault("default", self._pending_checkpoint.get("default", ""))
                tool_input.setdefault("decision_key", self._pending_checkpoint.get("decision_key", ""))
            result_json = self._handle_ask_user(tool_input)
        # Special handling for install_package - requires approval
        elif tool_name == "install_package":
            result_json = self._handle_install_package(tool_input)
        # Special handling for run_code - show what's being executed
        elif tool_name == "run_code":
            self._print(f"\n[Tool] {tool_name}")
            self._print(f"    Description: {tool_input.get('description', 'custom code')}")
            if self.verbose:
                code_preview = tool_input.get('code', '')[:100]
                if len(tool_input.get('code', '')) > 100:
                    code_preview += "..."
                self._print(f"    Code: {code_preview}")
            # Pass output directory for code file saving
            if self.run_manager:
                tool_input["output_dir"] = str(self.run_manager.run_dir)
            # Run without spinner for code (it may have its own output)
            result_json, self.adata = process_tool_call(
                tool_name,
                tool_input,
                self.adata,
                world_state=self.world_state,
                run_manager=self.run_manager,
            )
        else:
            # Run with spinner for other tools
            if self.verbose:
                with console.status(f"[bold cyan]Running {tool_name}...", spinner="dots") as status:
                    result_json, self.adata = process_tool_call(
                        tool_name,
                        tool_input,
                        self.adata,
                        world_state=self.world_state,
                        run_manager=self.run_manager,
                    )
                console.print(f"[green]✓[/green] {tool_name} complete")
            else:
                result_json, self.adata = process_tool_call(
                    tool_name,
                    tool_input,
                    self.adata,
                    world_state=self.world_state,
                    run_manager=self.run_manager,
                )

        # Check for image in result and store for vision
        try:
            result_data = json.loads(result_json)
            self._sync_world_state()

            # If there's an image, store it for the next message
            if "image_base64" in result_data:
                image_context = result_data.get("image_context", {})
                self._pending_image = {
                    "base64": result_data["image_base64"],
                    "mime": result_data.get("image_mime", "image/png"),
                    "path": (
                        result_data.get("output_path")
                        or result_data.get("figure_path")
                        or image_context.get("output_path")
                        or "figure.png"
                    ),
                }
                # Remove base64 from JSON to keep response small
                del result_data["image_base64"]
                if "image_mime" in result_data:
                    del result_data["image_mime"]
                result_data["image_included"] = True
                result_json = json.dumps(result_data, indent=2)

            status = result_data.get("status", "unknown")

            if tool_name == "run_gsea" and status == "ok" and self.run_manager:
                evidence_reports = self._generate_gsea_evidence_reports(result_data)
                if evidence_reports:
                    result_data.update(evidence_reports)
            result_data = self._ensure_standard_tool_result(
                tool_name,
                tool_input,
                result_data,
                before_snapshot,
            )
            checkpoint = self._build_checkpoint_payload(tool_name, tool_input, result_data)
            if checkpoint is None:
                checkpoint = self._build_recovery_checkpoint(tool_name, tool_input, result_data)
            if tool_name == "ask_user":
                prior_checkpoint = self._pending_checkpoint
                selected_action = result_data.get("selected_action")
                self._clear_pending_checkpoint(result_data.get("user_response"))
                auto_execution = self._execute_checkpoint_action(selected_action, prior_checkpoint or {}) if prior_checkpoint else None
                if auto_execution is not None:
                    result_data["auto_execution"] = auto_execution
            if checkpoint is not None:
                result_data["checkpoint_required"] = True
                result_data["checkpoint"] = checkpoint
                self._set_pending_checkpoint(checkpoint)
            self.world_state.apply_tool_result(tool_name, result_data, adata=self.adata)
            result_json = json.dumps(result_data, indent=2)

            if self.run_manager:
                self.run_manager.log_step(
                    tool=tool_name,
                    input_path=tool_input.get("data_path"),
                    output_path=result_data.get("output_path") or tool_input.get("output_path"),
                    parameters=tool_input,
                    result=result_data,
                )
                for w in result_data.get("warnings", []):
                    self.run_manager.add_warning(w)
                for artifact in result_data.get("artifacts_created", []):
                    self.run_manager.add_artifact(artifact)
                for decision in result_data.get("decisions_raised", []):
                    self.run_manager.add_user_decision(decision)
                if result_data.get("verification"):
                    self.run_manager.add_verification(result_data["verification"])
                self._record_world_state_snapshot()
                self.run_manager.append_log(
                    f"END {tool_name} status={status} output={result_data.get('output_path', '')}"
                )

            if status == "ok":
                if tool_name == "generate_figure" and result_data.get("output_path"):
                    self._interaction_state["shown_figures"].append({
                        "path": result_data["output_path"],
                        "kind": result_data.get("plot_type", "figure"),
                        "color_by": result_data.get("color_by"),
                    })
                elif tool_name == "review_figure" and result_data.get("figure_path"):
                    self._interaction_state["reviewed_figures"].append({
                        "path": result_data["figure_path"],
                        "question": result_data.get("question", ""),
                    })
                elif tool_name == "compare_clusterings":
                    for comparison in result_data.get("comparisons", []):
                        if comparison.get("figure_path"):
                            self._interaction_state["shown_figures"].append({
                                "path": comparison["figure_path"],
                                "kind": "umap",
                                "color_by": comparison.get("cluster_key"),
                            })
                elif tool_name == "ask_user":
                    self._interaction_state["asked_questions"].append({
                        "question": result_data.get("question", ""),
                        "options": result_data.get("options", []),
                    })
                # Show key results inline
                details = []
                if "after" in result_data:
                    details.append(f"{result_data['after'].get('n_cells', '?')} cells")
                if "n_clusters" in result_data:
                    details.append(f"{result_data['n_clusters']} clusters")
                if "n_types" in result_data:
                    details.append(f"{result_data['n_types']} cell types")
                if "shape" in result_data and tool_name == "run_code":
                    details.append(f"{result_data['shape']['n_cells']} cells")
                if details:
                    self._print(f"    → {', '.join(details)}")
            elif status == "needs_input":
                pass  # Handled by ask_user
            elif status == "error":
                self._print(f"    [red]✗ Error:[/red] {result_data.get('message', '')}")
            verification_status = (result_data.get("verification") or {}).get("status")
            if verification_status in {"warning", "failed"}:
                self._print(
                    f"    [yellow]Verification {verification_status}:[/yellow] "
                    f"{result_data['verification'].get('summary', '')}"
                )

        except json.JSONDecodeError:
            pass

        return result_json

    def _get_best_annotation_key(self) -> Optional[str]:
        """Return the most useful annotation column available on the current AnnData."""
        if self.adata is None:
            return None
        from ..analysis import get_best_annotation_key
        return get_best_annotation_key(self.adata)

    def _cluster_annotation_summary(self, cluster_id: str, groupby: str, annotation_key: str) -> Optional[Dict[str, Any]]:
        """Return dominant-label summary for one annotation source within a cluster."""
        if self.adata is None:
            return None
        from ..analysis import cluster_annotation_summary
        return cluster_annotation_summary(self.adata, cluster_id, groupby, annotation_key)

    def _normalize_annotation_lineage(self, label: str) -> str:
        """Map detailed annotation labels to broad lineages for sanity checks."""
        from ..analysis import normalize_annotation_lineage
        return normalize_annotation_lineage(label)

    def _annotation_agreement_summary(self, primary_label: str, secondary_label: Optional[str]) -> Dict[str, Any]:
        """Summarize whether annotation sources agree at fine or broad lineage level."""
        from ..analysis import annotation_agreement_summary
        return annotation_agreement_summary(primary_label, secondary_label)

    def _get_cluster_top_markers(self, cluster_id: str, groupby: str, n_genes: int = 10) -> List[str]:
        """Return top marker genes for a cluster from the current DEG result if available."""
        if self.adata is None:
            return []
        from ..analysis import get_cluster_top_markers
        return get_cluster_top_markers(self.adata, cluster_id, groupby, n_genes=n_genes)

    def _expected_marker_panel(self, label: str) -> Dict[str, Any]:
        """Return a coarse canonical marker panel for the inferred label."""
        from ..analysis import expected_marker_panel
        return expected_marker_panel(label)

    def _marker_support_summary(self, label: str, markers: List[str]) -> Dict[str, Any]:
        """Assess whether cluster markers support the inferred broad lineage."""
        from ..analysis import marker_support_summary
        return marker_support_summary(label, markers)

    def _get_biological_context(self, extra_text: Optional[str] = None) -> Dict[str, Any]:
        """Infer and cache biological context from the active data and request text."""
        if self.adata is None:
            return {}

        from ..analysis import infer_biological_context

        text_parts: List[str] = []
        if self.run_manager:
            if self.run_manager.manifest.request:
                text_parts.append(self.run_manager.manifest.request)
            text_parts.extend(self.run_manager.manifest.input_files)
        if extra_text:
            text_parts.append(extra_text)

        biological_context = infer_biological_context(
            self.adata,
            text_context=" ".join(part for part in text_parts if part),
        ).to_dict()
        self.biological_context = biological_context

        if self.run_manager:
            self.run_manager.manifest.parameters["biological_context"] = biological_context
            self.run_manager._save_manifest()

        return biological_context

    def _infer_cluster_context(self, cluster_id: str, groupby: str) -> Dict[str, Any]:
        """Infer a cluster's likely context, annotation agreement, and marker support."""
        if self.adata is None:
            return {
                "cluster": str(cluster_id),
                "groupby": groupby,
                "cell_type": f"cluster {cluster_id}",
                "confidence_level": "low",
                "confidence_score": 0.0,
                "interpretation_cautions": ["No AnnData object is currently loaded."],
            }
        from ..analysis import infer_cluster_confidence
        return infer_cluster_confidence(self.adata, cluster_id, groupby=groupby).to_dict()

    def _select_pathways_for_evidence(self, cluster_result: Dict[str, Any], max_pathways: int = 2) -> List[Dict[str, Any]]:
        """Select the most informative pathways from a cluster GSEA result."""
        candidates: List[Dict[str, Any]] = []
        for direction, key in [
            ("upregulated", "upregulated_pathways"),
            ("downregulated", "downregulated_pathways"),
        ]:
            for pathway in cluster_result.get(key, []):
                entry = dict(pathway)
                entry["direction"] = direction
                candidates.append(entry)

        if not candidates:
            return []

        candidates.sort(key=lambda item: (item.get("fdr", 1.0), -abs(item.get("nes", 0.0))))
        significant_candidates = [item for item in candidates if item.get("fdr", 1.0) < 0.25]
        ranked_candidates = significant_candidates or candidates

        selected: List[Dict[str, Any]] = []
        seen_terms = set()
        for pathway in ranked_candidates:
            term = pathway.get("term")
            if not term or term in seen_terms:
                continue
            selected.append(pathway)
            seen_terms.add(term)
            if len(selected) >= max_pathways:
                break

        return selected

    def _render_pathway_interpretation_summary(
        self,
        pathway_interpretation: Dict[str, Any],
        research_data: Dict[str, Any],
    ) -> str:
        """Render a concise markdown-friendly narrative from structured interpretation."""
        pieces = [pathway_interpretation.get("biological_meaning", "").strip()]

        statistical_confidence = pathway_interpretation.get("statistical_confidence")
        if statistical_confidence == "strong":
            pieces.append("The statistical support for this pathway call is strong.")
        elif statistical_confidence == "moderate":
            pieces.append("The statistical support for this pathway call is moderate.")
        else:
            pieces.append("The statistical support for this pathway call is weak or exploratory.")

        plausibility = pathway_interpretation.get("plausibility")
        if plausibility == "expected":
            pieces.append("This is biologically well aligned with the current cluster identity and context.")
        elif plausibility == "plausible":
            pieces.append("This is biologically plausible, but should still be interpreted with context-aware caution.")
        elif plausibility == "provisional":
            pieces.append("This is biologically provisional because upstream identity or evidence is not fully settled.")
        else:
            pieces.append("This remains biologically uncertain in the current evidence context.")

        papers = research_data.get("findings", {}).get("selected_papers") or []
        if papers:
            top_paper = papers[0]
            reasons = top_paper.get("match_reasons", [])
            if reasons:
                pieces.append(f"Top literature match was selected due to {'; '.join(reasons[:2])}.")
        else:
            pieces.append("Literature support was limited for this exact pathway/cell-type combination.")

        caveats = pathway_interpretation.get("caveats", [])
        if caveats:
            pieces.append(f"Key caveats: {'; '.join(caveats[:3])}.")

        return " ".join(piece for piece in pieces if piece)

    def _generate_gsea_evidence_reports(self, gsea_result: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate markdown and JSON evidence reports for GSEA results."""
        if not self.run_manager:
            return None

        from ..analysis import context_query_hint, infer_pathway_interpretation

        results = gsea_result.get("results", {})
        if not results:
            return None

        groupby = "leiden"
        if self.adata is not None and "rank_genes_groups" in self.adata.uns:
            groupby = self.adata.uns["rank_genes_groups"].get("params", {}).get("groupby", "leiden")

        max_clusters_with_literature = 5
        max_pathways_per_cluster = 2

        # Get DEG validity metadata
        deg_validity = None
        deg_caveats = []
        cluster_caveats = gsea_result.get("cluster_caveats", {})
        biological_context = self._get_biological_context()
        if self.adata is not None:
            deg_validity = self.adata.uns.get("deg_validity")
            deg_caveats = self.adata.uns.get("deg_caveats", [])

        report_payload: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "groupby": groupby,
            "gene_sets": gsea_result.get("gene_sets"),
            "clusters_analyzed": gsea_result.get("clusters_analyzed", []),
            "literature_limits": {
                "max_clusters_with_literature": max_clusters_with_literature,
                "max_pathways_per_cluster": max_pathways_per_cluster,
            },
            "deg_validity": deg_validity,
            "deg_caveats": deg_caveats,
            "biological_context": biological_context,
            "clusters": [],
        }

        md_lines = [
            "# GSEA Evidence Report",
            "",
            f"Generated: {report_payload['generated_at']}",
            f"Gene sets: `{gsea_result.get('gene_sets', 'unknown')}`",
            f"Cluster key: `{groupby}`",
            "",
            "This report combines pathway enrichment results with targeted PubMed searches.",
            "Cluster sections include cross-annotation agreement and marker-support cues so pathway narratives can be weighted by cluster identity confidence.",
            "",
        ]

        if biological_context:
            md_lines.append("## Biological Context")
            md_lines.append("")
            md_lines.append(f"- Tissue: `{biological_context.get('tissue', 'unknown')}`")
            md_lines.append(f"- Species: `{biological_context.get('species', 'unknown')}`")
            md_lines.append(f"- Sample type: `{biological_context.get('sample_type', 'unknown')}`")
            md_lines.append(f"- Condition: `{biological_context.get('condition', 'unknown')}`")
            if biological_context.get("expected_celltypes"):
                md_lines.append(f"- Expected cell types: {', '.join(biological_context['expected_celltypes'])}")
            if biological_context.get("confidence") is not None:
                md_lines.append(f"- Context confidence: {biological_context['confidence']:.2f}")
            provenance = biological_context.get("provenance", {})
            if provenance:
                md_lines.append("- Provenance:")
                for key, value in provenance.items():
                    md_lines.append(f"  - `{key}` from `{value}`")
            for note in biological_context.get("notes", []):
                md_lines.append(f"- Note: {note}")
            md_lines.append("")

        # Add DEG validity summary if present
        if deg_validity:
            md_lines.append("## DEG Validity")
            md_lines.append("")
            if deg_validity.get("is_valid") and not deg_validity.get("has_warnings"):
                md_lines.append("- Status: **VALID** (no issues detected)")
            elif deg_validity.get("is_valid"):
                md_lines.append(f"- Status: **VALID** with {deg_validity.get('n_warnings', 0)} warning(s)")
            else:
                md_lines.append(f"- Status: **ISSUES DETECTED** ({deg_validity.get('n_errors', 0)} error(s))")
            md_lines.append(f"- Matrix type: `{deg_validity.get('matrix_type', 'unknown')}`")
            md_lines.append(f"- Species: `{deg_validity.get('data_species', 'unknown')}`")
            md_lines.append(f"- Gene format: `{deg_validity.get('gene_id_format', 'unknown')}`")
            md_lines.append("")
            if deg_caveats:
                md_lines.append("### Caveats (apply to all clusters)")
                md_lines.append("")
                for caveat in deg_caveats:
                    md_lines.append(f"- {caveat}")
                md_lines.append("")

        cluster_ids = [str(cid) for cid in gsea_result.get("clusters_analyzed", [])]
        cluster_ids.sort(
            key=lambda cid: (
                results.get(cid, {}).get("total_significant", 0),
                max(
                    [abs(p.get("nes", 0.0)) for p in (
                        results.get(cid, {}).get("upregulated_pathways", []) +
                        results.get(cid, {}).get("downregulated_pathways", [])
                    )] or [0.0]
                ),
            ),
            reverse=True,
        )
        any_significant = any(results.get(cid, {}).get("total_significant", 0) > 0 for cid in cluster_ids)

        md_lines.extend([
            f"- Clusters analyzed: {len(cluster_ids)}",
            f"- Clusters with significant pathways: {sum(results.get(cid, {}).get('total_significant', 0) > 0 for cid in cluster_ids)}",
            "",
        ])

        if any_significant:
            md_lines.append("Clusters are ordered by the number of statistically significant pathways and pathway effect size.")
        else:
            md_lines.append("No clusters passed the default significance cutoff; exploratory pathway summaries are shown instead.")
        md_lines.append("")

        for idx, cluster_key in enumerate(cluster_ids):
            cluster_result = results.get(cluster_key, {})
            cluster_context = self._infer_cluster_context(cluster_key, groupby)
            specific_caveats = list(cluster_caveats.get(cluster_key, []))
            for context_caveat in cluster_context.get("interpretation_cautions", []):
                if context_caveat not in specific_caveats:
                    specific_caveats.append(context_caveat)

            cluster_entry: Dict[str, Any] = {
                "cluster": cluster_key,
                "context": cluster_context,
                "total_significant": cluster_result.get("total_significant"),
                "caveats": specific_caveats,
                "pathways": [],
            }

            md_lines.append(f"## Cluster {cluster_key}")
            md_lines.append("")
            md_lines.append(f"- Inferred cell type: `{cluster_context['cell_type']}`")
            if cluster_context.get("cell_type_fraction") is not None:
                md_lines.append(f"- Annotation support: {cluster_context['cell_type_fraction']:.1%} of annotated cells in this cluster")
            if cluster_context.get("secondary_cell_type"):
                md_lines.append(f"- Secondary annotation: `{cluster_context['secondary_cell_type']}` from `{cluster_context.get('secondary_annotation_key')}`")
            if cluster_context.get("annotation_agreement_note"):
                agreement_label = cluster_context.get("annotation_agreement", "unknown").replace("_", " ")
                md_lines.append(f"- Cross-annotation agreement: `{agreement_label}`")
                md_lines.append(f"  {cluster_context['annotation_agreement_note']}")
            if cluster_context.get("confidence_level"):
                md_lines.append(
                    f"- Cluster confidence: `{cluster_context['confidence_level']}`"
                    f" ({cluster_context.get('confidence_score', 0.0):.2f})"
                )
            if cluster_context.get("marker_support") and cluster_context.get("marker_support") != "unknown":
                matched = ", ".join(cluster_context.get("marker_support_markers", [])[:4]) or "no canonical markers among current top markers"
                md_lines.append(f"- Marker support: `{cluster_context['marker_support']}` for `{cluster_context.get('marker_lineage', 'unknown')}` lineage ({matched})")
            if cluster_context.get("top_markers"):
                md_lines.append(f"- Top markers: {', '.join(cluster_context['top_markers'][:5])}")
            if cluster_context.get("n_cells") is not None:
                md_lines.append(f"- Cells in cluster: {cluster_context['n_cells']}")
            md_lines.append(f"- Significant pathways (FDR < 0.25): {cluster_result.get('total_significant', 0)}")
            if cluster_result.get("total_significant", 0) == 0:
                md_lines.append("- Note: no pathways passed the significance threshold; any literature below is exploratory context only.")

            if specific_caveats:
                md_lines.append(f"- **Interpretation caveats for this cluster:**")
                for caveat in specific_caveats:
                    md_lines.append(f"  - {caveat}")

            md_lines.append("")

            if "error" in cluster_result:
                md_lines.append(f"Pathway analysis error: {cluster_result['error']}")
                md_lines.append("")
                cluster_entry["error"] = cluster_result["error"]
                report_payload["clusters"].append(cluster_entry)
                continue

            selected_pathways = self._select_pathways_for_evidence(
                cluster_result,
                max_pathways=max_pathways_per_cluster if (
                    idx < max_clusters_with_literature and (
                        cluster_result.get("total_significant", 0) > 0 or not any_significant
                    )
                ) else 0,
            )

            if not selected_pathways:
                if any_significant and cluster_result.get("total_significant", 0) == 0:
                    md_lines.append("Skipped automatic literature expansion because other clusters had stronger statistically significant pathway signals.")
                else:
                    md_lines.append("No automatically researched pathways for this cluster.")
                md_lines.append("")
                report_payload["clusters"].append(cluster_entry)
                continue

            for pathway in selected_pathways:
                context_hint = " | ".join(
                    part for part in [
                        context_query_hint(biological_context) if biological_context else "",
                        self.run_manager.manifest.request or "",
                    ] if part
                )
                research_json, _ = process_tool_call(
                    "research_findings",
                    {
                        "pathway": pathway["term"],
                        "cell_type": cluster_context["cell_type"],
                        "genes": pathway.get("genes", []),
                        "context": context_hint,
                        "cluster_confidence": cluster_context.get("confidence_score"),
                        "recent_years": 3,
                    },
                    self.adata,
                    world_state=self.world_state,
                    run_manager=self.run_manager,
                )
                research_data = json.loads(research_json)
                papers = research_data.get("findings", {}).get("selected_papers", [])
                reviews = research_data.get("findings", {}).get("review_articles", [])
                fdr_value = pathway.get("fdr")
                statistically_significant = (fdr_value if fdr_value is not None else 1.0) < 0.25
                structured_interpretation = infer_pathway_interpretation(
                    pathway,
                    cluster_context,
                    research_data,
                    biological_context=biological_context,
                    interpretation_cautions=specific_caveats,
                )
                interpretation = self._render_pathway_interpretation_summary(
                    structured_interpretation.to_dict(),
                    research_data,
                )

                pathway_entry = {
                    "term": pathway["term"],
                    "direction": pathway.get("direction"),
                    "nes": pathway.get("nes"),
                    "fdr": pathway.get("fdr"),
                    "genes": pathway.get("genes", []),
                    "research": research_data,
                    "structured_interpretation": structured_interpretation.to_dict(),
                    "interpretation": interpretation,
                    "statistically_significant": statistically_significant,
                }
                cluster_entry["pathways"].append(pathway_entry)

                md_lines.append(f"### {pathway['term']}")
                md_lines.append("")
                md_lines.append(f"- Direction: {pathway.get('direction', 'unknown')}")
                md_lines.append(f"- NES: {pathway.get('nes', 'NA')}")
                md_lines.append(f"- FDR q-value: {pathway.get('fdr', 'NA')}")
                md_lines.append(f"- Evidence tier: {'significant' if statistically_significant else 'exploratory'}")
                md_lines.append(f"- Leading-edge genes: {', '.join(pathway.get('genes', [])[:5]) or 'NA'}")
                md_lines.append(f"- Papers found: {research_data.get('total_papers_found', 0)}")
                md_lines.append(
                    f"- Statistical confidence: `{structured_interpretation.statistical_confidence}`"
                )
                md_lines.append(
                    f"- Biological plausibility: `{structured_interpretation.plausibility}`"
                )
                md_lines.append("")
                md_lines.append(f"Interpretation: {interpretation}")
                md_lines.append("")

                if structured_interpretation.suggested_validation:
                    md_lines.append("Suggested validation:")
                    for suggestion in structured_interpretation.suggested_validation[:3]:
                        md_lines.append(f"- {suggestion}")
                    md_lines.append("")

                if reviews:
                    top_review = reviews[0]
                    md_lines.append("Review article:")
                    md_lines.append(
                        f"- PMID {top_review.get('pmid')}: {top_review.get('title')} ({top_review.get('year')}, {top_review.get('journal')})"
                    )
                    md_lines.append("")

                if papers:
                    md_lines.append("Recent primary literature:")
                    for paper in papers[:3]:
                        reasons = paper.get("match_reasons", [])
                        reason_suffix = f" [matched on: {', '.join(reasons[:2])}]" if reasons else ""
                        md_lines.append(
                            f"- PMID {paper.get('pmid')}: {paper.get('title')} ({paper.get('year')}, {paper.get('journal')}){reason_suffix}"
                        )
                    md_lines.append("")
                else:
                    md_lines.append("No recent primary literature matched this pathway/cell-type query.")
                    md_lines.append("")

            report_payload["clusters"].append(cluster_entry)

        md_lines.extend([
            "## Notes",
            "",
            "- Documentation lookup and literature lookup are intentionally separated.",
            "- Pathway interpretation is based on PubMed searches anchored to pathway term, inferred cell type, and leading-edge genes.",
            "- DEG validity checks are run before GSEA; caveats propagate to cluster interpretations.",
            "- Clusters with caveats (small size, batch confounding, etc.) should be interpreted with caution.",
            "- Use the JSON companion report for downstream programmatic inspection.",
            "",
        ])

        md_path = self.run_manager.write_text_report("gsea_evidence", "\n".join(md_lines), ext="md")
        json_path = self.run_manager.write_json_report("gsea_evidence", report_payload)
        self.run_manager.append_log(f"Generated GSEA evidence reports: {md_path}, {json_path}")

        return {
            "gsea_evidence_report": md_path,
            "gsea_evidence_json": json_path,
        }

    def _handle_ask_user(self, tool_input: Dict[str, Any]) -> str:
        """Handle ask_user tool - prompt user for input."""
        question = tool_input["question"]
        options = tool_input.get("options", [])
        option_actions = tool_input.get("option_actions", [])
        default = tool_input.get("default", "")
        decision_key = tool_input.get("decision_key", "")

        if self.collaborative and not sys.stdin.isatty():
            return json.dumps({
                "status": "error",
                "tool": "ask_user",
                "question": question,
                "options": options,
                "option_actions": option_actions,
                "default": default,
                "decision_key": decision_key,
                "message": "Collaborative checkpoints require an interactive terminal.",
            }, indent=2)
        if not self.collaborative:
            response = default or "proceed"
            return json.dumps({
                "status": "ok",
                "tool": "ask_user",
                "question": question,
                "options": options,
                "option_actions": option_actions,
                "default": default,
                "decision_key": decision_key,
                "user_response": response,
                "auto_selected": True,
            }, indent=2)

        print()
        print(question)
        if options and not re.search(r"(^|\n)\s*1\.\s", question):
            for idx, option in enumerate(options, 1):
                print(f"{idx}. {option}")
        if default:
            print(f"[Default: {default}]")

        # Get user input
        try:
            response = input("> ").strip()
            if not response and default:
                response = default
        except (EOFError, KeyboardInterrupt):
            response = default or "no response"

        raw_response = response
        selected_option = None
        selected_action = None
        selected_index = None
        if options and response.isdigit():
            option_index = int(response) - 1
            if 0 <= option_index < len(options):
                selected_option = options[option_index]
                selected_index = option_index
                if option_index < len(option_actions):
                    selected_action = option_actions[option_index]
                response = selected_option
        elif response in options:
            selected_index = options.index(response)
            if selected_index < len(option_actions):
                selected_action = option_actions[selected_index]

        return json.dumps({
            "status": "ok",
            "tool": "ask_user",
            "question": question,
            "options": options,
            "option_actions": option_actions,
            "default": default,
            "decision_key": decision_key,
            "user_response": response,
            "raw_user_response": raw_response,
            "selected_option": selected_option,
            "selected_action": selected_action,
            "selected_index": selected_index,
        }, indent=2)

    def _handle_install_package(self, tool_input: Dict[str, Any]) -> str:
        """Handle install_package tool - requires user approval."""
        import subprocess
        import sys

        package = tool_input["package"]
        reason = tool_input["reason"]

        # Ask for approval
        print(f"\n{'='*50}")
        print(f"PACKAGE INSTALL REQUEST")
        print(f"Package: {package}")
        print(f"Reason: {reason}")
        print('='*50)

        try:
            response = input("Approve? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"

        if response in ("y", "yes"):
            # Try uv first (faster, works with uv-managed venvs), fall back to pip
            python_path = sys.executable
            install_commands = [
                ["uv", "pip", "install", "--python", python_path, package],
                [python_path, "-m", "pip", "install", package],
            ]

            last_error = None
            for cmd in install_commands:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=180
                    )
                    if result.returncode == 0:
                        return json.dumps({
                            "status": "ok",
                            "tool": "install_package",
                            "package": package,
                            "message": f"Successfully installed {package}"
                        }, indent=2)
                    else:
                        last_error = result.stderr
                except subprocess.TimeoutExpired:
                    last_error = "Installation timed out"
                except FileNotFoundError:
                    # Command not found (e.g., uv not installed), try next
                    continue

            # All install methods failed
            return json.dumps({
                "status": "error",
                "tool": "install_package",
                "package": package,
                "message": f"Installation failed: {last_error}"
            }, indent=2)
        else:
            return json.dumps({
                "status": "denied",
                "tool": "install_package",
                "package": package,
                "message": "User denied package installation"
            }, indent=2)

    def chat(self, message: str) -> str:
        """
        Send a single message and get a response.

        This is a simpler interface for quick questions without full analysis.

        Parameters
        ----------
        message : str
            Question or instruction.

        Returns
        -------
        str
            Agent's response.
        """
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._build_system_prompt(),
                messages=[{"role": "user", "content": message}],
            )
            for content in response.content:
                if hasattr(content, "text"):
                    return content.text
            return ""
        elif self.provider in {"openai", "groq"}:
            # OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=4096,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": message},
                ],
            )
            return response.choices[0].message.content or ""
        elif self.provider == "codex":
            prompt = (
                f"{self._build_system_prompt()}\n\n"
                "Answer the following user question directly. Return kind='final' in the "
                "required JSON schema; do not call tools for this lightweight chat method.\n\n"
                f"User question: {message}"
            )
            decision = self.client.complete_json(prompt, CODEX_DECISION_SCHEMA)
            if decision.get("kind") == "final":
                return decision.get("content") or ""
            return (
                "Codex requested an analysis tool for this question. "
                "Use `scagent analyze --provider codex` for tool-using runs."
            )
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def inspect(self, data_path: str) -> Dict[str, Any]:
        """
        Inspect a data file and return structured state.

        Parameters
        ----------
        data_path : str
            Path to the h5ad file.

        Returns
        -------
        Dict
            Structured data state.
        """
        result_json, self.adata = process_tool_call(
            "inspect_data",
            {"data_path": data_path},
            self.adata,
            world_state=self.world_state,
            run_manager=self.run_manager,
        )
        result = json.loads(result_json)
        self.biological_context = result.get("biological_context")
        self._sync_world_state(extra_text=data_path)
        self.world_state.apply_tool_result("inspect_data", result, adata=self.adata)
        self._record_world_state_snapshot()
        return result

    def reset_conversation(self):
        """
        Clear conversation history to start a fresh conversation.

        Useful when switching to a completely different analysis topic.
        Note: This does NOT unload the data - call reset() for that.
        """
        self._conversation_history = []
        self._interaction_state = {
            "shown_figures": [],
            "reviewed_figures": [],
            "asked_questions": [],
        }
        self.world_state = AgentWorldState()
        self._sync_world_state()
        self._print("Conversation history cleared.")

    def reset(self):
        """
        Reset agent state completely.

        Clears conversation history, loaded data, and run manager.
        """
        self._conversation_history = []
        self.adata = None
        self.run_manager = None
        self.biological_context = None
        self._pending_image = None
        self._interaction_state = {
            "shown_figures": [],
            "reviewed_figures": [],
            "asked_questions": [],
        }
        self.world_state = AgentWorldState()
        self._print("Agent state reset.")

    def recommend(self, goal: str) -> List[str]:
        """
        Get recommended analysis steps for a goal.

        Parameters
        ----------
        goal : str
            Analysis goal: 'qc', 'cluster', 'annotate', 'umap', 'deg', 'batch_correct'

        Returns
        -------
        List[str]
            Recommended analysis steps.
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call inspect() first.")

        from ..core import inspect_data, recommend_next_steps

        state = inspect_data(self.adata)
        return recommend_next_steps(state, goal)
