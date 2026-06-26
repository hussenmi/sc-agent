"""
Main agent class for autonomous single-cell analysis.

Uses Claude or OpenAI API with tools to perform single-cell analysis tasks.
Returns structured JSON from tools for reliable LLM reasoning.
Creates run directories with manifests for reproducibility.
"""

import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from . import (
    tracing as _tracing,  # optional OpenTelemetry per-step tracing (no-op unless SCAGENT_TRACE)
)
from .codex_bridge import CODEX_DECISION_SCHEMA, CodexCLIClient, CodexCLIError
from .decision_policy import (
    decision_for_clustering_selection,
)
from .prompts import SYSTEM_PROMPT
from .run_manager import RunManager, create_run
from .tools import (
    encode_image_base64,
    get_image_mime_type,
    get_openai_tools,
    get_tools,
    process_tool_call,
    write_h5ad_safe,
)
from .vision_sidecar import VisionSidecar
from .world_state import AgentWorldState, artifact_id_from_path

logger = logging.getLogger(__name__)

Provider = Literal["anthropic", "openai", "groq", "codex", "gemini"]

_SMART_AUTONOMOUS_PROMPT = """

## Smart Autonomous Mode

You are running in smart autonomous mode. **Drive the analysis forward without pausing** — you are the expert. Execute, narrate your reasoning, and keep going. Only stop when you genuinely cannot proceed without information the user alone can provide.

### How to work

**Narrate the WHY, not just the WHAT.** Before each tool call, write one sentence that includes your reasoning — not just the action. Examples:
- "Running PCA on 30 components — standard for this cell count and matches our lab defaults."
- "Using Leiden at resolution 1.0 as a starting point; I'll report cluster count and you can adjust if needed."
- "Running CellTypist with Immune_All_High — this dataset looks like immune cells based on the marker genes."

**Do NOT present numbered options at the end of every turn.** After completing a phase, give a brief status summary (what you found, what's next) and continue unless there's a real reason to stop. Options menus are for decisions, not routine narration.

When a real decision is needed, explain the evidence and call `pause_and_ask`
with concise labels and stable action identifiers. The runtime renders the
interactive selector; do not duplicate its numbered menu in prose.

**Proceed without pausing for:** standard preprocessing (normalization, HVG, PCA, neighbors, UMAP), algorithm parameter choices with established best practices, reversible steps you can re-run with different settings.

**Multi-sample data is the one exception that overrides autonomous mode.** When
inspection finds multiple sample-like groups and no `multi_sample_strategy` has
been selected, the runtime raises a `multi_sample_strategy` checkpoint and
**blocks the preprocessing tools until it is resolved.** Your single next action
is to present the choice with `pause_and_ask` (investigate / integrate with scVI
/ keep combined uncorrected / analyze separately / describe the experiment) and
end your turn. Do not deliberate about whether you can run QC or normalization
first — you cannot, they are blocked — so there is nothing to weigh. Do not infer
that correction is required from metadata names or group count. **"Investigate"
is an option you OFFER, not something you do before asking**: only after the user
selects `investigate_integration` do you run the uncorrected first pass
(PCA → neighbors → UMAP → clustering) and `diagnose_batch_effect`, after which the
runtime re-opens the decision. Explicit integration uses scVI unless the user or
a source workflow specifies another method.

### When to use `pause_and_ask`

Use it — and only use it — when:
1. **You need information only the user has.** Multiple equally plausible batch keys (e.g. `sample_id`, `batch`, `donor`) and nothing in the data resolves the ambiguity. Experimental design details that affect the analysis direction. Cell type context for manual annotation.
2. **Results are surprising in a consequential way.** Doublet rate >15%. QC removes >30% of cells at any reasonable threshold. Clustering reveals clear batch structure rather than biology. Anything that changes what should happen next in a non-obvious way.
3. **A genuine fork with large downstream consequences.** Not "which resolution?" — try one and explain. But "should I integrate across disease and control, or analyze them separately?" — that requires the user's scientific intent.

After calling `pause_and_ask`, write the question clearly in your response and end your turn. Do not call any more tools.

### Automatic checkpoints

Before batch correction, the system automatically saves an h5ad checkpoint. When the tool result includes `auto_checkpoint_saved`, mention it: "A checkpoint was saved at `<path>` — use `load_data('<path>')` to return to this state if needed."

### The pipeline is a continuous flow — a tool result is an input, not a stopping point

You are executing an analysis pipeline. When a tool returns results, your response is: **one sentence noting the key number, then the next tool call**. That's it. You do not write comprehensive reports between pipeline steps. You do not end your turn with text only unless you have reached a genuine pause point.

**Standard pipeline for an open-ended "analyze this" request:**

```
load_data
  → inspect_data
  → run_qc                    [narrate: MT% range, doublet rate, n_genes shape — 2 sentences max]
  → normalize_and_hvg         [narrate: "X HVGs selected."]
  → run_pca                   [narrate: "PCA done, 30 components."]
  → run_neighbors             [narrate: "Neighbor graph built."]
  → run_umap                  [narrate: "UMAP computed." → then run_code for QC overlay → then run_clustering]
  → run_clustering(res=1.5)   [narrate: "N clusters."]
  → run_cluster_qc            [narrate full table + evidence]
  → run_cluster_structure_qc  [for proposed/ambiguous clusters; narrate synthesis + heatmap evidence]
  → remove evidence-supported cleanup clusters when below the pause threshold, otherwise pause for review
  → [loop: normalize_and_hvg → run_pca → run_neighbors → run_umap → run_clustering → run_cluster_qc → run_cluster_structure_qc → until clean]
  → run_celltypist and/or run_scimilarity when organism/model compatibility allows
  → prepare_annotation with reference annotation keys
  → bc_get_panglaodb_marker_genes only for clusters flagged as requiring external adjudication
  → finalize_annotation with DEG + reference-label evidence, plus PanglaoDB evidence where required
  → final UMAP
```

**Each `→` is a tool call in the same response turn. The only places you stop and wait are: high-impact cleanup review, user domain knowledge, or surprising results.**

After `run_cluster_qc`: if any clusters are proposed/ambiguous, first call `run_cluster_structure_qc` to add covariance/Moran evidence. If structure QC synthesizes a cleanup set below the 15% pause threshold, remove exactly those clusters with `run_code`, explain the biological and technical evidence, and rerun the embedding/QC loop. If structure QC synthesizes no removal set, explicitly say that the reviewed clusters are being kept for now and proceed; do not keep asking about the stale metric-QC proposal. In user-facing narration, do not mention internal authorization or confirmation mechanics; frame it as an evidence-supported cleanup decision. Pause for review only when the structure-synthesized removal is at or above 15% or the tool explicitly marks the decision as high-impact/uncertain.
"""

FAILURE_PATTERNS = [
    r"\berror:",
    r"\bfailed to\b",
    r"\bfailed\b",
    r"\bfailure\b",
    r"\bexception\b",
    r"\btraceback\b",
    r"\bi couldn't\b",
    r"\bi could not\b",
    r"\bi can't\b",
    r"\bi cannot\b",
    r"\bunable to\b",
    r"\bnot installed\b",
    r"\bno module named\b",
    r"\bmodule not found\b",
    r"\bmissing dependency\b",
    r"\bmissing package\b",
    r"\bmissing module\b",
    r"\btry again\b",
]
NON_FAILURE_PATTERNS = [
    r"\bno errors?\b",
    r"\bwithout errors?\b",
    r"\b0 errors?\b",
]
# If any of these are present the response is considered complete, even if failure
# keywords appear (e.g. the agent correctly explains a fallback and offers next steps).
SUCCESS_OVERRIDE_PATTERNS = [
    r"\bwhat would you like\b",
    r"\bwhat would you like to do next\b",
    r"\bhow would you like\b",
    r"\bwhat do you think\b",
    r"\bwould you like me to\b",
    r"\bif you want.*i can\b",
    r"\bi can proceed\b",
    r"\bone of these ways\b",
    r"\bone of the following\b",
    r"\breadyfor\b",  # "ready for downstream"
    r"\bready for\b",
    r"\bsuccessfully applied\b",
    r"\bsuccessfully completed\b",
    r"\bsuccessfully run\b",
    r"\bsuccessfully computed\b",
    r"\bconverged after\b",
    r"\bbatch.corrected and ready\b",
    r"\bnow batch.corrected\b",
    # Any numbered next-step list ("1 Run", "1 Bypass", "1 Inspect", etc.)
    r"\b1[\.\)]\s+\w+\b.*\b2[\.\)]\s+\w+\b",
]
AUTO_RECOVERY_ATTEMPTS = 2
# Bounded re-prompts when the run tries to END with a scientific-spine obligation
# unmet (e.g. annotation staged-but-not-finalized). After these, the terminal gate
# stops nudging and forces a safe fallback so the run never ends silently incomplete.
OBLIGATION_NUDGES = 2

ACTION_TOOL_NAMES = {
    "load_data",
    "run_cellbender",
    "run_qc",
    "normalize_and_hvg",
    "run_pca",
    "run_neighbors",
    "run_umap",
    "run_clustering",
    "compare_clusterings",
    "run_celltypist",
    "run_scimilarity",
    "prepare_annotation",
    "stage_annotation_evidence",
    "finalize_annotation",
    "diagnose_batch_effect",
    "run_batch_correction",
    "score_integration",
    "benchmark_integration",
    "run_deg",
    "run_pseudobulk_deg",
    "run_gsea",
    "run_spectra",
    "score_gene_signature",
    "query_cells",
    "save_data",
    "run_cluster_qc",
    "run_cluster_structure_qc",
    "run_code",
    "write_report",
    "write_json",
    "run_shell",
    "install_package",
}

INSPECTION_TOOL_NAMES = {
    "inspect_data",
    "inspect_session",
    "list_celltypist_models",
    "check_celltypist_model",
    "list_artifacts",
    "get_cluster_sizes",
    "get_top_markers",
    "summarize_qc_metrics",
    "get_celltypes",
    "list_obs_columns",
    "review_figure",
    "review_artifact",
    "inspect_run_state",
    "inspect_data_inputs",
    "inspect_workspace",
    "read_file",
    "search_papers",
    "fetch_url",
    "web_search",
    "research_findings",
    "describe_image",
    "record_inspection",
}

# Load .env file if present
def _load_dotenv():
    """Load .env config, merging from lowest to highest precedence.

    Order: the package/repo root, then the current working directory, then
    ``$SCAGENT_HOME/.env`` last. Every existing file is loaded (not first-wins)
    with ``override=True``, so the one read last wins. ``$SCAGENT_HOME/.env`` is
    last on purpose: when scagent is installed as a shared module, that is the
    centrally-managed lab config, and editing it must control every user's setup
    regardless of which directory they run from. A local ``./.env`` can still add
    vars the shared file doesn't set, but cannot override it.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False  # python-dotenv not installed

    scagent_home = os.environ.get("SCAGENT_HOME", "").strip()
    # Lowest precedence first; $SCAGENT_HOME/.env last so it wins.
    candidates = [
        Path(__file__).parent.parent.parent / ".env",  # scagent/agent -> scagent -> project root
        Path.cwd() / ".env",
    ]
    if scagent_home:
        candidates.append(Path(scagent_home) / ".env")

    loaded = False
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.exists():
            continue  # skip duplicates (e.g. cwd == SCAGENT_HOME in dev)
        seen.add(resolved)
        load_dotenv(path, override=True)
        logger.info(f"Loaded config from {path}")
        loaded = True
    return loaded

_load_dotenv()


def _model_get(obj, name):
    """Read a single field from a /v1/models entry, backend- and SDK-agnostically.

    The OpenAI Python SDK parses each model into a pydantic object with
    ``extra="allow"``, so server-specific fields (``max_model_len``, ``meta``,
    ``aliases``, …) are reachable both as attributes and via ``model_extra``.
    Tests and some servers hand us plain dicts instead. Handle all three:
    plain dict, attribute, and ``model_extra`` fallback. Returns None if absent.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    val = getattr(obj, name, None)
    if val is not None:
        return val
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    return None


def _coerce_positive_int(value):
    """Coerce to a positive int, or return None.

    Servers report these limits as ints, but be tolerant of strings ("262144")
    and floats — and reject zero/negative/garbage so a bogus advertisement can
    never widen the context window past what the server actually allocated.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _server_context_limit(model):
    """Resolve the per-request context limit a serving backend advertises.

    Returns ``(limit, source)`` or ``(None, None)``. Two backends, two fields:

      * vLLM advertises top-level ``max_model_len`` — the GPU-constrained limit.
      * llama.cpp advertises ``meta.n_ctx`` — the *per-slot* context, i.e.
        already divided by ``--parallel``, which is exactly the per-request cap.

    Both are the real, memory-bound per-request limit. ``max_model_len`` wins
    when both are present (vLLM never sets ``meta``; this just makes precedence
    explicit and order-independent).
    """
    limit = _coerce_positive_int(_model_get(model, "max_model_len"))
    if limit:
        return limit, "max_model_len"
    n_ctx = _coerce_positive_int(_model_get(_model_get(model, "meta"), "n_ctx"))
    if n_ctx:
        return n_ctx, "meta.n_ctx"
    return None, None


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

    # After this many genuine finalize_annotation attempts fail validation, the
    # annotation save guard stops hard-blocking and lets save_data write a
    # clearly-marked UNVALIDATED dataset, so a run never ends with nothing saved.
    MAX_FINALIZE_ATTEMPTS_BEFORE_UNVALIDATED_SAVE = 2

    def __init__(
        self,
        provider: Optional[Provider] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = True,
        collaborative: bool = False,
        smart_autonomous: bool = True,
        create_run_dir: bool = True,
        output_dir: str = ".",
        save_checkpoints: bool = False,
        show_context_usage: bool = False,
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
        self.smart_autonomous = smart_autonomous
        self.create_run_dir = create_run_dir
        self.output_dir = output_dir
        self.save_checkpoints = save_checkpoints
        self.adata = None
        self.run_manager: Optional[RunManager] = None
        self.world_state = AgentWorldState()
        self.biological_context: Optional[Dict[str, Any]] = None
        self._pending_images: List[Dict[str, str]] = []  # For vision support (list of figure dicts)
        # Vision sidecar — used only when the main model is text-only AND
        # SCAGENT_VISION_MODEL is configured. None otherwise.
        self._vision_sidecar: Optional[VisionSidecar] = VisionSidecar.from_env()
        # Tracks the most recent producing-tool image_context per figure path so
        # describe_image can re-use plot_type / color_by / cluster_key on followups.
        self._figure_context_index: Dict[str, Dict[str, Any]] = {}
        self._next_llm_status_message: Optional[str] = None
        self._conversation_history: List[Dict[str, Any]] = []  # For interactive mode
        self._active_request: str = ""
        self._active_request_is_followup: bool = False
        self._interaction_state: Dict[str, List[Dict[str, Any]]] = {
            "shown_figures": [],
            "reviewed_figures": [],
            "asked_questions": [],
        }
        self._pending_checkpoint: Optional[Dict[str, Any]] = None
        self._active_cleanup_authorization: Optional[Dict[str, Any]] = None
        self._context_limit: int = 128_000  # overwritten by _init_* below
        self._vertex_key_file: Optional[str] = None
        self._vertex_project: Optional[str] = None
        self._vertex_region: Optional[str] = None
        self._vertex_token_expiry: float = 0.0
        self._last_estimated_tokens: int = 0
        self._last_actual_tokens: int = 0   # exact count from API response usage field
        # Per-response output cap, applied to every LLM call AND used as the context
        # completion reserve (kept in sync). 4096 truncated long report/finalize
        # outputs; raise via SCAGENT_MAX_OUTPUT_TOKENS for heavier write workloads.
        self._max_output_tokens: int = int(os.environ.get("SCAGENT_MAX_OUTPUT_TOKENS", "8192"))
        self._context_display_tokens: int = 0
        self._context_display_source: str = ""
        self._context_display_trim_target: int = 0
        self._context_display_hard_limit: int = 0
        self._tool_schema_tokens: int = 0   # precomputed at init; refreshed after MCP merge
        self._token_estimate_calibration: float = 1.0  # ratchets up after each API response
        self.show_context_usage: bool = show_context_usage

        self._silence_noisy_loggers()
        self._mcp_client: Optional[Any] = None  # MCPClientManager, if available

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
        elif provider == "gemini":
            self._init_gemini(api_key, model)
        elif provider == "vertex":
            self._init_vertex(api_key, model)
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'anthropic', 'openai', 'groq', 'codex', 'gemini', or 'vertex'."
            )

        # Connect to MCP servers after provider init so self.tools is already set
        self._init_mcp()

        # Apply SCAGENT_CONTEXT_LIMIT env override last — wins over any provider default
        _env_limit = os.environ.get("SCAGENT_CONTEXT_LIMIT")
        if _env_limit:
            try:
                self._context_limit = int(_env_limit)
                logger.info(f"Context limit overridden by SCAGENT_CONTEXT_LIMIT: {self._context_limit:,}")
            except ValueError:
                pass  # warning already emitted in _resolve_context_limit if called; otherwise silent

    @staticmethod
    def _silence_noisy_loggers() -> None:
        """Suppress INFO-level chatter from third-party libraries.

        Libraries like httpx, lightning, and openai log routine HTTP requests
        and training progress at INFO level, which clutters the terminal when
        the root logger is set to INFO (e.g. after scVI/lightning initialise).
        We push them to WARNING so only genuine problems surface.
        """
        import logging as _logging
        for name in (
            "httpx",
            "httpcore",
            "httpcore.http11",
            "httpcore.connection",
            "openai",
            "openai._base_client",
            "anthropic",
            "anthropic._base_client",
            "lightning",
            "lightning.pytorch",
            "pytorch_lightning",
            "harmonypy",
        ):
            _logging.getLogger(name).setLevel(_logging.WARNING)

    def close(self) -> None:
        """Shut down MCP connections and any other resources. Safe to call multiple times."""
        mcp_client = getattr(self, "_mcp_client", None)
        if mcp_client is not None:
            try:
                mcp_client.stop()
            except Exception:
                pass
            self._mcp_client = None

    def __del__(self) -> None:
        self.close()

    def _init_mcp(self) -> None:
        """Connect to MCP servers configured in .mcp.json and merge their tools."""
        try:
            from ..mcp.client import MCPClientManager
        except ImportError:
            logger.debug("mcp package not installed — MCP tools unavailable")
            return

        try:
            manager = MCPClientManager.from_config()
            manager.start()
            if not manager.connected_servers:
                if self.verbose and manager._server_configs:
                    logger.warning("MCP: no servers connected (check .mcp.json and server binaries)")
                return

            self._mcp_client = manager
            # Convert MCP schemas (always Anthropic format) to match self.tools format.
            # OpenAI format uses {"type": "function", "function": {..., "parameters": ...}};
            # Anthropic format uses {"name": ..., "description": ..., "input_schema": ...}.
            mcp_schemas = manager.tool_schemas
            if self.tools and self.tools[0].get("type") == "function":
                # self.tools is already in OpenAI format — convert MCP schemas to match
                mcp_schemas = [
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                        },
                    }
                    for t in mcp_schemas
                ]
            self.tools = self.tools + mcp_schemas
            self._tool_schema_tokens = self._estimate_tokens(self.tools)

            if self.verbose:
                tool_count = len(manager.tool_schemas)
                servers = ", ".join(
                    f"{s} ({sum(1 for t in manager._tool_to_server.values() if t == s)} tools)"
                    for s in manager.connected_servers
                )
                logger.info("MCP: connected — %s | %d tools added", servers, tool_count)

        except Exception as e:
            logger.warning("MCP init failed: %s — continuing without MCP tools", e)

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
        self.tools = get_tools(include_describe_image=self._use_sidecar_for_images())
        self._context_limit = 200_000  # all Claude models support 200K
        self._tool_schema_tokens = self._estimate_tokens(self.tools)

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
        self.tools = get_openai_tools(include_describe_image=self._use_sidecar_for_images())
        self._context_limit = self._resolve_context_limit()
        self._tool_schema_tokens = self._estimate_tokens(self.tools)

    def _init_codex(self, model: Optional[str]):
        """Initialize Codex CLI bridge for ChatGPT-login-backed runs."""
        codex_model = model or os.environ.get("SCAGENT_CODEX_MODEL")
        self.client = CodexCLIClient(model=codex_model, cwd=os.getcwd())
        self.model = codex_model or "codex-default"
        self.tools = get_openai_tools(include_describe_image=self._use_sidecar_for_images())
        self._tool_schema_tokens = self._estimate_tokens(self.tools)

    def _init_gemini(self, api_key: Optional[str], model: Optional[str]):
        """Initialize Google Gemini via its OpenAI-compatible API.

        Google exposes a drop-in OpenAI-compatible endpoint at
        https://generativelanguage.googleapis.com/v1beta/openai/
        so the standard OpenAI SDK works directly — just swap the base_url
        and pass GOOGLE_API_KEY as the api_key.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No API key provided. Set GOOGLE_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = model or "gemini-3.5-flash"
        self.tools = get_openai_tools(include_describe_image=self._use_sidecar_for_images())
        self._context_limit = self._resolve_context_limit()
        self._tool_schema_tokens = self._estimate_tokens(self.tools)

    def _init_vertex(self, api_key: Optional[str], model: Optional[str]):
        """Initialize Google Vertex AI via its OpenAI-compatible endpoint.

        Vertex AI authenticates with short-lived OAuth2 Bearer tokens, not API keys.
        Tokens are obtained via gcloud using the service account at
        GOOGLE_APPLICATION_CREDENTIALS (or SCAGENT_VERTEX_KEY_FILE).
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        key_file = (
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("SCAGENT_VERTEX_KEY_FILE")
        )
        project = os.environ.get("SCAGENT_VERTEX_PROJECT")
        region = os.environ.get("SCAGENT_VERTEX_REGION", "us-central1")

        if project is None and key_file:
            import json as _json
            with open(key_file) as _f:
                _sa = _json.load(_f)
            project = _sa.get("project_id")

        if not project:
            raise ValueError(
                "Vertex AI requires a project ID. "
                "Set SCAGENT_VERTEX_PROJECT or GOOGLE_APPLICATION_CREDENTIALS."
            )

        self._vertex_key_file = key_file
        self._vertex_project = project
        self._vertex_region = region

        token = self._get_vertex_token()
        base_url = (
            f"https://{region}-aiplatform.googleapis.com/v1beta1"
            f"/projects/{project}/locations/{region}/endpoints/openapi"
        )
        self.client = OpenAI(api_key=token, base_url=base_url)
        self._vertex_token_expiry = __import__("time").time() + 3600

        m = model or "gemini-3.5-flash"
        self.model = m if m.startswith("google/") else f"google/{m}"
        self.tools = get_openai_tools(include_describe_image=self._use_sidecar_for_images())
        self._context_limit = int(os.environ.get("SCAGENT_CONTEXT_LIMIT", "1000000"))
        self._tool_schema_tokens = self._estimate_tokens(self.tools)

    def _get_vertex_token(self) -> str:
        """Get an OAuth2 access token from gcloud for Vertex AI."""
        import subprocess
        if self._vertex_key_file:
            activate = subprocess.run(
                [
                    "gcloud", "auth", "activate-service-account",
                    "--key-file", self._vertex_key_file, "--quiet",
                ],
                capture_output=True, text=True,
            )
            if activate.returncode != 0:
                raise RuntimeError(
                    f"gcloud service account activation failed: {activate.stderr.strip()}"
                )
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gcloud auth print-access-token failed: {result.stderr.strip()}"
            )
        token = result.stdout.strip()
        if not token:
            raise RuntimeError("gcloud returned an empty access token")
        return token

    def _refresh_vertex_token_if_needed(self):
        """Recreate the Vertex AI client with a fresh Bearer token when close to expiry."""
        if self.provider != "vertex":
            return
        import time
        if time.time() < self._vertex_token_expiry - 300:
            return  # still valid for 5+ minutes
        from openai import OpenAI
        token = self._get_vertex_token()
        region = self._vertex_region
        project = self._vertex_project
        base_url = (
            f"https://{region}-aiplatform.googleapis.com/v1beta1"
            f"/projects/{project}/locations/{region}/endpoints/openapi"
        )
        self.client = OpenAI(api_key=token, base_url=base_url)
        self._vertex_token_expiry = time.time() + 3600

    # Map of LaTeX commands to Unicode/text used inside inline math blocks
    _LATEX_COMMANDS = {
        r"\rightarrow": "→",
        r"\leftarrow": "←",
        r"\Rightarrow": "⇒",
        r"\Leftarrow": "⇐",
        r"\leftrightarrow": "↔",
        r"\uparrow": "↑",
        r"\downarrow": "↓",
        r"\approx": "≈",
        r"\geq": "≥",
        r"\leq": "≤",
        r"\neq": "≠",
        r"\times": "×",
        r"\pm": "±",
        r"\cdot": "·",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\sigma": "σ",
        r"\infty": "∞",
        r"\sum": "Σ",
        r"\prod": "Π",
        r"\in": "∈",
        r"\notin": "∉",
        r"\subset": "⊂",
        r"\cup": "∪",
        r"\cap": "∩",
        r"\sqrt": "√",
        r"\log": "log",
        r"\exp": "exp",
    }

    @classmethod
    def _resolve_inline_math(cls, math_content: str) -> str:
        """Convert the interior of a $...$ block to readable plain text."""
        result = math_content
        # Replace known LaTeX commands (longest first to avoid partial matches)
        for cmd, uni in sorted(cls._LATEX_COMMANDS.items(), key=lambda x: -len(x[0])):
            result = result.replace(cmd, uni)
        # Strip any remaining backslash commands we don't know (e.g. \text{...} → content)
        result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\{([^}]*)\}", r"\1", result)  # bare braces → content
        result = re.sub(r"\\[a-zA-Z]+", "", result)     # drop unknown commands
        return result.strip()

    @classmethod
    def _delatex(cls, text: str) -> str:
        """Replace LaTeX math expressions with readable Unicode equivalents.

        Handles both single-symbol blocks ($\\times$) and compound inline
        math expressions ($151,370 \\times 0.00035 \\approx 53$).
        """
        if not text or "$" not in text:
            return text
        # Replace compound inline math $...$ blocks (non-greedy, no newlines inside)
        def _replace_math(m):
            return cls._resolve_inline_math(m.group(1))
        text = re.sub(r"\$([^$\n]+?)\$", _replace_math, text)
        return text

    # Patterns produced by local models' internal thinking/channel tokens that leak
    # into assistant text content and should be stripped before display.
    _ARTIFACT_PATTERNS = [
        # Gemma 4 / Gemma-style channel tokens:  <|channel>thought\n...\n<channel|>
        re.compile(r"<\|channel\>[^\|]*\n.*?<channel\|>\n?", re.DOTALL),
        # Qwen / DeepSeek thinking blocks: <think>...</think>
        re.compile(r"<think>.*?</think>\n?", re.DOTALL),
        # Generic angle-bracket model tokens: <|...|> on their own line
        re.compile(r"^<\|[^|]+\|>\s*$", re.MULTILINE),
    ]

    # Capturing versions of the above — used to extract thinking content for display.
    _THINKING_EXTRACT_PATTERNS = [
        # Gemma 4: <|channel>thought\n{content}\n<channel|>
        re.compile(r"<\|channel\>[^\|]*\n(.*?)<channel\|>\n?", re.DOTALL),
        # Qwen / DeepSeek: <think>{content}</think>
        re.compile(r"<think>(.*?)</think>\n?", re.DOTALL),
    ]

    @classmethod
    def _strip_model_artifacts(cls, text: str) -> str:
        """Remove internal thinking/channel tokens that local models sometimes leak."""
        if not text:
            return text
        for pat in cls._ARTIFACT_PATTERNS:
            text = pat.sub("", text)
        return text.strip()

    def _print(self, message: str, style: str = None, markdown: bool = False):
        """Print message if verbose using rich formatting.

        If markdown=True or message contains markdown patterns, renders as markdown.
        """
        if self.verbose:
            from rich.console import Console
            from rich.markdown import Markdown
            console = Console()
            message = self._strip_model_artifacts(message)
            message = self._delatex(message)

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

    def _split_reasoning_channels(self, message):
        """Split a tool-calling assistant message into (narration, chain_of_thought).

        Reasoning models emit two separate channels per turn: the user-facing
        narration in `content`, and the raw chain-of-thought under
        `reasoning_content` (Gemini/DeepSeek) or `reasoning` (vLLM parsers like
        nemotron_v3/glm45). Both raw values are returned (narration/CoT, or None
        when empty); display and persistence decisions are left to the caller.
        """
        extra = getattr(message, "model_extra", None) or {}
        cot = (
            extra.get("reasoning_content")
            or extra.get("reasoning")
            or getattr(message, "reasoning", None)
        )
        content = getattr(message, "content", None)
        narration = content if (content and content.strip()) else None
        return narration, (cot if (cot and cot.strip()) else None)

    def _save_thinking(self, cot: str, iteration: int):
        """Append a chain-of-thought trace to <run_dir>/reasoning.log.

        Returns the log path if written, else None. Gated by SCAGENT_SAVE_THINKING
        (default on); a no-op when saving is disabled, there is no run directory,
        or the trace is empty after artifact stripping.
        """
        if os.environ.get("SCAGENT_SAVE_THINKING", "1") != "1":
            return None
        rm = getattr(self, "run_manager", None)
        run_dir = getattr(rm, "run_dir", None) if rm is not None else None
        if run_dir is None:
            return None
        cot = self._strip_model_artifacts(cot)
        if not cot or not cot.strip():
            return None
        log_path = run_dir / "reasoning.log"
        header = f"\n{'=' * 60}\n# iteration {iteration} · {self.model}\n{'=' * 60}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(header + cot.rstrip() + "\n")
        return log_path

    def _print_thinking(self, message: str, dim: bool = False):
        """Print agent narration before a tool call.

        dim=True greys the text out — used for raw chain-of-thought
        (reasoning_content), shown only when SCAGENT_SHOW_THINKING=1.
        """
        if not message or not message.strip():
            return
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()
        message = self._strip_model_artifacts(message)
        message = self._delatex(message)
        if not message.strip():
            return
        marker = "[dim]…[/dim]" if dim else "[cyan]…[/cyan]"
        # Render as markdown if it contains markdown patterns, otherwise inline
        md_patterns = ["**", "##", "```", "- ", "1. "]
        if any(p in message for p in md_patterns):
            console.print(marker)
            if dim:
                console.print(Markdown(message), style="dim")
            else:
                console.print(Markdown(message))
        else:
            if dim:
                console.print(f"{marker} [dim]{message}[/dim]")
            else:
                console.print(f"{marker} {message}")

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

    def _followup_state_checkpoint(self, request: str) -> str:
        """Build a compact authoritative-state reminder for follow-up turns."""
        data_summary = self.world_state.data_summary or {}
        capabilities = data_summary.get("capabilities", {})
        compact_data_summary = {
            "shape": data_summary.get("shape"),
            "processing": data_summary.get("processing"),
            "cluster_key": data_summary.get("cluster_key"),
            "n_clusters": data_summary.get("n_clusters"),
            "cell_type_key": data_summary.get("cell_type_key"),
            "batch_key": data_summary.get("batch_key"),
            "biological_context": data_summary.get("biological_context"),
            "available_cluster_keys": capabilities.get("cluster_keys", []),
            "annotation_keys": capabilities.get("annotation_keys", []),
        }
        checkpoint = {
            "latest_user_request": request,
            "analysis_stage": self.world_state.analysis_stage,
            "current_data_summary": compact_data_summary,
            "last_action": self.world_state.last_action,
            "recent_events": self.world_state.recent_events[-8:],
            "recent_step_log": self.world_state.step_log[-12:],
            "outstanding_decisions": [
                d.to_dict() if hasattr(d, "to_dict") else d
                for d in self.world_state.outstanding_decisions[-5:]
            ],
            "resolved_decisions": [
                d.to_dict() if hasattr(d, "to_dict") else d
                for d in self.world_state.resolved_decisions[-5:]
            ],
        }
        guidance = (
            "Follow-up state checkpoint: the live in-memory AnnData and world "
            "state below are authoritative. Use earlier transcript only as "
            "background. Do not replay destructive actions from older transcript "
            "text, such as removing clusters or cells, unless the current state "
            "still validates that exact action. If a destructive action depends "
            "on cluster IDs, cell counts, or annotations, inspect or verify the "
            "current state first."
        )
        return f"{guidance}\n\n```json\n{json.dumps(checkpoint, indent=2, default=str)}\n```"

    def _is_gemma_model(self) -> bool:
        return "gemma" in (self.model or "").lower()

    def _is_gemini_model(self) -> bool:
        return self.provider == "gemini" or "gemini" in (self.model or "").lower()

    def _is_thinking_model(self) -> bool:
        """Return True for models that have controllable thinking/reasoning modes."""
        m = (self.model or "").lower()
        return "gemma" in m or "qwen" in m or "deepseek" in m or "gemini" in m

    def _thinking_extra(self) -> dict:
        """Return extra kwargs to control thinking mode and reasoning effort.

        DeepSeek API (cloud): thinking ON by default.
          SCAGENT_THINKING=0  → disable thinking entirely.
          SCAGENT_THINKING_EFFORT=high|max  → reasoning depth (default: high).

        Gemini (cloud): provider default unless explicitly configured.
          SCAGENT_THINKING=1  → set OpenAI-compatible reasoning_effort.
          SCAGENT_THINKING_EFFORT=minimal|low|medium|high.
          SCAGENT_THINKING_BUDGET=N  → legacy budget mapped to an effort level.

        vLLM / TensorRT-LLM local models (Qwen/Gemma): thinking OFF by default,
        sent explicitly via chat_template_kwargs (not relying on a server-side
        default, which TensorRT-LLM's trtllm-serve does not provide).
          SCAGENT_THINKING=1  → enable via chat_template_kwargs.
        """
        if not self._is_thinking_model():
            return {}
        m = (self.model or "").lower()
        if "deepseek" in m:
            thinking_on = os.environ.get("SCAGENT_THINKING", "1") != "0"
            if not thinking_on:
                return {"extra_body": {"thinking": {"type": "disabled"}}}
            effort = os.environ.get("SCAGENT_THINKING_EFFORT", "high")
            kwargs: dict = {"extra_body": {"thinking": {"type": "enabled"}}}
            if effort in ("high", "max"):
                kwargs["reasoning_effort"] = effort
            return kwargs
        if "gemini" in m:
            if os.environ.get("SCAGENT_THINKING", "0") == "1":
                effort = os.environ.get("SCAGENT_THINKING_EFFORT")
                if effort not in {"minimal", "low", "medium", "high"}:
                    budget = int(os.environ.get("SCAGENT_THINKING_BUDGET", "8000"))
                    effort = "low" if budget <= 1024 else "medium" if budget <= 8192 else "high"
                return {"reasoning_effort": effort}
            return {}
        # vLLM / TensorRT-LLM local models (Qwen/Gemma): send enable_thinking
        # EXPLICITLY in both directions. vLLM can default this server-side
        # (start_vllm.sh --default-chat-template-kwargs), but TensorRT-LLM's
        # trtllm-serve has no such flag — relying on a server default leaves
        # thinking ON, which leaks chain-of-thought into content and starves the
        # tool call of tokens. Being explicit is backend-agnostic and matches the
        # vLLM behavior either way.
        enable_thinking = os.environ.get("SCAGENT_THINKING", "0") == "1"
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}}

    def _is_action_tool(self, tool_name: str) -> bool:
        # MCP tools never mutate adata — treat them as inspection tools
        if self._mcp_client and self._mcp_client.has_tool(tool_name):
            return False
        return tool_name in ACTION_TOOL_NAMES

    def _is_inspection_tool(self, tool_name: str) -> bool:
        # MCP tools are always read-only
        if self._mcp_client and self._mcp_client.has_tool(tool_name):
            return True
        return tool_name in INSPECTION_TOOL_NAMES

    def _annotation_validation_guard(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prevent final save/report before annotation consensus is finalized."""
        validation = getattr(self.world_state, "annotation_validation", {}) or {}
        if not validation.get("required"):
            return None
        if validation.get("finalized") or validation.get("status") == "validated_and_finalized":
            return None

        is_final_save = tool_name == "save_data"
        is_final_report_code = False
        if tool_name == "run_code":
            text = " ".join(
                str(tool_input.get(key, ""))
                for key in ("description", "code")
            ).lower()
            final_terms = (
                "write_report",
                "final report",
                "summary report",
                "analysis complete",
                "save final",
                ".write_h5ad",
                "write_h5ad",
            )
            validation_terms = (
                "panglaodb",
                "reference marker",
                "marker validation",
                "validate annotation",
                "validate cell type",
            )
            is_final_report_code = any(term in text for term in final_terms) and not any(
                term in text for term in validation_terms
            )

        if not (is_final_save or is_final_report_code):
            return None

        # Escape hatch: never let the run end with no dataset on disk. Once the
        # agent has genuinely attempted finalize_annotation and it keeps failing
        # validation (or the caller explicitly passes allow_unvalidated), stop
        # hard-blocking and let the save proceed. save_data then degrades it to
        # an honestly-labeled UNVALIDATED file (uns flag + filename suffix +
        # manifest warning) rather than silently saving a "clean" dataset.
        finalize_attempts = int(validation.get("finalize_attempts", 0) or 0)
        allow_unvalidated = is_final_save and bool(tool_input.get("allow_unvalidated"))
        if allow_unvalidated or finalize_attempts >= self.MAX_FINALIZE_ATTEMPTS_BEFORE_UNVALIDATED_SAVE:
            return None

        attempts_note = (
            f" ({finalize_attempts} genuine finalize attempt(s) so far; after "
            f"{self.MAX_FINALIZE_ATTEMPTS_BEFORE_UNVALIDATED_SAVE} the save is allowed as a "
            f"clearly-marked UNVALIDATED file)"
            if finalize_attempts
            else ""
        )
        return {
            "status": "needs_validation",
            "tool": tool_name,
            "message": (
                "Cell-type annotation candidates are present, but the annotation consensus "
                "has not been finalized. Do not save or report the analysis as complete from "
                "CellTypist/Scimilarity/PanglaoDB snippets alone; run the full consensus path "
                "and finalize a curated annotation first." + attempts_note
            ),
            "annotation_validation": validation,
            "required_next_steps": [
                "Run both run_celltypist and run_scimilarity when compatible; if one cannot run, record the concrete reason.",
                "Run run_deg by the primary cluster key if marker DEGs are not already available.",
                "Call prepare_annotation with CellTypist and Scimilarity columns as reference_annotation_keys.",
                "Query PanglaoDB only for clusters flagged as requiring external adjudication, including plausible competitors and staged reverse marker lookup genes.",
                "Use search_papers/web_search as supporting context for ambiguous labels, but PanglaoDB remains the structured external adjudicator in v1.",
                "Stage per-cluster evidence with stage_annotation_evidence, then call finalize_annotation.",
                "If finalize genuinely cannot pass after honest attempts, call save_data with allow_unvalidated=true to write a clearly-marked UNVALIDATED dataset instead of losing the analysis.",
            ],
        }

    def _supports_vision(self) -> bool:
        """True iff the main model's API accepts image_url content.

        Resolution order (first match wins):
          1. SCAGENT_FORCE_TEXT_VISION=1 — test override, always text-only.
          2. SCAGENT_MAIN_HAS_VISION (0/1) — authoritative per-model override.
             This is the reliable knob: a served model name does NOT encode
             modality, so set it in .env whenever the heuristic can't be trusted.
          3. Name allowlist of known-multimodal families. Default is text-only
             (route to the sidecar) when unknown — that degrades gracefully,
             whereas wrongly sending image_url to a text-only server hard-errors.

        Text-only main models (Nemotron, GLM-5.2, DeepSeek, Llama-3.x, most Qwen
        text variants, …) therefore correctly return False and use the sidecar.
        """
        if os.environ.get("SCAGENT_FORCE_TEXT_VISION", "").lower() in ("1", "true", "yes"):
            return False
        override = (os.environ.get("SCAGENT_MAIN_HAS_VISION") or "").strip().lower()
        if override:
            return override in ("1", "true", "yes")
        m = (self.model or "").lower()
        # Known multimodal families (cloud + self-hosted). "-vl" catches the
        # Qwen/Intern *-VL variants generically.
        VISION_FAMILIES = (
            "gpt-4o", "gpt-4-turbo", "gpt-5", "claude", "gemini",
            "gemma-4", "qwen3.6", "qwen3.5", "-vl", "qwen2.5-vl",
            "internvl", "pixtral", "llama-4", "molmo",
        )
        return any(k in m for k in VISION_FAMILIES)

    def _use_sidecar_for_images(self) -> bool:
        """True iff main model is text-only AND a vision sidecar is configured."""
        return (not self._supports_vision()) and self._vision_sidecar is not None

    def _build_sidecar_text_message(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the vision sidecar over pending figures and wrap its output as a user message.

        On sidecar error, falls back to the path-only placeholder used today.
        Emits a run_manager event per call for auditing.
        """
        sidecar = self._vision_sidecar
        paths = ", ".join(img.get("path", "?") for img in images)
        if sidecar is None:
            return {"role": "user", "content": f"Figure(s) saved at {paths}."}

        # Attach any cached image_context for each image so the sidecar can use
        # plot_type / color_by / cluster_key when describing.
        enriched: List[Dict[str, Any]] = []
        for img in images:
            ctx = self._figure_context_index.get(img.get("path") or "", {})
            enriched.append({**img, "image_context": ctx})

        try:
            world_state = self.world_state.snapshot()
        except Exception:
            world_state = None

        result = sidecar.describe(
            enriched,
            world_state=world_state,
            comparative=(len(enriched) > 1),
        )
        if self.run_manager:
            try:
                self.run_manager.append_event(
                    "vision_sidecar_call",
                    {
                        "model": result.get("model"),
                        "n_images": result.get("n_images"),
                        "latency_ms": result.get("latency_ms"),
                        "cache_hits": result.get("cache_hits"),
                        "status": result.get("status"),
                        "paths": [img.get("path") for img in images],
                    },
                )
                self.run_manager.append_log(
                    f"vision_sidecar status={result.get('status')} "
                    f"model={result.get('model')} n_images={result.get('n_images')} "
                    f"latency_ms={result.get('latency_ms')} "
                    f"cache_hits={result.get('cache_hits')} "
                    + (f"error={result.get('error')!r} " if result.get('status') != 'ok' else "")
                    + f"paths={[img.get('path') for img in images]}"
                )
            except Exception:
                pass

        if result.get("status") != "ok" or not result.get("text"):
            return {
                "role": "user",
                "content": (
                    f"Figure(s) saved at {paths}. Vision sidecar unavailable "
                    f"({result.get('error', 'unknown error')})."
                ),
            }

        header = (
            f"[Figure described by vision sidecar — model={result.get('model')}; "
            "main model is text-only]\n"
            f"Path(s): {paths}\n\n"
        )
        footer = (
            "\n\nIf you need to look again or ask a focused question about any of "
            "these figures, call describe_image(figure_path=..., question=\"...\")."
        )
        return {"role": "user", "content": header + result["text"] + footer}

    def _build_image_message(self, images: List[Dict[str, str]], provider: str) -> Dict[str, Any]:
        """Build a user message containing one or more figures with a role-aware prompt."""
        roles = {img.get("role", "figure") for img in images}
        paths_str = ", ".join(img["path"] for img in images)
        has_qc_figure = "qc_figure" in roles or any(
            "qc" in Path(img.get("path", "")).name.lower()
            for img in images
        )
        if has_qc_figure:
            prompt_text = (
                f"Here {'is the QC figure' if len(images) == 1 else 'are the QC figures'} ({paths_str}). "
                "Use it as a quick sanity check for the flag-only QC pass: briefly note whether the distributions "
                "look broadly healthy and whether low-count, low-gene, high-MT, or doublet tails are present. "
                "Do not propose global filtering thresholds, do not ask the user to confirm QC-only filtering, "
                "and do not stop after this visual review. The standard workflow decides removals after "
                "normalization, embedding, clustering, and run_cluster_qc. Continue with the next pipeline tool."
            )
        else:
            n = len(images)
            prompt_text = (
                f"Here {'is the' if n == 1 else 'are the'} generated "
                f"figure{'s' if n > 1 else ''} ({paths_str}). "
                "Interpret it in the context of the current analysis — what does it show, "
                "what are the key observations, and is there anything the user should act on? "
                "If the original user request is an ongoing analysis pipeline and no genuine decision point "
                "has been reached, keep the interpretation brief and continue with the next appropriate tool."
            )

        content: List[Any] = [{"type": "text", "text": prompt_text}]
        for img in images:
            if provider == "anthropic":
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": img["mime"], "data": img["base64"]},
                })
            else:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime']};base64,{img['base64']}"},
                })
        return {"role": "user", "content": content}

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
        checkpoint = dict(checkpoint)
        options = [str(option) for option in checkpoint.get("options", []) or []]
        actions = [str(action) for action in checkpoint.get("option_actions", []) or []]
        if len(actions) != len(options):
            actions = self._stable_option_actions(options)
        checkpoint["options"] = options
        checkpoint["option_actions"] = actions
        checkpoint.setdefault("decision_key", checkpoint.get("kind", "pending_decision"))
        checkpoint.setdefault("allow_custom", True)
        self._pending_checkpoint = checkpoint
        if self.run_manager:
            self.run_manager.append_event("checkpoint_pending", checkpoint)

    @property
    def has_pending_decision(self) -> bool:
        return self._pending_checkpoint is not None

    @staticmethod
    def _stable_option_actions(options: List[str]) -> List[str]:
        """Generate deterministic action ids when a caller supplied labels only."""
        actions: List[str] = []
        seen: Dict[str, int] = {}
        for index, option in enumerate(options, 1):
            slug = re.sub(r"[^a-z0-9]+", "_", option.lower()).strip("_")
            slug = slug[:64] or f"option_{index}"
            seen[slug] = seen.get(slug, 0) + 1
            if seen[slug] > 1:
                slug = f"{slug}_{seen[slug]}"
            actions.append(slug)
        return actions

    @staticmethod
    def _multi_sample_partition_from_result(
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Extract the best sample-like partition from an inspection result."""
        batch = result_data.get("batch") or {}
        candidates = batch.get("candidates") or result_data.get("metadata_candidates") or []
        candidate = candidates[0] if candidates else {}
        column = (
            batch.get("confirmed_batch_key")
            or batch.get("inferred_batch_key")
            or batch.get("recommended_batch_key")
            or candidate.get("column")
        )
        n_groups = int(batch.get("n_batches") or candidate.get("n_unique") or 0)
        if not column or n_groups < 2:
            return None
        return {
            "column": str(column),
            "n_groups": n_groups,
            "role": batch.get("recommended_role") or candidate.get("role") or "sample",
            "status": batch.get("status") or "candidate",
            "needs_key_confirmation": bool(batch.get("needs_confirmation")),
            "reason": batch.get("reason") or candidate.get("rationale") or "",
            "examples": candidate.get("examples") or [],
        }

    def _multi_sample_strategy_checkpoint(
        self,
        partition: Dict[str, Any],
        *,
        post_investigation: bool = False,
    ) -> Dict[str, Any]:
        """Build the user-owned strategy decision for multi-sample data.

        With ``post_investigation=True`` this re-asks the same
        ``multi_sample_strategy`` decision after the uncorrected first pass is
        complete: the ``investigate_integration`` option is dropped (the
        investigation has already run) and the framing asks the user to commit
        to integrate / keep / separate based on the diagnostic. Because it
        reuses ``decision_key="multi_sample_strategy"``, the user's answer
        overwrites the recorded strategy so the downstream guards take over.
        """
        column = partition["column"]
        n_groups = partition["n_groups"]
        experiment_design = self.world_state.get_confirmed_value("experiment_design")
        if post_investigation:
            context = (
                "I finished the uncorrected first pass (PCA → neighbors → UMAP → "
                "clustering) you asked for to investigate whether integration is "
                f"needed for the {n_groups} groups in `{column}`. Review the "
                "sample-colored UMAP and per-cluster sample composition: if clusters "
                "separate by sample beyond the biology you expect, integration is "
                "justified; if the samples already mix, keep them combined. Metadata "
                "alone does not justify correction — this is your call."
            )
        else:
            context = (
                f"I found {n_groups} groups in the {partition.get('role', 'sample')}-like "
                f"column `{column}`. Their presence does not by itself justify batch correction, "
                "so I will not integrate them automatically."
            )
        if experiment_design:
            context += f"\n\nExperiment context you provided:\n{experiment_design}"

        if post_investigation:
            option_specs = [
                ("Integrate the samples with scVI", "integrate_scvi"),
                ("Keep samples combined without integration", "keep_unintegrated"),
                ("Analyze samples separately", "analyze_separately"),
                ("Describe the experiment first", "describe_experiment"),
            ]
            question = "Investigation complete — how should I handle the samples now?"
        else:
            option_specs = [
                ("Investigate whether integration is needed (recommended)", "investigate_integration"),
                ("Integrate the samples with scVI", "integrate_scvi"),
                ("Keep samples combined without integration", "keep_unintegrated"),
                ("Analyze samples separately", "analyze_separately"),
                ("Describe the experiment first", "describe_experiment"),
            ]
            question = "How should I handle these samples?"
        options, option_actions = self._checkpoint_options(option_specs)
        return {
            "kind": "multi_sample_strategy",
            "decision_key": "multi_sample_strategy",
            "question": question,
            "context": context,
            "summary": context,
            "options": options,
            "option_actions": option_actions,
            "default": options[0],
            "recommendation": options[0],
            "allow_custom": True,
            "custom_label": "Type something else...",
            "custom_prompt": "Describe another strategy: ",
            "custom_placeholder": (
                "For example: integrate within each condition, but keep conditions separate"
            ),
            "text_input_actions": {
                "describe_experiment": {
                    "prompt": "Describe the experiment: ",
                    "placeholder": (
                        "Samples, donors, conditions, tissues, protocols, known technical "
                        "batches, and the comparisons that matter. You can paste a table."
                    ),
                }
            },
            "partition": partition,
            "action_inputs": {
                "investigate_integration": {
                    "batch_key": column,
                    "mode": "progressive",
                },
                "integrate_scvi": {
                    "batch_key": column,
                    "method": "scvi",
                    "batch_key_needs_confirmation": partition.get("needs_key_confirmation", False),
                },
                "keep_unintegrated": {"batch_key": column},
                "analyze_separately": {"sample_key": column},
            },
            "artifacts": [],
        }

    def _build_multi_sample_strategy_checkpoint(
        self,
        tool_name: str,
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if tool_name != "inspect_data" or result_data.get("status") != "ok":
            return None
        if self.world_state.get_confirmed_value("multi_sample_strategy"):
            return None
        if (result_data.get("batch") or {}).get("batch_correction_applied"):
            return None
        partition = self._multi_sample_partition_from_result(result_data)
        if partition is None:
            return None
        return self._multi_sample_strategy_checkpoint(partition)

    def _post_investigation_strategy_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Re-ask the integration decision once the investigation first pass is done.

        Fires when the recorded ``multi_sample_strategy`` is
        ``investigate_integration`` and the uncorrected first pass has produced
        a structured batch-effect diagnostic without applying correction.
        Returning the user to a concrete integrate / keep / separate choice is
        what stops the agent from treating "I investigated and concluded
        integration is needed" as self-authorization to integrate on its own.
        """
        selected_strategy = self.world_state.get_confirmed_value("multi_sample_strategy")
        strategy_action = (
            selected_strategy.get("action")
            if isinstance(selected_strategy, dict)
            else selected_strategy
        )
        if strategy_action != "investigate_integration":
            return None
        ds = self.world_state.data_summary or {}
        if ds.get("batch_correction_applied"):
            return None
        diagnostic = {}
        if getattr(self, "adata", None) is not None:
            try:
                diagnostic = dict(self.adata.uns.get("batch_effect_diagnostic") or {})
            except Exception:
                diagnostic = {}
        if diagnostic.get("status") != "ok":
            return None
        column = ds.get("batch_key") or ds.get("recommended_batch_key")
        column = column or diagnostic.get("batch_key")
        n_groups = int(ds.get("n_batches") or 0)
        if not n_groups:
            n_groups = int(diagnostic.get("n_batches") or 0)
        if not column or n_groups < 2:
            return None
        checkpoint = self._multi_sample_strategy_checkpoint(
            {
                "column": str(column),
                "n_groups": n_groups,
                "role": "sample",
                "needs_key_confirmation": False,
            },
            post_investigation=True,
        )
        evidence_bits = []
        verdict = diagnostic.get("verdict")
        if verdict:
            evidence_bits.append(f"Diagnostic verdict: {verdict}.")
        recommendation = diagnostic.get("recommendation")
        if recommendation:
            evidence_bits.append(f"Recommendation: {recommendation}")
        support = diagnostic.get("support_reasons") or []
        if support:
            evidence_bits.append("Support: " + "; ".join(map(str, support[:4])) + ".")
        cautions = diagnostic.get("caution_reasons") or []
        if cautions:
            evidence_bits.append("Cautions: " + "; ".join(map(str, cautions[:4])) + ".")
        shared = diagnostic.get("shared_cross_cell_type_signatures") or []
        if shared:
            evidence_bits.append(
                f"Shared sample-associated signatures: {len(shared)} recurring gene/direction entries."
            )
        cluster_summary = diagnostic.get("cluster_sample_summary") or {}
        if cluster_summary:
            evidence_bits.append(
                "Sample-dominated clusters: "
                f"{cluster_summary.get('n_sample_dominated_clusters', 0)} "
                f"({cluster_summary.get('fraction_cells_in_sample_dominated_clusters', 0)} of cells)."
            )
        if evidence_bits:
            context = checkpoint.get("context") or ""
            checkpoint["context"] = context + "\n\nBatch-effect diagnostic evidence:\n" + "\n".join(
                f"- {bit}" for bit in evidence_bits
            )
            checkpoint["summary"] = checkpoint["context"]
        checkpoint["diagnostic"] = {
            "verdict": diagnostic.get("verdict"),
            "recommendation": diagnostic.get("recommendation"),
            "support_reasons": support[:6],
            "caution_reasons": cautions[:6],
            "evidence_limits": (diagnostic.get("evidence_limits") or [])[:6],
            "artifacts_created": diagnostic.get("artifacts_created") or [],
        }
        if diagnostic.get("verdict") == "batch_effect_supported":
            checkpoint["recommendation"] = checkpoint["options"][0]
        elif diagnostic.get("verdict") == "no_correction_needed" and len(checkpoint["options"]) > 1:
            checkpoint["recommendation"] = checkpoint["options"][1]
        else:
            checkpoint["recommendation"] = checkpoint["options"][0]
        return checkpoint

    def _build_post_concatenation_strategy_checkpoint(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Open the integration decision immediately after a successful concat."""
        if tool_name != "run_code" or result_data.get("status") != "ok":
            return None
        code = str(tool_input.get("code") or "")
        if not re.search(r"\b(?:anndata|ad)\.concat\s*\(|\bconcat_datasets\s*\(", code):
            return None
        if self.world_state.get_confirmed_value("multi_sample_strategy"):
            return None

        summary = self.world_state.data_summary or {}
        candidates = self.world_state.metadata_candidates or []
        candidate = candidates[0] if candidates else {}

        # Prefer the batch column the concat code itself named (anndata.concat
        # uses label=, concat_datasets uses batch_key=), then fall back to
        # anything a prior inspect_data recorded in world_state. Relying only on
        # world_state fails when the concat runs before any successful
        # inspect_data — which is exactly the case this safety net must cover.
        column = None
        m = re.search(r"\b(?:label|batch_key)\s*=\s*['\"]([^'\"]+)['\"]", code)
        if m:
            column = m.group(1)
        column = (
            column
            or summary.get("batch_key")
            or summary.get("recommended_batch_key")
            or candidate.get("column")
        )

        # Resolve the group count from the live concatenated AnnData (the source
        # of truth right after the concat), falling back to recorded metadata.
        # If we still have no column, sniff obs for a sample-like column the
        # concat may have created (e.g. anndata.concat's default 'batch' label).
        n_groups = 0
        obs = getattr(getattr(self, "adata", None), "obs", None)
        if obs is not None:
            if not column:
                for cand_col in ("batch", "sample", "replicate", "donor", "library", "dataset"):
                    if cand_col in obs.columns:
                        column = cand_col
                        break
            if column and column in obs.columns:
                try:
                    n_groups = int(obs[column].nunique())
                except Exception:
                    n_groups = 0
        if not n_groups:
            n_groups = int(summary.get("n_batches") or candidate.get("n_unique") or 0)

        if not column or n_groups < 2:
            return None
        return self._multi_sample_strategy_checkpoint({
            "column": str(column),
            "n_groups": n_groups,
            "role": candidate.get("role") or "sample",
            "status": "post_concatenation",
            "needs_key_confirmation": False,
            "reason": candidate.get("rationale") or "",
            "examples": candidate.get("examples") or [],
        })

    def _multi_dataset_loading_checkpoint(
        self,
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Ask how multiple files should become analysis objects before loading."""
        source_datasets = result_data.get("source_datasets") or []
        if len(source_datasets) < 2:
            return None
        if self.world_state.get_confirmed_value("multi_dataset_loading_strategy"):
            return None

        names = [str(dataset.get("name") or dataset.get("path")) for dataset in source_datasets]
        preview = "\n".join(f"- {name}" for name in names[:8])
        if len(names) > 8:
            preview += f"\n- ... and {len(names) - 8} more"
        context = (
            f"I found {len(source_datasets)} source datasets.\n\n{preview}\n\n"
            "Outer join is the recommended default for compatible replicate matrices: "
            "it keeps the union of genes and fills genes absent from a dataset with zero. "
            "If these datasets use different assays, panels, or feature definitions, "
            "separate analysis or custom handling may be safer."
        )
        likely_outputs = result_data.get("likely_combined_outputs") or []
        if likely_outputs:
            output_names = ", ".join(
                str(item.get("name") or item.get("path")) for item in likely_outputs[:4]
            )
            context += (
                "\n\nI also found file(s) that look like previous combined outputs and "
                f"excluded them from the source count: {output_names}."
            )

        options, option_actions = self._checkpoint_options([
            (
                "Concatenate with an outer join (recommended; keep all genes from all datasets)",
                "concatenate_outer",
            ),
            (
                "Concatenate with an inner join (keep only genes shared by every dataset)",
                "concatenate_inner",
            ),
            ("Analyze each dataset separately", "analyze_separately"),
        ])
        return {
            "kind": "multi_dataset_loading",
            "decision_key": "multi_dataset_loading_strategy",
            "question": "How should I handle these datasets before analysis?",
            "context": context,
            "summary": context,
            "options": options,
            "option_actions": option_actions,
            "default": options[0],
            "recommendation": options[0],
            "allow_custom": True,
            "custom_label": "Type something else...",
            "custom_prompt": "Describe how these datasets should be handled: ",
            "custom_placeholder": (
                "For example: concatenate Rep1 and Rep2, but analyze the control separately"
            ),
            "datasets": source_datasets,
            "action_inputs": {
                "concatenate_outer": {
                    "join": "outer",
                    "keep_genes": "union",
                    "datasets": source_datasets,
                },
                "concatenate_inner": {
                    "join": "inner",
                    "keep_genes": "intersection",
                    "datasets": source_datasets,
                },
                "analyze_separately": {
                    "datasets": source_datasets,
                },
            },
            "artifacts": [],
        }

    def _build_multi_dataset_loading_checkpoint(
        self,
        tool_name: str,
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if tool_name != "inspect_data_inputs" or result_data.get("status") != "ok":
            return None
        return self._multi_dataset_loading_checkpoint(result_data)

    def _checkpoint_options(
        self,
        entries: List[tuple[str, str]],
    ) -> tuple[List[str], List[str]]:
        options = [label for label, _ in entries]
        actions = [action for _, action in entries]
        return options, actions

    def _clear_pending_checkpoint(self, user_response: Any = None) -> None:
        if self._pending_checkpoint and self.run_manager:
            payload = dict(self._pending_checkpoint)
            if user_response is not None:
                payload["user_response"] = user_response
            self.run_manager.append_event("checkpoint_resolved", payload)
        self._pending_checkpoint = None

    def prompt_pending_decision(self, *, force_text_fallback: bool = False):
        """Render the current checkpoint and return a normalized selection."""
        if not self._pending_checkpoint:
            return None
        from ..terminal import DecisionChoice, prompt_for_decision

        checkpoint = self._pending_checkpoint
        options = checkpoint.get("options", []) or []
        actions = checkpoint.get("option_actions", []) or []
        text_input_actions = checkpoint.get("text_input_actions") or {}
        choices = []
        for index, label in enumerate(options):
            action = actions[index]
            text_input = text_input_actions.get(action) or {}
            choices.append(
                DecisionChoice(
                    label=label,
                    action=action,
                    requires_text=bool(text_input),
                    text_prompt=text_input.get("prompt", "Your response: "),
                    placeholder=text_input.get("placeholder", ""),
                )
            )
        default = checkpoint.get("default")
        default_index = options.index(default) if default in options else 0
        context = str(checkpoint.get("context") or checkpoint.get("summary") or "").strip()
        question = str(checkpoint.get("question") or "How should I proceed?").strip()
        if context:
            question = f"{context}\n\n{question}"
        return prompt_for_decision(
            question,
            choices,
            default_index=default_index,
            allow_custom=bool(checkpoint.get("allow_custom", True)),
            custom_label=checkpoint.get("custom_label", "Type something else..."),
            custom_prompt=checkpoint.get("custom_prompt", "Your response: "),
            custom_placeholder=checkpoint.get("custom_placeholder", ""),
            force_text_fallback=force_text_fallback,
        )

    def resolve_pending_decision_text(self, response: str):
        """Resolve a plain-text reply against the newest pending decision."""
        if not self._pending_checkpoint:
            return None
        from ..terminal import DecisionChoice, resolve_decision_response

        checkpoint = self._pending_checkpoint
        options = checkpoint.get("options", []) or []
        actions = checkpoint.get("option_actions", []) or []
        choices = [
            DecisionChoice(label=label, action=actions[index])
            for index, label in enumerate(options)
        ]
        default = checkpoint.get("default")
        default_index = options.index(default) if default in options else None
        return resolve_decision_response(
            response,
            choices,
            default_index=default_index,
            allow_custom=bool(checkpoint.get("allow_custom", True)),
        )

    def resolve_pending_decision(self, selection) -> Dict[str, Any]:
        """Commit a normalized selection and return the payload given to the model."""
        if not self._pending_checkpoint:
            raise RuntimeError("No pending decision to resolve.")
        checkpoint = dict(self._pending_checkpoint)
        selected_action = selection.action
        selected_value = selection.value

        if checkpoint.get("kind") == "cluster_qc_cleanup":
            self._authorize_pending_cleanup_from_user(selected_action)

        decision_key = checkpoint.get("decision_key", checkpoint.get("kind", "pending_decision"))
        reprompt_checkpoint = None
        if (
            checkpoint.get("kind") == "multi_sample_strategy"
            and selected_action == "describe_experiment"
        ):
            self.world_state.resolve_decision(
                "experiment_design",
                selected_value,
                source="user",
                message=selected_value,
            )
            self.world_state.add_context_hint(f"Experiment design: {selected_value}")
            decision_key = "experiment_design"
            reprompt_checkpoint = self._multi_sample_strategy_checkpoint(
                checkpoint.get("partition") or {}
            )
        else:
            applied_value: Any = selected_action or selected_value
            if selected_action == "custom":
                applied_value = {
                    "action": "custom",
                    "details": selected_value,
                }
            self.world_state.resolve_decision(
                decision_key,
                applied_value,
                source="user",
                message=selected_value,
            )
        payload = {
            "decision_key": decision_key,
            "checkpoint_kind": checkpoint.get("kind"),
            "question": checkpoint.get("question", ""),
            "selected_action": selected_action,
            "selected_label": selection.label,
            "selected_index": selection.index,
            "selected_value": selected_value,
            "raw_response": selection.raw_response,
            "input_mode": selection.input_mode,
            "custom": selection.custom,
            "action_input": (checkpoint.get("action_inputs") or {}).get(selected_action),
            "context": checkpoint.get("context") or checkpoint.get("summary") or "",
        }
        if checkpoint.get("proposal") is not None:
            payload["proposal"] = checkpoint["proposal"]
        if self._pending_checkpoint is not None:
            self._clear_pending_checkpoint(payload)
        if reprompt_checkpoint is not None:
            payload["reprompt"] = True
            self._set_pending_checkpoint(reprompt_checkpoint)
        return payload

    def structured_decision_request(self, selection) -> str:
        """Turn a selector result into an unambiguous model-facing user message."""
        payload = self.resolve_pending_decision(selection)
        instruction = (
            "Treat selected_action as authoritative. Carry out that choice, using "
            "selected_value as the user's text only when custom is true."
        )
        if payload.get("selected_action") == "investigate_integration":
            instruction += (
                " For investigate_integration, run the uncorrected first pass only "
                "(PCA, neighbors, UMAP, clustering), then call diagnose_batch_effect "
                "with the selected batch_key. Do not run batch correction until the "
                "post-diagnostic selector records a new integration decision."
            )
        return (
            "[Structured user decision]\n"
            f"{json.dumps(payload, indent=2, default=str)}\n\n"
            f"{instruction}"
        )

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
            qc_input["confirm_filtering"] = True
            run_step("run_qc", qc_input)
            return {"selected_action": selected_action, "steps": steps}

        if selected_action == "run_normalize_and_hvg":
            normalize_input = dict(action_inputs.get("run_normalize_and_hvg", {}))
            run_step("normalize_and_hvg", normalize_input)
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

        if selected_action == "run_cluster_qc":
            cluster_qc_input = dict(action_inputs.get("run_cluster_qc", {}))
            run_step("run_cluster_qc", cluster_qc_input)
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
        "record_inspection",  # Read-only judgment record; never mutates adata.
        "inspect_data_inputs",
        "inspect_session",
        "list_artifacts",
        "get_cluster_sizes",
        "get_top_markers",
        "summarize_qc_metrics",
        "get_celltypes",
        "list_obs_columns",
        "review_figure",
        "review_artifact",
        "run_cluster_structure_qc",  # Refines an existing cleanup checkpoint.
        "diagnose_batch_effect",  # Produces evidence before the post-investigation selector.
        "generate_figure",  # Visualization doesn't change state
        "write_report",  # Writing a report doesn't change state
        "write_json",  # Writing a JSON file doesn't change state
        "save_data",  # Saving is always ok
        "read_file",
        "search_papers",
        "fetch_url",
        "web_search",
        "research_findings",
    }

    def _multi_dataset_loading_guard(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Optional[str]:
        """Prevent silent concatenation or a join that differs from the user's choice."""
        if tool_name != "run_code":
            return None
        code = str(tool_input.get("code") or "")
        concatenates = bool(re.search(
            r"\b(?:anndata|ad)\.concat\s*\(|\bconcat_datasets\s*\(",
            code,
        ))
        if not concatenates:
            return None

        strategy = self.world_state.get_confirmed_value("multi_dataset_loading_strategy")
        action = strategy.get("action") if isinstance(strategy, dict) else strategy
        if not action:
            return json.dumps({
                "status": "error",
                "tool": "run_code",
                "message": (
                    "Multiple datasets cannot be concatenated before the user chooses how "
                    "to handle them. Run inspect_data_inputs on the input directory first."
                ),
                "requires_user_decision": True,
                "decision_key": "multi_dataset_loading_strategy",
            }, indent=2)
        if action == "analyze_separately":
            return json.dumps({
                "status": "error",
                "tool": "run_code",
                "message": "The user selected separate analyses, so concatenation is not allowed.",
                "selected_strategy": action,
            }, indent=2)

        expected_join = {
            "concatenate_outer": "outer",
            "concatenate_inner": "inner",
        }.get(action)
        if expected_join:
            join_pattern = rf"\bjoin\s*=\s*['\"]{expected_join}['\"]"
            if not re.search(join_pattern, code):
                return json.dumps({
                    "status": "error",
                    "tool": "run_code",
                    "message": (
                        f"The user selected a {expected_join} join. The concatenation code "
                        f"must explicitly pass join='{expected_join}'."
                    ),
                    "selected_strategy": action,
                    "required_join": expected_join,
                }, indent=2)
        return None

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
                "required_next_action": "resolve_pending_decision",
            },
            indent=2,
        )

    def _build_checkpoint_payload(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.collaborative or self.smart_autonomous or result_data.get("status") != "ok":
            return None

        checkpoint: Optional[Dict[str, Any]] = None
        artifacts = self._checkpoint_artifact_paths(result_data)

        if tool_name == "run_qc":
            if tool_input.get("preview_only", False):
                # Get doublet count from the correct location
                qc_decisions = result_data.get("qc_decisions", {})
                filtering_plan = result_data.get("filtering_plan", {})
                plan_params = filtering_plan.get("parameters", {})
                projected = filtering_plan.get("projected_after_filtering", {})
                mt_threshold = plan_params.get("mt_threshold", tool_input.get("mt_threshold", "auto"))
                min_cells = plan_params.get("min_cells_per_gene")
                min_genes = plan_params.get("min_genes")
                doublet_count = int(qc_decisions.get("doublet_detection", {}).get("cells_flagged", 0))
                mt_decision = qc_decisions.get("mt_threshold", {})
                filter_mt = bool(mt_decision.get("filter_enabled", True))
                high_mt_cells = int(mt_decision.get("cells_flagged", 0))
                min_genes_cells = int(qc_decisions.get("min_genes", {}).get("cells_flagged", 0))
                before_cells = result_data.get("before", {}).get("n_cells", 0)
                projected_removals = projected.get("cells_removed")
                projected_cells = projected.get("cells_retained")
                if projected_cells is None:
                    projected_cells = before_cells - projected_removals if before_cells and projected_removals else "?"
                projected_genes_removed = projected.get("genes_removed_before_cell_filtering")
                question = (
                    "QC preview is complete. I summarized the thresholds, parameters, and proposed removals. "
                    "What should I do next?"
                )
                options, option_actions = self._checkpoint_options([
                    (
                        (
                            "Apply the proposed QC filters "
                            f"({projected_removals} cells and ~{projected_genes_removed} genes projected for removal)"
                        ),
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
                    f"QC preview: parameters mt_threshold={mt_threshold}, "
                    f"min_genes={min_genes}, min_cells_per_gene={min_cells}; "
                    f"{min_genes_cells} low-gene cells flagged, "
                    f"{high_mt_cells} high-MT cells {'to remove' if filter_mt else 'reported only'}, "
                    f"{doublet_count} doublets flagged. Estimated {projected_cells} cells retained "
                    f"and ~{projected_genes_removed} genes removed before cell filtering."
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
                            "filter_mt": tool_input.get("filter_mt", True),
                            "min_genes": tool_input.get("min_genes"),
                            "min_cells": tool_input.get("min_cells"),
                            "remove_ribo": tool_input.get("remove_ribo", True),
                            "remove_mt": tool_input.get("remove_mt", False),
                            "detect_doublets_flag": tool_input.get("detect_doublets_flag", True),
                            "remove_doublets": tool_input.get("remove_doublets", False),
                            "scrublet_expected_doublet_rate": tool_input.get("scrublet_expected_doublet_rate"),
                            "scrublet_sim_doublet_ratio": tool_input.get("scrublet_sim_doublet_ratio"),
                            "scrublet_n_prin_comps": tool_input.get("scrublet_n_prin_comps"),
                            "scrublet_min_counts": tool_input.get("scrublet_min_counts"),
                            "scrublet_min_cells": tool_input.get("scrublet_min_cells"),
                            "scrublet_min_gene_variability_pctl": tool_input.get("scrublet_min_gene_variability_pctl"),
                            "scrublet_random_state": tool_input.get("scrublet_random_state"),
                            "force_doublet_recompute": tool_input.get("force_doublet_recompute", False),
                            "batch_key": tool_input.get("batch_key"),
                            "confirm_filtering": True,
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
                ("Run PCA", "run_pca"),
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
                    "run_pca": {},
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
                selected_strategy = self.world_state.get_confirmed_value("multi_sample_strategy")
                strategy_action = (
                    selected_strategy.get("action")
                    if isinstance(selected_strategy, dict)
                    else selected_strategy
                )
                if strategy_action == "investigate_integration":
                    try:
                        diagnostic_done = bool(
                            getattr(self, "adata", None) is not None
                            and self.adata.uns.get("batch_effect_diagnostic", {}).get("status") == "ok"
                        )
                    except Exception:
                        diagnostic_done = False
                    if not diagnostic_done:
                        return None
                post_investigation = self._post_investigation_strategy_checkpoint()
                if post_investigation is not None:
                    return post_investigation
                cluster_key = result_data.get("cluster_key", "clustering")
                n_clusters = result_data.get("n_clusters", "?")
                summary = f"Clustering produced {n_clusters} clusters in '{cluster_key}'."
                processing = self.world_state.data_summary.get("processing", {}) if self.world_state else {}
                qc_ready = bool(processing.get("has_qc_metrics"))
                cluster_qc_needed = qc_ready
                if cluster_qc_needed:
                    options, option_actions = self._checkpoint_options([
                        ("Run cluster-level QC before annotation", "run_cluster_qc"),
                        ("Compare alternative clustering resolutions before QC", "compare_clusterings"),
                        ("Proceed without cluster-level QC for this clustering", "run_annotation"),
                        ("Something else", "custom"),
                    ])
                    default_action_input = {
                        "run_cluster_qc": {
                            "cluster_key": result_data.get("primary_cluster_key") or cluster_key,
                        }
                    }
                    question = "Clustering is complete. Should I run cluster-level QC before continuing?"
                    recommendation = options[0]
                else:
                    options, option_actions = self._checkpoint_options([
                        ("Proceed to cell type annotation", "run_annotation"),
                        ("Compare alternative clustering resolutions before annotating", "compare_clusterings"),
                        ("Run marker analysis or cluster-size review before annotating", "review_cluster_markers"),
                        ("Something else", "custom"),
                    ])
                    default_action_input = {}
                    question = "Clustering is complete. What should I do next?"
                    recommendation = options[0]
                checkpoint = {
                    "kind": "clustering",
                    "question": question,
                    "options": options,
                    "default": options[0],
                    "decision_key": "clustering_next_step",
                    "summary": summary,
                    "recommendation": recommendation,
                    "option_actions": option_actions,
                    "action_inputs": {
                        **default_action_input,
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

        elif tool_name == "diagnose_batch_effect":
            post_investigation = self._post_investigation_strategy_checkpoint()
            if post_investigation is not None:
                post_investigation["artifacts"] = artifacts
                return post_investigation

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
            summary = f"Applied {method} batch correction using '{batch_key}'. UMAP must be recomputed before plotting or clustering."
            options, option_actions = self._checkpoint_options([
                ("Recompute UMAP from the corrected graph", "run_umap"),
                ("Inspect batch mixing quality before computing UMAP", "inspect_batch_mixing"),
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
                    "run_umap": {},
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
                ("Run PCA, then neighbors, then UMAP", "run_pca"),
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
                "action_inputs": {"run_pca": {}},
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
        prompt = SYSTEM_PROMPT
        if self._is_gemma_model():
            # Gemma 4 puts all output inside thinking blocks and produces no narration
            # text outside them. This instruction mirrors how Claude/GPT behave: brief
            # narration sentence after thinking, before the tool call.
            prompt += (
                "\n\n## Narration Requirement\n"
                "After your thinking block, always write one brief sentence describing "
                "what you are about to do (e.g. 'I'll inspect the data to check QC metrics.'), "
                "then call the appropriate tool. Do not include this sentence inside the "
                "thinking block — write it as plain response text after the closing tag."
            )
        if self._is_gemini_model():
            # Gemini via OpenAI-compatible API often returns tool calls with no text content.
            # Instruct it to narrate before each tool call so users see reasoning steps.
            prompt += (
                "\n\n## Narration Requirement\n"
                "Before each tool call, always write one brief sentence describing what you "
                "are about to do (e.g. 'I'll load the data to inspect its structure.'). "
                "Write this as plain text content in the same response as the tool call."
            )
        if self.smart_autonomous:
            prompt += _SMART_AUTONOMOUS_PROMPT
        if os.environ.get("SCAGENT_MODEL_INSPECTION") == "1":
            from .prompts import MODEL_INSPECTION_PROMPT
            prompt += MODEL_INSPECTION_PROMPT
        if self._use_sidecar_for_images():
            sidecar_model = self._vision_sidecar.model if self._vision_sidecar else "(unconfigured)"
            prompt += (
                "\n\n## Figure Handling (text-only main model + vision sidecar active)\n"
                "- You cannot see figures directly. A separate vision model "
                f"(`{sidecar_model}`) describes any figure produced by a tool. The "
                "description is injected into the conversation as a user message with "
                "sections WHAT_THIS_IS / KEY_OBSERVATIONS / NUMBERS_VISIBLE / ANOMALIES "
                "/ ACTIONABLE_FLAGS / OPEN_QUESTIONS. Treat that description as your "
                "view of the figure.\n"
                "- To re-inspect a figure or ask a focused question (e.g. 'do clusters "
                "4 and 7 separate by batch?'), call `describe_image(figure_path=..., "
                "question=\"...\")`.\n"
                "- Do not claim you 'see' the figure — you have a faithful textual "
                "description. If a description is missing a detail you need, request "
                "it via `describe_image`."
            )
        return f"{prompt}\n\n## Runtime Interaction State\n{self._runtime_guidance()}"

    def _current_capabilities(self) -> Dict[str, Any]:
        return (self.world_state.data_summary or {}).get("capabilities", {})

    def _checkpoint_action_from_user_response(
        self,
        checkpoint: Dict[str, Any],
        value: str,
    ) -> Optional[str]:
        """Resolve common natural replies to a pending checkpoint action."""
        text = " ".join((value or "").strip().lower().split())
        text = text.strip(" .,!?:;")
        option_actions = checkpoint.get("option_actions") or []
        options = checkpoint.get("options") or []

        for action in option_actions:
            if text == str(action).strip().lower():
                return action

        numbered_match = re.match(r"^(?:option|choice|number|#)?\s*([1-9][0-9]*)\b", text)
        if text.isdigit() or numbered_match:
            number = numbered_match.group(1) if numbered_match else text
            index = int(number) - 1
            if 0 <= index < len(option_actions):
                return option_actions[index]

        ordinal_index = {
            "first": 0,
            "1st": 0,
            "one": 0,
            "second": 1,
            "2nd": 1,
            "two": 1,
            "third": 2,
            "3rd": 2,
            "three": 2,
            "fourth": 3,
            "4th": 3,
            "four": 3,
        }
        for ordinal, index in ordinal_index.items():
            if re.match(rf"^(?:(?:option|choice|number)\s+)?{ordinal}\b", text):
                if 0 <= index < len(option_actions):
                    return option_actions[index]

        request_words = {w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 2}
        for option, action in zip(options, option_actions):
            option_text = " ".join(str(option).strip().lower().split()).strip(" .,!?:;")
            if text and text == option_text:
                return action
            option_words = {w for w in re.findall(r"[a-z0-9]+", option_text) if len(w) > 2}
            if option_words and len(option_words & request_words) >= min(3, len(option_words)):
                return action
        return None

    def _is_strict_cleanup_yes(self, value: str) -> bool:
        text = " ".join((value or "").strip().lower().split())
        text = text.strip(" .,!?:;")
        if text in {
            "y",
            "yes",
            "1",
            "ok",
            "okay",
            "sure",
            "do it",
            "remove",
            "remove them",
            "remove it",
            "go ahead",
            "proceed with removal",
        }:
            return True
        if re.match(r"^(yes|y|ok|okay|sure|go ahead|do it)\b", text):
            return not re.search(r"\b(no|not|don't|dont|keep)\b", text)
        return bool(
            re.search(r"\b(remove|drop|filter|exclude)\b", text)
            and not re.search(r"\b(no|not|don't|dont|keep|without removing)\b", text)
        )

    def _is_cleanup_no(self, value: str) -> bool:
        text = " ".join((value or "").strip().lower().split())
        text = text.strip(" .,!?:;")
        if text in {
            "n",
            "no",
            "nope",
            "keep",
            "keep them",
            "keep it",
            "do not remove",
            "don't remove",
            "dont remove",
            "proceed without removing",
        }:
            return True
        return bool(
            re.match(r"^(no|n|nope)\b", text)
            or re.search(r"\b(keep|do not remove|don't remove|dont remove|without removing)\b", text)
        )

    def _mentioned_cluster_labels(self, value: str) -> set[str]:
        text = (value or "").lower()
        labels: set[str] = set()
        for match in re.finditer(
            r"\bclusters?\s+([0-9a-z_,\s]+?)(?=\b(?:or|and|then|before|after|to|from|with|but|because|instead|$))",
            text,
        ):
            chunk = match.group(1)
            labels.update(re.findall(r"\b[0-9]+[a-z]?\b", chunk))
        return labels

    def _cleanup_policy(self) -> Dict[str, Any]:
        policy = self.world_state.get_confirmed_value("cluster_cleanup_policy")
        if isinstance(policy, dict):
            return policy
        return {
            "mode": "confirm",
            "max_pct_without_confirmation": 0.0,
            "stop_for_ambiguous": True,
            "require_exact_count": True,
        }

    def _is_auto_cleanup_allowed(self, proposal: Dict[str, Any]) -> tuple[bool, str]:
        policy = self._cleanup_policy()
        if policy.get("mode") != "auto_obvious":
            return False, "cluster cleanup policy requires user confirmation"
        if proposal.get("ambiguous"):
            return False, "ambiguous clusters require confirmation"
        proposed = proposal.get("proposed_removal") or []
        if not proposed:
            return False, "no proposed removals"
        max_pct = float(policy.get("max_pct_without_confirmation", 5.0))
        pct = float(proposal.get("pct_proposed") or 0.0)
        if pct > max_pct:
            return False, f"proposed removal {pct:.1f}% exceeds auto-cleanup limit {max_pct:.1f}%"
        cluster_decisions = proposal.get("cluster_decisions") or {}
        not_obvious = []
        for cluster in proposed:
            decision = cluster_decisions.get(str(cluster), {})
            if (
                decision.get("recommended_action") != "propose_removal"
                or decision.get("severity") != "obvious"
            ):
                not_obvious.append(str(cluster))
        if not_obvious:
            return False, f"cluster cleanup requires confirmation for: {', '.join(not_obvious)}"
        return True, "user granted auto-cleanup for obvious cluster-level QC removals"

    def _cluster_cleanup_checkpoint_from_result(self, result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        proposed = [str(cluster) for cluster in result_data.get("proposed_removal", []) or []]
        if not proposed:
            return None
        cells = int(result_data.get("cells_in_proposed_removal") or 0)
        pct = float(result_data.get("pct_proposed") or 0.0)
        cluster_key = result_data.get("cluster_key") or self.world_state.data_summary.get("cluster_key") or "leiden"
        cluster_decisions = result_data.get("cluster_decisions") or {}
        proposal = {
            "cluster_key": cluster_key,
            "proposed_removal": proposed,
            "cluster_decisions": cluster_decisions,
            "cells_in_proposed_removal": cells,
            "pct_proposed": pct,
            "cells_remaining_if_removed": result_data.get("cells_remaining_if_removed"),
            "ambiguous": [str(cluster) for cluster in result_data.get("ambiguous", []) or []],
            "n_clusters": result_data.get("n_clusters"),
            "thresholds_used": result_data.get("thresholds_used", {}),
            "checkpoint_path": result_data.get("checkpoint_path"),
            "cluster_table": result_data.get("cluster_table", []),
        }
        auto_allowed, auto_reason = self._is_auto_cleanup_allowed(proposal)
        options, option_actions = self._checkpoint_options([
            ("Remove the proposed low-quality clusters and rerun embedding", "remove_proposed_clusters"),
            ("Keep the proposed clusters and proceed", "keep_proposed_clusters"),
            ("Inspect cluster QC details before deciding", "review_cluster_qc"),
            ("Something else", "custom"),
        ])
        summary = (
            f"Cluster QC proposes removing {len(proposed)} cluster(s) "
            f"({', '.join(proposed)}) from '{cluster_key}', totaling {cells} cells "
            f"({pct:.1f}%)."
        )
        return {
            "kind": "cluster_qc_cleanup",
            "question": "Cluster QC found low-quality clusters. Should I remove them before continuing?",
            "options": options,
            "default": options[0],
            "decision_key": "cluster_qc_cleanup",
            "summary": summary,
            "recommendation": options[0],
            "option_actions": option_actions,
            "proposal": proposal,
            "auto_allowed": auto_allowed,
            "auto_reason": auto_reason,
            "requires_user_confirmation": not auto_allowed,
        }

    def _cluster_structure_qc_required_after_metric_qc(self, result_data: Dict[str, Any]) -> bool:
        if result_data.get("status") != "ok":
            return False
        proposed = result_data.get("proposed_removal") or []
        ambiguous = result_data.get("ambiguous") or []
        return bool(proposed or ambiguous)

    def _auto_structure_cleanup_allowed(self, proposal: Dict[str, Any]) -> tuple[bool, str]:
        proposed = [str(cluster) for cluster in proposal.get("proposed_removal", []) or []]
        if not proposed:
            return False, "no structure-synthesized removal candidates"
        pct = float(proposal.get("pct_proposed") or 0.0)
        max_pct = 15.0
        if pct >= max_pct:
            return False, f"structure-synthesized removal {pct:.1f}% is at or above the {max_pct:.1f}% pause threshold"
        return True, (
            "Structure QC synthesized a removal set below the 15% pause threshold; "
            "proceeding with the evidence-supported cleanup after saving the pre-cleanup checkpoint."
        )

    def _refine_cluster_cleanup_checkpoint_from_structure(
        self,
        result_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        checkpoint = self._pending_checkpoint or {}
        if checkpoint.get("kind") == "cluster_qc_cleanup":
            proposal = dict(checkpoint.get("proposal") or {})
        else:
            cluster_key = result_data.get("cluster_key") or self.world_state.data_summary.get("cluster_key") or "leiden"
            metric_record = (self.world_state.cluster_qc_registry or {}).get(str(cluster_key), {})
            proposal = {
                "cluster_key": cluster_key,
                "proposed_removal": [str(c) for c in metric_record.get("proposed_removal", []) or []],
                "cluster_decisions": metric_record.get("cluster_decisions", {}),
                "cells_in_proposed_removal": metric_record.get("cells_in_proposed_removal"),
                "pct_proposed": metric_record.get("pct_proposed"),
                "ambiguous": [str(c) for c in metric_record.get("ambiguous", []) or []],
                "thresholds_used": metric_record.get("thresholds_used", {}),
                "cluster_table": metric_record.get("cluster_table", []),
            }
        metric_proposed = [
            str(cluster)
            for cluster in proposal.get("metric_proposed_removal", proposal.get("proposed_removal", [])) or []
        ]
        synthesized = [str(cluster) for cluster in result_data.get("synthesized_removal", []) or []]
        cells = int(result_data.get("cells_in_synthesized_removal") or 0)
        pct = float(result_data.get("pct_synthesized_removal") or 0.0)
        proposal.update(
            {
                "metric_proposed_removal": metric_proposed,
                "proposed_removal": synthesized,
                "cells_in_proposed_removal": cells,
                "pct_proposed": pct,
                "cells_remaining_if_removed": (
                    None
                    if self.adata is None
                    else int(self.adata.n_obs) - cells
                ),
                "structure_evidence": result_data.get("structure_evidence_by_cluster", {}),
                "structure_clusters_analyzed": result_data.get("clusters_analyzed", []),
                "synthesized_removal": synthesized,
                "rescued_clusters": result_data.get("rescued_clusters", []),
                "confirmed_junk": result_data.get("confirmed_junk", []),
                "conflicting": result_data.get("conflicting", []),
                "structure_thresholds_used": result_data.get("thresholds_used", {}),
            }
        )
        auto_allowed, auto_reason = self._auto_structure_cleanup_allowed(proposal)

        if synthesized:
            options, option_actions = self._checkpoint_options([
                ("Remove the structure-supported low-quality clusters and rerun embedding", "remove_proposed_clusters"),
                ("Keep these clusters and proceed", "keep_proposed_clusters"),
                ("Inspect cluster structure details before deciding", "review_cluster_qc"),
                ("Something else", "custom"),
            ])
            summary = (
                f"Structure QC synthesized removing {len(synthesized)} cluster(s) "
                f"({', '.join(synthesized)}) from '{proposal.get('cluster_key', 'leiden')}', "
                f"totaling {cells} cells ({pct:.1f}%)."
            )
            recommendation = options[0]
        else:
            return {
                "kind": "cluster_qc_cleanup_resolved",
                "resolved_action": "keep_structure_reviewed_clusters",
                "summary": (
                    "Structure QC did not synthesize any removal candidates; "
                    "the reviewed metric-flagged clusters should be kept for now."
                ),
                "proposal": proposal,
                "review_clusters": result_data.get("conflicting", []) or [],
                "auto_allowed": False,
                "requires_user_confirmation": False,
                "structure_refined": True,
                "resolved": True,
            }

        refined = dict(checkpoint) if checkpoint.get("kind") == "cluster_qc_cleanup" else {}
        refined.update(
            {
                "kind": "cluster_qc_cleanup",
                "question": "Cluster structure QC refined the cleanup proposal. How should I proceed?",
                "options": options,
                "option_actions": option_actions,
                "default": recommendation,
                "recommendation": recommendation,
                "decision_key": "cluster_qc_cleanup",
                "summary": summary,
                "proposal": proposal,
                "auto_allowed": auto_allowed,
                "auto_reason": auto_reason,
                "requires_user_confirmation": not auto_allowed,
                "structure_refined": True,
            }
        )
        return refined

    def _authorize_pending_cleanup_from_user(self, request: str) -> bool:
        checkpoint = self._pending_checkpoint or {}
        if checkpoint.get("kind") != "cluster_qc_cleanup":
            return False
        proposal = checkpoint.get("proposal") or {}
        selected_action = self._checkpoint_action_from_user_response(checkpoint, request)
        if selected_action == "remove_proposed_clusters":
            mentioned = self._mentioned_cluster_labels(request)
            proposed = {str(label) for label in proposal.get("proposed_removal", [])}
            if mentioned and mentioned != proposed:
                return False
            self._active_cleanup_authorization = {
                "source": "user_confirmation",
                "proposal": proposal,
                "reason": "User explicitly confirmed the pending cluster cleanup.",
            }
            self.world_state.resolve_decision(
                "cluster_qc_cleanup",
                "remove_proposed_clusters",
                source="user",
                message=request,
            )
            self._clear_pending_checkpoint(request)
            return True
        if selected_action == "keep_proposed_clusters":
            self.world_state.resolve_decision(
                "cluster_qc_cleanup",
                "keep_proposed_clusters",
                source="user",
                message=request,
            )
            self._clear_pending_checkpoint(request)
            return True
        return False

    def _looks_like_direct_cleanup_request(self, message: str) -> bool:
        text = (message or "").lower()
        return bool(
            re.search(r"\b(remove|drop|filter|exclude|subset out)\b", text)
            and re.search(r"\b(cluster|clusters|cells?)\b", text)
        )

    def _cleanup_authorization_for_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if tool_name != "run_code":
            return None
        if self._active_cleanup_authorization:
            return dict(self._active_cleanup_authorization)
        checkpoint = self._pending_checkpoint or {}
        if checkpoint.get("kind") == "cluster_qc_cleanup":
            proposal = checkpoint.get("proposal") or {}
            auto_allowed, auto_reason = self._is_auto_cleanup_allowed(proposal)
            if auto_allowed:
                return {
                    "source": "auto_policy",
                    "proposal": proposal,
                    "reason": auto_reason,
                }
            return {
                "source": "user_confirmation",
                "proposal": proposal,
                "reason": (
                    "A cluster cleanup checkpoint is pending. The model interpreted the "
                    "latest user reply in that context; destructive-code preflight must "
                    "verify the exact proposed clusters and cell count before execution."
                ),
            }
        if self._looks_like_direct_cleanup_request(self._active_request):
            return {
                "source": "direct_user_request",
                "proposal": {},
                "reason": "The latest user request directly requested cell or cluster removal.",
            }
        return None

    def _sync_world_state(self, extra_text: Optional[str] = None) -> None:
        """Refresh the unified world state from the active AnnData and request context."""
        self.world_state.set_active_request(self._active_request)
        self.world_state.sync_from_adata(
            self.adata,
            request_text=extra_text or self._active_request,
        )

    def _complete_run(self, final_result: str) -> None:
        """Finalize a turn: guarantee a comprehensive analysis report exists, then
        append the findings-log entry (full prompt + tool counts + report link).

        A deterministic ``analysis_record.md`` is always written from stored
        session state via ``_assemble_analysis_record`` so a full report exists
        even when the model did not call ``write_report``. When the model did
        call it, that narrative report layers on top of the same record.
        """
        if not self.run_manager:
            return
        report_path: Optional[str] = None
        try:
            from .tools import _assemble_analysis_record
            record = _assemble_analysis_record(self.world_state, self.adata)
            if record:
                header = f"# Comprehensive Analysis Record — {self.run_manager.run_id}\n\n"
                req = (self._active_request or "").strip()
                if req:
                    header += f"**Original request:**\n\n{req}\n\n---\n\n"
                report_path = self.run_manager.write_text_report(
                    "analysis_record", header + record, ext="md"
                )
        except Exception:
            report_path = None
        self.run_manager.complete(
            summary=final_result,
            request=self._active_request,
            report_path=report_path,
        )

    def _record_world_state_snapshot(self) -> None:
        """Persist a compact world-state snapshot into the run ledger."""
        if self.run_manager:
            self.run_manager.append_world_state_snapshot(self.world_state.snapshot())

    def _remember_user_preferences(self, message: str) -> None:
        """Persist explicit user corrections so later tools can reuse them mechanically."""
        if not message:
            return

        text = " ".join(message.lower().split())

        confirm_cleanup_patterns = [
            r"\bask me\b.*\b(before|prior to)\b.*\b(remove|filter|drop|exclude)\b",
            r"\bconfirm\b.*\b(before|prior to)\b.*\b(remove|filter|drop|exclude)\b",
            r"\bdon't remove\b.*\bwithout\b.*\b(confirm|ask)",
            r"\bdo not remove\b.*\bwithout\b.*\b(confirm|ask)",
        ]
        auto_cleanup_patterns = [
            r"\b(don't|dont|do not)\s+ask\b.*\b(remove|filter|drop|exclude|cleanup|clean up)\b",
            r"\b(remove|filter|drop|exclude|cleanup|clean up)\b.*\bwithout asking\b",
            r"\b(you|agent)\s+(decide|choose)\b.*\b(threshold|remove|filter|cleanup|clean up)\b",
            r"\bchoose\b.*\b(threshold|cutoff|cutoffs)\b.*\b(remove|filter|drop|exclude)\b",
            r"\bif\b.*\b(removing|filtering|cleanup|cleaning)\b.*\b(needs|should)\b.*\b(just )?(do it|remove|proceed)\b",
            r"\bautomatically\b.*\b(remove|filter|drop|exclude|cleanup|clean up)\b",
            r"\bbe autonomous\b.*\b(remove|filter|drop|exclude|cleanup|clean up)\b",
        ]
        if any(re.search(pattern, text) for pattern in confirm_cleanup_patterns):
            policy = {
                "mode": "confirm",
                "max_pct_without_confirmation": 0.0,
                "stop_for_ambiguous": True,
                "require_exact_count": True,
                "set_by_user_message": message,
            }
            self.world_state.resolve_decision(
                "cluster_cleanup_policy",
                policy,
                source="user",
                message=message,
            )
            if self.run_manager:
                self.run_manager.add_user_decision(
                    {
                        "key": "cluster_cleanup_policy",
                        "policy_action": "user_preference",
                        "status": "user_corrected",
                        "applied_value": policy,
                        "user_message": message,
                    }
                )
        elif any(re.search(pattern, text) for pattern in auto_cleanup_patterns):
            policy = {
                "mode": "auto_obvious",
                "max_pct_without_confirmation": 5.0,
                "stop_for_ambiguous": True,
                "require_exact_count": True,
                "require_checkpoint": True,
                "set_by_user_message": message,
            }
            self.world_state.resolve_decision(
                "cluster_cleanup_policy",
                policy,
                source="user",
                message=message,
            )
            if self.run_manager:
                self.run_manager.add_user_decision(
                    {
                        "key": "cluster_cleanup_policy",
                        "policy_action": "user_preference",
                        "status": "user_corrected",
                        "applied_value": policy,
                        "user_message": message,
                    }
                )

        loading_strategy = None
        if re.search(
            r"\b(concatenate|concat|combine|merge)\b.*\bouter(?:\s+join)?\b"
            r"|\bouter\s+join\b.*\b(concatenate|concat|combine|merge)\b",
            text,
        ):
            loading_strategy = "concatenate_outer"
        elif re.search(
            r"\b(concatenate|concat|combine|merge)\b.*\binner(?:\s+join)?\b"
            r"|\binner\s+join\b.*\b(concatenate|concat|combine|merge)\b",
            text,
        ):
            loading_strategy = "concatenate_inner"
        elif re.search(
            r"\b(analy[sz]e|process|run)\b.*\b(datasets?|files?)\b.*\bseparately\b",
            text,
        ):
            loading_strategy = "analyze_separately"
        if loading_strategy:
            self.world_state.resolve_decision(
                "multi_dataset_loading_strategy",
                loading_strategy,
                source="user",
                message=message,
            )

        strategy_patterns = [
            (
                "integrate_scvi",
                [
                    r"\b(integrate|batch[- ]?correct)\b.*\bscvi\b",
                    r"\buse\s+scvi\b.*\b(integrat|batch)",
                    r"\bintegrate\s+(?:the\s+)?(?:samples?|datasets?)\b",
                ],
            ),
            (
                "keep_unintegrated",
                [
                    r"\b(do not|don't|dont|no)\s+(integrate|batch[- ]?correct)\b",
                    r"\bkeep\b.*\b(unintegrated|uncorrected)\b",
                    r"\bleave\b.*\b(unintegrated|uncorrected|as is)\b",
                ],
            ),
            (
                "investigate_integration",
                [
                    r"\binvestigate\b.*\b(batch|integrat)",
                    r"\b(check|assess|decide)\b.*\b(whether|if)\b.*\b(integrat|batch[- ]?correct)",
                ],
            ),
            (
                "analyze_separately",
                [
                    r"\b(analy[sz]e|process|run)\b.*\b(samples?|datasets?)\b.*\bseparately\b",
                    r"\bseparate\b.*\b(sample|dataset)[- ]specific\b.*\banalys",
                ],
            ),
        ]
        for strategy, patterns in strategy_patterns:
            if not any(re.search(pattern, text) for pattern in patterns):
                continue
            self.world_state.resolve_decision(
                "multi_sample_strategy",
                strategy,
                source="user",
                message=message,
            )
            if self.run_manager:
                self.run_manager.add_user_decision(
                    {
                        "key": "multi_sample_strategy",
                        "policy_action": "recommend_and_confirm",
                        "status": "user_corrected",
                        "applied_value": strategy,
                        "user_message": message,
                    }
                )
            break

        explicit_method_match = re.search(r"\b(harmony|bbknn|scanorama)\b", text)
        if explicit_method_match and re.search(
            r"\b(use|run|apply|integrate|integration|integrating|batch[- ]?correct)\b",
            text,
        ):
            method = explicit_method_match.group(1)
            self.world_state.resolve_decision(
                "multi_sample_strategy",
                {
                    "action": "custom",
                    "details": f"Integrate using {method}.",
                    "method": method,
                },
                source="user",
                message=message,
            )

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
        for preflight in result_data.get("preflight_checks", []) or []:
            checks.append(
                {
                    "name": preflight.get("name", "preflight_check"),
                    "status": preflight.get("status", "warning"),
                    "details": json.dumps(preflight, default=str),
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

        if tool_name == "bc_get_panglaodb_marker_genes":
            result_data.setdefault(
                "marker_query",
                {
                    "species": tool_input.get("species"),
                    "cell_type": tool_input.get("cell_type"),
                    "min_sensitivity": tool_input.get("min_sensitivity"),
                },
            )
            if isinstance(result_data.get("markers"), list) and len(result_data["markers"]) == 0:
                result_data["no_markers_found"] = True
                result_data["next_step"] = (
                    "No PanglaoDB entries matched this cell_type string. "
                    "Call bc_get_panglaodb_options once (if not already called this session) "
                    "to retrieve the valid vocabulary, pick the closest matching term, "
                    "retry bc_get_panglaodb_marker_genes with that term, and record the "
                    "substitution in panglaodb_label_used on the evidence entry."
                )

        if "state_delta" not in result_data:
            before_stage = before_snapshot.get("analysis_stage", "uninitialized")
            after_stage = after_snapshot.get("analysis_stage", before_stage)
            before_summary = before_snapshot.get("data_summary") or {}
            after_summary = after_snapshot.get("data_summary") or {}
            before_shape = before_summary.get("shape")
            after_shape = after_summary.get("shape")
            before_processing = (before_snapshot.get("data_summary") or {}).get("processing", {})
            after_processing = (after_snapshot.get("data_summary") or {}).get("processing", {})
            changed_flags = {}
            for key in sorted(set(before_processing.keys()) | set(after_processing.keys())):
                if before_processing.get(key) != after_processing.get(key):
                    changed_flags[key] = {
                        "before": before_processing.get(key),
                        "after": after_processing.get(key),
                    }
            dataset_changed = tool_name in {
                "run_qc",
                "normalize_and_hvg",
                "run_pca",
                "run_neighbors",
                "run_umap",
                "run_clustering",
                "compare_clusterings",
                "run_celltypist",
                "run_scimilarity",
                "run_batch_correction",
                "run_deg",
            }
            if tool_name == "run_code":
                dataset_changed = bool(changed_flags) or before_shape != after_shape
            result_data["state_delta"] = {
                "tool": tool_name,
                "summary": result_data.get("message") or f"{tool_name} completed.",
                "dataset_changed": dataset_changed,
                "stage_before": before_stage,
                "stage_after": after_stage,
                "shape_before": before_shape,
                "shape_after": after_shape,
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
        result_data["verification"] = self._generic_verification(tool_name, tool_input, result_data)
        return result_data

    def _looks_like_failure(self, message: str) -> bool:
        """Heuristic for assistant responses that likely need recovery.

        Returns False immediately when the response contains clear completion
        indicators (next-step options, success confirmations) even if it also
        mentions past failures that were handled gracefully.
        """
        normalized = " ".join((message or "").lower().split())
        if not normalized:
            return False
        # If the response ends with next-step options or confirms success, the
        # agent has already resolved any issues — no recovery needed.
        if any(re.search(p, normalized) for p in SUCCESS_OVERRIDE_PATTERNS):
            return False
        scrubbed = normalized
        for pattern in NON_FAILURE_PATTERNS:
            scrubbed = re.sub(pattern, "", scrubbed)
        return any(re.search(pattern, scrubbed) for pattern in FAILURE_PATTERNS)

    def _build_auto_recovery_instruction(self, error_msg: str, attempt: int) -> str:
        """Prompt the model to self-correct before asking the user for help."""
        return (
            f"Your previous response indicates an unresolved issue.\n\n"
            f"Issue:\n{error_msg}\n\n"
            f"This is automatic recovery attempt {attempt} of {AUTO_RECOVERY_ATTEMPTS}. "
            "Try to resolve the problem yourself before asking the user for help. "
            "Use the available tools to inspect state, fix missing prerequisites, adjust parameters, "
            "or try a better approach. If you can recover, do so and then give a normal user-facing "
            "summary. Only if you still cannot proceed after genuinely trying should you ask the user "
            "a concise follow-up question."
        )

    def _print_auto_recovery_notice(self, attempt: int, error_snippet: str = "") -> None:
        """Tell the user the agent is trying to recover automatically."""
        from rich.console import Console
        console = Console()
        if error_snippet:
            # Trim to one line for display
            first_line = error_snippet.strip().splitlines()[0][:120]
            console.print(f"[yellow]↺ Error:[/yellow] {first_line}")
        console.print(
            f"[yellow]  Trying to recover automatically ({attempt}/{AUTO_RECOVERY_ATTEMPTS})...[/yellow]"
        )

    def _maybe_continue_after_failure(
        self,
        final_result: str,
        messages: List[Dict[str, Any]],
        auto_recovery_attempts: int,
        suggestions: Optional[List[str]] = None,
    ):
        """Try bounded automatic recovery before interrupting the user."""
        if not self._looks_like_failure(final_result):
            return False, auto_recovery_attempts

        if auto_recovery_attempts < AUTO_RECOVERY_ATTEMPTS:
            next_attempt = auto_recovery_attempts + 1
            logger.warning(
                "Assistant final response looked like a failure; starting automatic recovery attempt %s/%s",
                next_attempt,
                AUTO_RECOVERY_ATTEMPTS,
            )
            self._print_auto_recovery_notice(next_attempt, error_snippet=final_result)
            messages.append({
                "role": "user",
                "content": self._build_auto_recovery_instruction(final_result, next_attempt),
            })
            self._conversation_history = messages
            return True, next_attempt

        if self.verbose:
            user_input = self._ask_continue(final_result, suggestions=suggestions)
            if user_input.lower() not in ["quit", "exit", "q"]:
                messages.append({"role": "user", "content": user_input})
                self._conversation_history = messages
                return True, 0

        return False, auto_recovery_attempts

    def _maybe_continue_for_obligations(
        self,
        messages: List[Dict[str, Any]],
        obligation_attempts: int,
    ):
        """Floor: don't let the run END while a scientific-spine obligation is unmet.

        The save/report guard (`_annotation_validation_guard`) already hard-blocks
        the *tool-exit* door (save_data/write_report before finalize). This closes the
        other door — a no-tool-call ``stop``/``length`` turn that calls `_complete_run`
        directly and bypasses that guard (the verified GLM non-convergence exit).

        Mechanism mirrors `_maybe_continue_after_failure`: re-prompt with the unmet
        obligation's guidance, bounded by `OBLIGATION_NUDGES`. Per "blocking > nudging",
        the bound is enforced: once exhausted, a *completion* obligation triggers a
        forced safe fallback (`save_data(allow_unvalidated=true)`) so the run never
        ends silently incomplete. Returns (should_continue, next_attempt).
        """
        # If the agent is correctly paused at a collaborative checkpoint, it has
        # SURFACED a decision (e.g. pause_and_ask for multi_sample_strategy) and is
        # awaiting the user — ending the turn to wait is the right behavior, not a
        # silent exit. Do not override it (this is the interactive case; in
        # autonomous mode pause_and_ask auto-resolves so no checkpoint lingers).
        # The GLM completion bug had NO pending checkpoint, so it is still caught.
        if getattr(self, "_pending_checkpoint", None):
            return False, obligation_attempts
        ws = getattr(self, "world_state", None)
        if ws is None or not hasattr(ws, "unmet_obligations"):
            return False, obligation_attempts
        blocking = [o for o in ws.unmet_obligations() if o.get("blocks_terminal")]
        if not blocking:
            return False, obligation_attempts

        if obligation_attempts < OBLIGATION_NUDGES:
            next_attempt = obligation_attempts + 1
            keys = ", ".join(o.get("key", "?") for o in blocking)
            logger.warning(
                "Run tried to end with unmet spine obligation(s) [%s]; nudge %s/%s",
                keys, next_attempt, OBLIGATION_NUDGES,
            )
            if hasattr(ws, "note_spine_intervention"):
                ws.note_spine_intervention([o.get("key", "?") for o in blocking], "nudge")
            guidance = "\n".join(f"- {o['guidance']}" for o in blocking)
            messages.append({
                "role": "user",
                "content": (
                    "You are ending the run, but a required step is not complete:\n"
                    f"{guidance}\n\n"
                    "Do not stop here. Take the action above now (emit the tool call)."
                ),
            })
            self._conversation_history = messages
            return True, next_attempt

        # Bound exhausted: force a safe fallback for completion obligations so the
        # analysis is never silently lost (worst case: an explicit UNVALIDATED save,
        # never a null result). Entry obligations have no harness-side fallback here.
        completion_unmet = any(o.get("kind") == "completion" for o in blocking)
        if completion_unmet:
            logger.warning(
                "Spine obligation still unmet after %s nudges; forcing "
                "save_data(allow_unvalidated=true).", OBLIGATION_NUDGES,
            )
            if hasattr(ws, "note_spine_intervention"):
                ws.note_spine_intervention(
                    [o.get("key", "?") for o in blocking if o.get("kind") == "completion"],
                    "forced_fallback",
                )
            try:
                result_json = self._execute_tool("save_data", {"allow_unvalidated": True})
                messages.append({
                    "role": "user",
                    "content": (
                        "Auto-saved an UNVALIDATED dataset because a required step was "
                        "not completed after repeated prompts:\n" + result_json
                    ),
                })
                self._conversation_history = messages
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Forced unvalidated save failed: %s", exc)
        return False, obligation_attempts

    def _ask_continue(self, error_msg: str, suggestions: list = None) -> str:
        """Ask user how to proceed after automatic recovery was not enough."""
        from rich.console import Console
        from rich.panel import Panel
        console = Console()

        console.print()
        console.print(Panel(
            "The agent ran into an issue and could not fully recover automatically.",
            title="⚠️  Issue Detected",
            border_style="yellow"
        ))

        if suggestions:
            console.print("\n[bold]Suggestions:[/bold]")
            for i, s in enumerate(suggestions, 1):
                console.print(f"  {i}. {s}")

        try:
            from ..terminal import DecisionChoice, prompt_for_decision

            selection = prompt_for_decision(
                "What would you like to do?",
                [
                    DecisionChoice("Try automatic recovery again", "retry"),
                    DecisionChoice("Enter a new instruction", "custom"),
                    DecisionChoice("Stop this analysis", "stop"),
                ],
                default_index=0,
                allow_custom=False,
            )
            if selection.action == "retry":
                return "try to recover from the error"
            if selection.action == "stop":
                return "quit"
            return selection.value
        except (EOFError, KeyboardInterrupt):
            return "quit"

    def _handle_max_iterations_reached(
        self,
        messages: List[Dict[str, Any]],
        max_iterations: int,
        *,
        message_format: str = "openai",
    ) -> str:
        """Persist and report an explicit resumable pause at the tool-call limit."""
        final_result = (
            f"Paused because this turn reached the tool-call limit "
            f"({max_iterations}). The current AnnData state is still in memory "
            "and the run manifest has been updated. Send a follow-up such as "
            "`continue` to resume from the current state; completed filtering "
            "or cleanup steps should not be repeated unless the live data still "
            "validates that action."
        )

        if message_format == "anthropic":
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": final_result}],
            })
        else:
            messages.append({"role": "assistant", "content": final_result})

        self._conversation_history = messages
        if self.run_manager:
            self.run_manager.append_event(
                "iteration_limit_reached",
                {
                    "max_iterations": max_iterations,
                    "request": self._active_request,
                    "last_action": self.world_state.last_action,
                },
            )
            self._complete_run(final_result)

        self._print("\n" + "-" * 50)
        self._print(final_result)
        if self.run_manager:
            self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")
        return final_result

    def _handle_unexpected_provider_stop(
        self,
        messages: List[Dict[str, Any]],
        reason: str,
        *,
        message_format: str = "openai",
    ) -> str:
        """Persist and report a resumable pause for unexpected provider stops."""
        final_result = (
            f"Paused because the model provider returned an unexpected stop "
            f"reason: {reason}. The current AnnData state is still in memory "
            "and the run manifest has been updated. Send a follow-up with how "
            "you want to proceed, or say `continue` to resume from the current state."
        )
        if message_format == "anthropic":
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": final_result}],
            })
        else:
            messages.append({"role": "assistant", "content": final_result})

        self._conversation_history = messages
        if self.run_manager:
            self.run_manager.append_event(
                "unexpected_provider_stop",
                {
                    "reason": reason,
                    "request": self._active_request,
                    "last_action": self.world_state.last_action,
                },
            )
            self._complete_run(final_result)

        self._print("\n" + "-" * 50)
        self._print(final_result)
        if self.run_manager:
            self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")
        return final_result

    def analyze(
        self,
        request: str,
        data_path: Optional[str] = None,
        run_name: Optional[str] = None,
        max_iterations: int = 75,
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
        max_iterations : int, default 75
            Maximum number of tool calls per turn before an explicit resumable pause.
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

        # Print the panel immediately so the terminal never looks stuck while
        # _sync_world_state (which calls inspect_data on the full adata) runs.
        if self.verbose:
            from rich.console import Console
            from rich.rule import Rule
            console = Console()
            console.print(Rule(style="cyan"))

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

        # Resolve text-only clients against the newest pending checkpoint before
        # the request enters the provider conversation.
        if self._pending_checkpoint:
            selection = self.resolve_pending_decision_text(request)
            if selection is not None:
                request = self.structured_decision_request(selection)
                self._active_request = request
                self.world_state.set_active_request(request)

        # Build initial message
        user_message = request
        if data_path:
            user_message += f"\n\nData file: {data_path}"
        elif is_followup:
            # Inform agent that data is already loaded
            user_message += f"\n\n[Data already loaded in memory: {self.adata.n_obs} cells x {self.adata.n_vars} genes]"
            user_message += f"\n\n{self._followup_state_checkpoint(request)}"
        if self.run_manager:
            user_message += f"\nOutput directory: {self.run_manager.run_dir}"

        if self.collaborative and not self.smart_autonomous and not sys.stdin.isatty():
            message = (
                "Collaborative agent mode requires an interactive terminal because the agent must "
                "pause and ask for decisions at checkpoints."
            )
            if self.run_manager:
                self.run_manager.fail(message)
            raise RuntimeError(message)

        # Route to provider-specific implementation, wrapped in a tracing root span so
        # per-iteration LLM/tool spans nest under one trace (and under NAT's eval
        # workflow span when a W3C traceparent is propagated in via the environment).
        _tracing.start_root("scagent.analyze", {
            "scagent.provider": self.provider,
            "scagent.model": str(self.model),
        })
        try:
            if self.provider == "anthropic":
                return self._analyze_anthropic(user_message, max_iterations, continue_conversation)
            elif self.provider in {"openai", "groq", "gemini", "vertex"}:
                return self._analyze_openai(user_message, max_iterations, continue_conversation)
            elif self.provider == "codex":
                return self._analyze_codex(user_message, max_iterations, continue_conversation)
            raise RuntimeError(f"Unsupported provider: {self.provider}")
        finally:
            _tracing.end_root()

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
            "When a tool result or runtime state says checkpoint_required or pending_checkpoint, "
            "return kind='final' with a clear explanation of the evidence and why a decision is "
            "needed. The runtime renders the options as an interactive selector, so do not "
            "duplicate them as a numbered menu. When the next request contains a Structured "
            "user decision, treat selected_action as authoritative.\n\n"
            "## Current Request, History, Runtime State, and Tools\n"
            f"{json.dumps(payload, indent=2, default=str)}"
        )

    @staticmethod
    def _format_token_count(value: int) -> str:
        if value >= 1000:
            return f"{value // 1000}K"
        return str(value)

    def _context_bar_str(self) -> str:
        """Compact context usage indicator: '▓▓▓░░░░░░░ 28% · ~21K/77K'."""
        used = self._context_display_tokens or self._last_actual_tokens or self._last_estimated_tokens
        limit = self._context_limit
        if limit <= 0 or used <= 0:
            return ""
        pct = min(used / limit, 1.0)
        filled = int(pct * 10)
        bar = "▓" * filled + "░" * (10 - filled)
        prefix = "~" if self._context_display_source.startswith("estimated") else ""
        used_k = self._format_token_count(used)
        limit_k = self._format_token_count(limit)
        return f"{bar} {pct:.0%} · {prefix}{used_k}/{limit_k}"

    def _update_context_bar(self) -> None:
        """
        Write the context usage bar to the bottom-right corner of the terminal.

        Writes directly to /dev/tty (the controlling terminal) to bypass any
        stdout wrapping by Rich. Uses ANSI cursor-save/restore so nothing else
        on screen is disturbed. Called both before and after each model call so
        the bar persists after the spinner clears.
        """
        if not self.show_context_usage:
            return
        try:
            self._refresh_context_usage_display(source="estimated next prompt")
            if (self._context_display_tokens or self._last_actual_tokens or self._last_estimated_tokens) <= 0:
                return

            import shutil
            size = shutil.get_terminal_size(fallback=(0, 0))
            cols, rows = size.columns, size.lines
            if cols <= 0 or rows <= 0:
                return

            # Full bar: " ▓▓▓░░░░░░░ 28% · 21K/77K " (~26 chars)
            bar = f" {self._context_bar_str()} "
            if len(bar) > cols:
                # Compact: just percentage and counts, no block bar
                used = self._context_display_tokens or self._last_actual_tokens or self._last_estimated_tokens
                pct = min(used / self._context_limit, 1.0)
                prefix = "~" if self._context_display_source.startswith("estimated") else ""
                used_k = self._format_token_count(used)
                limit_k = self._format_token_count(self._context_limit)
                bar = f" {pct:.0%} {prefix}{used_k}/{limit_k} "
            if len(bar) > cols:
                return  # terminal too narrow even for compact form

            col = cols - len(bar) + 1
            with open("/dev/tty", "w") as tty:
                tty.write(f"\0337\033[{rows};{col}H\033[2m{bar}\033[0m\0338")
                tty.flush()
        except Exception:
            return

    def _with_llm_status(self, action):
        """Run a blocking model call with provider-neutral terminal feedback."""
        status_messages = [
            "Analyzing...",
            "Working...",
            "Thinking...",
            'Doing...',
            'Thinkering...',
            'Processing...',
        ]
        if self.verbose:
            from rich.console import Console

            console = Console()
            message = self._next_llm_status_message or random.choice(status_messages)
            self._next_llm_status_message = None
            with console.status(message, spinner="dots"):
                return action()
        self._next_llm_status_message = None
        return action()

    def _request_codex_decision(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ask Codex for the next step while keeping terminal feedback provider-neutral."""
        prompt = self._build_codex_decision_prompt(messages)
        return self._with_llm_status(
            lambda: self.client.complete_json(prompt, CODEX_DECISION_SCHEMA)
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
        auto_recovery_attempts = 0
        obligation_nudge_attempts = 0

        try:
            for _iteration in range(max_iterations):
                decision = self._request_codex_decision(messages)

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

                    if self._pending_images:
                        if self._vision_sidecar is not None:
                            messages.append(self._build_sidecar_text_message(self._pending_images))
                            self._next_llm_status_message = "Reading figure description..."
                        else:
                            paths = ", ".join(img["path"] for img in self._pending_images)
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Figure(s) generated at {paths}. The Codex CLI bridge does not "
                                    "support inline image bytes; call review_figure if visual review is needed."
                                ),
                            })
                        self._pending_images = []
                    continue

                if kind == "final":
                    final_result = decision.get("content") or ""
                    messages.append({"role": "assistant", "content": final_result})
                    self._print("\n" + "-" * 50)
                    self._print(final_result)
                    should_continue, auto_recovery_attempts = self._maybe_continue_after_failure(
                        final_result,
                        messages,
                        auto_recovery_attempts,
                        suggestions=["Provide additional instructions", "Try a different approach"],
                    )
                    if should_continue:
                        continue
                    should_continue, obligation_nudge_attempts = self._maybe_continue_for_obligations(
                        messages, obligation_nudge_attempts,
                    )
                    if should_continue:
                        continue
                    self._conversation_history = messages
                    if self.run_manager and not self._pending_checkpoint:
                        self._complete_run(final_result)
                        self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")
                    return final_result

                raise CodexCLIError(f"Codex returned unknown decision kind: {kind}")

            return self._handle_max_iterations_reached(
                messages,
                max_iterations,
                message_format="openai",
            )
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
        auto_recovery_attempts = 0
        obligation_nudge_attempts = 0

        try:
            for iteration in range(max_iterations):
                system_prompt = self._build_system_prompt()
                trim_target, hard_limit = self._compute_message_budget(system_prompt)
                messages = self._trim_messages_if_needed(
                    messages, anthropic=True,
                    trim_target=trim_target, hard_limit=hard_limit,
                )
                _t0_llm = _tracing.now_ns()
                try:
                    response = self._with_llm_status(
                        lambda: self.client.messages.create(
                            model=self.model,
                            max_tokens=self._max_output_tokens,
                            system=system_prompt,
                            tools=self.tools,
                            messages=messages,
                        )
                    )
                except Exception as _api_err:
                    if self._is_context_overflow_error(_api_err):
                        messages = self._emergency_trim(messages, anthropic=True)
                        try:
                            response = self._with_llm_status(
                                lambda: self.client.messages.create(
                                    model=self.model,
                                    max_tokens=self._max_output_tokens,
                                    system=self._build_system_prompt(),
                                    tools=self.tools,
                                    messages=messages,
                                )
                            )
                        except Exception as _retry_err:
                            if self._is_context_overflow_error(_retry_err):
                                self._print(
                                    "[red]Context window exhausted — unable to recover after "
                                    "emergency trim. Start a new session. Your analysis state "
                                    "is preserved in world state.[/red]"
                                )
                                if self.run_manager:
                                    self.run_manager.fail("context_window_exhausted")
                                return final_result or ""
                            raise
                    else:
                        raise

                if response.usage and hasattr(response.usage, 'input_tokens') and response.usage.input_tokens:
                    self._last_actual_tokens = response.usage.input_tokens
                    self._context_display_tokens = self._last_actual_tokens
                    self._context_display_source = "last actual prompt"
                    # Calibrate estimate ratio — ratchets upward, never down, capped at 4x
                    _current_est = self._context_usage_snapshot(
                        messages,
                        system_prompt=system_prompt,
                        anthropic=True,
                        trim_target=trim_target,
                        hard_limit=hard_limit,
                    )["prompt_estimate"]
                    if _current_est > 0 and self._last_actual_tokens > _current_est:
                        _new_ratio = self._last_actual_tokens / _current_est
                        self._token_estimate_calibration = max(
                            self._token_estimate_calibration,
                            min(_new_ratio, 4.0),
                        )

                _usage = getattr(response, "usage", None)
                _tracing.record_llm(
                    iteration, self.model,
                    getattr(_usage, "input_tokens", None) if _usage else None,
                    getattr(_usage, "output_tokens", None) if _usage else None,
                    _t0_llm,
                )
                _tracing.record_llm_io(iteration, self.model, messages, response.content, _t0_llm)

                if response.stop_reason == "tool_use":
                    tool_results = []
                    assistant_content = []

                    for content in response.content:
                        if content.type == "text":
                            assistant_content.append(content)
                            self._print_thinking(content.text)

                        elif content.type == "tool_use":
                            assistant_content.append(content)
                            _t0_tool = _tracing.now_ns()
                            result_json = self._execute_tool(content.name, content.input)
                            _tracing.record_tool(content.name, iteration, _t0_tool)

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result_json,
                            })

                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({"role": "user", "content": tool_results})

                    # If there are pending figures, inject them as a vision user message
                    if self._pending_images:
                        if self._supports_vision():
                            messages.append(self._build_image_message(self._pending_images, "anthropic"))
                            self._next_llm_status_message = "Analyzing figure..."
                        elif self._vision_sidecar is not None:
                            messages.append(self._build_sidecar_text_message(self._pending_images))
                            self._next_llm_status_message = "Reading figure description..."
                        else:
                            paths = ", ".join(img["path"] for img in self._pending_images)
                            messages.append({
                                "role": "user",
                                "content": f"Figure(s) saved at {paths}.",
                            })
                        self._pending_images = []

                elif response.stop_reason == "end_turn":
                    # Add final assistant message to history
                    messages.append({"role": "assistant", "content": response.content})

                    text_parts = [content.text for content in response.content if hasattr(content, "text")]
                    final_result = "\n".join(part for part in text_parts if part)
                    if final_result:
                        self._print("\n" + "-" * 50)
                        self._print(final_result)

                    should_continue, auto_recovery_attempts = self._maybe_continue_after_failure(
                        final_result,
                        messages,
                        auto_recovery_attempts,
                        suggestions=["Provide additional instructions", "Try a different approach"],
                    )
                    if should_continue:
                        continue

                    should_continue, obligation_nudge_attempts = self._maybe_continue_for_obligations(
                        messages, obligation_nudge_attempts,
                    )
                    if should_continue:
                        continue

                    # Save conversation history for potential follow-ups
                    self._conversation_history = messages

                    if self.run_manager and not self._pending_checkpoint:
                        self._complete_run(final_result)
                        self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")

                    return final_result

                else:
                    logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                    return self._handle_unexpected_provider_stop(
                        messages,
                        str(response.stop_reason),
                        message_format="anthropic",
                    )

            return self._handle_max_iterations_reached(
                messages,
                max_iterations,
                message_format="anthropic",
            )

        except Exception as e:
            if self.run_manager:
                self.run_manager.fail(str(e))
            raise

        return final_result

    def _parse_xml_tool_calls(self, text: str) -> list:
        """Parse tool calls from local model text output.

        Handles three formats emitted by local models (e.g. Qwen2.5-Coder via vLLM):
          1. <tool_call>{"name": ..., "arguments": ...}</tool_call>
          2. <tools>{"name": ..., "arguments": ...}</tools>
          3. Bare JSON: {"name": ..., "arguments": ...}  (no wrapper)
        Returns list of dicts with 'id', 'name', 'arguments' keys, or empty list.
        """
        import re
        import uuid

        def _extract(raw):
            try:
                parsed = json.loads(raw.strip())
                if isinstance(parsed, dict) and "name" in parsed:
                    return {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", parsed.get("parameters", {})),
                    }
            except (json.JSONDecodeError, KeyError):
                pass
            return None

        results = []

        # 1 & 2: XML-wrapped
        for tag in ("tool_call", "tools"):
            for match in re.finditer(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL):
                tc = _extract(match.group(1))
                if tc:
                    results.append(tc)

        if results:
            return results

        # 3: Bare JSON object(s) — try whole text, then scan for {...} blocks
        tc = _extract(text)
        if tc:
            return [tc]

        for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,[^{}]*\}', text, re.DOTALL):
            tc = _extract(match.group(0))
            if tc:
                results.append(tc)

        return results

    def _resolve_context_limit(self) -> int:
        """
        Determine the context window size for the current model.

        Priority order (highest wins):
          1. SCAGENT_CONTEXT_LIMIT env var — user override
          2. Self-hosted /v1/models limit — the actual GPU-constrained, per-request
             window reported by the serving backend (vLLM `max_model_len` or
             llama.cpp `meta.n_ctx`). See _server_context_limit.
          3. K-size parsed from model name (e.g. "262k", "128k")
          4. Cloud model name dict — known limits by provider
          5. Generic Qwen fallback — 32K with a visible warning
          6. Hard default — 128K

        Never silently falls through: each step logs what was found and why.
        """
        # --- Priority 1: env var override ---
        env_limit = os.environ.get("SCAGENT_CONTEXT_LIMIT")
        if env_limit:
            try:
                limit = int(env_limit)
                logger.info(f"Context limit from SCAGENT_CONTEXT_LIMIT env var: {limit:,}")
                self._print(f"[dim]Context limit set by SCAGENT_CONTEXT_LIMIT: {limit:,} tokens[/dim]")
                return limit
            except ValueError:
                logger.warning(
                    f"SCAGENT_CONTEXT_LIMIT={env_limit!r} is not a valid integer — ignoring"
                )

        model = (self.model or "").lower()

        # --- Priority 2: self-hosted /v1/models limit (vLLM or llama.cpp) ---
        try:
            base_url = str(getattr(self.client, 'base_url', ''))
            cloud_hosts = (
                "api.openai.com",
                "api.anthropic.com",
                "api.groq.com",
                "api.deepseek.com",
                "generativelanguage.googleapis.com",
                "aiplatform.googleapis.com",
            )
            is_cloud = any(h in base_url for h in cloud_hosts)
            if not is_cloud and base_url:
                for m in self.client.models.list().data:
                    if _model_get(m, "id") != self.model:
                        continue
                    limit, source = _server_context_limit(m)
                    if limit:
                        backend = "vLLM" if source == "max_model_len" else "llama.cpp"
                        logger.info(
                            f"Context limit from {backend} ({source}): {limit:,} tokens"
                        )
                        self._print(
                            f"[dim]Context limit from {backend}: {limit:,} tokens[/dim]"
                        )
                        return limit
                    break  # matched our model, but it advertised no usable limit
        except Exception as e:
            logger.warning(
                f"Self-hosted model query failed ({e}); falling through to name-based detection"
            )

        # --- Priority 3: parse K-size from model name ---
        k_match = re.search(r'(\d+)k\b', model)
        if k_match:
            limit = int(k_match.group(1)) * 1024
            logger.info(f"Context limit parsed from model name '{self.model}': {limit:,} tokens")
            self._print(
                f"[dim]Context limit inferred from model name: {limit:,} tokens[/dim]"
            )
            return limit

        # --- Priority 4: cloud model name dict ---
        def _known(limit: int, source: str) -> int:
            logger.info(f"Context limit from known model table ({source}): {limit:,} tokens")
            self._print(f"[dim]Context limit from known model table: {limit:,} tokens[/dim]")
            return limit

        # Claude: Opus 4.6/4.7, Sonnet 4.6, and Mythos are 1M; Haiku 4.5 and older are 200K
        if "claude" in model:
            if any(m in model for m in ("opus-4-6", "opus-4-7", "sonnet-4-6", "mythos")):
                return _known(1_000_000, self.model)
            return _known(200_000, self.model)
        # GPT-5.x: mini variants (gpt-5-mini, gpt-5.4-mini) are 400K; all others are 1M
        if "gpt-5" in model:
            if "mini" in model:
                return _known(400_000, self.model)
            return _known(1_000_000, self.model)
        if "gpt-4o" in model:
            return _known(128_000, self.model)
        if "gpt-4-turbo" in model:
            return _known(128_000, self.model)
        if "llama" in model or "mixtral" in model or "gemma" in model:
            return _known(128_000, self.model)
        # Gemini 3.x and 2.5: Pro, Flash, and Flash-Lite all support 1M context
        if "gemini" in model:
            return _known(1_048_576, self.model)
        # DeepSeek v4: pro/flash both support 1M context, 384K max output.
        # Legacy aliases (deepseek-chat / deepseek-reasoner) point at v4-flash.
        if "deepseek" in model:
            return _known(1_000_000, self.model)
        # Qwen3.5 / Qwen3.6 family: 262,144 (256K) native window
        # (max_position_embeddings). Sits below the vLLM probe (priority 2) on
        # purpose — vLLM still reports the exact GPU-constrained limit; this only
        # fires for OpenAI-compatible servers that don't advertise max_model_len
        # (e.g. TensorRT-LLM's trtllm-serve, which serves it at max_seq_len=262144).
        if "qwen3.6" in model or "qwen3.5" in model or "qwen3_5" in model:
            return _known(262_144, self.model)

        # --- Priority 5: generic Qwen fallback ---
        if "qwen" in model:
            limit = 32_768
            msg = (
                f"WARNING: Qwen model '{self.model}' detected but context size could not be "
                f"determined from the model name or vLLM. Defaulting to {limit:,} tokens — "
                f"this is conservative. Set SCAGENT_CONTEXT_LIMIT=<actual_size> in your "
                f"environment to override, or use a vLLM server that reports max_model_len."
            )
            logger.warning(msg)
            print(f"\n\033[33m{msg}\033[0m\n")  # visible yellow, bypasses Rich
            return limit

        # --- Priority 6: hard default ---
        logger.info(
            f"Context limit: using default 128,000 tokens "
            f"(model '{self.model}' not recognized by name)"
        )
        return 128_000

    @staticmethod
    def _estimate_tokens(messages: list) -> int:
        """Rough token count: 1 token ≈ 4 chars of JSON-serialized content."""
        try:
            text = json.dumps(messages, default=str)
            # Base64 image blobs (data:image/...;base64,<data>) can be 200KB+ of
            # characters but vision models process them as ~256-2048 image tokens,
            # not text tokens. Strip them and substitute a flat 4000-char estimate
            # (~1000 tokens) per image so the text-token budget stays accurate.
            text = re.sub(
                r'data:[^;"\s]+;base64,[A-Za-z0-9+/=]+',
                'data:image/placeholder_1000_tokens_estimated',
                text,
            )
            return len(text) // 4
        except Exception:
            return 0

    def _calibrated_estimate(self, content) -> int:
        """Return _estimate_tokens scaled by the calibration factor.

        Calibration starts at 1.0 and ratchets upward (never down, capped at 4x)
        as we compare char-based estimates against actual API token counts. This
        compensates for tokenizers (e.g. Qwen) that produce more tokens per
        character than the 1-token-per-4-chars baseline.
        """
        raw = self._estimate_tokens(content)
        return int(raw * self._token_estimate_calibration)

    @staticmethod
    def _history_messages_for_budget(messages: list, *, anthropic: bool = False) -> list:
        """Return conversation-history messages without provider system overhead."""
        if (
            not anthropic
            and messages
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
        ):
            return messages[1:]
        return messages

    def _context_usage_snapshot(
        self,
        messages: Optional[list] = None,
        *,
        system_prompt: Optional[str] = None,
        anthropic: bool = False,
        trim_target: int = -1,
        hard_limit: int = -1,
    ) -> Dict[str, int]:
        """Estimate the full next prompt and the history portion used for trimming.

        The model sees system prompt + tool schemas + message history. Only the
        message history can be compacted, so trimming decisions compare history
        tokens against a history budget. The terminal bar reports the fuller
        next-prompt estimate so it does not under-report after tool results have
        been appended since the last API call.
        """
        messages = messages if messages is not None else self._conversation_history
        system_prompt = system_prompt if system_prompt is not None else self._build_system_prompt()
        if trim_target == -1 or hard_limit == -1:
            trim_target, hard_limit = self._compute_message_budget(system_prompt)

        system_tokens = int((len(system_prompt) // 4) * self._token_estimate_calibration)
        tool_schema_tokens = int(self._tool_schema_tokens * self._token_estimate_calibration)
        history_messages = self._history_messages_for_budget(messages or [], anthropic=anthropic)
        history_estimate = self._calibrated_estimate(history_messages)
        overhead = tool_schema_tokens + system_tokens
        completion_reserve = self._max_output_tokens  # in sync with the per-call output cap
        safety_margin = max(2000, int(self._context_limit * 0.03))
        hard_prompt_limit = max(0, self._context_limit - completion_reserve - safety_margin)

        return {
            "history_estimate": history_estimate,
            "prompt_estimate": overhead + history_estimate,
            "system_tokens": system_tokens,
            "tool_schema_tokens": tool_schema_tokens,
            "overhead_tokens": overhead,
            "trim_target": trim_target,
            "hard_limit": hard_limit,
            "trim_target_prompt": overhead + trim_target,
            "hard_prompt_limit": hard_prompt_limit,
        }

    def _refresh_context_usage_display(
        self,
        messages: Optional[list] = None,
        *,
        system_prompt: Optional[str] = None,
        anthropic: bool = False,
        source: str = "estimated next prompt",
    ) -> Dict[str, int]:
        """Refresh the context bar from current history instead of stale API usage."""
        if messages is None:
            messages = self._conversation_history
        if not messages:
            if self._last_actual_tokens:
                self._context_display_tokens = self._last_actual_tokens
                self._context_display_source = "last actual prompt"
            return {}

        snapshot = self._context_usage_snapshot(
            messages,
            system_prompt=system_prompt,
            anthropic=anthropic,
        )
        self._last_estimated_tokens = snapshot["prompt_estimate"]
        self._context_display_tokens = snapshot["prompt_estimate"]
        self._context_display_source = source
        self._context_display_trim_target = snapshot["trim_target_prompt"]
        self._context_display_hard_limit = snapshot["hard_prompt_limit"]
        return snapshot

    def _compute_message_budget(self, system_prompt: str) -> tuple:
        """Compute available token budget for message history.

        Fixed overhead on every API call:
          - Tool schemas: precomputed at init in self._tool_schema_tokens
          - System prompt: re-estimated each call (world_state snapshot grows)
          - Completion reserve: 4096 (matches max_tokens / max_completion_tokens)
          - Safety margin: 3% of context_limit or 2000, whichever is larger

        Returns (trim_target, hard_limit):
          trim_target  — aim to be below this before the API call (proactive trim)
          hard_limit   — absolute ceiling; triggers emergency trim if exceeded
        """
        COMPLETION_RESERVE = self._max_output_tokens  # in sync with the per-call output cap
        system_tokens = int((len(system_prompt) // 4) * self._token_estimate_calibration)
        tool_schema_tokens = int(self._tool_schema_tokens * self._token_estimate_calibration)
        safety_margin = max(2000, int(self._context_limit * 0.03))
        available = (
            self._context_limit
            - tool_schema_tokens
            - system_tokens
            - COMPLETION_RESERVE
            - safety_margin
        )
        available = max(available, 4000)
        trim_target = int(available * 0.85)
        hard_limit = available
        return trim_target, hard_limit

    def _report_trim(
        self,
        freed: int,
        before: int,
        after: int,
        emergency: bool = False,
        *,
        trim_target: Optional[int] = None,
        hard_limit: Optional[int] = None,
    ) -> None:
        """Log and print context compaction results."""
        if freed <= 0:
            return
        tier = "EMERGENCY" if emergency else "normal"
        limit_k = self._format_token_count(self._context_limit)
        detail = f"(next prompt ~{before:,} → ~{after:,} / {limit_k}"
        if trim_target is not None:
            detail += f"; trim target ~{trim_target:,}"
        if hard_limit is not None:
            detail += f"; hard prompt cap ~{hard_limit:,}"
        detail += ")"
        self._print(
            f"[dim]Context compacted ({tier}) — freed ~{freed:,} estimated prompt tokens "
            f"{detail}. Analysis state preserved in world state.[/dim]"
        )
        if self.run_manager:
            self.run_manager.append_log(
                "CONTEXT_COMPACT "
                f"tier={tier} freed={freed} before={before} after={after} "
                f"trim_target={trim_target} hard_limit={hard_limit}"
            )

    def _trim_messages_if_needed(
        self,
        messages: list,
        *,
        anthropic: bool = False,
        trim_target: int = -1,
        hard_limit: int = -1,
        force_emergency: bool = False,
    ) -> list:
        """
        Trim old tool results and assistant narrations when approaching context budget.

        Three tiers based on how far over budget the messages are:
          1. token_basis <= trim_target                  → no-op
          2. trim_target < token_basis <= hard_limit     → normal trim
               pass 1: replace old tool results (largest first)
               pass 1.5: replace old tool-call arguments (largest first)
               pass 2: truncate assistant narrations to 600 chars
          3. token_basis > hard_limit OR force_emergency=True  → emergency trim
               clear ALL trimmable tool results (no min-size threshold)
               clear ALL tool-call arguments (no min-size threshold)
               truncate ALL assistant narrations to 200 chars

        When trim_target / hard_limit are -1 (default), the budget is computed
        internally from _compute_message_budget — preserves backward compat.

        After trimming, if still above hard_limit: log a visible warning but do
        not crash. The overflow catch in the analysis loop is the safety net.

        Always preserved:
          - System message (index 0, OpenAI format)
          - The most recent KEEP_TAIL messages verbatim
          - All user messages (never trimmed)
        """
        if trim_target == -1 or hard_limit == -1:
            system_prompt = self._build_system_prompt()
            trim_target, hard_limit = self._compute_message_budget(system_prompt)
        else:
            system_prompt = self._build_system_prompt()

        KEEP_TAIL = 6
        TRIM_MIN_CHARS = 300
        MAX_ASSISTANT_CHARS = 600
        EMERGENCY_ASSISTANT_CHARS = 200
        PLACEHOLDER = (
            "[trimmed — result was processed; "
            "current analysis state is reflected in the system prompt above]"
        )
        ARGS_PLACEHOLDER_OPENAI = '{"_trimmed":true}'
        ARGS_PLACEHOLDER_ANTHROPIC = {"_trimmed": True}

        before_snapshot = self._context_usage_snapshot(
            messages,
            system_prompt=system_prompt,
            anthropic=anthropic,
            trim_target=trim_target,
            hard_limit=hard_limit,
        )
        token_basis = before_snapshot["history_estimate"]
        self._last_estimated_tokens = before_snapshot["prompt_estimate"]
        self._context_display_tokens = before_snapshot["prompt_estimate"]
        self._context_display_source = "estimated next prompt"
        self._context_display_trim_target = before_snapshot["trim_target_prompt"]
        self._context_display_hard_limit = before_snapshot["hard_prompt_limit"]

        if token_basis <= trim_target:
            return messages

        emergency = force_emergency or (token_basis > hard_limit)

        messages = list(messages)  # shallow copy — entries are replaced, not mutated

        first_trimmable = 0
        if not anthropic and messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            first_trimmable = 1

        protected_from = max(first_trimmable, len(messages) - KEEP_TAIL)
        trimmed_before = before_snapshot["prompt_estimate"]

        # --- Pass 1: replace tool results, largest first ---
        # Collect candidates with their size, sort descending so the biggest go first
        candidates = []
        for i in range(first_trimmable, protected_from):
            msg = messages[i]
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "tool":
                content = msg.get("content", "")
                if isinstance(content, str):
                    char_len = len(content)
                    if emergency or char_len > TRIM_MIN_CHARS:
                        candidates.append((i, char_len, "openai_tool"))
            elif role == "user" and isinstance(msg.get("content"), list):
                total = sum(
                    len(b.get("content", ""))
                    for b in msg["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                    and isinstance(b.get("content", ""), str)
                    and (emergency or len(b.get("content", "")) > TRIM_MIN_CHARS)
                )
                if total > 0:
                    candidates.append((i, total, "anthropic_tool"))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for i, _char_len, fmt in candidates:
            msg = messages[i]
            if fmt == "openai_tool":
                messages[i] = {**msg, "content": PLACEHOLDER}
            else:
                new_blocks, changed = [], False
                for block in msg["content"]:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and isinstance(block.get("content", ""), str)
                        and (emergency or len(block.get("content", "")) > TRIM_MIN_CHARS)
                    ):
                        new_blocks.append({**block, "content": PLACEHOLDER})
                        changed = True
                    else:
                        new_blocks.append(block)
                if changed:
                    messages[i] = {**msg, "content": new_blocks}

        post_pass1_snapshot = self._context_usage_snapshot(
            messages,
            system_prompt=system_prompt,
            anthropic=anthropic,
            trim_target=trim_target,
            hard_limit=hard_limit,
        )
        if not emergency and post_pass1_snapshot["history_estimate"] <= trim_target:
            after_prompt = post_pass1_snapshot["prompt_estimate"]
            freed = trimmed_before - after_prompt
            self._last_estimated_tokens = after_prompt
            self._context_display_tokens = after_prompt
            self._context_display_source = "estimated next prompt"
            self._context_display_trim_target = post_pass1_snapshot["trim_target_prompt"]
            self._context_display_hard_limit = post_pass1_snapshot["hard_prompt_limit"]
            self._report_trim(
                freed,
                trimmed_before,
                after_prompt,
                trim_target=post_pass1_snapshot["trim_target_prompt"],
                hard_limit=post_pass1_snapshot["hard_prompt_limit"],
            )
            return messages

        # --- Pass 1.5: replace large tool-call arguments in old assistant messages ---
        # Tool results (Pass 1) and tool-call arguments are stored separately:
        #   OpenAI:   role="assistant" → tool_calls[].function.arguments (JSON string)
        #   Anthropic: role="assistant" → content[type="tool_use"].input (dict)
        # Once a tool has been executed, its code/argument payload is redundant —
        # the result was already processed and world_state records the outcome.
        candidates_15 = []
        for i in range(first_trimmable, protected_from):
            msg = messages[i]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if not anthropic:
                tcs = msg.get("tool_calls") or []
                total = sum(
                    len(tc.get("function", {}).get("arguments", ""))
                    for tc in tcs
                    if isinstance(tc, dict)
                    and (emergency or len(tc.get("function", {}).get("arguments", "")) > TRIM_MIN_CHARS)
                )
                if total > 0:
                    candidates_15.append((i, total))
            else:
                blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
                total = sum(
                    len(json.dumps(b.get("input", {})))
                    for b in blocks
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                    and isinstance(b.get("input"), dict)
                    and (emergency or len(json.dumps(b.get("input", {}))) > TRIM_MIN_CHARS)
                )
                if total > 0:
                    candidates_15.append((i, total))

        candidates_15.sort(key=lambda x: x[1], reverse=True)

        for i, _ in candidates_15:
            msg = messages[i]
            if not anthropic:
                tcs = msg.get("tool_calls") or []
                new_tcs, changed = [], False
                for tc in tcs:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        args = func.get("arguments", "")
                        if isinstance(args, str) and (emergency or len(args) > TRIM_MIN_CHARS):
                            new_tcs.append({**tc, "function": {**func, "arguments": ARGS_PLACEHOLDER_OPENAI}})
                            changed = True
                            continue
                    new_tcs.append(tc)
                if changed:
                    messages[i] = {**msg, "tool_calls": new_tcs}
            else:
                blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
                new_blocks, changed = [], False
                for block in blocks:
                    if (
                        isinstance(block, dict) and block.get("type") == "tool_use"
                        and isinstance(block.get("input"), dict)
                    ):
                        if emergency or len(json.dumps(block["input"])) > TRIM_MIN_CHARS:
                            new_blocks.append({**block, "input": ARGS_PLACEHOLDER_ANTHROPIC})
                            changed = True
                            continue
                    new_blocks.append(block)
                if changed:
                    messages[i] = {**msg, "content": new_blocks}

        post_pass15_snapshot = self._context_usage_snapshot(
            messages,
            system_prompt=system_prompt,
            anthropic=anthropic,
            trim_target=trim_target,
            hard_limit=hard_limit,
        )
        if not emergency and post_pass15_snapshot["history_estimate"] <= trim_target:
            after_prompt = post_pass15_snapshot["prompt_estimate"]
            freed = trimmed_before - after_prompt
            self._last_estimated_tokens = after_prompt
            self._context_display_tokens = after_prompt
            self._context_display_source = "estimated next prompt"
            self._context_display_trim_target = post_pass15_snapshot["trim_target_prompt"]
            self._context_display_hard_limit = post_pass15_snapshot["hard_prompt_limit"]
            self._report_trim(
                freed,
                trimmed_before,
                after_prompt,
                trim_target=post_pass15_snapshot["trim_target_prompt"],
                hard_limit=post_pass15_snapshot["hard_prompt_limit"],
            )
            return messages

        # --- Pass 2: truncate assistant narrations ---
        max_chars = EMERGENCY_ASSISTANT_CHARS if emergency else MAX_ASSISTANT_CHARS
        for i in range(first_trimmable, protected_from):
            msg = messages[i]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str) and len(content) > max_chars:
                messages[i] = {**msg, "content": content[:max_chars] + " [truncated]"}
            elif isinstance(content, list):
                new_blocks, changed = [], False
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if len(text) > max_chars:
                            new_blocks.append({**block, "text": text[:max_chars] + " [truncated]"})
                            changed = True
                            continue
                    new_blocks.append(block)
                if changed:
                    messages[i] = {**msg, "content": new_blocks}

        # --- Pass 3: emergency-only tail compaction ---
        #
        # The last few messages are normally protected so the model can see the
        # immediate tool result it just requested. That breaks down for batched
        # evidence-query workflows: a single protected assistant/user pair can
        # contain dozens of PanglaoDB tool_use/tool_result blocks and remain
        # larger than the whole history budget. In emergency mode, compact
        # machine payloads even inside the tail while preserving natural user
        # text and the structural tool ids the provider expects.
        if emergency:
            for i in range(first_trimmable, len(messages)):
                msg = messages[i]
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "tool":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content != PLACEHOLDER:
                        messages[i] = {**msg, "content": PLACEHOLDER}
                    continue

                if role == "user" and isinstance(msg.get("content"), list):
                    new_blocks, changed = [], False
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            if block.get("content") != PLACEHOLDER:
                                new_blocks.append({**block, "content": PLACEHOLDER})
                                changed = True
                            else:
                                new_blocks.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if isinstance(text, str) and len(text) > EMERGENCY_ASSISTANT_CHARS:
                                new_blocks.append({**block, "text": text[:EMERGENCY_ASSISTANT_CHARS] + " [truncated]"})
                                changed = True
                            else:
                                new_blocks.append(block)
                        else:
                            new_blocks.append(block)
                    if changed:
                        messages[i] = {**msg, "content": new_blocks}
                    continue

                if role != "assistant":
                    continue

                if not anthropic:
                    tcs = msg.get("tool_calls") or []
                    if tcs:
                        new_tcs, changed = [], False
                        for tc in tcs:
                            if isinstance(tc, dict):
                                func = tc.get("function", {})
                                args = func.get("arguments", "")
                                if isinstance(args, str) and args != ARGS_PLACEHOLDER_OPENAI:
                                    new_tcs.append({**tc, "function": {**func, "arguments": ARGS_PLACEHOLDER_OPENAI}})
                                    changed = True
                                    continue
                            new_tcs.append(tc)
                        if changed:
                            messages[i] = {**msg, "tool_calls": new_tcs}
                    content = messages[i].get("content")
                    if isinstance(content, str) and len(content) > EMERGENCY_ASSISTANT_CHARS:
                        messages[i] = {**messages[i], "content": content[:EMERGENCY_ASSISTANT_CHARS] + " [truncated]"}
                else:
                    blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
                    if blocks:
                        new_blocks, changed = [], False
                        for block in blocks:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                if block.get("input") != ARGS_PLACEHOLDER_ANTHROPIC:
                                    new_blocks.append({**block, "input": ARGS_PLACEHOLDER_ANTHROPIC})
                                    changed = True
                                else:
                                    new_blocks.append(block)
                            elif isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if isinstance(text, str) and len(text) > EMERGENCY_ASSISTANT_CHARS:
                                    new_blocks.append({**block, "text": text[:EMERGENCY_ASSISTANT_CHARS] + " [truncated]"})
                                    changed = True
                                else:
                                    new_blocks.append(block)
                            else:
                                new_blocks.append(block)
                        if changed:
                            messages[i] = {**msg, "content": new_blocks}
                    content = messages[i].get("content")
                    if isinstance(content, str) and len(content) > EMERGENCY_ASSISTANT_CHARS:
                        messages[i] = {**messages[i], "content": content[:EMERGENCY_ASSISTANT_CHARS] + " [truncated]"}

        after_snapshot = self._context_usage_snapshot(
            messages,
            system_prompt=system_prompt,
            anthropic=anthropic,
            trim_target=trim_target,
            hard_limit=hard_limit,
        )
        remaining = after_snapshot["history_estimate"]
        after_prompt = after_snapshot["prompt_estimate"]
        freed = trimmed_before - after_prompt
        self._last_estimated_tokens = after_prompt
        self._context_display_tokens = after_prompt
        self._context_display_source = "estimated next prompt"
        self._context_display_trim_target = after_snapshot["trim_target_prompt"]
        self._context_display_hard_limit = after_snapshot["hard_prompt_limit"]
        self._report_trim(
            freed,
            trimmed_before,
            after_prompt,
            emergency=emergency,
            trim_target=after_snapshot["trim_target_prompt"],
            hard_limit=after_snapshot["hard_prompt_limit"],
        )

        if remaining > hard_limit:
            logger.warning(
                f"Context still above hard_limit after trimming "
                f"({remaining:,} > {hard_limit:,}). Overflow catch in API loop is the safety net."
            )
            self._print(
                f"[yellow]⚠ Context still large after compaction "
                f"(history ~{remaining:,}, next prompt ~{after_prompt:,}). "
                f"If the next API call fails, an emergency retry will be attempted.[/yellow]"
            )

        return messages

    @staticmethod
    def _is_context_overflow_error(e: Exception) -> bool:
        """Return True if the exception indicates a context window overflow.

        Intentionally broad: a false positive (extra emergency trim) is cheaper
        than a false negative (the session crashing). Checks OpenAI, Anthropic,
        and generic vLLM / local server error messages.
        """
        try:
            import openai as _openai
            if isinstance(e, _openai.BadRequestError):
                msg = str(e).lower()
                if any(kw in msg for kw in (
                    "context_length_exceeded", "maximum context",
                    "context window", "too many tokens", "exceed",
                )):
                    return True
        except ImportError:
            pass

        try:
            import anthropic as _anthropic
            if isinstance(e, _anthropic.BadRequestError):
                msg = str(e).lower()
                if any(kw in msg for kw in (
                    "context_length_exceeded", "maximum context",
                    "context window", "too many tokens", "prompt is too long",
                )):
                    return True
        except ImportError:
            pass

        # Generic fallback: covers Groq, vLLM, and other OpenAI-compatible servers
        msg = str(e).lower()
        return any(kw in msg for kw in (
            "context_length_exceeded",
            "context window",
            "maximum context length",
            "too many tokens",
            "exceeds the model",
        ))

    def _emergency_trim(self, messages: list, *, anthropic: bool = False) -> list:
        """Force maximum trimming regardless of budget — called on overflow error."""
        self._print(
            "[yellow]Context overflow detected — applying emergency trim before retry.[/yellow]"
        )
        logger.warning("Context overflow error caught; applying emergency trim")
        # trim_target=0: never skip early (token_basis > 0 always)
        # hard_limit=real_value: post-trim warning shows actual budget, not "X > 0"
        # force_emergency=True: guarantees max-aggression even if estimate shows under-budget
        system_prompt = self._build_system_prompt()
        _, hard_limit = self._compute_message_budget(system_prompt)
        return self._trim_messages_if_needed(
            messages, anthropic=anthropic,
            trim_target=0, hard_limit=hard_limit, force_emergency=True,
        )

    def _analyze_openai(self, user_message: str, max_iterations: int, continue_conversation: bool = False) -> str:
        """Run analysis loop using OpenAI API."""
        self._refresh_vertex_token_if_needed()
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
        auto_recovery_attempts = 0
        obligation_nudge_attempts = 0
        thinking_extra = self._thinking_extra()

        try:
            for iteration in range(max_iterations):
                system_prompt = self._build_system_prompt()
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = system_prompt
                trim_target, hard_limit = self._compute_message_budget(system_prompt)
                messages = self._trim_messages_if_needed(
                    messages, trim_target=trim_target, hard_limit=hard_limit,
                )
                _t0_llm = _tracing.now_ns()
                try:
                    response = self._with_llm_status(
                        lambda: self.client.chat.completions.create(
                            model=self.model,
                            max_completion_tokens=self._max_output_tokens,
                            tools=self.tools,
                            messages=messages,
                            **thinking_extra,
                        )
                    )
                except Exception as _api_err:
                    if self._is_context_overflow_error(_api_err):
                        messages = self._emergency_trim(messages)
                        if messages and messages[0].get("role") == "system":
                            messages[0]["content"] = self._build_system_prompt()
                        try:
                            response = self._with_llm_status(
                                lambda: self.client.chat.completions.create(
                                    model=self.model,
                                    max_completion_tokens=self._max_output_tokens,
                                    tools=self.tools,
                                    messages=messages,
                                    **thinking_extra,
                                )
                            )
                        except Exception as _retry_err:
                            if self._is_context_overflow_error(_retry_err):
                                self._print(
                                    "[red]Context window exhausted — unable to recover after "
                                    "emergency trim. Start a new session. Your analysis state "
                                    "is preserved in world state.[/red]"
                                )
                                if self.run_manager:
                                    self.run_manager.fail("context_window_exhausted")
                                return final_result or ""
                            raise
                    else:
                        raise

                choice = response.choices[0]
                message = choice.message

                # Capture exact token count from the API response (zero overhead —
                # already returned). Used for the context bar display.
                if response.usage and response.usage.prompt_tokens:
                    self._last_actual_tokens = response.usage.prompt_tokens
                    self._context_display_tokens = self._last_actual_tokens
                    self._context_display_source = "last actual prompt"
                    # Calibrate estimate ratio — ratchets upward, never down, capped at 4x
                    _current_est = self._context_usage_snapshot(
                        messages,
                        system_prompt=system_prompt,
                        trim_target=trim_target,
                        hard_limit=hard_limit,
                    )["prompt_estimate"]
                    if _current_est > 0 and self._last_actual_tokens > _current_est:
                        _new_ratio = self._last_actual_tokens / _current_est
                        self._token_estimate_calibration = max(
                            self._token_estimate_calibration,
                            min(_new_ratio, 4.0),
                        )

                _usage = getattr(response, "usage", None)
                _tracing.record_llm(
                    iteration, self.model,
                    getattr(_usage, "prompt_tokens", None) if _usage else None,
                    getattr(_usage, "completion_tokens", None) if _usage else None,
                    _t0_llm,
                )
                _tracing.record_llm_io(iteration, self.model, messages, message, _t0_llm)

                if choice.finish_reason == "tool_calls" and message.tool_calls:
                    # Add assistant message with tool calls
                    messages.append(message)

                    # Reasoning models emit two separate channels per turn:
                    #   reasoning_content / reasoning → raw chain-of-thought
                    #   content                       → user-facing narration
                    # OpenAI-compatible backends name the CoT field differently:
                    # Gemini/DeepSeek use `reasoning_content`, vLLM reasoning
                    # parsers (nemotron_v3, glm45, …) use `reasoning`.
                    #
                    # Print the narration (content) always; the chain-of-thought
                    # is noisy and hidden unless SCAGENT_SHOW_THINKING=1, in which
                    # case it's rendered dimmed. Previously these were collapsed
                    # with `or`, so the CoT was dumped verbatim AND the real
                    # narration was dropped on tool-calling turns.
                    _narration, _cot = self._split_reasoning_channels(message)
                    # Causal order: the model reasons first, then states intent,
                    # then acts. Print/persist the chain-of-thought BEFORE the
                    # narration so the transcript reads top-to-bottom as it happened.
                    if _cot:
                        _cot_log = self._save_thinking(_cot, iteration)
                        if os.environ.get("SCAGENT_SHOW_THINKING", "0") == "1":
                            self._print_thinking(_cot, dim=True)
                        elif _cot_log is not None:
                            self._print(
                                f"[dim]… reasoning hidden ({len(_cot)} chars) — "
                                f"{_cot_log}[/dim]"
                            )
                    if _narration:
                        self._print_thinking(_narration)

                    # Process each tool call
                    for tool_call in message.tool_calls:
                        tool_input = json.loads(tool_call.function.arguments)
                        _t0_tool = _tracing.now_ns()
                        result_json = self._execute_tool(tool_call.function.name, tool_input)
                        _tracing.record_tool(tool_call.function.name, iteration, _t0_tool)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_json,
                        })

                    # If there are pending figures, inject them as a vision user message
                    if self._pending_images:
                        if self._supports_vision():
                            messages.append(self._build_image_message(self._pending_images, "openai"))
                            self._next_llm_status_message = "Analyzing figure..."
                        elif self._vision_sidecar is not None:
                            messages.append(self._build_sidecar_text_message(self._pending_images))
                            self._next_llm_status_message = "Reading figure description..."
                        else:
                            paths = ", ".join(img["path"] for img in self._pending_images)
                            messages.append({"role": "user", "content": f"Figure(s) saved at {paths}."})
                        self._pending_images = []

                elif choice.finish_reason == "stop":
                    # Check for XML tool calls in text (local models like Qwen2.5-Coder
                    # emit <tool_call> or <tools> tags instead of structured tool_calls)
                    xml_calls = self._parse_xml_tool_calls(message.content or "")
                    if xml_calls:
                        # Treat as tool calls — build a synthetic assistant message
                        synthetic_tool_calls = [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["arguments"]),
                                },
                            }
                            for tc in xml_calls
                        ]
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": synthetic_tool_calls,
                        })
                        for tc in xml_calls:
                            tool_input = tc["arguments"] if isinstance(tc["arguments"], dict) else json.loads(tc["arguments"])
                            result_json = self._execute_tool(tc["name"], tool_input)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result_json,
                            })
                        continue

                    # Add final assistant message to history
                    messages.append({"role": "assistant", "content": message.content})

                    final_result = message.content or ""

                    self._print("\n" + "-" * 50)
                    self._print(final_result)

                    should_continue, auto_recovery_attempts = self._maybe_continue_after_failure(
                        final_result,
                        messages,
                        auto_recovery_attempts,
                        suggestions=["Provide additional instructions", "Try a different approach"],
                    )
                    if should_continue:
                        continue

                    # Spine floor: don't end with a scientific obligation unmet (the
                    # no-tool-call exit that bypasses the save/report guard).
                    should_continue, obligation_nudge_attempts = self._maybe_continue_for_obligations(
                        messages, obligation_nudge_attempts,
                    )
                    if should_continue:
                        continue

                    # Save conversation history for potential follow-ups
                    self._conversation_history = messages

                    if self.run_manager and not self._pending_checkpoint:
                        self._complete_run(final_result)
                        self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")

                    return final_result

                elif choice.finish_reason == "length":
                    # Response was truncated due to length
                    self._print("\n[Warning: Response truncated due to length]")
                    final_result = message.content or ""
                    messages.append({"role": "assistant", "content": final_result})
                    self._print(final_result)

                    # Same spine floor on the length-truncation exit.
                    should_continue, obligation_nudge_attempts = self._maybe_continue_for_obligations(
                        messages, obligation_nudge_attempts,
                    )
                    if should_continue:
                        continue

                    self._conversation_history = messages
                    if self.run_manager and not self._pending_checkpoint:
                        self._complete_run(final_result)
                        self._print(f"\n[dim]Run manifest: {self.run_manager.run_dir}/manifest.json[/dim]")
                    return final_result

                else:
                    self._print(f"\n[Debug: finish_reason={choice.finish_reason}]")
                    logger.warning(f"Unexpected finish reason: {choice.finish_reason}")
                    return self._handle_unexpected_provider_stop(
                        messages,
                        str(choice.finish_reason),
                        message_format="openai",
                    )

            return self._handle_max_iterations_reached(
                messages,
                max_iterations,
                message_format="openai",
            )

        except Exception as e:
            if self.run_manager:
                self.run_manager.fail(str(e))
            raise

        return final_result

    _TOOL_LABELS = {
        "load_data":            "Loading dataset",
        "run_cellbender":       "Running CellBender",
        "run_qc":               "Running QC",
        "score_integration":    "Scoring integration quality",
        "benchmark_integration": "Benchmarking integration (scib-metrics)",
        "normalize_and_hvg":    "Normalizing",
        "run_pca":              "Running PCA",
        "run_neighbors":        "Computing neighbors",
        "run_umap":             "Computing UMAP",
        "run_clustering":       "Clustering",
        "compare_clusterings":  "Comparing clusterings",
        "list_celltypist_models": "Listing CellTypist models",
        "check_celltypist_model": "Checking CellTypist model",
        "run_celltypist":       "Cell type annotation",
        "run_scimilarity":      "Scimilarity annotation",
        "prepare_annotation":   "Preparing annotation proposal",
        "stage_annotation_evidence": "Staging annotation evidence",
        "finalize_annotation":  "Finalizing annotation",
        "run_batch_correction": "Batch correction",
        "run_deg":              "Differential expression",
        "run_pseudobulk_deg":  "Pseudobulk DEG (DESeq2)",
        "run_gsea":             "GSEA",
        "run_spectra":          "Spectra factor analysis",
        "score_gene_signature": "Scoring gene signature",
        "query_cells":          "Querying Scimilarity reference database",
        "save_data":            "Saving data",
        "run_cluster_qc":       "Cluster QC assessment",
        "run_cluster_structure_qc": "Analyzing cluster structure",
        "run_code":             "Running code",
        "write_report":         "Writing report",
        "write_json":           "Writing JSON file",
        "run_shell":            "Running shell command",
        "install_package":      "Installing package",
        "generate_figure":      "Generating figure",
        "inspect_data":         "Inspecting data",
        "record_inspection":    "Recording data interpretation",
        "inspect_data_inputs":  "Inspecting data inputs",
        "search_papers":        "Searching papers",
        "research_findings":    "Searching literature",
        "web_search":           "Searching web",
        "review_artifact":      "Reviewing artifact",
        "read_file":            "Reading file",
        "pause_and_ask":        "Pausing for guidance",
        "describe_image":       "Describing figure (vision sidecar)",
    }

    _MCP_TOOL_LABELS = {
        # biocontext marker and annotation validation
        "bc_get_panglaodb_marker_genes": "Querying PanglaoDB markers",
        "bc_get_panglaodb_options": "Loading PanglaoDB options",
        "bc_get_cell_ontology_terms": "Searching Cell Ontology",
        "bc_search_ontology_terms": "Searching ontology terms",
        "bc_get_term_details": "Loading ontology term details",
        "bc_get_term_hierarchical_children": "Loading ontology hierarchy",
        "bc_get_available_ontologies": "Loading ontology list",
        # biocontext gene, protein, and pathway evidence
        "bc_get_human_protein_atlas_info": "Querying Human Protein Atlas",
        "bc_get_go_terms_by_gene": "Querying GO terms",
        "bc_get_reactome_info_by_identifier": "Querying Reactome",
        "bc_get_string_id": "Mapping STRING identifier",
        "bc_get_string_interactions": "Querying STRING interactions",
        "bc_get_string_network_image": "Generating STRING network",
        "bc_get_string_similarity_scores": "Querying STRING similarity",
        "bc_get_ensembl_id_from_gene_symbol": "Mapping Ensembl gene ID",
        "bc_get_kegg_id_by_gene_symbol": "Mapping KEGG gene ID",
        "bc_query_kegg": "Querying KEGG",
        "bc_get_uniprot_id_by_protein_symbol": "Mapping UniProt protein ID",
        "bc_get_uniprot_protein_info": "Querying UniProt",
        "bc_get_alphafold_info_by_protein_symbol": "Querying AlphaFold",
        "bc_get_protein_domains": "Querying protein domains",
        "bc_get_interpro_entry": "Querying InterPro entry",
        "bc_search_interpro_entries": "Searching InterPro",
        # literature MCPs
        "search_abstracts": "Searching PubMed abstracts",
        "bc_get_europepmc_articles": "Searching Europe PMC",
        "bc_get_europepmc_fulltext": "Fetching Europe PMC full text",
        "bc_get_biorxiv_preprint_details": "Fetching bioRxiv preprint",
        "bc_get_recent_biorxiv_preprints": "Searching bioRxiv preprints",
        "bc_search_google_scholar_publications": "Searching Google Scholar",
        # translational/drug/trial MCPs
        "bc_query_open_targets_graphql": "Querying Open Targets",
        "bc_get_open_targets_query_examples": "Loading Open Targets examples",
        "bc_get_open_targets_graphql_schema": "Loading Open Targets schema",
        "bc_search_drugs_fda": "Searching FDA drugs",
        "bc_get_drug_label_info": "Fetching FDA drug label",
        "bc_search_studies": "Searching clinical trials",
        "bc_get_study_details": "Fetching clinical trial details",
    }

    # Tools that should leave persistent start/done lines in the terminal. This
    # includes tools with their own tqdm/progress output and the main analysis
    # actions so the user can see the pipeline history after each step finishes.
    _STREAMING_TOOLS = {
        "load_data",
        "run_cellbender",
        "normalize_and_hvg",
        "run_pca",
        "run_neighbors",
        "run_clustering",
        "compare_clusterings",
        "run_celltypist",
        "run_scimilarity",
        "prepare_annotation",     # runs rank_genes_groups, can be slow
        "stage_annotation_evidence",
        "run_batch_correction",   # scVI tqdm training bar, Scanorama verbose
        "diagnose_batch_effect",  # multi-step diagnostic; leave a durable done line
        "run_umap",               # UMAP can take minutes on large datasets
        "run_qc",                 # Scrublet progress on large datasets
        "score_integration",
        "benchmark_integration",  # scib-metrics runs many metrics
        "run_deg",                # rank_genes_groups can be slow
        "run_pseudobulk_deg",     # DESeq2 fitting
        "run_gsea",               # GSEA permutations
        "run_spectra",
        "score_gene_signature",
        "query_cells",
        "save_data",
        "run_cluster_qc",
        "run_cluster_structure_qc",
        "generate_figure",
        "run_code",               # unknown — user code may print progress
        "run_shell",              # external system checks/commands should be visible in terminal history
    }

    def _display_label_for_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a concise terminal label for native and MCP tools."""
        if tool_name == "run_code":
            return self._TOOL_LABELS.get(tool_name, tool_input.get("description", "Running code"))

        if tool_name in self._TOOL_LABELS:
            return self._TOOL_LABELS[tool_name]
        if tool_name in self._MCP_TOOL_LABELS:
            return self._MCP_TOOL_LABELS[tool_name]

        if self._mcp_client and self._mcp_client.has_tool(tool_name):
            if tool_name.startswith("bc_get_"):
                stem = tool_name.removeprefix("bc_get_")
                return "Querying " + stem.replace("_", " ").title()
            if tool_name.startswith("bc_search_"):
                stem = tool_name.removeprefix("bc_search_")
                return "Searching " + stem.replace("_", " ").title()
            if tool_name.startswith("bc_query_"):
                stem = tool_name.removeprefix("bc_query_")
                return "Querying " + stem.replace("_", " ").title()
            return "Using MCP tool " + tool_name.replace("_", " ")

        return tool_name.replace("_", " ").title()

    def _should_print_persistent_tool_progress(self, tool_name: str) -> bool:
        """Whether verbose mode should leave durable start/done lines."""
        if tool_name in self._STREAMING_TOOLS:
            return True
        return bool(self._mcp_client and self._mcp_client.has_tool(tool_name))

    def _print_terminal_summary(self, tool_name: str, result_data: dict) -> None:
        """Surface a reasoning/diagnostic tool's own findings to the terminal.

        Tools opt in by returning ``terminal_summary`` (a list of short strings,
        or a single string) in their result — e.g. diagnose_batch_effect's
        verdict + evidence + recommendation, or run_cluster_structure_qc's
        per-cluster decisions. This lets the user see the analytical reasoning
        rather than only a spinner and a checkmark. No-op if the field is absent.
        """
        summary = result_data.get("terminal_summary")
        if not summary:
            return
        if isinstance(summary, str):
            summary = [summary]
        from rich.console import Console

        console = Console()
        try:
            console.print(f"[dim]  ⤷ {tool_name}:[/dim]")
            for line in summary:
                # markup=False: finding text may contain brackets/markup chars.
                console.print(f"    {line}", style="dim", markup=False)
        except Exception:  # pragma: no cover - display must never break a run
            pass

    @staticmethod
    def _cluster_qc_terminal_summary(result_data: dict) -> list[str] | None:
        """Compose terminal findings for the cluster-QC tools from their already
        post-processed result (per-cluster decisions + the synthesized cleanup
        recommendation). Returns None when there is nothing meaningful to show."""
        decisions = result_data.get("cluster_decisions") or {}
        lines: list[str] = []
        if decisions:
            from collections import Counter

            actions = Counter(str(d.get("recommended_action", "keep")) for d in decisions.values())
            lines.append(
                f"{len(decisions)} clusters reviewed: "
                + ", ".join(f"{n} {a}" for a, n in actions.items())
            )
            for clu, d in decisions.items():
                action = str(d.get("recommended_action", "keep"))
                if action == "keep":
                    continue
                reasons = d.get("reasons") or []
                detail = "; ".join(str(r) for r in reasons[:2]) if reasons else str(d.get("severity", ""))
                lines.append(f"cluster {clu}: {action} ({detail})")
        rec = (
            result_data.get("recommended_next_action")
            or result_data.get("metric_qc_interpretation")
        )
        if result_data.get("cleanup_resolved"):
            lines.append(f"resolved: {result_data['cleanup_resolved']}")
        if rec:
            lines.append(f"→ {rec}")
        return lines[:25] or None

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool and return JSON result."""
        from pathlib import Path

        from rich.console import Console
        from rich.status import Status

        console = Console()
        logger.debug("Tool call: %s with %s", tool_name, tool_input)

        # Only block truly pipeline-progressing tools when checkpoint pending
        # Allow flexible tools (run_code, inspection, visualization) to proceed
        if self._pending_checkpoint and self._is_action_tool(tool_name):
            if (
                self._pending_checkpoint.get("kind") == "multi_dataset_loading"
                and tool_name in {"run_code", "load_data"}
            ):
                return self._blocked_by_checkpoint_result(tool_name)
            if (
                self._pending_checkpoint.get("kind") == "multi_sample_strategy"
                and tool_name == "run_code"
            ):
                return self._blocked_by_checkpoint_result(tool_name)
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
                "run_pca",
                "run_neighbors",
                "run_umap",
                "run_clustering",
                "run_celltypist",
                "run_scimilarity",
                "run_batch_correction",
                "run_deg",
            }

            run_root = self.run_manager.run_dir

            def _inside_run_root(p: Path) -> bool:
                """True if path is the run dir or one of its descendants."""
                try:
                    p_resolved = p.resolve() if p.is_absolute() else (run_root / p).resolve()
                    p_resolved.relative_to(run_root.resolve())
                    return True
                except (ValueError, OSError):
                    return False

            if tool_name in figure_tools:
                requested = tool_input.get("output_path")
                if not requested:
                    stem = f"{tool_input.get('plot_type', 'figure')}_{tool_input.get('color_by', 'plot')}"
                    tool_input["output_path"] = self.run_manager.get_figure_path(_sanitize_name(stem))
                else:
                    requested_path = Path(requested)
                    if not requested_path.is_absolute():
                        # Any relative path (e.g. "figures/umap.png", "umap.png") gets
                        # routed through the run manager so the directory always exists.
                        tool_input["output_path"] = self.run_manager.get_figure_path(
                            _sanitize_name(requested_path.stem),
                            ext=requested_path.suffix.lstrip(".") or "png",
                        )
                    elif not _inside_run_root(requested_path):
                        # Absolute path outside the run dir — agent invented a custom
                        # location. Re-route under the run dir for consistency.
                        rerouted = self.run_manager.get_figure_path(
                            _sanitize_name(requested_path.stem),
                            ext=requested_path.suffix.lstrip(".") or "png",
                        )
                        logger.info(
                            "Rerouting generate_figure output_path from %s to %s "
                            "(outside run dir).", requested, rerouted,
                        )
                        tool_input["output_path"] = rerouted

            if tool_name == "run_qc":
                requested_dir = tool_input.get("figure_dir")
                run_figures_dir = str(self.run_manager._ensure(self.run_manager.dirs["figures"]))
                if not requested_dir:
                    tool_input["figure_dir"] = run_figures_dir
                elif not _inside_run_root(Path(requested_dir)):
                    logger.info(
                        "Rerouting run_qc figure_dir from %s to %s (outside run dir).",
                        requested_dir, run_figures_dir,
                    )
                    tool_input["figure_dir"] = run_figures_dir

            if tool_name == "compare_clusterings" and tool_input.get("generate_figures"):
                requested_dir = tool_input.get("figure_dir")
                run_figures_dir = str(self.run_manager._ensure(self.run_manager.dirs["figures"]))
                if not requested_dir:
                    tool_input["figure_dir"] = run_figures_dir
                elif not _inside_run_root(Path(requested_dir)):
                    logger.info(
                        "Rerouting compare_clusterings figure_dir from %s to %s (outside run dir).",
                        requested_dir, run_figures_dir,
                    )
                    tool_input["figure_dir"] = run_figures_dir

            if tool_name == "run_gsea":
                requested_dir = tool_input.get("output_dir")
                gsea_dir = str(self.run_manager._ensure(self.run_manager.dirs["gsea"]))
                if not requested_dir:
                    tool_input["output_dir"] = gsea_dir
                else:
                    requested_path = Path(requested_dir)
                    if not requested_path.is_absolute():
                        if requested_path == Path(".") or requested_path.name == run_root.name:
                            tool_input["output_dir"] = gsea_dir
                    elif not _inside_run_root(requested_path):
                        logger.info(
                            "Rerouting run_gsea output_dir from %s to %s (outside run dir).",
                            requested_dir, gsea_dir,
                        )
                        tool_input["output_dir"] = gsea_dir

            # When save_checkpoints is False, NEVER save intermediate h5ad files
            # Only save when save_checkpoints is True OR when it's save_data tool
            if tool_name in checkpoint_tools and not self.save_checkpoints:
                # Always remove output_path for checkpoint tools when not saving intermediates
                tool_input.pop("output_path", None)

            if self.save_checkpoints and tool_name in checkpoint_tools and not tool_input.get("output_path"):
                tool_input["output_path"] = self.run_manager.get_intermediate_path(_sanitize_name(tool_name))

        _prepare_tool_paths()
        loading_guard = self._multi_dataset_loading_guard(tool_name, tool_input)
        if loading_guard is not None:
            return loading_guard
        auto_checkpoint_path = self._maybe_auto_checkpoint(tool_name, tool_input)
        self._apply_world_state_overrides(tool_name, tool_input)
        # Don't re-sync here — we already synced at the start of analyze() and after
        # the previous tool call. Re-syncing before execution hits adata.X on every
        # tool call without any adata change having occurred.
        before_snapshot = self.world_state.snapshot()
        if self.run_manager:
            self.run_manager.append_log(f"START {tool_name} {json.dumps(tool_input, default=str)}")

        for hint_key in ("context", "biological_context_hint"):
            hint = tool_input.get(hint_key)
            if hint:
                self.world_state.add_context_hint(str(hint))

        if tool_name == "pause_and_ask":
            result_json = self._handle_pause_and_ask(tool_input)
        # Special handling for install_package - requires approval
        elif tool_name == "install_package":
            result_json = self._handle_install_package(tool_input)
        elif tool_name == "describe_image":
            result_json = self._handle_describe_image(tool_input)
        else:
            # For run_code, inject the output_dir before dispatch
            if tool_name == "run_code" and self.run_manager:
                tool_input["output_dir"] = str(self.run_manager.run_dir)
            if tool_name in {"run_celltypist", "run_scimilarity"}:
                biological_context = self._get_biological_context(self._active_request)
                if biological_context:
                    tool_input.setdefault("biological_context", biological_context)
                    species = str(biological_context.get("species") or "").lower()
                    if species in {"human", "mouse"}:
                        tool_input.setdefault("organism", species)
            cleanup_authorization = self._cleanup_authorization_for_tool(tool_name, tool_input)
            if cleanup_authorization is not None:
                tool_input["cleanup_authorization"] = cleanup_authorization
            annotation_validation_block = self._annotation_validation_guard(tool_name, tool_input)

            # Build the display label
            label = self._display_label_for_tool(tool_name, tool_input)

            def _dispatch():
                """Call process_tool_call (native tools) or MCP client (external tools)."""
                # Route to MCP client if this is an external MCP tool
                if self._mcp_client and self._mcp_client.has_tool(tool_name):
                    try:
                        result_dict = self._mcp_client.call_tool(tool_name, tool_input)
                        return json.dumps(result_dict, indent=2, default=str), self.adata
                    except Exception as mcp_err:
                        err_payload = {
                            "status": "error",
                            "tool": tool_name,
                            "message": str(mcp_err),
                            "recovery_options": [
                                "Check that the MCP server is running and the tool name is correct.",
                                "Try run_code as a fallback for custom analysis.",
                            ],
                        }
                        return json.dumps(err_payload, indent=2), self.adata

                # Native tool dispatch with Python warning capture
                if annotation_validation_block is not None:
                    return json.dumps(annotation_validation_block, indent=2), self.adata

                import warnings as _warnings
                with _warnings.catch_warnings(record=True) as _caught:
                    _warnings.simplefilter("always")
                    _rj, _ad = process_tool_call(
                        tool_name,
                        tool_input,
                        self.adata,
                        world_state=self.world_state,
                        run_manager=self.run_manager,
                    )
                if _caught:
                    try:
                        _rd = json.loads(_rj)
                        _msgs = list({str(w.message) for w in _caught})
                        existing = _rd.get("runtime_warnings") or []
                        _rd["runtime_warnings"] = existing + _msgs
                        _rj = json.dumps(_rd, indent=2)
                    except Exception:
                        pass
                return _rj, _ad

            if self.verbose and self._should_print_persistent_tool_progress(tool_name):
                # Streaming tools produce their own tqdm/progress output. Using
                # Rich's Live (console.status) fights with tqdm and blanks the
                # terminal. MCP calls should also leave a durable audit trail in
                # the live terminal because they are external evidence queries.
                console.print(f"[cyan]▶[/cyan] {label}...")
                result_json, self.adata = _dispatch()
                console.print(f"[green]✓[/green] {label} done")
            elif self.verbose:
                with console.status(f"{label}...", spinner="dots"):
                    result_json, self.adata = _dispatch()
            else:
                result_json, self.adata = _dispatch()

        # Check for image in result and store for vision
        try:
            result_data = json.loads(result_json)
            # Only re-sync when the tool actually modifies adata (action tools).
            # Inspection, figure, and search tools don't change the matrix, so
            # syncing them would hit adata.X for no benefit.
            # Invalidate the inspect cache first so the sync re-runs inspect_data
            # with the fresh adata state (e.g. after QC filtering, normalization).
            if self._is_action_tool(tool_name):
                self.world_state.invalidate_inspect_cache()
                self._sync_world_state()

            # If there's an image directly embedded in the result, queue it
            if "image_base64" in result_data:
                image_context = result_data.get("image_context", {})
                figure_path = (
                    result_data.get("output_path")
                    or result_data.get("figure_path")
                    or image_context.get("output_path")
                    or "figure.png"
                )
                self._pending_images.append({
                    "base64": result_data["image_base64"],
                    "mime": result_data.get("image_mime", "image/png"),
                    "path": figure_path,
                    "role": "figure",
                })
                # Remember context for sidecar describe_image follow-ups.
                ctx_for_index = dict(image_context)
                ctx_for_index.setdefault("plot_type", result_data.get("plot_type"))
                ctx_for_index.setdefault("color_by", result_data.get("color_by"))
                ctx_for_index.setdefault("producing_tool", tool_name)
                self._figure_context_index[figure_path] = {
                    k: v for k, v in ctx_for_index.items() if v is not None
                }
                # Remove base64 from JSON to keep response small
                del result_data["image_base64"]
                if "image_mime" in result_data:
                    del result_data["image_mime"]
                result_data["image_included"] = True
                result_json = json.dumps(result_data, indent=2)

            # Auto-load any figure artifacts (e.g. QC plots, UMAP) that weren't
            # embedded directly — encode them so the LLM can see and interpret them.
            already_loaded = {img["path"] for img in self._pending_images}
            for artifact in result_data.get("artifacts_created", []) or []:
                if artifact.get("kind") != "figure":
                    continue
                path = artifact.get("path", "")
                if not path or path in already_loaded or not os.path.exists(path):
                    continue
                try:
                    self._pending_images.append({
                        "base64": encode_image_base64(path),
                        "mime": get_image_mime_type(path),
                        "path": path,
                        "role": artifact.get("role", "figure"),
                    })
                    already_loaded.add(path)
                    self._figure_context_index.setdefault(path, {
                        "role": artifact.get("role", "figure"),
                        "producing_tool": tool_name,
                    })
                except Exception as enc_err:
                    logger.warning("Failed to encode figure %s for vision: %s", path, enc_err)

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
            if auto_checkpoint_path and result_data.get("status") == "ok":
                result_data["auto_checkpoint_saved"] = str(auto_checkpoint_path)
            checkpoint = None
            if tool_name == "run_cluster_qc" and result_data.get("status") == "ok":
                cleanup_checkpoint = self._cluster_cleanup_checkpoint_from_result(result_data)
                if cleanup_checkpoint is not None:
                    result_data["cluster_cleanup_proposal"] = cleanup_checkpoint.get("proposal", {})
                    result_data["cleanup_policy"] = self._cleanup_policy()
                    if self._cluster_structure_qc_required_after_metric_qc(result_data):
                        result_data["cluster_structure_qc_required"] = True
                        result_data["recommended_next_tool"] = "run_cluster_structure_qc"
                        result_data["metric_qc_interpretation"] = (
                            "Metric QC has flagged problematic clusters for structure review; "
                            "this is not yet a removal decision."
                        )
                        result_data["recommended_next_tool_input"] = {
                            "cluster_key": result_data.get("cluster_key"),
                            "clusters_to_analyze": [
                                str(c)
                                for c in (
                                    result_data.get("proposed_removal", [])
                                    + result_data.get("ambiguous", [])
                                )
                            ],
                        }
                    elif cleanup_checkpoint.get("auto_allowed"):
                        self._active_cleanup_authorization = {
                            "source": "auto_policy",
                            "proposal": cleanup_checkpoint.get("proposal", {}),
                            "reason": cleanup_checkpoint.get("auto_reason", ""),
                        }
                        result_data["cleanup_authorization_available"] = self._active_cleanup_authorization
                    else:
                        checkpoint = cleanup_checkpoint
                else:
                    self._active_cleanup_authorization = None
            if tool_name == "run_cluster_structure_qc" and result_data.get("status") == "ok":
                refined_checkpoint = self._refine_cluster_cleanup_checkpoint_from_structure(result_data)
                if refined_checkpoint is not None:
                    result_data["cluster_cleanup_proposal"] = refined_checkpoint.get("proposal", {})
                    result_data["cleanup_policy"] = {
                        "mode": "auto_after_structure_qc",
                        "max_auto_cleanup_pct": 15.0,
                        "requires_structure_qc": True,
                    }
                    if refined_checkpoint.get("resolved"):
                        self._active_cleanup_authorization = None
                        self._clear_pending_checkpoint(refined_checkpoint.get("summary"))
                        result_data["cleanup_resolved"] = refined_checkpoint.get("resolved_action")
                        result_data["review_clusters_kept"] = refined_checkpoint.get("review_clusters", [])
                        result_data["recommended_next_action"] = (
                            "Proceed without cell removal; structure QC did not synthesize "
                            "a cleanup set from the reviewed metric-flagged clusters."
                        )
                    elif refined_checkpoint.get("auto_allowed"):
                        self._active_cleanup_authorization = {
                            "source": "auto_structure_qc",
                            "proposal": refined_checkpoint.get("proposal", {}),
                            "reason": refined_checkpoint.get("auto_reason", ""),
                        }
                        result_data["cleanup_authorization_available"] = self._active_cleanup_authorization
                        result_data["recommended_next_tool"] = "run_code"
                        result_data["recommended_next_action"] = (
                            "Remove exactly the structure-synthesized cleanup clusters, "
                            "then rerun normalize_and_hvg, PCA, neighbors, UMAP, clustering, "
                            "cluster QC, and structure QC."
                        )
                    else:
                        checkpoint = refined_checkpoint
            if checkpoint is None:
                checkpoint = self._build_multi_dataset_loading_checkpoint(
                    tool_name,
                    result_data,
                )
            if checkpoint is None:
                checkpoint = self._build_post_concatenation_strategy_checkpoint(
                    tool_name,
                    tool_input,
                    result_data,
                )
            if checkpoint is None:
                checkpoint = self._build_multi_sample_strategy_checkpoint(
                    tool_name,
                    result_data,
                )
            if checkpoint is None:
                checkpoint = self._build_checkpoint_payload(tool_name, tool_input, result_data)
            if checkpoint is None:
                checkpoint = self._build_recovery_checkpoint(tool_name, tool_input, result_data)
            if checkpoint is not None:
                result_data["checkpoint_required"] = True
                result_data["checkpoint"] = checkpoint
                self._set_pending_checkpoint(checkpoint)
            self.world_state.apply_tool_result(tool_name, result_data, adata=self.adata)

            # Surface a reasoning/diagnostic tool's own findings to the terminal.
            # diagnose_batch_effect emits result["terminal_summary"] itself; the
            # cluster-QC tools' decisions are composed here from post-processed
            # fields. Any tool that sets terminal_summary gets printed.
            if (
                tool_name in ("run_cluster_qc", "run_cluster_structure_qc")
                and "terminal_summary" not in result_data
            ):
                cqc = self._cluster_qc_terminal_summary(result_data)
                if cqc:
                    result_data["terminal_summary"] = cqc
            if self.verbose:
                self._print_terminal_summary(tool_name, result_data)

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
                if tool_name == "run_code" and tool_input.get("cleanup_authorization"):
                    self._active_cleanup_authorization = None
                    cleanup_executed = any(
                        check.get("name") == "destructive_cluster_removal"
                        and check.get("status") == "passed"
                        for check in result_data.get("preflight_checks", []) or []
                    )
                    if (
                        cleanup_executed
                        and self._pending_checkpoint
                        and self._pending_checkpoint.get("kind") == "cluster_qc_cleanup"
                    ):
                        self._clear_pending_checkpoint("cleanup executed")
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
            elif status == "error":
                err_msg = result_data.get('message') or result_data.get('error', '')
                err_short = err_msg[:120] + ("…" if len(err_msg) > 120 else "")
                self._print(f"    [red]✗ Error:[/red] {err_short}")
                # Show captured output before the crash if available
                pre_crash = result_data.get('output', '')
                if pre_crash and pre_crash.strip():
                    self._print(f"    [dim]Output before error:[/dim] {pre_crash.strip()[:200]}")
            # Only show verification failures when the tool didn't already report
            # an error — otherwise we'd print two lines saying the same thing.
            if status != "error":
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
        text_parts.extend(getattr(self.world_state, "context_hints", []) or [])
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

    def _handle_pause_and_ask(self, tool_input: Dict[str, Any]) -> str:
        """Handle pause_and_ask tool — create a pending checkpoint from LLM-initiated pause."""
        if self._pending_checkpoint:
            checkpoint = self._pending_checkpoint or {}
            return json.dumps({
                "status": "ok",
                "tool": "pause_and_ask",
                "paused": True,
                "question": checkpoint.get("question", tool_input.get("question", "")),
                "context": checkpoint.get("summary", tool_input.get("context", "")),
                "options": checkpoint.get("options", tool_input.get("options", [])),
                "option_actions": checkpoint.get("option_actions", []),
                "decision_key": checkpoint.get("decision_key", ""),
                "kind": checkpoint.get("kind"),
                "action_inputs": checkpoint.get("action_inputs", {}),
                "message": (
                    "Analysis is already paused at a structured runtime checkpoint. "
                    "End the turn; the runtime renders the existing choices."
                ),
            }, indent=2)

        question = tool_input.get("question", "")
        context = tool_input.get("context", "")
        options = tool_input.get("options") or []
        post_investigation = self._post_investigation_strategy_checkpoint()
        if post_investigation is not None:
            decision_text = " ".join(
                [str(question), str(context)]
                + [str(option) for option in options]
                + [str(action) for action in (tool_input.get("option_actions") or [])]
            ).lower()
            if re.search(r"\b(scvi|integrat|batch[- ]?correct|uncorrected|keep)\b", decision_text):
                self._set_pending_checkpoint(post_investigation)
                return json.dumps({
                    "status": "ok",
                    "tool": "pause_and_ask",
                    "paused": True,
                    "question": post_investigation.get("question", question),
                    "context": post_investigation.get("context", context),
                    "options": post_investigation.get("options", options),
                    "option_actions": post_investigation.get("option_actions", []),
                    "decision_key": post_investigation.get("decision_key", "multi_sample_strategy"),
                    "kind": post_investigation.get("kind", "multi_sample_strategy"),
                    "action_inputs": post_investigation.get("action_inputs", {}),
                    "message": (
                        "Using the structured post-diagnostic multi-sample strategy "
                        "checkpoint instead of an ad hoc integration pause."
                    ),
                }, indent=2)
        cleanup_text = " ".join([str(question), str(context)] + [str(option) for option in options]).lower()
        if (
            re.search(r"\b(remove|drop|filter|exclude|subset)\b", cleanup_text)
            and re.search(r"\b(cluster|clusters|cells?)\b", cleanup_text)
        ):
            return json.dumps({
                "status": "error",
                "tool": "pause_and_ask",
                "message": (
                    "Do not create a manual cleanup decision after structure QC unless a "
                    "structure-refined cluster cleanup checkpoint is pending. If structure QC "
                    "synthesized no removal set, keep the reviewed clusters for now, document "
                    "the caveat, and continue."
                ),
                "recovery_options": [
                    "Proceed to annotation with the structure-reviewed clusters kept.",
                    "If the user explicitly asks for stricter cleanup, run a new structure-aware proposal instead of inventing a keep-mask removal.",
                ],
            }, indent=2)
        option_actions = tool_input.get("option_actions") or self._stable_option_actions(options)
        if len(option_actions) != len(options):
            option_actions = self._stable_option_actions(options)

        checkpoint = {
            "kind": "llm_pause",
            "question": question,
            "context": context,
            "options": options,
            "option_actions": option_actions,
            "summary": context,
            "decision_key": tool_input.get("decision_key") or "llm_pause",
            "allow_custom": bool(tool_input.get("allow_custom", True)),
            "artifacts": [],
        }
        self._set_pending_checkpoint(checkpoint)
        return json.dumps({
            "status": "ok",
            "tool": "pause_and_ask",
            "paused": True,
            "question": question,
            "context": context,
            "options": options,
            "option_actions": option_actions,
            "decision_key": checkpoint["decision_key"],
            "message": (
                "Analysis paused. Explain why input is needed and end the turn; "
                "the runtime renders the choices."
            ),
        }, indent=2)

    def _maybe_auto_checkpoint(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[str]:
        """Save adata to a checkpoint file before destructive operations in smart_autonomous mode."""
        if not self.smart_autonomous or self.adata is None:
            return None

        is_qc_filter = tool_name == "run_qc" and tool_input.get("confirm_filtering")
        is_batch_correction = tool_name == "run_batch_correction"
        if not (is_qc_filter or is_batch_correction):
            return None

        label = "pre_qc_filter" if is_qc_filter else "pre_batch_correction"
        try:
            if self.run_manager:
                out_path = self.run_manager.get_intermediate_path(label)
            else:
                from pathlib import Path as _Path
                out_path = str(_Path(self.output_dir) / f"checkpoint_{label}.h5ad")

            if self.verbose:
                print(f"▶ Saving checkpoint {label}...")
            save_details = write_h5ad_safe(self.adata, out_path)
            if self.verbose:
                print(f"✓ Checkpoint saved: {out_path}")
                for warning in save_details.get("warnings", []):
                    print(f"  {warning}")
            return out_path
        except Exception as exc:
            logger.warning("Auto-checkpoint save failed for %s: %s", label, exc)
            return None

    def _handle_describe_image(self, tool_input: Dict[str, Any]) -> str:
        """Run a saved figure through the vision sidecar and return text-only output.

        The agent only exposes this tool when ``_use_sidecar_for_images()`` is True,
        but we still defensively handle the case where the sidecar is missing.
        """
        figure_path = (tool_input.get("figure_path") or "").strip()
        question = tool_input.get("question") or ""

        if not figure_path:
            return json.dumps({
                "status": "error",
                "tool": "describe_image",
                "message": "figure_path is required.",
            }, indent=2)

        sidecar = self._vision_sidecar
        if sidecar is None:
            return json.dumps({
                "status": "unavailable",
                "tool": "describe_image",
                "message": (
                    "SCAGENT_VISION_MODEL is not configured; describe_image cannot run. "
                    "If the main model is multimodal, use review_figure instead."
                ),
            }, indent=2)

        abs_path = os.path.abspath(figure_path)
        if not os.path.exists(abs_path):
            return json.dumps({
                "status": "error",
                "tool": "describe_image",
                "figure_path": abs_path,
                "message": f"Figure not found: {abs_path}",
            }, indent=2)

        # Reuse base64 from _pending_images when it matches; otherwise read from disk.
        b64 = None
        mime = None
        for pending in self._pending_images:
            if os.path.abspath(pending.get("path", "")) == abs_path:
                b64 = pending.get("base64")
                mime = pending.get("mime")
                break
        if b64 is None:
            try:
                b64 = encode_image_base64(abs_path)
                mime = get_image_mime_type(abs_path)
            except Exception as exc:
                return json.dumps({
                    "status": "error",
                    "tool": "describe_image",
                    "figure_path": abs_path,
                    "message": f"Failed to read figure: {exc}",
                }, indent=2)

        ctx = self._figure_context_index.get(abs_path) or self._figure_context_index.get(figure_path) or {}
        img_payload = [{
            "base64": b64,
            "mime": mime or "image/png",
            "path": abs_path,
            "role": "figure",
            "image_context": ctx,
        }]

        try:
            world_state = self.world_state.snapshot()
        except Exception:
            world_state = None

        result = sidecar.describe(
            img_payload,
            world_state=world_state,
            question=question or None,
            comparative=False,
        )
        if self.run_manager:
            try:
                self.run_manager.append_event(
                    "vision_sidecar_call",
                    {
                        "model": result.get("model"),
                        "n_images": 1,
                        "latency_ms": result.get("latency_ms"),
                        "cache_hits": result.get("cache_hits"),
                        "status": result.get("status"),
                        "paths": [abs_path],
                        "via_tool": "describe_image",
                    },
                )
                self.run_manager.append_log(
                    f"vision_sidecar via=describe_image status={result.get('status')} "
                    f"model={result.get('model')} latency_ms={result.get('latency_ms')} "
                    f"cache_hit={bool(result.get('cache_hits'))} "
                    f"question={(question or '')[:80]!r} path={abs_path}"
                )
            except Exception:
                pass

        if result.get("status") != "ok":
            return json.dumps({
                "status": "error",
                "tool": "describe_image",
                "figure_path": abs_path,
                "model": result.get("model"),
                "message": result.get("error", "vision sidecar failed"),
            }, indent=2)

        return json.dumps({
            "status": "ok",
            "tool": "describe_image",
            "figure_path": abs_path,
            "model": result.get("model"),
            "question": question or None,
            "description": result.get("text", ""),
            "cache_hit": bool(result.get("cache_hits")),
            "latency_ms": result.get("latency_ms"),
        }, indent=2)

    def _handle_install_package(self, tool_input: Dict[str, Any]) -> str:
        """Handle install_package tool - requires user approval."""
        import subprocess
        import sys

        package = tool_input["package"]
        reason = tool_input["reason"]

        print(f"\n{'='*50}")
        print(f"PACKAGE INSTALL REQUEST")
        print(f"Package: {package}")
        print(f"Reason: {reason}")
        print('='*50)

        try:
            from ..terminal import DecisionChoice, prompt_for_decision

            selection = prompt_for_decision(
                "Approve this package installation?",
                [
                    DecisionChoice("Do not install", "deny"),
                    DecisionChoice(f"Install {package}", "approve"),
                ],
                default_index=0,
                allow_custom=False,
            )
            approved = selection.action == "approve"
        except (EOFError, KeyboardInterrupt):
            approved = False

        if approved:
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
            response = self._with_llm_status(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=self._max_output_tokens,
                    system=self._build_system_prompt(),
                    messages=[{"role": "user", "content": message}],
                )
            )
            for content in response.content:
                if hasattr(content, "text"):
                    return content.text
            return ""
        elif self.provider in {"openai", "groq"}:
            # OpenAI
            response = self._with_llm_status(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=self._max_output_tokens,
                    messages=[
                        {"role": "system", "content": self._build_system_prompt()},
                        {"role": "user", "content": message},
                    ],
                    **self._thinking_extra(),
                )
            )
            return self._strip_model_artifacts(response.choices[0].message.content or "")
        elif self.provider == "codex":
            messages = [{
                "role": "user",
                "content": (
                    "Answer the following user question directly. Return kind='final' in the "
                    "required JSON schema; do not call tools for this lightweight chat method.\n\n"
                    f"User question: {message}"
                ),
            }]
            decision = self._request_codex_decision(messages)
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
        self._pending_images = []
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
