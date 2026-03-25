"""
Main agent class for autonomous single-cell analysis.

Uses Claude or OpenAI API with tools to perform single-cell analysis tasks.
Returns structured JSON from tools for reliable LLM reasoning.
Creates run directories with manifests for reproducibility.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
import logging
from datetime import datetime

from .tools import get_tools, get_openai_tools, process_tool_call
from .prompts import SYSTEM_PROMPT
from .run_manager import RunManager, create_run

logger = logging.getLogger(__name__)

Provider = Literal["anthropic", "openai"]

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

    Uses Claude or OpenAI API to analyze single-cell data following lab best practices.
    All tool calls return structured JSON for reliable LLM reasoning.
    Optionally creates run directories with manifests for reproducibility.

    Parameters
    ----------
    provider : str, default "anthropic"
        API provider: "anthropic" or "openai".
    api_key : str, optional
        API key. If not provided, reads from ANTHROPIC_API_KEY or OPENAI_API_KEY.
    model : str, optional
        Model to use. Defaults: "claude-sonnet-4-20250514" (Anthropic) or "gpt-4o" (OpenAI).
    verbose : bool, default True
        Print agent outputs.
    create_run_dir : bool, default True
        Create structured run directory with manifest.
    output_dir : str, default "."
        Base directory for run outputs.

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
        verbose: bool = True,
        create_run_dir: bool = True,
        output_dir: str = ".",
        save_checkpoints: bool = False,
    ):
        # Use environment defaults if not specified
        if provider is None:
            provider = os.environ.get("SCAGENT_PROVIDER", "anthropic")
        if model is None:
            model = os.environ.get("SCAGENT_MODEL")  # None = use provider default

        self.provider = provider
        self.verbose = verbose
        self.create_run_dir = create_run_dir
        self.output_dir = output_dir
        self.save_checkpoints = save_checkpoints
        self.adata = None
        self.run_manager: Optional[RunManager] = None
        self._pending_image: Optional[Dict[str, str]] = None  # For vision support
        self._conversation_history: List[Dict[str, Any]] = []  # For interactive mode

        if provider == "anthropic":
            self._init_anthropic(api_key, model)
        elif provider == "openai":
            self._init_openai(api_key, model)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")

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

    def _init_openai(self, api_key: Optional[str], model: Optional[str]):
        """Initialize OpenAI client."""
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

        self.client = OpenAI(api_key=api_key)
        self.model = model or "gpt-4o"
        self.tools = get_openai_tools()

    def _print(self, message: str, style: str = None):
        """Print message if verbose using rich formatting."""
        if self.verbose:
            from rich.console import Console
            console = Console()
            if style:
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

        # Create run directory only for first analysis
        if self.create_run_dir and self.run_manager is None:
            self.run_manager = create_run(
                base_dir=self.output_dir,
                run_name=run_name,
                mode="agent"
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
        elif is_followup and self.run_manager:
            # Log follow-up request in existing manifest
            self.run_manager.log_step(
                tool="follow_up",
                parameters={"request": request},
                result={"status": "starting"}
            )

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

        # Route to provider-specific implementation
        if self.provider == "anthropic":
            return self._analyze_anthropic(user_message, max_iterations, continue_conversation)
        else:
            return self._analyze_openai(user_message, max_iterations, continue_conversation)

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
                    system=SYSTEM_PROMPT,
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        final_result = ""

        try:
            for iteration in range(max_iterations):
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

            if tool_name == "run_gsea":
                requested_dir = tool_input.get("output_dir")
                if not requested_dir:
                    tool_input["output_dir"] = str(self.run_manager.dirs["gsea"])
                else:
                    requested_path = Path(requested_dir)
                    if not requested_path.is_absolute():
                        if requested_path == Path(".") or requested_path.name == run_root.name:
                            tool_input["output_dir"] = str(self.run_manager.dirs["gsea"])

            if tool_name in checkpoint_tools and not self.save_checkpoints:
                requested = tool_input.get("output_path")
                if requested:
                    requested_path = Path(requested)
                    if requested_path.suffix.lower() != '.h5ad':
                        tool_input.pop("output_path", None)

            if self.save_checkpoints and tool_name in checkpoint_tools and not tool_input.get("output_path"):
                tool_input["output_path"] = self.run_manager.get_intermediate_path(_sanitize_name(tool_name))

        _prepare_tool_paths()
        if self.run_manager:
            self.run_manager.append_log(f"START {tool_name} {json.dumps(tool_input, default=str)}")

        # Special handling for ask_user - get input from user
        if tool_name == "ask_user":
            return self._handle_ask_user(tool_input)

        # Special handling for install_package - requires approval
        if tool_name == "install_package":
            return self._handle_install_package(tool_input)

        # Special handling for run_code - show what's being executed
        if tool_name == "run_code":
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
            result_json, self.adata = process_tool_call(tool_name, tool_input, self.adata)
        else:
            # Run with spinner for other tools
            if self.verbose:
                with console.status(f"[bold cyan]Running {tool_name}...", spinner="dots") as status:
                    result_json, self.adata = process_tool_call(tool_name, tool_input, self.adata)
                console.print(f"[green]✓[/green] {tool_name} complete")
            else:
                result_json, self.adata = process_tool_call(tool_name, tool_input, self.adata)

        # Check for image in result and store for vision
        try:
            result_data = json.loads(result_json)

            # If there's an image, store it for the next message
            if "image_base64" in result_data:
                self._pending_image = {
                    "base64": result_data["image_base64"],
                    "mime": result_data.get("image_mime", "image/png"),
                    "path": result_data.get("output_path", "figure.png"),
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
                self.run_manager.append_log(
                    f"END {tool_name} status={status} output={result_data.get('output_path', '')}"
                )

            if status == "ok":
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

        except json.JSONDecodeError:
            pass

        return result_json

    def _get_best_annotation_key(self) -> Optional[str]:
        """Return the most useful annotation column available on the current AnnData."""
        if self.adata is None:
            return None

        candidates = [
            "celltypist_majority_voting",
            "celltypist_predicted_labels",
            "scimilarity_representative_prediction",
            "scimilarity_predictions_unconstrained",
            "cell_type",
            "celltype",
        ]
        for key in candidates:
            if key in self.adata.obs.columns:
                return key
        return None

    def _cluster_annotation_summary(self, cluster_id: str, groupby: str, annotation_key: str) -> Optional[Dict[str, Any]]:
        """Return dominant-label summary for one annotation source within a cluster."""
        if self.adata is None or groupby not in self.adata.obs.columns or annotation_key not in self.adata.obs.columns:
            return None

        mask = self.adata.obs[groupby].astype(str) == str(cluster_id)
        labels = self.adata.obs.loc[mask, annotation_key].dropna().astype(str)
        if labels.empty:
            return None

        counts = labels.value_counts()
        return {
            "annotation_key": annotation_key,
            "label": str(counts.index[0]),
            "fraction": float(counts.iloc[0] / counts.sum()),
            "n_annotated": int(counts.sum()),
        }

    def _normalize_annotation_lineage(self, label: str) -> str:
        """Map detailed annotation labels to broad lineages for sanity checks."""
        lowered = (label or "").lower()
        if not lowered:
            return "unknown"
        if "plasmablast" in lowered or "plasma" in lowered:
            return "plasma"
        if "pdc" in lowered or "plasmacytoid" in lowered:
            return "pdc"
        if "dc" in lowered or "dendritic" in lowered:
            return "dendritic"
        if any(tok in lowered for tok in ["monocyte", "macroph", "myelo", "myelocyte"]):
            return "monocyte"
        if "nk" in lowered or "natural killer" in lowered:
            return "nk"
        if "b cell" in lowered or "b-cell" in lowered or lowered.startswith("b "):
            return "b_cell"
        if any(tok in lowered for tok in ["t cell", "helper t", "cytotoxic t", "regulatory t", "treg", "mait", "trm", "tem", "tcm"]):
            return "t_cell"
        return "unknown"

    def _annotation_agreement_summary(self, primary_label: str, secondary_label: Optional[str]) -> Dict[str, Any]:
        """Summarize whether annotation sources agree at fine or broad lineage level."""
        primary_lineage = self._normalize_annotation_lineage(primary_label)
        secondary_lineage = self._normalize_annotation_lineage(secondary_label or "")
        summary = {
            "status": "unknown",
            "note": "Only one annotation source available.",
            "primary_lineage": primary_lineage,
            "secondary_lineage": secondary_lineage,
        }
        if not secondary_label:
            return summary

        if primary_label.lower() == secondary_label.lower():
            summary.update(status="aligned", note="Annotation systems agree on the same label.")
            return summary

        if primary_lineage != "unknown" and primary_lineage == secondary_lineage:
            summary.update(status="broadly_aligned", note="Annotation systems agree on the broad lineage but not the fine-grained label.")
            return summary

        myeloid = {"monocyte", "dendritic", "pdc"}
        lymphoid = {"t_cell", "nk", "b_cell", "plasma"}
        if primary_lineage in myeloid and secondary_lineage in myeloid:
            summary.update(status="myeloid_disagreement", note="Annotation systems agree this cluster is myeloid, but disagree on the finer identity.")
        elif primary_lineage in lymphoid and secondary_lineage in lymphoid:
            summary.update(status="lymphoid_disagreement", note="Annotation systems agree this cluster is lymphoid, but disagree on the finer identity.")
        else:
            summary.update(status="conflicting", note="Annotation systems disagree on the broad lineage assignment for this cluster.")
        return summary

    def _get_cluster_top_markers(self, cluster_id: str, groupby: str, n_genes: int = 10) -> List[str]:
        """Return top marker genes for a cluster from the current DEG result if available."""
        if self.adata is None or "rank_genes_groups" not in self.adata.uns:
            return []
        params = self.adata.uns["rank_genes_groups"].get("params", {})
        if params.get("groupby") != groupby:
            return []

        try:
            import scanpy as sc
            markers_df = sc.get.rank_genes_groups_df(self.adata, group=str(cluster_id)).head(n_genes)
        except Exception:
            return []

        return [str(name) for name in markers_df["names"].tolist()]

    def _expected_marker_panel(self, label: str) -> Dict[str, Any]:
        """Return a coarse canonical marker panel for the inferred label."""
        lineage = self._normalize_annotation_lineage(label)
        lowered = (label or "").lower()
        panels = {
            "cytotoxic_t": {"CCL5", "NKG7", "CTSW", "CST7", "GZMA", "PRF1", "GNLY", "CD3D"},
            "treg": {"IL32", "TIGIT", "LTB", "IL7R", "CTLA4", "FOXP3", "LTB"},
            "t_cell": {"CD3D", "CD3E", "TRAC", "LTB", "IL7R", "MALAT1"},
            "nk": {"GNLY", "NKG7", "KLRD1", "PRF1", "KLRF1", "CST7", "CTSW", "TYROBP"},
            "b_cell": {"MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "HLA-DRA", "CD37"},
            "plasma": {"JCHAIN", "MZB1", "SDC1", "IGHG1", "IGKC", "XBP1"},
            "monocyte": {"LYZ", "S100A8", "S100A9", "FCN1", "CTSS", "SAT1", "LST1", "VCAN", "MNDA", "FCER1G"},
            "dendritic": {"HLA-DRA", "HLA-DPA1", "HLA-DPB1", "CD74", "FCER1A", "CLEC10A", "CST3", "HLA-DQA1"},
            "pdc": {"IL3RA", "GZMB", "JCHAIN", "TCF4", "IRF7", "IFITM1"},
        }

        if any(tok in lowered for tok in ["regulatory t", "treg"]):
            panel_key = "treg"
        elif lineage == "t_cell" and any(tok in lowered for tok in ["cytotoxic", "trm", "tem"]):
            panel_key = "cytotoxic_t"
        elif lineage in panels:
            panel_key = lineage
        else:
            panel_key = "t_cell" if lineage == "t_cell" else "unknown"

        return {
            "panel_key": panel_key,
            "lineage": lineage,
            "markers": panels.get(panel_key, set()),
        }

    def _marker_support_summary(self, label: str, markers: List[str]) -> Dict[str, Any]:
        """Assess whether cluster markers support the inferred broad lineage."""
        panel = self._expected_marker_panel(label)
        expected = {marker.upper() for marker in panel["markers"]}
        observed = [marker for marker in markers if marker.upper() in expected]
        n_matched = len(observed)

        if not markers:
            status = "unknown"
        elif n_matched >= 3:
            status = "strong"
        elif n_matched == 2:
            status = "moderate"
        elif n_matched == 1:
            status = "weak"
        else:
            status = "absent"

        return {
            "status": status,
            "lineage": panel["lineage"],
            "panel_key": panel["panel_key"],
            "matched_markers": observed[:5],
        }

    def _infer_cluster_context(self, cluster_id: str, groupby: str) -> Dict[str, Any]:
        """Infer a cluster's likely context, annotation agreement, and marker support."""
        context = {
            "cluster": str(cluster_id),
            "groupby": groupby,
            "annotation_key": None,
            "cell_type": f"cluster {cluster_id}",
            "cell_type_fraction": None,
            "secondary_annotation_key": None,
            "secondary_cell_type": None,
            "secondary_cell_type_fraction": None,
            "annotation_agreement": "unknown",
            "annotation_agreement_note": None,
            "marker_support": "unknown",
            "marker_lineage": "unknown",
            "marker_support_markers": [],
            "top_markers": [],
            "interpretation_cautions": [],
            "n_cells": None,
        }

        if self.adata is None or groupby not in self.adata.obs.columns:
            return context

        mask = self.adata.obs[groupby].astype(str) == str(cluster_id)
        n_cells = int(mask.sum())
        context["n_cells"] = n_cells
        if n_cells == 0:
            return context

        primary_key = self._get_best_annotation_key()
        context["annotation_key"] = primary_key
        primary_summary = self._cluster_annotation_summary(cluster_id, groupby, primary_key) if primary_key else None
        if primary_summary:
            context["cell_type"] = primary_summary["label"]
            context["cell_type_fraction"] = primary_summary["fraction"]

        secondary_summary = None
        for secondary_key in [
            "scimilarity_representative_prediction",
            "celltypist_majority_voting",
            "celltypist_predicted_labels",
        ]:
            if secondary_key == primary_key:
                continue
            secondary_summary = self._cluster_annotation_summary(cluster_id, groupby, secondary_key)
            if secondary_summary:
                break

        if secondary_summary:
            context["secondary_annotation_key"] = secondary_summary["annotation_key"]
            context["secondary_cell_type"] = secondary_summary["label"]
            context["secondary_cell_type_fraction"] = secondary_summary["fraction"]

        agreement = self._annotation_agreement_summary(
            context["cell_type"],
            context.get("secondary_cell_type"),
        )
        context["annotation_agreement"] = agreement["status"]
        context["annotation_agreement_note"] = agreement["note"]

        markers = self._get_cluster_top_markers(cluster_id, groupby, n_genes=10)
        context["top_markers"] = markers[:5]
        marker_support = self._marker_support_summary(context["cell_type"], markers)
        context["marker_support"] = marker_support["status"]
        context["marker_lineage"] = marker_support["lineage"]
        context["marker_support_markers"] = marker_support["matched_markers"]

        cautions = []
        if context.get("cell_type_fraction") is not None and context["cell_type_fraction"] < 0.75:
            cautions.append("Dominant annotation covers less than 75% of the cluster.")
        if agreement["status"] in {"myeloid_disagreement", "lymphoid_disagreement", "conflicting"}:
            cautions.append(agreement["note"])
        if marker_support["status"] in {"weak", "absent"}:
            if marker_support["status"] == "weak":
                cautions.append("Top markers only weakly support the inferred lineage.")
            else:
                cautions.append("Top markers do not clearly support the inferred lineage.")
        context["interpretation_cautions"] = cautions
        return context

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

    def _pathway_function_hint(self, pathway_term: str) -> Optional[str]:
        """Return a lightweight biological interpretation hint for common pathway families."""
        term = (pathway_term or "").lower()
        hints = [
            (["allograft", "rejection"], "broad immune activation, antigen presentation, and cytotoxic lymphocyte programs rather than literal transplant biology"),
            (["p53"], "cell stress, DNA damage response, and apoptosis-related programs"),
            (["il2", "stat5"], "cytokine signaling and immune activation programs"),
            (["tnf", "nf-kb"], "inflammatory signaling and innate immune activation"),
            (["tnf", "nfkb"], "inflammatory signaling and innate immune activation"),
            (["interferon", "gamma"], "interferon-driven inflammatory and antigen-presentation responses"),
            (["interferon"], "interferon-driven antiviral and inflammatory responses"),
            (["oxidative", "phosphorylation"], "mitochondrial respiration and energy metabolism"),
            (["glycolysis"], "glycolytic metabolism and rapid energy-demand programs"),
            (["hypoxia"], "cellular hypoxia and stress adaptation"),
            (["mtor"], "growth, nutrient sensing, and anabolic signaling"),
            (["e2f"], "cell-cycle progression and proliferation"),
            (["apoptosis"], "programmed cell death and survival control"),
            (["apical", "junction"], "cell adhesion, cytoskeletal remodeling, and tissue-interaction programs"),
            (["kras"], "RAS/MAPK-linked activation and signaling programs"),
            (["coagulation"], "coagulation-linked inflammatory and immunothrombotic programs"),
        ]
        for keywords, hint in hints:
            if all(keyword in term for keyword in keywords):
                return hint
        return None

    def _summarize_pathway_interpretation(
        self,
        pathway: Dict[str, Any],
        cluster_context: Dict[str, Any],
        research_data: Dict[str, Any],
        statistically_significant: bool,
    ) -> str:
        """Create a concise interpretation sentence for a pathway."""
        direction = pathway.get("direction", "unknown")
        term = pathway.get("term", "pathway")
        cell_type = cluster_context.get("cell_type", "this cluster")
        hint = self._pathway_function_hint(term)
        fdr = pathway.get("fdr")
        papers = research_data.get("findings", {}).get("selected_papers") or research_data.get("findings", {}).get("pubmed_results", [])

        direction_phrase = {
            "upregulated": "relative enrichment of",
            "downregulated": "relative depletion of",
        }.get(direction, "altered activity of")

        pieces = [f"This result suggests {direction_phrase} `{term}` in `{cell_type}`."]
        if hint:
            pieces.append(f"`{term}` is commonly associated with {hint}.")
        if pathway.get("genes"):
            pieces.append(f"Leading-edge genes include {', '.join(pathway['genes'][:3])}.")
        if statistically_significant:
            pieces.append(f"The statistical signal is stronger here (FDR {fdr}).")
        else:
            pieces.append(f"This pathway did not pass the default significance cutoff (FDR {fdr}), so treat it as exploratory.")

        agreement_status = cluster_context.get("annotation_agreement")
        if agreement_status in {"myeloid_disagreement", "lymphoid_disagreement", "conflicting"}:
            pieces.append("Cluster identity is not fully settled across annotation sources, so lineage-specific claims should be treated cautiously.")
        if cluster_context.get("marker_support") in {"weak", "absent"}:
            pieces.append("Marker support for the inferred lineage is limited, so this interpretation should be considered provisional.")

        if papers:
            top_paper = papers[0]
            reasons = top_paper.get("match_reasons", [])
            if reasons:
                pieces.append(f"Top literature match was selected due to {'; '.join(reasons[:2])}.")
        else:
            pieces.append("Literature support was limited for this exact pathway/cell-type combination.")
        return " ".join(pieces)

    def _generate_gsea_evidence_reports(self, gsea_result: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate markdown and JSON evidence reports for GSEA results."""
        if not self.run_manager:
            return None

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
                research_json, _ = process_tool_call(
                    "research_findings",
                    {
                        "pathway": pathway["term"],
                        "cell_type": cluster_context["cell_type"],
                        "genes": pathway.get("genes", []),
                        "context": self.run_manager.manifest.request or "",
                        "recent_years": 3,
                    },
                    self.adata,
                )
                research_data = json.loads(research_json)
                papers = research_data.get("findings", {}).get("selected_papers", [])
                reviews = research_data.get("findings", {}).get("review_articles", [])
                fdr_value = pathway.get("fdr")
                statistically_significant = (fdr_value if fdr_value is not None else 1.0) < 0.25
                interpretation = self._summarize_pathway_interpretation(
                    pathway,
                    cluster_context,
                    research_data,
                    statistically_significant=statistically_significant,
                )

                pathway_entry = {
                    "term": pathway["term"],
                    "direction": pathway.get("direction"),
                    "nes": pathway.get("nes"),
                    "fdr": pathway.get("fdr"),
                    "genes": pathway.get("genes", []),
                    "research": research_data,
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
                md_lines.append("")
                md_lines.append(f"Interpretation: {interpretation}")
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
        default = tool_input.get("default", "")

        # Format the question
        print(f"\n{'='*50}")
        print(f"AGENT QUESTION: {question}")
        if options:
            print(f"Options: {', '.join(options)}")
        if default:
            print(f"Default: {default}")
        print('='*50)

        # Get user input
        try:
            response = input("Your answer: ").strip()
            if not response and default:
                response = default
        except (EOFError, KeyboardInterrupt):
            response = default or "no response"

        return json.dumps({
            "status": "ok",
            "tool": "ask_user",
            "question": question,
            "user_response": response,
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
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}],
            )
            for content in response.content:
                if hasattr(content, "text"):
                    return content.text
            return ""
        else:
            # OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
            )
            return response.choices[0].message.content or ""

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
            self.adata
        )
        return json.loads(result_json)

    def reset_conversation(self):
        """
        Clear conversation history to start a fresh conversation.

        Useful when switching to a completely different analysis topic.
        Note: This does NOT unload the data - call reset() for that.
        """
        self._conversation_history = []
        self._print("Conversation history cleared.")

    def reset(self):
        """
        Reset agent state completely.

        Clears conversation history, loaded data, and run manager.
        """
        self._conversation_history = []
        self.adata = None
        self.run_manager = None
        self._pending_image = None
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
