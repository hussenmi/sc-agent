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
        self.adata = None
        self.run_manager: Optional[RunManager] = None
        self._pending_image: Optional[Dict[str, str]] = None  # For vision support

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

    def _print(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message, flush=True)

    def analyze(
        self,
        request: str,
        data_path: Optional[str] = None,
        run_name: Optional[str] = None,
        max_iterations: int = 20,
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
        run_name : str, optional
            Name for the run directory.
        max_iterations : int, default 20
            Maximum number of tool calls.

        Returns
        -------
        str
            Summary of the analysis performed.
        """
        # Create run directory
        if self.create_run_dir:
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

            self._print(f"Run directory: {self.run_manager.run_dir}")

            if data_path:
                self.run_manager.add_input(data_path)

        # Build initial message
        user_message = request
        if data_path:
            user_message += f"\n\nData file: {data_path}"
        if self.run_manager:
            user_message += f"\nOutput directory: {self.run_manager.run_dir}"

        self._print(f"\nAnalyzing: {request}")
        self._print("-" * 50)

        # Route to provider-specific implementation
        if self.provider == "anthropic":
            return self._analyze_anthropic(user_message, max_iterations)
        else:
            return self._analyze_openai(user_message, max_iterations)

    def _analyze_anthropic(self, user_message: str, max_iterations: int) -> str:
        """Run analysis loop using Anthropic API."""
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
                                self._print(f"\n{content.text}")

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
                    for content in response.content:
                        if hasattr(content, "text"):
                            final_result = content.text
                            self._print("\n" + "-" * 50)
                            self._print(final_result)

                    if self.run_manager:
                        self.run_manager.complete(summary=final_result)
                        self._print(f"\nRun manifest: {self.run_manager.run_dir}/manifest.json")

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

    def _analyze_openai(self, user_message: str, max_iterations: int) -> str:
        """Run analysis loop using OpenAI API."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        final_result = ""

        try:
            for iteration in range(max_iterations):
                self._print(f"\n[Iteration {iteration+1}/{max_iterations}]")
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=4096,
                    tools=self.tools,
                    messages=messages,
                )
                self._print(f"[finish_reason: {response.choices[0].finish_reason}]")

                choice = response.choices[0]
                message = choice.message

                if choice.finish_reason == "tool_calls" and message.tool_calls:
                    # Add assistant message with tool calls
                    messages.append(message)

                    # Print any text content
                    if message.content:
                        self._print(f"\n{message.content}")

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
                    final_result = message.content or ""
                    self._print("\n" + "-" * 50)
                    self._print(final_result)

                    if self.run_manager:
                        self.run_manager.complete(summary=final_result)
                        self._print(f"\nRun manifest: {self.run_manager.run_dir}/manifest.json")

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
        self._print(f"\n[Tool] {tool_name}")
        logger.info(f"Tool call: {tool_name} with {tool_input}")

        # Special handling for ask_user - get input from user
        if tool_name == "ask_user":
            return self._handle_ask_user(tool_input)

        # Special handling for install_package - requires approval
        if tool_name == "install_package":
            return self._handle_install_package(tool_input)

        # Special handling for run_code - show what's being executed
        if tool_name == "run_code":
            self._print(f"    Description: {tool_input.get('description', 'custom code')}")
            if self.verbose:
                code_preview = tool_input.get('code', '')[:100]
                if len(tool_input.get('code', '')) > 100:
                    code_preview += "..."
                self._print(f"    Code: {code_preview}")
            # Pass output directory for code file saving
            if self.run_manager:
                tool_input["output_dir"] = str(self.run_manager.run_dir)

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

            if self.run_manager:
                self.run_manager.log_step(
                    tool=tool_name,
                    input_path=tool_input.get("data_path"),
                    output_path=tool_input.get("output_path"),
                    parameters=tool_input,
                    result=result_data,
                )
                for w in result_data.get("warnings", []):
                    self.run_manager.add_warning(w)

            if status == "ok":
                if "after" in result_data:
                    self._print(f"    → {result_data['after'].get('n_cells', '?')} cells")
                if "n_clusters" in result_data:
                    self._print(f"    → {result_data['n_clusters']} clusters")
                if "shape" in result_data and tool_name == "run_code":
                    self._print(f"    → {result_data['shape']['n_cells']} cells, {result_data['shape']['n_genes']} genes")
            elif status == "needs_input":
                pass  # Handled by ask_user
            else:
                self._print(f"    → {status}: {result_data.get('message', '')}")

        except json.JSONDecodeError:
            pass

        return result_json

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
            # Install the package
            try:
                result = subprocess.run(
                    ["pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    return json.dumps({
                        "status": "ok",
                        "tool": "install_package",
                        "package": package,
                        "message": f"Successfully installed {package}"
                    }, indent=2)
                else:
                    return json.dumps({
                        "status": "error",
                        "tool": "install_package",
                        "package": package,
                        "message": f"Installation failed: {result.stderr}"
                    }, indent=2)
            except subprocess.TimeoutExpired:
                return json.dumps({
                    "status": "error",
                    "tool": "install_package",
                    "message": "Installation timed out"
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
