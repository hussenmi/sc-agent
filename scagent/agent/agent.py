"""
Main agent class for autonomous single-cell analysis.

Uses Claude API with tools to perform single-cell analysis tasks.
Returns structured JSON from tools for reliable LLM reasoning.
Creates run directories with manifests for reproducibility.
"""

import os
import json
from typing import Optional, Dict, Any, List
import logging

from .tools import get_tools, process_tool_call
from .prompts import SYSTEM_PROMPT
from .run_manager import RunManager, create_run

logger = logging.getLogger(__name__)


class SCAgent:
    """
    Autonomous single-cell RNA-seq analysis agent.

    Uses Claude API to analyze single-cell data following lab best practices.
    All tool calls return structured JSON for reliable LLM reasoning.
    Optionally creates run directories with manifests for reproducibility.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
    model : str, default "claude-sonnet-4-20250514"
        Claude model to use.
    verbose : bool, default True
        Print agent outputs.
    create_run_dir : bool, default True
        Create structured run directory with manifest.
    output_dir : str, default "."
        Base directory for run outputs.

    Examples
    --------
    >>> agent = SCAgent()
    >>> result = agent.analyze(
    ...     "Perform QC and cluster this PBMC data",
    ...     data_path="pbmc_10k.h5"
    ... )
    >>> print(result)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
        create_run_dir: bool = True,
        output_dir: str = ".",
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic not installed. Install with: pip install anthropic"
            )

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.verbose = verbose
        self.create_run_dir = create_run_dir
        self.output_dir = output_dir
        self.tools = get_tools()
        self.adata = None
        self.run_manager: Optional[RunManager] = None

    def _print(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

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
            self.run_manager.set_model(self.model)

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

        messages = [{"role": "user", "content": user_message}]

        self._print(f"\nAnalyzing: {request}")
        self._print("-" * 50)

        final_result = ""

        try:
            for iteration in range(max_iterations):
                # Call Claude
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=self.tools,
                    messages=messages,
                )

                # Process response
                if response.stop_reason == "tool_use":
                    # Extract tool calls
                    tool_results = []
                    assistant_content = []

                    for content in response.content:
                        if content.type == "text":
                            assistant_content.append(content)
                            if self.verbose:
                                self._print(f"\n{content.text}")

                        elif content.type == "tool_use":
                            assistant_content.append(content)
                            tool_name = content.name
                            tool_input = content.input
                            tool_id = content.id

                            self._print(f"\n[Tool] {tool_name}")
                            logger.info(f"Tool call: {tool_name} with {tool_input}")

                            # Process the tool call
                            result_json, self.adata = process_tool_call(
                                tool_name, tool_input, self.adata
                            )

                            # Parse result for logging
                            try:
                                result_data = json.loads(result_json)
                                status = result_data.get("status", "unknown")

                                # Log to run manager
                                if self.run_manager:
                                    self.run_manager.log_step(
                                        tool=tool_name,
                                        input_path=tool_input.get("data_path"),
                                        output_path=tool_input.get("output_path"),
                                        parameters=tool_input,
                                        result=result_data,
                                    )

                                    # Track warnings
                                    for w in result_data.get("warnings", []):
                                        self.run_manager.add_warning(w)

                                # Print compact summary
                                if status == "ok":
                                    if "after" in result_data:
                                        self._print(f"    → {result_data['after'].get('n_cells', '?')} cells")
                                    if "n_clusters" in result_data:
                                        self._print(f"    → {result_data['n_clusters']} clusters")
                                else:
                                    self._print(f"    → {status}: {result_data.get('message', '')}")

                            except json.JSONDecodeError:
                                pass

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_json,
                            })

                    # Add assistant message and tool results
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({"role": "user", "content": tool_results})

                elif response.stop_reason == "end_turn":
                    # Extract final text response
                    for content in response.content:
                        if hasattr(content, "text"):
                            final_result = content.text
                            self._print("\n" + "-" * 50)
                            self._print(final_result)

                    # Complete run
                    if self.run_manager:
                        summary_path = self.run_manager.complete(summary=final_result)
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
