#!/usr/bin/env python
"""
Command-line interface for scagent.

Usage:
    scagent analyze "your request" --data path/to/data.h5
    scagent analyze --data path/to/data.h5  # Auto-analyze
    scagent inspect path/to/data.h5ad
    scagent chat "question about single-cell analysis"
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="scagent - Autonomous single-cell RNA-seq analysis agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with custom request
  scagent analyze "QC, cluster, and annotate cell types" --data pbmc.h5

  # Auto-analyze (agent decides what to do)
  scagent analyze --data pbmc.h5

  # Run once and exit (skip follow-up prompt)
  scagent analyze --data pbmc.h5 --single-run

  # Just inspect data state
  scagent inspect clustered_data.h5ad

  # Ask a question
  scagent chat "What's the best way to handle batch effects?"

  # Use specific provider/model
  scagent analyze --data pbmc.h5 --provider openai --model gpt-4o
        """
    )

    parser.add_argument("--version", action="store_true", help="Show version")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # === analyze command ===
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run autonomous analysis on single-cell data"
    )
    analyze_parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help="What to analyze (e.g., 'QC and cluster'). If omitted, agent auto-analyzes."
    )
    analyze_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to input data file (h5ad or 10X h5)"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory (default: current directory)"
    )
    analyze_parser.add_argument(
        "--name", "-n",
        default=None,
        help="Run name for output directory"
    )
    analyze_parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider (default: from .env or anthropic)"
    )
    analyze_parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (default: provider default)"
    )
    analyze_parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max tool call iterations (default: 20)"
    )
    analyze_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output"
    )
    analyze_mode = analyze_parser.add_mutually_exclusive_group()
    analyze_mode.add_argument(
        "--interactive", "-i",
        dest="interactive",
        action="store_true",
        help="Keep the conversation open for follow-up requests after the initial run (default)"
    )
    analyze_mode.add_argument(
        "--single-run",
        dest="interactive",
        action="store_false",
        help="Exit after the initial analysis summary"
    )
    analyze_parser.set_defaults(interactive=True)
    collaboration_mode = analyze_parser.add_mutually_exclusive_group()
    collaboration_mode.add_argument(
        "--collaborative",
        dest="collaborative",
        action="store_true",
        help="Pause at major analysis checkpoints to summarize findings and ask before moving on (default)"
    )
    collaboration_mode.add_argument(
        "--autonomous",
        dest="collaborative",
        action="store_false",
        help="Let the agent run end-to-end without checkpoint prompts unless recovery is needed"
    )
    analyze_parser.set_defaults(collaborative=True)
    analyze_parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Save intermediate h5ad files (default: only save final)"
    )

    # === inspect command ===
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect data state without running analysis"
    )
    inspect_parser.add_argument(
        "data",
        help="Path to h5ad file"
    )
    inspect_parser.add_argument(
        "--goal", "-g",
        choices=["qc", "cluster", "annotate", "deg", "batch_correct"],
        default=None,
        help="Get recommendations for a specific goal"
    )

    # === qc command (direct, no agent) ===
    qc_parser = subparsers.add_parser(
        "qc",
        help="Run QC pipeline directly (no agent)"
    )
    qc_parser.add_argument("data_path", help="Input data path")
    qc_parser.add_argument("output_path", help="Output h5ad path")
    qc_parser.add_argument("--mt-threshold", type=float, help="MT percentage threshold")

    # === chat command ===
    chat_parser = subparsers.add_parser(
        "chat",
        help="Ask a question (no data analysis)"
    )
    chat_parser.add_argument(
        "question",
        help="Question to ask"
    )
    chat_parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider"
    )

    args = parser.parse_args()

    if args.version:
        from scagent import __version__
        print(f"scagent {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    # Handle commands
    if args.command == "analyze":
        return run_analyze(args)
    elif args.command == "inspect":
        return run_inspect(args)
    elif args.command == "qc":
        return run_qc(args)
    elif args.command == "chat":
        return run_chat(args)

    return 0


def run_analyze(args):
    """Run autonomous analysis."""
    from scagent.agent import SCAgent

    # Build request
    if args.request:
        request = args.request
    else:
        # Auto-analyze: let agent decide
        request = (
            "Analyze this single-cell data. First inspect it to understand what processing "
            "has been done. Then run appropriate analysis steps based on the data state and "
            "what would be most useful. Include QC if needed, clustering, and cell type "
            "annotation. Provide a summary of your findings."
        )

    # Create agent
    agent = SCAgent(
        provider=args.provider,
        model=args.model,
        verbose=not args.quiet,
        collaborative=args.collaborative,
        output_dir=args.output,
        save_checkpoints=args.checkpoints,
    )

    print(f"Data: {args.data}")
    print(f"Provider: {agent.provider}:{agent.model}")
    print(f"Session mode: {'interactive follow-up' if args.interactive else 'single-run'}")
    print(f"Analysis style: {'collaborative checkpoints' if args.collaborative else 'autonomous run'}")
    print(f"Checkpoints: {'enabled' if args.checkpoints else 'final only'}")
    print(f"Request: {request[:100]}{'...' if len(request) > 100 else ''}")
    print("-" * 50)

    # First analysis
    result = agent.analyze(
        request=request,
        data_path=args.data,
        run_name=args.name,
        max_iterations=args.max_iterations,
    )

    # Interactive mode - continue conversation
    if args.interactive:
        if not sys.stdin.isatty():
            print("\nInteractive mode requested, but stdin is not a TTY. Exiting after the initial run.")
            return 0

        # Simple prompt - no banner that implies "finished"
        while True:
            try:
                # Just show a prompt - the conversation is ongoing
                user_input = input("\n> ").strip()

                if user_input.lower() in ['done', 'exit', 'quit', 'q']:
                    print("Exiting interactive mode.")
                    break

                if not user_input:
                    continue

                # Continue analysis with the same agent (preserves state and conversation)
                result = agent.analyze(
                    request=user_input,
                    data_path=None,  # Use existing loaded data
                    max_iterations=args.max_iterations,
                    continue_conversation=True,  # Keep conversation history
                )

            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive mode.")
                break

    return 0


def run_inspect(args):
    """Inspect data state."""
    from scagent.core import load_data, inspect_data, recommend_next_steps
    from scagent.core.inspector import summarize_state

    print(f"Inspecting: {args.data}")
    print("-" * 50)

    adata = load_data(args.data)
    state = inspect_data(adata)

    print(summarize_state(state))

    if args.goal:
        print(f"\nRecommended steps for '{args.goal}':")
        steps = recommend_next_steps(state, args.goal)
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

    return 0


def run_qc(args):
    """Run QC directly without agent."""
    from scagent.core import load_data, run_qc_pipeline

    print(f"Running QC on: {args.data_path}")

    adata = load_data(args.data_path)
    n_before = adata.n_obs

    run_qc_pipeline(adata, mt_threshold=args.mt_threshold)

    adata.write_h5ad(args.output_path)
    print(f"Filtered: {n_before} -> {adata.n_obs} cells")
    print(f"Saved to: {args.output_path}")

    return 0


def run_chat(args):
    """Chat with the agent."""
    from scagent.agent import SCAgent

    agent = SCAgent(
        provider=args.provider,
        create_run_dir=False,
        verbose=False,
    )

    response = agent.chat(args.question)
    print(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())
