"""
Command-line interface for scagent.

Usage:
    scagent inspect <data_path>
    scagent qc <data_path> <output_path>
    scagent analyze <request> --data <data_path>
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="scagent: Single-cell RNA-seq Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect data state")
    inspect_parser.add_argument("data_path", help="Path to h5ad or 10X h5 file")

    # QC command
    qc_parser = subparsers.add_parser("qc", help="Run QC pipeline")
    qc_parser.add_argument("data_path", help="Input data path")
    qc_parser.add_argument("output_path", help="Output h5ad path")
    qc_parser.add_argument("--mt-threshold", type=float, help="MT percentage threshold")

    # Version
    parser.add_argument("--version", action="store_true", help="Show version")

    args = parser.parse_args()

    if args.version:
        from scagent import __version__
        print(f"scagent {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "inspect":
        from scagent.core import load_data, inspect_data
        from scagent.core.inspector import summarize_state

        adata = load_data(args.data_path)
        state = inspect_data(adata)
        print(summarize_state(state))
        return 0

    if args.command == "qc":
        from scagent.core import load_data, run_qc_pipeline

        adata = load_data(args.data_path)
        run_qc_pipeline(adata, mt_threshold=args.mt_threshold)
        adata.write_h5ad(args.output_path)
        print(f"Saved to {args.output_path}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
