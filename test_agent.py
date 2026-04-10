#!/usr/bin/env python
"""
Quick test of the SCAgent with either provider.

Usage:
    # Test with Anthropic (default)
    export ANTHROPIC_API_KEY=your_key
    python test_agent.py

    # Test with OpenAI
    export OPENAI_API_KEY=your_key
    python test_agent.py --provider openai
"""

import argparse
import sys

# Test data path
PBMC_PATH = "/data1/peerd/sharmar1/workshop_2025_files/workshop_data/pbmc_data/pbmc_10k_v3_filtered_feature_bc_matrix.h5"
SECOND_PATH = "/data1/peerd/ibrahih3/cs_agent/test_data/GSE155249_main.h5ad.gz"


def test_inspect(provider: str):
    """Test simple inspection (no LLM call, just tools)."""
    from scagent.agent.tools import process_tool_call

    print(f"Testing inspect_data tool on PBMC data...")
    result_json, adata = process_tool_call(
        "inspect_data",
        {"data_path": PBMC_PATH},
        None
    )

    import json
    result = json.loads(result_json)
    print(f"Status: {result['status']}")
    print(f"Shape: {result['shape']}")
    print(f"Data type: {result['data_type']}")
    print(f"Gene format: {result.get('genes', {})}")
    print(f"State: {result['state']}")
    return result


def test_agent(provider: str, simple: bool = True):
    """Test the agent with a simple request."""
    from scagent.agent import SCAgent

    print(f"\nTesting SCAgent with provider: {provider}")
    print("=" * 50)

    try:
        agent = SCAgent(
            provider=provider,
            verbose=True,
            create_run_dir=True,
            output_dir="test_output",
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nSet {'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'} and retry.")
        return None

    if simple:
        # Just inspect - no heavy processing
        request = "Inspect this data and tell me what processing steps are needed to cluster it."
    else:
        # Full QC pipeline
        request = "Run QC on this PBMC data and identify the major cell populations."

    result = agent.analyze(
        request=request,
        data_path=PBMC_PATH,
        run_name="test",
        max_iterations=5,
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SCAgent")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--full", action="store_true", help="Run full analysis (not just inspect)")
    parser.add_argument("--tools-only", action="store_true", help="Test tools without LLM")
    args = parser.parse_args()

    if args.tools_only:
        test_inspect(args.provider)
    else:
        test_agent(args.provider, simple=not args.full)
