#!/usr/bin/env python3
"""Tool-call reliability benchmark for OpenAI-compatible LLM endpoints.

The throughput benches (bench_llm.py, bench_prefix_cache.py) measure how *fast*
a server is. This measures whether it is *usable as an agent backend at all*:
given scagent's real ~50 tool schemas, does the server emit a syntactically
valid tool call that maps to the user's intent?

For an agent, this is the make-or-break axis. A server that is 2x faster but
mangles tool calls (wrong JSON, hallucinated tool names, or answering in prose
instead of calling a tool) is a net regression. This is exactly the place where
swapping vLLM (with a hand-picked --tool-call-parser) for NIM/TensorRT-LLM can
silently break — so we measure it head-to-head, same as the speed benches.

What it does:
  - Loads scagent's actual tool schemas via get_openai_tools() (51 tools).
  - Sends a set of unambiguous single-step requests with tools=... and
    tool_choice="auto" (non-streaming, so tool_calls parse cleanly).
  - For each response, classifies the outcome:
      ok            -> emitted a real tool with parseable JSON arguments
      wrong_tool    -> valid tool call, but not the intent-matching tool
      bad_json      -> tool call whose arguments don't parse as JSON
      hallucinated  -> tool name not in scagent's tool set
      no_call       -> answered in prose, emitted no tool call
      error         -> request failed
  - Reports valid-call rate (ok+wrong_tool: did tool-calling work at all) and
    intent-match rate (ok: did it pick the right tool), per endpoint.

Usage:
  # single endpoint
  python bench_toolcall.py --url http://host:8000/v1 --model Qwen2.5-Coder-32B-Instruct --label vllm

  # head-to-head (this is the point): vLLM vs NIM, same model, same GPU
  python bench_toolcall.py \\
    --pair http://host:8000/v1,Qwen2.5-Coder-32B-Instruct,vllm-trtllm-off \\
    --pair http://host:8001/v1,qwen2.5-coder-32b-instruct,nim-trtllm \\
    --repeats 3 --out results/toolcall.csv

Notes:
  - Run after `source setup.sh` so scagent imports resolve.
  - tool_choice defaults to "auto" (what scagent uses). Pass --force to send
    tool_choice="required" and isolate "can it emit valid tool JSON at all"
    from "does it decide to use a tool".
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import requests

# scagent's real tool schemas — the whole point is to test against these, not toys.
try:
    from scagent.agent.tools import get_openai_tools
except Exception as e:  # pragma: no cover - import-time guard
    print(f"ERROR: could not import scagent tool schemas ({e}).\n"
          f"Run `source setup.sh` first so the venv + package are active.", file=sys.stderr)
    sys.exit(1)

TOOLS = get_openai_tools()
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

SYSTEM = (
    "You are scagent, a single-cell RNA-seq analysis assistant. You operate by "
    "calling the provided tools. For each user request, call the single most "
    "appropriate tool with sensible arguments. Do not answer in prose when a tool fits."
)

# (user request, expected tool). Each request is deliberately unambiguous so a
# capable model+parser should hit the expected tool. Expected names are checked
# against the live tool set at startup, so a renamed tool surfaces loudly.
SCENARIOS = [
    ("Load the dataset at /data/pbmc.h5ad.", "load_data"),
    ("Run quality control: minimum 200 genes per cell and at most 20 percent mitochondrial counts.", "run_qc"),
    ("Normalize the counts and select 3000 highly variable genes.", "normalize_and_hvg"),
    ("Run PCA with 50 components.", "run_pca"),
    ("Compute the neighbors graph using 15 neighbors over 30 PCs.", "run_neighbors"),
    ("Generate a UMAP embedding.", "run_umap"),
    ("Cluster the cells with Leiden at resolution 0.5.", "run_clustering"),
    ("Run differential expression across the leiden clusters using the wilcoxon method.", "run_deg"),
    ("Annotate the clusters with CellTypist using the Immune_All_High model.", "run_celltypist"),
    ("Run Harmony batch correction on the 'sample' column.", "run_batch_correction"),
    ("Make a UMAP plot colored by the leiden clustering.", "generate_figure"),
    ("Save the current dataset to analysis.h5ad.", "save_data"),
    ("Show me the current state of the dataset.", "inspect_data"),
    ("What are the top marker genes for each cluster?", "get_top_markers"),
    ("Search recent papers on microglia marker genes.", "search_papers"),
]


def classify(message, expected):
    """Map one assistant response to an outcome label (see module docstring)."""
    tool_calls = (message or {}).get("tool_calls") or []
    if not tool_calls:
        return "no_call", None
    call = tool_calls[0]
    fn = call.get("function") or {}
    name = fn.get("name")
    if name not in TOOL_NAMES:
        return "hallucinated", name
    args = fn.get("arguments")
    # OpenAI spec: arguments is a JSON *string*. Some servers emit a dict directly.
    if isinstance(args, str):
        try:
            json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, ValueError):
            return "bad_json", name
    elif not isinstance(args, (dict, type(None))):
        return "bad_json", name
    return ("ok" if name == expected else "wrong_tool"), name


def one_request(base_url, model, user_msg, force):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user_msg}],
        "tools": TOOLS,
        "tool_choice": "required" if force else "auto",
        "max_tokens": 512,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/chat/completions", json=body, timeout=180)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    message = r.json()["choices"][0]["message"]
    return message, latency


def run_endpoint(label, base_url, model, repeats, force):
    print(f"\n=== {label}  ({base_url}, {model}) ===")
    print(f"  {'tool (expected)':<24} {'outcome':<13} {'got':<22} {'lat':>6}")
    print(f"  {'-'*70}")
    rows = []
    for user_msg, expected in SCENARIOS:
        for rep in range(repeats):
            try:
                message, latency = one_request(base_url, model, user_msg, force)
                outcome, got = classify(message, expected)
            except Exception as e:
                outcome, got, latency = "error", None, 0.0
                print(f"  {expected:<24} {'error':<13} {str(e)[:20]:<22} {'-':>6}")
                rows.append({"label": label, "expected": expected, "outcome": outcome,
                             "got": "", "latency_s": ""})
                continue
            if rep == 0:
                print(f"  {expected:<24} {outcome:<13} {str(got or '-'):<22} {latency:>5.2f}s")
            rows.append({"label": label, "expected": expected, "outcome": outcome,
                         "got": got or "", "latency_s": round(latency, 3)})
    return rows


def summarize(label, rows):
    if not rows:
        return
    n = len(rows)
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    valid = counts.get("ok", 0) + counts.get("wrong_tool", 0)
    lats = [r["latency_s"] for r in rows if isinstance(r["latency_s"], (int, float)) and r["latency_s"]]
    print(f"\n  {label} summary  (n={n}):")
    print(f"    valid tool-call rate : {valid/n*100:5.1f}%   (emitted a real tool with parseable args)")
    print(f"    intent-match rate    : {counts.get('ok', 0)/n*100:5.1f}%   (picked the expected tool)")
    breakdown = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"    breakdown            : {breakdown}")
    if lats:
        print(f"    latency              : median={statistics.median(lats):.2f}s")


CSV_FIELDS = ["label", "expected", "outcome", "got", "latency_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="single endpoint base URL (e.g. http://host:8000/v1)")
    ap.add_argument("--model", help="served model name (matches SCAGENT_MODEL)")
    ap.add_argument("--label", default="endpoint")
    ap.add_argument("--pair", action="append", default=[],
                    help="repeatable: url,model,label")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--force", action="store_true",
                    help="tool_choice='required' — isolate JSON validity from tool-use decision")
    ap.add_argument("--out", help="optional CSV path to append per-scenario rows")
    args = ap.parse_args()

    targets = []
    if args.url:
        targets.append((args.label, args.url, args.model))
    for p in args.pair:
        url, model, label = p.split(",", 2)
        targets.append((label, url, model))
    if not targets:
        ap.error("provide --url/--model or one or more --pair")

    # Loud check: expected tools must exist in the live schema set.
    missing = sorted({exp for _, exp in SCENARIOS} - TOOL_NAMES)
    if missing:
        print(f"WARNING: scenarios reference tools not in the current schema set: {missing}\n"
              f"         (a tool was renamed/removed — update SCENARIOS)", file=sys.stderr)

    print(f"Loaded {len(TOOL_NAMES)} scagent tool schemas; {len(SCENARIOS)} scenarios "
          f"x {args.repeats} repeats; tool_choice={'required' if args.force else 'auto'}.")

    all_rows = []
    results = {}
    for label, url, model in targets:
        rows = run_endpoint(label, url, model, args.repeats, args.force)
        results[label] = rows
        all_rows.extend(rows)

    print("\n" + "=" * 60)
    for label, rows in results.items():
        summarize(label, rows)

    if args.out and all_rows:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(all_rows)
        print(f"\n  Appended {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
