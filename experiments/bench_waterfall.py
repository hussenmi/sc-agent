#!/usr/bin/env python3
"""
Single-endpoint waterfall benchmark.

Measures decode throughput and TTFT (cold + warm) across the 5-config
optimization stack:
  baseline → +CUDA graphs → +FP8 KV → +prefix cache → +MTP (full)

Uses the real scagent system prompt + tool schemas (~33K tokens) as the system
message on every request, matching actual agent usage and giving prefix caching
something meaningful to cache.

Cold vs warm TTFT:
  The first measured request (request_order=0) is always a cold start — the
  server has not yet cached the system prompt KV blocks. All subsequent requests
  share that cached prefix and are labelled warm. The warmup call (a simple
  "Hello" with no system prompt) confirms the server is responsive without
  polluting the cache.

Usage:
  python bench_waterfall.py \\
    --url http://iscb016:8000/v1 \\
    --model Qwen3.6-27B \\
    --label "A100 + MTP (full)" \\
    --hardware A100 \\
    --out results/waterfall.csv

See bench_waterfall.sh for the full 5-config sequence and exact server commands.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime

import requests


# ── System prompt ─────────────────────────────────────────────────────────────
# Load the real scagent system prompt + tool schemas so the benchmark reflects
# actual agent usage. Falls back to a size-matched placeholder if the import
# fails (e.g. running from a node where setup.sh was not sourced).

def _build_system_prompt() -> str:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scagent.agent.prompts import SYSTEM_PROMPT
        from scagent.agent.tools import get_tools
        tools_block = json.dumps(get_tools(), indent=2)
        content = SYSTEM_PROMPT + "\n\n## Available Tools\n\n" + tools_block
        print(f"  [system prompt] loaded from scagent: "
              f"{len(content):,} chars (~{len(content)//4:,} tokens)")
        return content
    except Exception as exc:
        # Generate a ~33K-token placeholder that exercises the same KV cache
        # pressure as the real prompt without requiring the scagent import.
        print(f"  [system prompt] scagent import failed ({exc}) — "
              f"using size-matched placeholder (~33K tokens)")
        base = (
            "You are an expert single-cell RNA-seq analysis agent. "
            "You help researchers analyze their data following established "
            "best practices for quality control, normalization, dimensionality "
            "reduction, clustering, and cell type annotation. "
        )
        # Repeat until we reach ~132K chars ≈ 33K tokens
        placeholder = (base * ((132_000 // len(base)) + 1))[:132_000]
        return placeholder


SYSTEM_PROMPT = _build_system_prompt()


# ── Prompts ───────────────────────────────────────────────────────────────────
# All prompts generate ≥ 50 completion tokens so decode throughput measurements
# are meaningful. "agent" prompts mirror the actual distribution of scagent
# outputs; "medium" and "long" cover diverse output lengths.
# No trivial 1-sentence prompts — short outputs (~4 tokens) produce decode
# throughput measurements dominated by timing noise rather than GPU speed.

PROMPTS = [
    ("agent",
     "You are running a single-cell RNA-seq analysis. Leiden clustering at "
     "resolution 0.5 produced 9 clusters. Generate a JSON tool call for "
     "`run_cluster_qc` to assess cluster quality, including appropriate "
     "parameters. Output only the JSON object."),

    ("agent",
     "You just ran a UMAP on single-cell RNA-seq data. The plot shows 9 clusters: "
     "clusters 0–4 form a tight group at the top-left, clusters 5–7 are scattered "
     "across the bottom, and cluster 8 is isolated on the right with high "
     "mitochondrial gene expression. Write a 2-paragraph scientific narration "
     "of these observations for the lab notebook."),

    ("agent",
     "QC metrics for a PBMC dataset show median 12% mitochondrial content with a "
     "tail reaching 45%, median 1,800 genes per cell, and 3 clusters with n_genes "
     "under 400 visible after initial UMAP. Narrate a 2-paragraph interpretation: "
     "what each observation suggests about data quality and what the recommended "
     "next step is before finalising cell type annotation."),

    ("medium",
     "Explain how a transformer self-attention block works in 3 paragraphs, "
     "covering the query/key/value projections, the scaled dot-product, "
     "and why masking is needed during decoder training."),

    ("medium",
     "Write a Python function that finds the longest common subsequence of two "
     "strings using dynamic programming. Include a brief docstring and a concrete "
     "example showing the expected output."),

    ("medium",
     "List 8 differences between TCP and UDP, with one-line explanations for each. "
     "Format them as a numbered list."),

    ("long",
     "Write a 400-word essay on the role of mitochondria in cellular apoptosis, "
     "covering the intrinsic pathway, cytochrome c release, caspase activation, "
     "and the regulatory role of Bcl-2 family proteins."),

    ("long",
     "Summarize the major causes, key events, and consequences of World War I "
     "in detail, covering the alliance systems, the assassination of Archduke "
     "Franz Ferdinand, the major fronts including the Western and Eastern fronts, "
     "the entry of the United States, and the post-war settlement at Versailles."),
]


CSV_FIELDS = [
    "timestamp", "label", "hardware",
    "prompt_type", "rep", "request_order", "cold_start",
    "prompt_tokens", "completion_tokens",
    "ttft_ms", "decode_tps", "total_s",
    "url",
]


# ── Benchmark core ────────────────────────────────────────────────────────────

def bench_one(base_url: str, model: str, messages: list, max_tokens: int) -> dict:
    body = {
        "model":          model,
        "messages":       messages,
        "max_tokens":     max_tokens,
        "temperature":    0,
        "stream":         True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = None
    prompt_tokens = None

    with requests.post(
        f"{base_url}/chat/completions",
        json=body,
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = raw[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            now = time.perf_counter()
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") and ttft is None:
                    ttft = now - t0
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")

    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    decode_time = max(total - ttft, 1e-6)
    decode_tps = (completion_tokens / decode_time) if completion_tokens else None
    return {
        "ttft_ms":           ttft * 1000,
        "decode_tps":        decode_tps,
        "total_s":           total,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def warmup(base_url: str, model: str) -> bool:
    """Simple warmup — no system prompt so it does not populate the prefix cache."""
    print("  warmup (no system prompt — cache stays cold)... ", end="", flush=True)
    try:
        bench_one(
            base_url, model,
            [{"role": "user", "content": "Hello."}],
            max_tokens=8,
        )
        print("ok")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--url",        required=True,
                    help="OpenAI-compatible /v1 base URL")
    ap.add_argument("--model",      required=True,
                    help="served model name from start_vllm.sh")
    ap.add_argument("--label",      required=True,
                    help="config name, e.g. 'A100 + MTP (full)'")
    ap.add_argument("--hardware",   required=True, choices=["A100", "H100"])
    ap.add_argument("--out",        default="results/waterfall.csv")
    ap.add_argument("--repeats",    type=int, default=4,
                    help="full passes through all 8 prompts (total = repeats × 8 requests)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--no-warmup",  action="store_true")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.exists(args.out)

    total_requests = args.repeats * len(PROMPTS)
    print(f"\n{'='*60}")
    print(f"Waterfall benchmark: {args.label}")
    print(f"  URL:            {args.url}")
    print(f"  Model:          {args.model}")
    print(f"  Hardware:       {args.hardware}")
    print(f"  Repeats:        {args.repeats}  ({total_requests} requests total)")
    print(f"  Max tokens:     {args.max_tokens}")
    print(f"  System prompt:  {len(SYSTEM_PROMPT):,} chars (~{len(SYSTEM_PROMPT)//4:,} tokens)")
    print(f"  Output:         {args.out}")
    print(f"{'='*60}")

    if not args.no_warmup:
        if not warmup(args.url, args.model):
            sys.exit(1)

    rows = []
    ts = datetime.now().isoformat(timespec="seconds")
    request_order = 0

    # Outer loop: repeats — each pass sees all 8 prompts in sequence.
    # request_order=0 (first prompt, first pass) is always the cold start;
    # everything after has the system prompt in the KV cache.
    for rep in range(args.repeats):
        for prompt_type, prompt_text in PROMPTS:
            cold = (request_order == 0)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt_text},
            ]
            try:
                r = bench_one(args.url, args.model, messages, args.max_tokens)
            except Exception as e:
                print(f"  [rep{rep} {prompt_type:6s}{'  [COLD]' if cold else '        '}] "
                      f"FAILED: {e}")
                request_order += 1
                continue

            row = {
                "timestamp":         ts,
                "label":             args.label,
                "hardware":          args.hardware,
                "prompt_type":       prompt_type,
                "rep":               rep,
                "request_order":     request_order,
                "cold_start":        cold,
                "prompt_tokens":     r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "ttft_ms":           round(r["ttft_ms"], 1),
                "decode_tps":        round(r["decode_tps"], 2) if r["decode_tps"] else "",
                "total_s":           round(r["total_s"], 3),
                "url":               args.url,
            }
            rows.append(row)

            cold_tag   = "  [COLD]" if cold else "       "
            decode_str = f"{r['decode_tps']:>6.1f}" if r["decode_tps"] is not None else "   N/A"
            print(
                f"  [rep{rep} {prompt_type:6s}{cold_tag}]  "
                f"prompt={str(r['prompt_tokens'] or '?'):>5}  "
                f"out={str(r['completion_tokens'] or '?'):>4}  "
                f"TTFT={r['ttft_ms']:>7.0f} ms  "
                f"decode={decode_str} tok/s"
            )
            request_order += 1

    if not rows:
        print("No results collected — is the server running?")
        sys.exit(1)

    cold_rows = [r for r in rows if r["cold_start"]]
    warm_rows = [r for r in rows if not r["cold_start"]]

    print(f"\n  ── Summary: {args.label} ──")
    if cold_rows:
        print(f"    TTFT cold    {cold_rows[0]['ttft_ms']:.0f} ms  "
              f"(n=1, first request — system prompt not yet cached)")
    if warm_rows:
        warm_ttfts = [r["ttft_ms"] for r in warm_rows]
        print(f"    TTFT warm    median={statistics.median(warm_ttfts):.0f} ms  "
              f"p90={statistics.quantiles(warm_ttfts, n=10)[8]:.0f} ms  "
              f"(n={len(warm_ttfts)}, system prompt cached)")
    tpss = [r["decode_tps"] for r in rows if r["decode_tps"]]
    if tpss:
        print(f"    decode       median={statistics.median(tpss):.1f} tok/s  "
              f"p90={statistics.quantiles(tpss, n=10)[1]:.1f} tok/s  "
              f"(n={len(tpss)})")

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Appended {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
