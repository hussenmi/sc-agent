#!/usr/bin/env python3
"""
Perplexity benchmark across context lengths — for capturing baseline numbers
before any dynamic-YaRN work, and for re-running afterwards to measure the
quality effect.

What this measures:
  At each target context length L, the script extracts non-overlapping
  L-token chunks from a long natural-text file, sends each chunk to vLLM
  with echo=True + logprobs=1, and computes perplexity from the per-token
  logprobs returned by the server. Lower PPL = better next-token prediction.

  This is the standard metric for evaluating whether RoPE scaling
  (linear / NTK / YaRN — static or dynamic) hurts short-context quality.

The expected pattern for YaRN configurations:
  no-YaRN          → low PPL at 4K/32K, fails (or wraps) past native cap
  static factor=4  → higher PPL at 4K/32K (the cost), works at 600K+
  dynamic YaRN     → low PPL at 4K/32K AND works at 600K+ (the claim)

Usage:
  # Baseline against current server (no YaRN, native 262K cap):
  python bench_yarn_ppl.py \\
    --url http://localhost:8000/v1 \\
    --model Qwen3.6-27B \\
    --text-file data/war_and_peace.txt \\
    --ctx-lengths 4096 32768 131072 \\
    --chunks-per-length 8 \\
    --label no_yarn \\
    --out results/baseline_no_yarn.json

  # Smoke test (1 short chunk per length, quick sanity check):
  python bench_yarn_ppl.py --url http://localhost:8000/v1 \\
    --model Qwen3.6-27B --text-file data/war_and_peace.txt \\
    --ctx-lengths 1024 --chunks-per-length 1 --label smoke \\
    --out results/smoke.json

Notes:
  - Uses vLLM-specific /tokenize endpoint + token-id prompts for exact length
    control (avoids re-tokenization mismatches).
  - Skips the first token's logprob (always None — no preceding context).
  - max_tokens=1 with echo=True forces a single decode step; nearly all the
    wall time is prefill on the echoed prompt, which is exactly what we want
    to measure quality on.
  - Output JSON includes per-chunk NLL so results can be re-aggregated later.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests


def vllm_tokenize(base_url: str, model: str, text: str) -> list[int]:
    root = base_url.rstrip("/").removesuffix("/v1")
    r = requests.post(
        f"{root}/tokenize",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["tokens"]


def measure_chunk(base_url: str, model: str, token_ids: list[int]) -> tuple[float, int]:
    """Send token_ids as prompt with echo+logprobs, return (sum_nll, num_scored_tokens)."""
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 1,
        "echo": True,
        "logprobs": 1,
        "temperature": 0,
    }
    r = requests.post(
        f"{base_url.rstrip('/')}/completions",
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    lp = data["choices"][0]["logprobs"]["token_logprobs"]
    # First entry is None (no preceding context). Drop the trailing generated
    # token's logprob too — we only want PPL of the echoed prompt.
    prompt_lp = lp[1:len(token_ids)]
    nll = -sum(x for x in prompt_lp if x is not None)
    n = sum(1 for x in prompt_lp if x is not None)
    return nll, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--text-file", required=True, type=Path)
    ap.add_argument("--ctx-lengths", type=int, nargs="+", required=True,
                    help="Target prompt lengths in tokens, e.g. 4096 32768 131072")
    ap.add_argument("--chunks-per-length", type=int, default=8)
    ap.add_argument("--label", required=True,
                    help="Tag for this run, e.g. no_yarn / static_yarn_f4 / dynamic_yarn")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-tokens", type=int, default=2000,
                    help="Skip leading N tokens (book header / license boilerplate)")
    args = ap.parse_args()

    text = args.text_file.read_text(encoding="utf-8", errors="replace")
    print(f"[bench] tokenizing {len(text):,} chars via vLLM /tokenize ...", flush=True)
    all_tokens = vllm_tokenize(args.url, args.model, text)
    print(f"[bench] got {len(all_tokens):,} tokens", flush=True)

    available = all_tokens[args.skip_tokens:]
    results = {
        "label": args.label,
        "model": args.model,
        "url": args.url,
        "text_file": str(args.text_file),
        "total_tokens": len(all_tokens),
        "skip_tokens": args.skip_tokens,
        "per_length": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for L in args.ctx_lengths:
        if L > len(available):
            print(f"[bench] skipping L={L}: only {len(available)} tokens available", flush=True)
            continue
        # Non-overlapping chunks, capped by what fits in the text and the request limit.
        max_chunks = min(args.chunks_per_length, len(available) // L)
        per_chunk = []
        total_nll = 0.0
        total_n = 0
        for i in range(max_chunks):
            chunk = available[i * L : (i + 1) * L]
            t0 = time.time()
            try:
                nll, n = measure_chunk(args.url, args.model, chunk)
            except requests.HTTPError as e:
                print(f"[bench] L={L} chunk={i} HTTP error: {e.response.status_code} {e.response.text[:200]}", flush=True)
                continue
            elapsed = time.time() - t0
            ppl = math.exp(nll / n) if n else float("nan")
            print(f"[bench] L={L} chunk={i+1}/{max_chunks}  "
                  f"PPL={ppl:.3f}  tokens_scored={n}  {elapsed:.1f}s",
                  flush=True)
            per_chunk.append({"chunk": i, "nll": nll, "tokens": n, "ppl": ppl, "elapsed_s": elapsed})
            total_nll += nll
            total_n += n
        agg_ppl = math.exp(total_nll / total_n) if total_n else float("nan")
        results["per_length"][str(L)] = {
            "target_length": L,
            "chunks": per_chunk,
            "aggregate_ppl": agg_ppl,
            "aggregate_tokens": total_n,
        }
        print(f"[bench] L={L} AGGREGATE PPL = {agg_ppl:.3f} over {total_n} tokens", flush=True)

    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
