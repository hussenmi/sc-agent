#!/usr/bin/env python3
"""A/B benchmark for OpenAI-compatible LLM endpoints.

Usage:
  python bench_llm.py --url http://iscb009:8000/v1 --model Qwen3.6-27B --label A100-BF16
  python bench_llm.py --url http://isch003:8000/v1 --model Qwen3.6-27B --label H100-FP8

Or compare two in one shot:
  python bench_llm.py \
    --pair http://iscb009:8000/v1,Qwen3.6-27B,A100-BF16 \
    --pair http://isch003:8000/v1,Qwen3.6-27B,H100-FP8  \
    --repeats 2
"""
import argparse
import json
import statistics
import sys
import time

import requests

PROMPTS = [
    ("short", "What is 17 * 23? Answer with just the number."),
    ("medium", "Explain how a transformer attention block works in 3 paragraphs."),
    ("medium", "Write a Python function that finds the longest common subsequence of two strings."),
    ("medium", "List 8 differences between TCP and UDP, with one-line explanations."),
    ("long", "Write a 400-word story about a marine biologist who discovers a new species of bioluminescent jellyfish."),
    ("long", "Summarize the major causes, key events, and consequences of World War I in detail."),
]


def bench_one(base_url, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = None
    prompt_tokens = None
    text_chars = 0

    with requests.post(
        f"{base_url}/chat/completions",
        json=body,
        stream=True,
        timeout=180,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            now = time.perf_counter()
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = now - t0
                    text_chars += len(content)
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
        "ttft": ttft,
        "total": total,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tps": decode_tps,
        "text_chars": text_chars,
    }


def run_endpoint(label, base_url, model, repeats=2, max_tokens=300, warmup=True):
    print(f"\n=== {label}  ({base_url}, {model}) ===")
    if warmup:
        print("warmup…", end=" ", flush=True)
        try:
            bench_one(base_url, model, "Hello.", max_tokens=8)
        except Exception as e:
            print(f"FAILED: {e}")
            return None
        print("done")

    rows = []
    for tag, prompt in PROMPTS:
        for rep in range(repeats):
            try:
                r = bench_one(base_url, model, prompt, max_tokens=max_tokens)
            except Exception as e:
                print(f"  [{tag} rep{rep}] FAILED: {e}")
                continue
            r["tag"] = tag
            rows.append(r)
            print(
                f"  [{tag:6s} rep{rep}]  "
                f"prompt={r['prompt_tokens']:>4}  "
                f"out={r['completion_tokens']:>4}  "
                f"TTFT={r['ttft']*1000:>6.0f} ms  "
                f"decode={r['decode_tps']:>6.1f} tok/s  "
                f"total={r['total']:>5.2f} s"
            )
    return rows


def summarize(label, rows):
    if not rows:
        return
    ttfts = [r["ttft"] * 1000 for r in rows]
    tpss = [r["decode_tps"] for r in rows if r["decode_tps"]]
    print(f"\n  {label} summary:")
    print(f"    TTFT     median={statistics.median(ttfts):.0f} ms   "
          f"p90={statistics.quantiles(ttfts, n=10)[8]:.0f} ms")
    print(f"    decode   median={statistics.median(tpss):.1f} tok/s   "
          f"p90={statistics.quantiles(tpss, n=10)[1]:.1f} tok/s   (low p90 = slower tail)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="single endpoint base URL (e.g. http://host:8000/v1)")
    ap.add_argument("--model", help="served model name (matches SCAGENT_MODEL)")
    ap.add_argument("--label", default="endpoint")
    ap.add_argument("--pair", action="append", default=[],
                    help="repeatable: url,model,label  e.g. http://host:8000/v1,Qwen3.6-27B,H100-FP8")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    targets = []
    if args.url:
        targets.append((args.label, args.url, args.model))
    for p in args.pair:
        url, model, label = p.split(",", 2)
        targets.append((label, url, model))
    if not targets:
        ap.error("provide --url/--model or one or more --pair")

    results = {}
    for label, url, model in targets:
        results[label] = run_endpoint(
            label, url, model,
            repeats=args.repeats,
            max_tokens=args.max_tokens,
            warmup=not args.no_warmup,
        )

    print("\n" + "=" * 60)
    for label, rows in results.items():
        summarize(label, rows)

    if len(results) == 2:
        a_label, b_label = list(results.keys())
        a, b = results[a_label], results[b_label]
        if a and b:
            a_tps = statistics.median([r["decode_tps"] for r in a if r["decode_tps"]])
            b_tps = statistics.median([r["decode_tps"] for r in b if r["decode_tps"]])
            a_ttft = statistics.median([r["ttft"] for r in a])
            b_ttft = statistics.median([r["ttft"] for r in b])
            print(f"\n  {b_label} vs {a_label}:")
            print(f"    decode speedup: {b_tps / a_tps:.2f}x   "
                  f"({a_label} {a_tps:.1f} → {b_label} {b_tps:.1f} tok/s)")
            print(f"    TTFT speedup:   {a_ttft / b_ttft:.2f}x   "
                  f"({a_label} {a_ttft*1000:.0f} → {b_label} {b_ttft*1000:.0f} ms)")


if __name__ == "__main__":
    main()
