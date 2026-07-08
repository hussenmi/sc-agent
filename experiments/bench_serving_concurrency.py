#!/usr/bin/env python3
"""Measure end-to-end and aggregate throughput of an OpenAI-compatible endpoint."""

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

import requests


PROMPTS = [
    "Explain how consensus protocols handle leader failure and network partitions.",
    "Design a resilient scheduler for a heterogeneous GPU computing cluster.",
    "Compare optimistic and pessimistic concurrency control in database systems.",
    "Write a technical overview of memory management in a modern operating system.",
    "Analyze failure recovery strategies for a distributed object store.",
    "Explain how speculative decoding accelerates autoregressive language models.",
    "Design an observability architecture for a large microservice deployment.",
    "Discuss the tradeoffs among data, tensor, pipeline, and expert parallelism.",
]


def request_one(base_url: str, model: str, index: int, max_tokens: int) -> dict:
    prompt = f"{PROMPTS[index % len(PROMPTS)]} Give a detailed answer. Nonce {index:04d}."
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat/completions",
        json=body,
        timeout=300,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    payload = response.json()
    usage = payload.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "index": index,
        "elapsed_s": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion_tokens,
        "e2e_output_tps": completion_tokens / elapsed if elapsed else 0.0,
    }


def run_level(
    base_url: str,
    model: str,
    concurrency: int,
    request_count: int,
    max_tokens: int,
) -> dict:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(request_one, base_url, model, index, max_tokens)
            for index in range(request_count)
        ]
        rows = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    total_output_tokens = sum(row["completion_tokens"] for row in rows)
    latencies = [row["elapsed_s"] for row in rows]
    per_request_tps = [row["e2e_output_tps"] for row in rows]
    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "wall_s": wall_s,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tps": total_output_tokens / wall_s,
        "median_request_latency_s": statistics.median(latencies),
        "median_request_output_tps": statistics.median(per_request_tps),
        "requests": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrencies", default="1,4,8")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    print("Warmup...", flush=True)
    request_one(args.url, args.model, 9999, 32)

    levels = []
    for concurrency in (int(value) for value in args.concurrencies.split(",")):
        print(f"Running concurrency={concurrency}...", flush=True)
        level = run_level(
            args.url,
            args.model,
            concurrency,
            args.requests,
            args.max_tokens,
        )
        levels.append(level)
        print(
            f"  aggregate={level['aggregate_output_tps']:.1f} tok/s  "
            f"median_request={level['median_request_output_tps']:.1f} tok/s  "
            f"median_latency={level['median_request_latency_s']:.2f}s"
        )

    result = {
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "levels": levels,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
