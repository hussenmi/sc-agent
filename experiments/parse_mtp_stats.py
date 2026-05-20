#!/usr/bin/env python3
"""
Extract MTP / speculative decoding statistics from vLLM server logs.

vLLM logs SpecDecoding metrics every 10 seconds during active inference:

  INFO [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 1.87,
  Accepted throughput: 31.29 tokens/s, Drafted throughput: 36.78 tokens/s,
  Accepted: 313 tokens, Drafted: 368 tokens,
  Per-position acceptance rate: 0.851, Avg Draft acceptance rate: 85.1%

This script parses all matching lines from one or more log files, outputs a
structured CSV, and prints a summary table for quick review.

Usage:
  # Parse one log file:
  python parse_mtp_stats.py logs/vllm_isch003_Qwen_Qwen3.6-27B-FP8.log

  # Parse multiple log files and combine:
  python parse_mtp_stats.py logs/vllm_iscb007*.log logs/vllm_iscg002*.log \\
    --out results/mtp_stats.csv

  # Filter to only lines logged during a specific benchmark run
  # (use --after / --before to scope by timestamp):
  python parse_mtp_stats.py logs/vllm_isch003_Qwen_Qwen3.6-27B-FP8.log \\
    --after "05-07 15:45" --before "05-07 15:50"

Output CSV columns:
  log_file, timestamp_raw, mean_acceptance_length,
  accepted_tps, drafted_tps, accepted_tokens, drafted_tokens,
  per_position_acceptance_rate, avg_acceptance_rate_pct
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Pattern matches the full SpecDecoding metrics line
_PATTERN = re.compile(
    r"(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"SpecDecoding metrics:\s*"
    r"Mean acceptance length:\s*(?P<mean_acc_len>[\d.]+),\s*"
    r"Accepted throughput:\s*(?P<acc_tps>[\d.]+) tokens/s,\s*"
    r"Drafted throughput:\s*(?P<draft_tps>[\d.]+) tokens/s,\s*"
    r"Accepted:\s*(?P<accepted_tokens>\d+) tokens,\s*"
    r"Drafted:\s*(?P<drafted_tokens>\d+) tokens,\s*"
    r"Per-position acceptance rate:\s*(?P<per_pos_rate>[\d.]+),\s*"
    r"Avg Draft acceptance rate:\s*(?P<avg_rate>[\d.]+)%"
)

CSV_FIELDS = [
    "log_file", "timestamp_raw",
    "mean_acceptance_length",
    "accepted_tps", "drafted_tps",
    "accepted_tokens", "drafted_tokens",
    "per_position_acceptance_rate",
    "avg_acceptance_rate_pct",
]


def parse_log(path: str, after: str = "", before: str = "") -> list:
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _PATTERN.search(line)
            if not m:
                continue
            ts = m.group("ts")
            if after and ts < after:
                continue
            if before and ts > before:
                continue
            rows.append({
                "log_file": os.path.basename(path),
                "timestamp_raw": ts,
                "mean_acceptance_length": float(m.group("mean_acc_len")),
                "accepted_tps": float(m.group("acc_tps")),
                "drafted_tps": float(m.group("draft_tps")),
                "accepted_tokens": int(m.group("accepted_tokens")),
                "drafted_tokens": int(m.group("drafted_tokens")),
                "per_position_acceptance_rate": float(m.group("per_pos_rate")),
                "avg_acceptance_rate_pct": float(m.group("avg_rate")),
            })
    return rows


def summarize(rows: list, label: str):
    if not rows:
        print(f"  {label}: no data")
        return

    rates = [r["avg_acceptance_rate_pct"] for r in rows]
    lengths = [r["mean_acceptance_length"] for r in rows]
    acc_tps = [r["accepted_tps"] for r in rows]

    # Filter out near-zero throughput lines (logged during idle periods)
    active = [r for r in rows if r["accepted_tps"] > 1.0]
    active_rates = [r["avg_acceptance_rate_pct"] for r in active] or rates

    def pct(lst, p):
        lst = sorted(lst)
        idx = max(0, int(len(lst) * p / 100) - 1)
        return lst[idx]

    print(f"\n  {label}  ({len(rows)} log lines, {len(active)} during active inference)")
    print(f"  {'Metric':<30} {'min':>7} {'median':>8} {'p90':>7} {'max':>7}")
    print(f"  {'-'*60}")
    print(f"  {'Acceptance rate (%)':<30} "
          f"{min(active_rates):>7.1f} "
          f"{sorted(active_rates)[len(active_rates)//2]:>8.1f} "
          f"{pct(active_rates, 90):>7.1f} "
          f"{max(active_rates):>7.1f}")
    print(f"  {'Mean acceptance length':<30} "
          f"{min(lengths):>7.2f} "
          f"{sorted(lengths)[len(lengths)//2]:>8.2f} "
          f"{pct(lengths, 90):>7.2f} "
          f"{max(lengths):>7.2f}")
    print(f"  {'Accepted throughput (tok/s)':<30} "
          f"{min(acc_tps):>7.1f} "
          f"{sorted(acc_tps)[len(acc_tps)//2]:>8.1f} "
          f"{pct(acc_tps, 90):>7.1f} "
          f"{max(acc_tps):>7.1f}")

    # Interpretation
    med_rate = sorted(active_rates)[len(active_rates)//2]
    if med_rate >= 70:
        verdict = f"✓ Strong — {med_rate:.0f}% median acceptance means ~{1 + med_rate/100:.1f} tokens per forward pass"
    elif med_rate >= 40:
        verdict = f"~ Marginal — {med_rate:.0f}% median acceptance; near breakeven"
    else:
        verdict = f"✗ Counterproductive — {med_rate:.0f}% below breakeven (~40%); MTP is adding overhead"
    print(f"\n  Assessment: {verdict}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("logs", nargs="+",
                    help="vLLM log file(s) — supports glob patterns")
    ap.add_argument("--out", default="results/mtp_stats.csv",
                    help="output CSV path")
    ap.add_argument("--after",  default="",
                    help="only include lines at or after this timestamp, e.g. '05-07 15:45'")
    ap.add_argument("--before", default="",
                    help="only include lines at or before this timestamp")
    args = ap.parse_args()

    # Expand globs
    paths = []
    for pattern in args.logs:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(pattern):
            paths.append(pattern)
        else:
            print(f"WARNING: no files matched '{pattern}'", file=sys.stderr)

    if not paths:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    by_file = defaultdict(list)

    for path in sorted(set(paths)):
        rows = parse_log(path, args.after, args.before)
        all_rows.extend(rows)
        by_file[path] = rows
        print(f"  {os.path.basename(path)}: {len(rows)} SpecDecoding metric lines")

    if not all_rows:
        print("\nNo SpecDecoding metrics found. Check that:")
        print("  1. The server was started without --disable-log-stats")
        print("  2. At least one inference request was made while the server was running")
        print("  3. The log file is from a server with MTP/speculative decoding enabled")
        sys.exit(1)

    # Per-file summaries
    for path, rows in by_file.items():
        label = os.path.basename(path).replace("vllm_", "").replace(".log", "")
        summarize(rows, label)

    # Combined summary if multiple files
    if len(by_file) > 1:
        summarize(all_rows, "ALL FILES COMBINED")

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Wrote {len(all_rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
