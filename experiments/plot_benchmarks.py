#!/usr/bin/env python3
"""
Generate article figures from benchmark CSV files.

Figures produced:
  1. waterfall.png       — decode throughput and TTFT for each optimization config
  2. prefix_cache.png    — TTFT per agent iteration: cached vs uncached
  3. mtp_acceptance.png  — MTP acceptance rate distribution with context
  4. a100_vs_h100.png    — final side-by-side comparison (fully optimized)

Usage:
  python plot_benchmarks.py \\
    --waterfall    results/waterfall.csv \\
    --prefix-cache results/prefix_cache.csv \\
    --mtp-stats    results/mtp_stats.csv \\
    --out          figures/

  # Generate only specific figures:
  python plot_benchmarks.py --waterfall results/waterfall.csv --out figures/
  python plot_benchmarks.py --prefix-cache results/prefix_cache.csv --out figures/
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib is required.  pip install matplotlib")
    sys.exit(1)

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Color palette — A100 blues, H100 greens
A100_COLORS = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]   # light → dark
H100_COLORS = ["#C8E6C9", "#66BB6A", "#2E7D32", "#1B5E20"]
CACHED_COLOR   = "#1E88E5"
UNCACHED_COLOR = "#E53935"
MTP_COLOR      = "#2E7D32"
DFLASH_COLOR   = "#E53935"


# ── Waterfall ─────────────────────────────────────────────────────────────────

# Expected config order for the waterfall (configs appear in this order on chart)
WATERFALL_ORDER = [
    "baseline",
    "+ CUDA graphs",
    "+ FP8 KV",
    "+ prefix cache",
    "+ MTP (full)",
]


def _normalize_label(label: str) -> str:
    """Map raw labels to canonical waterfall step names."""
    label_lower = label.lower()
    if "baseline" in label_lower:
        return "baseline"
    if "cuda" in label_lower:
        return "+ CUDA graphs"
    if "fp8 kv" in label_lower or "fp8kv" in label_lower:
        return "+ FP8 KV"
    if "prefix" in label_lower:
        return "+ prefix cache"
    if "mtp" in label_lower or "full" in label_lower:
        return "+ MTP (full)"
    return label


def load_waterfall(path: str) -> dict:
    """
    Returns:
      {hardware: {config_label: {
          "ttft_cold": [ms, ...],   # request_order==0 rows (1 per config run)
          "ttft_warm": [ms, ...],   # all other rows (system prompt cached)
          "decode_tps": [tok/s, ...]
      }}}

    Backward-compatible: CSVs without a cold_start column treat all TTFT
    rows as warm (old data will show only the warm bar).
    """
    data = defaultdict(lambda: defaultdict(
        lambda: {"ttft_cold": [], "ttft_warm": [], "decode_tps": []}
    ))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            hw  = row["hardware"]
            lbl = _normalize_label(row["label"])
            cold = row.get("cold_start", "False").strip().lower() in ("true", "1", "yes")
            try:
                ttft = float(row["ttft_ms"])
                if cold:
                    data[hw][lbl]["ttft_cold"].append(ttft)
                else:
                    data[hw][lbl]["ttft_warm"].append(ttft)
            except (ValueError, KeyError):
                pass
            try:
                tps = float(row["decode_tps"])
                data[hw][lbl]["decode_tps"].append(tps)
            except (ValueError, KeyError):
                pass
    return data


def plot_waterfall(path: str, out_dir: str):
    data = load_waterfall(path)
    if not data:
        print(f"  waterfall: no data in {path}")
        return

    hardwares = sorted(data.keys())
    configs = [c for c in WATERFALL_ORDER if any(c in data[hw] for hw in hardwares)]
    if not configs:
        configs = sorted({c for hw in hardwares for c in data[hw]})

    n_hw      = len(hardwares)
    bar_h     = 0.28
    group_gap = 0.20
    # Each config row holds: n_hw warm bars + n_hw cold bars = 2*n_hw bars
    group_h   = (2 * n_hw) * bar_h + group_gap

    fig_height = max(6, len(configs) * group_h * 1.5 + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_height))

    hw_colors = {
        "A100": {"warm": A100_COLORS[2], "cold": A100_COLORS[1]},
        "H100": {"warm": H100_COLORS[2], "cold": H100_COLORS[1]},
    }

    for ax_idx, (panel, xlabel, title) in enumerate([
        ("decode",  "Decode throughput (tok/s)",  "Decode Throughput"),
        ("ttft",    "Median TTFT (ms)",            "Time to First Token\n(warm cache — repeated requests)"),
    ]):
        ax = axes[ax_idx]
        yticks, ylabels = [], []

        for ci, cfg in enumerate(configs):
            base_y = ci * group_h

            if panel == "decode":
                # One bar per GPU — median over all requests (cold/warm mixed;
                # decode throughput is not affected by cache state)
                for hi, hw in enumerate(hardwares):
                    vals = data[hw].get(cfg, {}).get("decode_tps", [])
                    if not vals:
                        continue
                    med   = statistics.median(vals)
                    color = hw_colors.get(hw, {}).get("warm", "#888888")
                    y_pos = base_y + hi * bar_h
                    ax.barh(y_pos, med, height=bar_h * 0.85,
                            color=color, edgecolor="white", linewidth=0.6)
                    ax.text(med * 1.015, y_pos, f"{med:.0f}",
                            va="center", fontsize=8.5, color="#333333")
                yticks.append(base_y + (n_hw - 1) * bar_h / 2)

            else:
                # Warm TTFT only — one bar per GPU per config.
                # For pre-caching configs (baseline through +FP8 KV), prefix caching
                # is disabled so every request pays full prefill cost; warm ≈ cold.
                # Showing warm only keeps the axis readable and lets the dramatic
                # drop at "+prefix cache" read clearly. The dedicated prefix_cache
                # figure handles the cold-vs-warm comparison in detail.
                for hi, hw in enumerate(hardwares):
                    warm_vals = data[hw].get(cfg, {}).get("ttft_warm", [])
                    if not warm_vals:
                        continue
                    warm_med = statistics.median(warm_vals)
                    color = hw_colors.get(hw, {}).get("warm", "#888888")
                    y_pos = base_y + hi * bar_h
                    ax.barh(y_pos, warm_med, height=bar_h * 0.85,
                            color=color, edgecolor="white", linewidth=0.6)
                    ax.text(warm_med * 1.015, y_pos, f"{warm_med:.0f}",
                            va="center", fontsize=8.5, color="#333333")

                yticks.append(base_y + (n_hw - 1) * bar_h / 2)

            ylabels.append(cfg)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.invert_yaxis()
        ax.set_xlim(left=0)

        legend_patches = [
            mpatches.Patch(color=A100_COLORS[2], label="A100 (BF16 weights)"),
            mpatches.Patch(color=H100_COLORS[2], label="H100 (FP8 weights)"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    fig.suptitle(
        "Cumulative effect of each optimization — Qwen3.6-27B · 2× GPU · single-user session\n"
        "Each config adds one optimization on top of all previous ones",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "waterfall.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Prefix cache ──────────────────────────────────────────────────────────────
#
# Derived from waterfall data: "+FP8 KV" config (caching disabled) vs
# "+prefix cache" config (caching enabled, all else equal).  This is the
# cleanest A/B because the only variable that changes between those two
# waterfall steps is prefix caching.

def plot_prefix_cache(waterfall_path: str, out_dir: str):
    """Generate cold-vs-warm TTFT bar chart from the waterfall CSV.

    Uses "+FP8 KV" rows as the "without caching" condition and
    "+prefix cache" rows as the "with caching" condition.
    """
    data = load_waterfall(waterfall_path)
    if not data:
        print(f"  prefix cache: no data in {waterfall_path}")
        return

    hardwares = sorted(data.keys())

    fig, axes = plt.subplots(1, len(hardwares),
                             figsize=(6 * len(hardwares), 5), squeeze=False)

    bar_w = 0.32
    phases  = ["Cold start\n(first request)", "Warm cache\n(repeated requests)"]
    x_pos   = [0, 1]

    for hi, hw in enumerate(hardwares):
        ax = axes[0][hi]
        colors_hw = A100_COLORS if hw == "A100" else H100_COLORS
        hw_data   = data.get(hw, {})

        no_cache  = hw_data.get("+ FP8 KV",       {})
        yes_cache = hw_data.get("+ prefix cache",  {})

        def safe_median(lst):
            return statistics.median(lst) if lst else 0

        pairs = [
            (colors_hw[2],    "With prefix caching",
             safe_median(yes_cache.get("ttft_cold", [])),
             safe_median(yes_cache.get("ttft_warm", []))),
            (UNCACHED_COLOR,  "Without prefix caching",
             safe_median(no_cache.get("ttft_cold", [])),
             safe_median(no_cache.get("ttft_warm", []))),
        ]

        max_val = max(v for _, _, c, w in pairs for v in (c, w) if v > 0)

        for mi, (bar_color, label, cold_val, warm_val) in enumerate(pairs):
            vals   = [cold_val, warm_val]
            offset = (mi - 0.5) * bar_w
            bars   = ax.bar([x + offset for x in x_pos], vals,
                            width=bar_w * 0.9, color=bar_color,
                            label=label, edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + max_val * 0.02,
                            f"{val:.0f}ms",
                            ha="center", va="bottom",
                            fontsize=9, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(phases, fontsize=11)
        ax.set_ylabel("Median TTFT (ms)", fontsize=11)
        ax.set_title(f"{hw}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.set_ylim(bottom=0, top=max_val * 1.22)

    fig.suptitle(
        "Prefix caching: TTFT for a ~33K token fixed prefix (system prompt + tool schemas)\n"
        "Cold start = first request (cache empty); warm = all subsequent requests",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "prefix_cache.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ── MTP acceptance rate ───────────────────────────────────────────────────────

def load_mtp_stats(path: str) -> dict:
    """Returns {log_file: [acceptance_rate_pct, ...]}"""
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                # Filter near-zero throughput lines (idle server logging)
                if float(row.get("accepted_tps", 0)) < 1.0:
                    continue
                log_file = row["log_file"]
                rate = float(row["avg_acceptance_rate_pct"])
                data[log_file].append(rate)
            except (ValueError, KeyError):
                continue
    return data


def plot_mtp_acceptance(path: str, out_dir: str):
    data = load_mtp_stats(path)
    if not data:
        print(f"  MTP stats: no data in {path}")
        return

    def hw_label(fname):
        return "H100 (FP8 weights)" if "iscg" in fname.lower() else "A100 (BF16 weights)"

    # Build ordered list: A100 first, H100 second
    series = sorted(
        [(hw_label(f), f, rates) for f, rates in data.items() if rates],
        key=lambda x: x[0],
    )

    import random
    rng = random.Random(42)

    fig, ax = plt.subplots(figsize=(9, 4))

    row_colors = {"A100 (BF16 weights)": A100_COLORS[2],
                  "H100 (FP8 weights)":  H100_COLORS[2]}
    y_positions = {label: i for i, (label, _, _) in enumerate(series)}

    for label, _, rates in series:
        color = row_colors[label]
        y_base = y_positions[label]
        # Jitter dots vertically within ±0.15 of their row
        ys = [y_base + rng.uniform(-0.15, 0.15) for _ in rates]
        ax.scatter(rates, ys, color=color, s=55, alpha=0.75, zorder=3)
        med = statistics.median(rates)
        # Median as a wide horizontal tick
        ax.plot([med, med], [y_base - 0.28, y_base + 0.28],
                color=color, linewidth=3, zorder=4)
        ax.text(med, y_base + 0.35, f"median {med:.0f}%",
                ha="center", va="bottom", fontsize=10,
                color=color, fontweight="bold")

    # Y-axis labels
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(
        [f"{lbl}\n(n={len(rates)})" for lbl, _, rates in series],
        fontsize=11,
    )
    ax.set_ylim(-0.6, len(series) - 0.4)

    # X-axis: zoom to where data is
    ax.set_xlim(60, 102)
    ax.set_xlabel("MTP acceptance rate per 10-second window (%)", fontsize=11)

    ax.set_title(
        "MTP speculative decoding acceptance rate during inference\n"
        "Qwen3.6-27B — auxiliary prediction head baked into model weights",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "mtp_acceptance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ── A100 vs H100 final comparison ─────────────────────────────────────────────

def plot_a100_vs_h100(waterfall_path: str, out_dir: str):
    data = load_waterfall(waterfall_path)

    # Use only the fully optimized config for this chart
    full_config = "+ MTP (full)"
    metrics = [
        ("decode_tps", "Decode throughput (tok/s)"),
        ("ttft_warm",  "Median TTFT — warm cache (ms)"),
    ]
    hardwares = ["A100", "H100"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        vals_by_hw = {}
        for hw in hardwares:
            vals = data.get(hw, {}).get(full_config, {}).get(metric, [])
            if vals:
                vals_by_hw[hw] = statistics.median(vals)

        if not vals_by_hw:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        colors = [A100_COLORS[2], H100_COLORS[2]]
        bars = ax.bar(
            list(vals_by_hw.keys()),
            list(vals_by_hw.values()),
            color=colors[:len(vals_by_hw)],
            width=0.5,
            edgecolor="white",
        )

        for bar, (hw, val) in zip(bars, vals_by_hw.items()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val * 1.02, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

        # Annotate the ratio
        if len(vals_by_hw) == 2 and "A100" in vals_by_hw and "H100" in vals_by_hw:
            a = vals_by_hw["A100"]
            h = vals_by_hw["H100"]
            if metric == "decode_tps":
                ratio = h / a
                ax.text(0.5, 0.92,
                        f"H100 is {ratio:.2f}× faster",
                        ha="center", transform=ax.transAxes,
                        fontsize=11, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.8))
            else:
                ratio = a / h if h > 0 else 0
                label = f"H100 is {ratio:.1f}× faster (TTFT)" if ratio > 1 else "Equal TTFT"
                ax.text(0.5, 0.92, label,
                        ha="center", transform=ax.transAxes,
                        fontsize=11, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.8))

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(bottom=0, top=max(vals_by_hw.values()) * 1.25)

    fig.suptitle(
        "A100 vs H100 — fully optimized (BF16 vs FP8 weights, FP8 KV, MTP)\n"
        "Qwen3.6-27B · 2× GPU · single-user interactive session",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "a100_vs_h100.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--waterfall",    help="path to waterfall.csv")
    ap.add_argument("--mtp-stats",    help="path to mtp_stats.csv")
    ap.add_argument("--out",          default="figures/",
                    help="output directory for PNG files")
    args = ap.parse_args()

    if not any([args.waterfall, args.mtp_stats]):
        ap.error("provide at least one of --waterfall, --mtp-stats")

    os.makedirs(args.out, exist_ok=True)
    print(f"Writing figures to: {args.out}")

    if args.waterfall and os.path.exists(args.waterfall):
        print(f"\nWaterfall chart ({args.waterfall}):")
        plot_waterfall(args.waterfall, args.out)
        print(f"A100 vs H100 comparison:")
        plot_a100_vs_h100(args.waterfall, args.out)
        print(f"Prefix cache chart (derived from waterfall):")
        plot_prefix_cache(args.waterfall, args.out)
    elif args.waterfall:
        print(f"WARNING: {args.waterfall} not found — skipping waterfall figures")

    if args.mtp_stats and os.path.exists(args.mtp_stats):
        print(f"\nMTP acceptance chart ({args.mtp_stats}):")
        plot_mtp_acceptance(args.mtp_stats, args.out)
    elif args.mtp_stats:
        print(f"WARNING: {args.mtp_stats} not found — skipping MTP figure")

    print("\nDone.")


if __name__ == "__main__":
    main()
