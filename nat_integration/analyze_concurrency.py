"""Analyze a NAT concurrency sweep: how does serving latency scale with load.

Reads each conc_<K>/standardized_data_all.csv produced by run_concurrency_sweep.sh,
reconstructs per-LLM-call intervals (LLM_START/END paired by UUID), and computes:
  - per-NOMINAL-level LLM latency (mean/median) + end-to-end workflow latency
  - per-INSTANTANEOUS-concurrency LLM latency: for each call, how many other LLM
    calls were in flight at its midpoint (extracted from timestamps) -> the clean
    serving-scaling curve, independent of nominal level.

Outputs latency_vs_concurrency.png + sweep_summary.{json,md}.

  /usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/python nat_integration/analyze_concurrency.py \
    --sweep nat_sweep_<stamp> --out nat_sweep_<stamp>/figs
"""

import argparse
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _llm_intervals(csv_path: str):
    """[(start_s, end_s, dur_s)] for each LLM call, paired by UUID."""
    df = pd.read_csv(csv_path)
    starts = {r.UUID: r.event_timestamp for r in df[df.event_type == "LLM_START"].itertuples()}
    out = []
    for r in df[df.event_type == "LLM_END"].itertuples():
        t0 = starts.get(r.UUID)
        if t0 is None:
            continue
        out.append((float(t0), float(r.event_timestamp), float(r.event_timestamp) - float(t0)))
    return out, df


def _workflow_dur(df) -> float | None:
    s = df[df.event_type == "WORKFLOW_START"].event_timestamp
    e = df[df.event_type == "WORKFLOW_END"].event_timestamp
    if len(s) and len(e):
        return float(e.max()) - float(s.min())
    return None


def _instantaneous_conc(intervals):
    """For each call, count calls (incl. itself) in flight at its midpoint."""
    out = []
    for s, e, d in intervals:
        mid = (s + e) / 2
        k = sum(1 for s2, e2, _ in intervals if s2 <= mid <= e2)
        out.append((k, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.sweep, "figs")
    os.makedirs(out, exist_ok=True)

    levels = {}
    all_intervals = []  # (nominal_level, start, end, dur)
    for d in sorted(glob.glob(os.path.join(args.sweep, "conc_*"))):
        try:
            K = int(os.path.basename(d).split("_")[1])
        except (IndexError, ValueError):
            continue
        csv = os.path.join(d, "standardized_data_all.csv")
        if not os.path.exists(csv):
            continue
        intervals, df = _llm_intervals(csv)
        durs = [x[2] for x in intervals]
        levels[K] = {
            "n_llm_calls": len(durs),
            "llm_latency_mean": st.mean(durs) if durs else None,
            "llm_latency_median": st.median(durs) if durs else None,
            "llm_latency_p95": (sorted(durs)[int(0.95 * len(durs))] if durs else None),
            "workflow_latency_s": _workflow_dur(df),
        }
        for s, e, dd in intervals:
            all_intervals.append((K, s, e, dd))

    # Instantaneous concurrency curve (pool all calls; overlap is naturally per-level
    # because levels run sequentially in time).
    flat = [(s, e, dd) for _, s, e, dd in all_intervals]
    inst = _instantaneous_conc(flat)
    by_k = {}
    for k, d in inst:
        by_k.setdefault(k, []).append(d)
    inst_curve = {k: {"n": len(v), "median_latency": st.median(v), "mean_latency": st.mean(v)}
                  for k, v in sorted(by_k.items())}

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ks = sorted(levels)
    ax1.plot(ks, [levels[k]["llm_latency_median"] for k in ks], "o-", label="LLM call (median)", color="#C44E52")
    ax1b = ax1.twinx()
    ax1b.plot(ks, [levels[k]["workflow_latency_s"] for k in ks], "s--", label="end-to-end", color="#4C72B0")
    ax1.set_xlabel("nominal max_concurrency")
    ax1.set_ylabel("LLM call latency (s)", color="#C44E52")
    ax1b.set_ylabel("end-to-end workflow (s)", color="#4C72B0")
    ax1.set_title("Latency vs nominal concurrency")
    ax1.set_xticks(ks)
    ax1.grid(alpha=0.3)

    ik = sorted(inst_curve)
    ax2.plot(ik, [inst_curve[k]["median_latency"] for k in ik], "o-", color="#55A868")
    for k in ik:
        ax2.annotate(f"n={inst_curve[k]['n']}", (k, inst_curve[k]["median_latency"]),
                     fontsize=7, xytext=(0, 6), textcoords="offset points", ha="center")
    ax2.set_xlabel("instantaneous # LLM calls in flight (from timestamps)")
    ax2.set_ylabel("LLM call latency (s, median)")
    ax2.set_title("Serving scaling: latency vs concurrent in-flight requests")
    ax2.set_xticks(ik)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "latency_vs_concurrency.png"), dpi=150)
    plt.close(fig)

    summary = {"nominal_levels": levels, "instantaneous_curve": inst_curve}
    json.dump(summary, open(os.path.join(out, "sweep_summary.json"), "w"), indent=2)
    lines = ["# Concurrency sweep summary", "", "## Per nominal level"]
    for k in ks:
        v = levels[k]
        lines.append(f"- conc={k}: LLM median={v['llm_latency_median']:.2f}s "
                     f"p95={v['llm_latency_p95']:.2f}s | end-to-end={v['workflow_latency_s']:.0f}s "
                     f"| {v['n_llm_calls']} calls")
    lines += ["", "## Latency vs instantaneous in-flight requests (the serving curve)"]
    for k in ik:
        v = inst_curve[k]
        lines.append(f"- {k} in flight: median latency {v['median_latency']:.2f}s (n={v['n']})")
    open(os.path.join(out, "sweep_summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nFigure + summary -> {out}/")


if __name__ == "__main__":
    main()
