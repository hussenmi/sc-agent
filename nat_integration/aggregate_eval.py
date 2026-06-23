"""Aggregate + plot NAT eval/profiler outputs into the collab figures.

Consumes one or more NAT eval output dirs (each = one backend/model/config run,
possibly with --reps) and produces:
  1. accuracy_consistency.png  — per-rep coarse & major accuracy (the consistency band)
  2. trajectory_cost.png       — per-rep LLM calls + prompt tokens (trajectory variance)
  3. bottleneck.png            — per-op avg duration (where time goes)
plus summary.json / summary.md with the aggregated numbers.

Multi-backend ready: pass --run LABEL:DIR repeatedly to group backends side by side
(e.g. vLLM vs NIM vs Nemotron). With no --run, defaults to nat_luca_out.

Run under the nat venv (has pandas + matplotlib):
  /usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/python nat_integration/aggregate_eval.py \
    --run vLLM-Qwen3.6-27B:/data1/peerd/ibrahih3/cs_agent/nat_luca_out --out nat_eval_figs
"""

import argparse
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_ACC_RE = re.compile(r"acc\[(\w+)\]=([0-9.]+)")


def _load_accuracy(run_dir: str) -> list[dict]:
    """Per-rep accuracy from luca_atlas_output.json (coarse+major parsed from reasoning)."""
    path = os.path.join(run_dir, "luca_atlas_output.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    items = d.get("eval_output_items") or d.get("items") or []
    rows = []
    for it in items:
        accs = dict(_ACC_RE.findall(str(it.get("reasoning", ""))))
        row = {"id": it.get("id"), "major": it.get("score")}
        for k, v in accs.items():
            tag = "coarse" if "coarse" in k else ("major" if "major" in k else k)
            row[tag] = float(v)
        rows.append(row)
    return rows


def _load_profile(run_dir: str):
    """(per-rep dict list, per-op avg-duration dict) from standardized_data_all.csv."""
    path = os.path.join(run_dir, "standardized_data_all.csv")
    if not os.path.exists(path):
        return [], {}
    df = pd.read_csv(path)

    # Per-rep (example_number) LLM workload.
    per_rep = []
    le = df[df.event_type == "LLM_END"]
    for ex, sub in le.groupby("example_number"):
        per_rep.append({
            "rep": int(ex),
            "llm_calls": int(len(sub)),
            "prompt_tokens": int(sub.prompt_tokens.sum()),
            "completion_tokens": int(sub.completion_tokens.sum()),
            "peak_prompt_tokens": int(sub.prompt_tokens.max()),
        })

    # Per-op avg duration: pair START/END by UUID, name = tool_name else llm_name.
    durs: dict[str, list[float]] = {}
    starts = {r.UUID: r.event_timestamp for r in df[df.event_type.str.endswith("_START")].itertuples()}
    for r in df[df.event_type.str.endswith("_END")].itertuples():
        t0 = starts.get(r.UUID)
        if t0 is None:
            continue
        name = r.tool_name if isinstance(r.tool_name, str) and r.tool_name else (
            r.llm_name if isinstance(r.llm_name, str) and r.llm_name else None)
        if not name:
            continue
        durs.setdefault(name, []).append(float(r.event_timestamp) - float(t0))
    avg_dur = {k: st.mean(v) for k, v in durs.items() if v}
    return per_rep, avg_dur


def _stats(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if not vals:
        return None
    return {"n": len(vals), "mean": st.mean(vals), "sd": st.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "spread": max(vals) - min(vals)}


def aggregate(runs: list[tuple[str, str]]):
    data = {}
    for label, d in runs:
        acc = _load_accuracy(d)
        per_rep, avg_dur = _load_profile(d)
        data[label] = {"accuracy": acc, "per_rep": per_rep, "avg_dur": avg_dur,
                       "summary": {
                           "coarse": _stats([a.get("coarse") for a in acc]),
                           "major": _stats([a.get("major") for a in acc]),
                           "llm_calls": _stats([p["llm_calls"] for p in per_rep]),
                           "prompt_tokens": _stats([p["prompt_tokens"] for p in per_rep]),
                       }}
    return data


def plot_accuracy(data, out):
    labels = list(data)
    fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(labels)), 4.5))
    for i, lab in enumerate(labels):
        acc = data[lab]["accuracy"]
        for metric, dx, color in (("coarse", -0.12, "#4C72B0"), ("major", 0.12, "#C44E52")):
            ys = [a.get(metric) for a in acc if a.get(metric) is not None]
            xs = [i + dx] * len(ys)
            ax.scatter(xs, ys, color=color, s=55, zorder=3,
                       label=metric if i == 0 else None)
            s = _stats(ys)
            if s:
                ax.hlines(s["mean"], i + dx - 0.08, i + dx + 0.08, color=color, lw=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("annotation accuracy vs LuCA atlas")
    ax.set_title("Accuracy consistency across repeated runs")
    ax.set_ylim(0, 1.02)
    ax.legend(title="granularity")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "accuracy_consistency.png"), dpi=150)
    plt.close(fig)


def plot_trajectory(data, out):
    labels = list(data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, 2.4 * len(labels)), 4.5))
    for i, lab in enumerate(labels):
        pr = data[lab]["per_rep"]
        calls = [p["llm_calls"] for p in pr]
        toks = [p["prompt_tokens"] / 1e6 for p in pr]
        ax1.scatter([i] * len(calls), calls, s=55, color="#55A868", zorder=3)
        ax2.scatter([i] * len(toks), toks, s=55, color="#8172B3", zorder=3)
        for ax, ys in ((ax1, calls), (ax2, toks)):
            s = _stats(ys)
            if s:
                ax.hlines(s["mean"], i - 0.1, i + 0.1, color="black", lw=2)
    for ax, ylab, title in ((ax1, "# LLM calls / run", "Trajectory length variance"),
                            (ax2, "prompt tokens / run (millions)", "Token-cost variance")):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trajectory_cost.png"), dpi=150)
    plt.close(fig)


def plot_bottleneck(data, out, top=12):
    labels = list(data)
    fig, axes = plt.subplots(1, len(labels), figsize=(max(6, 5 * len(labels)), 5), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        ad = data[lab]["avg_dur"]
        items = sorted(ad.items(), key=lambda kv: kv[1], reverse=True)[:top]
        names = [k for k, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        colors = ["#C44E52" if n == "Qwen3.6-27B" or "LLM" in n else "#4C72B0" for n in names]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("avg duration (s)")
        ax.set_title(f"{lab}: top ops by avg duration")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bottleneck.png"), dpi=150)
    plt.close(fig)


def write_summary(data, out):
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({k: v["summary"] for k, v in data.items()}, f, indent=2)
    lines = ["# NAT eval aggregate summary", ""]
    for lab, v in data.items():
        s = v["summary"]
        lines.append(f"## {lab}")

        def fmt(m):
            x = s.get(m)
            return (f"mean={x['mean']:.4g} sd={x['sd']:.4g} spread={x['spread']:.4g} "
                    f"(min {x['min']:.4g}, max {x['max']:.4g}, n={x['n']})") if x else "n/a"
        lines += [f"- coarse accuracy: {fmt('coarse')}",
                  f"- major accuracy:  {fmt('major')}",
                  f"- LLM calls/run:   {fmt('llm_calls')}",
                  f"- prompt tokens/run: {fmt('prompt_tokens')}", ""]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL:DIR (repeatable). Default: vLLM-Qwen3.6-27B:nat_luca_out")
    ap.add_argument("--out", default="nat_eval_figs")
    args = ap.parse_args()
    runs = [tuple(r.split(":", 1)) for r in args.run] or \
        [("vLLM-Qwen3.6-27B", "nat_luca_out")]
    os.makedirs(args.out, exist_ok=True)
    data = aggregate(runs)
    plot_accuracy(data, args.out)
    plot_trajectory(data, args.out)
    plot_bottleneck(data, args.out)
    summary = write_summary(data, args.out)
    print(summary)
    print(f"\nFigures + summary written to {args.out}/")


if __name__ == "__main__":
    main()
