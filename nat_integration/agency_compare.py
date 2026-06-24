"""Compare agent *agency* between two NAT eval runs of the same model/task —
typically pre-fix vs post-fix (the spine+agency contract).

Agency is measured by trajectory proxies parsed from each rep's run manifest:
  - investigative run_code   (custom inspection/analysis, the main agency lever)
  - generate_figure          (plots beyond tool-enforced ones)
  - run_deg                  (marker investigation depth)
  - bc_get_panglaodb_*       (external adjudication depth)
  - total tool calls
Controls (must stay in band — agency added, not traded for correctness):
  - finalized (completion), and accuracy is reported separately by aggregate_eval.

Usage (under any python with stdlib):
  python nat_integration/agency_compare.py \
    --pre  Qwen-pre:nat_luca_out_qwen \
    --post Qwen-post:nat_luca_out_qwen_postfix
  (repeat --pre/--post for more models; pairs are matched by label prefix.)
"""
import argparse
import glob
import json
import os
import statistics as st

INVESTIGATIVE = "run_code"
FIG = "generate_figure"
DEG = "run_deg"
PANGLAO_PREFIX = "bc_get_panglaodb"


def _run_dirs(eval_dir: str) -> list[str]:
    """Per-rep scagent run dirs, from the eval's workflow_output.json."""
    p = os.path.join(eval_dir, "workflow_output.json")
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    items = d if isinstance(d, list) else d.get("eval_output_items") or d.get("items") or []
    dirs = []
    for it in items:
        ga = it.get("generated_answer") or it.get("output") or {}
        if isinstance(ga, str):
            try:
                ga = json.loads(ga)
            except Exception:
                ga = {}
        rd = ga.get("run_dir") if isinstance(ga, dict) else None
        if rd and os.path.isdir(rd):
            dirs.append(rd)
    return dirs


def _proxies(run_dir: str) -> dict | None:
    mp = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(mp):
        return None
    d = json.load(open(mp))
    tools = [s.get("tool") for s in d.get("steps_completed", [])]
    return {
        "n_tools": len(tools),
        "investigative_run_code": tools.count(INVESTIGATIVE),
        "generate_figure": tools.count(FIG),
        "run_deg": tools.count(DEG),
        "panglaodb_queries": sum(1 for t in tools if t and t.startswith(PANGLAO_PREFIX)),
        "finalized": int("finalize_annotation" in tools),
    }


def _agg(eval_dir: str) -> dict:
    rows = [p for d in _run_dirs(eval_dir) if (p := _proxies(d))]
    keys = ["n_tools", "investigative_run_code", "generate_figure", "run_deg",
            "panglaodb_queries", "finalized"]
    out = {"n_reps": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows]
        out[k] = {"mean": round(st.mean(vals), 2), "min": min(vals), "max": max(vals)} if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", action="append", default=[], help="LABEL:DIR (pre-fix)")
    ap.add_argument("--post", action="append", default=[], help="LABEL:DIR (post-fix)")
    args = ap.parse_args()
    pre = {r.split(":", 1)[0].rsplit("-", 1)[0]: r.split(":", 1)[1] for r in args.pre}
    post = {r.split(":", 1)[0].rsplit("-", 1)[0]: r.split(":", 1)[1] for r in args.post}

    for model in sorted(set(pre) | set(post)):
        print(f"\n===== {model} =====")
        a = _agg(pre[model]) if model in pre else None
        b = _agg(post[model]) if model in post else None
        keys = ["n_tools", "investigative_run_code", "generate_figure", "run_deg",
                "panglaodb_queries", "finalized"]
        print(f"  {'proxy':24} {'pre(mean)':>12} {'post(mean)':>12}  delta")
        for k in keys:
            pv = a[k]["mean"] if a and a.get(k) else None
            qv = b[k]["mean"] if b and b.get(k) else None
            delta = (round(qv - pv, 2) if pv is not None and qv is not None else "")
            print(f"  {k:24} {str(pv):>12} {str(qv):>12}  {delta:>+}" if isinstance(delta, (int, float))
                  else f"  {k:24} {str(pv):>12} {str(qv):>12}")
        if a:
            print(f"  (pre n_reps={a['n_reps']})", end="")
        if b:
            print(f"  (post n_reps={b['n_reps']})")


if __name__ == "__main__":
    main()
