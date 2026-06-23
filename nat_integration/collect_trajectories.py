"""Merge per-run trajectory JSONLs into one distillation/flywheel dataset.

scagent writes <run_name>.trajectory.jsonl (one line per LLM call: input messages
-> assistant action) when SCAGENT_TRAJECTORY_LOG is set (the NAT workflow sets it
when collect_trajectory: true). This gathers them into a single dataset, tagging
each example with its source run, and prints a quick summary.

  /usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/python nat_integration/collect_trajectories.py \
    --src nat_runs --out trajectories/dataset.jsonl
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="nat_runs", help="dir holding *.trajectory.jsonl")
    ap.add_argument("--out", default="trajectories/dataset.jsonl")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.trajectory.jsonl")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_runs = n_ex = 0
    with open(args.out, "w") as out:
        for f in files:
            run_id = os.path.basename(f).replace(".trajectory.jsonl", "")
            rows = 0
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["run_id"] = run_id
                out.write(json.dumps(rec) + "\n")
                rows += 1
            if rows:
                n_runs += 1
                n_ex += rows
                print(f"  {run_id}: {rows} LLM calls")

    print(f"\nMerged {n_ex} examples from {n_runs} runs -> {args.out}")
    if n_ex == 0:
        print("  (none found — run with collect_trajectory: true, or SCAGENT_TRAJECTORY_LOG set)")


if __name__ == "__main__":
    main()
