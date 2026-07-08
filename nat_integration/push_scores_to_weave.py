"""Attach our LuCA accuracy scores to the matching Weave trace calls.

NAT's eval scores (from the `luca_atlas` evaluator) live in luca_atlas_output.json,
separate from the telemetry trace exported to Weave. This script joins them: for
each eval item it finds the Weave WORKFLOW_START call (matched by the unique
run_dir in the workflow output) and attaches the accuracy via apply_scorer — so it
renders in the call's **Scores** tab — plus a feedback note as a Feedback-tab backup.

Run under the nat venv with WANDB_API_KEY exported:
  export WANDB_API_KEY=$(grep ^WANDB_API_KEY= .env | cut -d= -f2- | tr -d '"')
  /usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/python nat_integration/push_scores_to_weave.py \
    --project scagent-nat-luca --eval-out nat_luca_out
"""

import argparse
import asyncio
import json
import os
import re

import weave

_ACC_RE = re.compile(r"acc\[(\w+)\]=([0-9.]+)")
_WORKFLOW_OP = "WORKFLOW_START.scagent_analyze"


class LucaAtlasScore(weave.Scorer):
    """A precomputed scorer: returns the accuracy we already computed (no recompute)."""
    coarse_accuracy: float
    major_accuracy: float

    @weave.op()
    def score(self, output) -> dict:  # output is the call's output; unused (precomputed)
        return {"coarse_accuracy": self.coarse_accuracy, "major_accuracy": self.major_accuracy}


def _load_scores(eval_out: str) -> dict:
    """id -> {major, coarse} from luca_atlas_output.json."""
    d = json.load(open(os.path.join(eval_out, "luca_atlas_output.json")))
    out = {}
    for it in d.get("eval_output_items", []):
        accs = {("coarse" if "coarse" in k else "major" if "major" in k else k): float(v)
                for k, v in _ACC_RE.findall(str(it.get("reasoning", "")))}
        out[it["id"]] = {"major": accs.get("major", it.get("score")), "coarse": accs.get("coarse")}
    return out


def _load_run_dirs(eval_out: str) -> dict:
    """id -> run_dir, parsed from workflow_output.json generated_answer."""
    items = json.load(open(os.path.join(eval_out, "workflow_output.json")))
    out = {}
    for it in items:
        try:
            ga = json.loads(it["generated_answer"])
            out[it["id"]] = ga.get("run_dir")
        except Exception:
            pass
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="scagent-nat-luca")
    ap.add_argument("--eval-out", default="nat_luca_out")
    args = ap.parse_args()

    scores = _load_scores(args.eval_out)
    run_dirs = _load_run_dirs(args.eval_out)
    client = weave.init(args.project)

    # Index the project's workflow calls by the run_dir embedded in their output.
    calls = [c for c in client.get_calls(limit=500)
             if _WORKFLOW_OP in str(getattr(c, "op_name", ""))]
    by_run_dir = {}
    for c in calls:
        s = str(getattr(c, "output", ""))
        for _id, rd in run_dirs.items():
            if rd and rd in s:
                by_run_dir[rd] = c

    n = 0
    for _id, acc in scores.items():
        rd = run_dirs.get(_id)
        call = by_run_dir.get(rd)
        if call is None:
            print(f"  [skip] {_id}: no Weave call matched run_dir={rd}")
            continue
        scorer = LucaAtlasScore(coarse_accuracy=float(acc.get("coarse") or 0.0),
                                major_accuracy=float(acc.get("major") or 0.0))
        await call.apply_scorer(scorer)
        # Feedback-tab backup (human-readable).
        call.feedback.add("luca_atlas_accuracy",
                          {"major": acc.get("major"), "coarse": acc.get("coarse")})
        print(f"  [ok] {_id}: major={acc.get('major')} coarse={acc.get('coarse')} -> {call.ui_url}")
        n += 1
    print(f"\nAttached scores to {n}/{len(scores)} calls in project '{args.project}'.")


if __name__ == "__main__":
    asyncio.run(main())
