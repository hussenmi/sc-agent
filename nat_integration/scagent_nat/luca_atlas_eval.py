"""NAT evaluator that wraps scagent's own LuCA atlas scorer.

Per the division of labor for the NVIDIA collab: NAT owns profiling/serving/
trajectory, but accuracy is scored by our richer, already-validated
`compare_run_to_atlas.py` (majority-mapped annotation accuracy + ARI/NMI + purity
vs the LuCA atlas truth, joined by barcode). Rather than reimplement a thinner
metric inside NAT (the `annotation_ari` evaluator does that), this evaluator
*invokes the canonical scorer* and surfaces its result as a NAT eval score — so
accuracy shows up in NAT's eval artifacts while the logic stays in one place.

The scorer needs anndata 0.12 (the run h5ad), which lives in the `scagent_rapids`
env, not this py3.11 NAT venv — so it runs as a cross-env subprocess.

`expected_output` format in the dataset: "<truth_csv>::<study>" (study optional).
`output` is the JSON string returned by the scagent_analyze workflow.
"""

import json
import os
import subprocess
import tempfile

from nat.builder.builder import Builder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.plugins.eval.data_models.evaluator_io import EvalOutput, EvalOutputItem


class LucaAtlasConfig(EvaluatorBaseConfig, name="luca_atlas"):
    """Score = scagent annotation accuracy vs LuCA atlas (via compare_run_to_atlas.py)."""
    scorer_python: str = "/usersoftware/peerd/ibrahih3/envs/scagent_rapids/bin/python"
    scorer_script: str = "/home/ibrahih3/luca_bench/compare_run_to_atlas.py"
    annotation_col: str = "cell_type"
    # Which accuracy to report as the NAT score: "major" (harder, e.g. 24-class)
    # or "coarse" (e.g. 7-lineage). Full metrics go into the reasoning string.
    primary_metric: str = "major"
    timeout_s: int = 1800


def _run_scorer(pred_h5ad: str, truth_csv: str, study: str | None, cfg: LucaAtlasConfig) -> dict:
    """Invoke the canonical scorer in the rapids env; return its metrics.json dict."""
    with tempfile.TemporaryDirectory(prefix="luca_eval_") as tmp:
        cmd = [cfg.scorer_python, cfg.scorer_script,
               "--h5ad", pred_h5ad, "--truth", truth_csv,
               "--annotation-col", cfg.annotation_col, "--out", tmp]
        if study:
            cmd += ["--study", study]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_s)
        metrics_path = os.path.join(tmp, "metrics.json")
        if not os.path.exists(metrics_path):
            tail = (proc.stderr or proc.stdout or "")[-500:]
            raise RuntimeError(f"scorer produced no metrics.json (rc={proc.returncode}): {tail}")
        with open(metrics_path) as fh:
            return json.load(fh)


def _score_one(output_obj, expected_output_obj, cfg: LucaAtlasConfig):
    """Return (score, reasoning). Raises on hard failure (caught by caller)."""
    out = json.loads(output_obj) if isinstance(output_obj, str) else output_obj
    pred_path = out.get("annotated_h5ad")
    if not pred_path:
        raise ValueError(f"workflow produced no annotated_h5ad (rc={out.get('returncode')})")

    spec = str(expected_output_obj)
    truth_csv, study = (spec.rsplit("::", 1) + [None])[:2] if "::" in spec else (spec, None)

    m = _run_scorer(pred_path, truth_csv, study, cfg)
    acc = m.get("accuracy") or {}
    coarse_ref, major_ref = m.get("coarse_ref"), m.get("major_ref")
    ref = major_ref if cfg.primary_metric == "major" else coarse_ref
    score = acc.get(ref)
    if score is None:
        raise ValueError(f"scorer returned no accuracy for ref '{ref}' (accuracy keys: {list(acc)})")

    reasoning = (
        f"acc[{coarse_ref}]={acc.get(coarse_ref)}, acc[{major_ref}]={acc.get(major_ref)} "
        f"| primary={ref}:{score:.3f} | n_labeled={m.get('n_labeled')}/{m.get('n_cells')} "
        f"| purity_wtd={m.get('purity_size_weighted')}"
    )
    return float(score), reasoning


@register_evaluator(config_type=LucaAtlasConfig)
async def register_luca_atlas(config: LucaAtlasConfig, builder: Builder):
    async def evaluate_fn(eval_input):
        items, scores = [], []
        for it in eval_input.eval_input_items:
            try:
                score, reasoning = _score_one(it.output_obj, it.expected_output_obj, config)
                items.append(EvalOutputItem(id=it.id, score=score, reasoning=reasoning))
                scores.append(score)
            except Exception as e:  # noqa: BLE001
                items.append(EvalOutputItem(id=it.id, score=float("nan"), reasoning="", error=str(e)))
        avg = sum(scores) / len(scores) if scores else float("nan")
        return EvalOutput(average_score=avg, eval_output_items=items)

    yield EvaluatorInfo(config=config, evaluate_fn=evaluate_fn,
                        description="scagent annotation accuracy vs LuCA atlas (compare_run_to_atlas.py)")
