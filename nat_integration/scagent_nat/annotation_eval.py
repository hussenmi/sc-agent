"""NAT evaluator: cell-type annotation quality vs published ground truth.

Thin-slice metric = Adjusted Rand Index (ARI) between scagent's clustering and
the authors' cell-type labels, computed on the barcodes the two share (scagent
filters cells in QC, so we intersect on obs_names). ARI is label-invariant, so
it needs no label-name mapping — ideal as the zero-curation first metric. The
"consistent + overall-right" framing is served by running this across repeats
and backends and comparing the distributions.

Lineage-level accuracy (needs a curated label map) comes next, as a second
registered evaluator; this one stays simple and objective.

`expected_output` format in the dataset: "<ground_truth_h5ad>::<obs_column>".
`output` is the JSON string returned by the scagent_analyze workflow.
"""

import json

from nat.builder.builder import Builder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.plugins.eval.data_models.evaluator_io import EvalOutput, EvalOutputItem


class AnnotationARIConfig(EvaluatorBaseConfig, name="annotation_ari"):
    """Score = ARI(scagent clusters, ground-truth labels) on shared barcodes."""
    cluster_key: str = "leiden"        # scagent's cluster column in the annotated .h5ad
    fallback_key: str = "cell_type"    # used if cluster_key is absent
    min_overlap: int = 50              # below this many shared cells, flag low confidence


def _score_one(output_obj, expected_output_obj, cfg: AnnotationARIConfig):
    """Return (score, reasoning). Raises on hard failure (caught by caller)."""
    import anndata as ad
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    out = json.loads(output_obj) if isinstance(output_obj, str) else output_obj
    pred_path = out.get("annotated_h5ad")
    if not pred_path:
        raise ValueError(f"workflow produced no annotated_h5ad (rc={out.get('returncode')})")

    gt_path, gt_col = str(expected_output_obj).rsplit("::", 1)

    pred = ad.read_h5ad(pred_path, backed="r")
    gt = ad.read_h5ad(gt_path, backed="r")

    key = cfg.cluster_key if cfg.cluster_key in pred.obs.columns else cfg.fallback_key
    if key not in pred.obs.columns:
        raise ValueError(f"neither '{cfg.cluster_key}' nor '{cfg.fallback_key}' in predicted obs")
    if gt_col not in gt.obs.columns:
        raise ValueError(f"ground-truth column '{gt_col}' not in {gt_path}")

    common = pred.obs_names.intersection(gt.obs_names)
    n = len(common)
    if n < cfg.min_overlap:
        return float("nan"), f"only {n} shared barcodes (<{cfg.min_overlap}); cannot score reliably"

    p = pred.obs.loc[common, key].astype(str).to_numpy()
    g = gt.obs.loc[common, gt_col].astype(str).to_numpy()
    ari = float(adjusted_rand_score(g, p))
    nmi = float(normalized_mutual_info_score(g, p))
    reasoning = (f"ARI={ari:.3f}, NMI={nmi:.3f} on {n} shared cells "
                 f"(pred key='{key}', {len(set(p))} clusters vs {len(set(g))} GT labels '{gt_col}')")
    return ari, reasoning


@register_evaluator(config_type=AnnotationARIConfig)
async def register_annotation_ari(config: AnnotationARIConfig, builder: Builder):
    async def evaluate_fn(eval_input):  # nat.data_models.evaluator.EvalInput
        items, scores = [], []
        for it in eval_input.eval_input_items:
            try:
                score, reasoning = _score_one(it.output_obj, it.expected_output_obj, config)
                items.append(EvalOutputItem(id=it.id, score=score, reasoning=reasoning))
                if score == score:  # not NaN
                    scores.append(score)
            except Exception as e:  # noqa: BLE001
                items.append(EvalOutputItem(id=it.id, score=float("nan"), reasoning="", error=str(e)))
        avg = sum(scores) / len(scores) if scores else float("nan")
        return EvalOutput(average_score=avg, eval_output_items=items)

    yield EvaluatorInfo(config=config, evaluate_fn=evaluate_fn,
                        description="Adjusted Rand Index between scagent clustering and ground-truth labels")
