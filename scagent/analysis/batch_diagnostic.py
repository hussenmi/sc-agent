"""Batch-effect diagnostic for multi-sample single-cell data.

The core is a **gene-first investigation** (``batch_gene_investigation``): find
sample-enriched cluster regions, characterize each with a within-sample identity
DEG, match the same population across samples by shared identity genes, and — as
secondary support — compare matched regions directly and look for a program that
recurs across populations. Sample composition, neighborhood-mixing entropy and
cluster/sample ARI-NMI are kept only as *context*; they never drive the verdict.
The verdict is derived from two independent axes (gene evidence x experimental
design) and is never labeled "conclusive".
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.artifact_docs import ArtifactGroupDoc

# Neighborhood batch-mixing entropy thresholds. Entropy is normalized against the
# *global* batch-proportion ceiling (the value a neighborhood would have if it
# mirrored the dataset-wide batch mix), so these ratios are robust to batch-size
# imbalance. A cell is "segregated" when its neighborhood reaches less than half
# the achievable mixing; the dataset reads as poorly mixed when the mean does.
ENTROPY_DEFAULT_USE_REP = "X_pca"
ENTROPY_DEFAULT_N_NEIGHBORS = 50
ENTROPY_SEGREGATED_CELL_FRACTION = 0.5  # per-cell threshold, as a fraction of the ceiling
ENTROPY_LOW_MIXING_RATIO = 0.5  # dataset-level mean/ceiling below which mixing is "low"

# Cluster<->sample agreement (ARI / NMI). Both are ~0 when clusters are independent of
# sample (well mixed) and rise as clusters come to correspond to individual samples.
# ARI is chance-adjusted; NMI is information-theoretic and tends to read higher, so we
# fire on either. A "high" tier only sharpens the wording. These are advisory: a high
# score can reflect donor/patient-private biology (e.g. donor-specific epithelial states,
# or malignant clones in tumors) as much as a technical batch effect.
CLUSTER_BATCH_ARI_SUPPORT = 0.2
CLUSTER_BATCH_NMI_SUPPORT = 0.3
CLUSTER_BATCH_ARI_HIGH = 0.5
CLUSTER_BATCH_NMI_HIGH = 0.6


def _expression_view(adata):
    """The expression matrix + gene names used for DEGs.

    Prefers ``adata.raw`` (the full-gene log-normalized matrix scanpy stores
    before HVG subsetting), else ``adata.X``.
    """
    if getattr(adata, "raw", None) is not None:
        return adata.raw.X, list(map(str, adata.raw.var_names))
    return adata.X, list(map(str, adata.var_names))


def _matrix_source_label(matrix: Any) -> str:
    """Best-effort label for the expression scale, for provenance ('lognorm'/'counts')."""
    if hasattr(matrix, "data"):
        sample = np.asarray(matrix.data[:100000], dtype=np.float64)
    else:
        sample = np.asarray(matrix, dtype=np.float64).ravel()[:100000]
    if sample.size == 0:
        return "unknown"
    return "counts" if np.allclose(sample, np.round(sample)) else "lognorm"


def _safe_entropy(fractions: Iterable[float]) -> float:
    vals = [float(v) for v in fractions if float(v) > 0]
    if not vals:
        return 0.0
    denom = math.log(len(vals)) if len(vals) > 1 else 1.0
    if denom <= 0:
        return 1.0
    return -sum(v * math.log(v) for v in vals) / denom


def _candidate_condition_keys(adata, batch_key: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return [key for key in explicit if key in adata.obs.columns and key != batch_key]
    patterns = re.compile(
        r"(condition|group|disease|diagnosis|treatment|stim|status|phenotype|time|tissue|organ|sex|genotype)",
        re.I,
    )
    keys: list[str] = []
    for key in adata.obs.columns:
        if key == batch_key:
            continue
        if key.startswith("_") or key in {"leiden", "louvain"}:
            continue
        series = adata.obs[key]
        nunique = int(series.nunique(dropna=True))
        if 1 < nunique <= 25 and patterns.search(str(key)):
            keys.append(str(key))
    return keys[:8]


def _confounding_summary(adata, batch_key: str, condition_keys: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch = adata.obs[batch_key].astype(str)
    for key in condition_keys:
        condition = adata.obs[key].astype(str)
        table = pd.crosstab(batch, condition)
        if table.empty:
            continue
        batch_purity = table.div(table.sum(axis=1), axis=0).max(axis=1)
        condition_purity = table.div(table.sum(axis=0), axis=1).max(axis=0)
        max_batch_purity = float(batch_purity.max())
        median_batch_purity = float(batch_purity.median())
        max_condition_purity = float(condition_purity.max())
        median_condition_purity = float(condition_purity.median())
        confounded = (
            median_batch_purity >= 0.95
            and median_condition_purity >= 0.95
            and table.shape[0] > 1
            and table.shape[1] > 1
        )
        rows.append(
            {
                "condition_key": key,
                "n_conditions": int(table.shape[1]),
                "max_batch_purity": round(max_batch_purity, 4),
                "median_batch_purity": round(median_batch_purity, 4),
                "max_condition_purity": round(max_condition_purity, 4),
                "median_condition_purity": round(median_condition_purity, 4),
                "confounded_with_batch": bool(confounded),
            }
        )
    return rows


def _cluster_sample_composition(adata, batch_key: str, cluster_key: str) -> list[dict[str, Any]]:
    table = pd.crosstab(adata.obs[cluster_key].astype(str), adata.obs[batch_key].astype(str))
    rows: list[dict[str, Any]] = []
    for cluster, counts in table.iterrows():
        total = int(counts.sum())
        if total <= 0:
            continue
        fractions = counts / total
        dominant_batch = str(fractions.idxmax())
        dominant_fraction = float(fractions.max())
        rows.append(
            {
                "cluster": str(cluster),
                "n_cells": total,
                "dominant_batch": dominant_batch,
                "dominant_fraction": round(dominant_fraction, 4),
                "normalized_sample_entropy": round(_safe_entropy(fractions.values), 4),
                "sample_exclusive": bool(dominant_fraction >= 0.98),
                "sample_dominated": bool(dominant_fraction >= 0.80),
                "batch_counts": {str(k): int(v) for k, v in counts.items()},
            }
        )
    return rows


def _neighborhood_batch_mixing(
    adata,
    batch_key: str,
    cluster_key: str,
    state_by_cluster: dict[str, str],
    *,
    use_rep: str,
    n_neighbors: int,
    obs_key: str = "batch_diagnostic_neighborhood_entropy",
) -> dict[str, Any] | None:
    """Per-cell neighborhood batch-mixing entropy on a low-dimensional embedding.

    This is the *continuous* complement to the cluster-composition and UMAP
    centroid checks. Those only register a batch effect that forms discrete
    clusters or pulls a cell type's per-sample centroids apart; they are blind to
    a batch that smears continuously through a shared region without splitting it.
    For every cell we take the Shannon entropy of the batch labels among its k
    nearest neighbors in ``use_rep`` (uncorrected PCA by default) and normalize by
    ``log2(n_batches)`` to land in [0, 1].

    A raw entropy is hard to read because imbalanced batches cannot reach 1.0 even
    when perfectly mixed. So we also compute the *global* ceiling — the entropy a
    neighborhood would have if it mirrored the dataset-wide batch proportions — and
    report ``mixing_ratio = mean / ceiling``. Ratio near 1 means neighborhoods look
    like the whole dataset (well mixed); near 0 means each cell sits among its own
    sample (segregated). Per-cell entropy is also written to ``adata.obs[obs_key]``
    so the agent can paint it on the UMAP and show *where* mixing fails.

    Returns ``None`` when ``use_rep`` is absent (so the diagnostic degrades to its
    discrete checks instead of failing), or a ``{"skipped": True, ...}`` dict when
    the embedding exists but entropy could not be computed.
    """
    if use_rep not in adata.obsm:
        return None
    try:
        from ..batch.entropy import compute_batch_entropy

        ent = compute_batch_entropy(
            adata,
            batch_key=batch_key,
            use_rep=use_rep,
            n_neighbors=n_neighbors,
        )
    except Exception as exc:  # keep the diagnostic alive; entropy is one signal among many
        return {"skipped": True, "reason": str(exc), "use_rep": use_rep}

    per_cell = np.asarray(ent["per_cell_entropy"], dtype=float)
    # Per-cell value kept in obs so a downstream UMAP can show where batches fail to mix.
    adata.obs[obs_key] = per_cell

    # Global ceiling: normalized entropy of the dataset-wide batch proportions. This is
    # the most mixing achievable, so dividing by it corrects for batch-size imbalance.
    batch_labels = adata.obs[batch_key].astype(str).values
    _, global_counts = np.unique(batch_labels, return_counts=True)
    n_batches = len(global_counts)
    global_probs = global_counts / global_counts.sum()
    global_probs = global_probs[global_probs > 0]
    max_entropy = math.log2(n_batches) if n_batches > 1 else 1.0
    global_ceiling = float(-np.sum(global_probs * np.log2(global_probs)) / max_entropy)

    mean_entropy = float(ent["entropy_mean"])
    mixing_ratio = float(mean_entropy / global_ceiling) if global_ceiling > 0 else None

    # Fraction of cells whose neighborhood reaches < half the achievable mixing.
    seg_threshold = ENTROPY_SEGREGATED_CELL_FRACTION * global_ceiling
    fraction_segregated = float(np.mean(per_cell < seg_threshold)) if global_ceiling > 0 else 0.0

    # Per-broad-label means corroborate the UMAP centroid-separation check: a cell type
    # that is both centroid-separated and low-entropy is segregating by sample.
    states = adata.obs[cluster_key].astype(str).map(state_by_cluster).fillna("Unknown")
    batch_series = adata.obs[batch_key].astype(str)
    per_label: list[dict[str, Any]] = []
    for state in sorted(set(states) - {"Unknown"}):
        mask = (states == state).values
        if not mask.any():
            continue
        per_label.append(
            {
                "broad_label": str(state),
                "n_cells": int(mask.sum()),
                "n_batches_present": int(batch_series[mask].nunique()),
                "mean_entropy": round(float(np.mean(per_cell[mask])), 4),
            }
        )
    per_label.sort(key=lambda row: row["mean_entropy"])  # most segregated first

    return {
        "skipped": False,
        "use_rep": use_rep,
        "n_neighbors": int(ent["n_neighbors"]),
        "n_batches": int(n_batches),
        "obs_key": obs_key,
        "mean_entropy": round(mean_entropy, 4),
        "median_entropy": round(float(ent["entropy_median"]), 4),
        "global_ceiling": round(global_ceiling, 4),
        "mixing_ratio": round(mixing_ratio, 4) if mixing_ratio is not None else None,
        "fraction_segregated_cells": round(fraction_segregated, 4),
        "entropy_per_batch": {
            str(k): round(float(v), 4) for k, v in ent["entropy_per_batch"].items()
        },
        "entropy_per_broad_label": per_label,
    }


def _cluster_batch_concordance(adata, batch_key: str, cluster_key: str) -> dict[str, Any]:
    """Global agreement between the clustering and the sample labels (ARI + NMI).

    ARI and NMI compress, into a single number, how strongly the uncorrected clusters
    line up with sample identity — the global counterpart to the per-cluster dominance
    and per-cell entropy checks:

    - ~0   -> clusters are independent of sample; cells from different samples share the
              same clusters (well mixed).
    - high -> clusters largely correspond to individual samples (strong sample structure).

    ARI (adjusted Rand index) is corrected for chance; NMI (normalized mutual information)
    is information-theoretic and tends to read higher, so we report both and act on
    either. A high value is *not* proof of a technical batch effect: donor/patient-private
    biology also drives clusters to track sample. Epithelial cells are a common example
    in any tissue — donor-specific epithelial states and genetic background in normal
    tissue, or malignant clones and CNVs in tumors. Treat it as advisory; do not assume
    the tissue is a tumor.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    batch = adata.obs[batch_key].astype(str).to_numpy()
    cluster = adata.obs[cluster_key].astype(str).to_numpy()
    ari = float(adjusted_rand_score(batch, cluster))
    nmi = float(normalized_mutual_info_score(batch, cluster))
    if ari >= CLUSTER_BATCH_ARI_HIGH or nmi >= CLUSTER_BATCH_NMI_HIGH:
        interpretation = "clusters largely correspond to individual samples"
    elif ari >= CLUSTER_BATCH_ARI_SUPPORT or nmi >= CLUSTER_BATCH_NMI_SUPPORT:
        interpretation = "clusters partly track sample labels"
    else:
        interpretation = "clusters are largely independent of sample (well mixed)"
    return {
        "cluster_key": cluster_key,
        "batch_key": batch_key,
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "tracks_sample": bool(ari >= CLUSTER_BATCH_ARI_SUPPORT or nmi >= CLUSTER_BATCH_NMI_SUPPORT),
        "interpretation": interpretation,
    }


def _batch_diagnostic_group_doc(params: dict[str, Any]) -> ArtifactGroupDoc:
    """Static documentation for the five batch-diagnostic CSVs.

    Authored next to the tool because it documents this tool's own output; the
    dataset-specific interpretation is left to the model.
    """
    from ..core.artifact_docs import ArtifactGroupDoc, FileDoc

    return ArtifactGroupDoc(
        group="diagnose_batch_effect",
        title="Batch-effect diagnostic — how to read these files",
        overview=(
            "A gene-first check of whether an uncorrected multi-sample dataset carries "
            "sample-associated expression differences, and if so whether they recur across "
            "cell populations. The primary evidence is the within-sample DEGs; the direct "
            "cross-sample comparison and recurrence are secondary. None of these tables prove "
            "a difference is technical rather than real per-sample biology — only the "
            "experimental design can separate those, and q-values here rank cell-level "
            "separation, not replicate-level biology (cells are not independent replicates)."
        ),
        params=params,
        files=[
            FileDoc(
                filename="batch_diagnostic_sample_enriched_regions.csv",
                purpose=(
                    "Which cluster-sample regions hold far more of a sample than its overall "
                    "size predicts — where to look. Enrichment over baseline, not raw purity."
                ),
                computation=(
                    "For each cluster and sample: the fraction of the cluster made of that "
                    "sample, divided by the sample's fraction of the whole dataset. Kept when "
                    "the region has enough cells and enrichment above the threshold."
                ),
                columns={
                    "cluster": "Cluster id.",
                    "sample": "Sample/batch.",
                    "n_cells": "Cells of this sample in this cluster.",
                    "n_cluster": "Total cells in the cluster.",
                    "frac_of_cluster": "Fraction of the cluster that is this sample.",
                    "sample_baseline_frac": "This sample's fraction of the whole dataset.",
                    "enrichment": "frac_of_cluster / sample_baseline_frac (2 = twice expected).",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_within_sample_degs.csv",
                purpose=(
                    "PRIMARY evidence: within each sample, the genes that identify a region "
                    "compared with the rest of that same sample (batch held constant)."
                ),
                computation=(
                    "For each enriched region, a differential-expression test of that cluster's "
                    "cells against all other cells of the SAME sample. Because both sides are "
                    "one sample, the genes describe the population's identity, not batch."
                ),
                how_to_read=(
                    "expression_effect is the primary effect (mean_target - mean_reference on the "
                    "matrix_source scale); higher_in names the side. engine_log2fc is the test's "
                    "raw fold-change, secondary. q_value ranks genes; it is not replicate evidence."
                ),
                columns={
                    "region": "cluster/sample being characterized.",
                    "comparison": "Exactly what was compared (region vs rest of its sample).",
                    "target_cells": "Cells in the region.",
                    "reference_cells": "Other cells of the same sample.",
                    "gene": "Gene symbol.",
                    "higher_in": "Which side the gene is higher in (region or rest of sample).",
                    "expression_effect": "mean_target - mean_reference (primary, oriented effect).",
                    "q_value": "BH-adjusted p-value (ranking aid; small values in scientific notation).",
                    "engine_log2fc": "The DE engine's raw log-fold-change (secondary).",
                    "pct_expressed_target": "Fraction of region cells expressing the gene.",
                    "pct_expressed_reference": "Fraction of the same-sample reference expressing it.",
                    "de_engine": "diffxpy or scanpy (the engine that actually ran).",
                    "de_test": "rank / wilcoxon / wald.",
                    "matrix_source": "Expression scale used (lognorm or counts).",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_population_pairs.csv",
                purpose=(
                    "The cross-sample region pairs we investigated: nominated as looking like the "
                    "same cell type, then confirmed (or not) by their within-sample identity genes."
                ),
                computation=(
                    "Candidate pairs are first nominated cheaply by mean-expression profile "
                    "similarity (profile_correlation) between regions from DIFFERENT samples; the "
                    "top few non-redundant candidates are then given within-sample identity DEGs, "
                    "and the overlap of their top identity genes is reported. Housekeeping genes "
                    "are excluded from the numerical overlap only."
                ),
                how_to_read=(
                    "identity_match_supported=True means the shared identity genes are enough to "
                    "treat the pair as the same population worth comparing directly — NOT that the "
                    "two populations are definitively identical."
                ),
                columns={
                    "cluster_a": "First region's cluster.", "sample_a": "First region's sample.",
                    "cluster_b": "Second region's cluster.", "sample_b": "Second region's sample.",
                    "profile_correlation": "How similar the two regions' mean-expression profiles are (nomination signal).",
                    "n_shared_top25": "Shared genes among the top-25 identity genes.",
                    "n_shared_top50": "Shared genes among the top-50 identity genes.",
                    "jaccard_top50": "Jaccard overlap of the top-50 identity genes.",
                    "identity_match_supported": "Whether the identity-gene overlap confirms them as one population.",
                    "shared_top25_genes": "The actual shared top-25 identity genes.",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_direct_pair_degs.csv",
                purpose=(
                    "SECONDARY: for each matched pair, how the two regions differ across samples, "
                    "gene by gene. Characterizes the difference; does not outrank the within-sample "
                    "evidence and is not proof of a technical cause."
                ),
                computation=(
                    "A differential-expression test of region A's cells vs region B's cells "
                    "directly. ALL genes are kept (stress / mitochondrial / ribosomal / ambient "
                    "genes may be the informative ones)."
                ),
                columns={
                    "cluster_a": "Region A cluster.", "sample_a": "Region A sample.",
                    "cluster_b": "Region B cluster.", "sample_b": "Region B sample.",
                    "gene": "Gene symbol.",
                    "higher_in": "The sample the gene is higher in.",
                    "expression_effect": "mean(sample_a) - mean(sample_b) (oriented effect).",
                    "q_value": "BH-adjusted p-value (ranking aid, not replicate evidence).",
                    "pct_expressed_sample_a": "Fraction of region-A cells expressing the gene.",
                    "pct_expressed_sample_b": "Fraction of region-B cells expressing the gene.",
                    "de_engine": "diffxpy or scanpy.",
                    "de_test": "rank / wilcoxon / wald.",
                    "matrix_source": "Expression scale used.",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_design_check.csv",
                purpose=(
                    "Whether the sample variable can be separated from condition/donor/treatment. "
                    "If not, technical and biological effects cannot be told apart."
                ),
                computation=(
                    "Cross-tabulation of sample against each candidate condition column; how "
                    "cleanly each maps to the other. Missing metadata is recorded as status "
                    "'unknown' rather than as 'not confounded'."
                ),
                how_to_read=(
                    "status: 'unknown' = no design metadata to test; 'confounded' = sample and "
                    "condition are largely redundant (cannot separate technical from biology); "
                    "'orthogonal' = a condition column exists and is not confounded (but that alone "
                    "does not make sample-wide differences technical — donor effects can remain)."
                ),
                columns={
                    "condition_key": "The obs column tested against sample (or '(none available)').",
                    "n_conditions": "Number of distinct condition values.",
                    "median_batch_purity": "How cleanly conditions map to a single sample.",
                    "median_condition_purity": "How cleanly samples map to a single condition.",
                    "confounded_with_batch": "True if redundant with sample (blank when unknown).",
                    "status": "unknown / confounded / orthogonal.",
                },
            ),
        ],
    )


def _write_outputs(
    output_dir: str | None,
    tables: dict[str, pd.DataFrame],
    group_doc: ArtifactGroupDoc | None = None,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if not output_dir:
        return artifacts
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Per-file authored docs keyed by both full filename and stem.
    file_docs: dict[str, Any] = {}
    if group_doc is not None:
        for fdoc in group_doc.files:
            file_docs[fdoc.filename] = fdoc
            file_docs[Path(fdoc.filename).stem] = fdoc

    for stem, df in tables.items():
        path = root / f"{stem}.csv"
        df.to_csv(path, index=False)
        metadata: dict[str, Any] = {"kind": stem}
        doc_for_stem = file_docs.get(stem) or file_docs.get(f"{stem}.csv")
        if doc_for_stem is not None:
            from ..core.artifact_docs import artifact_column_metadata

            metadata.update(artifact_column_metadata(doc_for_stem, df))
        artifacts.append({"path": str(path), "role": "artifact", "metadata": metadata})

    # A single README documenting every file, plus a placeholder Interpretation
    # section the model fills in later via annotate_artifact_group.
    if group_doc is not None:
        from ..core.artifact_docs import write_group_doc

        frames = {f"{stem}.csv": df for stem, df in tables.items()}
        readme_path = write_group_doc(root, group_doc, frames=frames)
        artifacts.append(
            {
                "path": str(readme_path),
                "role": "artifact_readme",
                "metadata": {"kind": "artifact_readme", "group": group_doc.group},
            }
        )
    return artifacts


# ---------------------------------------------------------------------------
# Artifact tables — each answers one distinct question (see the group doc)
# ---------------------------------------------------------------------------

def _engine_test(engine: str) -> tuple[str, str]:
    """Split an engine tag into (engine, test).

    Tags: 'diffxpy_<test>'; 'scanpy_wilcoxon'; or
    'scanpy_wilcoxon_diffxpy_unavailable' (diffxpy was requested but the env was
    not available, so it visibly fell back to Wilcoxon). The de_engine column
    records the ran engine ('scanpy'); the full tag is preserved in
    de_engines_used so the requested-but-unavailable fallback stays visible.
    """
    if engine and engine.startswith("diffxpy_"):
        return "diffxpy", engine.split("_", 1)[1]
    if engine and engine.startswith("scanpy_wilcoxon"):
        return "scanpy", "wilcoxon"
    return str(engine), ""


def _regions_table(regions: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(regions)


def _within_sample_deg_table(
    investigation: dict[str, Any], matrix_source: str, *, top_per_region: int = 50
) -> pd.DataFrame:
    """Every selected region's top identity genes vs the rest of its own sample."""
    from .batch_gene_investigation import top_positive_genes

    rows: list[dict[str, Any]] = []
    for (cluster, sample), deg in investigation["identity_degs"].items():
        if deg is None:
            continue
        engine, test = _engine_test(str(deg.attrs.get("engine", "")))
        n_t = int(deg.attrs.get("n_target", 0))
        n_r = int(deg.attrs.get("n_reference", 0))
        region = f"{cluster}/{sample}"
        top_genes = top_positive_genes(deg, top_per_region, exclude_nuisance=False, min_effect=0.0)
        sub = deg[deg["gene"].isin(top_genes)]
        for _, g in sub.iterrows():
            rows.append(
                {
                    "region": region,
                    "comparison": f"{region} vs rest of {sample}",
                    "target_cells": n_t,
                    "reference_cells": n_r,
                    "gene": g["gene"],
                    "higher_in": region if g["expression_effect"] > 0 else f"rest of {sample}",
                    "expression_effect": round(float(g["expression_effect"]), 4),
                    "q_value": float(g["qval"]),
                    "engine_log2fc": round(float(g.get("engine_log2fc", float("nan"))), 4),
                    "pct_expressed_target": round(float(g["pct_target"]), 4),
                    "pct_expressed_reference": round(float(g["pct_reference"]), 4),
                    "de_engine": engine,
                    "de_test": test,
                    "matrix_source": matrix_source,
                }
            )
    return pd.DataFrame(rows)


def _population_pairs_table(pairs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in pairs:
        rows.append(
            {
                "cluster_a": p["cluster_a"], "sample_a": p["sample_a"],
                "cluster_b": p["cluster_b"], "sample_b": p["sample_b"],
                "profile_correlation": p.get("profile_correlation"),
                "n_shared_top25": p["n_shared_top25"],
                "n_shared_top50": p["n_shared_top50"],
                "jaccard_top50": p["jaccard_top50"],
                "identity_match_supported": p["identity_match_supported"],
                "shared_top25_genes": ", ".join(p["shared_top25_genes"]),
            }
        )
    return pd.DataFrame(rows)


def _direct_pair_table(
    direct_results: list[dict[str, Any]], matrix_source: str, *, top_each: int = 25
) -> pd.DataFrame:
    """Genes separating each matched pair across samples (secondary evidence)."""
    rows: list[dict[str, Any]] = []
    for r in direct_results:
        engine, test = _engine_test(str(r.get("engine", "")))
        deg = r["deg"]
        up_a = deg[deg["expression_effect"] > 0].sort_values("expression_effect", ascending=False).head(top_each)
        up_b = deg[deg["expression_effect"] < 0].sort_values("expression_effect").head(top_each)
        for side, g in [("a", row) for _, row in up_a.iterrows()] + [("b", row) for _, row in up_b.iterrows()]:
            higher = r["sample_a"] if side == "a" else r["sample_b"]
            rows.append(
                {
                    "cluster_a": r["cluster_a"], "sample_a": r["sample_a"],
                    "cluster_b": r["cluster_b"], "sample_b": r["sample_b"],
                    "gene": g["gene"],
                    "higher_in": higher,
                    "expression_effect": round(float(g["expression_effect"]), 4),
                    "q_value": float(g["qval"]),
                    "pct_expressed_sample_a": round(float(g["pct_target"]), 4),
                    "pct_expressed_sample_b": round(float(g["pct_reference"]), 4),
                    "de_engine": engine,
                    "de_test": test,
                    "matrix_source": matrix_source,
                }
            )
    return pd.DataFrame(rows)


def _design_check_table(
    confounding: list[dict[str, Any]], condition_cols: list[str]
) -> pd.DataFrame:
    """Whether sample can be separated from each condition/covariate.

    Missing metadata is recorded as an explicit ``status='unknown'`` row, never as
    ``confounded=False`` (absence of a test is not evidence of no confounding).
    """
    if not condition_cols:
        return pd.DataFrame(
            [
                {
                    "condition_key": "(none available)",
                    "n_conditions": 0,
                    "median_batch_purity": None,
                    "median_condition_purity": None,
                    "confounded_with_batch": None,
                    "status": "unknown",
                }
            ]
        )
    rows = []
    for row in confounding:
        status = "confounded" if row.get("confounded_with_batch") else "orthogonal"
        rows.append(
            {
                "condition_key": row["condition_key"],
                "n_conditions": row["n_conditions"],
                "median_batch_purity": row["median_batch_purity"],
                "median_condition_purity": row["median_condition_purity"],
                "confounded_with_batch": row["confounded_with_batch"],
                "status": status,
            }
        )
    return pd.DataFrame(rows)


_VERDICT_PLAIN = {
    "cannot_determine_technical_vs_biological": (
        "the technical-versus-biological origin of these differences cannot be "
        "determined from the genes alone, so the dataset should not be integrated "
        "automatically"
    ),
    "do_not_integrate_based_on_current_evidence": (
        "the current gene evidence does not justify integrating the dataset"
    ),
    "integration_optional_if_replicates": (
        "integration may be reasonable only if the samples are intended as "
        "comparable replicates"
    ),
    "integration_supported": (
        "the recurring, design-documented technical differences support integrating "
        "the samples"
    ),
}


def _deterministic_interpretation(
    investigation: dict[str, Any],
    gene_evidence: str,
    design_interpretation: str,
    verdict: dict[str, str],
    batch_key: str,
) -> str:
    """Readable prose summary of the results, built from the structured evidence.

    Written into the README so the artifact reads on its own; the model may still
    expand it. No hardcoded biology — every gene name comes from the results.
    """
    pairs = investigation.get("selected_pairs") or []
    direct_by = {
        (d["cluster_a"], d["sample_a"], d["cluster_b"], d["sample_b"]): d
        for d in investigation.get("direct_results") or []
    }
    n_regions = len(investigation.get("regions") or [])

    paras: list[str] = []
    paras.append(
        f"This diagnostic examined {n_regions} sample-enriched cluster region(s) and "
        f"investigated {len(pairs)} cross-sample population pair(s) whose mean-expression "
        f"profiles looked alike. For each pair it compared the cluster against the rest of "
        f"its own sample (holding {batch_key} constant) to describe the population, then "
        f"compared the two matched regions directly."
    )

    for p in pairs:
        key = (p["cluster_a"], p["sample_a"], p["cluster_b"], p["sample_b"])
        d = direct_by.get(key)
        shared = ", ".join(p.get("shared_top25_genes", [])[:8]) or "shared identity genes"
        sent = (
            f"Cluster {p['cluster_a']} in {p['sample_a']} and cluster {p['cluster_b']} in "
            f"{p['sample_b']} share {p.get('n_shared_top25', 0)} of their top identity genes "
            f"({shared}), so they appear to be the same population present in both samples."
        )
        if d is not None:
            hi_a = ", ".join(d["higher_in_a"][:6]) or "n/a"
            hi_b = ", ".join(d["higher_in_b"][:6]) or "n/a"
            sent += (
                f" Comparing them directly, {d['sample_a']} is higher for {hi_a}, while "
                f"{d['sample_b']} is higher for {hi_b}. This describes how the regions differ; "
                f"it does not by itself show the difference is technical."
            )
        paras.append(sent)

    recurrent = investigation.get("recurrent_programs") or []
    if recurrent:
        by_group: dict[str, list[str]] = {}
        for r in recurrent:
            by_group.setdefault(r["associated_batch_group"], []).append(r["gene"])
        bits = [
            f"a {group}-associated program ({', '.join(gs[:8])})"
            for group, gs in by_group.items()
        ]
        paras.append(
            "The same sample-associated shift recurs across more than one population: "
            + "; ".join(bits)
            + ". A program that recurs across populations points to a sample-wide effect, "
            "but sample-wide is not the same as technical — a real systemic biological "
            "difference would look the same."
        )
    else:
        paras.append(
            "No sample-associated program recurred across more than one population, so any "
            "differences appear localized rather than sample-wide."
        )

    design_plain = {
        "unknown": "no experimental-design metadata is available to test whether sample is "
                   "confounded with a biological condition",
        "confounded_with_biology": "the sample variable is confounded with a biological "
                                   "condition, so technical and biological effects cannot be separated",
        "orthogonal_but_not_known_technical": "a condition column exists and is not confounded "
                                              "with sample, but that alone does not make the differences technical",
        "documented_technical_batch": "the design documents this as a technical batch variable "
                                      "separable from biological condition",
    }.get(design_interpretation, design_interpretation)
    paras.append(
        f"On the experimental design, {design_plain}. Taken together, "
        f"{_VERDICT_PLAIN.get(verdict['recommendation'], verdict['recommendation'])}. "
        "Note that the q-values here rank cell-level separation and are not replicate-level "
        "evidence, because the cells are not independent replicates; weigh the expression "
        "effects, percent-expressed, recurrence, and study design instead."
    )
    return "\n\n".join(paras)


def _fill_readme_interpretation(artifacts: list[dict[str, Any]], interpretation: str) -> None:
    """Replace the README's Interpretation section with deterministic prose."""
    from ..core.artifact_docs import set_interpretation

    readme = next(
        (a["path"] for a in artifacts if a.get("role") == "artifact_readme"), None
    )
    if not readme:
        return
    p = Path(readme)
    try:
        p.write_text(set_interpretation(p.read_text(), interpretation))
    except OSError:
        pass


def diagnose_batch_effect(
    adata,
    *,
    batch_key: str,
    cluster_key: str = "leiden",
    condition_keys: list[str] | None = None,
    min_cells_per_cluster_sample: int = 30,
    n_top_genes: int = 25,
    entropy_use_rep: str = ENTROPY_DEFAULT_USE_REP,
    entropy_n_neighbors: int = ENTROPY_DEFAULT_N_NEIGHBORS,
    output_dir: str | None = None,
    prefer_diffxpy: bool = False,
    min_enrichment: float = 2.0,
    technical_batch_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Gene-first batch-effect diagnostic with a two-axis, design-gated verdict.

    The gene investigation (``batch_gene_investigation``) is the evidence; sample
    composition, neighborhood entropy and cluster/sample ARI-NMI are secondary
    context only. ``technical_batch_keys`` lets a caller assert which columns are
    documented technical batch variables separable from biology; the verdict only
    reaches ``integration_supported`` when ``batch_key`` itself is among them (a
    non-empty list of unrelated keys does not count). The harness never infers
    "technical" on its own.
    """
    from .batch_gene_investigation import (
        build_terminal_summary,
        derive_design_interpretation,
        derive_gene_evidence,
        derive_verdict,
        run_gene_investigation,
    )

    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs.")
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"cluster_key '{cluster_key}' not found in adata.obs. Run clustering first.")
    batches = list(map(str, adata.obs[batch_key].dropna().astype(str).unique()))
    if len(batches) < 2:
        raise ValueError(f"batch_key '{batch_key}' has fewer than two groups.")

    matrix, genes = _expression_view(adata)
    matrix_source = _matrix_source_label(matrix)

    # --- gene evidence (steps 1-5) ---
    investigation = run_gene_investigation(
        adata, matrix, genes,
        batch_key=batch_key, cluster_key=cluster_key,
        prefer_diffxpy=prefer_diffxpy,
        min_cells=min_cells_per_cluster_sample,
        min_enrichment=min_enrichment,
    )

    # --- design gate ---
    condition_cols = _candidate_condition_keys(adata, batch_key, condition_keys)
    confounding = _confounding_summary(adata, batch_key, condition_cols)
    # A documented technical batch requires the caller to name THIS batch variable
    # specifically — not merely to pass some non-empty list of unrelated keys.
    technical_documented = batch_key in (technical_batch_keys or [])
    gene_evidence = derive_gene_evidence(investigation)
    design_interpretation = derive_design_interpretation(
        confounding,
        condition_columns_present=bool(condition_cols),
        technical_batch_documented=technical_documented,
    )
    verdict = derive_verdict(gene_evidence, design_interpretation)

    # --- secondary context (never drives the verdict) ---
    composition = _cluster_sample_composition(adata, batch_key, cluster_key)
    clusters = [str(c) for c in adata.obs[cluster_key].astype(str).unique()]
    per_cluster_label = {c: f"cluster {c}" for c in clusters}
    entropy_mixing = _neighborhood_batch_mixing(
        adata, batch_key, cluster_key, per_cluster_label,
        use_rep=entropy_use_rep, n_neighbors=entropy_n_neighbors,
    )
    concordance = _cluster_batch_concordance(adata, batch_key, cluster_key)

    # --- artifact tables ---
    tables = {
        "batch_diagnostic_sample_enriched_regions": _regions_table(investigation["regions"]),
        "batch_diagnostic_within_sample_degs": _within_sample_deg_table(investigation, matrix_source),
        "batch_diagnostic_population_pairs": _population_pairs_table(investigation["population_pairs"]),
        "batch_diagnostic_direct_pair_degs": _direct_pair_table(investigation["direct_results"], matrix_source),
        "batch_diagnostic_design_check": _design_check_table(confounding, condition_cols),
    }
    artifacts = _write_outputs(
        output_dir, tables,
        group_doc=_batch_diagnostic_group_doc(
            {
                "batch_key": batch_key, "cluster_key": cluster_key,
                "de_engine": (
                    "scanpy Wilcoxon (default, in-process). diffxpy is opt-in "
                    "(prefer_diffxpy=True) and runs the same rank test through its engine; "
                    "the diffxpy bridge also offers an NB Wald count model, not used here"
                ),
                "engines_used": ", ".join(investigation["engines_used"]) or "n/a",
                "matrix_source": matrix_source,
                "min_cells_per_region": min_cells_per_cluster_sample,
                "min_enrichment": min_enrichment,
                "condition_keys_tested": condition_cols or [],
            }
        ),
    )
    # Fill the README's Interpretation section with deterministic prose so the
    # artifact reads without depending on the model (the model may still expand it).
    _fill_readme_interpretation(
        artifacts,
        _deterministic_interpretation(
            investigation, gene_evidence, design_interpretation, verdict, batch_key
        ),
    )

    # --- deterministic, already-readable terminal summary ---
    terminal_summary = build_terminal_summary(
        investigation, gene_evidence, design_interpretation, verdict
    )
    terminal_summary.append(
        "Note: q-values rank cell-level separation and, because cells are not "
        "independent replicates, are NOT sample-level replication evidence. Weigh "
        "expression effect, percent-expressed, recurrence and study design instead."
    )

    result = {
        "status": "ok",
        "tool": "diagnose_batch_effect",
        "batch_key": batch_key,
        "cluster_key": cluster_key,
        "n_batches": int(len(batches)),
        "matrix_source": matrix_source,
        "de_engines_used": investigation["engines_used"],
        "gene_evidence": gene_evidence,
        "design_interpretation": design_interpretation,
        "recommendation": verdict["recommendation"],
        "recommendation_reason": verdict["reason"],
        "terminal_summary": terminal_summary,
        "sample_enriched_regions": investigation["regions"][:50],
        "population_pairs": [
            {k: v for k, v in p.items() if k != "shared_top50_genes"}
            for p in investigation["population_pairs"][:30]
        ],
        "selected_pairs": [
            {"cluster_a": p["cluster_a"], "sample_a": p["sample_a"],
             "cluster_b": p["cluster_b"], "sample_b": p["sample_b"],
             "n_shared_top25": p["n_shared_top25"],
             "shared_top25_genes": p["shared_top25_genes"][:15]}
            for p in investigation["selected_pairs"]
        ],
        "direct_pair_summaries": [
            {"cluster_a": r["cluster_a"], "sample_a": r["sample_a"],
             "cluster_b": r["cluster_b"], "sample_b": r["sample_b"],
             "engine": r["engine"],
             "higher_in_a": r["higher_in_a"][:15], "higher_in_b": r["higher_in_b"][:15]}
            for r in investigation["direct_results"]
        ],
        "recurrent_programs": investigation["recurrent_programs"][:50],
        # secondary context
        "context": {
            "cluster_sample_composition": composition[:30],
            "neighborhood_batch_entropy": entropy_mixing,
            "cluster_batch_concordance": concordance,
            "condition_confounding": confounding,
        },
        "evidence_limits": [
            "This is a descriptive gene diagnostic, not proof a sample-associated difference is technical.",
            "Matching within-sample identity genes support a candidate population match; they do not prove definitive identity.",
            "A recurring program is sample-wide, which is not the same as technical; only experimental design separates the two.",
            "q-values rank cell-level separation and are not replicate-level evidence (cells are not independent samples).",
            "Context signals (composition, entropy, ARI/NMI) are advisory and do not drive the recommendation.",
        ],
        "artifacts_created": artifacts,
    }
    adata.uns["batch_effect_diagnostic"] = result
    return result
