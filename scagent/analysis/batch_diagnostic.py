"""Descriptive batch-effect diagnostics for multi-sample single-cell data.

This module deliberately avoids heavy integration benchmarks.  It summarizes
whether uncorrected clusters/states separate by sample, whether sample-linked
expression shifts recur across broad cell types, and whether sample is
confounded with condition-like metadata.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import math
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


NUISANCE_PATTERNS = [
    r"^MT-",
    r"^mt-",
    r"^RPL",
    r"^RPS",
    r"^MRPL",
    r"^MRPS",
    r"^MALAT1$",
    r"^HBA",
    r"^HBB",
    r"^HB[ABDEGMQZ]",
    r"\.\d+$",
]

BROAD_MARKER_MODULES: Dict[str, set[str]] = {
    "T cell": {"CD3D", "CD3E", "TRAC", "IL7R", "CD4", "CD8A", "CD8B", "LTB"},
    "B cell": {"MS4A1", "CD79A", "CD79B", "BANK1", "CD74"},
    "Plasma cell": {"MZB1", "XBP1", "JCHAIN", "SDC1", "IGHG1", "IGKC"},
    "NK cell": {"NKG7", "GNLY", "KLRD1", "PRF1", "GZMB", "KLRF1"},
    "Myeloid": {"LYZ", "LST1", "S100A8", "S100A9", "FCGR3A", "CD14", "CTSS"},
    "Dendritic cell": {"FCER1A", "CLEC10A", "CST3", "IRF8", "LILRA4", "CLEC4C"},
    "Epithelial": {"EPCAM", "KRT8", "KRT18", "KRT19", "KRT5", "KRT17"},
    "Endothelial": {"PECAM1", "VWF", "KDR", "CLDN5", "ESAM"},
    "Fibroblast": {"COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "TAGLN"},
    "Cycling": {"MKI67", "TOP2A", "PCLAF", "STMN1", "UBE2C"},
    "Mast cell": {"TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2"},
    "Platelet": {"PPBP", "PF4", "GP9", "ITGA2B", "NRGN"},
    "Erythroid": {"HBA1", "HBA2", "HBB", "GYPA", "ALAS2"},
}

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


def _is_nuisance_gene(gene: str) -> bool:
    return any(re.search(pattern, gene) for pattern in NUISANCE_PATTERNS)


def _to_dense_1d(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    arr = np.asarray(values)
    return np.ravel(arr)


def _mean_expression(matrix: Any) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.array([])
    means = matrix.mean(axis=0)
    return _to_dense_1d(means)


def _expression_view(adata):
    if getattr(adata, "raw", None) is not None:
        return adata.raw.X, list(map(str, adata.raw.var_names))
    return adata.X, list(map(str, adata.var_names))


def _safe_entropy(fractions: Iterable[float]) -> float:
    vals = [float(v) for v in fractions if float(v) > 0]
    if not vals:
        return 0.0
    denom = math.log(len(vals)) if len(vals) > 1 else 1.0
    if denom <= 0:
        return 1.0
    return -sum(v * math.log(v) for v in vals) / denom


def _candidate_condition_keys(adata, batch_key: str, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return [key for key in explicit if key in adata.obs.columns and key != batch_key]
    patterns = re.compile(
        r"(condition|group|disease|diagnosis|treatment|stim|status|phenotype|time|tissue|organ|sex|genotype)",
        re.I,
    )
    keys: List[str] = []
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


def _confounding_summary(adata, batch_key: str, condition_keys: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def _compute_cluster_markers(adata, cluster_key: str, key_added: str, n_top_genes: int) -> Dict[str, List[str]]:
    import scanpy as sc

    if key_added not in adata.uns:
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            method="wilcoxon",
            use_raw=getattr(adata, "raw", None) is not None,
            n_genes=max(n_top_genes, 50),
            key_added=key_added,
        )
    markers: Dict[str, List[str]] = {}
    clusters = list(adata.obs[cluster_key].astype(str).unique())
    for cluster in clusters:
        try:
            df = sc.get.rank_genes_groups_df(adata, group=cluster, key=key_added)
        except Exception:
            markers[cluster] = []
            continue
        genes = [str(g) for g in df.get("names", pd.Series(dtype=str)).head(n_top_genes).tolist()]
        markers[cluster] = [gene for gene in genes if gene and gene.lower() != "nan"]
    return markers


def _broad_label_from_markers(markers: List[str]) -> Dict[str, Any]:
    upper = [gene.upper() for gene in markers if not _is_nuisance_gene(gene)]
    ranked = {gene: rank for rank, gene in enumerate(upper[:50])}
    scores = []
    for label, module in BROAD_MARKER_MODULES.items():
        overlap = sorted(set(ranked) & module, key=lambda gene: ranked[gene])
        if overlap:
            rank_weight = sum(1.0 / (ranked[gene] + 1.0) for gene in overlap)
            scores.append((len(overlap), rank_weight, label, overlap))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 2:
        return {"label": "Unknown", "confidence": "low", "supporting_markers": []}
    confidence = "high" if scores[0][0] >= 4 else "medium"
    return {
        "label": scores[0][2],
        "confidence": confidence,
        "supporting_markers": scores[0][3],
    }


def _cluster_sample_composition(adata, batch_key: str, cluster_key: str) -> List[Dict[str, Any]]:
    table = pd.crosstab(adata.obs[cluster_key].astype(str), adata.obs[batch_key].astype(str))
    rows: List[Dict[str, Any]] = []
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


def _umap_state_separation(adata, state_by_cluster: Dict[str, str], batch_key: str, cluster_key: str) -> List[Dict[str, Any]]:
    if "X_umap" not in adata.obsm:
        return []
    coords = np.asarray(adata.obsm["X_umap"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        return []
    obs = adata.obs[[batch_key, cluster_key]].copy()
    obs["_state"] = obs[cluster_key].astype(str).map(state_by_cluster).fillna("Unknown")
    obs["_x"] = coords[:, 0]
    obs["_y"] = coords[:, 1]
    global_scale = float(np.sqrt(np.var(coords[:, 0]) + np.var(coords[:, 1]))) or 1.0
    rows: List[Dict[str, Any]] = []
    for state, sub in obs.groupby("_state", observed=False):
        if state == "Unknown":
            continue
        centroids = sub.groupby(batch_key, observed=False)[["_x", "_y"]].mean()
        if len(centroids) < 2:
            continue
        distances = [
            float(np.linalg.norm(centroids.loc[a].values - centroids.loc[b].values))
            for a, b in combinations(centroids.index, 2)
        ]
        rows.append(
            {
                "broad_label": str(state),
                "n_batches_present": int(len(centroids)),
                "mean_batch_centroid_distance": round(float(np.mean(distances)), 4),
                "max_batch_centroid_distance": round(float(np.max(distances)), 4),
                "mean_distance_over_global_umap_scale": round(float(np.mean(distances)) / global_scale, 4),
            }
        )
    return rows


def _neighborhood_batch_mixing(
    adata,
    batch_key: str,
    cluster_key: str,
    state_by_cluster: Dict[str, str],
    *,
    use_rep: str,
    n_neighbors: int,
    obs_key: str = "batch_diagnostic_neighborhood_entropy",
) -> Optional[Dict[str, Any]]:
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
    per_label: List[Dict[str, Any]] = []
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


def _cluster_batch_concordance(adata, batch_key: str, cluster_key: str) -> Dict[str, Any]:
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


def _state_expression_shifts(
    adata,
    batch_key: str,
    cluster_key: str,
    state_by_cluster: Dict[str, str],
    *,
    min_cells_per_cluster_sample: int,
    n_top_genes: int,
) -> Dict[str, Any]:
    matrix, genes = _expression_view(adata)
    gene_array = np.asarray(genes)
    usable_gene_mask = np.array([not _is_nuisance_gene(g) for g in genes], dtype=bool)
    obs = adata.obs[[batch_key, cluster_key]].copy()
    obs["_state"] = obs[cluster_key].astype(str).map(state_by_cluster).fillna("Unknown")
    shifts: List[Dict[str, Any]] = []
    recurring: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    for state, state_obs in obs.groupby("_state", observed=False):
        if state == "Unknown":
            continue
        state_idx = np.flatnonzero(obs["_state"].values == state)
        if len(state_idx) < 2 * min_cells_per_cluster_sample:
            continue
        for batch in sorted(state_obs[batch_key].astype(str).unique()):
            in_batch = np.flatnonzero((obs["_state"].values == state) & (obs[batch_key].astype(str).values == batch))
            out_batch = np.setdiff1d(state_idx, in_batch, assume_unique=False)
            if len(in_batch) < min_cells_per_cluster_sample or len(out_batch) < min_cells_per_cluster_sample:
                continue
            mean_in = _mean_expression(matrix[in_batch, :])
            mean_out = _mean_expression(matrix[out_batch, :])
            if mean_in.size == 0 or mean_out.size == 0:
                continue
            delta = mean_in - mean_out
            delta = np.where(usable_gene_mask, delta, 0.0)
            order_up = np.argsort(delta)[::-1]
            order_down = np.argsort(delta)
            top_up = [
                {"gene": str(gene_array[i]), "delta": round(float(delta[i]), 4)}
                for i in order_up[:n_top_genes]
                if delta[i] > 0
            ][:n_top_genes]
            top_down = [
                {"gene": str(gene_array[i]), "delta": round(float(delta[i]), 4)}
                for i in order_down[:n_top_genes]
                if delta[i] < 0
            ][:n_top_genes]
            shifts.append(
                {
                    "broad_label": str(state),
                    "batch": str(batch),
                    "n_cells_in_batch": int(len(in_batch)),
                    "n_cells_other_batches": int(len(out_batch)),
                    "top_up": top_up[:10],
                    "top_down": top_down[:10],
                }
            )
            for direction, geneset in (("up", top_up), ("down", top_down)):
                for entry in geneset[:15]:
                    key = (str(batch), direction, entry["gene"])
                    rec = recurring.setdefault(
                        key,
                        {
                            "batch": str(batch),
                            "direction": direction,
                            "gene": entry["gene"],
                            "broad_labels": set(),
                            "max_abs_delta": 0.0,
                        },
                    )
                    rec["broad_labels"].add(str(state))
                    rec["max_abs_delta"] = max(rec["max_abs_delta"], abs(float(entry["delta"])))

    shared = []
    for rec in recurring.values():
        if len(rec["broad_labels"]) < 2:
            continue
        shared.append(
            {
                "batch": rec["batch"],
                "direction": rec["direction"],
                "gene": rec["gene"],
                "n_broad_labels": int(len(rec["broad_labels"])),
                "broad_labels": sorted(rec["broad_labels"]),
                "max_abs_delta": round(float(rec["max_abs_delta"]), 4),
            }
        )
    shared.sort(key=lambda r: (r["n_broad_labels"], r["max_abs_delta"]), reverse=True)
    return {"state_sample_expression_shifts": shifts, "shared_cross_cell_type_signatures": shared[:50]}


def _cross_sample_identity_deg(
    adata,
    batch_key: str,
    cluster_key: str,
    composition: List[Dict[str, Any]],
    cluster_markers: Dict[str, List[str]],
    *,
    min_cells: int,
    n_top_genes: int,
    max_pairs: int = 12,
    marker_overlap_min: float = 0.15,
    signature_similarity_min: float = 0.30,
) -> Dict[str, Any]:
    """Paired within-sample identity DEG — the clean control for a batch effect.

    For two clusters dominated by DIFFERENT samples that nonetheless look like the
    same cell type (their one-vs-all markers overlap), compute each cluster's DEG
    against the rest of ITS OWN sample, then compare the two identity signatures.

    Because each DEG is computed entirely *within one sample*, no batch signal can
    contaminate it. So if the two within-sample "what makes me distinct"
    signatures match, the clusters are the same biological population separated
    only by sample — i.e. a batch effect, and they should merge under integration.
    This is the clean control that comparing the same state ACROSS samples cannot
    give (that conflates biology and batch); here the batch is held constant
    inside each DEG, so a match is conclusive.
    """
    matrix, genes = _expression_view(adata)
    gene_array = np.asarray(genes)
    usable = np.array([not _is_nuisance_gene(g) for g in genes], dtype=bool)
    cluster_vals = adata.obs[cluster_key].astype(str).values
    batch_vals = adata.obs[batch_key].astype(str).values

    # Dominant sample per sample-dominated cluster (the batch-split candidates).
    dom = {row["cluster"]: row["dominant_batch"] for row in composition if row.get("sample_dominated")}

    def _markers(cluster_id: str) -> set:
        return {g for g in (cluster_markers.get(cluster_id) or [])[:n_top_genes] if not _is_nuisance_gene(g)}

    # Candidate cross-sample same-type pairs, nominated by marker overlap.
    candidates: List[tuple] = []
    dom_clusters = sorted(dom)
    for a_i in range(len(dom_clusters)):
        for b_i in range(a_i + 1, len(dom_clusters)):
            ca, cb = dom_clusters[a_i], dom_clusters[b_i]
            if dom[ca] == dom[cb]:
                continue  # same dominant sample → not a cross-sample pair
            ma, mb = _markers(ca), _markers(cb)
            if not ma or not mb:
                continue
            jac = len(ma & mb) / len(ma | mb)
            if jac >= marker_overlap_min:
                candidates.append((jac, ca, cb))
    candidates.sort(reverse=True)
    candidates = candidates[:max_pairs]

    def _within_sample_signature(cluster_id: str, sample: str):
        in_mask = (cluster_vals == cluster_id) & (batch_vals == sample)
        out_mask = (cluster_vals != cluster_id) & (batch_vals == sample)
        n_in, n_out = int(in_mask.sum()), int(out_mask.sum())
        if n_in < min_cells or n_out < min_cells:
            return None, n_in, n_out
        delta = _mean_expression(matrix[in_mask, :]) - _mean_expression(matrix[out_mask, :])
        if delta.size == 0:
            return None, n_in, n_out
        # Exclude nuisance genes from the identity signature.
        delta = np.where(usable, delta, -np.inf)
        top_idx = np.argsort(delta)[::-1][:n_top_genes]
        sig = [str(gene_array[k]) for k in top_idx if np.isfinite(delta[k]) and delta[k] > 0]
        return sig, n_in, n_out

    pairs_out: List[Dict[str, Any]] = []
    for jac, ca, cb in candidates:
        sig_a, na_in, _ = _within_sample_signature(ca, dom[ca])
        sig_b, nb_in, _ = _within_sample_signature(cb, dom[cb])
        if not sig_a or not sig_b:
            continue
        set_a, set_b = set(sig_a), set(sig_b)
        shared_genes = [g for g in sig_a if g in set_b]  # keep sig_a ordering
        similarity = len(set_a & set_b) / len(set_a | set_b)
        pairs_out.append({
            "cluster_a": ca, "sample_a": dom[ca], "n_cells_a": na_in,
            "cluster_b": cb, "sample_b": dom[cb], "n_cells_b": nb_in,
            "marker_overlap_jaccard": round(jac, 3),
            "within_sample_signature_a": sig_a[:15],
            "within_sample_signature_b": sig_b[:15],
            "shared_identity_genes": shared_genes[:20],
            "signature_similarity": round(similarity, 3),
            "conclusive_batch_effect": bool(similarity >= signature_similarity_min),
        })

    pairs_out.sort(key=lambda e: e["signature_similarity"], reverse=True)
    conclusive = [p for p in pairs_out if p["conclusive_batch_effect"]]
    return {
        "cross_sample_identity_pairs": pairs_out[:max_pairs],
        "conclusive_pairs": conclusive,
        "n_conclusive": len(conclusive),
        "signature_similarity_min": signature_similarity_min,
    }


def _batch_diagnostic_group_doc(params: Dict[str, Any]) -> "ArtifactGroupDoc":
    """Static documentation for the batch-diagnostic CSVs.

    Describes what each table computes and what its columns mean — authored next
    to the tool (not the harness) because it documents this tool's own output.
    The dataset-specific interpretation is left to the model.
    """
    from ..core.artifact_docs import ArtifactGroupDoc, FileDoc

    return ArtifactGroupDoc(
        group="diagnose_batch_effect",
        title="Batch-effect diagnostic — how to read these files",
        overview=(
            "Descriptive diagnostic for whether an uncorrected multi-sample dataset "
            "separates by sample/batch. Each CSV captures one line of evidence; "
            "together they inform whether to integrate, keep unintegrated, or analyze "
            "samples separately. None of these tables prove a difference is technical "
            "rather than real per-sample biology — they are evidence, not proof."
        ),
        params=params,
        files=[
            FileDoc(
                filename="batch_diagnostic_cluster_sample_composition.csv",
                purpose=(
                    "How each cluster's cells split across samples/batches — the primary "
                    "signal for sample-private clusters."
                ),
                computation=(
                    "Cross-tabulation of cluster x batch; per cluster the dominant batch and "
                    "its fraction, plus a normalized entropy of the sample mixture."
                ),
                columns={
                    "cluster": "Cluster id from the clustering used.",
                    "n_cells": "Total cells in the cluster.",
                    "dominant_batch": "Sample/batch contributing the most cells to the cluster.",
                    "dominant_fraction": "Fraction of the cluster's cells from dominant_batch (1.0 = one sample).",
                    "normalized_sample_entropy": "Evenness of the sample mixture, 0 (one sample) to 1 (even split).",
                    "sample_exclusive": "True if dominant_fraction >= 0.98 (essentially one sample).",
                    "sample_dominated": "True if dominant_fraction >= 0.80 (mostly one sample).",
                    "batch_counts": "Per-batch cell counts (dict serialized as text).",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_broad_cluster_labels.csv",
                purpose=(
                    "Provisional broad lineage label per cluster, used only to group clusters "
                    "for this diagnostic — NOT a final annotation."
                ),
                computation=(
                    "Per-cluster marker genes (rank_genes_groups) matched against a small "
                    "built-in broad-lineage marker set; highest-scoring lineage wins."
                ),
                columns={
                    "cluster": "Cluster id.",
                    "broad_label": "Provisional broad lineage (e.g. epithelial, myeloid) — do not reuse as annotation.",
                    "confidence": "Confidence of the broad-label assignment.",
                    "supporting_markers": "Markers that drove the label.",
                    "top_markers": "Top marker genes for the cluster.",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_condition_confounding.csv",
                purpose=(
                    "Whether each supplied condition/covariate is confounded with the "
                    "batch/sample variable — if so, batch and biology cannot be separated."
                ),
                computation=(
                    "Cross-tabulation of batch x condition; purity of batches within conditions "
                    "and vice versa; flagged confounded above a purity threshold."
                ),
                columns={
                    "condition_key": "The obs column tested against batch.",
                    "n_conditions": "Number of distinct condition values.",
                    "max_batch_purity": "How cleanly the purest condition maps to a single batch.",
                    "median_batch_purity": "Median of that batch-purity across conditions.",
                    "max_condition_purity": "How cleanly the purest batch maps to a single condition.",
                    "median_condition_purity": "Median of that condition-purity across batches.",
                    "confounded_with_batch": "True if the two variables are largely redundant (confounded).",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_shared_signatures.csv",
                purpose=(
                    "Genes that shift the same way with a given sample across MULTIPLE cell "
                    "types — a hallmark of a technical (batch-wide) effect, not one cell type's biology."
                ),
                computation=(
                    "Per broad label, per-sample expression deltas vs the other samples; genes "
                    "recurring in the same direction across >=2 broad labels are retained."
                ),
                columns={
                    "batch": "Sample/batch the shift is associated with.",
                    "direction": "'up' or 'down' in that batch.",
                    "gene": "Gene symbol.",
                    "n_broad_labels": "Number of distinct broad cell types showing this shift.",
                    "broad_labels": "Which broad labels show it.",
                    "max_abs_delta": "Largest absolute expression delta observed.",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_neighborhood_entropy.csv",
                purpose=(
                    "Per broad-label batch-mixing entropy in the chosen embedding — low entropy "
                    "where a cell type spans several samples means it segregates by sample (batch-like)."
                ),
                computation=(
                    "kNN graph in the chosen representation; per cell, entropy of its neighbors' "
                    "batch labels; averaged per broad label."
                ),
                columns={
                    "broad_label": "Provisional broad lineage.",
                    "n_cells": "Cells with this label.",
                    "n_batches_present": "How many batches contribute cells to this label.",
                    "mean_entropy": "Mean neighborhood batch entropy (higher = better mixed; lower = more segregated).",
                },
            ),
            FileDoc(
                filename="batch_diagnostic_cross_sample_identity_deg.csv",
                purpose=(
                    "The most conclusive test — pairs of sample-private clusters that are the same "
                    "cell population separated only by sample (a batch split integration should merge)."
                ),
                computation=(
                    "Candidate pairs nominated by one-vs-all marker overlap; each cluster is then "
                    "DEG'd against the rest of its OWN sample (batch held constant) and the two "
                    "within-sample signatures compared."
                ),
                how_to_read=(
                    "High signature_similarity with conclusive_batch_effect=True means the split is "
                    "technical (integrate to merge); low similarity means genuinely different "
                    "populations, and the separation may be real biology."
                ),
                columns={
                    "cluster_a": "First cluster of the pair.",
                    "sample_a": "Sample of cluster_a.",
                    "cluster_b": "Second cluster of the pair.",
                    "sample_b": "Sample of cluster_b.",
                    "marker_overlap_jaccard": "Jaccard overlap of the two clusters' one-vs-all markers (nominates the pair).",
                    "signature_similarity": "Similarity of the two within-sample identity signatures (higher = more likely same population).",
                    "conclusive_batch_effect": "True when similarity exceeds the threshold — conclusive that the split is batch, not biology.",
                    "shared_identity_genes": "Genes shared by both within-sample signatures.",
                },
            ),
        ],
    )


def _write_outputs(
    output_dir: Optional[str],
    tables: Dict[str, pd.DataFrame],
    group_doc: Optional["ArtifactGroupDoc"] = None,
) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    if not output_dir:
        return artifacts
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Per-file authored docs keyed by both full filename and stem.
    file_docs: Dict[str, Any] = {}
    if group_doc is not None:
        for fdoc in group_doc.files:
            file_docs[fdoc.filename] = fdoc
            file_docs[Path(fdoc.filename).stem] = fdoc

    for stem, df in tables.items():
        path = root / f"{stem}.csv"
        df.to_csv(path, index=False)
        metadata: Dict[str, Any] = {"kind": stem}
        fdoc = file_docs.get(stem) or file_docs.get(f"{stem}.csv")
        if fdoc is not None:
            from ..core.artifact_docs import artifact_column_metadata

            metadata.update(artifact_column_metadata(fdoc, df))
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


def diagnose_batch_effect(
    adata,
    *,
    batch_key: str,
    cluster_key: str = "leiden",
    condition_keys: Optional[List[str]] = None,
    min_cells_per_cluster_sample: int = 30,
    n_top_genes: int = 25,
    entropy_use_rep: str = ENTROPY_DEFAULT_USE_REP,
    entropy_n_neighbors: int = ENTROPY_DEFAULT_N_NEIGHBORS,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs.")
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"cluster_key '{cluster_key}' not found in adata.obs. Run clustering first.")

    batches = list(map(str, adata.obs[batch_key].dropna().astype(str).unique()))
    if len(batches) < 2:
        raise ValueError(f"batch_key '{batch_key}' has fewer than two groups.")

    marker_key = f"batch_diag_markers_{cluster_key}"
    try:
        cluster_markers = _compute_cluster_markers(adata, cluster_key, marker_key, n_top_genes)
        marker_error = None
    except Exception as exc:
        cluster_markers = {str(c): [] for c in adata.obs[cluster_key].astype(str).unique()}
        marker_error = str(exc)

    broad_by_cluster: Dict[str, Dict[str, Any]] = {}
    state_by_cluster: Dict[str, str] = {}
    for cluster, markers in cluster_markers.items():
        label_info = _broad_label_from_markers(markers)
        broad_by_cluster[str(cluster)] = {
            "cluster": str(cluster),
            "broad_label": label_info["label"],
            "confidence": label_info["confidence"],
            "supporting_markers": label_info["supporting_markers"],
            "top_markers": markers[:10],
        }
        state_by_cluster[str(cluster)] = label_info["label"]

    composition = _cluster_sample_composition(adata, batch_key, cluster_key)
    n_clusters = len(composition)
    sample_dominated = [row for row in composition if row["sample_dominated"]]
    sample_exclusive = [row for row in composition if row["sample_exclusive"]]
    cells_in_dominated = sum(row["n_cells"] for row in sample_dominated)
    dominated_cell_fraction = cells_in_dominated / max(int(adata.n_obs), 1)

    separation = _umap_state_separation(adata, state_by_cluster, batch_key, cluster_key)
    entropy_mixing = _neighborhood_batch_mixing(
        adata,
        batch_key,
        cluster_key,
        state_by_cluster,
        use_rep=entropy_use_rep,
        n_neighbors=entropy_n_neighbors,
    )
    shift_payload = _state_expression_shifts(
        adata,
        batch_key,
        cluster_key,
        state_by_cluster,
        min_cells_per_cluster_sample=min_cells_per_cluster_sample,
        n_top_genes=n_top_genes,
    )
    shared = shift_payload["shared_cross_cell_type_signatures"]
    max_shared_labels = max([entry["n_broad_labels"] for entry in shared], default=0)
    n_shared_genes = len(shared)

    # Paired within-sample identity DEG — the clean control (batch held constant
    # inside each DEG). See _cross_sample_identity_deg.
    identity_deg = _cross_sample_identity_deg(
        adata,
        batch_key,
        cluster_key,
        composition,
        cluster_markers,
        min_cells=min_cells_per_cluster_sample,
        n_top_genes=n_top_genes,
    )

    support_reasons: List[str] = []
    caution_reasons: List[str] = []
    condition_cols = _candidate_condition_keys(adata, batch_key, condition_keys)
    confounding = _confounding_summary(adata, batch_key, condition_cols)
    any_confounded = any(row["confounded_with_batch"] for row in confounding)
    if not condition_cols:
        caution_reasons.append(
            "no condition-like metadata columns were available, so sample-condition confounding was not tested"
        )

    sample_prefixes = sorted({str(batch).split("_", 1)[0] for batch in batches if "_" in str(batch)})
    mixed_source_like_samples = len(sample_prefixes) >= 3
    if mixed_source_like_samples:
        caution_reasons.append(
            "sample names suggest multiple source/procedure groups; sample effects may include real tissue or collection-method biology"
        )

    concordance = _cluster_batch_concordance(adata, batch_key, cluster_key)

    if dominated_cell_fraction >= 0.30 or (n_clusters and len(sample_dominated) / n_clusters >= 0.30):
        support_reasons.append("many clusters are sample-dominated")
    if sample_exclusive:
        support_reasons.append("some clusters are nearly sample-exclusive")
    if max_shared_labels >= 3 or (max_shared_labels >= 2 and n_shared_genes >= 5):
        support_reasons.append("sample-associated expression shifts recur across broad cell types")
    if separation and max(row["mean_distance_over_global_umap_scale"] for row in separation) >= 0.35:
        support_reasons.append("broad cell types have separated sample centroids on the UMAP")
    if concordance["tracks_sample"]:
        support_reasons.append(
            f"clusters track sample labels (ARI {concordance['ari']:.2f}, NMI {concordance['nmi']:.2f} "
            f"between {cluster_key} and {batch_key}): {cluster_key} clusters largely correspond to "
            "individual samples rather than shared cell states"
        )
    if identity_deg["n_conclusive"]:
        top = identity_deg["conclusive_pairs"][0]
        shared_str = ", ".join(top["shared_identity_genes"][:8]) or "shared identity genes"
        support_reasons.append(
            f"the same cell type appears split across samples: cluster {top['cluster_a']} (sample "
            f"{top['sample_a']}) and cluster {top['cluster_b']} (sample {top['sample_b']}) carry almost "
            f"the same marker genes when each is compared against the rest of its OWN sample "
            f"(similarity {top['signature_similarity']:.2f}; shared genes: {shared_str}). Because each "
            f"comparison stays inside a single sample, batch is held constant — so the match means these "
            f"are one cell population pulled apart only by sample, a batch effect that integration should merge"
        )

    # Name the markers behind sample-segregated Epithelial clusters and caveat them: the
    # epithelial compartment is frequently donor/patient-private regardless of tissue
    # (donor-specific epithelial states and genetic background in normal tissue; malignant
    # clones and CNVs in tumors), so this pattern is often real biology rather than a
    # technical batch effect — and integrating it across samples can erase genuine
    # per-sample differences. Do not assume the tissue is a tumor.
    segregated_clusters: Dict[str, Dict[str, Any]] = {row["cluster"]: row for row in sample_dominated}
    for row in sample_exclusive:
        segregated_clusters.setdefault(row["cluster"], row)
    epithelial_private_markers: List[str] = []
    epithelial_private_clusters: List[str] = []
    for cluster_id in segregated_clusters:
        info = broad_by_cluster.get(str(cluster_id))
        if info and info.get("broad_label") == "Epithelial":
            epithelial_private_clusters.append(str(cluster_id))
            markers = info.get("supporting_markers") or info.get("top_markers") or []
            epithelial_private_markers.extend(str(gene) for gene in markers[:4])
    if epithelial_private_clusters:
        marker_str = ", ".join(dict.fromkeys(epithelial_private_markers)) or "epithelial markers"
        caution_reasons.append(
            f"sample-segregated Epithelial cluster(s) {', '.join(epithelial_private_clusters)} "
            f"(driven by {marker_str}) may be donor/patient-private epithelial biology rather than a "
            "technical batch effect; epithelium is often sample-specific (donor-specific epithelial "
            "states in normal tissue, or malignant clones/CNVs in tumors), and integrating it across "
            "samples can erase real per-sample differences"
        )
    if entropy_mixing is None:
        caution_reasons.append(
            f"neighborhood batch-mixing entropy was skipped because '{entropy_use_rep}' is not in "
            "adata.obsm — run the uncorrected PCA first to enable this continuous check"
        )
    elif entropy_mixing.get("skipped"):
        caution_reasons.append(
            "neighborhood batch-mixing entropy could not be computed "
            f"({entropy_mixing.get('reason')})"
        )
    else:
        ratio = entropy_mixing["mixing_ratio"]
        if ratio is not None and ratio <= ENTROPY_LOW_MIXING_RATIO:
            support_reasons.append(
                f"neighborhood batch-mixing entropy in {entropy_mixing['use_rep']} is low "
                f"(mean {entropy_mixing['mean_entropy']:.2f} vs achievable {entropy_mixing['global_ceiling']:.2f}, "
                f"ratio {ratio:.2f}): cells tend to neighbor their own sample even within shared regions"
            )
    if marker_error:
        caution_reasons.append(f"cluster marker calculation failed: {marker_error}")
    if any_confounded:
        caution_reasons.append("sample/batch is confounded with at least one condition-like metadata column")

    if any_confounded and support_reasons:
        verdict = "confounded_with_condition"
        recommendation = (
            "Ask the user whether the confounded sample/condition structure is expected before integration."
        )
    elif support_reasons:
        verdict = "batch_effect_supported"
        if mixed_source_like_samples or not condition_cols:
            recommendation = (
                "Offer scVI integration, but frame the evidence as sample/source/procedure effects "
                "and ask the user to confirm whether to integrate all samples together or within comparable groups."
            )
        else:
            recommendation = "Offer scVI integration, but wait for the user's confirmation."
    elif n_clusters < 2 or not shift_payload["state_sample_expression_shifts"]:
        verdict = "insufficient_evidence"
        recommendation = "Ask the user; descriptive evidence is limited for this dataset."
    else:
        verdict = "no_correction_needed"
        recommendation = "Proceed without integration unless the user has external design knowledge."

    composition_df = pd.DataFrame(
        [
            {k: v for k, v in row.items() if k != "batch_counts"}
            | {f"count_{k}": v for k, v in row["batch_counts"].items()}
            for row in composition
        ]
    )
    broad_df = pd.DataFrame(list(broad_by_cluster.values()))
    confounding_df = pd.DataFrame(confounding)
    shared_df = pd.DataFrame(shared)
    # Narrow to a non-None dict only when entropy was actually computed, so the
    # downstream summary/result blocks can index it safely.
    entropy_ok: Optional[Dict[str, Any]] = (
        entropy_mixing
        if entropy_mixing is not None and not entropy_mixing.get("skipped")
        else None
    )
    entropy_df = (
        pd.DataFrame(entropy_ok["entropy_per_broad_label"])
        if entropy_ok is not None
        else pd.DataFrame()
    )
    identity_df = pd.DataFrame([
        {
            "cluster_a": p["cluster_a"], "sample_a": p["sample_a"],
            "cluster_b": p["cluster_b"], "sample_b": p["sample_b"],
            "marker_overlap_jaccard": p["marker_overlap_jaccard"],
            "signature_similarity": p["signature_similarity"],
            "conclusive_batch_effect": p["conclusive_batch_effect"],
            "shared_identity_genes": ", ".join(p["shared_identity_genes"]),
        }
        for p in identity_deg["cross_sample_identity_pairs"]
    ])
    artifacts = _write_outputs(
        output_dir,
        {
            "batch_diagnostic_cluster_sample_composition": composition_df,
            "batch_diagnostic_broad_cluster_labels": broad_df,
            "batch_diagnostic_condition_confounding": confounding_df,
            "batch_diagnostic_shared_signatures": shared_df,
            "batch_diagnostic_neighborhood_entropy": entropy_df,
            "batch_diagnostic_cross_sample_identity_deg": identity_df,
        },
        group_doc=_batch_diagnostic_group_doc(
            {
                "batch_key": batch_key,
                "cluster_key": cluster_key,
                "n_top_genes": n_top_genes,
                "min_cells_per_cluster_sample": min_cells_per_cluster_sample,
                "entropy_use_rep": entropy_use_rep,
                "entropy_n_neighbors": entropy_n_neighbors,
                "condition_keys": condition_keys or [],
            }
        ),
    )

    # Human-readable findings shown in the terminal (the agent prints
    # result["terminal_summary"] for reasoning tools). Keep it concise.
    # Human-readable verdict label (the machine slug stays on result["verdict"]).
    verdict_label = {
        "confounded_with_condition": (
            "Sample and experimental condition overlap, so a batch effect can't be "
            "separated from real biology"
        ),
        "batch_effect_supported": "Evidence points to a technical batch effect across samples",
        "insufficient_evidence": "Not enough evidence to call this a batch effect either way",
        "no_correction_needed": "Samples look well mixed — no batch correction appears needed",
    }.get(verdict, verdict)
    terminal_summary = [f"Verdict: {verdict_label}"]
    terminal_summary += [f"• {r}" for r in support_reasons]
    terminal_summary += [f"⚠ {r}" for r in caution_reasons]
    if sample_dominated:
        terminal_summary.append(
            f"{len(sample_dominated)} cluster(s) are made up almost entirely of one sample "
            f"({dominated_cell_fraction * 100:.0f}% of all cells sit in them) — a sign cells are "
            "grouping by which sample they came from rather than by cell type"
        )
    if entropy_ok is not None:
        if entropy_ok["mixing_ratio"] is not None:
            terminal_summary.append(
                "Neighborhood mixing: on average each cell's nearest neighbors span "
                f"{entropy_ok['mixing_ratio'] * 100:.0f}% of the sample variety you'd see if the "
                "samples were perfectly intermixed — lower means cells tend to sit next to others "
                "from their own sample (a batch signature)"
            )
        else:
            terminal_summary.append(
                "Neighborhood mixing across samples: entropy "
                f"{entropy_ok['mean_entropy']:.2f} (higher = better intermixed)"
            )
    terminal_summary.append(
        f"How closely clusters track samples: ARI {concordance['ari']:.2f}, NMI "
        f"{concordance['nmi']:.2f} (0 = clusters unrelated to sample, 1 = clusters exactly follow "
        f"sample; {concordance['interpretation']})"
    )
    if identity_deg["n_conclusive"]:
        top = identity_deg["conclusive_pairs"][0]
        terminal_summary.append(
            f"Same cell type split across samples: {identity_deg['n_conclusive']} clear case(s). "
            f"For example, cluster {top['cluster_a']} (sample {top['sample_a']}) and cluster "
            f"{top['cluster_b']} (sample {top['sample_b']}) carry almost the same marker genes when "
            f"each is compared against the rest of its OWN sample (match "
            f"{top['signature_similarity']:.2f}) — i.e. one population pulled apart by sample, "
            "which is a batch effect integration should merge"
        )
    terminal_summary.append(f"→ {recommendation}")

    result = {
        "status": "ok",
        "tool": "diagnose_batch_effect",
        "batch_key": batch_key,
        "cluster_key": cluster_key,
        "n_batches": int(len(batches)),
        "n_clusters": int(n_clusters),
        "verdict": verdict,
        "recommendation": recommendation,
        "support_reasons": support_reasons,
        "caution_reasons": caution_reasons,
        "terminal_summary": terminal_summary,
        "cluster_sample_summary": {
            "n_sample_dominated_clusters": int(len(sample_dominated)),
            "n_sample_exclusive_clusters": int(len(sample_exclusive)),
            "fraction_cells_in_sample_dominated_clusters": round(float(dominated_cell_fraction), 4),
            "sample_dominated_clusters": sample_dominated[:20],
            "sample_exclusive_clusters": sample_exclusive[:20],
        },
        "broad_cluster_labels": list(broad_by_cluster.values()),
        "umap_broad_label_sample_separation": separation[:30],
        "neighborhood_batch_entropy": entropy_mixing,
        "cluster_batch_concordance": concordance,
        "condition_confounding": confounding,
        "state_sample_expression_shifts": shift_payload["state_sample_expression_shifts"][:50],
        "shared_cross_cell_type_signatures": shared[:30],
        "cross_sample_identity_deg": {
            "n_conclusive": identity_deg["n_conclusive"],
            "signature_similarity_min": identity_deg["signature_similarity_min"],
            "conclusive_pairs": identity_deg["conclusive_pairs"][:10],
            "candidate_pairs": identity_deg["cross_sample_identity_pairs"][:10],
            "interpretation": (
                "Each cluster in a pair was DEG'd against the rest of its OWN sample, so batch is held "
                "constant inside each test. A high signature_similarity means the two sample-private "
                "clusters are the same cell population separated only by sample — conclusive evidence "
                "that the split is technical (batch) and integration should merge them. Low similarity "
                "means they are genuinely different populations and the separation may be real biology. "
                "Candidate pairs are nominated by one-vs-all marker overlap; the within-sample "
                "signature match is the confirmation."
            ),
        },
        "evidence_limits": [
            "This is a descriptive diagnostic, not proof that sample-associated differences are technical.",
            "One-sample-per-condition or sample-condition confounding cannot distinguish batch from biology.",
            "If sample names encode tissue, procedure, site, or disease state, sample-exclusive clusters may reflect real biology as well as technical effects.",
            "Cluster–sample ARI/NMI measure how strongly clusters correspond to samples; high values can reflect donor/patient-private biology (e.g. donor-specific epithelial states, or malignant clones in tumors) as much as a technical batch effect.",
            "Broad labels are provisional and must not be reused as final annotation.",
        ],
        "artifacts_created": artifacts,
    }
    if entropy_ok is not None:
        result["evidence_limits"].append(
            f"Neighborhood batch-mixing entropy reflects mixing in {entropy_ok['use_rep']} and, "
            "like the cluster and centroid evidence, cannot prove a sample-associated difference is "
            "technical rather than real per-sample biology; it is normalized against the batch-size "
            "ceiling but very small or rare batches can still depress the score."
        )
    adata.uns["batch_effect_diagnostic"] = result
    return result
