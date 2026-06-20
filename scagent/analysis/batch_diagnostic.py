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


def _write_outputs(output_dir: Optional[str], tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    if not output_dir:
        return artifacts
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for stem, df in tables.items():
        path = root / f"{stem}.csv"
        df.to_csv(path, index=False)
        artifacts.append({"path": str(path), "role": "artifact", "metadata": {"kind": stem}})
    return artifacts


def diagnose_batch_effect(
    adata,
    *,
    batch_key: str,
    cluster_key: str = "leiden",
    condition_keys: Optional[List[str]] = None,
    min_cells_per_cluster_sample: int = 30,
    n_top_genes: int = 25,
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

    if dominated_cell_fraction >= 0.30 or (n_clusters and len(sample_dominated) / n_clusters >= 0.30):
        support_reasons.append("many clusters are sample-dominated")
    if sample_exclusive:
        support_reasons.append("some clusters are nearly sample-exclusive")
    if max_shared_labels >= 3 or (max_shared_labels >= 2 and n_shared_genes >= 5):
        support_reasons.append("sample-associated expression shifts recur across broad cell types")
    if separation and max(row["mean_distance_over_global_umap_scale"] for row in separation) >= 0.35:
        support_reasons.append("broad cell types have separated sample centroids on the UMAP")
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
    artifacts = _write_outputs(
        output_dir,
        {
            "batch_diagnostic_cluster_sample_composition": composition_df,
            "batch_diagnostic_broad_cluster_labels": broad_df,
            "batch_diagnostic_condition_confounding": confounding_df,
            "batch_diagnostic_shared_signatures": shared_df,
        },
    )

    # Human-readable findings shown in the terminal (the agent prints
    # result["terminal_summary"] for reasoning tools). Keep it concise.
    terminal_summary = [f"verdict: {verdict}"]
    terminal_summary += [f"• {r}" for r in support_reasons]
    terminal_summary += [f"⚠ {r}" for r in caution_reasons]
    if sample_dominated:
        terminal_summary.append(
            f"{len(sample_dominated)} sample-dominated cluster(s); "
            f"{dominated_cell_fraction * 100:.0f}% of cells in them"
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
        "condition_confounding": confounding,
        "state_sample_expression_shifts": shift_payload["state_sample_expression_shifts"][:50],
        "shared_cross_cell_type_signatures": shared[:30],
        "evidence_limits": [
            "This is a descriptive diagnostic, not proof that sample-associated differences are technical.",
            "One-sample-per-condition or sample-condition confounding cannot distinguish batch from biology.",
            "If sample names encode tissue, procedure, site, or disease state, sample-exclusive clusters may reflect real biology as well as technical effects.",
            "Broad labels are provisional and must not be reused as final annotation.",
        ],
        "artifacts_created": artifacts,
    }
    adata.uns["batch_effect_diagnostic"] = result
    return result
