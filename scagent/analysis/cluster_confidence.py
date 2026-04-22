"""
Cluster confidence and annotation sanity checks for scagent.

Phase 3 turns cluster-identity heuristics into a reusable analysis module so
reports and downstream interpretation can rely on the same confidence logic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from anndata import AnnData


DEFAULT_ANNOTATION_PRIORITY = [
    "celltypist_majority_voting",
    "celltypist_predicted_labels",
    "scimilarity_representative_prediction",
    "scimilarity_predictions_unconstrained",
    "cell_type",
    "celltype",
]


@dataclass
class ClusterConfidence:
    cluster: str
    groupby: str
    n_cells: Optional[int] = None

    annotation_key: Optional[str] = None
    cell_type: str = "unknown"
    cell_type_fraction: Optional[float] = None

    secondary_annotation_key: Optional[str] = None
    secondary_cell_type: Optional[str] = None
    secondary_cell_type_fraction: Optional[float] = None

    annotation_agreement: str = "unknown"
    annotation_agreement_note: Optional[str] = None

    marker_support: str = "unknown"
    marker_lineage: str = "unknown"
    marker_support_markers: List[str] = field(default_factory=list)
    top_markers: List[str] = field(default_factory=list)

    confidence_score: float = 0.0
    confidence_level: str = "low"
    interpretation_cautions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {}, "")}


def get_best_annotation_key(adata: AnnData, candidates: Optional[List[str]] = None) -> Optional[str]:
    """Return the most useful annotation column available on the AnnData object."""
    for key in candidates or DEFAULT_ANNOTATION_PRIORITY:
        if key in adata.obs.columns:
            return key
    try:
        from ..core.inspector import rank_obs_semantic_candidates

        semantic = rank_obs_semantic_candidates(adata, roles={"cell_type"})
        ranked = semantic.get("cell_type", [])
        if ranked:
            return ranked[0].column
    except Exception:
        pass
    return None


def cluster_annotation_summary(
    adata: AnnData,
    cluster_id: str,
    groupby: str,
    annotation_key: str,
) -> Optional[Dict[str, Any]]:
    """Return dominant-label summary for one annotation source within a cluster."""
    if groupby not in adata.obs.columns or annotation_key not in adata.obs.columns:
        return None

    mask = adata.obs[groupby].astype(str) == str(cluster_id)
    labels = adata.obs.loc[mask, annotation_key].dropna().astype(str)
    if labels.empty:
        return None

    counts = labels.value_counts()
    return {
        "annotation_key": annotation_key,
        "label": str(counts.index[0]),
        "fraction": float(counts.iloc[0] / counts.sum()),
        "n_annotated": int(counts.sum()),
    }


def normalize_annotation_lineage(label: str) -> str:
    """Map detailed annotation labels to broad lineages for sanity checks."""
    lowered = (label or "").lower()
    if not lowered:
        return "unknown"
    if "plasmablast" in lowered or "plasma" in lowered:
        return "plasma"
    if "pdc" in lowered or "plasmacytoid" in lowered:
        return "pdc"
    if "dc" in lowered or "dendritic" in lowered:
        return "dendritic"
    if any(tok in lowered for tok in ["monocyte", "macroph", "myelo", "myelocyte"]):
        return "monocyte"
    if "nk" in lowered or "natural killer" in lowered:
        return "nk"
    if "b cell" in lowered or "b-cell" in lowered or lowered.startswith("b "):
        return "b_cell"
    if any(tok in lowered for tok in ["t cell", "helper t", "cytotoxic t", "regulatory t", "treg", "mait", "trm", "tem", "tcm"]):
        return "t_cell"
    return "unknown"


def annotation_agreement_summary(primary_label: str, secondary_label: Optional[str]) -> Dict[str, Any]:
    """Summarize whether annotation sources agree at fine or broad lineage level."""
    primary_lineage = normalize_annotation_lineage(primary_label)
    secondary_lineage = normalize_annotation_lineage(secondary_label or "")
    summary = {
        "status": "unknown",
        "note": "Only one annotation source available.",
        "primary_lineage": primary_lineage,
        "secondary_lineage": secondary_lineage,
    }
    if not secondary_label:
        return summary

    if primary_label.lower() == secondary_label.lower():
        summary.update(status="aligned", note="Annotation systems agree on the same label.")
        return summary

    if primary_lineage != "unknown" and primary_lineage == secondary_lineage:
        summary.update(
            status="broadly_aligned",
            note="Annotation systems agree on the broad lineage but not the fine-grained label.",
        )
        return summary

    myeloid = {"monocyte", "dendritic", "pdc"}
    lymphoid = {"t_cell", "nk", "b_cell", "plasma"}
    if primary_lineage in myeloid and secondary_lineage in myeloid:
        summary.update(
            status="myeloid_disagreement",
            note="Annotation systems agree this cluster is myeloid, but disagree on the finer identity.",
        )
    elif primary_lineage in lymphoid and secondary_lineage in lymphoid:
        summary.update(
            status="lymphoid_disagreement",
            note="Annotation systems agree this cluster is lymphoid, but disagree on the finer identity.",
        )
    else:
        summary.update(
            status="conflicting",
            note="Annotation systems disagree on the broad lineage assignment for this cluster.",
        )
    return summary


def expected_marker_panel(label: str) -> Dict[str, Any]:
    """Return a coarse canonical marker panel for the inferred label."""
    lineage = normalize_annotation_lineage(label)
    lowered = (label or "").lower()
    panels = {
        "cytotoxic_t": {"CCL5", "NKG7", "CTSW", "CST7", "GZMA", "PRF1", "GNLY", "CD3D"},
        "treg": {"IL32", "TIGIT", "LTB", "IL7R", "CTLA4", "FOXP3", "LTB"},
        "t_cell": {"CD3D", "CD3E", "TRAC", "LTB", "IL7R", "MALAT1"},
        "nk": {"GNLY", "NKG7", "KLRD1", "PRF1", "KLRF1", "CST7", "CTSW", "TYROBP"},
        "b_cell": {"MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "HLA-DRA", "CD37"},
        "plasma": {"JCHAIN", "MZB1", "SDC1", "IGHG1", "IGKC", "XBP1"},
        "monocyte": {"LYZ", "S100A8", "S100A9", "FCN1", "CTSS", "SAT1", "LST1", "VCAN", "MNDA", "FCER1G"},
        "dendritic": {"HLA-DRA", "HLA-DPA1", "HLA-DPB1", "CD74", "FCER1A", "CLEC10A", "CST3", "HLA-DQA1"},
        "pdc": {"IL3RA", "GZMB", "JCHAIN", "TCF4", "IRF7", "IFITM1"},
    }

    if any(tok in lowered for tok in ["regulatory t", "treg"]):
        panel_key = "treg"
    elif lineage == "t_cell" and any(tok in lowered for tok in ["cytotoxic", "trm", "tem"]):
        panel_key = "cytotoxic_t"
    elif lineage in panels:
        panel_key = lineage
    else:
        panel_key = "t_cell" if lineage == "t_cell" else "unknown"

    return {
        "panel_key": panel_key,
        "lineage": lineage,
        "markers": panels.get(panel_key, set()),
    }


def marker_support_summary(label: str, markers: List[str]) -> Dict[str, Any]:
    """Assess whether cluster markers support the inferred broad lineage."""
    panel = expected_marker_panel(label)
    expected = {marker.upper() for marker in panel["markers"]}
    observed = [marker for marker in markers if marker.upper() in expected]
    n_matched = len(observed)

    if not markers:
        status = "unknown"
    elif n_matched >= 3:
        status = "strong"
    elif n_matched == 2:
        status = "moderate"
    elif n_matched == 1:
        status = "weak"
    else:
        status = "absent"

    return {
        "status": status,
        "lineage": panel["lineage"],
        "panel_key": panel["panel_key"],
        "matched_markers": observed[:5],
    }


def get_cluster_top_markers(
    adata: AnnData,
    cluster_id: str,
    groupby: str,
    n_genes: int = 10,
) -> List[str]:
    """Return top marker genes for a cluster from the current DEG result if available."""
    if "rank_genes_groups" not in adata.uns:
        return []
    params = adata.uns["rank_genes_groups"].get("params", {})
    if params.get("groupby") != groupby:
        return []

    try:
        import scanpy as sc
        markers_df = sc.get.rank_genes_groups_df(adata, group=str(cluster_id)).head(n_genes)
    except Exception:
        return []

    return [str(name) for name in markers_df["names"].tolist()]


def _confidence_level(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "moderate"
    return "low"


def infer_cluster_confidence(
    adata: AnnData,
    cluster_id: str,
    groupby: str = "leiden",
    *,
    annotation_priority: Optional[List[str]] = None,
    marker_gene_count: int = 10,
) -> ClusterConfidence:
    """Infer cluster identity confidence from annotations, markers, and size."""
    result = ClusterConfidence(cluster=str(cluster_id), groupby=groupby)

    if groupby not in adata.obs.columns:
        result.interpretation_cautions.append(f"Cluster key '{groupby}' not found.")
        return result

    mask = adata.obs[groupby].astype(str) == str(cluster_id)
    n_cells = int(mask.sum())
    result.n_cells = n_cells
    if n_cells == 0:
        result.interpretation_cautions.append("Cluster has no cells in the current object.")
        return result

    primary_key = get_best_annotation_key(adata, annotation_priority)
    result.annotation_key = primary_key
    primary_summary = cluster_annotation_summary(adata, cluster_id, groupby, primary_key) if primary_key else None
    if primary_summary:
        result.cell_type = primary_summary["label"]
        result.cell_type_fraction = primary_summary["fraction"]
    else:
        result.cell_type = f"cluster {cluster_id}"

    secondary_summary = None
    for secondary_key in [
        "scimilarity_representative_prediction",
        "celltypist_majority_voting",
        "celltypist_predicted_labels",
    ]:
        if secondary_key == primary_key:
            continue
        secondary_summary = cluster_annotation_summary(adata, cluster_id, groupby, secondary_key)
        if secondary_summary:
            break

    if secondary_summary:
        result.secondary_annotation_key = secondary_summary["annotation_key"]
        result.secondary_cell_type = secondary_summary["label"]
        result.secondary_cell_type_fraction = secondary_summary["fraction"]

    agreement = annotation_agreement_summary(result.cell_type, result.secondary_cell_type)
    result.annotation_agreement = agreement["status"]
    result.annotation_agreement_note = agreement["note"]

    markers = get_cluster_top_markers(adata, cluster_id, groupby, n_genes=marker_gene_count)
    result.top_markers = markers[:5]
    marker_support = marker_support_summary(result.cell_type, markers)
    result.marker_support = marker_support["status"]
    result.marker_lineage = marker_support["lineage"]
    result.marker_support_markers = marker_support["matched_markers"]

    cautions: List[str] = []
    score = 0.0

    if result.cell_type_fraction is not None:
        if result.cell_type_fraction >= 0.9:
            score += 0.35
        elif result.cell_type_fraction >= 0.75:
            score += 0.25
        elif result.cell_type_fraction >= 0.6:
            score += 0.15
        else:
            cautions.append("Dominant annotation covers less than 60% of the cluster.")
    else:
        cautions.append("No dominant annotation was available for this cluster.")

    agreement_score = {
        "aligned": 0.3,
        "broadly_aligned": 0.22,
        "myeloid_disagreement": 0.12,
        "lymphoid_disagreement": 0.12,
        "conflicting": 0.02,
        "unknown": 0.1,
    }.get(result.annotation_agreement, 0.1)
    score += agreement_score
    if result.annotation_agreement in {"myeloid_disagreement", "lymphoid_disagreement", "conflicting"} and result.annotation_agreement_note:
        cautions.append(result.annotation_agreement_note)

    marker_score = {
        "strong": 0.25,
        "moderate": 0.18,
        "weak": 0.08,
        "absent": 0.0,
        "unknown": 0.1,
    }.get(result.marker_support, 0.0)
    score += marker_score
    if result.marker_support == "weak":
        cautions.append("Top markers only weakly support the inferred lineage.")
    elif result.marker_support == "absent":
        cautions.append("Top markers do not clearly support the inferred lineage.")

    if n_cells < 50:
        cautions.append("Small cluster size may make markers and annotation less stable.")
    elif n_cells < 100:
        cautions.append("Cluster is relatively small; fine-grained interpretation should stay cautious.")
        score -= 0.05

    result.confidence_score = round(max(0.0, min(score, 1.0)), 2)
    result.confidence_level = _confidence_level(result.confidence_score)
    result.interpretation_cautions = cautions
    return result
