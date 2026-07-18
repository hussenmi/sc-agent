"""The Per-Cluster Annotation Evidence table is simplified to show the independent
evidence side by side: Scimilarity prediction, CellTypist prediction, Supporting
genes, DEG prediction — then Final label + Reasoning. The old bureaucratic columns
(Validation tier, PanglaoDB label, Support level, QC cap, Competing labels) are gone.

Pins the redesign requested after run_2026_07_15_121111 (breast dataset where the
reference predictions were buried in one combined column).
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from scagent.agent.tools import _assemble_analysis_record, _split_reference_predictions


def test_split_reference_predictions_prefers_representative_and_majority():
    ev = {
        "reference_annotation_support": {
            "scimilarity_representative_prediction": "luminal epithelial cell of mammary gland",
            "scimilarity_predictions_unconstrained": "epithelial cell",
            "celltypist_majority_voting": "Epithelial cells",
            "celltypist_predicted_labels": "Luminal",
        }
    }
    scim, ctyp = _split_reference_predictions(ev)
    assert scim == "luminal epithelial cell of mammary gland"  # representative preferred
    assert ctyp == "Epithelial cells"  # majority-voting preferred


def test_split_reference_predictions_falls_back_and_marks_not_run():
    # Only the raw per-cell scimilarity key present; no celltypist at all.
    ev = {"reference_annotation_support": {"scimilarity_predictions_unconstrained": "fibroblast"}}
    scim, ctyp = _split_reference_predictions(ev)
    assert scim == "fibroblast"
    assert ctyp == "not run"
    # No reference support recorded at all.
    assert _split_reference_predictions({}) == ("not run", "not run")


def _adata_with_annotation():
    a = ad.AnnData(X=np.ones((4, 3), dtype="float32"))
    a.uns["annotation_validation"] = {
        "label_counts": {"Epithelial cells": 2, "Fibroblasts": 2},
        "per_cluster_evidence": {
            "0": {
                "label": "Epithelial cells",
                "reference_annotation_support": {
                    "scimilarity_representative_prediction": "luminal epithelial cell of mammary gland",
                    "celltypist_majority_voting": "Epithelial cells",
                },
                "supporting_genes": ["KRT8", "KRT18", "EPCAM"],
                "deg_derived_label": "epithelial cell",
                "reasoning": "Both references epithelial; DEGs KRT8/KRT18 confirm.",
                # legacy fields that must NOT appear as columns anymore:
                "validation_tier": "reference_consensus_plus_deg",
                "panglaodb_label_used": "epithelial cells",
                "competing_labels_considered": ["Fibroblasts"],
            },
        },
    }
    return a


def test_analysis_record_table_has_new_columns_and_drops_old():
    md = _assemble_analysis_record(world_state=None, adata=_adata_with_annotation())
    header = next(ln for ln in md.splitlines() if ln.startswith("| Cluster |"))
    for col in ("Scimilarity prediction", "Celltypist prediction", "Supporting genes",
                "DEG prediction", "Reasoning"):
        assert col in header, f"missing column: {col}"
    for gone in ("Tier", "PanglaoDB", "Competing", "Confidence"):
        assert gone not in header, f"removed column still present: {gone}"


def test_analysis_record_table_splits_predictions_into_cells():
    md = _assemble_analysis_record(world_state=None, adata=_adata_with_annotation())
    row = next(ln for ln in md.splitlines() if ln.startswith("| 0 |"))
    assert "luminal epithelial cell of mammary gland" in row  # scimilarity column
    assert "Epithelial cells" in row  # celltypist column
    assert "epithelial cell" in row  # DEG prediction column
    assert "KRT8" in row  # supporting genes
    # A legacy value that lived only in a removed column must not leak into the row.
    assert "reference_consensus_plus_deg" not in row
