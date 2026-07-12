"""Regression tests: PanglaoDB is OPTIONAL supplementary evidence, never a gate.

A cluster the references/Cytopus don't resolve rests on its own DEGs
(``deg_primary`` tier). It is a valid, honest call — NOT flagged for a PanglaoDB
query and NOT penalised to low confidence for lacking one. Clusters resolvable by
references/Cytopus/DEG keep their stronger tiers. (Earlier versions forced such
clusters to low confidence and flagged them ``needs_external_adjudication``; that
PanglaoDB penalty is gone — DEGs are the decision basis.)
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import _validate_annotation_evidence


def _adata():
    a = ad.AnnData(X=np.zeros((10, 4), dtype="float32"))
    a.obs["leiden"] = pd.Categorical(["0"] * 5 + ["1"] * 5)
    return a


def _hard_cluster_proposal():
    """A cluster references disagree on and Cytopus doesn't cover (CMP)."""
    return {
        "cluster_ids": ["0"],
        "cluster_key": "leiden",
        "annotation_key": "cell_type",
        "ambiguous_clusters": ["0"],
        "reference_annotation_keys": [
            "celltypist_majority_voting", "scimilarity_predictions_unconstrained"
        ],
        "clusters": [{
            "cluster_id": "0",
            "top_degs": ["MPO", "PRSS57", "AZU1", "FNDC3B", "ELANE"],
            "discriminating_degs": ["MPO", "PRSS57", "AZU1", "ELANE"],
            "suggested_supporting_genes": ["MPO", "PRSS57", "AZU1"],
            "reference_annotations": [
                {"annotation_key": "celltypist_majority_voting",
                 "top_label": "Classical monocytes", "top_fraction": 0.6},
                {"annotation_key": "scimilarity_predictions_unconstrained",
                 "top_label": "erythroid cell", "top_fraction": 0.55},
            ],
            "competing_labels": [{"label": "monocyte"}, {"label": "erythroid cell"}],
        }],
    }


def _hard_cluster_evidence(confidence="high", panglaodb_queried=False, panglaodb_label_used=None):
    e = {
        "label": "CMP",
        "deg_derived_label": "CMP",
        "confidence": confidence,
        "panglaodb_queried": panglaodb_queried,
        "supporting_genes": ["MPO", "PRSS57", "AZU1"],
        "reasoning": "CMP from myeloid progenitor DEGs MPO/PRSS57/AZU1; references disagree.",
        "source_synthesis": {
            "agreement": "CellTypist=monocyte, Scimilarity=erythroid; DEGs favor myeloid progenitor",
            "final_decision_basis": "DEG-driven CMP call; external adjudication attempted but unresolved",
        },
        "competing_labels_considered": ["monocyte", "erythroid cell"],
        "reference_annotation_support": {
            "celltypist_majority_voting": "Classical monocytes",
            "scimilarity_predictions_unconstrained": "erythroid cell",
        },
    }
    if panglaodb_label_used is not None:
        e["panglaodb_label_used"] = panglaodb_label_used
    return {"0": e}


def _validate(proposal, evidence):
    return _validate_annotation_evidence(
        adata=_adata(), proposal=proposal, evidence=evidence,
        world_state=None, apply_auto_fixes=True,
    )


def test_unresolved_rests_on_degs_not_flagged():
    rep = _validate(_hard_cluster_proposal(), _hard_cluster_evidence())
    assert rep["validation_failures"] == []
    pc = rep["per_cluster_validation"]["0"]
    # references disagree + no Cytopus → the label rests on its DEGs
    assert pc["validation_tier"] == "deg_primary"
    # no cluster is ever flagged as requiring a PanglaoDB query
    assert "0" not in rep["panglaodb_required_clusters"]


def test_unresolved_no_longer_capped_to_low():
    # The old PanglaoDB penalty (force unresolved clusters to low) is gone —
    # a DEG-grounded call keeps its confidence.
    for conf in ("high", "medium"):
        rep = _validate(_hard_cluster_proposal(), _hard_cluster_evidence(confidence=conf))
        assert rep["validation_failures"] == []
        assert rep["per_cluster_validation"]["0"]["confidence"] != "low"


def test_no_panglaodb_queried_must_be_true_failure():
    # the exact failure from run_2026_06_11_112506 must no longer appear
    rep = _validate(_hard_cluster_proposal(), _hard_cluster_evidence(panglaodb_queried=False))
    assert not any("panglaodb_queried must be true" in f for f in rep["validation_failures"])


def test_incompatible_panglaodb_claim_dropped_not_failed():
    # agent forces an incompatible label onto a hard cluster: drop it, don't hard-fail
    rep = _validate(
        _hard_cluster_proposal(),
        _hard_cluster_evidence(panglaodb_queried=True, panglaodb_label_used="T cells"),
    )
    assert not any("biologically compatible" in f for f in rep["validation_failures"])
    assert rep["validation_failures"] == []
    # the incompatible PanglaoDB claim is dropped, not turned into a low-confidence penalty
    assert rep["per_cluster_validation"]["0"].get("panglaodb_queried") in (False, None)


def test_resolvable_cluster_unaffected():
    # references AGREE + DEG support → reference_consensus_plus_deg, NOT downgraded
    proposal = {
        "cluster_ids": ["0"],
        "cluster_key": "leiden",
        "annotation_key": "cell_type",
        "ambiguous_clusters": [],
        "reference_annotation_keys": [
            "celltypist_majority_voting", "scimilarity_predictions_unconstrained"
        ],
        "clusters": [{
            "cluster_id": "0",
            "top_degs": ["LYZ", "CD14", "S100A8", "S100A9", "FCN1"],
            "discriminating_degs": ["LYZ", "CD14", "S100A8", "FCN1"],
            "suggested_supporting_genes": ["LYZ", "CD14", "S100A8"],
            "reference_annotations": [
                {"annotation_key": "celltypist_majority_voting",
                 "top_label": "Classical monocytes", "top_fraction": 0.7},
                {"annotation_key": "scimilarity_predictions_unconstrained",
                 "top_label": "monocyte", "top_fraction": 0.65},
            ],
            "competing_labels": [],
        }],
    }
    evidence = {"0": {
        "label": "Classical monocyte",
        "deg_derived_label": "Classical monocyte",
        "confidence": "high",
        "panglaodb_queried": False,
        "supporting_genes": ["LYZ", "CD14", "S100A8"],
        "reasoning": "Classical monocyte: LYZ/CD14/S100A8 with concordant CellTypist+Scimilarity.",
        "source_synthesis": {
            "agreement": "CellTypist and Scimilarity both monocyte; DEGs concordant",
            "final_decision_basis": "two-source reference consensus + DEG support",
        },
        "competing_labels_considered": [],
        "reference_annotation_support": {
            "celltypist_majority_voting": "Classical monocytes",
            "scimilarity_predictions_unconstrained": "monocyte",
        },
    }}
    rep = _validate(proposal, evidence)
    pc = rep["per_cluster_validation"]["0"]
    # resolved by references/Cytopus/DEG — NOT the unadjudicated tier
    assert pc["validation_tier"] in {
        "reference_consensus_plus_deg", "reference_partial_plus_deg", "cytopus_plus_deg"
    }
    assert pc["validation_tier"] != "reference_deg_unadjudicated"
    assert "external_adjudication_status" not in pc
    assert "0" not in rep["panglaodb_required_clusters"]
    assert pc["confidence"] in {"high", "medium"}  # not forced to low by the unresolved path
