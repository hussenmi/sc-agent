"""Tests for the annotation evidence scaffold.

`prepare_annotation` now emits a ready-to-edit evidence scaffold (every
mechanically-derivable field pre-filled, `reasoning` blank) and
`stage_annotation_evidence`/`finalize_annotation` overlay the model's submitted
fields on top of it. This removes the proposal→evidence reverse-engineering loop
and the reference_annotation_support / competing_labels_considered failures seen
in run_2026_07_02_002825 (32 clusters failing ref-support, cluster 17 failing on
empty competing labels, plus a premature finalize misread as a persistence bug).
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import (
    _build_annotation_evidence_scaffold,
    _merge_evidence_over_scaffold,
    _validate_annotation_evidence,
    process_tool_call,
)
from scagent.core.inspector import register_clustering

REF_KEYS = ["celltypist_majority_voting", "scimilarity_predictions_unconstrained"]


class _FakeAdata:
    """Minimal stand-in for AnnData: only .uns is touched by the merge helper."""

    def __init__(self, uns):
        self.uns = uns


def _adata_obs():
    a = ad.AnnData(X=np.zeros((10, 4), dtype="float32"))
    a.obs["leiden"] = pd.Categorical(["0"] * 5 + ["1"] * 5)
    return a


def _consensus_summary():
    return {
        "cluster_id": "1",
        "validation_tier": "reference_consensus_plus_deg",
        "proposed_label": "Endothelial cells",
        "suggested_supporting_genes": ["CLDN5", "PECAM1", "VWF"],
        "reference_annotations": [
            {"annotation_key": "celltypist_majority_voting", "top_label": "Endothelial cells"},
            {"annotation_key": "scimilarity_predictions_unconstrained", "top_label": "endothelial cell"},
        ],
        "reference_consensus": {"has_consensus": True},
        "competing_labels": [],
    }


def _reference_ambiguous_summary():
    """Cluster-17 pattern: references disagree, so competing_labels is empty."""
    return {
        "cluster_id": "17",
        "validation_tier": "reference_partial_plus_deg",
        "proposed_label": "Alveolar macrophages",
        "suggested_supporting_genes": ["MRC1", "INHBA", "OLR1"],
        "reference_annotations": [
            {"annotation_key": "celltypist_majority_voting", "top_label": "Alveolar macrophages"},
            {"annotation_key": "scimilarity_predictions_unconstrained", "top_label": "macrophage"},
        ],
        "reference_consensus": {"has_consensus": False},
        "competing_labels": [],
    }


# --------------------------------------------------------------------------- #
# scaffold builder
# --------------------------------------------------------------------------- #

def test_scaffold_prefills_derivable_fields_leaves_reasoning_blank():
    scaffold = _build_annotation_evidence_scaffold([_consensus_summary()], REF_KEYS)
    entry = scaffold["1"]
    assert entry["label"] == "Endothelial cells"
    assert entry["deg_derived_label"] == "Endothelial cells"
    assert entry["supporting_genes"] == ["CLDN5", "PECAM1", "VWF"]
    assert entry["confidence"] == "high"  # reference_consensus_plus_deg ceiling
    assert entry["panglaodb_queried"] is False
    assert entry["reference_annotation_support"] == {
        "celltypist_majority_voting": "Endothelial cells",
        "scimilarity_predictions_unconstrained": "endothelial cell",
    }
    assert entry["source_synthesis"]["agreement"] == "reference_consensus"
    # reasoning + final_decision_basis are the model's job and stay blank
    assert entry["reasoning"] == ""
    assert entry["source_synthesis"]["final_decision_basis"] == ""


def test_scaffold_confidence_ceiling_by_tier():
    tiers = {
        "reference_consensus_plus_deg": "high",
        "cytopus_plus_deg": "medium",
        "reference_partial_plus_deg": "medium",
        "needs_external_adjudication": "low",
    }
    for tier, conf in tiers.items():
        s = _consensus_summary()
        s["validation_tier"] = tier
        assert _build_annotation_evidence_scaffold([s], REF_KEYS)["1"]["confidence"] == conf


def test_scaffold_competing_labels_from_reference_when_proposal_empty():
    # The cluster-17 regression: competing_labels empty in the proposal, but the
    # differing reference label is the alternative that was considered.
    scaffold = _build_annotation_evidence_scaffold([_reference_ambiguous_summary()], REF_KEYS)
    assert scaffold["17"]["competing_labels_considered"] == ["macrophage"]


def test_scaffold_no_reference_keys_omits_reference_support():
    s = _consensus_summary()
    s["reference_annotations"] = []
    scaffold = _build_annotation_evidence_scaffold([s], [])
    assert "reference_annotation_support" not in scaffold["1"]
    assert scaffold["1"]["source_synthesis"]["agreement"] == "no_reference"


def test_scaffold_falls_back_to_discriminating_degs_for_supporting_genes():
    s = _consensus_summary()
    s["suggested_supporting_genes"] = []
    s["discriminating_degs"] = ["CLDN5", "PECAM1", "VWF", "SPARCL1", "EPAS1", "TIMP3", "A2M"]
    scaffold = _build_annotation_evidence_scaffold([s], REF_KEYS)
    assert scaffold["1"]["supporting_genes"] == ["CLDN5", "PECAM1", "VWF", "SPARCL1", "EPAS1", "TIMP3"]


# --------------------------------------------------------------------------- #
# merge helper
# --------------------------------------------------------------------------- #

def _adata_with_scaffold(scaffold, fingerprint="fp1"):
    return _FakeAdata({
        "annotation_evidence_scaffold": scaffold,
        "annotation_evidence_scaffold_fingerprint": fingerprint,
    })


def test_merge_overlays_reasoning_keeps_scaffold_defaults():
    scaffold = _build_annotation_evidence_scaffold(
        [_consensus_summary(), _reference_ambiguous_summary()], REF_KEYS
    )
    adata = _adata_with_scaffold(scaffold)
    merged = _merge_evidence_over_scaffold(adata, "fp1", {"1": {"reasoning": "Endo: CLDN5/PECAM1/VWF."}})
    # cluster 1 gets the model's reasoning but keeps scaffold label/genes
    assert merged["1"]["reasoning"] == "Endo: CLDN5/PECAM1/VWF."
    assert merged["1"]["label"] == "Endothelial cells"
    # cluster 17 still present (from scaffold) with blank reasoning
    assert merged["17"]["reasoning"] == ""


def test_merge_model_field_overrides_scaffold():
    scaffold = _build_annotation_evidence_scaffold([_consensus_summary()], REF_KEYS)
    adata = _adata_with_scaffold(scaffold)
    merged = _merge_evidence_over_scaffold(
        adata, "fp1", {"1": {"label": "Lymphatic EC", "reasoning": "PROX1 high"}}
    )
    assert merged["1"]["label"] == "Lymphatic EC"


def test_merge_ignores_scaffold_on_fingerprint_mismatch():
    scaffold = _build_annotation_evidence_scaffold([_consensus_summary()], REF_KEYS)
    adata = _adata_with_scaffold(scaffold, fingerprint="fp1")
    merged = _merge_evidence_over_scaffold(adata, "fp_other", {"1": {"reasoning": "x"}})
    assert set(merged.keys()) == {"1"}
    assert "label" not in merged["1"]  # scaffold NOT used → only model fields


def test_merge_no_scaffold_is_passthrough():
    adata = _FakeAdata({})
    merged = _merge_evidence_over_scaffold(adata, "fp1", {"1": {"reasoning": "x"}})
    assert merged == {"1": {"reasoning": "x"}}


# --------------------------------------------------------------------------- #
# validator auto-fills (defense-in-depth for direct submissions)
# --------------------------------------------------------------------------- #

def _proposal_two_ref_sources(cluster_id="0", ambiguous=False):
    return {
        "cluster_ids": [cluster_id],
        "cluster_key": "leiden",
        "annotation_key": "cell_type",
        "ambiguous_clusters": [cluster_id] if ambiguous else [],
        "reference_annotation_keys": REF_KEYS,
        "clusters": [{
            "cluster_id": cluster_id,
            "top_degs": ["MRC1", "INHBA", "OLR1", "GRN", "C1QA"],
            "discriminating_degs": ["MRC1", "INHBA", "OLR1", "GRN"],
            "suggested_supporting_genes": ["MRC1", "INHBA", "OLR1"],
            "reference_annotations": [
                {"annotation_key": "celltypist_majority_voting", "top_label": "Alveolar macrophages"},
                {"annotation_key": "scimilarity_predictions_unconstrained", "top_label": "macrophage"},
            ],
            "competing_labels": [],
        }],
    }


def _minimal_evidence(cluster_id="0", **overrides):
    ev = {
        "label": "Alveolar macrophages",
        "deg_derived_label": "Alveolar macrophages",
        "confidence": "medium",
        "panglaodb_queried": False,
        "supporting_genes": ["MRC1", "INHBA", "OLR1"],
        "reasoning": "Alveolar macrophage markers MRC1/INHBA/OLR1; both references agree on lineage.",
        "source_synthesis": {"agreement": "single_reference_source", "final_decision_basis": "DEG + reference"},
    }
    ev.update(overrides)
    return {cluster_id: ev}


def test_validator_autofills_reference_annotation_support():
    # No reference_annotation_support submitted -> derived from proposal, not a failure.
    rep = _validate_annotation_evidence(
        adata=_adata_obs(),
        proposal=_proposal_two_ref_sources(),
        evidence=_minimal_evidence(),
        world_state=None,
        apply_auto_fixes=True,
    )
    assert not any("reference_annotation_support" in f for f in rep["validation_failures"])
    filled = rep["evidence_str"]["0"]["reference_annotation_support"]
    assert filled["celltypist_majority_voting"] == "Alveolar macrophages"
    assert filled["scimilarity_predictions_unconstrained"] == "macrophage"
    assert any("reference_annotation_support" in fix for fix in rep["auto_fixes"])


def test_validator_autofills_competing_labels_from_reference_for_ambiguous():
    # Reference-ambiguous cluster (cluster-17 pattern): empty competing_labels in
    # the proposal, no competing_labels_considered submitted -> filled from the
    # differing reference label instead of hard-failing.
    rep = _validate_annotation_evidence(
        adata=_adata_obs(),
        proposal=_proposal_two_ref_sources(ambiguous=True),
        evidence=_minimal_evidence(),
        world_state=None,
        apply_auto_fixes=True,
    )
    assert not any("competing_labels_considered" in f for f in rep["validation_failures"])
    considered = rep["evidence_str"]["0"]["competing_labels_considered"]
    assert "macrophage" in considered


def test_validator_still_requires_reasoning():
    # The one field the scaffold cannot fill: blank reasoning must still fail.
    rep = _validate_annotation_evidence(
        adata=_adata_obs(),
        proposal=_proposal_two_ref_sources(),
        evidence=_minimal_evidence(reasoning=""),
        world_state=None,
        apply_auto_fixes=True,
    )
    assert any("reasoning" in f for f in rep["validation_failures"])


# --------------------------------------------------------------------------- #
# end-to-end through process_tool_call: prepare -> stage(reasoning only) -> finalize
# --------------------------------------------------------------------------- #

def _integration_adata(seed=1, n=90, g=40):
    rng = np.random.default_rng(seed)
    X = np.log1p(rng.poisson(1.0, size=(n, g)).astype("float32"))
    obs = pd.DataFrame(
        {"leiden": pd.Categorical(rng.integers(0, 3, n).astype(str))},
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    a.obsm["X_scVI"] = rng.normal(size=(n, 10)).astype("float32")
    a.obs["celltypist_majority_voting"] = pd.Categorical(
        rng.choice(["T cell", "B cell", "Monocyte"], n)
    )
    a.obs["scimilarity_predictions_unconstrained"] = pd.Categorical(
        rng.choice(["t cell", "b cell", "monocyte"], n)
    )
    register_clustering(a, cluster_key="leiden", method="leiden", resolution=1.0, use_rep="X_scVI")
    # prepare_annotation now gates on cluster structure QC having run for this
    # clustering; these tests exercise the scaffold, not that gate, so mark it done.
    a.uns["cluster_structure_qc"] = {"leiden": {"structure_qc_run_id": "test-structure-qc"}}
    return a


def test_prepare_writes_scaffold_to_uns():
    a = _integration_adata()
    res, a = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    assert json.loads(res).get("evidence_scaffold_ready") is True
    scaffold = a.uns["annotation_evidence_scaffold"]
    assert set(scaffold.keys()) == {"0", "1", "2"}
    assert all(entry["reasoning"] == "" for entry in scaffold.values())


def test_finalize_without_reasoning_gives_targeted_message_not_persistence_error():
    a = _integration_adata()
    _, a = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    res, _ = process_tool_call(
        "finalize_annotation", {"annotation_key": "cell_type", "cluster_key": "leiden"}, a
    )
    r = json.loads(res)
    assert r["status"] == "error"
    # names the scaffold + reasoning; must NOT read like a generic "evidence required"
    assert "scaffold" in r["message"].lower()
    assert "reasoning" in r["message"].lower()


def test_stage_with_reasoning_only_then_finalize_writes_labels():
    a = _integration_adata()
    _, a = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    clusters = list(a.uns["annotation_evidence_scaffold"].keys())
    # The model supplies ONLY reasoning; every other field comes from the scaffold.
    ev = {
        c: {"reasoning": f"Cluster {c}: lineage from reference consensus + DEG support; markers reviewed."}
        for c in clusters
    }
    res, a = process_tool_call("stage_annotation_evidence", {"evidence_summary": ev}, a)
    r = json.loads(res)
    assert r["ready_to_finalize"] is True
    assert r["clusters_failing"] == []
    assert r["coverage"]["n_with_reasoning"] == len(clusters)

    res, a = process_tool_call(
        "finalize_annotation", {"annotation_key": "cell_type", "cluster_key": "leiden"}, a
    )
    assert json.loads(res)["status"] == "ok"
    assert "cell_type" in a.obs.columns


def test_finalize_does_not_clobber_preexisting_annotation():
    # A pre-existing 'cell_type' (e.g. the source paper's labels) must be preserved;
    # scagent writes to 'cell_type_scagent' by default instead of overwriting.
    a = _integration_adata()
    a.obs["cell_type"] = pd.Categorical(["SourcePaperLabel"] * a.n_obs)
    _, a = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    clusters = list(a.uns["annotation_evidence_scaffold"].keys())
    ev = {c: {"reasoning": f"Cluster {c}: lineage from reference consensus + DEG support; reviewed."} for c in clusters}
    _, a = process_tool_call("stage_annotation_evidence", {"evidence_summary": ev}, a)
    res, a = process_tool_call("finalize_annotation", {"cluster_key": "leiden"}, a)
    r = json.loads(res)
    assert r["status"] == "ok"
    assert r["annotation_key"] == "cell_type_scagent"
    assert r["annotation_key_redirected_from"] == "cell_type"
    # original preserved untouched
    assert list(a.obs["cell_type"].astype(str).unique()) == ["SourcePaperLabel"]
    assert "cell_type_scagent" in a.obs.columns


def test_finalize_overwrite_true_replaces_preexisting():
    a = _integration_adata()
    a.obs["cell_type"] = pd.Categorical(["SourcePaperLabel"] * a.n_obs)
    _, a = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    clusters = list(a.uns["annotation_evidence_scaffold"].keys())
    ev = {c: {"reasoning": f"Cluster {c}: lineage from reference consensus + DEG support; reviewed."} for c in clusters}
    _, a = process_tool_call("stage_annotation_evidence", {"evidence_summary": ev}, a)
    res, a = process_tool_call(
        "finalize_annotation", {"cluster_key": "leiden", "annotation_key": "cell_type", "overwrite": True}, a
    )
    r = json.loads(res)
    assert r["annotation_key"] == "cell_type"
    assert r["annotation_key_redirected_from"] is None
    assert "SourcePaperLabel" not in list(a.obs["cell_type"].astype(str).unique())


def test_stage_reports_clusters_awaiting_reasoning():
    a = _integration_adata()
    _, a = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    # reasoning for only one cluster
    ev = {"0": {"reasoning": "Cluster 0: T-cell lineage from CD3-like DEG support; reviewed."}}
    res, a = process_tool_call("stage_annotation_evidence", {"evidence_summary": ev}, a)
    r = json.loads(res)
    assert r["ready_to_finalize"] is False
    assert set(r["clusters_awaiting_reasoning"]) == {"1", "2"}
    assert r["coverage"]["n_with_reasoning"] == 1
