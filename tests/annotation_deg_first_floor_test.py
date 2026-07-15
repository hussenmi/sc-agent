"""Floor 2 (reframed): DEG-first derivation process floor.

Markers have the final say — but the harness encodes NO biology. It only enforces
the *process*: the model must record the label the cluster's own top DEGs indicate
(`deg_derived_label`), and must justify any override when that differs from the
final label. The engine never judges what a gene means; the model owns the biology.

Reproduces the run_2026_06_29_202520 reference-domination failure (AT2 cluster
labeled "Tem/Effector helper T cell") at the process level: the DEG-first read
conflicts with the final label and an override justification is forced.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import (
    _build_annotation_evidence_scaffold,
    _merge_evidence_over_scaffold,
    _validate_annotation_evidence,
)

AT2_DEGS = ["SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "NAPSA"]


def _adata():
    a = ad.AnnData(X=np.zeros((10, 4), dtype="float32"))
    a.obs["leiden"] = pd.Categorical(["0"] * 5 + ["1"] * 5)
    return a


def _proposal():
    return {
        "cluster_ids": ["0"],
        "cluster_key": "leiden",
        "annotation_key": "cell_type",
        "ambiguous_clusters": [],
        "reference_annotation_keys": [],
        "clusters": [{
            "cluster_id": "0",
            "top_degs": AT2_DEGS,
            "discriminating_degs": AT2_DEGS,
            "suggested_supporting_genes": AT2_DEGS[:3],
            "reference_annotations": [],
            "competing_labels": [],
        }],
    }


def _evidence(label, *, deg_derived=None, override=None):
    e = {
        "label": label,
        "confidence": "medium",
        "panglaodb_queried": False,
        "supporting_genes": ["SFTPC", "SFTPB"],
        "reasoning": "Cluster annotated from its top differentially expressed genes.",
    }
    if deg_derived is not None:
        e["deg_derived_label"] = deg_derived
    if override is not None:
        e["deg_override_justification"] = override
    return {"0": e}


def _validate(evidence):
    return _validate_annotation_evidence(
        adata=_adata(), proposal=_proposal(), evidence=evidence,
        world_state=None, apply_auto_fixes=True,
    )


def test_missing_deg_derived_label_blocks():
    rep = _validate(_evidence("Type II pneumocyte"))  # no deg_derived_label
    assert any("deg_derived_label" in f for f in rep["validation_failures"])


def test_matching_deg_derived_label_passes_floor():
    rep = _validate(_evidence("Type II pneumocyte", deg_derived="Type II pneumocyte"))
    assert not any(
        "deg_derived_label" in f or "deg_override_justification" in f
        for f in rep["validation_failures"]
    )


def test_mismatch_without_justification_blocks():
    # the run-520 pattern: DEGs say AT2, final label says T cell, no justification
    rep = _validate(_evidence("Tem/Effector helper T cell", deg_derived="Type II pneumocyte"))
    assert any("deg_override_justification" in f for f in rep["validation_failures"])


def test_mismatch_with_justification_passes_floor():
    # honest-limit boundary: the harness enforces that an override was justified,
    # it does not (and cannot) judge whether the biology is correct.
    rep = _validate(_evidence(
        "Tem/Effector helper T cell",
        deg_derived="Type II pneumocyte",
        override="Cluster co-expresses CD3D/IL32 above background; treating SFTPC as ambient AT2.",
    ))
    assert not any(
        "deg_override_justification" in f or "deg_derived_label" in f
        for f in rep["validation_failures"]
    )


def test_no_hardcoded_biology_symbols_removed():
    # the engine must not carry hardcoded marker/lineage tables anymore
    import scagent.agent.tools as t
    assert not hasattr(t, "_BROAD_LINEAGE_MARKERS")
    assert not hasattr(t, "_deg_lineage_contradiction")
    assert not hasattr(t, "_is_immune_only_celltypist_model")


# --------------------------------------------------------------------------- #
# Regression: the scaffold must NOT pre-fill deg_derived_label from the
# reference-derived proposed_label. run_2026_07_13_172738 finalized mouse islet
# clusters topped by Sst/Rbp4 (delta markers) as "type B pancreatic cell"
# (beta) because a single confident Scimilarity prior said "mostly beta" and the
# prefilled "DEG-derived" label silently agreed — so the DEG-first floor never
# fired. The independent DEG read has to be the model's own act.
# --------------------------------------------------------------------------- #

_SST_DEGS = ["Sst", "Rbp4", "Fam159b", "Cd24a", "Arg1", "Hhex"]


class _UnsAdata:
    def __init__(self, uns):
        self.uns = uns
        import anndata as _ad
        import numpy as _np
        import pandas as _pd
        self._a = _ad.AnnData(X=_np.zeros((10, 4), dtype="float32"))
        self._a.obs["leiden"] = _pd.Categorical(["0"] * 5 + ["1"] * 5)


def _delta_summary():
    """A delta cluster (Sst/Rbp4 topped) whose reference-derived proposed_label
    is the majority-prior label 'type B pancreatic cell' (beta)."""
    return {
        "cluster_id": "3",
        "validation_tier": "reference_partial_plus_deg",
        "proposed_label": "type B pancreatic cell",
        "suggested_supporting_genes": _SST_DEGS[:3],
        "discriminating_degs": _SST_DEGS,
        "top_degs": _SST_DEGS,
        "reference_annotations": [
            {"annotation_key": "scimilarity_predictions_unconstrained", "top_label": "type B pancreatic cell"},
        ],
        "reference_consensus": {"has_consensus": False},
        "competing_labels": [],
    }


def _delta_proposal():
    return {
        "cluster_ids": ["3"],
        "cluster_key": "leiden",
        "annotation_key": "cell_type",
        "ambiguous_clusters": [],
        "reference_annotation_keys": ["scimilarity_predictions_unconstrained"],
        "clusters": [{
            "cluster_id": "3",
            "top_degs": _SST_DEGS,
            "discriminating_degs": _SST_DEGS,
            "suggested_supporting_genes": _SST_DEGS[:3],
            "reference_annotations": [
                {"annotation_key": "scimilarity_predictions_unconstrained", "top_label": "type B pancreatic cell"},
            ],
            "competing_labels": [],
        }],
    }


def test_scaffold_does_not_prefill_deg_derived_label_from_reference():
    scaffold = _build_annotation_evidence_scaffold(
        [_delta_summary()], ["scimilarity_predictions_unconstrained"]
    )
    # label may default to the reference proposal, but the DEG-derived read must
    # NOT be seeded from it — that is the whole bug.
    assert scaffold["3"]["label"] == "type B pancreatic cell"
    assert scaffold["3"]["deg_derived_label"] == ""


def _validate_delta(evidence):
    scaffold = _build_annotation_evidence_scaffold(
        [_delta_summary()], ["scimilarity_predictions_unconstrained"]
    )
    adata = _UnsAdata({
        "annotation_evidence_scaffold": scaffold,
        "annotation_evidence_scaffold_fingerprint": "fp",
    })
    merged = _merge_evidence_over_scaffold(adata, "fp", evidence)
    return _validate_annotation_evidence(
        adata=adata._a, proposal=_delta_proposal(), evidence=merged,
        world_state=None, apply_auto_fixes=True,
    )


def test_reasoning_only_submission_now_blocks_on_missing_deg_label():
    # The exact old happy path: submit only reasoning, let the scaffold fill the
    # rest. Before the fix this passed with a prefilled beta "DEG-derived" label;
    # now it must block because deg_derived_label is blank in the scaffold.
    rep = _validate_delta({"3": {"reasoning": "Islet endocrine cluster; markers reviewed and lineage assigned."}})
    assert any("deg_derived_label" in f for f in rep["validation_failures"])


def test_delta_degs_kept_as_beta_requires_override_justification():
    # Model does its independent read (delta) but keeps the reference's beta label
    # with no justification -> blocked. This is what should have stopped the run.
    rep = _validate_delta({"3": {
        "deg_derived_label": "pancreatic D cell",
        "reasoning": "Top DEGs Sst/Rbp4/Fam159b are delta markers.",
    }})
    assert any("deg_override_justification" in f for f in rep["validation_failures"])


def test_delta_call_matching_degs_passes_floor():
    # Correct DEG-first outcome: change the label to the delta call the markers
    # indicate; the floor is satisfied (harness judges process, not biology).
    rep = _validate_delta({"3": {
        "label": "pancreatic D cell",
        "deg_derived_label": "pancreatic D cell",
        "supporting_genes": ["Sst", "Rbp4", "Fam159b"],
        "reasoning": "Top DEGs Sst/Rbp4/Fam159b are delta markers; overrides the beta majority prior.",
    }})
    assert not any(
        "deg_derived_label" in f or "deg_override_justification" in f
        for f in rep["validation_failures"]
    )


# --------------------------------------------------------------------------- #
# Floor 2b: local DEG-vs-marker-DB consistency — catches confident MISREADS
# where deg_derived_label == final_label but both disagree with the cluster's
# markers. Fires ONLY where the local Cytopus KB covers the lineage and its
# markers positively contradict the declared label (immune/lung/etc.). It stays
# silent on Cytopus-uncovered lineages (e.g. pancreatic endocrine), because a
# forced check there is indistinguishable from a legitimate DEG-over-reference
# override — see test_pancreas_uncovered_misread_not_gated. Cleared by a label
# fix or a gene-level deg_marker_crosscheck_note.
# --------------------------------------------------------------------------- #

def _validate_case(proposal, evidence):
    return _validate_annotation_evidence(
        adata=_adata(), proposal=proposal, evidence=evidence,
        world_state=None, apply_auto_fixes=True,
    )


def _pancreas_proposal(ref_label="pancreatic D cell"):
    return {
        "cluster_ids": ["2"],
        "cluster_key": "leiden",
        "annotation_key": "cell_type",
        "ambiguous_clusters": [],
        "reference_annotation_keys": ["scimilarity_predictions_unconstrained"],
        "clusters": [{
            "cluster_id": "2",
            "top_degs": ["Sst", "Iapp", "Enho", "Fam159b", "Ly6h", "Pdyn"],
            "discriminating_degs": ["Sst", "Enho", "Fam159b", "Ly6h", "Pdyn"],
            "suggested_supporting_genes": ["Sst", "Fam159b", "Ly6h"],
            "reference_annotations": [
                {"annotation_key": "scimilarity_predictions_unconstrained",
                 "top_label": ref_label, "top_fraction": 0.6},
            ],
            "competing_labels": [],
        }],
    }


def _pancreas_ev(label="type B pancreatic cell", deg_derived="type B pancreatic cell", **extra):
    e = {
        "label": label,
        "deg_derived_label": deg_derived,
        "confidence": "medium",
        "panglaodb_queried": False,
        "supporting_genes": ["Sst", "Iapp"],
        "reasoning": "Islet endocrine cluster annotated from its top DEGs.",
    }
    e.update(extra)
    return {"2": e}


def test_pancreas_uncovered_misread_not_gated():
    # The honest limit: a Sst-topped cluster mislabeled beta on a Cytopus-uncovered
    # lineage is NOT force-flagged. Gating it here is indistinguishable from a
    # legitimate DEG-over-reference override, which deliberately rests on its DEGs.
    rep = _validate_case(_pancreas_proposal(), _pancreas_ev())
    assert not any(
        "bc_get_panglaodb" in f.lower() or "local marker DB" in f
        for f in rep["validation_failures"]
    )


# ---- Layer 1: covered tissue (immune) ----

def _immune_proposal():
    macs = ["C1QA", "C1QB", "MRC1", "APOE", "CD68", "LYZ", "CD163"]
    return {
        "cluster_ids": ["0"],
        "cluster_key": "leiden",
        "annotation_key": "cell_type",
        "ambiguous_clusters": [],
        "reference_annotation_keys": [],
        "clusters": [{
            "cluster_id": "0",
            "top_degs": macs,
            "discriminating_degs": macs,
            "suggested_supporting_genes": macs[:3],
            "reference_annotations": [],
            "competing_labels": [{"label": "macrophage"}],
        }],
    }


def test_layer1_covered_misread_flagged_by_local_db():
    # Declared "T cell" but the top DEGs are macrophage markers Cytopus knows.
    rep = _validate_case(_immune_proposal(), {"0": {
        "label": "T cell",
        "deg_derived_label": "T cell",
        "confidence": "medium",
        "panglaodb_queried": False,
        "supporting_genes": ["C1QA", "MRC1"],
        "reasoning": "Annotated from top DEGs of the cluster.",
    }})
    assert any("local marker DB" in f for f in rep["validation_failures"])


def test_layer1_cleared_when_deg_label_matches_local_db():
    # Correct call: declared macrophage, top DEGs are macrophage markers -> no flag.
    rep = _validate_case(_immune_proposal(), {"0": {
        "label": "macrophage",
        "deg_derived_label": "macrophage",
        "confidence": "medium",
        "panglaodb_queried": False,
        "supporting_genes": ["C1QA", "MRC1"],
        "reasoning": "Macrophage markers C1QA/C1QB/MRC1 dominate the cluster.",
    }})
    assert not any("local marker DB" in f for f in rep["validation_failures"])
