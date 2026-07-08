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

from scagent.agent.tools import _validate_annotation_evidence

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
