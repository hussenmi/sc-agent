"""Doublet detection must validate VALUES, not just column names.

run_2026_07_02_150701 (screenshot 2026-07-05): inspect_data reported
has_doublets=True on the misharin lung dataset, which has no doublet column at
all. The cell-type column 'scanvi_label' fuzzy-matched the 'doublet_label' role
on the "label" substring (name_score 0.63) and, with a plausible categorical
structure, crossed the confidence threshold — a verdict from a name match with no
real doublet call. The fix requires doublet-like VALUES for the doublet_label
role (content validation), while still catching genuine calls under non-canonical
names.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scagent.core.inspector import inspect_data, rank_obs_semantic_candidates


def _adata(n=20):
    return ad.AnnData(X=np.abs(np.random.default_rng(0).normal(size=(n, 5))).astype("float32"))


def test_celltype_label_column_is_not_a_doublet_label():
    a = _adata()
    # cell-type column whose name contains "label" (like scanvi_label)
    a.obs["scanvi_label"] = pd.Categorical(["Macrophages", "AT2", "Secretory", "Monocytes"] * 5)
    cands = rank_obs_semantic_candidates(a, roles={"doublet_label"}).get("doublet_label", [])
    assert cands == []
    assert inspect_data(a).has_doublet_scores is False


def test_genuine_doublet_call_still_detected_under_noncanonical_name():
    a = _adata()
    a.obs["scanvi_label"] = pd.Categorical(["Macrophages", "AT2"] * 10)  # decoy
    a.obs["scrublet_call"] = pd.Categorical(["singlet", "doublet"] * 10)  # real call
    cands = rank_obs_semantic_candidates(a, roles={"doublet_label"}).get("doublet_label", [])
    assert [c.column for c in cands] == ["scrublet_call"]
    assert inspect_data(a).has_doublet_scores is True


def test_boolean_predicted_doublet_column_detected():
    a = _adata()
    a.obs["predicted_doublet"] = ([True, False] * 10)
    # canonical name is matched by inspect_data's explicit check regardless
    assert inspect_data(a).has_doublet_scores is True


def test_score_like_label_column_does_not_trigger():
    # a column named to resemble a label but holding cell types must not count
    a = _adata()
    a.obs["annotation_label"] = pd.Categorical(["Fibroblast", "Endothelial", "T cell"] * 6 + ["T cell", "T cell"])
    assert inspect_data(a).has_doublet_scores is False
