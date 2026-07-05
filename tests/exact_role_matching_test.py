"""Column-role detection uses EXACT names + content, never fuzzy matching.

Fuzzy/approximate name matching (shared-token overlap, edit-distance ratio,
substring containment) produced false role verdicts — the clearest being a
cell-type column 'scanvi_label' scoring 0.63 for the 'doublet_label' role purely
because of the shared 'label' token (run_2026_07_02_150701 screenshot). The
harness now recognizes canonical names EXACTLY and otherwise defers to content
facts / the model. No hardcoded biology (cell-type value vocabulary) either.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scagent.core.inspector import (
    SEMANTIC_OBS_ROLE_ALIASES,
    _column_name_role_scores,
    rank_obs_semantic_candidates,
)


# --- the matcher itself: exact only ------------------------------------------
def test_exact_canonical_names_match():
    assert _column_name_role_scores("cell_type", SEMANTIC_OBS_ROLE_ALIASES).get("cell_type") == 1.0
    assert _column_name_role_scores("leiden", SEMANTIC_OBS_ROLE_ALIASES).get("cluster") == 1.0
    assert _column_name_role_scores("pct_counts_mt", SEMANTIC_OBS_ROLE_ALIASES).get("qc_pct_mt") == 1.0
    # punctuation/case-insensitive but still exact
    assert _column_name_role_scores("percent.MT", SEMANTIC_OBS_ROLE_ALIASES).get("qc_pct_mt") == 1.0
    assert _column_name_role_scores("CellType", SEMANTIC_OBS_ROLE_ALIASES).get("cell_type") == 1.0
    assert _column_name_role_scores("sample_id", SEMANTIC_OBS_ROLE_ALIASES).get("sample") == 1.0


def test_shared_token_does_not_match():
    # 'scanvi_label' shares the token 'label' with doublet_label/cell_label — must
    # NOT score for either role.
    scores = _column_name_role_scores("scanvi_label", SEMANTIC_OBS_ROLE_ALIASES)
    assert scores.get("doublet_label", 0.0) == 0.0
    assert scores.get("cell_type", 0.0) == 0.0
    # 'cluster_annotation' shares tokens with cluster and cell_type — no match.
    scores2 = _column_name_role_scores("cluster_annotation", SEMANTIC_OBS_ROLE_ALIASES)
    assert scores2.get("cluster", 0.0) == 0.0
    # a made-up batch-ish name is not matched without an exact alias
    assert _column_name_role_scores("my_batch_thing", SEMANTIC_OBS_ROLE_ALIASES).get("batch", 0.0) == 0.0


def test_id_suffix_is_not_guessed():
    # '_id' token guessing removed: only exact aliases match.
    assert _column_name_role_scores("channel_id", SEMANTIC_OBS_ROLE_ALIASES).get("batch", 0.0) == 0.0
    # but an explicit alias still matches
    assert _column_name_role_scores("donor_id", SEMANTIC_OBS_ROLE_ALIASES).get("donor") == 1.0


# --- end-to-end candidate ranking --------------------------------------------
def _adata_multi_annotation():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=np.abs(rng.normal(size=(30, 5))).astype("float32"))
    ct = ["T cell", "B cell", "Macrophage"] * 10
    a.obs["cell_type"] = pd.Categorical(ct)          # canonical name
    a.obs["scanvi_label"] = pd.Categorical(ct)       # same values, non-canonical name
    a.obs["ann_level_1"] = pd.Categorical(ct)        # non-canonical name
    return a


def test_only_canonically_named_celltype_is_autodetected():
    a = _adata_multi_annotation()
    cands = rank_obs_semantic_candidates(a, roles={"cell_type"}).get("cell_type", [])
    cols = [c.column for c in cands]
    assert "cell_type" in cols
    # non-canonical annotation columns are NOT auto-labeled (model decides from facts)
    assert "scanvi_label" not in cols
    assert "ann_level_1" not in cols


def test_celltype_detected_without_hardcoded_biology():
    # cell_type is found via exact name + categorical structure, NOT by matching
    # values against a baked-in list of cell-type names.
    rng = np.random.default_rng(1)
    a = ad.AnnData(X=np.abs(rng.normal(size=(30, 5))).astype("float32"))
    # values are NOT recognizable biology terms, but the column is named cell_type
    a.obs["cell_type"] = pd.Categorical(["type_alpha", "type_beta", "type_gamma"] * 10)
    cands = rank_obs_semantic_candidates(a, roles={"cell_type"}).get("cell_type", [])
    assert [c.column for c in cands] == ["cell_type"]
