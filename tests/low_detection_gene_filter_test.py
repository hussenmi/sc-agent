"""normalize_and_hvg applies the standard low-detection gene filter automatically.

Genes detected (nonzero counts) in too few cells carry no usable signal and add
noise to DEG. Runs that never call run_code (the common case — run_2026_07_12_170204
on Iris, run_2026_07_12_164515 on the Spark) used to skip this step entirely
because it was only issued as a model-driven run_code block. It is now a
guaranteed floor inside normalize_and_hvg, with a size-scaled default threshold
(2% of cells) and an absolute-count override.
"""

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from scagent.agent.tools import process_tool_call


def _adata_with_detection(n, detected_counts, n_filler=40):
    """Build an AnnData whose first genes have controlled detection: gene j is
    expressed (count 4) in exactly detected_counts[j] cells. Well-detected filler
    genes are appended so seurat_v3 HVG has enough features to fit (it segfaults on
    a handful of genes); fillers sit far above any low-detection threshold."""
    rng = np.random.RandomState(0)
    g0 = len(detected_counts)
    X = np.zeros((n, g0), dtype=np.float32)
    for j, d in enumerate(detected_counts):
        if d:
            X[rng.choice(n, size=int(d), replace=False), j] = 4
    filler = rng.poisson(1.0, size=(n, n_filler)).astype(np.float32)
    X = np.hstack([X, filler])
    a = AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]),
    )
    return a


def test_default_is_two_percent_of_cells():
    # n=200 -> threshold ceil(0.02*200)=4. Genes in <4 cells are dropped.
    n = 200
    detected = [1, 3, 4, 10, 100]  # first two (<4) dropped; 4,10,100 kept
    a = _adata_with_detection(n, detected)
    rj, a2 = process_tool_call("normalize_and_hvg", {"n_hvg": 20}, a)
    r = json.loads(rj)
    assert r["status"] == "ok"
    meta = r["feature_removals"]["low_detection_genes"]
    assert meta["enabled"] is True
    assert meta["threshold_basis"] == "fraction_of_cells"
    assert meta["min_cell_fraction"] == 0.02
    assert meta["min_cells"] == 4
    assert meta["n_removed"] == 2
    assert r["metrics"]["n_removed_low_detection_genes"] == 2
    # raw-counts layer stays gene-aligned with X after the slice
    assert a2.layers["raw_counts"].shape[1] == a2.n_vars


def test_threshold_scales_with_dataset_size():
    # Same relative structure, different n -> different absolute threshold.
    small = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20}, _adata_with_detection(100, [1, 5, 50])
    )
    big = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20}, _adata_with_detection(1000, [1, 5, 50])
    )
    m_small = json.loads(small[0])["feature_removals"]["low_detection_genes"]
    m_big = json.loads(big[0])["feature_removals"]["low_detection_genes"]
    assert m_small["min_cells"] == 2      # ceil(0.02*100)
    assert m_big["min_cells"] == 20       # ceil(0.02*1000)
    # a gene in 5 cells survives at n=100 but is dropped at n=1000
    assert m_small["n_removed"] == 1
    assert m_big["n_removed"] == 2


def test_fraction_zero_disables_the_filter():
    a = _adata_with_detection(200, [1, 2, 50])
    before = a.n_vars
    rj, _ = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20, "min_cell_fraction_per_gene": 0}, a
    )
    r = json.loads(rj)
    meta = r["feature_removals"]["low_detection_genes"]
    assert meta["enabled"] is False
    assert meta["n_removed"] == 0
    assert r["after"]["n_genes"] == before


def test_absolute_override_takes_precedence_over_fraction():
    # min_cells_per_gene=3 overrides the 2% fraction (which would be 4 at n=200).
    a = _adata_with_detection(200, [1, 3, 4, 50])
    rj, _ = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20, "min_cells_per_gene": 3}, a
    )
    r = json.loads(rj)
    meta = r["feature_removals"]["low_detection_genes"]
    assert meta["threshold_basis"] == "absolute_min_cells"
    assert meta["min_cells"] == 3
    # only the gene in 1 cell is < 3; the gene in exactly 3 survives
    assert meta["n_removed"] == 1


def test_absolute_override_zero_disables():
    a = _adata_with_detection(200, [1, 2, 50])
    rj, _ = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20, "min_cells_per_gene": 0}, a
    )
    meta = json.loads(rj)["feature_removals"]["low_detection_genes"]
    assert meta["enabled"] is False
    assert meta["n_removed"] == 0


def test_detection_counted_on_counts_not_scaled_X():
    """If X was already log-normalized, detection must still be computed from the
    counts layer's nonzero pattern (log1p preserves zeros)."""
    a = _adata_with_detection(200, [1, 2, 3, 80])  # threshold 4 -> drop first 3
    a.layers["raw_counts"] = a.X.copy()
    a.X = a.X.copy()
    a.X.data = np.log1p(a.X.data)
    rj, _ = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20, "normalization_source": "raw_counts"}, a
    )
    meta = json.loads(rj)["feature_removals"]["low_detection_genes"]
    assert meta["n_removed"] == 3
