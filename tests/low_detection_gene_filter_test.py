"""normalize_and_hvg applies the standard low-detection gene filter automatically.

Genes detected (nonzero counts) in fewer than min_cells_per_gene cells carry no
usable signal and add noise to DEG. Runs that never call run_code (the common
case — run_2026_07_12_170204 on Iris, run_2026_07_12_164515 on the Spark) used to
skip this step entirely because it was only issued as a model-driven run_code
block. It is now a guaranteed floor inside normalize_and_hvg.
"""

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from scagent.agent.tools import process_tool_call


def _adata_with_rare_genes(n=200, g=50, n_rare=8):
    """Counts matrix where the first n_rare genes are seen in <3 cells."""
    rng = np.random.RandomState(0)
    X = rng.poisson(0.6, size=(n, g)).astype(np.float32)
    for j in range(n_rare):
        X[:, j] = 0
        # detected in exactly (j % 3) cells -> 0, 1, or 2 (all < 3)
        k = j % 3
        if k:
            X[rng.choice(n, size=k, replace=False), j] = 4
    a = AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(g)]),
    )
    return a


def test_filter_removes_low_detection_genes_by_default():
    a = _adata_with_rare_genes(n_rare=8)
    before = a.n_vars
    rj, a2 = process_tool_call("normalize_and_hvg", {"n_hvg": 20}, a)
    r = json.loads(rj)
    assert r["status"] == "ok"
    meta = r["feature_removals"]["low_detection_genes"]
    assert meta["enabled"] is True
    assert meta["min_cells"] == 3
    assert meta["n_removed"] == 8
    assert r["after"]["n_genes"] == before - 8
    assert r["metrics"]["n_removed_low_detection_genes"] == 8
    # raw-counts layer stays gene-aligned with X after the slice
    assert a2.layers["raw_counts"].shape[1] == a2.n_vars


def test_min_cells_per_gene_zero_disables_the_filter():
    a = _adata_with_rare_genes(n_rare=8)
    before = a.n_vars
    rj, _ = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20, "min_cells_per_gene": 0}, a
    )
    r = json.loads(rj)
    meta = r["feature_removals"]["low_detection_genes"]
    assert meta["enabled"] is False
    assert meta["n_removed"] == 0
    # only genes that HVG/ribo touch change; the rare genes survive
    assert r["after"]["n_genes"] == before


def test_custom_threshold_removes_more():
    # min_cells=5 should remove more than the default 3 on the same data
    a = _adata_with_rare_genes(n=200, g=50, n_rare=8)
    rj_default, _ = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20}, _adata_with_rare_genes(n=200, g=50, n_rare=8)
    )
    rj_strict, _ = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20, "min_cells_per_gene": 5}, a
    )
    n_default = json.loads(rj_default)["feature_removals"]["low_detection_genes"]["n_removed"]
    n_strict = json.loads(rj_strict)["feature_removals"]["low_detection_genes"]["n_removed"]
    assert n_strict >= n_default


def test_detection_counted_on_counts_not_scaled_X():
    """If X was already log-normalized, detection must still be computed from the
    counts layer's nonzero pattern (log1p preserves zeros), so the same rare genes
    are dropped rather than none."""
    a = _adata_with_rare_genes(n_rare=6)
    # pre-populate a raw_counts layer and log-normalize X (simulate a re-run)
    a.layers["raw_counts"] = a.X.copy()
    a.X = a.X.copy()
    a.X.data = np.log1p(a.X.data)
    rj, _ = process_tool_call(
        "normalize_and_hvg", {"n_hvg": 20, "normalization_source": "raw_counts"}, a
    )
    r = json.loads(rj)
    assert r["feature_removals"]["low_detection_genes"]["n_removed"] == 6
