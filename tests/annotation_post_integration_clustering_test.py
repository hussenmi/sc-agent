"""Floor 1: annotation must bind to a post-integration clustering.

When the dataset has been batch-corrected, prepare_annotation must refuse to
annotate a clustering that was computed on a non-integrated representation
(the run_2026_06_29_202520 failure: annotation on a stale res-1.5 clustering
instead of the final post-scVI res-1.0 one). Reproduced via the clustering
registry's recorded use_rep.
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import process_tool_call
from scagent.core.inspector import register_clustering


def _adata_integrated(cluster_rep: str, *, with_integration: bool = True):
    rng = np.random.default_rng(0)
    n, g = 60, 30
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = pd.DataFrame(
        {"leiden": pd.Categorical(rng.integers(0, 4, n).astype(str))},
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{j}" for j in range(g)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    if with_integration:
        a.obsm["X_scVI"] = rng.normal(size=(n, 10)).astype("float32")
    register_clustering(
        a, cluster_key="leiden", method="leiden", resolution=1.0, use_rep=cluster_rep
    )
    return a


def _missing(result_json: str):
    return json.loads(result_json).get("missing_prerequisites") or []


def test_refuses_precorrection_clustering():
    a = _adata_integrated("X_pca")  # integrated data, but clustering on PCA
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    assert "post_integration_clustering" in _missing(res)


def test_refuses_unrecorded_rep_clustering():
    a = _adata_integrated(None)  # rep unknown -> conservative refuse under integration
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    assert "post_integration_clustering" in _missing(res)


def test_allows_postintegration_clustering():
    a = _adata_integrated("X_scVI")
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    assert "post_integration_clustering" not in _missing(res)


def test_override_bypasses_gate():
    a = _adata_integrated("X_pca")
    res, _ = process_tool_call(
        "prepare_annotation",
        {"cluster_key": "leiden", "allow_precorrection_clustering": True},
        a,
    )
    assert "post_integration_clustering" not in _missing(res)


def test_no_integration_no_gate():
    a = _adata_integrated("X_pca", with_integration=False)
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    assert "post_integration_clustering" not in _missing(res)
