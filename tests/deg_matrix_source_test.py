"""DEG must default to adata.X — the log-normalized, full-gene analysis matrix
this pipeline maintains — not scanpy's implicit adata.raw. (adata.raw here is a
redundant post-log1p snapshot that can go stale.) use_raw=True / a layer remain
available as explicit overrides."""

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.analysis.deg import _resolve_expression_source


def _adata_with_raw():
    X = np.abs(np.random.RandomState(0).randn(30, 5)).astype(np.float32)  # log-norm-like
    a = AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
                var=pd.DataFrame(index=[f"g{i}" for i in range(5)]))
    a.raw = a.copy()  # redundant post-log1p snapshot
    return a


def test_default_uses_adata_X_not_raw():
    a = _adata_with_raw()
    _, layer_used, resolved_use_raw, source, issues = _resolve_expression_source(a, None, None)
    assert resolved_use_raw is False
    assert source == "adata.X"
    assert layer_used is None


def test_explicit_use_raw_true_still_honored():
    a = _adata_with_raw()
    _, _, resolved_use_raw, source, _ = _resolve_expression_source(a, None, True)
    assert resolved_use_raw is True
    assert source == "adata.raw.X"


def test_explicit_use_raw_false_uses_X():
    a = _adata_with_raw()
    _, _, resolved_use_raw, source, _ = _resolve_expression_source(a, None, False)
    assert resolved_use_raw is False
    assert source == "adata.X"
