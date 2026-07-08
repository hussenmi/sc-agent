"""Regression: a raw-count X stored as float32 with no separate raw layer
(e.g. *_raw.h5ad files like Reyfman) must report raw counts as available.

Before the fix, the snapshot's `has_raw_counts` was wired to `has_raw_layer`
only, so such files showed `has_raw_counts: false` even though X *is* raw
integer counts — confusing the model and wrongly flagging normalize as blocked.
"""
from __future__ import annotations

import numpy as np
import anndata as ad

from scagent.core.inspector import inspect_data
from scagent.agent.returns import make_state_dict
from scagent.agent.world_state import AgentWorldState


def _raw_float32_adata():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.5, size=(200, 50)).astype("float32")  # integer values, float dtype
    a = ad.AnnData(X=X)
    a.var_names = [f"G{i}" for i in range(a.n_vars)]
    return a


def test_inspector_flags_float32_integer_X_as_counts():
    a = _raw_float32_adata()
    state = inspect_data(a)
    assert state.is_counts is True          # X recognised as raw counts
    assert state.has_raw_layer is False     # no separate raw layer
    assert state.is_normalized is False


def test_state_dict_reports_raw_available_for_raw_X():
    state = inspect_data(_raw_float32_adata())
    d = make_state_dict(state)
    assert d["has_raw_counts"] is True       # available (in X), not just-a-layer
    assert d["x_is_raw_counts"] is True


def test_world_state_snapshot_reports_raw_X():
    ws = AgentWorldState()
    ws.sync_from_adata(_raw_float32_adata())
    proc = ws.data_summary["processing"]
    assert proc["has_raw_counts"] is True
    assert proc["x_is_raw_counts"] is True
    assert proc["is_normalized"] is False
