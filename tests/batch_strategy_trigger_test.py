"""Regression tests for the two fixes that make the multi-sample integration
decision (`multi_sample_strategy`) fire reliably after a concat:

1. get_adata treats an empty/whitespace data_path as "use in-memory data" so
   inspect_data does not error with "No data available" (which silently skipped
   the integration checkpoint).
2. _build_post_concatenation_strategy_checkpoint derives the batch column from
   the concat code (label=/batch_key=) and the live AnnData, instead of relying
   only on world_state populated by a prior successful inspect_data.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.agent import SCAgent
from scagent.agent.tools import process_tool_call


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _adata_with(col: str | None, n_groups: int = 2, n_cells: int = 20) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.poisson(1.0, size=(n_cells, 6)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(a.n_vars)]
    if col is not None:
        labels = [str(i % n_groups) for i in range(n_cells)]
        a.obs[col] = pd.Categorical(labels)
    return a


class _FakeWorldState:
    def __init__(self, confirmed=None, data_summary=None, metadata_candidates=None):
        self._confirmed = confirmed or {}
        self.data_summary = data_summary or {}
        self.metadata_candidates = metadata_candidates or []

    def get_confirmed_value(self, key):
        return self._confirmed.get(key)


def _agent(adata, world_state) -> SCAgent:
    agent = object.__new__(SCAgent)
    agent.adata = adata
    agent.world_state = world_state
    return agent


# --------------------------------------------------------------------------- #
# Fix 1: empty / whitespace data_path falls back to in-memory data
# --------------------------------------------------------------------------- #

def test_inspect_data_empty_data_path_uses_memory():
    adata = _adata_with("replicate")
    res_json, _ = process_tool_call("inspect_data", {"data_path": "", "goal": "cluster"}, adata)
    res = json.loads(res_json)
    assert "No data available" not in res.get("message", "")
    assert res.get("status") != "error"


def test_inspect_data_whitespace_data_path_uses_memory():
    adata = _adata_with("replicate")
    res_json, _ = process_tool_call("inspect_data", {"data_path": "   ", "goal": "cluster"}, adata)
    res = json.loads(res_json)
    assert "No data available" not in res.get("message", "")
    assert res.get("status") != "error"


def test_inspect_data_no_memory_and_empty_path_still_errors():
    res_json, _ = process_tool_call("inspect_data", {"data_path": ""}, None)
    res = json.loads(res_json)
    assert res.get("status") == "error"
    assert "No data available" in res.get("message", "")


# --------------------------------------------------------------------------- #
# Fix 2: post-concat checkpoint derives the batch column without prior inspect
# --------------------------------------------------------------------------- #

OK = {"status": "ok"}


def test_checkpoint_from_anndata_concat_label_arg():
    # The exact shape from the failing run: anndata.concat(..., label='replicate')
    # with NO prior successful inspect_data (empty world_state).
    agent = _agent(_adata_with("replicate"), _FakeWorldState())
    code = "combined = anndata.concat(datasets, join='outer', label='replicate', keys=names)"
    cp = agent._build_post_concatenation_strategy_checkpoint("run_code", {"code": code}, OK)
    assert cp is not None
    assert cp["kind"] == "multi_sample_strategy"
    assert cp["partition"]["column"] == "replicate"
    assert cp["partition"]["n_groups"] == 2


def test_checkpoint_from_concat_datasets_batch_key_arg():
    agent = _agent(_adata_with("sample"), _FakeWorldState())
    code = "combined = concat_datasets(datasets, batch_key='sample', join='outer')"
    cp = agent._build_post_concatenation_strategy_checkpoint("run_code", {"code": code}, OK)
    assert cp is not None
    assert cp["partition"]["column"] == "sample"


def test_checkpoint_sniffs_default_batch_column_when_unnamed():
    # ad.concat(...) with no label= falls back to anndata's default 'batch' col.
    agent = _agent(_adata_with("batch"), _FakeWorldState())
    code = "combined = ad.concat(datasets, join='outer')"
    cp = agent._build_post_concatenation_strategy_checkpoint("run_code", {"code": code}, OK)
    assert cp is not None
    assert cp["partition"]["column"] == "batch"


def test_checkpoint_uses_live_group_count_over_stale_world_state():
    # world_state says 5 groups; the live adata has 2. Live data wins.
    ws = _FakeWorldState(data_summary={"batch_key": "replicate", "n_batches": 5})
    agent = _agent(_adata_with("replicate", n_groups=2), ws)
    code = "anndata.concat(datasets, label='replicate')"
    cp = agent._build_post_concatenation_strategy_checkpoint("run_code", {"code": code}, OK)
    assert cp is not None
    assert cp["partition"]["n_groups"] == 2


def test_no_checkpoint_single_group():
    agent = _agent(_adata_with("replicate", n_groups=1), _FakeWorldState())
    code = "anndata.concat(datasets, label='replicate')"
    cp = agent._build_post_concatenation_strategy_checkpoint("run_code", {"code": code}, OK)
    assert cp is None


def test_no_checkpoint_when_strategy_already_confirmed():
    ws = _FakeWorldState(confirmed={"multi_sample_strategy": "keep_unintegrated"})
    agent = _agent(_adata_with("replicate"), ws)
    code = "anndata.concat(datasets, label='replicate')"
    cp = agent._build_post_concatenation_strategy_checkpoint("run_code", {"code": code}, OK)
    assert cp is None


def test_no_checkpoint_for_non_concat_code():
    agent = _agent(_adata_with("replicate"), _FakeWorldState())
    code = "adata.obs['foo'] = 1"
    cp = agent._build_post_concatenation_strategy_checkpoint("run_code", {"code": code}, OK)
    assert cp is None


def test_no_checkpoint_on_error_status():
    agent = _agent(_adata_with("replicate"), _FakeWorldState())
    code = "anndata.concat(datasets, label='replicate')"
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_code", {"code": code}, {"status": "error"}
    )
    assert cp is None
