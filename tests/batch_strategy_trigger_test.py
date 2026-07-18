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
    def __init__(self, confirmed=None, data_summary=None, metadata_candidates=None,
                 group_count=None):
        self._confirmed = confirmed or {}
        self.data_summary = data_summary or {}
        self.metadata_candidates = metadata_candidates or []
        self._group_count = group_count

    def get_confirmed_value(self, key):
        return self._confirmed.get(key)

    def _multi_sample_group_count(self):
        if self._group_count is not None:
            return self._group_count
        n = int(self.data_summary.get("n_batches") or 0)
        for cand in self.metadata_candidates:
            try:
                n = max(n, int(cand.get("n_unique") or 0))
            except (TypeError, ValueError):
                continue
        return n


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
# Fix 2: post-concat checkpoint is STATE-based (single->multi transition), not a
# regex on the run_code source — so it fires however the combine was written.
# --------------------------------------------------------------------------- #

OK = {"status": "ok"}


def test_checkpoint_fires_on_multi_sample_transition_via_obs_sniff():
    # No recorded batch key: the column is sniffed from the combined obs. The
    # trigger is the single->multi transition, not any concat call in the code.
    ws = _FakeWorldState(group_count=2)
    agent = _agent(_adata_with("replicate"), ws)
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_code", {"code": "adata = build_from_frames(frames)"}, OK, before_group_count=1
    )
    assert cp is not None
    assert cp["kind"] == "multi_sample_strategy"
    assert cp["partition"]["column"] == "replicate"
    assert cp["partition"]["n_groups"] == 2


def test_checkpoint_uses_recorded_batch_key_when_present():
    ws = _FakeWorldState(confirmed={"batch_key": "sample"}, group_count=2)
    agent = _agent(_adata_with("sample"), ws)
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_code", {"code": "x = 1"}, OK, before_group_count=1
    )
    assert cp is not None
    assert cp["partition"]["column"] == "sample"


def test_checkpoint_fires_for_hand_built_combine_with_no_concat_call():
    # The run_2026_07_15_121111 bypass: a combine built from stacked frames with
    # NO anndata.concat / concat_datasets call. The old regex missed this; the
    # state-based transition catches it.
    ws = _FakeWorldState(group_count=40)
    agent = _agent(_adata_with("sample", n_groups=40, n_cells=80), ws)
    manual = (
        "mats = [pd.read_csv(f, index_col=0) for f in files]\n"
        "adata = anndata.AnnData(scipy.sparse.vstack([m.T.values for m in mats]))"
    )
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_code", {"code": manual}, OK, before_group_count=1
    )
    assert cp is not None
    assert cp["partition"]["n_groups"] == 40


def test_load_data_also_triggers_the_transition_checkpoint():
    ws = _FakeWorldState(group_count=2)
    agent = _agent(_adata_with("replicate"), ws)
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "load_data", {}, OK, before_group_count=1
    )
    assert cp is not None


def test_no_checkpoint_without_transition_when_already_multi_sample():
    # Data was already multi-sample before this tool ran — no transition, so the
    # post-concat net does not re-fire (inspect_data / the enforcement own it).
    ws = _FakeWorldState(group_count=2)
    agent = _agent(_adata_with("replicate"), ws)
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_code", {"code": "x = 1"}, OK, before_group_count=2
    )
    assert cp is None


def test_no_checkpoint_single_group():
    ws = _FakeWorldState(group_count=1)
    agent = _agent(_adata_with("replicate", n_groups=1), ws)
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_code", {"code": "x = 1"}, OK, before_group_count=0
    )
    assert cp is None


def test_no_checkpoint_when_strategy_already_confirmed():
    ws = _FakeWorldState(confirmed={"multi_sample_strategy": "keep_unintegrated"}, group_count=2)
    agent = _agent(_adata_with("replicate"), ws)
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_code", {"code": "x = 1"}, OK, before_group_count=1
    )
    assert cp is None


def test_no_checkpoint_for_unrelated_tool():
    ws = _FakeWorldState(group_count=2)
    agent = _agent(_adata_with("replicate"), ws)
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_pca", {}, OK, before_group_count=1
    )
    assert cp is None


def test_no_checkpoint_on_error_status():
    ws = _FakeWorldState(group_count=2)
    agent = _agent(_adata_with("replicate"), ws)
    cp = agent._build_post_concatenation_strategy_checkpoint(
        "run_code", {"code": "x = 1"}, {"status": "error"}, before_group_count=1
    )
    assert cp is None
