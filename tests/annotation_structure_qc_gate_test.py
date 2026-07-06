"""prepare_annotation must not proceed until cluster structure QC has run.

run_2026_07_05_225406 went run_cluster_qc -> straight to CellTypist/annotation,
skipping run_cluster_structure_qc entirely. The terminal obligation only fires
when the run tries to END, so a run that never converges annotation never runs
structure QC. This gate refuses prepare_annotation until structure QC has run on
the active clustering (freshness via world_state registry; adata.uns fallback),
with an explicit opt-out.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import process_tool_call
from scagent.agent.world_state import AgentWorldState
from scagent.core.inspector import register_clustering


def _adata(n=60, g=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.log1p(rng.poisson(1.0, size=(n, g)).astype("float32"))
    obs = pd.DataFrame(
        {"leiden": pd.Categorical(rng.integers(0, 3, n).astype(str))},
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    register_clustering(a, cluster_key="leiden", method="leiden", resolution=1.0)
    return a


def _missing(res):
    return json.loads(res).get("missing_prerequisites", []) or []


def test_prepare_refuses_without_structure_qc():
    a = _adata()
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    assert "cluster_structure_qc" in _missing(res)


def test_prepare_allowed_after_structure_qc_recorded_in_uns():
    # no world_state -> gate uses the adata.uns fallback
    a = _adata()
    a.uns["cluster_structure_qc"] = {"leiden": {"structure_qc_run_id": "sq1"}}
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    assert "cluster_structure_qc" not in _missing(res)


def test_prepare_allowed_when_world_state_registry_has_structure_qc():
    a = _adata()
    ws = AgentWorldState()
    ws.cluster_qc_registry = {
        "leiden": {"cluster_key": "leiden", "checked_at": "t0", "structure_qc_run_id": "sq1"}
    }
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a, world_state=ws)
    assert "cluster_structure_qc" not in _missing(res)


def test_world_state_registry_without_structure_qc_still_refuses():
    # metric QC ran but structure QC did not -> gate fires (the 225406 case)
    a = _adata()
    ws = AgentWorldState()
    ws.cluster_qc_registry = {"leiden": {"cluster_key": "leiden", "checked_at": "t0"}}
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a, world_state=ws)
    assert "cluster_structure_qc" in _missing(res)


def test_explicit_opt_out_bypasses_structure_qc_gate():
    a = _adata()
    res, _ = process_tool_call(
        "prepare_annotation", {"cluster_key": "leiden", "allow_skip_structure_qc": True}, a
    )
    assert "cluster_structure_qc" not in _missing(res)
