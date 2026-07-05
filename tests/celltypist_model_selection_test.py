"""CellTypist model-selection gate.

The default `Immune_All_Low.pkl` is an immune-only model; on non-immune tissue it
mislabels cells (run_2026_07_02_122548: lung epithelium annotated as T/NK). So
`run_celltypist` refuses to run the default model until the user has chosen a
tissue-appropriate model — the engine surfaces the catalog and enforces the
process, but which model fits the tissue is the model's judgment (no hardcoded
tissue->model table).
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import get_tools, process_tool_call
from scagent.agent.world_state import AgentWorldState


def _adata(with_leiden=False):
    a = ad.AnnData(X=np.abs(np.random.default_rng(0).normal(size=(20, 10))).astype("float32"))
    if with_leiden:
        a.obs["leiden"] = pd.Categorical(["0", "1"] * 10)
    return a


def _run(tool_input, adata=None, world_state=None):
    res, _ = process_tool_call("run_celltypist", tool_input, adata or _adata(), world_state=world_state)
    return json.loads(res)


def test_default_model_blocks_for_model_selection():
    d = _run({"organism": "human"})
    assert d["status"] == "needs_input"
    assert d["unavailable_reason"] == "model_selection_required"
    assert d["required_input"] == "celltypist_model_choice"
    # surfaces candidate models for the model to reason over
    assert "available_models" in d
    # tells the model to present options to the user and confirm
    assert any("pause_and_ask" in step for step in d["how_to_proceed"])


def test_confirmed_flag_passes_the_gate():
    # With confirmation, the model-selection gate is satisfied; the next
    # unmet prerequisite (clustering for majority voting) is what surfaces.
    d = _run({"organism": "human", "model_selection_confirmed": True})
    assert d.get("unavailable_reason") != "model_selection_required"
    assert "clustering" in (d.get("missing_prerequisites") or [])


def test_explicit_non_default_model_passes_the_gate():
    d = _run({"organism": "human", "model": "Healthy_Adult_Lung.pkl"})
    assert d.get("unavailable_reason") != "model_selection_required"


def test_prior_recorded_choice_passes_the_gate():
    ws = AgentWorldState()
    ws.user_preferences["celltypist_model"] = "Healthy_Adult_Lung.pkl"
    d = _run({"organism": "human"}, world_state=ws)
    assert d.get("unavailable_reason") != "model_selection_required"


def test_organism_resolved_before_model_selection():
    # Organism ambiguity is resolved first (can't list organism-compatible models
    # without it); the model-selection gate comes after.
    d = _run({})
    assert d["unavailable_reason"] == "organism_ambiguous"


def test_running_celltypist_records_model_choice_in_world_state():
    ws = AgentWorldState()
    # simulate a successful run result flowing through the world-state updater
    ws.apply_tool_result(
        "run_celltypist",
        {
            "status": "ok",
            "tool": "run_celltypist",
            "model": "Healthy_Adult_Lung.pkl",
            "annotation_key": "celltypist_majority_voting",
            "cell_type_breakdown": {"AT2": 5, "AT1": 5},
        },
    )
    assert ws.get_confirmed_value("celltypist_model") == "Healthy_Adult_Lung.pkl"


def test_schema_advertises_model_selection_confirmed():
    tools = {t["name"]: t for t in get_tools()}
    props = tools["run_celltypist"]["input_schema"]["properties"]
    assert "model_selection_confirmed" in props
