"""Safety net that makes SCAGENT_MODEL_INSPECTION safe to default: if the model
skips record_inspection, it's nudged once before the first analysis step; if it
still skips, the run falls back to the deterministic heuristic (logged once), so
the flag can never leave a run worse off than the heuristic path.
inspection_source records which path produced the judgments."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.agent.agent import SCAgent
from scagent.agent.world_state import AgentWorldState


def _agent():
    a = object.__new__(SCAgent)
    a.world_state = AgentWorldState()
    a.adata = SimpleNamespace()  # non-None sentinel (gate only checks is None)
    a._inspection_nudged = False
    a._inspection_fallback_logged = False
    return a


def _adata():
    obs = pd.DataFrame(
        {"donor": [f"D{i % 4}" for i in range(20)]},
        index=[f"c{i}" for i in range(20)],
    )
    return AnnData(
        X=np.zeros((20, 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["HLA-A", "CD3D", "GAPDH"]),
    )


# --- gate decision ------------------------------------------------------------

def test_gate_off_when_flag_off(monkeypatch):
    # Model inspection now defaults ON, so OFF must be set explicitly.
    monkeypatch.setenv("SCAGENT_MODEL_INSPECTION", "0")
    assert _agent()._inspection_gate_action("run_qc") is None


def test_gate_on_by_default(monkeypatch):
    # Default (unset) is ON: an analysis step without a recorded inspection nudges.
    monkeypatch.delenv("SCAGENT_MODEL_INSPECTION", raising=False)
    assert _agent()._inspection_gate_action("run_qc") == "nudge"


def test_gate_nudges_then_falls_back(monkeypatch):
    monkeypatch.setenv("SCAGENT_MODEL_INSPECTION", "1")
    a = _agent()
    assert a._inspection_gate_action("run_qc") == "nudge"
    a._inspection_nudged = True  # caller sets this after the one-time nudge
    assert a._inspection_gate_action("run_qc") == "fallback"


def test_gate_skips_non_gated_tools(monkeypatch):
    monkeypatch.setenv("SCAGENT_MODEL_INSPECTION", "1")
    a = _agent()
    for tool in ("load_data", "run_code", "inspect_data", "record_inspection", "save_data"):
        assert a._inspection_gate_action(tool) is None


def test_gate_silent_once_inspection_recorded(monkeypatch):
    monkeypatch.setenv("SCAGENT_MODEL_INSPECTION", "1")
    a = _agent()
    a.world_state.resolve_decision("inspection", {"cell_type_col": None}, source="model_inspection")
    assert a._inspection_gate_action("run_qc") is None


def test_gate_off_without_data(monkeypatch):
    monkeypatch.setenv("SCAGENT_MODEL_INSPECTION", "1")
    a = _agent()
    a.adata = None
    assert a._inspection_gate_action("run_qc") is None


def test_nudge_result_steers_to_record_inspection():
    r = json.loads(_agent()._inspection_nudge_result("run_qc"))
    assert r["status"] == "error"
    assert r["required_next_action"] == "record_inspection"
    assert "record_inspection" in r["message"]


# --- observability ------------------------------------------------------------

def test_inspection_source_reports_path():
    adata = _adata()
    ws = AgentWorldState()
    ws.sync_from_adata(adata)
    assert ws.data_summary["inspection_source"] == "heuristic"  # nothing recorded

    ws.record_inspection({"species": "human"}, adata=adata)
    assert ws.data_summary["inspection_source"] == "model"
