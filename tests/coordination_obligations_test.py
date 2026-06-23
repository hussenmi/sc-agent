"""Tests for the coordination-harness spine: the unmet-obligations view and the
terminal gate that enforces it.

Covers Stage 1 (completion floor) + the batch entry-obligation predicate:

1. `unmet_obligations()` reports `annotation_finalize` iff annotation was entered
   (prepare set `required`) but not finalized — and is silent otherwise (so it can
   never fire on a QC-only run).
2. `multi_sample_decision_unresolved()` is a floor: True only when multi-sample AND
   no strategy chosen; moot (False) on single-sample data.
3. `_maybe_continue_for_obligations()` re-prompts while bounded, then forces a safe
   `save_data(allow_unvalidated=true)` fallback — never a silent incomplete exit.
4. The floor never binds a model that already finalized (no nudge).
"""

from __future__ import annotations

import json

from scagent.agent.agent import SCAgent, OBLIGATION_NUDGES
from scagent.agent.world_state import AgentWorldState


# --------------------------------------------------------------------------- #
# 1. unmet_obligations: annotation completion
# --------------------------------------------------------------------------- #

def test_unmet_reports_annotation_when_staged_not_finalized():
    ws = AgentWorldState()
    ws.annotation_validation = {"required": True, "status": "evidence_staged_ready_to_finalize"}
    keys = [o["key"] for o in ws.unmet_obligations()]
    assert "annotation_finalize" in keys
    o = next(o for o in ws.unmet_obligations() if o["key"] == "annotation_finalize")
    assert o["kind"] == "completion" and o["blocks_terminal"] is True


def test_unmet_silent_when_finalized():
    ws = AgentWorldState()
    ws.annotation_validation = {"required": True, "status": "validated_and_finalized", "finalized": True}
    assert [o for o in ws.unmet_obligations() if o["key"] == "annotation_finalize"] == []


def test_unmet_silent_when_annotation_never_entered():
    # QC-only run: prepare_annotation never called -> no `required` -> no obligation.
    ws = AgentWorldState()
    assert ws.unmet_obligations() == []


# --------------------------------------------------------------------------- #
# 2. batch entry obligation is a floor (moot on single-sample)
# --------------------------------------------------------------------------- #

def test_batch_decision_unresolved_when_multisample_and_no_strategy():
    ws = AgentWorldState()
    ws.data_summary = {"n_batches": 8}
    assert ws.multi_sample_decision_unresolved() is True
    assert any(o["key"] == "batch_decision" for o in ws.unmet_obligations())


def test_batch_decision_moot_on_single_sample():
    ws = AgentWorldState()
    ws.data_summary = {"n_batches": 1}
    assert ws.multi_sample_decision_unresolved() is False
    assert not any(o["key"] == "batch_decision" for o in ws.unmet_obligations())


def test_batch_decision_satisfied_once_strategy_chosen():
    ws = AgentWorldState()
    ws.data_summary = {"n_batches": 8}
    ws.user_preferences["multi_sample_strategy"] = "investigate_integration"
    assert ws.multi_sample_decision_unresolved() is False


# --------------------------------------------------------------------------- #
# 3 + 4. terminal gate: nudge -> bounded -> forced fallback; no-op when satisfied
# --------------------------------------------------------------------------- #

def _agent_with(ws) -> SCAgent:
    agent = object.__new__(SCAgent)
    agent.world_state = ws
    agent._conversation_history = []
    return agent


def test_terminal_gate_nudges_then_forces_fallback():
    ws = AgentWorldState()
    ws.annotation_validation = {"required": True, "status": "evidence_staged_ready_to_finalize"}
    agent = _agent_with(ws)

    saved = {}

    def _fake_execute(tool_name, tool_input):
        saved["tool"] = tool_name
        saved["input"] = tool_input
        return json.dumps({"status": "ok", "unvalidated": True})

    agent._execute_tool = _fake_execute  # type: ignore[assignment]

    messages: list = []
    attempts = 0
    # First OBLIGATION_NUDGES calls should re-prompt (continue), no save yet.
    for i in range(OBLIGATION_NUDGES):
        cont, attempts = agent._maybe_continue_for_obligations(messages, attempts)
        assert cont is True
        assert "tool" not in saved  # not yet forced
    # Next call: bound exhausted -> forced unvalidated save, do not continue.
    cont, attempts = agent._maybe_continue_for_obligations(messages, attempts)
    assert cont is False
    assert saved.get("tool") == "save_data"
    assert saved["input"].get("allow_unvalidated") is True


def test_terminal_gate_noop_when_finalized():
    ws = AgentWorldState()
    ws.annotation_validation = {"required": True, "status": "validated_and_finalized", "finalized": True}
    agent = _agent_with(ws)
    agent._execute_tool = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not save"))  # type: ignore
    cont, attempts = agent._maybe_continue_for_obligations([], 0)
    assert cont is False and attempts == 0
