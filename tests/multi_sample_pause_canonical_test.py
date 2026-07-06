"""Regression tests for the multi-sample strategy "mismatch" bug.

When a multi-sample dataset needs a `multi_sample_strategy` decision, the model
must not hand-roll it with free-text `pause_and_ask` options: their action ids
are slugified labels (e.g.
``investigate_first_run_uncorrected_analysis_then_diagnose_batch_e``) that never
match the canonical enum the downstream gates check for
(``investigate_integration``, ``integrate_scvi``, ``keep_unintegrated``,
``analyze_separately``). In run_2026_07_06_015627 that mismatch permanently
blocked ``diagnose_batch_effect`` and the scVI re-prompt, so the investigation
the user asked for never ran.

These tests cover the two fixes:

1. ``record_inspection`` on multi-sample data raises the runtime's canonical
   strategy selector (previously only ``inspect_data`` did).
2. A model-authored batch-strategy ``pause_and_ask`` is intercepted and
   replaced with the canonical selector, so the user's pick maps to a real
   action value.
"""

from __future__ import annotations

import json

from scagent.agent.agent import SCAgent


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class _FakeWS:
    """Minimal world_state faithful to the predicates the fixes rely on."""

    def __init__(self, confirmed=None, data_summary=None, group_count=0):
        self._confirmed = confirmed or {}
        self.data_summary = data_summary or {}
        self._group_count = group_count

    def get_confirmed_value(self, key):
        return self._confirmed.get(key)

    def _multi_sample_group_count(self):
        return self._group_count

    def multi_sample_decision_unresolved(self):
        if self._confirmed.get("multi_sample_strategy"):
            return False
        return self._group_count >= 2


def _agent(world_state, adata=None) -> SCAgent:
    agent = object.__new__(SCAgent)
    agent.world_state = world_state
    agent.adata = adata
    agent.run_manager = None
    agent._pending_checkpoint = None
    return agent


_BATCH_PAUSE = {
    "question": "How would you like to handle the 4 datasets in this lung atlas?",
    "context": (
        "The dataset has 4 sample-like groups: Misharin_Budinger_2018, "
        "Misharin_2021, Jain_Misharin_2021_10Xv2, and Jain_Misharin_2021_10Xv1."
    ),
    "options": [
        "Investigate first — run uncorrected analysis, then diagnose batch effects before deciding",
        "Integrate with scVI — correct batch effects across all 4 datasets",
        "Keep combined uncorrected — analyze all samples together without correction",
        "Analyze samples separately — process each dataset independently",
    ],
}


# --------------------------------------------------------------------------- #
# _multi_sample_partition_from_state
# --------------------------------------------------------------------------- #


def test_partition_from_state_uses_confirmed_batch_key():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=4)
    partition = _agent(ws)._multi_sample_partition_from_state()
    assert partition == {
        "column": "dataset",
        "n_groups": 4,
        "role": "sample",
        "needs_key_confirmation": False,
    }


def test_partition_from_state_falls_back_to_data_summary():
    ws = _FakeWS(data_summary={"recommended_batch_key": "sample"}, group_count=3)
    partition = _agent(ws)._multi_sample_partition_from_state()
    assert partition is not None
    assert partition["column"] == "sample"
    assert partition["n_groups"] == 3


def test_partition_from_state_none_when_single_group():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=1)
    assert _agent(ws)._multi_sample_partition_from_state() is None


def test_partition_from_state_none_without_batch_key():
    ws = _FakeWS(group_count=4)
    assert _agent(ws)._multi_sample_partition_from_state() is None


# --------------------------------------------------------------------------- #
# record_inspection raises the canonical checkpoint
# --------------------------------------------------------------------------- #


def test_record_inspection_raises_canonical_strategy_checkpoint():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=4)
    cp = _agent(ws)._build_multi_sample_strategy_checkpoint(
        "record_inspection", {"status": "ok"}
    )
    assert cp is not None
    assert cp["kind"] == "multi_sample_strategy"
    assert cp["decision_key"] == "multi_sample_strategy"
    # canonical enum, not a slugified prose label
    assert cp["option_actions"][0] == "investigate_integration"
    assert set(cp["option_actions"]) >= {
        "investigate_integration",
        "integrate_scvi",
        "keep_unintegrated",
        "analyze_separately",
    }
    assert cp["partition"]["column"] == "dataset"


def test_inspect_data_still_raises_canonical_checkpoint():
    # The original trigger must keep working via the result-derived partition.
    ws = _FakeWS(group_count=0)
    result = {
        "status": "ok",
        "batch": {"recommended_batch_key": "dataset", "n_batches": 4},
    }
    cp = _agent(ws)._build_multi_sample_strategy_checkpoint("inspect_data", result)
    assert cp is not None
    assert cp["option_actions"][0] == "investigate_integration"


def test_record_inspection_no_checkpoint_when_strategy_confirmed():
    ws = _FakeWS(
        confirmed={"batch_key": "dataset", "multi_sample_strategy": "keep_unintegrated"},
        group_count=4,
    )
    cp = _agent(ws)._build_multi_sample_strategy_checkpoint(
        "record_inspection", {"status": "ok"}
    )
    assert cp is None


def test_record_inspection_no_checkpoint_single_sample():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=1)
    cp = _agent(ws)._build_multi_sample_strategy_checkpoint(
        "record_inspection", {"status": "ok"}
    )
    assert cp is None


def test_no_checkpoint_on_error_status():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=4)
    cp = _agent(ws)._build_multi_sample_strategy_checkpoint(
        "record_inspection", {"status": "error"}
    )
    assert cp is None


def test_unrelated_tool_never_raises_strategy_checkpoint():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=4)
    cp = _agent(ws)._build_multi_sample_strategy_checkpoint(
        "run_clustering", {"status": "ok"}
    )
    assert cp is None


# --------------------------------------------------------------------------- #
# pause_and_ask interception (the core regression)
# --------------------------------------------------------------------------- #


def test_pause_and_ask_batch_strategy_is_canonicalized():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=4)
    agent = _agent(ws)
    out = json.loads(agent._handle_pause_and_ask(dict(_BATCH_PAUSE)))

    assert out["kind"] == "multi_sample_strategy"
    assert out["decision_key"] == "multi_sample_strategy"
    # the "investigate" pick now maps to the enum diagnose_batch_effect checks for
    assert out["option_actions"][0] == "investigate_integration"
    # the slugified prose id that caused the mismatch is gone
    assert not any(
        a.startswith("investigate_first_run_uncorrected") for a in out["option_actions"]
    )
    # pending checkpoint is the canonical one
    assert agent._pending_checkpoint["decision_key"] == "multi_sample_strategy"
    assert agent._pending_checkpoint["option_actions"][0] == "investigate_integration"


def test_pause_and_ask_unrelated_question_not_hijacked():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=4)
    agent = _agent(ws)
    out = json.loads(
        agent._handle_pause_and_ask(
            {
                "question": "Which clustering resolution do you prefer?",
                "context": "Pick a resolution for the final clustering.",
                "options": ["0.5", "1.0", "2.0"],
            }
        )
    )
    assert out.get("kind") != "multi_sample_strategy"
    assert "investigate_integration" not in out.get("option_actions", [])


def test_pause_and_ask_not_canonicalized_when_strategy_resolved():
    ws = _FakeWS(
        confirmed={"batch_key": "dataset", "multi_sample_strategy": "keep_unintegrated"},
        group_count=4,
    )
    agent = _agent(ws)
    out = json.loads(agent._handle_pause_and_ask(dict(_BATCH_PAUSE)))
    # decision already made — the pause is not re-hijacked into the selector
    assert out.get("kind") != "multi_sample_strategy"


def test_pause_and_ask_not_canonicalized_single_sample():
    ws = _FakeWS(confirmed={"batch_key": "dataset"}, group_count=1)
    agent = _agent(ws)
    out = json.loads(agent._handle_pause_and_ask(dict(_BATCH_PAUSE)))
    assert out.get("kind") != "multi_sample_strategy"
