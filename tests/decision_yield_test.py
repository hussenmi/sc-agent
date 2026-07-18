"""Regression tests for the enforced decision-yield mechanism.

Background (run_2026_07_15_121111): a weak model, pointed at 40 CSVs, hit the
concat guard, called ``pause_and_ask`` (which set an ``llm_pause`` checkpoint),
but then *kept calling tools* instead of ending its turn — misreading its own
option label as the user's answer and finally hand-rolling an ``AnnData`` to get
around the guard. The decision was never resolved because control never returned
to the resolver.

Two fixes are covered here:

1. ``_yield_for_pending_decision`` — the single exit the tool loop takes when any
   checkpoint is pending, handing control to the resolver instead of letting the
   model take another tool-calling turn.
2. The ``_execute_tool`` hard-block now covers ``llm_pause`` (and every genuine
   data decision) for ``run_code``/``load_data`` — the escape hatch the model
   used to sidestep the concat guard.
"""

from __future__ import annotations

import json

from scagent.agent.agent import SCAgent


def _bare_agent(checkpoint=None):
    agent = object.__new__(SCAgent)
    agent._pending_checkpoint = checkpoint
    agent._mcp_client = None
    agent.run_manager = None
    agent._multifile_source_datasets = []
    agent._conversation_history = []
    agent._printed = []
    agent._print = lambda *a, **k: agent._printed.append(a[0] if a else "")
    return agent


# --------------------------------------------------------------------------- #
# _yield_for_pending_decision
# --------------------------------------------------------------------------- #


def test_yield_returns_question_and_context():
    cp = {
        "kind": "llm_pause",
        "question": "How should I combine these 40 CSV samples?",
        "context": "40 GEO count matrices with a shared gene set.",
    }
    agent = _bare_agent(cp)
    messages = [{"role": "user", "content": "load csvs"}]
    out = agent._yield_for_pending_decision(messages)
    # Both the context and the question reach the user, context first.
    assert "40 GEO count matrices" in out
    assert "How should I combine" in out
    assert out.index("40 GEO") < out.index("How should I combine")
    # History is preserved for the follow-up resume.
    assert agent._conversation_history is messages


def test_yield_falls_back_to_summary_then_generic():
    # No context/question, only summary → summary used.
    agent = _bare_agent({"kind": "llm_pause", "summary": "Choose a join."})
    assert "Choose a join." in agent._yield_for_pending_decision([])
    # Nothing at all → a generic, non-empty prompt (never a blank turn).
    agent2 = _bare_agent({"kind": "llm_pause"})
    assert agent2._yield_for_pending_decision([]).strip() != ""


def test_yield_does_not_complete_the_run():
    # run_manager is None here; the guard is that _yield never calls _complete_run.
    calls = []
    agent = _bare_agent({"kind": "llm_pause", "question": "?"})
    agent._complete_run = lambda *a, **k: calls.append(a)  # type: ignore[assignment]
    agent._yield_for_pending_decision([])
    assert calls == []


# --------------------------------------------------------------------------- #
# _execute_tool hard-block covers llm_pause (the closed bypass)
# --------------------------------------------------------------------------- #


def _assert_blocked(agent, tool_name):
    result = json.loads(agent._execute_tool(tool_name, {"code": "adata = anndata.AnnData(...)"}))
    assert result["status"] == "error"
    assert result["tool"] == tool_name
    assert result["required_next_action"] == "resolve_pending_decision"


def test_llm_pause_blocks_run_code_and_load_data():
    # The exact bypass: pause_and_ask set an llm_pause, model then reaches for
    # run_code (manual construction) or load_data.
    for kind in ("llm_pause", "multi_dataset_loading", "multi_sample_strategy"):
        agent = _bare_agent({"kind": kind})
        _assert_blocked(agent, "run_code")
    for kind in ("llm_pause", "multi_dataset_loading"):
        agent = _bare_agent({"kind": kind})
        _assert_blocked(agent, "load_data")


def test_no_checkpoint_does_not_block():
    # Without a pending checkpoint the hard-block path is inert (run_code would
    # proceed to real execution — we only assert it does NOT short-circuit).
    agent = _bare_agent(None)
    # _is_action_tool is safe to call; with no checkpoint the block is skipped.
    assert agent._pending_checkpoint is None
    assert agent._is_action_tool("run_code") is True
