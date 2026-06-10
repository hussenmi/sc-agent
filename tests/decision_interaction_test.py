"""Focused regression tests for structured terminal decisions."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scagent.agent.agent import SCAgent
from scagent.cli import _analyze_with_decisions, _maybe_save_on_exit
from scagent.terminal import (
    DecisionChoice,
    DecisionSelection,
    prompt_for_decision,
    resolve_decision_response,
)


CHOICES = [
    DecisionChoice("Integrate with scVI", "integrate_scvi"),
    DecisionChoice("Leave samples separate", "leave_separate"),
    DecisionChoice("Investigate first", "investigate"),
]


def _bare_agent(checkpoint=None):
    agent = object.__new__(SCAgent)
    agent._pending_checkpoint = checkpoint
    agent.run_manager = None
    agent._active_cleanup_authorization = None
    agent.world_state = SimpleNamespace(resolve_decision=lambda *args, **kwargs: None)
    return agent


def test_resolves_numeric_and_numbered_text():
    assert resolve_decision_response("2", CHOICES).action == "leave_separate"
    assert resolve_decision_response("option 3", CHOICES).action == "investigate"


def test_resolves_ordinals_labels_and_actions():
    assert resolve_decision_response("first", CHOICES).action == "integrate_scvi"
    assert resolve_decision_response("Investigate first", CHOICES).action == "investigate"
    assert resolve_decision_response("leave_separate", CHOICES).index == 1


def test_resolves_default_and_custom_text():
    assert resolve_decision_response("", CHOICES, default_index=2).action == "investigate"
    custom = resolve_decision_response("Compare donors only", CHOICES)
    assert custom.custom is True
    assert custom.value == "Compare donors only"


def test_numbered_text_fallback_maps_to_action():
    replies = iter(["2"])
    selected = prompt_for_decision(
        "Choose",
        CHOICES,
        force_text_fallback=True,
        input_reader=lambda _: next(replies),
    )
    assert selected.action == "leave_separate"
    assert selected.input_mode == "text"


def test_open_ended_question_returns_custom():
    selected = prompt_for_decision(
        "Which column?",
        [],
        force_text_fallback=True,
        input_reader=lambda _: "donor_id",
    )
    assert selected.custom is True
    assert selected.value == "donor_id"


def test_selector_result_uses_stable_action(monkeypatch):
    monkeypatch.setattr("scagent.terminal.sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("scagent.terminal.sys.stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("scagent.terminal._run_selector_app", lambda *args, **kwargs: 1)
    selected = prompt_for_decision("Choose", CHOICES, allow_custom=False)
    assert selected.action == "leave_separate"
    assert selected.index == 1
    assert selected.input_mode == "selector"


def test_existing_custom_choice_prompts_for_text(monkeypatch):
    choices = [DecisionChoice("Proceed", "proceed"), DecisionChoice("Something else", "custom")]
    monkeypatch.setattr("scagent.terminal.sys.stdin.isatty", lambda: False)
    replies = iter(["2", "Use donor as the grouping"])
    selected = prompt_for_decision(
        "Choose",
        choices,
        force_text_fallback=True,
        input_reader=lambda _: next(replies),
    )
    assert selected.action == "custom"
    assert selected.value == "Use donor as the grouping"


def test_pending_checkpoint_generates_stable_actions():
    agent = _bare_agent()
    agent._set_pending_checkpoint(
        {"kind": "test", "question": "Choose", "options": ["Run scVI", "Run scVI"]}
    )
    assert agent._pending_checkpoint["option_actions"] == ["run_scvi", "run_scvi_2"]


def test_structured_request_contains_world_state_action_input():
    checkpoint = {
        "kind": "integration_choice",
        "decision_key": "integration_choice",
        "question": "How should samples be handled?",
        "options": [choice.label for choice in CHOICES],
        "option_actions": [choice.action for choice in CHOICES],
        "action_inputs": {"integrate_scvi": {"method": "scvi"}},
    }
    agent = _bare_agent(checkpoint)
    selection = DecisionSelection(
        "integrate_scvi", "Integrate with scVI", 0, "Integrate with scVI",
        "Integrate with scVI", "selector",
    )
    request = agent.structured_decision_request(selection)
    payload = json.loads(request.split("\n", 1)[1].split("\n\n", 1)[0])
    assert payload["selected_action"] == "integrate_scvi"
    assert payload["action_input"] == {"method": "scvi"}
    assert agent.has_pending_decision is False


def test_text_only_reply_maps_against_pending_checkpoint():
    checkpoint = {
        "kind": "integration_choice",
        "question": "Choose",
        "options": [choice.label for choice in CHOICES],
        "option_actions": [choice.action for choice in CHOICES],
    }
    agent = _bare_agent(checkpoint)
    assert agent.resolve_pending_decision_text("3").action == "investigate"


def test_cleanup_action_authorizes_exact_proposal():
    checkpoint = {
        "kind": "cluster_qc_cleanup",
        "decision_key": "cluster_qc_cleanup",
        "question": "Remove?",
        "options": ["Remove proposed clusters", "Keep clusters"],
        "option_actions": ["remove_proposed_clusters", "keep_proposed_clusters"],
        "proposal": {"proposed_removal": ["4", "7"]},
    }
    agent = _bare_agent(checkpoint)
    selected = DecisionSelection(
        "remove_proposed_clusters", "Remove proposed clusters", 0,
        "Remove proposed clusters", "remove_proposed_clusters", "selector",
    )
    payload = agent.resolve_pending_decision(selected)
    assert payload["selected_action"] == "remove_proposed_clusters"
    assert agent._active_cleanup_authorization["proposal"]["proposed_removal"] == ["4", "7"]


def test_cli_resumes_until_no_pending_decision():
    selection = DecisionSelection("investigate", "Investigate first", 2, "Investigate first", "3", "text")

    class FakeAgent:
        def __init__(self):
            self.calls = []
            self.has_pending_decision = True

        def analyze(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 2:
                self.has_pending_decision = False
            return f"result-{len(self.calls)}"

        def prompt_pending_decision(self):
            return selection

        def structured_decision_request(self, value):
            assert value is selection
            return "structured"

    agent = FakeAgent()
    result = _analyze_with_decisions(agent, request="start", data_path="x.h5", max_iterations=9)
    assert result == "result-2"
    assert agent.calls[1]["request"] == "structured"
    assert agent.calls[1]["continue_conversation"] is True


def test_package_install_defaults_to_denial(monkeypatch):
    agent = _bare_agent()
    monkeypatch.setattr(
        "scagent.terminal.prompt_for_decision",
        lambda *args, **kwargs: DecisionSelection(
            "deny", "Do not install", 0, "Do not install", "", "selector"
        ),
    )
    result = json.loads(agent._handle_install_package({"package": "example", "reason": "test"}))
    assert result["status"] == "denied"


def test_error_recovery_accepts_custom_instruction(monkeypatch):
    agent = _bare_agent()
    agent.verbose = True
    monkeypatch.setattr(
        "scagent.terminal.prompt_for_decision",
        lambda *args, **kwargs: DecisionSelection(
            "custom", "Enter a new instruction", 1, "Use another model", "Use another model",
            "selector", True,
        ),
    )
    assert agent._ask_continue("failed") == "Use another model"


def test_exit_save_can_be_skipped(monkeypatch, tmp_path):
    class Adata:
        def write_h5ad(self, path):
            raise AssertionError("save should have been skipped")

    agent = SimpleNamespace(adata=Adata(), run_manager=None, output_dir=tmp_path)
    console = SimpleNamespace(print=lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "scagent.terminal.prompt_for_decision",
        lambda *args, **kwargs: DecisionSelection(
            "skip", "Exit without saving", 2, "Exit without saving", "", "selector"
        ),
    )
    _maybe_save_on_exit(agent, console)
