"""Focused regression tests for structured terminal decisions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from scagent.agent.agent import SCAgent
from scagent.agent.tools import get_tools, process_tool_call, write_h5ad_safe
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


class FakeWorldState:
    def __init__(self):
        self.user_preferences = {}
        self.context_hints = []
        self.data_summary = {}
        self.metadata_candidates = []

    def resolve_decision(self, key, value, **kwargs):
        self.user_preferences[key] = value

    def get_confirmed_value(self, key):
        return self.user_preferences.get(key)

    def add_context_hint(self, hint, *, source="model"):
        self.context_hints.append(hint)


def _bare_agent(checkpoint=None):
    agent = object.__new__(SCAgent)
    agent._pending_checkpoint = checkpoint
    agent._mcp_client = None
    agent.run_manager = None
    agent._active_cleanup_authorization = None
    agent.world_state = FakeWorldState()
    return agent


def _null_encoded_paths(path):
    paths = []
    with h5py.File(path, "r") as handle:
        def visit(name, obj):
            encoding = obj.attrs.get("encoding-type")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            if encoding == "null":
                paths.append(name)

        handle.visititems(visit)
    return paths


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


def test_text_entry_choice_keeps_its_action():
    choices = [
        DecisionChoice(
            "Describe the experiment first",
            "describe_experiment",
            requires_text=True,
            text_prompt="Describe: ",
            placeholder="Samples, donors, and conditions",
        )
    ]
    replies = iter(["1", "Three donors per condition; sequencing run is balanced."])
    selected = prompt_for_decision(
        "Choose",
        choices,
        allow_custom=False,
        force_text_fallback=True,
        input_reader=lambda _: next(replies),
    )
    assert selected.action == "describe_experiment"
    assert selected.custom is False
    assert selected.value.startswith("Three donors")


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
            self.has_pending_decision = False
            return "structured"

    agent = FakeAgent()
    result = _analyze_with_decisions(agent, request="start", data_path="x.h5", max_iterations=9)
    assert result == "result-2"
    assert agent.calls[1]["request"] == "structured"
    assert agent.calls[1]["continue_conversation"] is True


def test_multi_sample_checkpoint_has_expected_initial_choices():
    agent = _bare_agent()
    checkpoint = agent._build_multi_sample_strategy_checkpoint(
        "inspect_data",
        {
            "status": "ok",
            "batch": {
                "recommended_batch_key": "sample_id",
                "recommended_role": "sample",
                "n_batches": 4,
                "status": "auto_selected",
                "batch_correction_applied": False,
            },
        },
    )
    assert checkpoint["option_actions"] == [
        "investigate_integration",
        "integrate_scvi",
        "keep_unintegrated",
        "analyze_separately",
        "describe_experiment",
    ]
    assert checkpoint["default"] == checkpoint["options"][0]
    assert checkpoint["custom_label"] == "Type something else..."
    assert checkpoint["action_inputs"]["integrate_scvi"]["method"] == "scvi"


def test_multi_dataset_loading_checkpoint_precedes_concatenation():
    agent = _bare_agent()
    checkpoint = agent._build_multi_dataset_loading_checkpoint(
        "inspect_data_inputs",
        {
            "status": "ok",
            "source_datasets": [
                {"name": "Rep1.h5", "path": "/data/Rep1.h5", "format": "10x_h5"},
                {"name": "Rep2.h5", "path": "/data/Rep2.h5", "format": "10x_h5"},
            ],
            "likely_combined_outputs": [
                {"name": "combined.h5ad", "path": "/data/combined.h5ad"}
            ],
        },
    )

    assert checkpoint["kind"] == "multi_dataset_loading"
    assert checkpoint["option_actions"] == [
        "concatenate_outer",
        "concatenate_inner",
        "analyze_separately",
    ]
    assert checkpoint["default"] == checkpoint["options"][0]
    assert "recommended" in checkpoint["default"]
    assert "keep all genes" in checkpoint["default"]
    assert checkpoint["custom_label"] == "Type something else..."
    assert checkpoint["action_inputs"]["concatenate_outer"]["join"] == "outer"
    assert checkpoint["action_inputs"]["concatenate_inner"]["join"] == "inner"
    assert "combined.h5ad" in checkpoint["context"]


def test_single_dataset_does_not_open_loading_checkpoint():
    agent = _bare_agent()
    checkpoint = agent._build_multi_dataset_loading_checkpoint(
        "inspect_data_inputs",
        {
            "status": "ok",
            "source_datasets": [
                {"name": "Rep1.h5", "path": "/data/Rep1.h5", "format": "10x_h5"}
            ],
        },
    )
    assert checkpoint is None


def test_loading_choice_is_persisted_independently_from_integration_choice():
    checkpoint = {
        "kind": "multi_dataset_loading",
        "decision_key": "multi_dataset_loading_strategy",
        "question": "How should I handle these datasets?",
        "options": ["Outer", "Inner", "Separate"],
        "option_actions": [
            "concatenate_outer",
            "concatenate_inner",
            "analyze_separately",
        ],
        "action_inputs": {
            "concatenate_inner": {"join": "inner"},
        },
    }
    agent = _bare_agent(checkpoint)
    selection = DecisionSelection(
        "concatenate_inner", "Inner", 1, "Inner", "2", "text"
    )
    payload = agent.resolve_pending_decision(selection)

    assert payload["action_input"] == {"join": "inner"}
    assert (
        agent.world_state.get_confirmed_value("multi_dataset_loading_strategy")
        == "concatenate_inner"
    )
    assert agent.world_state.get_confirmed_value("multi_sample_strategy") is None


def test_concatenation_guard_requires_and_enforces_loading_choice():
    agent = _bare_agent()
    outer_code = (
        "from scagent.core import concat_datasets\n"
        "adata = concat_datasets(items, batch_names=names, join='outer')"
    )
    blocked = json.loads(agent._multi_dataset_loading_guard(
        "run_code", {"code": outer_code}
    ))
    assert blocked["requires_user_decision"] is True

    agent.world_state.user_preferences["multi_dataset_loading_strategy"] = "analyze_separately"
    separate_block = json.loads(agent._multi_dataset_loading_guard(
        "run_code", {"code": outer_code}
    ))
    assert "not allowed" in separate_block["message"]

    agent.world_state.user_preferences["multi_dataset_loading_strategy"] = "concatenate_inner"
    mismatch = json.loads(agent._multi_dataset_loading_guard(
        "run_code", {"code": outer_code}
    ))
    assert mismatch["required_join"] == "inner"

    inner_code = outer_code.replace("join='outer'", "join='inner'")
    assert agent._multi_dataset_loading_guard("run_code", {"code": inner_code}) is None


def test_pending_loading_checkpoint_blocks_run_code_before_selector_resolution():
    agent = _bare_agent({
        "kind": "multi_dataset_loading",
        "question": "How should I handle these datasets?",
        "options": ["Outer", "Inner", "Separate"],
        "option_actions": [
            "concatenate_outer",
            "concatenate_inner",
            "analyze_separately",
        ],
    })
    result = json.loads(agent._execute_tool(
        "run_code",
        {"code": "adata = ad.concat(items)", "description": "combine datasets"},
    ))

    assert result["status"] == "error"
    assert result["pending_checkpoint"]["kind"] == "multi_dataset_loading"


def test_successful_concatenation_opens_sample_strategy_checkpoint():
    agent = _bare_agent()
    agent.world_state.data_summary = {
        "recommended_batch_key": "sample",
        "n_batches": 2,
    }
    agent.world_state.metadata_candidates = [
        {"column": "sample", "role": "sample", "n_unique": 2}
    ]
    checkpoint = agent._build_post_concatenation_strategy_checkpoint(
        "run_code",
        {
            "code": (
                "adata = concat_datasets("
                "items, batch_names=names, join='outer')"
            )
        },
        {"status": "ok"},
    )

    assert checkpoint["kind"] == "multi_sample_strategy"
    assert checkpoint["partition"]["column"] == "sample"
    assert checkpoint["partition"]["n_groups"] == 2


def test_non_concatenation_code_does_not_open_sample_strategy_checkpoint():
    agent = _bare_agent()
    agent.world_state.data_summary = {
        "recommended_batch_key": "sample",
        "n_batches": 2,
    }
    checkpoint = agent._build_post_concatenation_strategy_checkpoint(
        "run_code",
        {"code": "adata.obs['total'] = 1"},
        {"status": "ok"},
    )
    assert checkpoint is None


def test_pending_sample_strategy_blocks_more_run_code():
    agent = _bare_agent({
        "kind": "multi_sample_strategy",
        "question": "How should I handle these samples?",
        "options": ["Investigate", "Integrate", "Keep unintegrated"],
        "option_actions": [
            "investigate_integration",
            "integrate_scvi",
            "keep_unintegrated",
        ],
    })
    result = json.loads(agent._execute_tool(
        "run_code",
        {"code": "sc.pp.normalize_total(adata)", "description": "normalize"},
    ))

    assert result["status"] == "error"
    assert result["pending_checkpoint"]["kind"] == "multi_sample_strategy"


def test_pause_and_ask_preserves_existing_runtime_checkpoint():
    checkpoint = {
        "kind": "multi_sample_strategy",
        "decision_key": "multi_sample_strategy",
        "question": "How should I handle these samples?",
        "summary": "Two samples were concatenated.",
        "options": ["Investigate", "Integrate"],
        "option_actions": ["investigate_integration", "integrate_scvi"],
        "action_inputs": {
            "integrate_scvi": {"batch_key": "sample", "method": "scvi"}
        },
    }
    agent = _bare_agent(checkpoint)
    result = json.loads(agent._handle_pause_and_ask({
        "question": "Should I integrate?",
        "options": ["Yes", "No"],
    }))

    assert agent._pending_checkpoint is checkpoint
    assert result["kind"] == "multi_sample_strategy"
    assert result["options"] == ["Investigate", "Integrate"]
    assert result["action_inputs"]["integrate_scvi"]["method"] == "scvi"


def test_explicit_loading_join_language_is_remembered():
    agent = _bare_agent()
    agent.adata = None
    agent._remember_user_preferences(
        "Concatenate these datasets with an inner join."
    )
    assert (
        agent.world_state.get_confirmed_value("multi_dataset_loading_strategy")
        == "concatenate_inner"
    )


def test_multi_sample_checkpoint_uses_ambiguous_candidate_counts():
    agent = _bare_agent()
    checkpoint = agent._build_multi_sample_strategy_checkpoint(
        "inspect_data",
        {
            "status": "ok",
            "batch": {
                "status": "needs_confirmation",
                "needs_confirmation": True,
                "recommended_batch_key": "donor",
                "recommended_role": "donor",
                "n_batches": 0,
                "candidates": [
                    {"column": "donor", "role": "donor", "n_unique": 6}
                ],
            },
        },
    )
    assert checkpoint["partition"]["n_groups"] == 6
    assert checkpoint["action_inputs"]["integrate_scvi"]["batch_key_needs_confirmation"] is True


def test_selected_strategy_suppresses_repeat_checkpoint():
    agent = _bare_agent()
    agent.world_state.user_preferences["multi_sample_strategy"] = "keep_unintegrated"
    checkpoint = agent._build_multi_sample_strategy_checkpoint(
        "inspect_data",
        {
            "status": "ok",
            "batch": {
                "recommended_batch_key": "sample",
                "n_batches": 3,
                "batch_correction_applied": False,
            },
        },
    )
    assert checkpoint is None


def test_explicit_strategy_language_is_remembered():
    agent = _bare_agent()
    agent.adata = None
    agent._remember_user_preferences("Integrate the samples.")
    assert agent.world_state.get_confirmed_value("multi_sample_strategy") == "integrate_scvi"


def test_explicit_nondefault_method_is_remembered():
    agent = _bare_agent()
    agent.adata = None
    agent._remember_user_preferences("Integrate the samples with Harmony.")
    strategy = agent.world_state.get_confirmed_value("multi_sample_strategy")
    assert strategy["action"] == "custom"
    assert strategy["method"] == "harmony"


def test_describe_experiment_is_stored_then_reprompts_strategy():
    agent = _bare_agent(
        {
            "kind": "multi_sample_strategy",
            "decision_key": "multi_sample_strategy",
            "question": "How should I handle these samples?",
            "options": ["Investigate", "Describe the experiment first"],
            "option_actions": ["investigate_integration", "describe_experiment"],
            "partition": {
                "column": "sample_id",
                "n_groups": 4,
                "role": "sample",
                "status": "auto_selected",
                "needs_key_confirmation": False,
            },
        }
    )
    selection = DecisionSelection(
        "describe_experiment",
        "Describe the experiment first",
        1,
        "Two conditions, three donors each, balanced across sequencing runs.",
        "Two conditions, three donors each, balanced across sequencing runs.",
        "selector",
    )
    request = agent.structured_decision_request(selection)
    assert "experiment_design" in request
    assert agent.world_state.get_confirmed_value("multi_sample_strategy") is None
    assert "balanced across sequencing runs" in agent.world_state.get_confirmed_value("experiment_design")
    assert agent.has_pending_decision is True
    assert agent._pending_checkpoint["kind"] == "multi_sample_strategy"
    assert "Experiment context you provided" in agent._pending_checkpoint["context"]


def _multi_option_checkpoint():
    return {
        "kind": "multi_sample_strategy",
        "decision_key": "multi_sample_strategy",
        "question": "How should I handle the 8 donors in this dataset?",
        "options": [
            "Investigate whether integration is needed (recommended)",
            "Integrate the samples with scVI",
            "Keep samples combined without integration",
            "Analyze samples separately",
            "Describe the experiment first",
        ],
        "option_actions": [
            "investigate_integration",
            "integrate_scvi",
            "keep_unintegrated",
            "analyze_separately",
            "describe_experiment",
        ],
    }


def _custom_selection(text):
    return DecisionSelection("custom", "Type something else...", 5, text, text, "selector", True)


def test_custom_reply_leaves_multi_option_decision_unresolved():
    # The batch-question bug: a free-text reply ("go ahead but skip scrublet") must
    # NOT silently close a genuine multiple-choice decision.
    agent = _bare_agent(_multi_option_checkpoint())
    payload = agent.resolve_pending_decision(
        _custom_selection("can you go ahead but not run scrublet")
    )
    assert payload["custom_needs_resolution"] is True
    # Decision stays open; the free text is preserved as a side-instruction hint.
    assert agent.world_state.get_confirmed_value("multi_sample_strategy") is None
    assert any("not run scrublet" in h for h in agent.world_state.context_hints)


def test_custom_reply_mandates_reask_in_model_message():
    agent = _bare_agent(_multi_option_checkpoint())
    request = agent.structured_decision_request(
        _custom_selection("can you go ahead but not run scrublet")
    )
    assert "MUST re-ask" in request
    assert "Investigate whether integration is needed (recommended)" in request  # options listed
    assert "custom_needs_resolution" in request


def test_custom_reply_to_single_option_checkpoint_still_resolves():
    # An escape-hatch custom on a checkpoint without a real branch fork is a valid
    # answer — it should resolve normally, not trigger a re-ask.
    agent = _bare_agent(
        {
            "kind": "free_form",
            "decision_key": "free_form",
            "question": "Anything to add?",
            "options": ["Proceed"],
            "option_actions": ["proceed"],
        }
    )
    payload = agent.resolve_pending_decision(_custom_selection("use inner join"))
    assert payload["custom_needs_resolution"] is False
    resolved = agent.world_state.get_confirmed_value("free_form")
    assert resolved == {"action": "custom", "details": "use inner join"}


def test_branch_selection_is_unaffected_by_the_guard():
    agent = _bare_agent(_multi_option_checkpoint())
    selection = DecisionSelection(
        "keep_unintegrated", "Keep samples combined without integration", 2,
        "Keep samples combined without integration", "keep_unintegrated", "selector",
    )
    payload = agent.resolve_pending_decision(selection)
    assert payload["custom_needs_resolution"] is False
    assert agent.world_state.get_confirmed_value("multi_sample_strategy") == "keep_unintegrated"


def test_cli_reprompts_without_model_turn_for_context_only_choice():
    describe = DecisionSelection(
        "describe_experiment", "Describe the experiment first", 4,
        "Balanced donors", "Balanced donors", "selector",
    )
    final = DecisionSelection(
        "investigate_integration", "Investigate first", 0,
        "Investigate first", "Investigate first", "selector",
    )

    class FakeAgent:
        def __init__(self):
            self.calls = []
            self.selections = iter([describe, final])
            self.has_pending_decision = True

        def analyze(self, **kwargs):
            self.calls.append(kwargs)
            return f"result-{len(self.calls)}"

        def prompt_pending_decision(self):
            return next(self.selections)

        def structured_decision_request(self, selection):
            self.has_pending_decision = selection.action == "describe_experiment"
            return f"structured:{selection.action}"

    agent = FakeAgent()
    result = _analyze_with_decisions(agent, request="start", data_path="x.h5")
    assert result == "result-2"
    assert len(agent.calls) == 2
    assert agent.calls[1]["request"] == "structured:investigate_integration"


def test_batch_correction_tool_schema_defaults_to_scvi_language():
    schema = next(
        tool for tool in get_tools()
        if tool["name"] == "run_batch_correction"
    )
    method_description = schema["input_schema"]["properties"]["method"]["description"]
    assert "default: scvi" in method_description
    assert "default: harmony" not in method_description


def test_batch_correction_refuses_to_run_without_user_strategy():
    adata = ad.AnnData(
        np.ones((4, 2)),
        obs=pd.DataFrame(
            {"sample": ["s1", "s1", "s2", "s2"]},
            index=[f"cell_{index}" for index in range(4)],
        ),
    )
    result_json, _ = process_tool_call(
        "run_batch_correction",
        {"batch_key": "sample"},
        adata,
        world_state=FakeWorldState(),
    )
    result = json.loads(result_json)
    assert result["status"] == "error"
    assert result["requires_user_strategy"] is True
    assert "opt-in" in result["message"]


def test_batch_correction_honors_nonintegration_strategy():
    world_state = FakeWorldState()
    world_state.user_preferences["multi_sample_strategy"] = "keep_unintegrated"
    adata = ad.AnnData(
        np.ones((4, 2)),
        obs=pd.DataFrame(
            {"sample": ["s1", "s1", "s2", "s2"]},
            index=[f"cell_{index}" for index in range(4)],
        ),
    )
    result_json, _ = process_tool_call(
        "run_batch_correction",
        {"batch_key": "sample", "method": "scvi"},
        adata,
        world_state=world_state,
    )
    result = json.loads(result_json)
    assert result["status"] == "error"
    assert result["selected_strategy"] == "keep_unintegrated"
    assert "not integration" in result["message"]


def test_batch_diagnostic_tool_is_advertised():
    schema = next(
        tool for tool in get_tools()
        if tool["name"] == "diagnose_batch_effect"
    )
    description = schema["description"]
    assert "shared cross-cell-type signatures" in description
    assert "user confirmation before scVI" in description


def test_batch_diagnostic_refuses_without_investigation_strategy():
    adata = ad.AnnData(
        np.ones((6, 3)),
        obs=pd.DataFrame(
            {
                "sample": ["s1", "s1", "s1", "s2", "s2", "s2"],
                "leiden": ["0", "0", "1", "0", "1", "1"],
            },
            index=[f"cell_{index}" for index in range(6)],
        ),
        var=pd.DataFrame(index=["CD3D", "CD3E", "IFIT1"]),
    )
    result_json, _ = process_tool_call(
        "diagnose_batch_effect",
        {"batch_key": "sample", "cluster_key": "leiden"},
        adata,
        world_state=FakeWorldState(),
    )
    result = json.loads(result_json)
    assert result["status"] == "error"
    assert result["requires_user_strategy"] is True


def test_post_investigation_checkpoint_waits_for_diagnostic():
    agent = _bare_agent()
    agent.world_state.user_preferences["multi_sample_strategy"] = "investigate_integration"
    agent.world_state.data_summary = {"batch_key": "sample", "n_batches": 2}
    agent.adata = ad.AnnData(
        np.ones((4, 2)),
        obs=pd.DataFrame(
            {"sample": ["s1", "s1", "s2", "s2"]},
            index=[f"cell_{index}" for index in range(4)],
        ),
    )

    assert agent._post_investigation_strategy_checkpoint() is None

    agent.adata.uns["batch_effect_diagnostic"] = {
        "status": "ok",
        "batch_key": "sample",
        "n_batches": 2,
        "gene_evidence": "recurring_sample_associated",
        "design_interpretation": "unknown",
        "recommendation": "cannot_determine_technical_vs_biological",
        "recommendation_reason": "a sample-associated program recurs but design is unknown",
        "selected_pairs": [],
        "recurrent_programs": [],
    }
    checkpoint = agent._post_investigation_strategy_checkpoint()
    assert checkpoint["kind"] == "multi_sample_strategy"
    assert checkpoint["option_actions"] == [
        "integrate_scvi",
        "keep_unintegrated",
        "analyze_separately",
        "describe_experiment",
    ]
    assert "Gene evidence: recurring_sample_associated" in checkpoint["context"]


def test_post_diagnostic_pause_is_normalized_to_runtime_checkpoint():
    agent = _bare_agent()
    agent.world_state.user_preferences["multi_sample_strategy"] = "investigate_integration"
    agent.world_state.data_summary = {"batch_key": "sample", "n_batches": 2}
    agent.adata = ad.AnnData(
        np.ones((4, 2)),
        obs=pd.DataFrame(
            {"sample": ["s1", "s1", "s2", "s2"]},
            index=[f"cell_{index}" for index in range(4)],
        ),
    )
    agent.adata.uns["batch_effect_diagnostic"] = {
        "status": "ok",
        "batch_key": "sample",
        "n_batches": 2,
        "gene_evidence": "recurring_sample_associated",
        "design_interpretation": "unknown",
        "recommendation": "cannot_determine_technical_vs_biological",
        "recommendation_reason": "a sample-associated program recurs but design is unknown",
        "selected_pairs": [],
        "recurrent_programs": [],
    }

    result = json.loads(agent._handle_pause_and_ask({
        "question": "Integrate with scVI?",
        "context": "The diagnostic found sample effects.",
        "options": ["Integrate with scVI", "Keep uncorrected"],
        "option_actions": ["integrate_scvi", "keep_unintegrated"],
        "decision_key": "integration_decision",
    }))

    assert result["kind"] == "multi_sample_strategy"
    assert result["decision_key"] == "multi_sample_strategy"
    assert agent._pending_checkpoint["kind"] == "multi_sample_strategy"
    assert agent._pending_checkpoint["option_actions"] == [
        "integrate_scvi",
        "keep_unintegrated",
        "analyze_separately",
        "describe_experiment",
    ]


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


def test_write_h5ad_safe_sanitizes_uns_null_values(tmp_path):
    adata = ad.AnnData(np.ones((3, 2)))
    adata.obs["mixed"] = ["a", None, {"bad": "object"}]
    adata.uns["nested"] = {"skip_reason": None, "items": [1, None, {"x": None}]}
    output_path = tmp_path / "safe.h5ad"

    details = write_h5ad_safe(adata, str(output_path))

    assert details["save_mode"] == "clean_obs_var_uns_preflight"
    assert _null_encoded_paths(output_path) == []
    restored = ad.read_h5ad(output_path, backed="r")
    try:
        assert restored.shape == (3, 2)
    finally:
        restored.file.close()


def test_auto_checkpoint_uses_safe_h5ad_writer(tmp_path):
    adata = ad.AnnData(np.ones((3, 2)))
    adata.uns["checkpoint_metadata"] = {"source": None}
    agent = SimpleNamespace(
        smart_autonomous=True,
        adata=adata,
        run_manager=None,
        output_dir=tmp_path,
        verbose=False,
    )

    output_path = SCAgent._maybe_auto_checkpoint(agent, "run_batch_correction", {})

    assert output_path == str(tmp_path / "checkpoint_pre_batch_correction.h5ad")
    assert _null_encoded_paths(output_path) == []
    restored = ad.read_h5ad(output_path, backed="r")
    try:
        assert restored.shape == (3, 2)
    finally:
        restored.file.close()
