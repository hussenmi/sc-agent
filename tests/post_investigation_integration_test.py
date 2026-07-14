"""Regression tests for the post-investigation integration return contract.

When the user selects `investigate_integration`, the runtime runs an uncorrected
first pass plus a structured batch-effect diagnostic, then returns to the user
with a concrete integrate / keep / separate decision instead of letting the
agent self-authorize integration.

The checkpoint is gated on the diagnostic having run: it fires only once
`adata.uns['batch_effect_diagnostic']['status'] == 'ok'`. The diagnostic verdict
also steers which option is pre-highlighted (scVI when a batch effect is
supported; keep-uncorrected when none is needed).

Context: in run_2026_06_17_182429 the recorded strategy was
`investigate_integration`, but after investigating the agent ran Harmony via
`run_code` on its own — there was no enforced point where the decision came back
to the user. This checkpoint is that point.
"""

from __future__ import annotations

import types

from scagent.agent.agent import SCAgent


class _FakeWorldState:
    def __init__(self, confirmed=None, data_summary=None):
        self._confirmed = confirmed or {}
        self.data_summary = data_summary or {}

    def get_confirmed_value(self, key):
        return self._confirmed.get(key)


def _diagnostic(
    status="ok", recommendation="integration_supported", batch_key="sample", n_batches=6
):
    return {
        "status": status,
        "recommendation": recommendation,
        "recommendation_reason": "test reason",
        "gene_evidence": "recurring_sample_associated",
        "design_interpretation": (
            "documented_technical_batch"
            if recommendation == "integration_supported"
            else "unknown"
        ),
        "batch_key": batch_key,
        "n_batches": n_batches,
        "selected_pairs": [],
        "recurrent_programs": [],
    }


def _agent(world_state, *, diagnostic=None, collaborative=True, smart_autonomous=False) -> SCAgent:
    agent = object.__new__(SCAgent)
    agent.world_state = world_state
    agent.collaborative = collaborative
    agent.smart_autonomous = smart_autonomous
    uns = {} if diagnostic is None else {"batch_effect_diagnostic": diagnostic}
    agent.adata = types.SimpleNamespace(uns=uns)
    return agent


def _investigating_ws(**overrides):
    data_summary = {
        "batch_key": "sample",
        "recommended_batch_key": "sample",
        "n_batches": 6,
        "batch_correction_applied": False,
        "processing": {"has_qc_metrics": True},
    }
    data_summary.update(overrides)
    return _FakeWorldState(
        confirmed={"multi_sample_strategy": "investigate_integration"},
        data_summary=data_summary,
    )


# --------------------------------------------------------------------------- #
# _post_investigation_strategy_checkpoint — fires once the diagnostic is done
# --------------------------------------------------------------------------- #

def test_fires_after_investigation_first_pass():
    agent = _agent(_investigating_ws(), diagnostic=_diagnostic())
    cp = agent._post_investigation_strategy_checkpoint()
    assert cp is not None
    assert cp["kind"] == "multi_sample_strategy"
    # Reuses the same decision_key so the answer overwrites the recorded strategy
    # and the existing run_batch_correction guard takes over.
    assert cp["decision_key"] == "multi_sample_strategy"


def test_investigate_option_is_dropped_after_investigation():
    agent = _agent(_investigating_ws(), diagnostic=_diagnostic())
    cp = agent._post_investigation_strategy_checkpoint()
    assert "investigate_integration" not in cp["option_actions"]
    for action in ("integrate_scvi", "keep_unintegrated", "analyze_separately"):
        assert action in cp["option_actions"]


def test_recommends_scvi_when_integration_supported():
    # Diagnostic supports a technical batch effect -> highlight scVI (policy method,
    # never Harmony) as the default.
    agent = _agent(
        _investigating_ws(), diagnostic=_diagnostic(recommendation="integration_supported")
    )
    cp = agent._post_investigation_strategy_checkpoint()
    assert cp["option_actions"][0] == "integrate_scvi"
    assert cp["recommendation"] == cp["options"][0]
    assert "scVI" in cp["recommendation"]


def test_recommends_keep_when_not_supported():
    # Any recommendation other than integration_supported -> highlight keeping
    # samples uncorrected (the conservative default; the user decides).
    agent = _agent(
        _investigating_ws(),
        diagnostic=_diagnostic(recommendation="do_not_integrate_based_on_current_evidence"),
    )
    cp = agent._post_investigation_strategy_checkpoint()
    assert cp["recommendation"] == cp["options"][1]
    assert cp["option_actions"][1] == "keep_unintegrated"


def test_strategy_dict_form_is_supported():
    ws = _investigating_ws()
    ws._confirmed["multi_sample_strategy"] = {"action": "investigate_integration"}
    cp = _agent(ws, diagnostic=_diagnostic())._post_investigation_strategy_checkpoint()
    assert cp is not None


# --------------------------------------------------------------------------- #
# diagnostic gating: does NOT fire until the diagnostic has run
# --------------------------------------------------------------------------- #

def test_no_checkpoint_without_diagnostic():
    # Investigation strategy selected, but the diagnostic has not run yet.
    cp = _agent(_investigating_ws(), diagnostic=None)._post_investigation_strategy_checkpoint()
    assert cp is None


def test_no_checkpoint_when_diagnostic_not_ok():
    cp = _agent(
        _investigating_ws(), diagnostic=_diagnostic(status="pending")
    )._post_investigation_strategy_checkpoint()
    assert cp is None


# --------------------------------------------------------------------------- #
# guards: does NOT fire outside the investigation-pending state
# --------------------------------------------------------------------------- #

def test_no_checkpoint_when_strategy_is_already_integrate():
    ws = _investigating_ws()
    ws._confirmed["multi_sample_strategy"] = "integrate_scvi"
    assert _agent(ws, diagnostic=_diagnostic())._post_investigation_strategy_checkpoint() is None


def test_no_checkpoint_when_batch_correction_already_applied():
    cp = _agent(
        _investigating_ws(batch_correction_applied=True), diagnostic=_diagnostic()
    )._post_investigation_strategy_checkpoint()
    assert cp is None


def test_no_checkpoint_when_no_batch_key():
    # No batch key in world_state *or* the diagnostic.
    cp = _agent(
        _investigating_ws(batch_key=None, recommended_batch_key=None),
        diagnostic=_diagnostic(batch_key=None),
    )._post_investigation_strategy_checkpoint()
    assert cp is None


def test_no_checkpoint_for_single_group():
    cp = _agent(
        _investigating_ws(n_batches=1), diagnostic=_diagnostic(n_batches=1)
    )._post_investigation_strategy_checkpoint()
    assert cp is None


def test_no_checkpoint_without_recorded_strategy():
    ws = _FakeWorldState(data_summary={"batch_key": "sample", "n_batches": 6})
    assert _agent(ws, diagnostic=_diagnostic())._post_investigation_strategy_checkpoint() is None


# --------------------------------------------------------------------------- #
# wiring: _build_checkpoint_payload routes run_clustering through the contract
# --------------------------------------------------------------------------- #

CLUSTERING_OK = {"status": "ok", "n_clusters": 12, "cluster_key": "leiden"}


def test_run_clustering_returns_post_investigation_checkpoint():
    agent = _agent(_investigating_ws(), diagnostic=_diagnostic())
    cp = agent._build_checkpoint_payload("run_clustering", {}, CLUSTERING_OK)
    assert cp is not None
    assert cp["kind"] == "multi_sample_strategy"
    assert "investigate_integration" not in cp["option_actions"]


def test_run_clustering_no_checkpoint_until_diagnostic_done():
    # Investigating but the diagnostic has not run -> no checkpoint, so the agent
    # continues (to run the diagnostic) rather than re-prompting prematurely.
    agent = _agent(_investigating_ws(), diagnostic=None)
    cp = agent._build_checkpoint_payload("run_clustering", {}, CLUSTERING_OK)
    assert cp is None


def test_run_clustering_falls_through_when_not_investigating():
    # No multi-sample investigation pending -> normal clustering checkpoint.
    ws = _FakeWorldState(data_summary={"processing": {"has_qc_metrics": True}})
    agent = _agent(ws)
    cp = agent._build_checkpoint_payload("run_clustering", {}, CLUSTERING_OK)
    assert cp is not None
    assert cp["decision_key"] == "clustering_next_step"
