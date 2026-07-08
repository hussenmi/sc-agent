"""Regression tests for the annotation-save escape hatch and self-correcting
finalize validation message.

These cover the three changes that prevent a run from ending with no dataset on
disk when finalize_annotation cannot pass consensus validation:

1. world_state counts only *genuine* finalize attempts (evidence evaluated),
   not the trivial "stage evidence first" rejection.
2. the save guard hard-blocks until attempts are exhausted, then releases (or on
   explicit allow_unvalidated).
3. save_data degrades the release into a clearly-marked UNVALIDATED artifact.
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.agent import SCAgent
from scagent.agent.tools import process_tool_call
from scagent.agent.world_state import AgentWorldState


# --------------------------------------------------------------------------- #
# 1. finalize_attempts counts genuine validation failures only
# --------------------------------------------------------------------------- #

def _fresh_required_state() -> AgentWorldState:
    ws = AgentWorldState()
    ws.annotation_validation = {"required": True, "status": "evidence_staged_ready_to_finalize"}
    return ws


def test_genuine_finalize_failure_increments_attempts():
    ws = _fresh_required_state()
    ws._update_annotation_validation(
        "finalize_annotation",
        {"status": "error", "message": "Evidence validation failed: 38 issues across 19 cluster(s)",
         "validation_failures": ["Cluster 0: ...", "Cluster 1: ..."]},
    )
    assert ws.annotation_validation["finalize_attempts"] == 1
    assert "validation failed" in ws.annotation_validation["last_finalize_error"].lower()


def test_stage_first_rejection_does_not_count():
    ws = _fresh_required_state()
    ws._update_annotation_validation(
        "finalize_annotation",
        {"status": "error",
         "message": "Annotation evidence is required and must map every cluster to a label with evidence."},
    )
    assert ws.annotation_validation.get("finalize_attempts", 0) == 0


def test_attempts_accumulate_across_genuine_failures():
    ws = _fresh_required_state()
    for _ in range(3):
        ws._update_annotation_validation(
            "finalize_annotation",
            {"status": "error", "message": "Evidence validation failed: X", "validation_failures": ["c0"]},
        )
    assert ws.annotation_validation["finalize_attempts"] == 3


# --------------------------------------------------------------------------- #
# 2. save guard: block until attempts exhausted, then release
# --------------------------------------------------------------------------- #

def _agent_with_validation(validation: dict) -> SCAgent:
    agent = object.__new__(SCAgent)
    agent.world_state = SimpleNamespace(annotation_validation=validation)
    agent._mcp_client = None
    return agent


def test_guard_blocks_when_required_and_no_attempts():
    agent = _agent_with_validation({"required": True, "finalized": False})
    block = agent._annotation_validation_guard("save_data", {"output_path": "out.h5ad"})
    assert block is not None and block["status"] == "needs_validation"


def test_guard_releases_after_attempts_exhausted():
    agent = _agent_with_validation(
        {"required": True, "finalized": False,
         "finalize_attempts": SCAgent.MAX_FINALIZE_ATTEMPTS_BEFORE_UNVALIDATED_SAVE},
    )
    assert agent._annotation_validation_guard("save_data", {"output_path": "out.h5ad"}) is None


def test_guard_releases_on_explicit_allow_unvalidated():
    agent = _agent_with_validation({"required": True, "finalized": False, "finalize_attempts": 0})
    block = agent._annotation_validation_guard(
        "save_data", {"output_path": "out.h5ad", "allow_unvalidated": True}
    )
    assert block is None


def test_guard_passthrough_when_finalized():
    agent = _agent_with_validation({"required": True, "finalized": True})
    assert agent._annotation_validation_guard("save_data", {"output_path": "out.h5ad"}) is None


def test_guard_passthrough_when_not_required():
    agent = _agent_with_validation({"required": False})
    assert agent._annotation_validation_guard("save_data", {"output_path": "out.h5ad"}) is None


def test_guard_block_message_mentions_attempts_and_escape():
    agent = _agent_with_validation({"required": True, "finalized": False, "finalize_attempts": 1})
    block = agent._annotation_validation_guard("save_data", {"output_path": "out.h5ad"})
    assert block is not None
    assert "1 genuine finalize attempt" in block["message"]
    assert any("allow_unvalidated=true" in s for s in block["required_next_steps"])


# --------------------------------------------------------------------------- #
# 3. save_data degrades the release into an UNVALIDATED artifact
# --------------------------------------------------------------------------- #

def _toy_adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.poisson(1.0, size=(20, 8)).astype("float32"))
    a.obs["leiden"] = pd.Categorical(["0", "1"] * 10)
    return a


def test_save_data_marks_unvalidated_annotation():
    adata = _toy_adata()
    ws = SimpleNamespace(annotation_validation={
        "required": True, "finalized": False, "status": "evidence_staged_ready_to_finalize",
        "finalize_attempts": 2, "last_finalize_error": "Evidence validation failed: ...",
    })
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "final.h5ad")
        res_json, _ = process_tool_call("save_data", {"output_path": out}, adata, world_state=ws)
        res = json.loads(res_json)

        assert res["status"] == "ok"
        assert res["annotation_unvalidated"] is True
        # filename suffixed, original name NOT written
        assert res["output_path"].endswith("_UNVALIDATED.h5ad")
        assert os.path.exists(res["output_path"])
        assert not os.path.exists(out)
        assert any("UNVALIDATED" in w for w in res["warnings"])

        # the stamp persisted into the saved file
        reloaded = ad.read_h5ad(res["output_path"])
        assert reloaded.uns["annotation_status"] == "unvalidated"
        assert reloaded.uns["annotation_validation_note"]["finalize_attempts"] == 2


def test_save_data_normal_when_not_annotation_run():
    adata = _toy_adata()
    ws = SimpleNamespace(annotation_validation={})
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "final.h5ad")
        res_json, _ = process_tool_call("save_data", {"output_path": out}, adata, world_state=ws)
        res = json.loads(res_json)
        assert res["status"] == "ok"
        assert res["annotation_unvalidated"] is False
        assert res["output_path"] == out
        assert os.path.exists(out)
        assert "annotation_status" not in adata.uns


def test_save_data_normal_when_finalized():
    adata = _toy_adata()
    ws = SimpleNamespace(annotation_validation={"required": True, "finalized": True})
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "final.h5ad")
        res_json, _ = process_tool_call("save_data", {"output_path": out}, adata, world_state=ws)
        res = json.loads(res_json)
        assert res["annotation_unvalidated"] is False
        assert res["output_path"] == out
