"""Regression tests for write_json string-payload robustness.

Weaker/self-hosted models tend to stringify a payload into the `data` argument.
write_json must (a) parse common quirks — markdown fences, python literals,
trailing commas — and (b) when a payload is genuinely truncated, return an
*actionable* error steering to the run_code+json.dump file path, not a dead-end
that gets retried the same way.
"""

from __future__ import annotations

import json
import os
import tempfile

import anndata as ad
import numpy as np

from scagent.agent.tools import process_tool_call


def _adata():
    return ad.AnnData(X=np.zeros((4, 3), dtype="float32"))


class _RunManager:
    """Minimal run_manager so write_json writes into a temp dir."""

    def __init__(self, d):
        self.d = d
        self.outputs = []

    def write_json_report(self, name, payload):
        p = os.path.join(self.d, f"{name}.json")
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
        self.outputs.append(p)
        return p


def _write(data, rm):
    res_json, _ = process_tool_call(
        "write_json", {"name": "ev", "data": data}, _adata(), run_manager=rm
    )
    return json.loads(res_json)


def test_accepts_fenced_json_string():
    with tempfile.TemporaryDirectory() as d:
        res = _write('```json\n{"0": {"label": "Pro-B"}}\n```', _RunManager(d))
        assert res["status"] == "ok"
        assert res["n_entries"] == 1
        assert os.path.exists(res["json_path"])
        assert json.load(open(res["json_path"]))["0"]["label"] == "Pro-B"


def test_accepts_python_literal_string():
    with tempfile.TemporaryDirectory() as d:
        res = _write("{'0': {'label': 'HSC/MPP', 'ok': True}}", _RunManager(d))
        assert res["status"] == "ok"
        assert res["n_entries"] == 1


def test_accepts_real_object():
    with tempfile.TemporaryDirectory() as d:
        res = _write({"0": {"label": "CMP"}, "1": {"label": "pDC"}}, _RunManager(d))
        assert res["status"] == "ok"
        assert res["n_entries"] == 2


def test_truncated_payload_gives_actionable_error():
    # unbalanced braces -> looks truncated -> steer to run_code/json.dump
    res = _write('{"0": {"label": "Pro-B", "supporting_genes": ["CD79', None)
    assert res["status"] == "error"
    assert "truncated" in res["message"].lower()
    assert any("json.dump" in opt for opt in res["recovery_options"])
    # must NOT just tell it to retry the same stringified blob
    assert any("register_artifact" in opt or "evidence_path" in opt
               for opt in res["recovery_options"])


def test_plain_nonjson_string_gives_object_hint():
    res = _write("just some prose, not json at all", None)
    assert res["status"] == "error"
    assert "must be a JSON object" in res["message"]
