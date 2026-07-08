"""run_code returns its captured stdout to the model. The old 2000-char cap was
too tight for inspecting structured data (annotation evidence across many
clusters), forcing the model to page through the same print across many calls.
The cap is now 8000 by default, env-configurable, and reports the true length."""

import json

from scagent.agent.tools import process_tool_call


def test_default_cap_is_8000_and_reports_total(monkeypatch):
    monkeypatch.delenv("SCAGENT_RUN_CODE_MAX_OUTPUT", raising=False)
    rj, _ = process_tool_call("run_code", {"code": 'print("x" * 9000)'}, adata=None)
    r = json.loads(rj)
    assert r["status"] == "ok"
    assert len(r["output"]) == 8000
    assert r["output_truncated"] is True
    assert r["output_total_chars"] == 9001  # 9000 'x' + newline


def test_env_override(monkeypatch):
    monkeypatch.setenv("SCAGENT_RUN_CODE_MAX_OUTPUT", "200")
    rj, _ = process_tool_call("run_code", {"code": 'print("y" * 500)'}, adata=None)
    r = json.loads(rj)
    assert len(r["output"]) == 200
    assert r["output_truncated"] is True


def test_small_output_not_truncated(monkeypatch):
    monkeypatch.delenv("SCAGENT_RUN_CODE_MAX_OUTPUT", raising=False)
    rj, _ = process_tool_call("run_code", {"code": 'print("hello")'}, adata=None)
    r = json.loads(rj)
    assert r["output"].strip() == "hello"
    assert "output_truncated" not in r
