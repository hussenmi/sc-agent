"""Tests for optional OpenTelemetry tracing of the agent loop (scagent.agent.tracing).

- Disabled by default (SCAGENT_TRACE unset): every call is a safe no-op and no
  tracer is initialized, so the default install/behavior is unchanged.
- Enabled (SCAGENT_TRACE=1): records a root span plus per-LLM and per-tool child
  spans carrying token-count attributes (this is what feeds the Phoenix view).
"""

from __future__ import annotations

import importlib
import json

_TRACE_ENV = ["SCAGENT_TRACE", "SCAGENT_OTLP_ENDPOINT", "SCAGENT_TRACE_CONSOLE",
              "SCAGENT_TRACE_PROJECT", "SCAGENT_TRACE_SERVICE", "SCAGENT_STEP_LOG",
              "SCAGENT_TRAJECTORY_LOG",
              "TRACEPARENT", "TRACESTATE", "OTEL_EXPORTER_OTLP_ENDPOINT"]


def _fresh(monkeypatch, **env):
    """Reload the tracing module with a clean, controlled environment."""
    for k in _TRACE_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import scagent.agent.tracing as t
    return importlib.reload(t)


def test_noop_when_disabled(monkeypatch):
    t = _fresh(monkeypatch)  # SCAGENT_TRACE unset
    # None of these should raise or initialize a tracer.
    t.start_root("scagent.analyze", {"scagent.provider": "openai"})
    t.record_llm(0, "m", 100, 5, t.now_ns())
    t.record_tool("run_qc", 0, t.now_ns())
    t.end_root()
    assert t.enabled() is False


def test_records_spans_when_enabled(monkeypatch):
    t = _fresh(monkeypatch, SCAGENT_TRACE="1", SCAGENT_TRACE_PROJECT="testproj")
    t.start_root("scagent.analyze", {"scagent.provider": "openai"})
    assert t.enabled() is True

    # Attach an in-memory exporter to the module's provider to capture spans.
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter = InMemorySpanExporter()
    t._provider.add_span_processor(SimpleSpanProcessor(exporter))

    t0 = t.now_ns()
    t.record_llm(2, "Qwen3.6-27B", 12345, 67, t0)
    t.record_tool("run_qc", 2, t0)
    t.end_root()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "llm.generate" in spans
    assert "tool.run_qc" in spans

    llm = spans["llm.generate"]
    assert llm.attributes.get("gen_ai.usage.input_tokens") == 12345
    assert llm.attributes.get("gen_ai.usage.output_tokens") == 67
    assert llm.attributes.get("scagent.iteration") == 2

    assert spans["tool.run_qc"].attributes.get("tool.name") == "run_qc"

    # Run-level totals are stamped on the root span (cumulative across the run).
    root = spans.get("scagent.analyze")
    assert root is not None
    assert root.attributes.get("scagent.run.input_tokens") == 12345
    assert root.attributes.get("scagent.run.output_tokens") == 67
    assert root.attributes.get("scagent.run.total_tokens") == 12412
    assert root.attributes.get("scagent.run.llm_calls") == 1


def test_step_log_written_when_tracing_disabled(monkeypatch, tmp_path):
    # The NAT step bridge must work even with OTel tracing OFF: record_llm/record_tool
    # append JSONL to SCAGENT_STEP_LOG regardless of whether a tracer is initialized.
    step_log = tmp_path / "steps.jsonl"
    t = _fresh(monkeypatch, SCAGENT_STEP_LOG=str(step_log))  # SCAGENT_TRACE unset
    assert t.enabled() is False

    t0 = t.now_ns()
    t.record_llm(1, "Qwen3.6-27B", 1000, 50, t0)
    t.record_tool("run_qc", 1, t0)

    rows = [json.loads(line) for line in step_log.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    llm = next(r for r in rows if r["type"] == "llm")
    assert llm["prompt_tokens"] == 1000
    assert llm["completion_tokens"] == 50
    assert llm["model"] == "Qwen3.6-27B"
    assert llm["end_ns"] >= llm["start_ns"]
    tool = next(r for r in rows if r["type"] == "tool")
    assert tool["name"] == "run_qc"
    assert tool["iteration"] == 1


def test_no_step_log_when_env_unset(monkeypatch):
    # Without SCAGENT_STEP_LOG, recording must not attempt any file write.
    t = _fresh(monkeypatch)
    t.record_llm(0, "m", 10, 1, t.now_ns())
    t.record_tool("run_qc", 0, t.now_ns())
    assert t._step_log() is None


def test_trajectory_log_captures_io_and_strips_base64(monkeypatch, tmp_path):
    # record_llm_io writes full (messages -> assistant) I/O as JSONL for distillation,
    # independent of OTel, with base64 image blobs stripped to keep files small.
    traj = tmp_path / "traj.jsonl"
    t = _fresh(monkeypatch, SCAGENT_TRAJECTORY_LOG=str(traj))
    assert t.enabled() is False  # works with OTel tracing off

    messages = [
        {"role": "system", "content": "You are a single-cell analyst."},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AAAABBBBCCCCDDDDEEEE=="}},
        ]},
    ]
    assistant = {"role": "assistant", "content": "Running QC.",
                 "tool_calls": [{"function": {"name": "run_qc", "arguments": "{}"}}]}
    t.record_llm_io(3, "Qwen3.6-27B", messages, assistant, t.now_ns())

    rows = [json.loads(line) for line in traj.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    r = rows[0]
    assert r["iteration"] == 3 and r["model"] == "Qwen3.6-27B"
    assert r["assistant"]["tool_calls"][0]["function"]["name"] == "run_qc"
    blob = json.dumps(r["messages"])
    assert "base64,AAAABBBB" not in blob       # the actual base64 payload is gone
    assert "<image_stripped>" in blob          # replaced with a placeholder


def test_no_trajectory_log_when_env_unset(monkeypatch):
    t = _fresh(monkeypatch)
    t.record_llm_io(0, "m", [{"role": "user", "content": "hi"}], {"content": "ok"}, t.now_ns())
    assert t._traj_log() is None
