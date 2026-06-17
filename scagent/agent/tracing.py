"""Optional OpenTelemetry tracing for the scagent agent loop.

No-op unless ``SCAGENT_TRACE`` is set AND opentelemetry is installed (the
``[tracing]`` extra) — so the default install and behavior are unchanged.

When enabled, ``analyze()`` opens a root span per turn, and the loop records a
child span per LLM call (with input/output token counts) and per tool call
(with name + latency). Spans are recorded *after the fact* with explicit
start/end timestamps, so the agent loop needs no re-indentation — just a
``now_ns()`` grab and a ``record_*`` call.

If a W3C ``traceparent`` is present in the environment (propagated from NAT's
eval workflow span into the scagent subprocess), the root nests under it, so the
whole run renders as one trace in Phoenix alongside NAT's eval score.

Env:
  SCAGENT_TRACE=1            enable tracing
  SCAGENT_OTLP_ENDPOINT=...  OTLP/HTTP collector (e.g. http://localhost:6006 for Phoenix)
  SCAGENT_TRACE_CONSOLE=1    also print spans to stderr (handy for validation)
  TRACEPARENT / TRACESTATE   W3C context to nest under (set by the NAT wrapper)
"""

import os
import time

_tracer = None
_provider = None
_root = None  # current root span

# Per-run token accumulators, stamped onto the root span at end_root so the
# whole-run input/output split is visible in the UI without querying. Reset when
# the run changes (detected via the propagated TRACEPARENT).
_run_key = None
_in_tokens = 0
_out_tokens = 0
_llm_calls = 0


def _truthy(v) -> bool:
    return str(v).lower() in {"1", "true", "yes", "on"}


def now_ns() -> int:
    return time.time_ns()


def enabled() -> bool:
    return _tracer is not None


def _init() -> None:
    global _tracer, _provider
    if _tracer is not None or not _truthy(os.environ.get("SCAGENT_TRACE", "")):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception:
        return  # opentelemetry not installed -> stay a no-op

    _res_attrs = {"service.name": os.environ.get("SCAGENT_TRACE_SERVICE", "scagent")}
    if os.environ.get("SCAGENT_TRACE_PROJECT"):
        # Same resource attribute Phoenix/NAT use, so scagent + NAT spans share a project.
        _res_attrs["openinference.project.name"] = os.environ["SCAGENT_TRACE_PROJECT"]
    prov = TracerProvider(resource=Resource.create(_res_attrs))
    endpoint = os.environ.get("SCAGENT_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            ep = endpoint if endpoint.endswith("/v1/traces") else endpoint.rstrip("/") + "/v1/traces"
            prov.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=ep)))
        except Exception:
            pass
    if _truthy(os.environ.get("SCAGENT_TRACE_CONSOLE", "")):
        prov.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(prov)
    _provider = prov
    _tracer = prov.get_tracer("scagent")  # bind to our provider (not the global) so exports are deterministic


def _parent_context():
    """Extract a W3C traceparent (propagated by NAT) from the environment."""
    try:
        from opentelemetry.propagate import extract
        carrier = {}
        if os.environ.get("TRACEPARENT"):
            carrier["traceparent"] = os.environ["TRACEPARENT"]
        if os.environ.get("TRACESTATE"):
            carrier["tracestate"] = os.environ["TRACESTATE"]
        return extract(carrier) if carrier else None
    except Exception:
        return None


def start_root(name: str, attributes: dict | None = None) -> None:
    global _root, _run_key, _in_tokens, _out_tokens, _llm_calls
    _init()
    if _tracer is None:
        return
    run_key = os.environ.get("TRACEPARENT") or "local"
    if run_key != _run_key:  # new run -> reset the per-run token accumulators
        _run_key, _in_tokens, _out_tokens, _llm_calls = run_key, 0, 0, 0
    attrs = {"openinference.span.kind": "AGENT"}  # so Phoenix labels the root 'AGENT', not 'unknown'
    if attributes:
        attrs.update(attributes)
    _root = _tracer.start_span(name, context=_parent_context(), attributes=attrs)


def end_root() -> None:
    global _root
    if _root is not None:
        # Stamp run-level token totals so the input/output split shows on the root
        # span in the UI (custom scagent.run.* keys — Phoenix won't double-count them
        # against its llm.token_count rollup). Cumulative across the run's turns.
        _root.set_attribute("scagent.run.llm_calls", _llm_calls)
        _root.set_attribute("scagent.run.input_tokens", _in_tokens)
        _root.set_attribute("scagent.run.output_tokens", _out_tokens)
        _root.set_attribute("scagent.run.total_tokens", _in_tokens + _out_tokens)
        _root.end()
        _root = None
    if _provider is not None:
        try:
            _provider.force_flush()  # short runs: ensure spans export before exit
        except Exception:
            pass


def _child(name: str, t0_ns: int, attributes: dict) -> None:
    if _tracer is None or _root is None:
        return
    from opentelemetry import trace
    span = _tracer.start_span(
        name, context=trace.set_span_in_context(_root), start_time=t0_ns, attributes=attributes)
    span.end(end_time=now_ns())


def record_llm(iteration, model, input_tokens, output_tokens, t0_ns) -> None:
    global _in_tokens, _out_tokens, _llm_calls
    if _tracer is None or _root is None:
        return
    _llm_calls += 1
    if input_tokens is not None:
        _in_tokens += int(input_tokens)
    if output_tokens is not None:
        _out_tokens += int(output_tokens)
    attrs = {
        "scagent.iteration": iteration,
        "openinference.span.kind": "LLM",
        "gen_ai.system": "openai-compatible",
        "gen_ai.request.model": str(model),
    }
    if input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = int(input_tokens)
        attrs["llm.token_count.prompt"] = int(input_tokens)
    if output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = int(output_tokens)
        attrs["llm.token_count.completion"] = int(output_tokens)
    _child("llm.generate", t0_ns, attrs)


def record_tool(name, iteration, t0_ns) -> None:
    if _tracer is None or _root is None:
        return
    _child(f"tool.{name}", t0_ns,
           {"scagent.iteration": iteration, "tool.name": name, "openinference.span.kind": "TOOL"})
