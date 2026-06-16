"""NAT workflow function that runs scagent end-to-end as a subprocess.

scagent keeps its own loop, tools, and OpenAI-compatible client (pointed at a
vLLM or NIM endpoint via SCAGENT_BASE_URL). NAT treats it as one opaque step:
it runs scagent on the dataset `input` and returns a JSON blob locating the
output (run dir + annotated .h5ad) so the evaluator can score it.

Backend is selected purely by config (base_url/model), which is how we get the
"same eval, different serving backend" matrix (vLLM vs NIM vs Nemotron).
"""

import asyncio
import glob
import hashlib
import json
import os
import time

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

# h5ad files that are NOT the final annotated object.
_NON_FINAL = ("checkpoint", "pre_cleanup", "pre_batch", "intermediate")


class SCAgentAnalyzeConfig(FunctionBaseConfig, name="scagent_analyze"):
    """Config for invoking scagent's CLI from the (separate) NAT venv."""
    scagent_bin: str = "/usersoftware/peerd/ibrahih3/envs/scagent/bin/scagent"
    scagent_repo: str = "/data1/peerd/ibrahih3/cs_agent"
    provider: str = "openai"
    model: str = "Qwen3.6-27B"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    output_root: str = "/data1/peerd/ibrahih3/cs_agent/nat_runs"
    max_iterations: int = 75
    timeout_s: int = 5400  # 90 min hard cap per run
    # Tracing: when true, tell scagent to emit OTel spans to Phoenix and nest them
    # under NAT's workflow span (so the eval score + per-step token trace are one view).
    trace: bool = False
    otlp_endpoint: str = ""               # e.g. http://localhost:6006 (Phoenix)
    trace_project: str = "scagent-nat-eval"


def _subdirs(root: str) -> set[str]:
    return {p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)}


def _find_annotated_h5ad(run_dir: str) -> str | None:
    """Locate scagent's final annotated .h5ad in a run dir (manifest first)."""
    candidates: list[str] = []
    manifest = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest):
        try:
            m = json.load(open(manifest))
            for f in (m.get("output_files") or []):
                fp = f if os.path.isabs(f) else os.path.join(run_dir, os.path.basename(f))
                if fp.endswith(".h5ad"):
                    candidates.append(fp)
        except Exception:
            pass
    candidates += glob.glob(os.path.join(run_dir, "*.h5ad"))
    # Prefer files that exist, aren't checkpoints; favor 'annot' in the name.
    real = [c for c in dict.fromkeys(candidates)
            if os.path.exists(c) and not any(k in os.path.basename(c).lower() for k in _NON_FINAL)]
    if not real:
        return None
    real.sort(key=lambda c: ("annot" not in os.path.basename(c).lower(), -os.path.getmtime(c)))
    return real[0]


_NAT_PROVIDER = None
_NAT_TRACER = None


def _nat_tracer(config):
    """Lazily build a TracerProvider exporting to Phoenix, for the nat-side parent span.

    NAT emits no OTel span for an opaque subprocess function, so we create our own
    parent span per eval item and nest scagent under it via the injected traceparent.
    We deliberately do NOT set the global provider (so NAT's own observability is
    untouched) — spans are created from this provider and attached to the OTel context.
    """
    global _NAT_PROVIDER, _NAT_TRACER
    if _NAT_TRACER is not None:
        return _NAT_TRACER
    if not (config.trace and config.otlp_endpoint):
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return None
    prov = TracerProvider(resource=Resource.create({
        "service.name": "scagent-nat",
        "openinference.project.name": config.trace_project,
    }))
    prov.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=config.otlp_endpoint.rstrip("/") + "/v1/traces")))
    _NAT_PROVIDER = prov
    _NAT_TRACER = prov.get_tracer("scagent-nat")
    return _NAT_TRACER


@register_function(config_type=SCAgentAnalyzeConfig)
async def scagent_analyze(config: SCAgentAnalyzeConfig, builder: Builder):

    os.makedirs(config.output_root, exist_ok=True)

    async def _run(request: str) -> str:
        # `request` is a JSON string: {"instruction": "...", "data_path": "..."}
        spec = json.loads(request) if isinstance(request, str) else dict(request)
        instruction = spec["instruction"]
        data_path = spec["data_path"]

        # Parent span per eval item; scagent's per-step spans nest under it via the
        # injected traceparent -> one Phoenix trace per item.
        tracer = _nat_tracer(config)
        parent = ctx_token = None
        if tracer is not None:
            from opentelemetry import context as _otctx
            from opentelemetry import trace as _ot
            parent = tracer.start_span("scagent.eval_item", attributes={
                "scagent.data_path": data_path,
                "scagent.instruction": instruction[:300],
                "openinference.span.kind": "CHAIN",
            })
            ctx_token = _otctx.attach(_ot.set_span_in_context(parent))

        def _finalize(rc=None, elapsed=None, run_dir=None, annotated=None, error=None):
            if parent is None:
                return
            from opentelemetry import context as _otctx2
            if rc is not None:
                parent.set_attribute("scagent.returncode", rc)
            if elapsed is not None:
                parent.set_attribute("scagent.elapsed_s", elapsed)
            if run_dir:
                parent.set_attribute("scagent.run_dir", run_dir)
            if annotated:
                parent.set_attribute("scagent.annotated_h5ad", annotated)
            if error:
                parent.set_attribute("error", error)
            if ctx_token is not None:
                _otctx2.detach(ctx_token)
            parent.end()
            if _NAT_PROVIDER is not None:
                try:
                    _NAT_PROVIDER.force_flush()
                except Exception:
                    pass

        run_name = "nat_%s_%d" % (hashlib.sha1(request.encode()).hexdigest()[:8], int(time.time()))
        env = {
            **os.environ,
            "SCAGENT_PROVIDER": config.provider,
            "SCAGENT_MODEL": config.model,
            "SCAGENT_BASE_URL": config.base_url,
            "SCAGENT_API_KEY": config.api_key,
            "OPENAI_API_KEY": config.api_key,
        }
        if config.trace:
            env["SCAGENT_TRACE"] = "1"
            if config.otlp_endpoint:
                env["SCAGENT_OTLP_ENDPOINT"] = config.otlp_endpoint
            if config.trace_project:
                env["SCAGENT_TRACE_PROJECT"] = config.trace_project
            try:  # propagate NAT's active span as W3C traceparent so scagent nests under it
                from opentelemetry.propagate import inject
                carrier: dict = {}
                inject(carrier)
                if carrier.get("traceparent"):
                    env["TRACEPARENT"] = carrier["traceparent"]
                if carrier.get("tracestate"):
                    env["TRACESTATE"] = carrier["tracestate"]
            except Exception:
                pass
        cmd = [
            config.scagent_bin, "analyze", instruction,
            "--data", data_path,
            "--output", config.output_root,
            "--name", run_name,
            "--single-run", "--quiet", "--smart",
            "--provider", config.provider, "--model", config.model,
            "--max-iterations", str(config.max_iterations),
        ]

        t0 = time.time()
        before = _subdirs(config.output_root)
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=config.scagent_repo, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=config.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            _finalize(rc=-1, error="timeout")
            return json.dumps({"error": "timeout", "run_name": run_name})
        tail = (out_b or b"").decode(errors="replace")[-2000:]

        # Locate the run dir: prefer name match, else newest dir created this call.
        named = [d for d in _subdirs(config.output_root) if run_name in os.path.basename(d)]
        fresh = [d for d in _subdirs(config.output_root) if d not in before]
        pool = named or fresh
        run_dir = max(pool, key=os.path.getmtime) if pool else None
        annotated = _find_annotated_h5ad(run_dir) if run_dir else None

        _finalize(rc=proc.returncode if proc.returncode is not None else -1,
                  elapsed=round(time.time() - t0, 1), run_dir=run_dir, annotated=annotated)
        return json.dumps({
            "run_dir": run_dir,
            "annotated_h5ad": annotated,
            "returncode": proc.returncode,
            "elapsed_s": round(time.time() - t0, 1),
            "stdout_tail": tail,
        })

    yield FunctionInfo.from_fn(
        _run,
        description=("Run scagent end-to-end scRNA-seq analysis + cell-type annotation on a "
                     "dataset. Input is JSON {instruction, data_path}; returns JSON with the "
                     "run directory and the path to the annotated .h5ad."),
    )
