# scagent under load: profiling & concurrency results (NAT Tier-1)

*Backend: Qwen3.6-27B served via vLLM (iscn008:8000). Workload: LuCA NSCLC
study annotation (`LUNG_T06`). Profiler: NeMo Agent Toolkit (NAT) v1.7.0.
Sweep: `nat_sweep_20260622_160026/`.*

## What this measures and why NAT

We ran the same scagent analysis at nominal concurrency 1, 2, and 4 (4 reps
each) and let NAT's profiler capture per-LLM-call and per-tool timings via the
per-step event bridge. The question is the one a *deployment* owner asks and a
single-user benchmark can't answer: **what happens to latency when several users
hit the agent at once, and where is the time actually going?**

## Finding 1 — Throughput scales; the cost is the tail, not the median

| Nominal concurrency | LLM median | LLM p95 | End-to-end wall-clock | LLM calls |
|---|---|---|---|---|
| 1 | 15.5 s | 59.0 s | 4156 s | 142 |
| 2 | 22.2 s | 98.1 s | 3029 s | 179 |
| 4 | 19.3 s | **148.0 s** | **1894 s** | 366 |

- Going 1→4 concurrent users cut total wall-clock **2.2×** — the vLLM backend
  genuinely absorbs parallel load rather than time-slicing one queue.
- Median per-call latency barely moves (15→22→19 s), but **p95 latency 2.5×'d**
  (59→148 s). The actionable risk under multi-user load is the *tail*: typical
  steps stay fast while the slow (long-context) steps get much slower. That's an
  SLA signal invisible to any single-user run.

## Finding 2 — The dominant bottleneck is a *tool*, not the LLM

NAT's bottleneck report (score = avg_duration × max_concurrency) ranks operations
across the whole workflow:

| Rank | Operation | Avg duration | Note |
|---|---|---|---|
| 1 | TOOL `run_scimilarity` | **73.3 s** | foundation-model cell annotation |
| 2 | LLM `Qwen3.6-27B` | 22.8 s | per-call mean |
| 3 | TOOL `run_neighbors` | 14.9 s | kNN graph |

The single most expensive step in a scagent run is **not the LLM** — it's the
SCimilarity annotation tool (3× a mean LLM call). This is exactly the
Anthropic "reliability lives in the validated tool layer" thesis showing up in
the *cost* profile: most of the wall-clock is deterministic scientific compute
the orchestrator is waiting on, not token generation. Practical implication —
optimizing/serving the embedding model buys more than swapping the LLM.

## Finding 3 — At this scale, contention isn't the latency driver

Bucketing every LLM call by the *instantaneous* number in flight (computed from
timestamps, not nominal level) gives a non-monotonic curve: e.g. 16-in-flight
shows a *lower* median (9.0 s) than 2-in-flight (33 s). That isn't noise to
explain away — it says latency here is **prompt-shape-dominated**: bursts of high
in-flight count are the cheap early QC fan-out, while the expensive long-context
annotation calls happen to run with few in flight. The backend has headroom at
conc=4; the variance is scagent's own heterogeneous step mix, not queueing.

*(This panel is supporting evidence, not an article headline — a cleaner serving
curve would hold prompt shape constant. Finding 1's left panel is the
publishable sizing result.)*

## Update 2026-06-23 — Nemotron-3-Ultra + VLM-NIM, full pipeline (single-stream)

*Backend swap: main reasoner **Nemotron-3-Ultra-550B-A55B (NVFP4)** via vLLM
(iscp001, TP=8, MTP k=5); vision via the **NVIDIA NIM** for
`nemotron-nano-12b-v2-vl` (iscn008:8000, NIM-packaged **vLLM**, FP8, TP=1).
Same LuCA `LUNG_T06` workload, single-stream (conc=1), profiling.yml, Weave-traced.
Run: `nat_runs/run_2026_06_23_000117_nat_e78cf76b_1782187264/`.*

This is the first end-to-end run of the **two-model NIM story**: a vLLM-served
frontier reasoner driving a NIM-served VLM for figure interpretation, the whole
agent profiled by NAT. It both validates the pipeline and turns Finding 3's
"prompt-shape-dominated" claim into a hard number.

### Result — the agent is **prefill-bound, not decode-bound**

| Metric | Value |
|---|---|
| End-to-end wall-clock | **1147 s** (~19 min) |
| Nemotron LLM calls | 53 |
| VLM-NIM calls | ~20 (figure interpretation) |
| **Avg prompt tokens / LLM call** | **~147,100** |
| Avg total tokens / call | ~147,850 (completions are tiny — prompts dominate) |
| LLM latency median / p95 | ~12 s / **28.4 s** |
| Concurrency | 1 |

The agent re-enters each turn carrying a **~147K-token context**, so per-call
latency is dominated by **prefilling ~147K tokens**, not by generation — even
though Nemotron decodes at ~264 tok/s in isolation. **For this agent the serving
metric that matters is long-context prefill throughput + prefix-cache hit rate,
not raw decode tok/s.** A NIM-packaged Nemotron would hit the identical wall; the
lever is prefix-caching / chunked-prefill config, not the engine. This is the
direct, single-stream confirmation of Finding 3 (measured here, not inferred from
the in-flight bucketing).

### Cross-model (vs the Qwen3.6-27B conc=1 row above)

| | Qwen3.6-27B (vLLM) | Nemotron-3-Ultra (vLLM) + VLM-NIM |
|---|---|---|
| LLM median | 15.5 s | ~12 s |
| LLM p95 | 59.0 s | **28.4 s** |
| Bottleneck #1 | `run_scimilarity` 73.3 s | `run_scimilarity` **69.0 s** |

Two things hold across models: (1) the **`run_scimilarity` tool is the #1
bottleneck regardless of LLM** (73 s ≈ 69 s) — Finding 2 is model-agnostic; and
(2) Nemotron-3-Ultra's **tail is ~2× tighter** (p95 28 s vs 59 s) — the larger
NVFP4 reasoner handles the long-context annotation turns more evenly than the 27B.

### NIM-in-the-loop note

The vision NIM auto-selected the `vllm-h200-nvl-fp8-tp1-pp1` profile — i.e. it
**is** packaged vLLM, not TRT-LLM. So "NIM vs vLLM" here is *productization*
(auto-profile, OpenAI+Anthropic endpoints, ops) over the **same engine**, not a
kernel difference. The VLM-NIM-under-Singularity boot bug is solved
(`--no-eval` + `NIM_MEDIA_IO_KWARGS='{}'`); see `docs/serving_findings.md`.

### One reliability flag

An **EMERGENCY context-compact** fired during annotation — context hit **263K,
just over Nemotron's 262K native cap** — and recovered. scagent's annotation
stage is context-heavy enough to saturate a 256K window; on a smaller-context
model this phase would force harder compaction.

### Artifacts (this run)

- Weave: `wandb.ai/.../scagent-nat-profiling/weave` (project `scagent-nat-profiling`)
- `nat_profiling_out/workflow_profiling_{report.txt,metrics.json}`, `gantt_chart.png`
- `nat_runs/run_2026_06_23_000117_.../` (run dir, `LUNG_T06_annotated.h5ad`)
- Reproduce: `nat eval --config_file nat_integration/configs/profiling.yml --override workflow.model Nemotron-3-Ultra --override workflow.base_url http://iscp001:8000/v1 --reps 1` (Weave needs `WANDB_API_KEY` in env)

## Artifacts

- `nat_sweep_20260622_160026/figs/latency_vs_concurrency.png` — sizing curve
- `nat_sweep_20260622_160026/figs/sweep_summary.{md,json}` — machine-readable
- `nat_sweep_20260622_160026/conc_{1,2,4}/` — per-level NAT profiler dumps
  (gantt chart, simple/nested bottleneck stacks, `standardized_data_all.csv`)

## Reproduce

```bash
# sweep (latency-sensitive; run on an idle node, Qwen on iscn008:8000)
bash nat_integration/run_concurrency_sweep.sh
# analyze any completed sweep dir -> figs/
/usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/python \
  nat_integration/analyze_concurrency.py --sweep nat_sweep_<TS>
```
