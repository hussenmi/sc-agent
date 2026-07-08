# NVIDIA Collaboration — Project Brief

> **Purpose of this doc.** Onboarding context for an AI coding agent (Claude Code)
> or a human picking up the NVIDIA collaboration work — including on the **DGX
> Spark**. It distills the goal, the pieces we're building, what we've found, and
> what's blocked on hardware. Host-specific facts are labeled **[Iris]** (the
> MSKCC HPC where most of this was built) vs **[Spark]** (the loaned DGX Spark).
> When they conflict, the Spark is the newer, more capable host — prefer it for
> anything the Iris section marks as blocked.

## TL;DR

We wrap **scagent** (this repo — an agentic single-cell RNA-seq analysis tool)
in NVIDIA's agent/serving stack and evaluate it end-to-end. The collaboration
began after NVIDIA reached out about the user's Towards Data Science serving
article and offered a DGX Spark loan + collaboration on the NeMo Agent Toolkit /
AI-Q. There are four arms, at different "levels":

| Arm | Layer | Status |
|-----|-------|--------|
| **NAT** (NeMo Agent Toolkit) | Observability / eval / profiling | **Built, working end-to-end** |
| **NIM / vLLM serving** | The model backend | **Working**; serving findings collected |
| **OpenShell / NemoClaw** | Secure sandbox runtime | **Blocked on Iris**; the Spark unblocks it |
| **AI-Q** | Higher-level RAG/report orchestration | Deferred; orthogonal, optional later |

## Why the DGX Spark matters

The Spark is not just "more compute" — it is the exact host that unblocks the
**OpenShell** arm we shelved, and it changes the serving story:

- **OpenShell can finally run.** On Iris it hit three hard, root-level blockers
  (see below). The Spark ships modern Ubuntu (kernel ≥ 6.x → **Landlock present**),
  **cgroups v2**, and normal **subuid/subgid** ranges — the three things Iris
  lacked. This is the priority work to move onto the Spark.
- **Native NVFP4.** The Spark is **Blackwell (GB10)**, so NVFP4 checkpoints
  (e.g. Nemotron-3-Ultra) run natively rather than being coerced onto Hopper.
- **Docker, not Singularity-only.** Much of the Iris pain (NIM-under-Singularity
  bugs, MPI/ray workarounds) came from having no Docker. The Spark has a normal
  container runtime, so NIM/NemoClaw install the intended way.

## The four arms in detail

### 1. NAT — NeMo Agent Toolkit (observability / eval / profiling)

The most built-out arm. scagent runs as an **opaque subprocess** wrapped by NAT;
a per-step event bridge feeds NAT's profiler so it sees real per-tool durations
and per-LLM token counts despite the subprocess boundary.

- Lives in `nat_integration/` in this repo (installed editable into a **separate
  Python 3.11 venv**, because NAT needs ≥3.11 and scagent's venv is 3.10 [Iris]).
- Key pieces: `scagent_nat/workflow.py` (runs scagent CLI, replays a JSONL step
  log into NAT's `IntermediateStep` bus), `scagent_nat/luca_atlas_eval.py` (our
  own accuracy scorer wrapped as a NAT custom evaluator), `annotation_eval.py`
  (ARI/NMI), `aggregate_eval.py` (cross-run figures).
- **Division of labor (decided):** NAT is *not* the accuracy judge — we score
  accuracy better ourselves against LuCA atlas truth. NAT's real value is the
  profiling / serving / trajectory / consistency layer: bottleneck analysis,
  concurrency/sizing, prefix-span tool-pattern mining, `dynamo_metrics` (scrape
  serving Prometheus), trajectory eval, and `--reps` consistency distributions.
- Observability exports to **Phoenix** (node-local) and **Weave** (W&B cloud,
  shareable). Weave dashboard:
  `wandb.ai/hussenmibrahim-memorial-sloan-kettering-cancer-center/scagent-nat-luca/weave`.
- scagent emits its own OpenTelemetry spans (guarded, off by default) via
  `scagent/agent/tracing.py`; env knobs `SCAGENT_TRACE`, `SCAGENT_OTLP_ENDPOINT`,
  `SCAGENT_STEP_LOG`, `SCAGENT_TRAJECTORY_LOG`.

### 2. NIM / vLLM serving

An article-grade head-to-head: **NVIDIA NIM (TensorRT-LLM) vs hand-tuned vLLM**
as scagent's LLM backend, plus a vision sidecar for figures.

- Backends exercised: Qwen3.6-27B (FP8+MTP on vLLM), Qwen2.5-Coder-32B (the
  mature TRT-LLM NIM), and **Nemotron-3-Ultra-550B NVFP4** on vLLM.
- Launchers: `start_vllm.sh` (generalized with `GPU_IDS` + per-model
  `MODEL_EXTRA`), `start_nim.sh`, `start_trtllm.sh`, `start_vlm.sh` (vision).
- **Vision sidecar:** text-only reasoners (Nemotron, GLM) can't see figures, so a
  separate VLM describes plots and the reasoner interrogates via a
  `describe_image` tool. `SCAGENT_MAIN_HAS_VISION` / `SCAGENT_VISION_MODEL` etc.
  control routing. A/B finding: a frontier general VLM (Qwen3.6-27B) is a more
  faithful describer of dense sci figures than the smaller Nemotron-Nano-VL,
  which hallucinates specifics on dense UMAPs.
- Benchmark harness in `experiments/`: `bench_llm.py`, `bench_prefix_cache.py`,
  `bench_toolcall.py`.

### 3. OpenShell / NemoClaw (secure sandbox runtime) — the Spark's headline job

scagent's `run_code` is a hand-rolled in-process sandbox (blocks `os`/`sys`/
`subprocess`; injects helpers). **NemoClaw = a thin wrapper; the real engine is
NVIDIA OpenShell** (separate repo, Rust+Python, Apache-2.0, alpha). OpenShell's
model — out-of-process kernel enforcement (Landlock + seccomp + netns) — is the
right way to harden scagent's sandbox, same threat model but real isolation.

- **Architecture:** CLI →gRPC→ gateway (host daemon, port 17670) →
  PodmanComputeDriver → Podman REST API → sandbox container; an
  `openshell-sandbox` supervisor inside the container applies Landlock + seccomp
  + a nested netns/veth egress proxy.
- **Install without root:** use the GitHub release tarballs (SHA256-verified).
  The **musl** static CLI is the glibc-independent artifact; the PyPI/RPM builds
  target glibc 2.39.
- **On the Spark:** this should just work — re-run the spike here. Evaluate
  OpenShell *directly*; skip the NemoClaw wrapper (it bundles a whole agent
  runtime + a Docker-socket mount we don't want).

### 4. AI-Q — deferred

The AI-Q *Blueprint* (deep-research/RAG report app built on NAT + LangChain Deep
Agents; NIM-only, Docker/Helm). Orthogonal to scagent; treat as a later optional
comparison. Only interesting seam = exposing our biocontext MCP through AI-Q.

## Key findings so far (the deliverables)

- **The agent is prefill-bound, not decode-bound.** Average prompt is ~147K
  tokens/call with tiny completions, so per-turn latency is dominated by
  prefilling context, not generation. **The serving metric that matters for
  scagent is long-context prefill throughput + prefix-cache hit rate, not decode
  tok/s.** The lever is prefix-caching / chunked-prefill, not the engine.
- **Output is consistent; the trajectory is not.** Same input, repeated runs:
  cluster/lineage labels are stable (coarse LuCA accuracy locked ~0.94), but the
  agent's *path* varies a lot — LLM calls swing 29→41, prompt tokens ±48% per
  run. Notably the *cheapest* run sometimes gets the *best* accuracy: more agent
  work ≠ better answer. This is a direct serving-cost + agent-reliability story,
  and it's what NAT made visible.
- **Fine-grained annotation has real run-to-run variance** even when lineage is
  rock-solid (24-class LuCA accuracy band ~0.79–0.87 vs coarse ~0.94).
- **Compute bottlenecks** are model-agnostic: `run_scimilarity`, `run_neighbors`,
  `run_qc` dominate; RAPIDS (`rapids_singlecell`) accelerates the compute chain
  ~20× on large data (see the `rapids-gpu-*` branches).

Written up in `docs/serving_findings.md` and
`nat_integration/RESULTS_concurrency_profiling.md`.

## Evaluation data & scorer

- **Benchmark:** scagent QC + annotation per study vs the **LuCA NSCLC atlas**
  ground-truth labels (Salcher 2022, CELLxGENE-curated).
- Staged inputs and truth CSVs live under `/home/ibrahih3/luca_bench/` **[Iris]**
  (LUNG_T06 3.4k for fast wiring, lung6 ~20k, Reyfman 43k pre-QC for QC testing,
  KimLee 208k whole-study). These will need re-staging on the Spark.
- **Scorer:** `/home/ibrahih3/luca_bench/compare_run_to_atlas.py` **[Iris]** —
  computes ARI/NMI, majority-map annotation accuracy (coarse + major), per-cluster
  purity, QC overlap. It is the objective layer; biological *interpretation* is a
  separate Claude Code pass written to `INTERPRETATION.md`.
- Never score against scagent's own prior annotated output (circular) — use
  independent atlas truth only.

## Iris-specific constraints — mostly DO NOT apply on the Spark

These shaped the Iris build but are the *reason* the Spark helps. Flagged so they
don't mislead on the new host:

- **[Iris] OpenShell blocked by three root-level gaps:** (1) node is **cgroups
  v1**, OpenShell requires v2 (hard stop at gateway init); (2) **no subuid/subgid**
  ranges → rootless podman single-UID mapping breaks privilege drop; (3) **kernel
  4.18 < 5.13 → no Landlock**, so filesystem isolation silently degrades to off.
  → **[Spark] all three are resolved** (cgroups v2, subuid ranges, modern kernel).
- **[Iris] No Docker — Singularity only.** NIMs pulled via `singularity pull
  docker://nvcr.io/...`; caused the NIM-VLM `media-io-kwargs` mangling bug and the
  TRT-LLM MPI/ray workarounds. → **[Spark] normal container runtime.**
- **[Iris] glibc 2.28** → OpenShell's PyPI wheel (manylinux_2_39) and RPMs
  (`.fc44`) don't run; only the musl CLI + the GLIBC_2.28-max gateway binary do.
  → **[Spark] modern glibc**, so the normal artifacts work.
- **[Iris] NVFP4 coerced onto Hopper H200** (worked at 4× H200, a config NVIDIA
  didn't list). → **[Spark] native Blackwell NVFP4.**
- **[Iris] `/data1/peerd` shared storage oscillates near-full** (intermittent
  `ENOSPC`); scratch went to node-local `/tmp` or `/home/wekafs`. Verify the
  Spark's own disk layout; this constraint is Iris-specific.
- **[Iris] Compute nodes change frequently** (iscn008 / iscp001 / iscb015…) and
  the user launches their own vLLM; always probe which node/GPU/port is live
  before running. Less relevant on a single dedicated Spark.

## Environment notes (verify on the Spark before trusting)

- **[Iris]** scagent runs under a `uv`-managed venv: `source setup.sh`, install
  with `uv pip install` (no bare `pip`). GPU runs use the `scagent_rapids` conda
  env. NAT lives in its own Python 3.11 venv. On the Spark, set these up fresh —
  paths under `/usersoftware/peerd/...` and `/home/ibrahih3/...` are Iris paths.
- Provider/model + API keys come from `.env` (NGC key, `WANDB_API_KEY`, serving
  `SCAGENT_BASE_URL`, etc.). Don't commit secrets.

## Where things live in this repo

- `nat_integration/` — the NAT wrapper, evaluators, configs, aggregation.
- `experiments/` — serving + tool-call + vision benchmarks.
- `start_vllm.sh` / `start_nim.sh` / `start_trtllm.sh` / `start_vlm.sh` — backend
  launchers (encode the container/parser/GPU recipes).
- `docs/serving_findings.md` — the serving write-up.
- `scagent/agent/tracing.py` — OTel spans + step/trajectory logging (gated).
- `scagent/core/` — the compute chain with RAPIDS GPU routing (`SCAGENT_GPU=1`).

## Suggested first moves on the Spark

1. Clone this repo; read `README.md`, `AGENTS.md`, and this file.
2. Stand up the scagent env + a serving backend (vLLM or a NIM) natively.
3. **Re-run the OpenShell spike** — this is the work the Spark uniquely unblocks;
   evaluate it as a replacement for scagent's in-process `run_code` sandbox.
4. Re-stage a small LuCA subset (LUNG_T06) + the scorer to reproduce the eval
   loop, then compare Spark serving numbers (native NVFP4) against the Iris runs.
