# scagent Serving & Model Findings

Running log of self-hosted serving work for scagent (frontier text reasoners +
vision sidecar) on the Iris H200 nodes. Captures configs, measured performance,
and the bugs/fixes found along the way. Source for the NVIDIA collaboration write-up.

_Last updated: 2026-06-22._

## Hardware

| Node | GPUs | Interconnect | Notes |
|---|---|---|---|
| iscp001 | 8× H200 (143 GB) | NVSwitch (full) | main serving box; TP up to 8 |
| iscn008 | 3× H200 NVL | NVLink-bridge / PCIe | hosts vision describer + a Qwen3.6 |
| iscn011 | 3× H200 NVL | NVLink-bridge / PCIe | spare |

Shared FS: `/data1/peerd/ibrahih3` (HF cache `…/hf`, SIFs, repo). ~18 TB free.

## Models & measured performance

Single-stream decode, TTFT-isolated (warm), via vLLM unless noted.

| Model | Quant / engine | Node / config | Decode | TTFT | Notes |
|---|---|---|---|---|---|
| Nemotron-3-Ultra (550B/55B MoE, hybrid Mamba) | NVFP4 / vLLM | iscp001 TP=8 | **~264 tok/s** | ~0.1 s | MTP k=5, accept ~4.1 tok/step (~62%); tools ✓ |
| Qwen3.6-27B | (start_vllm.sh) / vLLM | iscn008:8000 | **~84 tok/s** | ~0.13 s | dense 27B |
| GLM-5.2 (744B/40B MoE) | UD-Q4_K_XL GGUF / llama.cpp | iscp001 4-GPU layer-split | ~47–59 tok/s | — | **no MoE tensor-parallel in llama.cpp** → ~1 GPU's worth |
| GLM-5.2 | FP8 / vLLM | iscp001 TP=8 | _pending_ | _pending_ | 704 GB; full 1M ctx; glm47/glm45 parsers |

Headline: real MoE **tensor parallelism (vLLM TP=8) is ~4–5× the llama.cpp
layer-split** for the same class of model — llama.cpp has no TP for MoE, so extra
GPUs only add KV room, not speed.

## Launch scripts (all in repo root)

- `start_vllm.sh` — agentic main models via vLLM. Table-driven; per-model `MODEL_EXTRA`
  + `GPU_IDS` pinning. Knows Nemotron-3-Ultra (NVFP4, 0.22 SIF) and GLM-5.2-FP8
  (0.23 SIF). Hopper-NVSwitch path adds FP8 KV + DeepGEMM/SymmMem fusions.
- `start_llamacpp.sh` + `download_gguf.sh` — GLM-5.2 via llama.cpp (eval only; layer-split).
- `start_vlm.sh` — vision **describer** via vLLM (minimal: no tool/reasoning/spec machinery).
- `start_nim.sh` — generic NIM launcher (TRT-LLM/vLLM backend, auto profile select).
- `smoke_test_tools.sh`, `experiments/vision_ab.py` — tool-call + vision smoke/A-B harnesses.

SIFs: `vllm-openai_v0.22.0.sif` (Nemotron), `vllm-openai_v0.23.0.sif` (GLM-5.2),
`llamacpp-server-cuda.sif`, `nim-nemotron-nano-12b-v2-vl.sif`.

## Vision sidecar (for text-only main models)

Text-only reasoners (Nemotron, GLM-5.2) can't see figures. A VLM sidecar describes
them; the reasoner reasons over the text + can interrogate via `describe_image`.

- **Nemotron-Nano-12B-v2-VL** (BF16, public, 24.6 GB) served via `start_vlm.sh` on
  iscn008:8003. Works; **hallucinates specifics on dense UMAPs** (invents cluster
  sizes, chromosome regions, axis ranges, library versions).
- **Qwen3.6-27B** as describer is **more faithful** (A/B winner): stays at the right
  altitude, hedges, doesn't invent. Recommended sidecar; already on hand.
- Real flow passes `world_state` (true cluster counts etc.) → grounds the describer;
  the A/B was blind (no world_state) and still favored Qwen3.6.
- NIM route for the VLM **failed** (see bugs) → used plain vLLM.

`.env`: `SCAGENT_VISION_MODEL`, `SCAGENT_VISION_BASE_URL`, `SCAGENT_VISION_API_KEY=dummy`.

## Bugs found & fixes (all in working tree, uncommitted)

1. **`_supports_vision` mis-classified text-only models** (returned True for all but
   DeepSeek) → sent `image_url` to text-only servers → hard error. Fixed: allowlist of
   known-multimodal families + authoritative `SCAGENT_MAIN_HAS_VISION` override;
   default text-only (route to sidecar) when unknown. + tests.
2. **Context detection was vLLM-only** (`max_model_len`); llama.cpp advertises
   `meta.n_ctx`. Made `_resolve_context_limit` backend-agnostic (helpers
   `_model_get`/`_coerce_positive_int`/`_server_context_limit`, 26 tests).
3. **`start_vllm.sh` couldn't pin GPUs or carry per-model flags** → added `GPU_IDS`
   (env, conditional `export SINGULARITYENV_CUDA_VISIBLE_DEVICES` — a parameter-expanded
   `VAR=val` is NOT an assignment in bash) and a per-model `MODEL_EXTRA`/`SKIP_THINKING_KWARG`
   case mechanism.
4. **`ctx_k` units**: table is in 1024s. 262144 tokens = `ctx_k=256`, NOT 262
   (262×1024=268288 > model max → vLLM rejects).
5. **NIM VLM crashes under Singularity → SOLVED 2026-06-23.** `nemotron-nano-12b-v2-vl`
   NIM selected the vLLM backend, then died at boot: its SDK builds the vLLM subprocess
   via `shell=True` and mangled the JSON `--media-io-kwargs` into literal `{\"video\":…}`
   → `json.loads` rejected → exit. **Fix:** `singularity run --no-eval` (OCI-quoting,
   no shell re-eval) + `SINGULARITYENV_NIM_MEDIA_IO_KWARGS='{}'` + `SINGULARITYENV_TMPDIR=
   /opt/nim/.cache/tmp` + `--writable-tmpfs` — the live backend args then show a clean
   `"media_io_kwargs": {}`. Verified end-to-end on iscn008 (read a UMAP's title/axes/
   cluster-count correctly in 2.2 s). **GOTCHA:** the VLM NIM ignores `NIM_HTTP_API_PORT`
   and serves on **:8000**. All baked into `start_nim.sh`.
6. **`start_vlm.sh` host-env leak**: container used host `~/.local` transformers (missing
   `PYTHONNOUSERSITE=1`) and inherited host `HF_HOME=/data1/.../hf` (read-only auto-mount)
   → `--trust-remote-code` modules write failed. Fixed: pin `PYTHONNOUSERSITE`, `HF_HOME`,
   `HF_MODULES_CACHE` to the writable `/hf_cache` bind.
7. **Reasoning narration not shown** for vLLM reasoning parsers: scagent read only
   `reasoning_content`, but `nemotron_v3`/`glm45` emit narration under `reasoning` while
   `content` is empty on tool-calling turns → "only tool calls visible." Fixed
   `_reasoning` extraction to also read `reasoning`. (Restart scagent for it to apply.)
8. **Startup banner** (`cli.py`) now shows the **Vision** line (native / sidecar→model@url /
   none-warning) and the **main model's endpoint** for self-hosted servers.

## Open items / TODO

- Serve & benchmark **GLM-5.2-FP8 TP=8** (downloaded; launch via `start_vllm.sh`).
- Re-run vision A/B **with `world_state`** (apples-to-apples) — does context close
  Nemotron-VL's hallucination gap vs Qwen3.6?
- File the NIM-VLM-under-Singularity bug report (#5) with NVIDIA.
- Full end-to-end annotation run: Nemotron/GLM main + Qwen3.6 vision sidecar.
- Commit this session's working-tree changes (scripts, agent.py, cli.py, tests).
- Claude's SSH key access to compute nodes is intermittent — launches sometimes need
  the user; HTTP to running endpoints works regardless.

## Update 2026-06-22 — full 8×H200 NVSwitch node (iscp001)

Measured single-stream decode (TTFT-isolated, warm) on the full node:

| Model | Engine / config | Decode | Notes |
|---|---|---|---|
| Nemotron-3-Ultra NVFP4 | vLLM TP=8 + MTP + graphs | **~264 tok/s** | TTFT ~0.1s; MTP accept ~4.1 tok/step; all 8 GPUs; clean |
| Qwen3.6-27B | vLLM (iscn008) | ~84 tok/s | dense 27B |
| GLM-5.2 GGUF UD-Q4_K_XL | llama.cpp `-sm layer`, 4–8 GPU | ~50 tok/s | layer-split = ~1 GPU's compute |
| GLM-5.2-FP8 | vLLM TP=8 + EP, **eager** | ~21 tok/s | graphs hang → eager-only → slow |

**GLM-5.2 serving matrix (the hard part).** Its `glm_moe_dsa` (DeepSeek-sparse-attn MoE)
is under-supported in both engines on 8×H200:
- llama.cpp `-sm layer`: ✅ works ~50 tok/s (only ~1 GPU's compute; no MoE TP).
- llama.cpp `-sm tensor` (real TP): ❌ **aborts** — `ggml_abort` in
  `llama_meta_device_get_split_state`; build 9737 tensor-split doesn't support this MoE
  (works for dense only). The new `-sm tensor` exists but MoE is the same wall as old `-sm row`.
- vLLM TP=8 WITHOUT `--enable-expert-parallel`: ❌ collective **deadlock** at warmup
  (ranks spin at 100%, others 0% — the "only N of 8 GPUs" hang). **EP is REQUIRED for GLM MoE.**
- vLLM TP=8 + EP + `--disable-custom-all-reduce`, **eager**: ✅ works but ~21 tok/s.
- vLLM TP=8 + EP, **CUDA graphs**: ❌ graph-capture **hangs** (vLLM 0.23 + glm_moe_dsa bug).
  This is the blocker on fast GLM-5.2; eager is the only working vLLM mode today.

start_vllm.sh GLM entry now carries `--enable-expert-parallel --disable-custom-all-reduce
--enforce-eager` (the only working combo). Remove `--enforce-eager` once a newer vLLM
fixes glm_moe_dsa graph capture — that should restore ~200+ tok/s.

**Key takeaways:**
- **EP is mandatory for multi-GPU GLM MoE in vLLM** — this is the root cause of the
  recurring "only N of 8 GPUs" hang (NOT a GPU-visibility problem; all 8 always loaded).
- GLM-5.2 fast serving is currently **upstream-blocked**: needs newer vLLM (graph fix)
  or newer llama.cpp (MoE tensor support). Best working today = llama.cpp layer ~50 tok/s.
- Nemotron-3-Ultra is the clear fast main model (264 tok/s, no issues).
- llama.cpp now has `-sm tensor`/graph (real TP, splits weights+KV) — dense-only for now;
  recheck MoE support in newer builds (would make GGUF a fast path).
- Untried cheap option for fast GLM on vLLM: force `cudagraph_mode=PIECEWISE` (skip the
  FULL-graph capture that's hanging) before resorting to a vLLM version bump.

## Update 2026-06-23 — VLM NIM works; full Nemotron + NIM pipeline, NAT-profiled

**The NIM story is now real and running.** The realistic architecture for this
hardware (NVFP4 is Blackwell-only in the Nemotron-3-Ultra NIM, so NIM-on-H200
would force a 1.1 TB BF16 download that barely fits — see below) is:

| Layer | Serving | Where |
|---|---|---|
| Reasoning LLM | Nemotron-3-Ultra **NVFP4 via vLLM** (~264 tok/s) | iscp001 TP=8 |
| Vision | **NVIDIA NIM** `nemotron-nano-12b-v2-vl` (NIM-packaged vLLM, FP8) | iscn008:8000 |
| Orchestration / eval | scagent + **NeMo Agent Toolkit (NAT)** → Weave | iscb015 |

- **VLM NIM under Singularity: solved** (bug #5 above). Boots, serves OpenAI +
  Anthropic endpoints, verified on real figures.
- **Nemotron-3-Ultra NIM on H200 = BF16 only** (NVFP4 profile is Blackwell
  B200/GB200/B300; H200 profile is `bf16 tp8 pp1`, ~1.1 TB weights on 1128 GB →
  marginal KV). So keep Nemotron on **vLLM-NVFP4** (faster, 328 GB); the NIM is
  the **VLM** leg. The VLM NIM is itself vLLM-backed → "NIM vs vLLM" here is
  *productization over the same engine*, not a kernel difference.
- **Full end-to-end run (single-stream), profiled by NAT** — LuCA `LUNG_T06`,
  complete QC→clustering→annotation in **1147 s**; 53 Nemotron calls, ~20 VLM-NIM
  calls; LLM p95 **28.4 s**. **Headline: the agent is prefill-bound, not
  decode-bound** — avg prompt **~147K tokens/call**, so per-turn latency is
  dominated by prefilling context, *not* generation. The serving metric that
  matters for this agent is **long-context prefill + prefix-cache hit rate**, not
  decode tok/s. Top bottleneck is the **`run_scimilarity` tool (69 s)**, not the
  LLM (consistent with the Qwen sweep's 73 s). Full numbers + cross-model table:
  **`nat_integration/RESULTS_concurrency_profiling.md`** (Update 2026-06-23).
- Reliability flag: an EMERGENCY context-compact fired at 263K (> Nemotron's 262K
  native cap) during annotation — that phase saturates a 256K window.
