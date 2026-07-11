# Environments — starting a session on Iris and the Spark

scagent runs on two very different hosts. This page is the authoritative
reference for **how to start a session on each**, what each environment
provides, and how to confirm you're actually on the GPU. For the terse version,
see the table in `AGENTS.md`.

| Host | Arch / GPU | Env manager | Activate |
|---|---|---|---|
| **Iris HPC** | x86_64, Hopper (multi-GPU) | uv venv (CPU) + conda (GPU) | `source setup.sh` / `source setup_gpu.sh` |
| **DGX Spark** | aarch64, GB10 Blackwell (1 GPU, 128 GB unified) | pixi | `pixi shell -e gpu` |

The one thing scagent's code actually depends on is a small **env-var contract** —
`SCAGENT_GPU`, `SCAGENT_CELLBENDER`, `SCIMILARITY_MODEL_PATH` — plus a runtime GPU
probe. Both hosts satisfy the same contract with different machinery, so no
scagent code changes are needed to move between them; only the env differs.

---

## Iris — CPU / dev  (`source setup.sh`)

```bash
cd /data1/peerd/ibrahih3/cs_agent
source setup.sh            # add --local for ./.venv instead of /usersoftware
```

- uv-managed venv (Python 3.10) at `/usersoftware/peerd/$USER/envs/scagent`.
- Installs the package **editable** with all extras (`uv pip install -e ".[all]"`).
- Exports `SCAGENT_HOME`, `SCIMILARITY_MODEL_PATH`, and (if present) `SCAGENT_CELLBENDER`.
- **No `SCAGENT_GPU`** → the compute chain runs on scanpy (CPU). This is the env
  for tests, lint, type-checking, and quick iteration.
- **Install with `uv pip install`, never bare `pip`** (the venv has no `pip`; a
  bare `pip` hits the wrong miniconda).

## Iris — GPU  (`source setup_gpu.sh`)

```bash
cd /data1/peerd/ibrahih3/cs_agent
source setup_gpu.sh
```

- Activates the conda env `scagent_rapids` (Python 3.14, RAPIDS 26.04) at
  `/usersoftware/peerd/ibrahih3/envs/scagent_rapids`. scagent is already installed
  editable into it — this script **only activates + exports, never installs**.
- Sets `SCAGENT_GPU=1` (override with `export SCAGENT_GPU=0` before sourcing to
  force CPU while keeping the env).
- **`LD_PRELOAD` surgery is required here:** it preloads the conda
  `libstdc++.so.6` (numpy needs a newer `GLIBCXX`) and `libnvJitLink.so.13`
  (cuml's libcuvs needs a symbol the pip-torch copy lacks). Without it, imports
  fail. Do **not** add `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH`.
- Exports `SCIMILARITY_MODEL_PATH` (+ `_MOUSE`) and `SCAGENT_CELLBENDER` (a
  cellbender binary in a separate shared pixi env — see "Heavy tools" below).

## Spark — GPU  (`pixi shell -e gpu`)

```bash
cd /home/hussen/projects/sc-agent
pixi shell -e gpu                 # interactive session
# or, non-interactively:
pixi run -e gpu scagent analyze --data path/to.h5ad
```

- One `pixi.toml` defines two environments: the implicit **`default`**
  (CPU/agent + science stack) and **`gpu`** (`default` + the `gpu` feature).
  `pixi shell` alone is CPU-only; **`-e gpu` is what enables compute.**
- The `gpu` env adds torch (cu130) + RAPIDS (cuml/cugraph/cuvs) +
  `rapids-singlecell` + `scvi-tools`; CellTypist is present. It sets
  `SCAGENT_GPU=1`, `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=12.1a` (GB10 = sm_121),
  and `LD_PRELOAD` (see next).
- **`LD_PRELOAD` surgery is required here too** — different libs than Iris:
  ```toml
  # in [feature.gpu.activation.env]
  LD_PRELOAD = "$CONDA_PREFIX/lib/libcublas.so.13:$CONDA_PREFIX/lib/libcublasLt.so.13"
  ```
  It forces the conda RAPIDS cuBLAS/cuBLASLt to win over the copy bundled in
  pip-torch's `nvidia-cu13` wheels. Without it, `rapids_singlecell` fails to
  import (`undefined symbol cublasLt…MatmulAlgoGetHeuristicForStream`) and scagent
  **silently falls back to `scanpy_cpu`**. As on Iris, do **not** force system
  CUDA onto `LD_LIBRARY_PATH`.
- Deps are pixi-managed: edit `pixi.toml`, then `pixi install`. (`pixi.toml`'s
  own comments are the running log of ARM-specific build lessons.)

---

## The GPU backend — how it's chosen, and how to verify

scagent never hardcodes the device. `scagent/core/gpu.py` decides at runtime:

1. `SCAGENT_GPU` must be truthy (set by both GPU envs above).
2. `gpu_capability_report()` spawns a **throwaway subprocess** that imports
   `cupy` + `rapids_singlecell` and counts CUDA devices — so probing never leaves
   a CUDA context in the main process. It returns
   `{enabled, gpu, n_devices, rsc_version, reason}`.
3. If any of that fails, the heavy steps (PCA / neighbors / UMAP / Leiden) fall
   back transparently to scanpy on CPU.

**Always confirm before a long run** — the fallback is silent except for
`"backend": "scanpy_cpu"` in each step of `manifest.json`:

```bash
python -c "from scagent.core.gpu import gpu_capability_report as g; print(g())"
# GPU-ready looks like: {'enabled': True, 'gpu': True, 'n_devices': N, 'rsc_version': '...', 'reason': ''}
# If 'gpu': False, the 'reason' field says why (env not set, import error, no device).
```

`scagent start` / `scagent analyze` also print a GPU line at startup summarizing
the resolved device config and visible GPUs.

---

## Heavy tools — three different execution models

These are worth knowing because each behaves differently across hosts.

| Tool | How it runs | Iris | Spark |
|---|---|---|---|
| **scVI** | subprocess in the **same env** (`sys.executable -m scagent.batch._scvi_worker`), purely to isolate the CUDA context | ✅ | ✅ (scvi-tools in the gpu env) |
| **CellBender** | subprocess to a **separate env** via the `SCAGENT_CELLBENDER` binary path | ✅ (shared sail pixi env) | ⚠️ not set up yet (optional; only for ambient-RNA removal) |
| **scimilarity** | in-process import; model at `SCIMILARITY_MODEL_PATH` | ✅ | ❌ ARM-blocked (`tiledb-vector-search` has no aarch64 build) |
| **CellTypist** | in-process import | ✅ | ✅ |

Notes:

- **scVI** needs `scvi-tools` in the active env; the subprocess re-execs the same
  interpreter, so it works wherever the env has it (both GPU envs do).
- **CellBender** is the only true cross-env subprocess — it lives in its own env
  because its deps conflict with the RAPIDS/scanpy stack. To enable it on the
  Spark, add a dedicated pixi env and export `SCAGENT_CELLBENDER` to its
  `cellbender` binary (mirror the Iris shared env).
- **scimilarity** is unavailable on the Spark; annotation there leans on
  CellTypist + PanglaoDB/biocontext markers. Everything still runs, just with one
  fewer evidence source than Iris.

---

## Serving the model

scagent talks to an OpenAI-compatible endpoint set in `.env`
(`SCAGENT_BASE_URL`, `SCAGENT_MODEL`, plus optional `SCAGENT_VISION_*`). The
serving stack differs per host:

- **Iris:** `start_vllm.sh` (Singularity), plus `start_nim.sh` / `start_trtllm.sh`
  / `start_vlm.sh` for the NIM / TensorRT-LLM / vision variants.
- **Spark:** `start_vllm_spark.sh` (Docker, not Singularity), serving on
  `localhost`. `.env` points `SCAGENT_BASE_URL` at that local server.

Point `.env`'s `SCAGENT_BASE_URL` at whichever server is live (nodes and ports
change frequently — probe before a run). Current serving state and findings live
in `docs/serving_findings.md` and `docs/nvidia_collab.md`.

---

## Host-specific gotchas

- **Iris nodes change** (login/compute names shift between sessions); a service
  reachable on one node may not be on another. GPU runs use `source setup_gpu.sh`,
  CPU/dev + tests use `source setup.sh`.
- **Spark is aarch64** — some x86 wheels/images don't exist (NIM images, a few
  scientific wheels). Pull the compiled scientific stack from conda-forge; see the
  `pixi.toml` comments for the specific ARM lessons.
- **Both GPU envs need `LD_PRELOAD`** (different libs) and both break if you force
  system CUDA onto `LD_LIBRARY_PATH`.
- **The tell for a broken GPU env is `backend: scanpy_cpu` in the manifest.** Run
  the `gpu_capability_report()` check above whenever a "GPU" run looks CPU-slow.
