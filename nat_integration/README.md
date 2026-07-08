# scagent_nat — NeMo Agent Toolkit integration

Wraps the scagent scRNA-seq agent in NVIDIA's NeMo Agent Toolkit (NAT) for
**per-step observability** (token/latency tracing) and an **annotation
accuracy + consistency eval**, so we can measure not just "which serving
backend is faster" (vLLM vs NIM vs Nemotron) but "which is faster *and still
scientifically correct*".

## Why a separate venv

NAT needs Python ≥3.11; scagent runs on 3.10. So this package installs into the
dedicated **`nvidia-nat`** venv and invokes scagent's **CLI as a subprocess** —
the two dependency stacks never mix. The backend is chosen by config
(`base_url`/`model`), and scagent reads it from `SCAGENT_BASE_URL` env vars set
per run.

## Install (into the nvidia-nat venv)

```bash
NATPY=/usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/python
uv pip install --python "$NATPY" -e /data1/peerd/ibrahih3/cs_agent/nat_integration
```

## Run the eval

```bash
/usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/nat eval \
  --config_file /data1/peerd/ibrahih3/cs_agent/nat_integration/configs/annotation_eval.yml
```

Requires a serving backend reachable at the config's `base_url` (the Qwen3.6-27B
vLLM server on `:8000`, by default).

## Components

- `scagent_analyze` (workflow function) — runs scagent end-to-end on
  `{instruction, data_path}`, returns `{run_dir, annotated_h5ad, ...}`.
- `annotation_ari` (evaluator) — Adjusted Rand Index between scagent's `leiden`
  clustering and the ground-truth labels (`<h5ad>::<obs_col>`), computed on
  shared barcodes. Label-free, zero curation. Lineage-level accuracy (with a
  curated label map) is the planned next evaluator.

## Eval data

`test_data/GSE155249_*.h5ad` carry the published authors' cell-type labels in
`obs['Cluster']` (28 BAL cell types) — independent ground truth. Do **not**
score against scagent's own prior `run_*/` outputs (circular).
