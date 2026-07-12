# AGENTS.md

Operational guide for AI coding agents working in this repo. For architecture,
usage, and the analysis workflow, read `README.md` — this file only covers what
isn't obvious from the code.

## Environment — do this first, every session

This repo runs on two hosts — the **Iris HPC** (x86_64, Hopper) and the **DGX
Spark** (aarch64/GB10). Pick the environment for what you're doing and activate
it **before** any Python, tests, or `scagent` command:

| Host / mode | Activate (from the repo root) | Gives you |
|---|---|---|
| **Iris — CPU / dev** | `source setup.sh` | uv venv (Py 3.10), editable install, tests + lint. No GPU. |
| **Iris — GPU** | `source setup_gpu.sh` | conda `scagent_rapids` (Py 3.14, RAPIDS), `SCAGENT_GPU=1`; scVI + CellBender + scimilarity. |
| **Spark — GPU** | `pixi shell -e gpu` | pixi gpu env (RAPIDS + scVI + CellTypist + scimilarity), `SCAGENT_GPU=1`. |

`source setup.sh` accepts `--local` (venv in `./.venv` instead of
`/usersoftware`). On the Spark, `pixi shell` **alone** is CPU/agent-only — use
`-e gpu` for compute, or `pixi run -e gpu scagent …` non-interactively.

- **Confirm the backend before a real run.** A silent CPU fallback is easy to
  miss — it only shows as `"backend": "scanpy_cpu"` in `manifest.json`. Check:
  ```bash
  python -c "from scagent.core.gpu import gpu_capability_report as g; print(g())"
  # want {'gpu': True, 'n_devices': N}; if False, read the 'reason' field
  ```
- **On Iris, install with `uv pip install`, never bare `pip`** — the venv has no
  `pip`, and a bare `pip` resolves to the wrong miniconda. On the **Spark**, deps
  are pixi-managed: edit `pixi.toml`, then `pixi install`.
- Provider/model and API keys come from `.env` (see README). Tests don't need it.

See **`docs/environments.md`** for what each env provides, per-host tool
availability (scVI / CellBender / scimilarity), the GPU backend + verification,
serving, and host-specific gotchas.

## Tests, lint, types

Run from the repo root after sourcing `setup.sh`:

```bash
python -m pytest tests/              # full suite; coverage is auto-added via pyproject addopts
python -m pytest tests/foo_test.py   # a single file
ruff check scagent/                  # lint (line-length 100, E/F/W/I/UP/B)
mypy scagent/                        # type check
```

- Test files use the `*_test.py` suffix (not `test_*.py`); `testpaths = ["tests"]`.
- `addopts = -v --cov=scagent` is on by default, so don't pass `-p no:cov`
  (it errors). Coverage output is expected noise, not a failure.

## Repo-specific gotchas

- **Annotation / validation logic spans four files — change them together.** The
  PanglaoDB/Cytopus enforcement and the `prepare_annotation → stage_annotation_evidence
  → finalize_annotation` consensus path are split across
  `scagent/agent/tools.py` (the evidence validator + tool handlers),
  `scagent/agent/prompts.py` (the workflow rules the model follows),
  `scagent/agent/agent.py` (the save/finalize guard in the run loop), and
  `scagent/agent/world_state.py` (`annotation_validation` state machine). A change
  in one usually needs matching edits in the others.
- **`run_code` is sandboxed.** Generated analysis code cannot `import os`, `sys`,
  `subprocess`, or `shutil`; it uses injected helpers (`ensure_dir`, `write_report`,
  `register_artifact`, `output_dir`, `adata`, `sc`, `np`, `pd`, `plt`). Keep this in
  mind when editing tool prompts or the sandbox.
- **Don't hand-assign cell-type labels.** Annotation must go through the consensus
  tools, not `adata.obs['cell_type'] = ...` in `run_code`.
- Each agent run writes a self-contained `run_YYYY_MM_DD_HHMMSS/` directory;
  `manifest.json` is the provenance record. Don't edit prior run dirs.

## Conventions

- Match the surrounding style; keep changes minimal and focused. Comment only to
  explain a constraint, not to narrate.
- Add or update a `*_test.py` when changing behavior; run the suite before finishing.
- **Git:** branch off `main`; commit or push only when the user asks.
