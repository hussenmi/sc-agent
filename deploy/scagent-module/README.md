# scagent as an Lmod module — reference

How `module load scagent` is built and operated on the Iris HPC. Deployed +
verified on a GPU node (iscb007, CUDA 13.1) on 2026-06-21. Status: testing phase
for select members (not announced lab-wide yet).

```bash
module load scagent      # GPU node = accelerated; CPU node = scanpy fallback
scagent start            # interactive session; also: analyze | inspect | qc | chat | login
module unload scagent
```

Nothing is installed per-user — the whole environment lives once in a shared
tree, and the module just puts it on `PATH` and sets a few env vars.

**GPU is optional.** scagent defaults to CPU (scanpy); GPU only accelerates the
heavy steps (PCA/neighbors/UMAP/Leiden). The module detects a CUDA 12/13 GPU at
load: if present it sets `SCAGENT_GPU=1` and preloads the CUDA libs; if not, it
loads anyway in CPU mode (prints "CPU only"). The load message tells you which.

---

## 1. Where everything lives

Shared module tree (owned by group `grp_hpc_collab002`, on `$MODULEPATH` cluster-wide):

```
/usersoftware/collab002/sail/tools/Modules/
├── modulefiles/scagent/
│   ├── 0.1.0            # the TCL modulefile (what `module load` runs)
│   └── .version         # "0.1.0" — the default version
└── lib/scagent-0.1.0/   # the frozen ~15G env (relocated conda env)
    ├── bin/scagent      # console script (shebang -> this env's python)
    ├── lib/python3.14/site-packages/scagent/   # the frozen package
    └── .env             # the shared lab config (640, group-readable)
```

Source of truth for the deploy artifacts (this directory, version-controlled):
`deploy/scagent-module/` — installer, update script, modulefile, `.env.template`.

---

## 2. How it was built (brief)

The Iris module system is **Lmod 8.7** with classic-TCL modulefiles. The house
pattern (see `…/Modules/modulefiles/AGENTS.md`): install software in
`lib/<tool>-<ver>/`, write a modulefile that puts it on `PATH`, set `.version`.
`cellranger` is a one-line `PATH` prepend; `segger` is the richer GPU precedent.

scagent ships the **GPU/RAPIDS conda env** (the one `setup_gpu.sh` uses), so the
build is more involved than a single binary:

1. **Relocate the env with conda-pack.** The env is pip-built (everything
   `=pypi_0`), so `conda create --clone` can't reproduce it — it fails outright.
   `conda-pack` tars the whole prefix and `conda-unpack` rewrites the baked-in
   absolute paths at the new location. Result: a frozen ~15G copy.
2. **Freeze scagent itself.** The dev env has scagent installed *editable*
   (tracks the repo). The installer replaces that with a **non-editable wheel**
   built from the repo, so `module load` is a stable snapshot that does NOT shift
   when you edit your working tree.
3. **The modulefile** (`deploy/scagent-module/modulefile/0.1.0`) reproduces
   `setup_gpu.sh` in TCL:
   - a `nvidia-smi` **GPU detection** (sets `SCAGENT_GPU=1` + preloads
     `libnvJitLink` only when a CUDA 12/13 GPU is present; loads in CPU mode
     otherwise — does NOT block the load);
   - the env's `activate.d` deltas (GDAL/PROJ/glib/XML) **baked in as `setenv`**
     — this Lmod's TCL has no `source-sh`, so they're set directly;
   - the two `LD_PRELOAD` libs (`libstdc++`, `libnvJitLink`) in the right order;
   - `SCAGENT_HOME` (→ install dir, so the shared `.env` is found), `SCAGENT_GPU=1`,
     `SCIMILARITY_MODEL_PATH`, `SCAGENT_CELLBENDER`;
   - `prepend-path PATH <prefix>/bin`.
4. **Shared `.env`** placed at the install root, `chmod 640` group-readable.

### Config / API keys — how the `.env` is found

scagent's `_load_dotenv()` (`scagent/agent/agent.py`) loads, lowest→highest
precedence: the package/repo root, then `./.env` (cwd), then **`$SCAGENT_HOME/.env`
last so it wins**. The module sets `SCAGENT_HOME` to the install dir, so the
shared `.env` is authoritative — a stray `./.env` in someone's working dir can
add vars but cannot override it. **Editing the shared `.env` controls every user.**

---

## 3. Day-to-day operations

### Change config (key, model, base_url) — no rebuild
Edit the shared `.env` in place. The install dir is locked read-only, so:

```bash
DIR=/usersoftware/collab002/sail/tools/Modules/lib/scagent-0.1.0
chmod u+w "$DIR"
vi "$DIR/.env"            # e.g. point SCAGENT_BASE_URL at the running server
chmod u-w "$DIR"; chmod 640 "$DIR/.env"
```
Picked up on each user's **next `scagent` launch** (restart any open session).

Currently the lab defaults to the self-hosted endpoint
(`SCAGENT_PROVIDER=openai`, `SCAGENT_BASE_URL=http://iscp001:8004/v1`,
`SCAGENT_MODEL=Qwen3.6-27B`), copied from the personal `.env`. So the module
depends on that vLLM server being up; switch to `anthropic` + a key for a stable
default.

### Push working-tree CODE changes into the module (in place)
```bash
bash deploy/scagent-module/update_scagent_code.sh
```
Rebuilds the wheel from the repo, **verifies every tracked source file is in the
wheel** (guards the `.gitignore` trap below), reinstalls into the prefix, re-locks.
Overwrites `0.1.0` in place — ideal for the test phase. Testers get it next launch.

### New dependency or conda-env change
Not covered by the code update. Either `pip install` the dep into
`$PREFIX/bin/python`, or re-pack the whole env via `install_scagent_module.sh`.

### Cut a new, immutable version (for stable release later)
See **§3a Versioning** below.

---

## 3a. Versioning — what changes, what to specify

There are **two independent "versions"** and it helps to keep them straight:

| Version | Where it's set | What it controls |
|---|---|---|
| **Module version** | `VERSION=` in `install_scagent_module.sh`; mirrored by the install dir name `lib/scagent-<ver>/`, the modulefile path `modulefiles/scagent/<ver>`, and `.version` | What users type: `module load scagent/<ver>`. The `.version` file picks the default when they just type `module load scagent`. |
| **Package version** | `version = "x.y.z"` in `pyproject.toml` (`[project]`) | The wheel's own version (`scagent --version`, what pip records). Independent of the module version, though usually kept in sync. |

### What a version change physically looks like
A new module version is a **new directory tree alongside the old one** — nothing
is overwritten:
```
lib/scagent-0.1.0/            modulefiles/scagent/0.1.0
lib/scagent-0.2.0/      +     modulefiles/scagent/0.2.0
                             modulefiles/scagent/.version  -> "0.2.0"
```
Both stay loadable; `.version` only decides the default. Users pin explicitly with
`module load scagent/0.1.0`.

### To cut a new version
1. (Usually) bump `version` in `pyproject.toml` so `scagent --version` reflects it.
2. Set `VERSION=` to the new value in `install_scagent_module.sh`.
3. Re-run `bash install_scagent_module.sh` → builds `lib/scagent-<new>/`, writes
   `modulefiles/scagent/<new>`, and updates `.version` to the new default.
4. Set the lab key/config in the new `<prefix>/.env` (a fresh deploy starts from
   `.env.template`; or copy the old version's `.env`).

> Note: step 3 re-packs the **whole conda env** (~15G, minutes). Only do this when
> the environment or dependencies changed, or when you genuinely want a frozen,
> immutable snapshot users can pin.

### Which method for which change

| Kind of change | Method | Bump version? |
|---|---|---|
| API key / model / `base_url` | edit shared `.env` in place | no |
| scagent **Python code** (testing phase) | `update_scagent_code.sh` — overwrites `0.1.0` in place | no (stay on 0.1.0) |
| scagent code, **stable release** people pin | bump version → `install_scagent_module.sh` | yes |
| **New pip dependency** | `pip install` into `$PREFIX/bin/python` (quick), or re-pack via installer | only if releasing |
| **conda-env / RAPIDS changes** | re-pack via `install_scagent_module.sh` | yes (new env = new version) |

Rule of thumb: **overwrite `0.1.0` in place while iterating with testers**
(everyone always wants your latest); **bump to a new version once others depend on
a stable build** and you don't want to move it under them.

### First-time full deploy (already done; for reference / a fresh version)
```bash
bash deploy/scagent-module/install_scagent_module.sh --dry-run   # validate, write nothing
bash deploy/scagent-module/install_scagent_module.sh             # conda-pack + freeze + install
# then set the lab key in <prefix>/.env
```
Run on a GPU node, as a `grp_hpc_collab002` member, with PyPI access.

---

## 4. Gotchas (so they don't bite again)

- **`source-sh` is unavailable in this Lmod's TCL** → modulefile errored with
  `invalid command name "source-sh"`. We bake the `activate.d` env vars as `setenv`
  instead. (A `.lua` modulefile with `source_sh` would also work.)
- **`.gitignore` packaging trap.** The pattern `run*` matched the tracked source
  file `scagent/agent/run_manager.py`; hatchling honors `.gitignore` via pathspec
  and **silently dropped it from every wheel** (git itself didn't, since it's
  tracked — only `git check-ignore --no-index` flags it). Broke `scagent --help`
  with `ModuleNotFoundError: scagent.agent.run_manager`. Fixed: `run*` → `run*/`.
  `update_scagent_code.sh` now hard-checks wheel completeness to catch a recurrence.
- **conda `--clone` can't relocate a pip-built env** — use conda-pack.
- **The wheel build needs PyPI access** (the hatchling backend isn't in the env);
  installing the built wheel does not.

---

## 5. Follow-ups (deferred)

1. **`SCIMILARITY_MODEL_PATH` (41G) + `SCAGENT_CELLBENDER` still point at personal
   `ibrahih3` dirs.** `/data1/peerd` is full; relocate to shared group storage and
   update the two `setenv` lines in the modulefile.
2. **Migrate the env to pixi** (house convention, like `segger`; lockfile-
   reproducible) instead of the opaque conda-pack blob. Seed `pixi.toml` with exact
   pins from the current env to de-risk the resolve.
3. **Stable default provider:** decide whether the lab `.env` should point at the
   self-hosted endpoint (current) or Anthropic (robust) before wider rollout.
```
