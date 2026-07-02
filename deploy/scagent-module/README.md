# scagent as an Lmod module — reference

How `module load scagent` is built and operated on the Iris HPC. Status: testing
phase for select members (not announced lab-wide yet).

**The live module is now the pixi build** (lockfile-reproducible; deployed +
verified on a GPU node with 8x H200 / CUDA 13.1 on 2026-06-21). The original
conda-pack build is the predecessor — its scripts (`install_scagent_module.sh`,
`update_scagent_code.sh`) are kept for reference, but the current path is
`install_pixi_module.sh` / `update_pixi_code.sh`. Differences called out inline.

```bash
module load scagent      # GPU node = accelerated; CPU node = scanpy fallback
scagent start            # interactive session; also: analyze | inspect | qc | chat | login
module unload scagent
```

Nothing is installed per-user — the whole environment lives once in a shared
tree, and the module just puts it on `PATH` and sets a few env vars.

> **Prerequisite: the module is only discoverable once its modulefiles dir is on
> `$MODULEPATH`.** This is NOT wired up cluster-wide yet, so a fresh user gets
> `Lmod ... unknown module: "scagent"`. See **§0** for the one line each user (or,
> better, the admin) must add. `module --ignore_cache` does NOT fix this — it's a
> missing MODULEPATH, not a stale cache.

**GPU is optional.** scagent defaults to CPU (scanpy); GPU only accelerates the
heavy steps (PCA/neighbors/UMAP/Leiden). The module detects a CUDA 12/13 GPU at
load: if present it sets `SCAGENT_GPU=1` and preloads the CUDA libs; if not, it
loads anyway in CPU mode (prints "CPU only"). The load message tells you which.

---

## 0. Making `module load scagent` discoverable (MODULEPATH)

The environment is fully deployed and world-readable, but Lmod can't find it
unless the modulefiles dir is on the user's `$MODULEPATH`:

```
/usersoftware/collab002/sail/tools/Modules/modulefiles
```

There is currently **no system-wide file that adds this path** — not in
`/etc/profile.d/`, and Iris has no `/etc/lmod/modulepath.d/` mechanism. The only
reason it works for the initial testers is a hand-added line in their personal
`~/.bashrc`. So a new user who tries `module load scagent` gets:

```
Lmod has detected the following error: The following module(s) are unknown: "scagent"
```

Permissions are NOT the problem: the whole tree is world-traversable and the
modulefile is world-readable (`o+rX`), so once the path is on MODULEPATH it just
works for anyone. Two ways to get it there:

### Per-user (immediate, no privileges)
Each user adds one line to their `~/.bashrc` (or `~/.bash_profile`):

```bash
module use --append /usersoftware/collab002/sail/tools/Modules/modulefiles
```

Then start a new shell (or `source ~/.bashrc`). `module avail scagent` should now
list it, and `module load scagent` works.

### Lab-wide (recommended — requires HPC admin)
So nobody has to edit their bashrc, ask the HPC admin / the tree owner
(`krauset`) to drop a one-line snippet in `/etc/profile.d/`, which every login
shell sources automatically:

```sh
# /etc/profile.d/z-collab002-modules.sh
module use --append /usersoftware/collab002/sail/tools/Modules/modulefiles
```

This is the standard site pattern and the only route to a truly hands-off
`module load scagent` for the whole cluster. `/etc/profile.d/` is root-owned, so
it can't be done without admin — until it's in place, distribute the per-user
line above.

---

## 1. Where everything lives

Shared module tree (owned by group `grp_hpc_collab002`; note the modulefiles dir
must be on `$MODULEPATH` — see §0):

```
/usersoftware/collab002/sail/tools/Modules/
├── modulefiles/scagent/
│   ├── 0.1.0                  # the TCL modulefile (what `module load` runs; pixi build)
│   └── .version               # "0.1.0" — the default version
├── lib/scagent-0.1.0-pixi/    # CURRENT: the pixi project dir
│   ├── pixi.toml              # frozen deploy manifest (scagent non-editable)
│   ├── pixi.lock              # exact solved pins (the reproducible record)
│   ├── .env                   # the shared lab config (640, group-readable)
│   └── .pixi/envs/default/    # the ~17G solved env
│       ├── bin/scagent        #   console script (shebang -> this env's python)
│       └── lib/python3.14/site-packages/scagent/   # the frozen package
└── lib/scagent-0.1.0/         # PREDECESSOR: the conda-pack env (kept as fallback)
```

Source of truth for the deploy artifacts (this directory, version-controlled):
`deploy/scagent-module/` — `pixi/` (manifest + lock), `install_pixi_module.sh`,
`update_pixi_code.sh`, `_freeze_scagent.sh`, `modulefile/0.1.0-pixi`,
`.env.template`, plus the predecessor conda scripts + `modulefile/0.1.0`.

Note: the modulefile points `prefix` at `.pixi/envs/default` and sets
`SCAGENT_HOME` to the project dir (`scagent-0.1.0-pixi`) so the shared `.env`,
which sits next to the manifest rather than inside the pixi-managed env, is found.

---

## 2. How it was built (brief)

The Iris module system is **Lmod 8.7** with classic-TCL modulefiles. The house
pattern (see `…/Modules/modulefiles/AGENTS.md`): install software in
`lib/<tool>-<ver>/`, write a modulefile that puts it on `PATH`, set `.version`.
`cellranger` is a one-line `PATH` prepend; `segger` is the richer GPU precedent.

scagent ships the **GPU/RAPIDS env** (the same stack `setup_gpu.sh` uses), so the
build is more involved than a single binary.

### Current: pixi (`install_pixi_module.sh`)

1. **Build the env from a lockfile.** `pixi/pixi.toml` declares the RAPIDS 26.04 /
   CUDA 13 stack as conda deps (channels `rapidsai/nvidia/conda-forge/bioconda`,
   `channel-priority = "disabled"` — RAPIDS can't use strict) plus the PyPI side
   (`scagent[all]`, `rapids-singlecell-cu13`, `gdown`). `pixi install` solves it
   into `pixi.lock` (committed) and a ~17G env. **No conda-pack, no relocation** —
   the env is built in place, so its paths are already correct.
2. **Freeze scagent itself.** The deploy manifest installs scagent **non-editable**
   from the repo, and the installer then **overwrites it with a fresh wheel**
   (`_freeze_scagent.sh`) so the deployed code always matches the repo — pixi/uv
   caches the built scagent wheel by version and would otherwise serve a stale copy
   on a code change without a version bump (see Gotchas).
3. **The modulefile** (`modulefile/0.1.0-pixi`) is much simpler than the conda one:
   - `nvidia-smi` **GPU detection** (sets `SCAGENT_GPU=1` when a CUDA 12/13 GPU is
     present; loads in CPU mode otherwise — does NOT block the load);
   - the env's `activate.d` deltas (GDAL/PROJ/glib/XML) **baked in as `setenv`** —
     this Lmod's TCL has no `source-sh`;
   - **NO `LD_PRELOAD`** — pixi's real conda solve gives consistent libs, so the
     `libstdc++` / `libnvJitLink` preload hack the conda build needed is gone
     (verified: cupy/cuml/rapids_singlecell + torch + scvi-tools all run GPU ops in
     one process without it);
   - `SCAGENT_HOME` (→ the project dir, where the shared `.env` lives),
     `SCAGENT_GPU=1`, `SCIMILARITY_MODEL_PATH`, `SCAGENT_CELLBENDER`;
   - `prepend-path PATH <env>/bin`.
4. **Shared `.env`** placed in the project dir, `chmod 640` group-readable.

### Predecessor: conda-pack (`install_scagent_module.sh`)

The first build relocated the pip-built `scagent_rapids` conda env with
**conda-pack** + `conda-unpack` (`conda create --clone` fails on an all-`pypi_0`
env), froze scagent via a non-editable wheel, and used `modulefile/0.1.0` with the
two `LD_PRELOAD` libs. Kept as a fallback while the pixi build is in testing.

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
DIR=/usersoftware/collab002/sail/tools/Modules/lib/scagent-0.1.0-pixi
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
bash deploy/scagent-module/update_pixi_code.sh
```
Rebuilds the wheel from the repo, **verifies every tracked source file is in the
wheel** (guards the `.gitignore` trap below), force-reinstalls scagent into the
pixi env, re-locks permissions. Overwrites `0.1.0` in place — ideal for the test
phase. Testers get it next launch. (Force-reinstall is required: `pixi install`
alone would NOT pick up a same-version code change — see Gotchas.)

### New dependency or env change
Not covered by the code update. Edit `pixi/pixi.toml` (add the dep), run
`pixi install` to refresh `pixi/pixi.lock`, commit the lock, then re-run
`install_pixi_module.sh` to rebuild the shared env from it.

### Cut a new, immutable version (for stable release later)
See **§3a Versioning** below.

---

## 3a. Versioning — what changes, what to specify

There are **two independent "versions"** and it helps to keep them straight:

| Version | Where it's set | What it controls |
|---|---|---|
| **Module version** | `VERSION=` in `install_pixi_module.sh`; mirrored by the install dir name `lib/scagent-<ver>-pixi/`, the modulefile path `modulefiles/scagent/<ver>`, and `.version` | What users type: `module load scagent/<ver>`. The `.version` file picks the default when they just type `module load scagent`. |
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
2. Set `VERSION=` to the new value in `install_pixi_module.sh`.
3. Re-run `bash install_pixi_module.sh` → builds `lib/scagent-<new>-pixi/`, writes
   `modulefiles/scagent/<new>`, and updates `.version` to the new default.
4. Set the lab key/config in the new `<dir>/.env` (a fresh deploy copies the
   predecessor's `.env` if present, else seeds from `.env.template`).

> Note: step 3 builds the **whole env** via `pixi install` (~17G, minutes). Only do
> this when the environment/deps changed (refresh `pixi/pixi.lock` first), or when
> you want a frozen, immutable snapshot users can pin.

### Which method for which change

| Kind of change | Method | Bump version? |
|---|---|---|
| API key / model / `base_url` | edit shared `.env` in place | no |
| scagent **Python code** (testing phase) | `update_pixi_code.sh` — overwrites `0.1.0` in place | no (stay on 0.1.0) |
| scagent code, **stable release** people pin | bump version → `install_pixi_module.sh` | yes |
| **New pip / conda dependency** | edit `pixi/pixi.toml`, `pixi install` to refresh the lock, commit, then `install_pixi_module.sh` | only if releasing |
| **RAPIDS / CUDA bump** | edit pins in `pixi/pixi.toml`, re-lock, re-deploy | yes (new env = new version) |

Rule of thumb: **overwrite `0.1.0` in place while iterating with testers**
(everyone always wants your latest); **bump to a new version once others depend on
a stable build** and you don't want to move it under them.

### First-time full deploy (already done; for reference / a fresh version)
```bash
bash deploy/scagent-module/install_pixi_module.sh --dry-run   # validate, write nothing
bash deploy/scagent-module/install_pixi_module.sh             # pixi install + freeze + modulefile
# then set the lab key in <dir>/.env
```
Run on a GPU node, as a `grp_hpc_collab002` member, with PyPI access.

---

## 4. Gotchas (so they don't bite again)

- **pixi defaults to strict channel priority; RAPIDS needs it off.** Without
  `channel-priority = "disabled"` in `[workspace]`, the RAPIDS solve fails. This is
  the single most important line in `pixi.toml`.
- **`pixi install` won't pick up a same-version code change.** pixi/uv caches the
  built scagent wheel by version; if you edit code but leave `version = "0.1.0"`,
  `pixi install` sees the lock satisfied and no-ops, serving the stale build
  (verified). That's why both `install_pixi_module.sh` and `update_pixi_code.sh`
  force-reinstall scagent from a freshly built wheel (`_freeze_scagent.sh`).
- **Don't pin torch to the cu13 wheel.** It seems right (the driver is cu13), but
  `download.pytorch.org/whl/cu130` torch conflicts with the conda-pinned
  `cuda-bindings==13.3.1` and makes the solve unsatisfiable. The default `+cu128`
  torch runs fine on the cu13 driver.
- **`PYTHONPATH` shadows the frozen env.** A dev shell that sourced `setup*.sh`
  exports `PYTHONPATH=<repo>`, so `import scagent` resolves to the repo, not the
  frozen env — which can look like the freeze failed. `module load` sets no
  PYTHONPATH, so users are unaffected; the deploy/update scripts verify with
  `env -u PYTHONPATH` from `/tmp` to avoid the false alarm.
- **`scvi-tools` was an undeclared dependency.** scagent's scVI batch correction
  imports it but it was missing from `pyproject.toml` (hand-installed in the old
  conda env). Now declared as the `scvi` extra and pulled in via `scagent[all]`.
- **`source-sh` is unavailable in this Lmod's TCL** → modulefile errored with
  `invalid command name "source-sh"`. We bake the `activate.d` env vars as `setenv`
  instead. (A `.lua` modulefile with `source_sh` would also work.)
- **`.gitignore` packaging trap.** The pattern `run*` matched the tracked source
  file `scagent/agent/run_manager.py`; hatchling honors `.gitignore` via pathspec
  and **silently dropped it from every wheel** (git itself didn't, since it's
  tracked — only `git check-ignore --no-index` flags it). Broke `scagent --help`
  with `ModuleNotFoundError: scagent.agent.run_manager`. Fixed: `run*` → `run*/`.
  The wheel-freeze step (`_freeze_scagent.sh`) hard-checks wheel completeness to
  catch a recurrence.
- **conda `--clone` can't relocate a pip-built env** — use conda-pack.
- **The wheel build needs PyPI access** (the hatchling backend isn't in the env);
  installing the built wheel does not.

---

## 5. Follow-ups (deferred)

1. ~~SCIMILARITY_MODEL_PATH + SCAGENT_CELLBENDER point at personal dirs.~~ **DONE
   (2026-06-22).** Both now live in the shared sail tree, world-readable:
   - SCimilarity v2 models at `/data1/collab002/sail/shared/models/sci/{human_v2,
     mouse_v1}` (the module sets `SCIMILARITY_MODEL_PATH` + `SCIMILARITY_MODEL_PATH_MOUSE`;
     scagent auto-picks by organism). Opened with `chmod -R o+rX` on just those two
     model dirs — siblings (`human_v2_old`, `immune_cd8_nk_finetuned`) stay locked;
     the `sci`/`models` parents keep traverse-only (`o+x`), so nothing else is exposed.
   - CellBender rebuilt as a shared pixi env at
     `/data1/collab002/sail/shared/tools/cellbender` (manifest+lock in repo at
     `cellbender-pixi/`). Now torch 2.10+cu128 → **GPU-capable** (the old personal
     env's torch 1.13/cu117 predated Hopper, so it was CPU-only on the H200s).
2. **Retire the conda-pack predecessor.** Once the pixi build has had enough test
   mileage, remove `lib/scagent-0.1.0/` (~15G) and the conda scripts
   (`install_scagent_module.sh`, `update_scagent_code.sh`, `modulefile/0.1.0`).
   Reclaim with: `chmod -R u+w lib/scagent-0.1.0 && rm -rf lib/scagent-0.1.0`.
3. **pixi binary is in a personal dir.** It's at
   `/usersoftware/peerd/ibrahih3/.pixi/bin/pixi` (cache at
   `$PIXI_CACHE_DIR=/usersoftware/peerd/ibrahih3/.pixi/cache`). The lab tree already
   ships a shared `pixi` module (`lib/pixi`); consider standardizing on it so deploys
   don't depend on a personal install.
4. **Stable default provider:** decide whether the lab `.env` should point at the
   self-hosted endpoint (current) or Anthropic (robust) before wider rollout.
```
