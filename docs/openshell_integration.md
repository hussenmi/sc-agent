# OpenShell Integration — Findings & Working Plan

> **Purpose.** Dedicated log for hardening scagent's code-execution sandbox with
> **NVIDIA OpenShell**. This is Arm 3 of the NVIDIA collaboration (see
> `nvidia_collab.md` for the four-arm overview); it lives in its own doc because
> it's now active work with its own findings, design, and open questions.
> **Host:** the gifted DGX Spark (`spark-e5d5`, GB10/aarch64, kernel 6.17,
> cgroups v2) — the box that unblocks OpenShell after Iris couldn't run it.

## TL;DR / status

- OpenShell **v0.0.80 is installed and running** on the Spark: gateway
  `Connected` on `https://127.0.0.1:17670`.
- The **runtime spikes are done** (round-trip + isolation verified; scanpy image
  builds in 10s) — see Spike 1 / Spike 2 below.
- **The `run_code` integration is BUILT** (2026-07-12) — `run_code` now executes
  inside a per-session OpenShell sandbox when one is available, and keeps the
  in-process path where it isn't. Both paths verified end-to-end. See
  **"Integration (built)"** below.
- **Remaining (all optional / decided):** `run_shell` stays host-side by design;
  `/dev/shm` numba-parallelism not fixable in OpenShell v0.0.80 (perf-only);
  optional fat/GPU image; adata-churn optimization. See the next-steps list.

## Why replace the current sandbox

Today's model (`scagent/agent/tools.py`):
- `run_code` → `exec(code, namespace)` **in the agent's own process**
  (`tools.py:6549`), guarded by a **string denylist** (`import os`,
  `subprocess`, `eval(`, …) plus regex guards on `adata` mutation.
- `run_shell` → `subprocess.run(shell=True)` guarded by a small
  destructive-pattern denylist (`rm -rf /`, `mkfs`, fork-bomb, …).

Both are soft, same-process sandboxes: a denylist is bypassable and `exec`
shares memory with the agent (can read/corrupt agent state, exhaust host RAM,
reach the network, touch any file the user can). OpenShell replaces this with
**out-of-process OS-level enforcement** — the same threat model, real isolation.

## OpenShell interface (as verified on the Spark)

```
openshell sandbox create [--from <img|Dockerfile|community>] [--name N]
                         [--gpu[=N]] [--cpu C] [--memory M] [--no-keep]
openshell sandbox exec   -n <name> [--workdir D] [--timeout S] [--env K=V] -- <cmd...>
openshell sandbox upload   <name> <local_path> [dest]     # host -> sandbox
openshell sandbox download <name> <sandbox_path> [dest]   # sandbox -> host
openshell sandbox get/list/delete <name>
openshell policy get/set/update <name>                   # top-level (NOT `sandbox policy`)
```

Architecture (from `nvidia_collab.md`): CLI →gRPC→ gateway (host daemon :17670)
→ compute driver → sandbox container; an `openshell-sandbox` supervisor inside
applies Landlock + seccomp + a nested netns/veth egress proxy.

## Spike findings (verified live against `osh-test`, 2026-07-11)

### Plumbing round-trip — all working

| Step | Mechanism | Result |
|------|-----------|--------|
| Code **in** | `code-string \| openshell sandbox exec -n osh-test -- python3 -` | stdout captured, remote exit code passed through |
| Artifact **out** | `openshell sandbox download osh-test /sandbox/out DEST` | works; **flattens** the dir's children into DEST (no `out/` subdir) |
| Binary **in** (adata stand-in) | `openshell sandbox upload osh-test local /sandbox/in.bin` | md5 identical across the boundary |

### Isolation is real — and declarative per-sandbox

`openshell sandbox get osh-test` returns the enforced policy:

- **Filesystem (Landlock, `best_effort`):** read-write only `/sandbox`, `/tmp`,
  `/dev/null`; `/usr /lib /proc /etc /app /var/log /dev/urandom` read-only;
  everything else denied. Verified live: `touch /etc/pwned` and `touch /pwned`
  → *Permission denied*; `ls /home` → *Permission denied*; the host repo
  `/home/hussen/projects/sc-agent` is not visible inside the sandbox.
- **Network:** default-deny egress with an **allowlist**. `https://example.com`
  → blocked; the policy explicitly permits `api.anthropic.com:443`
  (`rest`, tls terminate, `enforce`). So we control exactly what `run_code`
  can reach.
- **Process:** runs as unprivileged `sandbox` (uid/gid 998), workdir `/sandbox`.

### Two findings that shape the integration

1. **The default sandbox image is bare** — Python 3.14 venv at `/sandbox/.venv`,
   **no numpy / pandas / scanpy**. A real `run_code` needs a scagent-deps image.
   On aarch64 this is the RAPIDS/scanpy build story again (see
   `spark_arm_findings` / `environments.md`). **→ Solved in Spike 2 below.**
2. **Gotcha:** multi-line code must be piped via **stdin** (`python3 -`), *not*
   passed with `-c` — `sandbox exec` rejects newlines in a command argument
   (`command argument contains newline or carriage return characters`).

## Spike 2 — real scanpy on real data, full round-trip (2026-07-11)

Built a scagent-deps sandbox image and ran a real pipeline end-to-end. **All
green.** This validates the whole `OpenShellExecutor` design with real data.

### The deps image is easy to build

`docker build` FROM the OpenShell community base + `uv pip install scanpy` →
**10s build, all wheels, zero compilation** for Py3.14/aarch64 (scanpy 1.12.2,
anndata 0.13.1). Image `scagent-sbx:cpu`, 3.53 GB. Dockerfile + runner live in
scratchpad (`scagent-sbx/`); promote into the repo when we build the executor.
Build FROM the community base (not a fresh micromamba/miniforge base) so the
supervisor / `sandbox` user / uv-venv wiring stays intact.

```dockerfile
FROM ghcr.io/nvidia/openshell-community/sandboxes/base:latest
RUN uv pip install --python /sandbox/.venv/bin/python scanpy anndata leidenalg igraph
```
Created a persistent sandbox from it: `openshell sandbox create --from
scagent-sbx:cpu --name scagent-sbx` (drops into the container shell if no
`-- cmd`; it becomes `Ready` regardless).

### Full round-trip validated on real pbmc3k (2700 × 32738)

Uploaded `_sandbox_runner.py` (the executor preamble/epilogue — loads adata,
defines `output_dir`/`ensure_dir`/`write_report`/`register_artifact`, runs user
code, emits an artifact **manifest** + writes adata back), plus a realistic
`user_code.py` (QC → normalize → log1p → HVG → scale → PCA → neighbors →
**leiden** → figure + report → reassign `adata` to a subset). Then:
`exec python3 /sandbox/_sandbox_runner.py` → downloaded `/sandbox/out`.

Verified host-side:
- **Ran for real:** loaded `(2700, 32738)`, leiden found 5 clusters, dropped the
  smallest (13 cells) → `(2687, 1000)`.
- **Artifacts round-trip:** `reports/qc_summary.md` (correct: median mt% 2.03),
  `figures/cluster_sizes.png` (valid PNG, 299×246 RGBA), and a `_manifest.json`
  the host uses to remap sandbox paths → host paths.
- **Manifest over stdout:** a `__MANIFEST__{...json...}` line is echoed so the
  host gets artifact list + `adata_shape` even without the download.
- **adata write-back reloads:** `adata_out.h5ad` reopens host-side as
  `(2687, 1000)`, 4 leiden clusters, `X_pca` present.
- **`download` of a dir** copies the dir's *contents* into DEST, preserving
  nested subdirs (`figures/`, `reports/`) but placing top-level files
  (`adata_out.h5ad`, `_manifest.json`) directly in DEST.

### One real caveat — `/dev/shm` is read-only → serial numba/joblib

Inside the sandbox numba/joblib warn `[Errno 13] Permission denied` on
`/dev/shm` and **fall back to serial mode** (no multiprocessing semaphores).
Correctness is unaffected but parallelism is lost — a real perf hit for heavy
scanpy/leiden. **Update (2026-07-12): not fixable per-sandbox in v0.0.80** — see
the resolved `/dev/shm` entry under Next steps for the full investigation.

## Target design — `OpenShellExecutor` (original plan; superseded by "Integration (built)")

> This was the pre-build sketch. What actually shipped is in **"Integration
> (built)"** below — notably the flag defaults to `auto` (not off), and
> `run_shell` was deliberately left host-side. Kept here for the design rationale.

Route `run_code` (and later `run_shell`) through OpenShell behind an env flag,
keeping the denylist as defense-in-depth.

```
SCAGENT_SANDBOX=openshell   # off by default; when set, run_code uses the sandbox

OpenShellExecutor.run_code(code, adata, run_dir):
  1. upload the run_dir h5ad -> /sandbox/adata.h5ad            (adata in)
  2. wrap `code` with a preamble that:
       - loads adata from /sandbox/adata.h5ad
       - defines namespace helpers: ensure_dir, write_report,
         register_artifact, output_dir=/sandbox/out, Path, sc/np/pd/plt
       - runs the user code
       - appends: dump an artifact manifest (JSON) to stdout/file, and
         write adata back to /sandbox/out if it was reassigned
  3. pipe the wrapped code via stdin -> `sandbox exec -n <sb> -- python3 -`
  4. capture stdout (user prints + manifest)
  5. download /sandbox/out -> run_dir; remap sandbox paths -> host paths
     when reconstructing result.artifacts_created
```

## Integration (built) — 2026-07-12

`run_code` now runs inside a per-session OpenShell sandbox when available. The
execution mode is resolved **once at agent startup** (capability-detected) and
shown in the `scagent start` welcome box (`Sandbox:` line). Where OpenShell is
absent (e.g. Iris) the agent transparently uses the in-process path — verified
byte-for-byte unchanged.

**Config (env)** — documented in `.env.example` and set in `.env`:
- `SCAGENT_SANDBOX` = `auto` (default; OpenShell iff available, else in-process)
  · `openshell` (force; startup errors loudly if unavailable) · `off` (in-process).
- `SCAGENT_SANDBOX_IMAGE` (default `scagent-sbx:cpu`).

**Pieces:**
- `scagent/agent/sandbox.py` — `OpenShellSandbox` (capability, lazy create,
  upload/exec/download, delete) + `build_from_env()` mode resolver.
- `scagent/agent/sandbox_runner.py` — self-contained in-sandbox runner (exact
  `run_code` namespace; emits a JSON manifest with stdout / error / warnings /
  artifacts / adata write-back + reassignment flag).
- `scagent/agent/tools.py` — `process_tool_call(..., sandbox=None)`; `run_code`
  branches to the sandbox then **reuses all downstream result-shaping** (artifacts,
  error hints, output cap, var/obs fix). Denylist kept as defense-in-depth.
- `scagent/agent/agent.py` — `SCAgent` owns `self._sandbox` (from `build_from_env`),
  passes it into `process_tool_call`, deletes it in `close()`.
- `scagent/cli.py` — `Sandbox:` welcome-box line; forced-but-unavailable stops loudly.
- `docker/scagent-sandbox/Dockerfile` — the CPU scanpy image.
- `tests/openshell_sandbox_test.py` — guarded (skips without OpenShell); success
  (adata + artifacts commit) and error (reassignment discarded) paths.

**Gotchas encoded (learned during the build):**
- `sandbox upload` applies `.gitignore` filtering by default and **silently drops
  files** → the manager always uses `--no-git-ignore`.
- `sandbox exec` blocks waiting on stdin when run non-interactively → the manager
  runs all CLI calls with `stdin=DEVNULL`.
- multi-line code must go via a file/stdin, never `-- python3 -c` (newline rejected).
- `create --from <img> --name N -- true` returns promptly and leaves the sandbox Ready.

**Verified end-to-end (2026-07-12):** both `run_code` paths through
`process_tool_call` — success commits the reassigned adata + downloads the report
artifact to the run dir; error discards the reassignment and keeps the original
adata; in-process parity (output cap, small output, obs mutation) unchanged; the
existing destructive-subsetting preflight still fires (guards run before both paths).

**Live model-driven session verified (2026-07-12):** `scagent analyze --single-run`
with a real served model (Qwen3.6-35B-A3B) and `SCAGENT_SANDBOX=auto` on the Spark.
The model loaded pbmc3k, called `run_code`, and got correct results (shape
2700×32738; top genes MALAT1/TMSB4X/B2M/…). Proof it ran in the sandbox: the run
dir contains the runner's `_manifest.json` (8-key schema) + `adata_out.h5ad`
(sandbox-only artifacts); the per-session sandbox was created lazily and torn down
on session end (no orphan). The `sc.read_h5ad`-in-run_code guard fired correctly
mid-session, and the model recovered via `load_data`.

## Open questions — resolved by the build

1. **Image.** ✅ Built `scagent-sbx:cpu` (`docker/scagent-sandbox/Dockerfile`),
   CPU scanpy subset. A fat/GPU image remains an optional follow-up.
2. **adata churn.** Shipped **(a) per-call upload/download** for correctness-first
   v1. Resident-sandbox / change-detection optimization is a tracked follow-up.
3. **Scope.** **`run_code` only.** `run_shell` deliberately stays host-side
   (see Next steps) — it needs host GPU/CLI/data access.
4. **Network policy.** Kept the image's **default-deny egress**; `run_code`
   needs no network (bio/web calls run in the agent process via
   `fetch_url`/`search_papers`, not in the sandbox).

## Next steps

- [x] **Build/provision a scagent-deps sandbox** and re-run this spike with real
      `scanpy` on a real h5ad (proves the adata round-trip + artifact remap).
      **Done — Spike 2 above; `scagent-sbx:cpu` image + `scagent-sbx` sandbox.**
- [x] Prototype `OpenShellExecutor` behind `SCAGENT_SANDBOX=openshell`. **Done —
      see "Integration (built)"; both paths verified end-to-end.**
- [x] `run_shell` — **decided: stays host-side by design** (not sandboxed).
      It is a host-introspection tool (`nvidia-smi`, `df -h`, `which`/run
      `cellbender` on host data, `ls /path/to/data`); the CPU sandbox has no GPU,
      no host CLI tools, and no host data paths, so sandboxing would break its
      real use cases for no security gain (destructive patterns are already
      denylisted). `run_code` is the isolated path; `run_shell` is the deliberate,
      documented residual.
- [~] `/dev/shm` (numba/joblib parallelism) — **investigated, not fixable
      per-sandbox in OpenShell v0.0.80.** `/dev/shm` exists (tmpfs, rw, 64 MB) but
      Landlock blocks writes because it isn't in the filesystem read_write list.
      Adding it via `openshell policy set/update` on a **live** sandbox does NOT
      grant write — Landlock rulesets are one-way (a running domain can be
      tightened, never loosened), so the grant must happen at sandbox creation.
      OpenShell's `create` doesn't take a policy file (only experimental
      `--driver-config-json`); the only create-time lever is a **gateway-global**
      policy (`openshell policy set --global`), which is too broad for a
      per-session tool and would affect every sandbox on the box. **Verdict:**
      leave numba serial in the sandbox (perf-only; correctness unaffected).
      Revisit if OpenShell exposes per-sandbox create-time filesystem policy.
- [ ] adata-churn optimization: avoid re-uploading a large h5ad every call
      (resident sandbox or change-detection). **Only remaining functional TODO.**
- [ ] Optional: fat/GPU sandbox image (torch/RAPIDS) if a `run_code` snippet ever
      needs them — and whether `install_package` needs a PyPI network carve-out
      (the sandbox blocks egress).
- [ ] Not committed yet — the change lives on branch `openshell-sandbox`.

## Reference: exact spike commands

```bash
# code in via stdin, stdout capture
printf 'print("hi"); print(sum(range(10)))\n' \
  | openshell sandbox exec -n osh-test -- python3 -

# write artifact in sandbox, download it out
printf 'from pathlib import Path; p=Path("/sandbox/out"); p.mkdir(exist_ok=True); (p/"r.md").write_text("x")\n' \
  | openshell sandbox exec -n osh-test --workdir /sandbox -- python3 -
openshell sandbox download osh-test /sandbox/out ./dest/    # children land in ./dest/

# binary round-trip (adata stand-in)
openshell sandbox upload osh-test ./blob.bin /sandbox/in.bin
openshell sandbox exec -n osh-test -- python3 -c 'import hashlib,pathlib; b=pathlib.Path("/sandbox/in.bin").read_bytes(); print(len(b), hashlib.md5(b).hexdigest())'

# inspect the enforced policy
openshell sandbox get osh-test
```
