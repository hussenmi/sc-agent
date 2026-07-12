"""OpenShell-backed sandbox for scagent's ``run_code``.

Runs model-generated Python inside a per-session NVIDIA OpenShell sandbox
(out-of-process, Landlock filesystem policy + network allowlist + unprivileged
user) instead of the in-process ``exec``. When OpenShell is unavailable (e.g. an
HPC without cgroups-v2 / Landlock), the agent keeps using the in-process path —
which execution mode is active is decided once at startup by :func:`capability`
and shown in the ``scagent start`` welcome box.

This module owns the infrastructure only; the result-shaping (artifacts_created,
error hints, output caps) stays in ``tools.py`` and is reused verbatim across
both paths. The in-sandbox half lives in :mod:`scagent.agent.sandbox_runner`.

See ``docs/openshell_integration.md`` for the design and the validating spikes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_IMAGE = "scagent-sbx:cpu"
_RUNNER = Path(__file__).with_name("sandbox_runner.py")
_MANIFEST_PREFIX = "__MANIFEST__"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Where the runner reads/writes inside the sandbox (mirrors sandbox_runner.py).
_SBX_RUNNER = "/sandbox/_sandbox_runner.py"
_SBX_CODE = "/sandbox/user_code.py"
_SBX_ADATA = "/sandbox/adata.h5ad"
_SBX_OUT = "/sandbox/out"


def _run(args: List[str], timeout: int = 120, check: bool = False) -> subprocess.CompletedProcess:
    """Invoke the ``openshell`` CLI (or ``docker``) and capture output.

    stdin is closed (DEVNULL): ``openshell sandbox exec`` forwards stdin and
    otherwise blocks waiting for it when run non-interactively.
    """
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=check,
        stdin=subprocess.DEVNULL,
    )


def capability(image: str = DEFAULT_IMAGE) -> Dict[str, Any]:
    """Cheap, no-container check of whether OpenShell can back ``run_code``.

    Returns ``{"available": bool, "reason": str, "image": image}``. Never raises.
    """
    out: Dict[str, Any] = {"available": False, "reason": "", "image": image}
    if shutil.which("openshell") is None:
        out["reason"] = "openshell CLI not found on PATH"
        return out
    try:
        status = _run(["openshell", "status"], timeout=15)
    except Exception as e:
        out["reason"] = f"openshell status failed: {e}"
        return out
    if "Connected" not in _ANSI.sub("", status.stdout + status.stderr):
        out["reason"] = "OpenShell gateway not connected"
        return out
    # If docker is present, require the image to exist locally (create pulls only
    # from a registry; our default image is a local build). If docker is absent
    # we can't verify cheaply — defer to create-time.
    if shutil.which("docker") is not None:
        try:
            insp = _run(["docker", "image", "inspect", image], timeout=15)
        except Exception as e:
            out["reason"] = f"docker image inspect failed: {e}"
            return out
        if insp.returncode != 0:
            out["reason"] = (
                f"sandbox image '{image}' not built — run: "
                f"docker build -t {image} docker/scagent-sandbox"
            )
            return out
    out["available"] = True
    out["reason"] = "ready"
    return out


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def resolve_mode() -> str:
    """SCAGENT_SANDBOX → one of 'auto' | 'openshell' | 'off'."""
    v = _env("SCAGENT_SANDBOX", "auto").lower()
    if v in ("off", "0", "false", "no", "inprocess", "in-process", "none"):
        return "off"
    if v in ("openshell", "on", "1", "true", "yes", "force"):
        return "openshell"
    return "auto"


def image_from_env() -> str:
    return _env("SCAGENT_SANDBOX_IMAGE") or DEFAULT_IMAGE


def build_from_env() -> Dict[str, Any]:
    """Resolve the run_code execution mode once (at agent startup).

    Returns a decision dict::

        {mode, image, sandbox: OpenShellSandbox|None, isolated: bool,
         reason: str, fatal: str|None}

    - ``off``       → in-process (current behavior).
    - ``auto``      → OpenShell iff available, else in-process (graceful).
    - ``openshell`` → OpenShell required; if unavailable, ``fatal`` is set so the
                      CLI can stop loudly instead of silently dropping isolation.
    """
    mode = resolve_mode()
    image = image_from_env()
    decision: Dict[str, Any] = {
        "mode": mode, "image": image, "sandbox": None,
        "isolated": False, "reason": "", "fatal": None,
    }
    if mode == "off":
        decision["reason"] = "in-process (SCAGENT_SANDBOX=off)"
        return decision
    cap = capability(image)
    if cap["available"]:
        decision["sandbox"] = OpenShellSandbox(image=image)
        decision["isolated"] = True
        decision["reason"] = f"OpenShell (isolated) · image {image}"
    elif mode == "openshell":
        decision["fatal"] = cap["reason"]
        decision["reason"] = f"OpenShell required but unavailable — {cap['reason']}"
    else:  # auto → fall back to in-process, gracefully
        decision["reason"] = f"in-process (OpenShell unavailable — {cap['reason']})"
    return decision


class SandboxInfraError(RuntimeError):
    """Raised when the sandbox itself (create/upload/exec/download) fails —
    as opposed to a user-code error, which is reported inside the manifest."""


class OpenShellSandbox:
    """A per-session OpenShell sandbox. Created lazily on first ``run_code``,
    deleted via :meth:`delete` (wired to ``SCAgent.close``)."""

    def __init__(self, image: str = DEFAULT_IMAGE, name: Optional[str] = None,
                 exec_timeout: int = 900):
        self.image = image
        self.name = name or f"scagent-sbx-{os.getpid()}-{int(time.time())}"
        self.exec_timeout = exec_timeout
        self._created = False
        self._runner_uploaded = False

    # --- lifecycle -------------------------------------------------------
    def ensure_created(self) -> None:
        if self._created:
            return
        # `-- true` runs a benign command and (without --no-keep) leaves the
        # sandbox Ready; the CLI returns promptly once the supervisor is up.
        res = _run(
            ["openshell", "sandbox", "create", "--from", self.image,
             "--name", self.name, "--", "true"],
            timeout=180,
        )
        if res.returncode != 0:
            raise SandboxInfraError(
                f"sandbox create failed: {_ANSI.sub('', res.stderr or res.stdout).strip()}"
            )
        self._wait_ready()
        self._created = True

    def _wait_ready(self, timeout: int = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = _run(["openshell", "sandbox", "get", self.name], timeout=15)
            if "Ready" in _ANSI.sub("", res.stdout):
                return
            time.sleep(0.5)
        raise SandboxInfraError(f"sandbox '{self.name}' did not reach Ready in {timeout}s")

    def delete(self) -> None:
        """Idempotent teardown."""
        if not self._created:
            return
        try:
            _run(["openshell", "sandbox", "delete", self.name], timeout=60)
        except Exception:
            pass
        self._created = False

    # --- helpers ---------------------------------------------------------
    def _upload(self, local: str, dest: str) -> None:
        # --no-git-ignore is REQUIRED: without it the CLI applies .gitignore
        # filtering and silently drops files (learned in the spike).
        res = _run(
            ["openshell", "sandbox", "upload", "--no-git-ignore", self.name, local, dest],
            timeout=300,
        )
        if res.returncode != 0:
            raise SandboxInfraError(
                f"upload {local} -> {dest} failed: {_ANSI.sub('', res.stderr or res.stdout).strip()}"
            )

    def _upload_runner(self) -> None:
        if self._runner_uploaded:
            return
        self._upload(str(_RUNNER), _SBX_RUNNER)
        self._runner_uploaded = True

    # --- the main entry point -------------------------------------------
    def run_code(self, code: str, adata: Any, run_dir: Path) -> Dict[str, Any]:
        """Execute ``code`` against ``adata`` inside the sandbox.

        Downloads artifacts into ``run_dir`` and returns a dict with host-side
        paths already rebased:
            stdout, error({type,message,traceback}|None), warnings[],
            artifacts[{path,role,metadata}], adata_out_path|None,
            adata_reassigned, output_path|None
        Raises :class:`SandboxInfraError` on infrastructure failure.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_created()
        self._upload_runner()

        # Clean any state from a previous call so stale artifacts aren't re-downloaded.
        _run(["openshell", "sandbox", "exec", "-n", self.name, "--",
              "sh", "-c", f"rm -rf {_SBX_OUT} {_SBX_ADATA} {_SBX_CODE}"], timeout=60)

        # Stage user code + the current AnnData, upload them.
        code_tmp = run_dir / ".sandbox_user_code.py"
        adata_tmp = run_dir / ".sandbox_in.h5ad"
        try:
            code_tmp.write_text(code)
            self._upload(str(code_tmp), _SBX_CODE)
            if adata is not None:
                adata.write_h5ad(adata_tmp)
                self._upload(str(adata_tmp), _SBX_ADATA)

            # Run the sandboxed runner; it always exits 0 and prints one
            # __MANIFEST__ line carrying success/failure.
            res = _run(
                ["openshell", "sandbox", "exec", "-n", self.name, "--workdir", "/sandbox",
                 "--timeout", str(self.exec_timeout), "--",
                 "python3", _SBX_RUNNER],
                timeout=self.exec_timeout + 60,
            )
            manifest = self._parse_manifest(res)

            # Pull produced artifacts back into the run directory (contents of
            # /sandbox/out land directly in run_dir, preserving subdirs).
            _run(["openshell", "sandbox", "download", self.name, _SBX_OUT, str(run_dir)],
                 timeout=300)
        finally:
            for f in (code_tmp, adata_tmp):
                try:
                    f.unlink()
                except OSError:
                    pass

        return self._rebase(manifest, run_dir)

    def _parse_manifest(self, res: subprocess.CompletedProcess) -> Dict[str, Any]:
        for line in res.stdout.splitlines():
            if line.startswith(_MANIFEST_PREFIX):
                try:
                    return json.loads(line[len(_MANIFEST_PREFIX):])
                except json.JSONDecodeError as e:
                    raise SandboxInfraError(f"could not parse sandbox manifest: {e}")
        # No manifest → the runner never completed (infra/OOM/timeout).
        tail = _ANSI.sub("", (res.stderr or res.stdout))[-800:].strip()
        raise SandboxInfraError(
            f"sandbox exec produced no manifest (exit {res.returncode}). Output tail:\n{tail}"
        )

    def _rebase(self, manifest: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        artifacts = []
        for a in manifest.get("artifacts", []):
            rel = a.get("rel")
            if not rel:
                continue  # written outside /sandbox/out — not downloaded
            artifacts.append({
                "path": str(run_dir / rel),
                "role": a.get("role", "artifact"),
                "metadata": a.get("metadata") or {},
            })
        adata_out = run_dir / "adata_out.h5ad" if manifest.get("adata_written") else None
        output_path = manifest.get("output_path")
        if output_path:
            # Rebase a user-set output_path if it pointed inside /sandbox/out.
            op = Path(output_path)
            try:
                output_path = str(run_dir / op.relative_to(_SBX_OUT))
            except Exception:
                output_path = str(op)
        return {
            "stdout": manifest.get("stdout", ""),
            "error": manifest.get("error"),
            "warnings": manifest.get("warnings", []),
            "artifacts": artifacts,
            "adata_out_path": str(adata_out) if adata_out else None,
            "adata_reassigned": bool(manifest.get("adata_reassigned")),
            "output_path": output_path,
        }
