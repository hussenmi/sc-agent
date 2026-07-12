"""In-sandbox runner for scagent's `run_code` under OpenShell.

This file runs INSIDE the OpenShell sandbox (a bare Python venv), so it must be
**fully self-contained** — no scagent imports, only the stdlib + the scientific
stack baked into the sandbox image. The host uploads it plus the user's snippet
and the current AnnData, execs it, and reads back a JSON manifest.

It reproduces the exact namespace scagent's in-process `run_code` provides
(see ``process_tool_call`` in ``tools.py``) so snippets behave identically:
``adata, sc, np, pd, plt, matplotlib, scanpy, output_dir, Path, ensure_dir,
write_report, register_artifact``.

Contract (everything lives under the sandbox's writable ``/sandbox``):
  /sandbox/adata.h5ad     input AnnData uploaded by the host        [optional]
  /sandbox/user_code.py   the user's run_code snippet               [required]
  /sandbox/out/           all produced artifacts (host downloads this)
  /sandbox/out/adata_out.h5ad   written iff the snippet ran cleanly and
                                ``adata`` is still an AnnData
The manifest is both written to ``/sandbox/out/_manifest.json`` and echoed to
stdout on a single line prefixed ``__MANIFEST__`` so the host gets it even
without the download.
"""
import io
import json
import sys
import traceback
import warnings
from contextlib import redirect_stdout
from pathlib import Path

SBX = Path("/sandbox")
OUT = SBX / "out"
MANIFEST_PREFIX = "__MANIFEST__"


def _build_namespace(out_dir, artifacts):
    """Recreate the run_code namespace. ``artifacts`` is the list the helpers
    append to; the host rebases each onto the real run directory."""

    def ensure_dir(path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _record(path, role=None, metadata=None):
        # Store the path both absolute (inside the sandbox) and, when it lives
        # under OUT, relative to OUT so the host can rebase it onto the run dir.
        p = Path(path)
        abs_p = p if p.is_absolute() else (out_dir / p)
        try:
            rel = str(abs_p.resolve().relative_to(out_dir.resolve()))
        except Exception:
            rel = None  # written outside OUT — host cannot download it
        rec = {
            "sandbox_path": str(abs_p),
            "rel": rel,
            "role": str(role) if role else "artifact",
            "metadata": dict(metadata) if isinstance(metadata, dict) else {},
        }
        for existing in artifacts:  # dedupe by sandbox path
            if existing.get("sandbox_path") == rec["sandbox_path"]:
                return existing
        artifacts.append(rec)
        return rec

    def register_artifact(path, role=None, metadata=None, columns=None):
        meta = dict(metadata) if isinstance(metadata, dict) else {}
        if columns:
            meta["columns"] = [
                {"name": str(k), "description": str(v)} for k, v in columns.items()
            ]
        return _record(path, role=role, metadata=meta or None)

    def write_report(name, content):
        reports_dir = ensure_dir(out_dir / "reports")
        # Match the host's safe_name derivation exactly (tools.py write_report).
        safe_name = name.replace(" ", "_").rstrip(".md")
        path = reports_dir / f"{safe_name}.md"
        path.write_text(content)
        _record(path, role="report", metadata={"name": safe_name})
        return str(path)

    ns = {
        "output_dir": str(out_dir),
        "Path": Path,
        "ensure_dir": ensure_dir,
        "write_report": write_report,
        "register_artifact": register_artifact,
    }
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc
        ns.update(np=np, pd=pd, plt=plt, matplotlib=matplotlib, sc=sc, scanpy=sc)
    except Exception as e:  # pure-python snippets still run
        ns["__import_error__"] = f"{type(e).__name__}: {e}"
    return ns


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = []
    manifest = {
        "stdout": "",
        "error": None,            # {type, message, traceback}
        "warnings": [],           # [{category, message}]
        "artifacts": [],          # [{rel, role, metadata, sandbox_path}]
        "adata_written": False,
        "adata_shape": None,
        "adata_reassigned": False,
        "output_path": None,
    }

    ns = _build_namespace(OUT, artifacts)
    if "__import_error__" in ns:
        manifest["import_warning"] = ns.pop("__import_error__")

    adata_path = SBX / "adata.h5ad"
    if adata_path.exists() and "sc" in ns:
        ns["adata"] = ns["sc"].read_h5ad(adata_path)
    orig_id = id(ns.get("adata"))

    code = (SBX / "user_code.py").read_text()

    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with redirect_stdout(buf):
                exec(compile(code, "<run_code>", "exec"), ns)
        except Exception as exc:
            manifest["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
    manifest["stdout"] = buf.getvalue()
    manifest["warnings"] = [
        {"category": w.category.__name__, "message": str(w.message)} for w in caught
    ]

    adata = ns.get("adata")
    manifest["adata_reassigned"] = adata is not None and id(adata) != orig_id

    # Commit adata only on a clean run — matches the host's all-or-nothing
    # discard of a reassignment when the snippet raises.
    if manifest["error"] is None and adata is not None and adata.__class__.__name__ == "AnnData":
        try:
            adata.write_h5ad(OUT / "adata_out.h5ad")
            manifest["adata_written"] = True
            manifest["adata_shape"] = list(adata.shape)
        except Exception as e:
            manifest["adata_write_error"] = f"{type(e).__name__}: {e}"

    # Surface a user-set output_path variable (host registers + echoes it).
    op = ns.get("output_path")
    if isinstance(op, (str, Path)):
        manifest["output_path"] = str(op)

    manifest["artifacts"] = artifacts
    try:
        (OUT / "_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    except Exception:
        pass
    print(MANIFEST_PREFIX + json.dumps(manifest, default=str))
    # Always exit 0: the manifest carries success/failure. A non-zero exit
    # would make the host's CLI wrapper conflate infra errors with user errors.


if __name__ == "__main__":
    main()
