"""Parent-side driver for diffxpy differential-expression tests.

diffxpy is a frozen 2020-era stack (``batchglm`` + TensorFlow) that would break
scagent's modern scanpy / anndata / RAPIDS environment if installed alongside it.
So it lives in its **own environment** and is invoked across a process boundary,
exactly like CellBender: the path to that environment's Python interpreter is
read from the ``SCAGENT_DIFFXPY`` env var. The batch diagnostic's DEFAULT engine
is the in-env scanpy Wilcoxon (identical rank statistic, instant); diffxpy is
opt-in and, in the investigation, runs the same rank test through its own engine.
This bridge also supports the NB Wald count model (``test='wald'``) — the reason
the diffxpy env exists at all — though the investigation does not use it by
default. diffxpy cold-starts TensorFlow in a subprocess per call, so it is not the
default. When ``SCAGENT_DIFFXPY`` is unset (or the interpreter is missing),
:func:`diffxpy_available` returns ``False`` and an explicit diffxpy request falls
back to Wilcoxon, so runs never break where it isn't built.

Typical use::

    from scagent.batch.diffxpy import diffxpy_available, run_two_group_de

    if diffxpy_available():
        res = run_two_group_de(matrix, group_mask, gene_names, test="wald",
                               noise_model="nb", size_factors="total_count")
    else:
        ...  # scanpy Wilcoxon fallback

``run_two_group_de`` returns a per-gene :class:`pandas.DataFrame` with columns
``gene, pval, qval, log2fc, mean`` (``log2fc`` is oriented group-vs-rest, so
positive means higher in the group of interest).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DiffxpyUnavailable(RuntimeError):
    """Raised when diffxpy is requested but no usable diffxpy environment exists."""


def group_mean_pct(matrix: Any, group: Any) -> dict[str, np.ndarray]:
    """Per-gene mean and detection rate on each side of a two-group split.

    THE single, shared definition of the displayed evidence columns. Both the
    diffxpy path (via :func:`run_two_group_de`) and the scanpy-Wilcoxon fallback
    go through :func:`attach_mean_pct` → here, so ``mean_*`` / ``pct_*`` are
    computed identically regardless of which test produced the p-values.

    ``mean_*`` is the arithmetic mean of the given expression matrix on each side;
    ``pct_*`` is the fraction of cells with non-zero expression. Returns arrays in
    the matrix's column (gene) order.
    """
    g = np.asarray(group).ravel().astype(bool)
    m_in, m_out = matrix[g], matrix[~g]

    def _mean(x):
        if x.shape[0] == 0:
            return np.zeros(x.shape[1])
        r = x.mean(axis=0)
        return np.asarray(r.todense()).ravel() if hasattr(r, "todense") else np.asarray(r).ravel()

    def _pct(x):
        if x.shape[0] == 0:
            return np.zeros(x.shape[1])
        nz = (x > 0).sum(axis=0)
        nz = np.asarray(nz.todense()).ravel() if hasattr(nz, "todense") else np.asarray(nz).ravel()
        return nz / x.shape[0]

    return {
        "mean_target": _mean(m_in),
        "mean_reference": _mean(m_out),
        "pct_target": _pct(m_in),
        "pct_reference": _pct(m_out),
    }


def attach_mean_pct(
    stats: pd.DataFrame, matrix: Any, group: Any, genes: list[str]
) -> pd.DataFrame:
    """Add the shared evidence columns to a stats frame.

    Maps the per-gene arrays from :func:`group_mean_pct` onto ``stats`` by gene
    name (so columns line up with whatever row order the test returned), then
    derives the primary displayed effect and demotes the engine's fold-change:

    - ``expression_effect = mean_target - mean_reference`` — THE primary,
      user-facing effect. On log-normalized input this is a mean log-expression
      difference; ``higher_in`` and every orientation decision are taken from its
      sign, never from the engine's fold change.
    - ``engine_log2fc`` — the raw fold-change the test reported, kept only as
      clearly-secondary engine output (a 6.75-vs-1.68 log-normalized comparison is
      not an unqualified ``log2fc`` and must not be presented as one).

    Shared by the diffxpy and Wilcoxon paths so both are byte-identical.
    """
    cols = group_mean_pct(matrix, group)
    gene_pos = {str(g): i for i, g in enumerate(genes)}
    idx = stats["gene"].astype(str).map(gene_pos)
    keep = idx.notna()
    out = stats[keep].copy()
    pos = idx[keep].astype(int).to_numpy()
    for name, arr in cols.items():
        out[name] = arr[pos]
    out["expression_effect"] = out["mean_target"] - out["mean_reference"]
    if "log2fc" in out.columns:
        out = out.rename(columns={"log2fc": "engine_log2fc"})
    return out


def diffxpy_python() -> str | None:
    """Path to the diffxpy environment's Python interpreter, or ``None``.

    Read from ``SCAGENT_DIFFXPY``. The value should point at the interpreter
    (``.../envs/scagent_diffxpy/bin/python``); a directory or the env root is
    also accepted and resolved to ``bin/python`` for convenience.
    """
    raw = os.environ.get("SCAGENT_DIFFXPY", "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_dir():
        cand = p / "bin" / "python"
        if cand.exists():
            return str(cand)
        cand = p / "python"
        if cand.exists():
            return str(cand)
        return None
    return str(p) if p.exists() else None


def diffxpy_available() -> bool:
    """True iff a diffxpy interpreter is configured and can import diffxpy.

    Does a cheap ``import diffxpy`` in the foreign interpreter so a misconfigured
    or half-built env reports unavailable rather than failing mid-run.
    """
    py = diffxpy_python()
    if not py:
        return False
    try:
        proc = subprocess.run(
            [py, "-c", "import diffxpy.api"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
            env=_worker_env(py),
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _worker_path() -> str:
    """Absolute path to the standalone worker script.

    The worker is invoked *by file path* (not ``python -m scagent...``) so the
    foreign interpreter never imports the ``scagent`` package — whose
    ``batch/__init__`` pulls in scvi/torch/scanpy that do not exist in the
    diffxpy env. The worker is deliberately self-contained (numpy/scipy/pandas/
    anndata/diffxpy only).
    """
    return str(Path(__file__).with_name("_diffxpy_worker.py"))


def _worker_env(python_exe: str) -> dict:
    """Build a *clean* environment for the foreign diffxpy interpreter.

    The parent scagent process usually runs inside a different conda env (e.g.
    RAPIDS on Iris) whose ``PYTHONHOME`` / ``PYTHONPATH`` / activation vars would
    point this py3.9 interpreter at the wrong stdlib and site-packages and break
    it outright. So we do NOT inherit ``os.environ``; we construct a minimal env
    from scratch — the diffxpy env's ``bin`` on ``PATH``, an empty
    ``PYTHONPATH`` (nothing from the parent env leaks in; the worker resolves its
    deps from the env's own site-packages), and a locale so Python's filesystem
    codec initialises. TensorFlow is kept quiet and off-GPU.
    """
    env_bin = str(Path(python_exe).resolve().parent)
    env: dict = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": os.pathsep.join([env_bin, "/usr/bin", "/bin"]),
        "PYTHONPATH": "",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "CUDA_VISIBLE_DEVICES": "",  # keep these small CPU tests off the GPU
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", os.environ.get("LANG", "C.UTF-8")),
    }
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        env["TMPDIR"] = tmpdir
    return env


def run_two_group_de(
    matrix: Any,
    group_mask: Sequence[bool] | np.ndarray,
    gene_names: Sequence[str],
    *,
    test: str = "rank",
    noise_model: str | None = None,
    size_factors: str | None = None,
    min_cells_per_group: int = 2,
    python_exe: str | None = None,
    timeout: int = 1800,
) -> pd.DataFrame:
    """Run one two-group diffxpy test (group-of-interest vs rest) in the diffxpy env.

    Parameters
    ----------
    matrix
        Cells x genes expression matrix (dense ``np.ndarray`` or ``scipy.sparse``).
        For ``noise_model='nb'`` this should be counts; for ``'norm'`` / ``t-test``
        / ``rank`` it may be normalized values.
    group_mask
        Boolean/0-1 array, length n_cells: ``True`` marks the group of interest.
    gene_names
        Gene symbols, length n_genes, aligned to the matrix columns.
    test
        diffxpy test: ``'rank'`` (Mann-Whitney), ``'t-test'`` (Welch) or
        ``'wald'`` (count GLM). ``'lrt'`` is not supported in this frozen stack;
        Wald tests the same single-coefficient hypothesis.
    noise_model
        Required for ``'wald'`` (``'nb'`` or ``'norm'``); ignored for
        ``'t-test'`` / ``'rank'``.
    size_factors
        Optional diffxpy size-factor spec (e.g. ``'total_count'``) for GLM tests.
    min_cells_per_group
        Each side (group of interest and rest) must have at least this many cells,
        else :class:`ValueError` — a DEG on a near-empty side is meaningless.

    Returns
    -------
    DataFrame sorted by ascending ``qval`` with columns ``gene, pval, qval,
    expression_effect, engine_log2fc, mean, mean_target, mean_reference,
    pct_target, pct_reference``. ``expression_effect = mean_target -
    mean_reference`` is the primary effect and the sole basis for orientation
    (positive = higher in the group of interest); ``engine_log2fc`` is the test's
    raw fold-change, kept only as secondary output. ``pct_*`` are detection rates.

    Raises
    ------
    DiffxpyUnavailable
        If no diffxpy interpreter is configured (caller should fall back).
    ValueError
        On shape mismatch, a missing/too-small group, or non-count input to NB Wald.
    RuntimeError
        If the worker subprocess fails (a real error — do NOT silently fall back).
    """
    py = python_exe or diffxpy_python()
    if not py:
        raise DiffxpyUnavailable(
            "SCAGENT_DIFFXPY is not set to a diffxpy Python interpreter; "
            "cannot run diffxpy. Fall back to the in-env Wilcoxon path."
        )

    group = np.asarray(group_mask).ravel().astype(np.int8)
    genes = [str(g) for g in gene_names]
    if group.shape[0] != _n_rows(matrix):
        raise ValueError(
            f"group_mask length {group.shape[0]} != matrix rows {_n_rows(matrix)}"
        )
    if len(genes) != _n_cols(matrix):
        raise ValueError(
            f"gene_names length {len(genes)} != matrix cols {_n_cols(matrix)}"
        )
    n_group = int((group != 0).sum())
    n_rest = int((group == 0).sum())
    if n_group < min_cells_per_group or n_rest < min_cells_per_group:
        raise ValueError(
            f"both groups need >= {min_cells_per_group} cells; got "
            f"group={n_group}, rest={n_rest}"
        )
    if test == "wald" and (noise_model or "nb") == "nb":
        _require_counts(matrix)
    work_dir = tempfile.mkdtemp(prefix="scagent_diffxpy_")
    try:
        spec: dict = {
            "genes_json": os.path.join(work_dir, "genes.json"),
            "group_npy": os.path.join(work_dir, "group.npy"),
            "result_csv": os.path.join(work_dir, "result.csv"),
            "error_out": os.path.join(work_dir, "error.txt"),
            "test": test,
            "noise_model": noise_model,
            "size_factors": size_factors,
        }
        _save_matrix(matrix, work_dir, spec)
        with open(spec["genes_json"], "w") as f:
            json.dump(genes, f)
        np.save(spec["group_npy"], group)
        spec_path = os.path.join(work_dir, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        proc = subprocess.run(
            [py, _worker_path(), spec_path],
            capture_output=True,
            env=_worker_env(py),
            timeout=timeout,
        )
        if proc.returncode != 0:
            detail = ""
            if os.path.exists(spec["error_out"]):
                with open(spec["error_out"]) as f:
                    detail = f.read()[-2000:]
            if not detail:
                detail = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]
            raise RuntimeError(
                f"diffxpy worker failed (exit {proc.returncode}).\n{detail}"
            )

        res = pd.read_csv(spec["result_csv"])
        # Add the shared per-side mean/pct evidence columns (same code path the
        # Wilcoxon fallback uses), then sort by significance.
        res = attach_mean_pct(res, matrix, group, genes)
        if "qval" in res.columns:
            res = res.sort_values("qval", kind="mergesort").reset_index(drop=True)
        return res
    finally:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


def _require_counts(matrix: Any) -> None:
    """Reject non-count input to the NB Wald GLM.

    The negative-binomial model is only meaningful on raw integer counts. A
    normalized/log matrix would be silently mis-modeled, so we fail loudly. Checks
    a bounded sample of the values (all of a sparse matrix's stored data; a capped
    dense sample) for negativity and non-integrality.
    """
    if hasattr(matrix, "tocsr"):
        data = np.asarray(matrix.tocsr().data, dtype=np.float64)
    else:
        arr = np.asarray(matrix, dtype=np.float64)
        data = arr.ravel()
    if data.size == 0:
        return
    sample = data[:2_000_000]
    if np.any(sample < 0):
        raise ValueError("NB Wald requires non-negative counts; matrix has negatives.")
    if not np.allclose(sample, np.round(sample)):
        raise ValueError(
            "NB Wald requires integer counts; matrix has non-integer values "
            "(looks normalized/log-transformed). Pass raw counts or use test='rank'."
        )


def diffxpy_versions() -> dict:
    """Package versions from the diffxpy env, for provenance. Best-effort, cached.

    Returns ``{}`` when the env is unavailable. Runs a tiny import in the foreign
    interpreter, so callers can record exactly which frozen stack produced a result.
    """
    global _VERSIONS_CACHE
    if _VERSIONS_CACHE is not None:
        return _VERSIONS_CACHE
    py = diffxpy_python()
    if not py:
        _VERSIONS_CACHE = {}
        return _VERSIONS_CACHE
    code = (
        "import json,diffxpy,numpy,scipy,pandas,anndata;"
        "print(json.dumps({'diffxpy':diffxpy.__version__,'numpy':numpy.__version__,"
        "'scipy':scipy.__version__,'pandas':pandas.__version__,"
        "'anndata':anndata.__version__}))"
    )
    try:
        proc = subprocess.run(
            [py, "-c", code],
            capture_output=True,
            timeout=180,
            env=_worker_env(py),
        )
        _VERSIONS_CACHE = json.loads(proc.stdout.decode("utf-8").strip()) if proc.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, ValueError):
        _VERSIONS_CACHE = {}
    return _VERSIONS_CACHE


_VERSIONS_CACHE: dict | None = None


def _n_rows(matrix: Any) -> int:
    return int(matrix.shape[0])


def _n_cols(matrix: Any) -> int:
    return int(matrix.shape[1])


def _save_matrix(matrix: Any, work_dir: str, spec: dict) -> None:
    """Persist the expression matrix, sparse or dense, for the worker to load."""
    if hasattr(matrix, "tocsr"):  # scipy sparse
        from scipy import sparse

        path = os.path.join(work_dir, "matrix.npz")
        sparse.save_npz(path, matrix.tocsr())
        spec["matrix_npz"] = path
    else:
        path = os.path.join(work_dir, "matrix.npy")
        np.save(path, np.asarray(matrix, dtype=np.float32))
        spec["matrix_npy"] = path


__all__ = [
    "DiffxpyUnavailable",
    "diffxpy_python",
    "diffxpy_available",
    "diffxpy_versions",
    "run_two_group_de",
]
