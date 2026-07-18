"""GPU→CPU handoff must be enforced so a leaked cupy matrix never reaches a CPU
consumer (scanpy DEG, scipy structure-QC, h5ad write, the run_code sandbox).

Root cause pinned here (run_2026_07_15_143719): once one tool leaves adata on the
GPU, on_gpu's re-entrancy (`moved = not is_on_gpu`) makes every later GPU step
think someone else owns the round-trip, so it never moves data back — a cascade
that broke Leiden, structure-QC, DEG, and prepare_annotation in the same run.

These run on the CPU-only test env: a lightweight fake stands in for a cupy
array/cupyx sparse matrix (module name starts with cupy/cupyx and exposes .get()),
which is exactly what ensure_cpu keys off.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scagent.core import gpu


class _FakeCupy:
    """Duck-types a cupyx sparse matrix: GPU-resident module + .get()->host."""

    __module__ = "cupyx.scipy.sparse._csr"

    def __init__(self, host):
        self._host = host
        self.shape = host.shape
        self.dtype = host.dtype

    def get(self):
        return self._host


class _DuckAnnData:
    """Minimal AnnData stand-in exposing exactly the slots ensure_cpu touches.

    Real AnnData re-validates aligned mappings (obsp/obsm/layers/varm) on read and
    rejects a non-registered fake, so a duck type is the only way to exercise the
    multi-slot conversion off-GPU. Real cupy arrays pass (anndata's CupyArray /
    CupySparseMatrix compat types) — this fake stands in for those.
    """

    def __init__(self):
        h = np.eye(4, dtype="float32")
        self.X = _FakeCupy(np.ones((4, 2), dtype="float32"))
        self.layers = {"counts": _FakeCupy(np.ones((4, 2), dtype="float32"))}
        self.obsm = {"X_pca": _FakeCupy(np.ones((4, 3), dtype="float32"))}
        self.obsp = {"connectivities": _FakeCupy(h.copy())}
        self.varm = {"PCs": _FakeCupy(np.ones((2, 3), dtype="float32"))}
        self.raw = None


def _leaked_adata():
    """A real AnnData whose X is 'on the GPU' (X getter does not re-validate)."""
    host = np.ones((6, 3), dtype="float32")
    a = ad.AnnData(X=host.copy())
    a._X = _FakeCupy(host.copy())  # bypass validation to inject a GPU-resident X
    return a


def test_is_gpu_array_and_to_host():
    fake = _FakeCupy(np.arange(4))
    assert gpu._is_gpu_array(fake) is True
    assert gpu._is_gpu_array(np.arange(4)) is False
    host = gpu._to_host(fake)
    assert isinstance(host, np.ndarray)
    assert gpu._to_host(np.arange(4)).tolist() == [0, 1, 2, 3]  # already host: passthrough


def test_has_gpu_arrays_detects_x():
    a = _leaked_adata()
    assert gpu.is_on_gpu(a) is True
    assert gpu.has_gpu_arrays(a) is True


def test_ensure_cpu_moves_x_back():
    a = _leaked_adata()
    moved = gpu.ensure_cpu(a)
    assert moved is True
    assert not gpu.has_gpu_arrays(a)
    assert isinstance(a.X, np.ndarray)


def test_ensure_cpu_converts_every_slot():
    """X plus layers/obsm/obsp/varm all get moved off the device."""
    a = _DuckAnnData()
    assert gpu.has_gpu_arrays(a) is True
    moved = gpu.ensure_cpu(a)
    assert moved is True
    assert gpu.has_gpu_arrays(a) is False
    assert isinstance(a.X, np.ndarray)
    for mapping in (a.layers, a.obsm, a.obsp, a.varm):
        for v in mapping.values():
            assert not gpu._is_gpu_array(v)


def test_ensure_cpu_noop_on_clean_adata():
    a = ad.AnnData(X=np.ones((5, 2), dtype="float32"))
    assert gpu.ensure_cpu(a) is False  # nothing to do → no work, no warning
    assert isinstance(a.X, np.ndarray)


def test_ensure_cpu_handles_none():
    assert gpu.ensure_cpu(None) is False


def test_process_tool_call_wrapper_enforces_cpu(monkeypatch):
    """The single dispatch boundary must hand back a host-resident AnnData even when
    the underlying tool leaked one on the GPU."""
    from scagent.agent import tools

    leaked = _leaked_adata()

    def _fake_impl(tool_name, tool_input, adata=None, **kwargs):
        return json.dumps({"status": "ok", "tool": tool_name}), leaked

    monkeypatch.setattr(tools, "_process_tool_call_impl", _fake_impl)
    result_json, out = tools.process_tool_call("some_tool", {}, ad.AnnData(X=np.ones((6, 3), "float32")))
    assert json.loads(result_json)["status"] == "ok"
    assert not gpu.has_gpu_arrays(out), "wrapper left a GPU-resident matrix on the returned adata"
    assert isinstance(out.X, np.ndarray)


def test_run_differential_expression_gives_clear_error_on_bad_matrix():
    """A non-float matrix must yield an actionable DEG error, not scanpy's opaque one."""
    from scagent.core.clustering import run_differential_expression

    # object-dtype X can't be log/ranked; scanpy fails deep inside — we want a clear message.
    a = ad.AnnData(X=np.array([["a", "b"], ["c", "d"], ["e", "f"]], dtype=object))
    a.obs["leiden"] = pd.Categorical(["0", "1", "0"])
    with pytest.raises(RuntimeError, match="Differential expression"):
        run_differential_expression(a, groupby="leiden", method="wilcoxon")
