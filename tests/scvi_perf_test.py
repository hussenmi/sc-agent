"""Tests for the scVI speed/placement fixes in scagent.batch.scvi:

1. Train on the HVG subset instead of the full ~30k-gene matrix.
2. Select the least-busy GPU instead of always cuda:0.
3. Early stopping (passed through to the worker via the job spec).

scVI now trains in a subprocess (scagent/batch/_scvi_worker.py) to isolate
torch's CUDA context from the parent. These tests stub that subprocess so the
parent-side orchestration (HVG subsetting, spec building, latent read-back,
normalized mapping) can be checked deterministically without scvi-tools or a GPU.
The worker's own train kwargs are exercised end-to-end in scvi_subprocess_test.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scagent.batch import scvi as scvi_mod

# --------------------------------------------------------------------------- #
# fixtures / stubs
# --------------------------------------------------------------------------- #

def _adata(n_cells=60, n_genes=400, n_hvg=150, batches=3):
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32")
    a = ad.AnnData(X=counts.copy())
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.layers["raw_counts"] = counts.copy()
    a.obs["sample"] = pd.Categorical([f"s{i % batches}" for i in range(n_cells)])
    hv = np.zeros(n_genes, dtype=bool)
    hv[:n_hvg] = True
    a.var["highly_variable"] = hv
    return a


@pytest.fixture
def stub_worker(monkeypatch):
    """Simulate the _scvi_worker subprocess.

    Reads the job spec the parent wrote, captures it (and the gene count of the
    minimal AnnData handed to the worker — i.e. the HVG subset), and writes the
    fake outputs the parent reads back. Returns a dict the tests assert on.
    """
    captured: dict = {}

    def fake_run(cmd, capture_output=True, text=True, **kwargs):
        spec_path = cmd[-1]
        with open(spec_path) as f:
            spec = json.load(f)
        child = ad.read_h5ad(spec["input_h5ad"])
        captured["spec"] = spec
        captured["worker_n_vars"] = child.n_vars
        captured["worker_n_obs"] = child.n_obs

        np.save(spec["latent_out"], np.zeros((child.n_obs, spec["n_latent"]), dtype="float32"))
        if spec.get("store_normalized"):
            np.save(
                spec["normalized_out"],
                np.ones((child.n_obs, child.n_vars), dtype="float32"),
            )
            with open(spec["columns_out"], "w") as f:
                json.dump([str(c) for c in child.var_names], f)
        return types.SimpleNamespace(returncode=0, stdout="SCVI_WORKER_OK", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


# --------------------------------------------------------------------------- #
# HVG subsetting (decided in the parent, before the worker is launched)
# --------------------------------------------------------------------------- #

def test_trains_on_hvg_subset(stub_worker):
    a = _adata(n_genes=400, n_hvg=150)
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False)
    # Worker received the 150 HVGs, not all 400 genes.
    assert stub_worker["worker_n_vars"] == 150
    # Latent still lands on the full adata.
    assert a.obsm["X_scVI"].shape == (a.n_obs, 30)


def test_falls_back_to_all_genes_without_hvg_flag(stub_worker):
    a = _adata(n_genes=400, n_hvg=150)
    del a.var["highly_variable"]
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False)
    assert stub_worker["worker_n_vars"] == 400


def test_falls_back_when_too_few_hvgs(stub_worker):
    a = _adata(n_genes=400, n_hvg=5)  # below the 100-gene guard
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False)
    assert stub_worker["worker_n_vars"] == 400


def test_use_hvg_false_trains_on_all_genes(stub_worker):
    a = _adata(n_genes=400, n_hvg=150)
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False, use_hvg=False)
    assert stub_worker["worker_n_vars"] == 400


# --------------------------------------------------------------------------- #
# early stopping / params flow through the job spec
# --------------------------------------------------------------------------- #

def test_early_stopping_passed_through(stub_worker):
    a = _adata()
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False)
    assert stub_worker["spec"]["early_stopping"] is True
    assert stub_worker["spec"]["use_gpu"] is False


def test_early_stopping_can_be_disabled(stub_worker):
    a = _adata()
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False, early_stopping=False)
    assert stub_worker["spec"]["early_stopping"] is False


def test_subprocess_failure_raises(stub_worker, monkeypatch):
    """A non-zero worker exit must surface as a clear error, not a silent pass."""
    def boom(cmd, capture_output=True, text=True, **kwargs):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="scvi blew up")

    monkeypatch.setattr(subprocess, "run", boom)
    a = _adata()
    with pytest.raises(RuntimeError, match="scVI training subprocess failed"):
        scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False)


# --------------------------------------------------------------------------- #
# normalized expression maps back to the full var axis (parent-side)
# --------------------------------------------------------------------------- #

def test_store_normalized_maps_back_to_full_var_axis(stub_worker):
    a = _adata(n_genes=400, n_hvg=150)
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False, store_normalized=True)
    layer = a.layers["scvi_normalized"]
    assert layer.shape == (a.n_obs, 400)
    # HVG columns were modeled (ones); the rest stay zero.
    assert np.all(layer[:, :150] == 1.0)
    assert np.all(layer[:, 150:] == 0.0)


# --------------------------------------------------------------------------- #
# GPU selection helper
# --------------------------------------------------------------------------- #

def _fake_pynvml(free_by_index):
    """Build a fake pynvml whose memory info follows free_by_index (list)."""
    mod = types.ModuleType("pynvml")
    mod.nvmlInit = lambda: None
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetCount = lambda: len(free_by_index)
    mod.nvmlDeviceGetHandleByIndex = lambda i: i
    mod.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(free=free_by_index[h])
    return mod


def test_select_gpu_device_picks_most_free_via_nvml(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # device 2 has the most free memory.
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([2_000, 1_000, 90_000]))
    assert scvi_mod._select_gpu_device() == 2


def test_select_gpu_device_maps_cuda_visible_devices(monkeypatch):
    # Only physical GPUs 4 and 6 are visible; their CUDA ordinals are 0 and 1.
    # NVML free memory favors physical 6, which is CUDA ordinal 1.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    free = {4: 5_000, 6: 99_000}
    mod = types.ModuleType("pynvml")
    mod.nvmlInit = lambda: None
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetCount = lambda: 8
    mod.nvmlDeviceGetHandleByIndex = lambda i: i
    mod.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(free=free[h])
    monkeypatch.setitem(sys.modules, "pynvml", mod)
    assert scvi_mod._select_gpu_device() == 1


def test_select_gpu_device_falls_back_to_torch_without_nvml(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import pynvml -> ImportError
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        device_count=lambda: 3,
        mem_get_info=lambda i: ([2_000, 1_000, 90_000][i], 100_000),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert scvi_mod._select_gpu_device() == 2


def test_select_gpu_device_defaults_to_zero_on_error(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "pynvml", None)

    def _boom(_i):
        raise RuntimeError("no info")

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 2, mem_get_info=_boom)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert scvi_mod._select_gpu_device() == 0
