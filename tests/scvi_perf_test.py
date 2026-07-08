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


# --------------------------------------------------------------------------- #
# multi-GPU opt-in flows through the job spec (n_devices + strategy)
# --------------------------------------------------------------------------- #

def test_single_gpu_is_the_default(stub_worker, monkeypatch):
    monkeypatch.delenv("SCAGENT_SCVI_DEVICES", raising=False)
    a = _adata()
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False)
    assert stub_worker["spec"]["n_devices"] == 1
    # The strategy is still recorded but only takes effect when n_devices != 1.
    assert stub_worker["spec"]["strategy"] == "ddp_find_unused_parameters_true"


@pytest.mark.parametrize(
    "env_value,expected",
    [("-1", -1), ("all", -1), ("4", 4), ("1", 1), ("0", 1), ("bogus", 1)],
)
def test_scagent_scvi_devices_env_resolves(stub_worker, monkeypatch, env_value, expected):
    monkeypatch.setenv("SCAGENT_SCVI_DEVICES", env_value)
    a = _adata()
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False)
    assert stub_worker["spec"]["n_devices"] == expected


def test_explicit_n_devices_overrides_env(stub_worker, monkeypatch):
    monkeypatch.setenv("SCAGENT_SCVI_DEVICES", "4")
    a = _adata()
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False, n_devices=2)
    assert stub_worker["spec"]["n_devices"] == 2


def test_scvi_strategy_env_override(stub_worker, monkeypatch):
    monkeypatch.setenv("SCAGENT_SCVI_STRATEGY", "ddp_notebook_find_unused_parameters_true")
    a = _adata()
    scvi_mod.run_scvi(a, batch_key="sample", use_gpu=False)
    assert stub_worker["spec"]["strategy"] == "ddp_notebook_find_unused_parameters_true"


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


# --------------------------------------------------------------------------- #
# worker-side device planning (_plan_devices) — pure decision logic
# --------------------------------------------------------------------------- #

from scagent.batch import _scvi_worker as worker  # noqa: E402


def _pick7():
    """Stand-in for _select_gpu_device: claims the least-busy device is index 7."""
    return 7


def test_plan_devices_cpu_when_gpu_disabled():
    assert worker._plan_devices(False, 1, True, 4, _pick7) == ("cpu", 1, False)


def test_plan_devices_cpu_when_no_cuda():
    assert worker._plan_devices(True, 1, False, 0, _pick7) == ("cpu", 1, False)


def test_plan_devices_single_gpu_picks_least_busy():
    assert worker._plan_devices(True, 1, True, 4, _pick7) == ("gpu", [7], False)


def test_plan_devices_all_visible_gpus():
    # n_devices = -1 -> every visible GPU, DDP.
    assert worker._plan_devices(True, -1, True, 4, _pick7) == ("gpu", 4, True)


def test_plan_devices_caps_request_at_available():
    # Asked for 8 but only 4 are visible -> use 4.
    assert worker._plan_devices(True, 8, True, 4, _pick7) == ("gpu", 4, True)


def test_plan_devices_multi_request_with_one_gpu_falls_back_to_single():
    # DDP over a single GPU only adds overhead -> single-GPU path, no DDP.
    assert worker._plan_devices(True, -1, True, 1, _pick7) == ("gpu", [7], False)
    assert worker._plan_devices(True, 4, True, 1, _pick7) == ("gpu", [7], False)


# --------------------------------------------------------------------------- #
# worker-side DDP rank guard (_rank_zero_and_teardown)
# --------------------------------------------------------------------------- #

import logging  # noqa: E402

_log = logging.getLogger("scvi_worker_test")


def _install_fake_dist(monkeypatch, *, initialized, rank, calls):
    """Register a fake torch.distributed (and a torch parent) in sys.modules.

    Stubbing the parent ``torch`` too keeps these unit tests independent of whether
    torch is importable in the test environment.
    """
    dist = types.ModuleType("torch.distributed")
    dist.is_available = lambda: True
    dist.is_initialized = lambda: initialized
    dist.get_rank = lambda: rank
    dist.barrier = lambda: calls.append("barrier")
    dist.destroy_process_group = lambda: calls.append("destroy")
    torch_mod = types.ModuleType("torch")
    torch_mod.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)


def test_rank_zero_teardown_rank0_returns_true_and_tears_down(monkeypatch):
    calls: list = []
    _install_fake_dist(monkeypatch, initialized=True, rank=0, calls=calls)
    assert worker._rank_zero_and_teardown(_log) is True
    assert calls == ["barrier", "destroy"]


def test_rank_zero_teardown_nonzero_rank_returns_false(monkeypatch):
    calls: list = []
    _install_fake_dist(monkeypatch, initialized=True, rank=3, calls=calls)
    assert worker._rank_zero_and_teardown(_log) is False
    assert calls == ["barrier", "destroy"]  # group is torn down on every rank


def test_rank_zero_teardown_uninitialized_falls_back_to_env(monkeypatch):
    # No active process group (e.g. ddp_spawn main process) -> use launcher env var.
    _install_fake_dist(monkeypatch, initialized=False, rank=0, calls=[])
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert worker._rank_zero_and_teardown(_log) is True
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert worker._rank_zero_and_teardown(_log) is False
