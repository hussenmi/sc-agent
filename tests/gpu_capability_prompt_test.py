"""Tests for the runtime compute-backend awareness added in layer 1.

Two pieces: the ground-truth probe (``gpu.gpu_capability_report``) and the pure
prompt renderer (``prompts.backend_prompt_block``) that turns the report into the
"Runtime environment (compute backend)" block appended to the system prompt.
These run on the CPU-only test environment; the renderer is exercised directly
against synthetic reports so it needs no GPU.
"""

from __future__ import annotations

import importlib.util

import pytest

from scagent.agent.prompts import backend_prompt_block
from scagent.core import gpu

_HAS_RAPIDS = importlib.util.find_spec("rapids_singlecell") is not None


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    gpu.gpu_capability_report.cache_clear()
    gpu.gpu_available.cache_clear()
    gpu.reset_backends_used()
    yield
    gpu.gpu_capability_report.cache_clear()
    gpu.gpu_available.cache_clear()
    gpu.reset_backends_used()


# --- renderer: GPU active -------------------------------------------------


def test_backend_block_gpu_active_states_ground_truth():
    block = backend_prompt_block(
        {"enabled": True, "gpu": True, "n_devices": 8, "rsc_version": "0.15.2", "reason": ""}
    )
    assert "ACTIVE" in block
    assert "rapids_singlecell 0.15.2" in block
    assert "8 CUDA devices" in block
    assert "`backend`" in block  # points the model at the per-tool field
    # The whole point: forbid the "attempt if available" hedging.
    assert "attempt rapids if available" in block
    assert "OFF" not in block
    # Every block also invites the model to inspect its environment itself.
    assert "inspect it, don't guess" in block
    assert "run_shell" in block and "run_code" in block


def test_backend_block_gpu_active_singular_device():
    block = backend_prompt_block(
        {"enabled": True, "gpu": True, "n_devices": 1, "rsc_version": "0.15.2", "reason": ""}
    )
    assert "1 CUDA device." in block
    assert "CUDA devices" not in block


def test_backend_block_gpu_active_missing_version_is_safe():
    block = backend_prompt_block(
        {"enabled": True, "gpu": True, "n_devices": 2, "rsc_version": None, "reason": ""}
    )
    assert "rapids_singlecell ?" in block


# --- renderer: GPU off ----------------------------------------------------


def test_backend_block_cpu_default():
    block = backend_prompt_block(
        {"enabled": False, "gpu": False, "n_devices": 0, "rsc_version": None,
         "reason": "SCAGENT_GPU not set"}
    )
    assert "OFF" in block
    assert "scanpy/CPU" in block
    assert "ACTIVE" not in block
    assert "fell back" not in block  # plain default, not a failed request


def test_backend_block_requested_but_failed_reports_fallback():
    block = backend_prompt_block(
        {"enabled": True, "gpu": False, "n_devices": 0, "rsc_version": None,
         "reason": "ModuleNotFoundError: No module named 'cupy'"}
    )
    assert "OFF" in block
    assert "fell back to CPU" in block
    assert "ModuleNotFoundError" in block  # surfaces the concrete reason


# --- probe: ground truth --------------------------------------------------


def test_capability_report_cpu_default_skips_subprocess(monkeypatch):
    # SCAGENT_GPU unset must return immediately (no subprocess spawn cost).
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    called = False

    def _fail(*a, **k):  # would run only if the fast path is broken
        nonlocal called
        called = True
        raise AssertionError("subprocess should not run when SCAGENT_GPU is unset")

    monkeypatch.setattr(gpu.subprocess, "run", _fail)
    report = gpu.gpu_capability_report()
    assert called is False
    assert report == {
        "enabled": False, "gpu": False, "n_devices": 0,
        "rsc_version": None, "reason": "SCAGENT_GPU not set",
    }


@pytest.mark.skipif(_HAS_RAPIDS, reason="rapids_singlecell present; fallback path not exercised")
def test_capability_report_falls_back_when_stack_absent(monkeypatch):
    # SCAGENT_GPU truthy but the GPU stack is missing -> a graceful CPU report,
    # never a raise. Exercises the real subprocess probe.
    monkeypatch.setenv("SCAGENT_GPU", "1")
    report = gpu.gpu_capability_report()
    assert report["gpu"] is False
    assert report["enabled"] is True
    assert report["reason"]  # non-empty explanation of the fallback


def test_capability_report_is_cached(monkeypatch):
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    first = gpu.gpu_capability_report()
    second = gpu.gpu_capability_report()
    assert first is second  # lru_cache returns the same object


# --- backend ledger -------------------------------------------------------


def test_backend_ledger_records_reset_and_dedupes():
    gpu.reset_backends_used()
    assert gpu.backends_used() == []
    gpu.record_backend(gpu.BACKEND_GPU)
    gpu.record_backend(gpu.BACKEND_GPU)  # deduped by the set
    gpu.record_backend("scvi:gpu")
    assert gpu.backends_used() == ["rapids_singlecell", "scvi:gpu"]
    gpu.reset_backends_used()
    assert gpu.backends_used() == []


def test_on_gpu_records_cpu_when_disabled(monkeypatch):
    # on_gpu is the compute layer's ground truth: with GPU off it yields False
    # AND records the CPU backend, so any tool that used it is tagged truthfully
    # without a hardcoded tool-name list.
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    gpu.reset_backends_used()
    import anndata as ad
    import numpy as np

    adata = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    with gpu.on_gpu(adata) as on:
        assert on is False
    assert gpu.backends_used() == [gpu.BACKEND_CPU]
