"""Tests for optional GPU acceleration (scagent.core.gpu) and its CPU fallback.

These run on the CPU-only test environment: they verify that GPU routing is
strictly opt-in via SCAGENT_GPU, that it degrades gracefully when the GPU stack
is absent, and that the dimred/clustering functions still produce correct
results on the scanpy (CPU) path after the on_gpu wrapping was added.
"""

from __future__ import annotations

import importlib.util

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp

from scagent.core import gpu
from scagent.core.clustering import run_leiden
from scagent.core.dimred import compute_neighbors, compute_umap, run_pca
from scagent.core.normalization import normalize_data, select_hvg
from scagent.core.qc import detect_doublets

_HAS_RAPIDS = importlib.util.find_spec("rapids_singlecell") is not None


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.poisson(1.0, size=(80, 50)).astype("float32"))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


@pytest.fixture(autouse=True)
def _clear_gpu_cache():
    gpu.gpu_available.cache_clear()
    yield
    gpu.gpu_available.cache_clear()


def test_gpu_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    assert gpu.gpu_available() is False


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_gpu_falsey_values_disable(monkeypatch, value):
    monkeypatch.setenv("SCAGENT_GPU", value)
    assert gpu.gpu_available() is False


@pytest.mark.skipif(_HAS_RAPIDS, reason="rapids_singlecell present; fallback path not exercised")
def test_gpu_requested_but_unavailable_falls_back(monkeypatch):
    # SCAGENT_GPU truthy but no GPU stack installed -> must return False, not raise.
    monkeypatch.setenv("SCAGENT_GPU", "1")
    assert gpu.gpu_available() is False


def test_is_on_gpu_false_for_host_array(adata):
    assert gpu.is_on_gpu(adata) is False


def test_on_gpu_noop_when_disabled(monkeypatch, adata):
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    x_before = adata.X.copy()
    with gpu.on_gpu(adata) as on:
        assert on is False
    assert gpu.is_on_gpu(adata) is False
    np.testing.assert_array_equal(adata.X, x_before)


def test_cpu_pipeline_still_works(monkeypatch, adata):
    """Regression: PCA -> neighbors -> UMAP -> Leiden on the CPU path."""
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    run_pca(adata, n_comps=10, mask_var=None, inplace=True)
    assert adata.obsm["X_pca"].shape == (80, 10)
    assert "variance_ratio" in adata.uns["pca"]

    compute_neighbors(adata, n_neighbors=15, inplace=True)
    assert "neighbors" in adata.uns

    compute_umap(adata, inplace=True)
    assert adata.obsm["X_umap"].shape == (80, 2)

    run_leiden(adata, resolution=1.0, inplace=True)
    assert "leiden" in adata.obs
    assert adata.obs["leiden"].nunique() >= 1


def test_detect_doublets_cpu_path(monkeypatch):
    """Regression: with GPU disabled, detect_doublets runs CPU Scrublet and the
    GPU branch is skipped (GPU correctness is validated separately on a GPU env)."""
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.5, size=(400, 200)).astype("float32")
    counts[:200, :50] += rng.poisson(3.0, size=(200, 50))  # population A markers
    counts[200:, 50:100] += rng.poisson(3.0, size=(200, 50))  # population B markers
    a = ad.AnnData(X=sp.csr_matrix(counts))

    detect_doublets(a, n_prin_comps=15, random_state=0)

    assert "doublet_score" in a.obs
    assert "predicted_doublet" in a.obs
    assert a.obs["predicted_doublet"].dtype == bool
    assert "scrublet_params" in a.uns
    assert not gpu.is_on_gpu(a)


def test_normalize_data_cpu_path(monkeypatch):
    """Regression: normalize_data on CPU writes the log1p marker and normalization meta."""
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, size=(60, 40)).astype("float32")))
    normalize_data(a, target_sum=1e4, log_transform=True)
    assert a.uns.get("log1p") == {"base": None}
    assert "normalization" in a.uns
    assert not gpu.is_on_gpu(a)


def test_select_hvg_cpu_path(monkeypatch):
    """Regression: select_hvg (seurat_v3 on a counts layer) on CPU flags n_top_genes."""
    monkeypatch.delenv("SCAGENT_GPU", raising=False)
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, size=(120, 60)).astype("float32")))
    a.layers["counts"] = a.X.copy()
    select_hvg(a, n_top_genes=20, flavor="seurat_v3", layer="counts")
    assert "highly_variable" in a.var
    assert int(a.var["highly_variable"].sum()) == 20
