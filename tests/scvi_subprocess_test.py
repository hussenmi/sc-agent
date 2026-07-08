"""scVI now trains in a subprocess (scagent/batch/_scvi_worker.py) to isolate
torch's CUDA context from the parent. This exercises the full handoff on CPU:
build minimal AnnData -> temp h5ad -> worker trains -> latent read back.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("scvi")

from scagent.batch.scvi import run_scvi


def _adata(n=80, g=60, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.6, size=(n, g)).astype("float32")
    # give two batches a mild offset so batch_key is meaningful
    counts[: n // 2, : g // 3] += rng.poisson(2.0, size=(n // 2, g // 3))
    a = ad.AnnData(X=sp.csr_matrix(counts))
    a.layers["raw_counts"] = a.X.copy()
    a.obs["batch"] = (["a"] * (n // 2) + ["b"] * (n - n // 2))
    a.var["highly_variable"] = True
    return a


def test_run_scvi_subprocess_cpu_writes_latent():
    a = _adata()
    run_scvi(
        a,
        batch_key="batch",
        n_latent=8,
        max_epochs=2,
        use_gpu=False,
        early_stopping=False,
        use_hvg=False,
    )
    assert "X_scVI" in a.obsm
    assert a.obsm["X_scVI"].shape == (a.n_obs, 8)
    assert np.isfinite(a.obsm["X_scVI"]).all()


def test_run_scvi_missing_batch_key_raises():
    a = _adata()
    with pytest.raises(ValueError, match="Batch key"):
        run_scvi(a, batch_key="nonexistent", use_gpu=False, max_epochs=2)


def test_run_scvi_missing_layer_raises():
    a = _adata()
    del a.layers["raw_counts"]
    with pytest.raises(ValueError, match="Layer"):
        run_scvi(a, batch_key="batch", layer="raw_counts", use_gpu=False, max_epochs=2)


def test_run_scvi_writes_training_diagnostics(tmp_path):
    """Explicit max_epochs -> diagnostics (history CSV, loss plot, metrics) are produced
    and the run is correctly reported as having hit the cap (no early stopping)."""
    a = _adata()
    run_scvi(
        a,
        batch_key="batch",
        n_latent=8,
        max_epochs=3,
        use_gpu=False,
        early_stopping=False,
        use_hvg=False,
        diagnostics_dir=str(tmp_path),
    )
    m = a.uns.get("scvi_training")
    assert m is not None
    assert m["resolved_max_epochs"] == 3
    assert m["epochs_trained"] == 3
    assert m["early_stopped"] is False  # ran the full cap
    assert m["final_elbo_train"] is not None
    assert (tmp_path / "scvi_training_history.csv").exists()
    assert (tmp_path / "scvi_training_loss.png").exists()
    assert m["loss_plot"].endswith("scvi_training_loss.png")


def test_run_scvi_unset_epochs_uses_scvi_heuristic():
    """max_epochs=None -> scVI's get_max_epochs_heuristic sets the cap; early stopping
    governs the actual run length."""
    from scvi.model._utils import get_max_epochs_heuristic

    a = _adata()
    run_scvi(a, batch_key="batch", n_latent=8, max_epochs=None, use_gpu=False, use_hvg=False)
    m = a.uns["scvi_training"]
    assert m["resolved_max_epochs"] == int(get_max_epochs_heuristic(a.n_obs))
    assert m["epochs_trained"] <= m["resolved_max_epochs"]
