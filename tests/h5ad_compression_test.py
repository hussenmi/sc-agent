"""h5ad writes must be gzip-compressed by default (anndata/h5py compress
nothing by default, so count matrices ballooned to ~9 GB per file)."""

import os

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from scagent.agent.tools import _resolve_h5ad_compression, write_h5ad_safe


def _adata(n_obs=400, n_var=800, seed=0):
    rng = np.random.default_rng(seed)
    # Sparse-ish integer counts — representative and compressible.
    X = sp.random(n_obs, n_var, density=0.05, random_state=seed, dtype="float32")
    X.data = np.round(X.data * 50).astype("float32")
    a = ad.AnnData(X=X.tocsr())
    a.obs["cluster"] = rng.integers(0, 8, n_obs).astype(str)
    return a


def test_default_compression_is_gzip(monkeypatch):
    monkeypatch.delenv("SCAGENT_H5AD_COMPRESSION", raising=False)
    monkeypatch.delenv("SCAGENT_H5AD_COMPRESSION_LEVEL", raising=False)
    assert _resolve_h5ad_compression() == ("gzip", None)


def test_compression_env_overrides(monkeypatch):
    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION", "none")
    assert _resolve_h5ad_compression() == (None, None)
    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION", "off")
    assert _resolve_h5ad_compression() == (None, None)
    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION", "lzf")
    assert _resolve_h5ad_compression() == ("lzf", None)
    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION", "gzip")
    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION_LEVEL", "6")
    assert _resolve_h5ad_compression() == ("gzip", 6)
    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION_LEVEL", "not-an-int")
    assert _resolve_h5ad_compression() == ("gzip", None)


def test_write_h5ad_safe_gzip_is_smaller_and_roundtrips(tmp_path, monkeypatch):
    a = _adata()

    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION", "none")
    p_none = tmp_path / "none.h5ad"
    details_none = write_h5ad_safe(a, str(p_none))
    assert details_none["compression"] is None

    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION", "gzip")
    p_gz = tmp_path / "gzip.h5ad"
    details_gz = write_h5ad_safe(a, str(p_gz))
    assert details_gz["compression"] == "gzip"

    # gzip must actually shrink the file, and the data must round-trip intact.
    assert p_gz.stat().st_size < p_none.stat().st_size
    back = ad.read_h5ad(p_gz)
    assert back.shape == a.shape
    np.testing.assert_allclose(
        np.asarray(back.X.todense()), np.asarray(a.X.todense())
    )


def test_write_h5ad_safe_reports_compression_in_details(tmp_path, monkeypatch):
    monkeypatch.setenv("SCAGENT_H5AD_COMPRESSION", "gzip")
    details = write_h5ad_safe(_adata(n_obs=50, n_var=100), str(tmp_path / "x.h5ad"))
    assert "compression" in details
    assert details["compression"] == "gzip"
