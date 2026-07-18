"""Count-magnitude colorings must use a LOG color scale, everywhere.

Right-skewed library-size metrics (total_counts, n_genes_by_counts) collapse to a
flat dark map on a linear colormap because a few ultra-high-count cells own the
whole scale (run_2026_07_15_121111 umap_total_counts.png). Every plot path — the
auto per-cell overlays and generate_figure — routes count colorings through
_log_color_norm so they are painted logged, not raw.
"""

from __future__ import annotations

import json
import os

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")

from scagent.agent.tools import _log_color_norm, _plot_umap_overlays, process_tool_call


def _adata():
    rng = np.random.default_rng(0)
    n = 400
    a = ad.AnnData(rng.poisson(1, (n, 15)).astype("float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"G{i}" for i in range(15)]
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    tc = rng.lognormal(8, 1.1, n)
    tc[:3] = tc.max() * 4  # heavy right tail
    a.obs["total_counts"] = tc
    a.obs["n_genes_by_counts"] = rng.lognormal(6, 0.7, n)
    a.obs["pct_counts_mt"] = rng.uniform(0, 20, n)
    return a


def test_log_color_norm_selects_count_metrics_only():
    a = _adata()
    for key in ("total_counts", "n_genes_by_counts"):
        norm, title = _log_color_norm(a, key)
        assert norm is not None
        assert "log scale" in title
    # bounded fractions and unknown keys stay linear
    assert _log_color_norm(a, "pct_counts_mt") == (None, None)
    assert _log_color_norm(a, "not_a_column") == (None, None)


def test_log_color_norm_falls_back_to_linear_on_nonpositive():
    a = _adata()
    a.obs["total_counts"] = a.obs["total_counts"].copy()
    a.obs.loc[a.obs_names[0], "total_counts"] = 0.0  # LogNorm rejects <= 0
    norm, title = _log_color_norm(a, "total_counts")
    assert norm is None and title is None


def test_overlays_log_scale_count_panels(tmp_path):
    a = _adata()
    saved = _plot_umap_overlays(a, ["total_counts", "pct_counts_mt"], str(tmp_path))
    assert len(saved) == 2
    assert all(os.path.exists(p) and os.path.getsize(p) > 0 for p in saved)


def test_generate_figure_uses_log_for_counts(tmp_path):
    a = _adata()
    out = str(tmp_path / "u.png")
    r, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "umap", "color_by": "total_counts", "include_image": False,
         "output_path": out},
        a,
    )
    d = json.loads(r)
    assert d["status"] == "ok"
    assert os.path.exists(d["output_path"]) and os.path.getsize(d["output_path"]) > 0
