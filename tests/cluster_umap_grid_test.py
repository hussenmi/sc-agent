"""A single overlaid UMAP with dozens of clusters is unreadable — the per-cluster
highlight grid is the legible companion, and it must be emitted BOTH by the
generate_figure tool and by the harness-auto UMAPs (run_cluster_qc, the
pre-integration batch diagnostic).

Pins two regressions from run_2026_07_15_121111 (61 Leiden clusters, no grid):
  * The grid was capped at n_clusters > 50 → returned None precisely for the
    fine-resolution clusterings (the ladder starts at Leiden 2.0) that need it most.
  * The harness auto-UMAPs (cluster_qc, pre_integration) emitted only the single
    overlaid UMAP, never the grid — so the figures the user actually looks at had
    no grid at all.
"""

from __future__ import annotations

import glob
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import process_tool_call


def _adata_many_clusters(n_clusters=61, n_per=6):
    n = n_clusters * n_per
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=np.abs(rng.normal(size=(n, 8))).astype("float32"))
    a.var_names = [f"G{i}" for i in range(8)]
    leiden = [str(i // n_per) for i in range(n)]
    a.obs["leiden"] = pd.Categorical(leiden, categories=[str(i) for i in range(n_clusters)])
    a.obs["total_counts"] = rng.uniform(2000, 5000, n).astype("float32")
    a.obs["n_genes_by_counts"] = rng.uniform(1000, 2500, n).astype("float32")
    a.obs["pct_counts_mt"] = rng.uniform(2, 8, n).astype("float32")
    a.obsm["X_umap"] = rng.normal(size=(n, 2)).astype("float32")
    return a


def test_generate_figure_emits_grid_at_61_clusters(tmp_path):
    # 61 clusters used to exceed the old cap (50) and silently drop the grid.
    a = _adata_many_clusters(n_clusters=61)
    out = str(tmp_path / "umap_leiden.png")
    res, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "umap", "color_by": "leiden", "output_path": out, "include_image": False},
        a,
    )
    d = json.loads(res)
    assert d["status"] == "ok"
    grid = d.get("cluster_grid_path")
    assert grid, "no cluster grid emitted for a 61-cluster UMAP"
    assert os.path.exists(grid)
    assert grid.endswith("_grid.png")


def test_run_cluster_qc_emits_cluster_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _adata_many_clusters(n_clusters=55)
    res, _ = process_tool_call(
        "run_cluster_qc",
        {"save_checkpoint": False, "auto_structure_qc": False},
        a,
    )
    d = json.loads(res)
    assert d["status"] in ("ok", "success")
    grid = d.get("cluster_grid_figure")
    assert grid, "run_cluster_qc did not emit a per-cluster highlight grid"
    assert os.path.exists(grid)
    assert "_grid" in os.path.basename(grid)


def test_grid_skipped_for_trivial_partition(tmp_path):
    # A 1-cluster partition has nothing to grid — no spurious file.
    a = _adata_many_clusters(n_clusters=1, n_per=30)
    out = str(tmp_path / "umap_one.png")
    res, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "umap", "color_by": "leiden", "output_path": out, "include_image": False},
        a,
    )
    d = json.loads(res)
    assert d["status"] == "ok"
    assert not d.get("cluster_grid_path")
    assert not glob.glob(str(tmp_path / "*_grid.png"))
