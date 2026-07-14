"""Cross-sample population matching + per-cell-metric UMAP overlay hints.

For two clusters enriched for DIFFERENT samples that look like the same cell type
(their within-sample identity genes overlap), the diagnostic records a supported
identity match and compares them directly — a candidate match, never a
"conclusive" batch call. And tools that write per-cell metrics advertise
`suggested_umap_overlays` so the agent paints them on the UMAP.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import _suggested_umap_overlays
from scagent.analysis.batch_diagnostic import diagnose_batch_effect


def _batch_split_adata(same_type_across_samples: bool):
    """4 clusters across 2 samples in a realistic-width gene space. Cluster 1 (A)
    is an X-type (identity genes G0-G9). When same_type_across_samples, cluster 3
    (B) is also X-type (a batch split of the same population); otherwise it is a
    distinct Y-type (identity genes G100-G109). All sample-B cells carry a batch
    signature on G50-G59."""
    rng = np.random.default_rng(0)
    n_per, n_genes = 20, 200
    genes = [f"G{i}" for i in range(n_genes)]
    x_type = slice(0, 10)
    y_type = slice(100, 110)
    blocks = [
        ("0", "A", None),
        ("1", "A", x_type),
        ("2", "B", None),
        ("3", "B", x_type if same_type_across_samples else y_type),
    ]
    X, leiden, sample = [], [], []
    for cl, smp, ident in blocks:
        base = np.abs(rng.normal(0.5, 0.2, size=(n_per, n_genes)))
        if ident is not None:
            base[:, ident] += 3.0
        if smp == "B":
            base[:, 50:60] += 2.5  # batch signature on all sample-B cells
        X.append(base)
        leiden += [cl] * n_per
        sample += [smp] * n_per
    a = ad.AnnData(X=np.vstack(X).astype("float32"), var=pd.DataFrame(index=genes))
    a.obs["leiden"] = pd.Categorical(leiden)
    a.obs["sample"] = pd.Categorical(sample)
    a.obsm["X_pca"] = a.X[:, :10].copy()
    return a


def test_same_type_across_samples_is_a_supported_identity_match():
    a = _batch_split_adata(same_type_across_samples=True)
    res = diagnose_batch_effect(a, batch_key="sample", cluster_key="leiden",
                                min_cells_per_cluster_sample=5, min_enrichment=1.5,
                                prefer_diffxpy=False)
    # clusters 1 (A) and 3 (B) are the same population -> a supported identity match.
    pair = next(
        (p for p in res["population_pairs"]
         if {p["cluster_a"], p["cluster_b"]} == {"1", "3"}),
        None,
    )
    assert pair is not None and pair["sample_a"] != pair["sample_b"]
    assert pair["identity_match_supported"]
    assert any(g in pair["shared_top25_genes"] for g in ["G0", "G1", "G2"])
    # The direct comparison surfaces the sample-B batch signature (G50-59) as higher in B.
    ds = next(
        d for d in res["direct_pair_summaries"]
        if {d["cluster_a"], d["cluster_b"]} == {"1", "3"}
    )
    higher_b = ds["higher_in_a"] if ds["sample_a"] == "B" else ds["higher_in_b"]
    assert any(g in higher_b for g in ["G50", "G51", "G52"])
    # A single split population does not recur -> localized -> do not integrate.
    assert res["gene_evidence"] == "localized"
    assert res["recommendation"] == "do_not_integrate_based_on_current_evidence"


def test_different_types_are_not_matched():
    # clusters 1 (A, G0-9) and 3 (B, G100-109) are different populations;
    # their within-sample identity signatures should NOT support a match.
    a = _batch_split_adata(same_type_across_samples=False)
    res = diagnose_batch_effect(a, batch_key="sample", cluster_key="leiden",
                                min_cells_per_cluster_sample=5, min_enrichment=1.5,
                                prefer_diffxpy=False)
    pair = next(
        (p for p in res["population_pairs"]
         if {p["cluster_a"], p["cluster_b"]} == {"1", "3"}),
        None,
    )
    assert pair is None or pair["identity_match_supported"] is False


def test_result_carries_structured_evidence_and_artifacts(tmp_path):
    a = _batch_split_adata(same_type_across_samples=True)
    res = diagnose_batch_effect(a, batch_key="sample", cluster_key="leiden",
                                min_cells_per_cluster_sample=5, min_enrichment=1.5,
                                prefer_diffxpy=False, output_dir=str(tmp_path))
    assert "selected_pairs" in res and "population_pairs" in res
    assert (tmp_path / "batch_diagnostic_within_sample_degs.csv").exists()
    assert (tmp_path / "batch_diagnostic_population_pairs.csv").exists()


# --- suggested_umap_overlays --------------------------------------------------
def _adata_with_umap(cols):
    n = 12
    a = ad.AnnData(X=np.abs(np.random.default_rng(0).normal(size=(n, 4))).astype("float32"))
    a.obsm["X_umap"] = np.random.default_rng(1).normal(size=(n, 2))
    for c, vals in cols.items():
        a.obs[c] = vals
    return a


def test_suggested_overlays_returns_per_cell_metrics_when_umap_present():
    a = _adata_with_umap({
        "pct_counts_mt": np.linspace(1, 9, 12),
        "doublet_score": np.linspace(0, 0.5, 12),
        "myprogram_score": np.linspace(0, 1, 12),          # *_score suffix
        "batch_diagnostic_neighborhood_entropy": np.linspace(0, 1, 12),
        "cell_type": pd.Categorical(["T", "B"] * 6),        # not a metric
    })
    overlays = _suggested_umap_overlays(a)
    assert "pct_counts_mt" in overlays
    assert "doublet_score" in overlays
    assert "myprogram_score" in overlays
    assert "batch_diagnostic_neighborhood_entropy" in overlays
    assert "cell_type" not in overlays


def test_suggested_overlays_empty_without_umap():
    a = ad.AnnData(X=np.abs(np.random.default_rng(0).normal(size=(6, 4))).astype("float32"))
    a.obs["pct_counts_mt"] = np.linspace(1, 9, 6)
    assert _suggested_umap_overlays(a) == []  # nowhere to paint it yet


# --- auto-generated per-cell UMAP overlays ------------------------------------
def test_plot_umap_overlays_saves_one_figure_per_metric(tmp_path):
    from scagent.agent.tools import _plot_umap_overlays
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=np.abs(rng.normal(size=(30, 5))).astype("float32"))
    a.obsm["X_umap"] = rng.normal(size=(30, 2))
    a.obs["batch_diagnostic_neighborhood_entropy"] = rng.uniform(0, 1, 30)
    a.obs["pct_counts_mt"] = rng.uniform(1, 9, 30)
    paths = _plot_umap_overlays(a, ["batch_diagnostic_neighborhood_entropy", "pct_counts_mt"], tmp_path / "ov")
    import os
    assert len(paths) == 2
    assert all(os.path.exists(p) for p in paths)
    assert any("neighborhood_entropy" in p for p in paths)


def test_plot_umap_overlays_noop_without_umap(tmp_path):
    from scagent.agent.tools import _plot_umap_overlays
    a = ad.AnnData(X=np.abs(np.random.default_rng(0).normal(size=(10, 4))).astype("float32"))
    a.obs["pct_counts_mt"] = np.linspace(1, 9, 10)
    assert _plot_umap_overlays(a, ["pct_counts_mt"], tmp_path / "ov") == []


def test_score_gene_signature_auto_plots_overlay(tmp_path, monkeypatch):
    import json

    from scagent.agent.tools import process_tool_call
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=np.abs(rng.normal(size=(40, 8))).astype("float32"),
                   var=pd.DataFrame(index=[f"G{j}" for j in range(8)]))
    a.obsm["X_umap"] = rng.normal(size=(40, 2))
    res, _ = process_tool_call("score_gene_signature", {"gene_list": ["G0", "G1", "G2"], "score_name": "prog_score"}, a)
    d = json.loads(res)
    assert d["status"] == "ok"
    assert d["overlay_figures"] and any("prog_score" in p for p in d["overlay_figures"])
    import os
    assert all(os.path.exists(p) for p in d["overlay_figures"])
