"""Within-sample paired-DEG batch check + per-cell-metric UMAP overlay hints.

The paired DEG is the clean control for a batch effect: for two clusters
dominated by DIFFERENT samples that look like the same cell type, DEG each
against the rest of its OWN sample and compare the signatures. Batch is held
constant inside each test, so a match means the same population split by sample
(a batch effect). And tools that write per-cell metrics now advertise
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


def test_paired_deg_flags_same_type_across_samples_as_conclusive():
    a = _batch_split_adata(same_type_across_samples=True)
    res = diagnose_batch_effect(a, batch_key="sample", cluster_key="leiden",
                                min_cells_per_cluster_sample=5, n_top_genes=15)
    idg = res["cross_sample_identity_deg"]
    # the same-type cross-sample pair (1 in A, 3 in B) is found and conclusive
    pair = next(
        (p for p in idg["conclusive_pairs"]
         if {p["cluster_a"], p["cluster_b"]} == {"1", "3"}),
        None,
    )
    assert pair is not None
    assert pair["sample_a"] != pair["sample_b"]
    # its shared within-sample identity is the planted identity genes
    assert {"G0", "G1", "G2"}.issubset(set(pair["shared_identity_genes"]))
    assert res["verdict"] == "batch_effect_supported"
    assert any("within-sample identity DEG" in r for r in res["support_reasons"])


def test_paired_deg_does_not_flag_different_types():
    # clusters 1 (A, genes G0-2) and 3 (B, genes G5-7) are different populations;
    # their within-sample identity signatures should NOT match.
    a = _batch_split_adata(same_type_across_samples=False)
    res = diagnose_batch_effect(a, batch_key="sample", cluster_key="leiden",
                                min_cells_per_cluster_sample=5, n_top_genes=15)
    idg = res["cross_sample_identity_deg"]
    pair = next(
        (p for p in idg["candidate_pairs"]
         if {p["cluster_a"], p["cluster_b"]} == {"1", "3"}),
        None,
    )
    # either not paired at all (markers don't overlap) or paired but not conclusive
    assert pair is None or pair["conclusive_batch_effect"] is False


def test_result_carries_identity_deg_and_artifact():
    a = _batch_split_adata(same_type_across_samples=True)
    res = diagnose_batch_effect(a, batch_key="sample", cluster_key="leiden",
                                min_cells_per_cluster_sample=5, n_top_genes=15)
    idg = res["cross_sample_identity_deg"]
    assert "interpretation" in idg and "within-sample" in idg["interpretation"].lower()
    assert "signature_similarity_min" in idg


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
