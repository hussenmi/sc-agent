"""Tests for user-facing output documentation: the analysis-record glossary/legends,
the DEG CSV column guide, and the auto-generated figures index README."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.agent.tools import (
    _assemble_analysis_record,
    _write_deg_column_doc,
    process_tool_call,
    unique_output_path,
)
from scagent.core import artifact_docs as ad

# --- analysis_record intro + legends -------------------------------------------


def _world_state():
    return SimpleNamespace(
        step_log=[
            {"tool": "run_qc", "cells_before": 100, "genes_before": 200},
            {"tool": "run_clustering", "method": "leiden", "resolution": 1.0,
             "n_clusters": 5, "cluster_key": "leiden"},
        ],
        cluster_qc_registry={
            "leiden": {
                "thresholds_used": {"mt_threshold": 25.0},
                "proposed_removal": [],
                "cluster_decisions": {
                    "0": {"recommended_action": "keep", "severity": "clean",
                          "reasons": ["within expected ranges"]},
                },
            }
        },
        data_summary={"shape": {"n_cells": 100, "n_genes": 200},
                      "data_type": "cells", "processing": {"has_pca": True}},
    )


class _Adata:
    uns = {
        "annotation_validation": {
            "external_validation_policy": "cytopus_local_primary_panglaodb_fallback",
            "label_counts": {"T cell": 50},
            "per_cluster_evidence": {
                "0": {"label": "T cell", "confidence": "low",
                      "validation_tier": "deg_primary", "supporting_genes": ["CD3D"],
                      "panglaodb_queried": False,
                      "competing_labels_considered": ["NK cell"],
                      "reasoning": "CD3D present"},
            },
        }
    }


def test_analysis_record_has_intro_and_legends():
    out = _assemble_analysis_record(_world_state(), _Adata())
    # Orientation preface for a non-expert reader
    assert "## How to read this record" in out
    # Friendly step names appear alongside the raw tool name
    assert "Cluster cells" in out
    assert "`run_clustering`" in out
    # Recurring-settings glossary
    assert "What the recurring settings mean" in out
    assert "**resolution**" in out
    # Per-table column guides
    assert out.count("_Column guide:_") == 2  # QC table + annotation table
    # Annotation table legend: independent evidence columns (Tier/Confidence removed)
    assert "**Scimilarity prediction**" in out
    assert "**Celltypist prediction**" in out
    assert "**DEG prediction**" in out
    assert "**Severity**" in out  # QC table legend


def test_analysis_record_empty_when_no_state():
    assert _assemble_analysis_record(None, None) == ""


# --- DEG column guide ----------------------------------------------------------


def test_deg_column_doc_documents_statistics(tmp_path):
    df = pd.DataFrame({
        "cluster": ["0", "0"], "gene": ["A", "B"],
        "log2fc": [2.4, 1.1], "pval_adj": [1e-9, 1e-3],
        "pval": [1e-11, 1e-4], "score": [30.0, 12.0],
    })
    csv = tmp_path / "deg_leiden_rank_genes_groups.csv"
    df.to_csv(csv, index=False)
    registered = []
    _write_deg_column_doc(str(csv), df, groupby="leiden",
                          run_manager=SimpleNamespace(add_output=registered.append))
    readme = tmp_path / "deg_leiden_rank_genes_groups.README.md"
    assert readme.exists()
    text = readme.read_text()
    assert ad.DOC_MARKER in text
    # Column meanings are spelled out in plain language
    assert "log2fc" in text and "fold-change" in text
    assert "Benjamini" in text  # pval_adj explanation
    # It is a reference doc, no interpretation section demanded
    assert ad.INTERPRETATION_HEADING not in text
    assert registered == [str(readme)]


# --- figures index README ------------------------------------------------------


def test_build_figures_readme_none_when_empty(tmp_path):
    (tmp_path / "figures").mkdir()
    assert ad.build_figures_readme(tmp_path / "figures") is None


def test_build_figures_readme_indexes_and_explains(tmp_path):
    fdir = tmp_path / "figures"
    (fdir / "pre_integration").mkdir(parents=True)
    (fdir / "post_integration").mkdir(parents=True)
    (fdir / "pre_integration" / "umap_donor_res1.0.png").write_bytes(b"")
    (fdir / "post_integration" / "umap_donor_res1.0.png").write_bytes(b"")
    (fdir / "scvi_training_loss.png").write_bytes(b"")
    text = ad.build_figures_readme(fdir)
    assert text is not None
    assert ad.DOC_MARKER in text
    # naming/folder convention explained
    assert "self-describing" in text
    # only present types are glossed
    assert "UMAP embedding" in text
    assert "scVI training curve" in text
    assert "Dot plot" not in text  # no dotplot in this run
    # files indexed under their subfolders
    assert "### `pre_integration`" in text
    assert "### `post_integration`" in text
    # no interpretation section (reference doc)
    assert ad.INTERPRETATION_HEADING not in text


def test_write_figures_readme_roundtrip(tmp_path):
    fdir = tmp_path / "figures"
    fdir.mkdir()
    (fdir / "umap_leiden_res1.0.png").write_bytes(b"")
    path = ad.write_figures_readme(fdir)
    assert path is not None and path.exists()
    assert path.name == "README.md"
    # opted-out README is not flagged as missing interpretation by the finalize scan
    assert not ad.interpretation_is_empty(path.read_text())


def test_figure_type_detection_specificity():
    # cluster_qc box-plot beats the generic umap/heatmap fallbacks
    assert ad._figure_type_for("qc_metrics_by_cluster_pass_001.png")[0] == "Per-cluster QC box plots"
    assert ad._figure_type_for("cluster_5_correlation.png")[0] == "Gene–gene correlation heatmap"
    assert ad._figure_type_for("umap_donor_post_integration.png")[0] == "UMAP embedding"
    assert ad._figure_type_for("something_unknown.png") is None


# --- render_readme opt-out wording ---------------------------------------------


def test_render_readme_optout_wording_no_interpretation_promise():
    doc = ad.ArtifactGroupDoc(group="g", title="t", overview="o", files=[],
                              interpretation_required=False)
    text = ad.render_readme(doc)
    assert "explanatory only" in text
    assert "Interpretation" not in text


# --- figure subfolder saves ----------------------------------------------------


def test_unique_output_path_preserves_subfolder(tmp_path):
    sub = tmp_path / "post_integration"
    sub.mkdir()
    p = sub / "umap_donor.png"
    p.write_bytes(b"")
    got = unique_output_path(str(p))
    assert got.endswith(os.path.join("post_integration", "umap_donor_2.png"))


# --- run_cluster_qc auto-generates a cluster-colored UMAP ----------------------


def _clustered_umap_adata():
    n, k, g = 120, 3, 60
    rng = np.random.RandomState(0)
    obs = pd.DataFrame(
        {
            "total_counts": rng.normal(5000, 200, n),
            "n_genes_by_counts": rng.normal(2000, 80, n),
            "pct_counts_mt": rng.normal(5, 0.5, n),
            "leiden_res_1_5": pd.Categorical([str(i % k) for i in range(n)]),
        },
        index=[f"c{i}" for i in range(n)],
    )
    X = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    a = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    a.obsm["X_umap"] = rng.randn(n, 2)
    return a


def test_run_cluster_qc_emits_cluster_umap(tmp_path):
    from scagent.agent.run_manager import RunManager

    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    rj, _ = process_tool_call(
        "run_cluster_qc",
        {"cluster_key": "leiden_res_1_5", "save_checkpoint": False},
        _clustered_umap_adata(),
        run_manager=rm,
    )
    r = json.loads(rj)
    assert r["status"] == "ok"
    umap = r.get("cluster_umap_figure")
    assert umap is not None
    # Saved next to the QC outputs for this clustering, descriptively named.
    assert os.path.exists(umap)
    assert os.path.join("cluster_qc", "leiden_res_1_5") in umap
    assert "umap_leiden_res_1_5" in os.path.basename(umap)


def test_run_cluster_qc_no_umap_when_embedding_absent(tmp_path):
    from scagent.agent.run_manager import RunManager

    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    adata = _clustered_umap_adata()
    del adata.obsm["X_umap"]  # no embedding -> gracefully skip the cluster UMAP
    rj, _ = process_tool_call(
        "run_cluster_qc",
        {"cluster_key": "leiden_res_1_5", "save_checkpoint": False},
        adata,
        run_manager=rm,
    )
    r = json.loads(rj)
    assert r["status"] == "ok"
    assert r.get("cluster_umap_figure") is None
