from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scagent.analysis.batch_diagnostic import diagnose_batch_effect


def test_batch_diagnostic_finds_shared_signature_and_confounding(tmp_path):
    genes = ["CD3D", "CD3E", "LYZ", "LST1", "IFIT1", "ISG15"]
    rows = []
    obs = []
    for cluster, marker_pair in [("0", ("CD3D", "CD3E")), ("1", ("LYZ", "LST1"))]:
        for sample in ["s1", "s2"]:
            for i in range(18):
                expr = np.ones(len(genes))
                expr[genes.index(marker_pair[0])] = 8
                expr[genes.index(marker_pair[1])] = 7
                if sample == "s2":
                    expr[genes.index("IFIT1")] = 9
                    expr[genes.index("ISG15")] = 8
                rows.append(expr)
                obs.append(
                    {
                        "sample": sample,
                        "condition": "treated" if sample == "s2" else "control",
                        "leiden": cluster,
                    }
                )
    adata = ad.AnnData(
        np.asarray(rows, dtype=float),
        obs=pd.DataFrame(obs, index=[f"cell_{i}" for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    adata.raw = adata.copy()
    adata.obsm["X_umap"] = np.column_stack(
        [
            np.where(adata.obs["sample"].values == "s2", 5.0, 0.0),
            np.where(adata.obs["leiden"].values == "1", 5.0, 0.0),
        ]
    )

    result = diagnose_batch_effect(
        adata,
        batch_key="sample",
        cluster_key="leiden",
        condition_keys=["condition"],
        min_cells_per_cluster_sample=10,
        output_dir=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert result["verdict"] == "confounded_with_condition"
    assert result["shared_cross_cell_type_signatures"]
    assert any(
        entry["gene"] == "IFIT1" and entry["n_broad_labels"] >= 2
        for entry in result["shared_cross_cell_type_signatures"]
    )
    assert result["condition_confounding"][0]["confounded_with_batch"] is True
    assert (tmp_path / "batch_diagnostic_cluster_sample_composition.csv").exists()
    # A self-documenting README is written alongside the CSVs, with the column
    # glossary and an empty (to be filled by the model) interpretation section.
    readme = tmp_path / "README.md"
    assert readme.exists()
    readme_text = readme.read_text()
    from scagent.core import artifact_docs as _ad

    assert _ad.DOC_MARKER in readme_text
    assert "signature_similarity" in readme_text
    assert _ad.interpretation_is_empty(readme_text) is True
    # Each CSV artifact carries a machine-readable column glossary in metadata.
    csv_arts = [
        a for a in result["artifacts_created"]
        if a.get("role") == "artifact" and str(a.get("path", "")).endswith(".csv")
    ]
    assert csv_arts and all("columns" in (a.get("metadata") or {}) for a in csv_arts)
    assert any(a.get("role") == "artifact_readme" for a in result["artifacts_created"])
    # terminal_summary: human-readable findings the agent prints to the terminal.
    # The verdict line is a plain-language label (the machine slug stays on
    # result["verdict"], asserted above), not the raw enum value.
    summary = result["terminal_summary"]
    assert summary[0].startswith("Verdict:")
    assert "confounded" not in summary[0].lower() or "condition" in summary[0].lower()
    assert "confounded_with_condition" not in summary[0]  # no raw slug leaks
    assert summary[-1].startswith("→ ")  # recommendation line


def test_batch_diagnostic_cautions_when_condition_metadata_missing():
    genes = ["CD3D", "CD3E", "LYZ", "LST1", "IFIT1", "ISG15", "IFIT2", "MX1", "OAS1"]
    rows = []
    obs = []
    for cluster, marker_pair in [("0", ("CD3D", "CD3E")), ("1", ("LYZ", "LST1"))]:
        for sample in ["LUNG_T01", "EBUS_02"]:
            for i in range(18):
                expr = np.ones(len(genes))
                expr[genes.index(marker_pair[0])] = 8
                expr[genes.index(marker_pair[1])] = 7
                if sample == "EBUS_02":
                    for gene in ["IFIT1", "ISG15", "IFIT2", "MX1", "OAS1"]:
                        expr[genes.index(gene)] = 8
                rows.append(expr)
                obs.append({"sample": sample, "leiden": cluster})
    adata = ad.AnnData(
        np.asarray(rows, dtype=float),
        obs=pd.DataFrame(obs, index=[f"cell_{i}" for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    adata.raw = adata.copy()

    result = diagnose_batch_effect(
        adata,
        batch_key="sample",
        cluster_key="leiden",
        min_cells_per_cluster_sample=10,
    )

    assert result["verdict"] == "batch_effect_supported"
    assert any("confounding was not tested" in reason for reason in result["caution_reasons"])
    assert "sample/source/procedure effects" in result["recommendation"]
    # No X_pca on this dataset -> the entropy check degrades gracefully, not crashes.
    assert result["neighborhood_batch_entropy"] is None
    assert any("batch-mixing entropy was skipped" in reason for reason in result["caution_reasons"])


def _two_cluster_two_sample_adata():
    genes = ["CD3D", "CD3E", "TRAC", "LYZ", "LST1", "S100A8"]
    rows = []
    obs = []
    for cluster, markers in [("0", ("CD3D", "CD3E", "TRAC")), ("1", ("LYZ", "LST1", "S100A8"))]:
        for sample in ["s1", "s2"]:
            for _ in range(40):
                expr = np.ones(len(genes))
                for marker in markers:
                    expr[genes.index(marker)] = 8
                rows.append(expr)
                obs.append({"sample": sample, "leiden": cluster})
    adata = ad.AnnData(
        np.asarray(rows, dtype=float),
        obs=pd.DataFrame(obs, index=[f"cell_{i}" for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    adata.raw = adata.copy()
    return adata


def test_neighborhood_entropy_flags_pca_segregated_samples(tmp_path):
    adata = _two_cluster_two_sample_adata()
    # X_pca: each sample sits in a far-apart region, so a cell's nearest neighbors
    # are all from its own sample -> neighborhood entropy ~0 -> low mixing ratio.
    sample_offset = np.where(adata.obs["sample"].values == "s2", 100.0, 0.0)
    rng = np.random.default_rng(0)
    adata.obsm["X_pca"] = np.column_stack([sample_offset, rng.normal(scale=0.01, size=adata.n_obs)])

    result = diagnose_batch_effect(
        adata,
        batch_key="sample",
        cluster_key="leiden",
        min_cells_per_cluster_sample=10,
        entropy_n_neighbors=10,
        output_dir=str(tmp_path),
    )

    entropy = result["neighborhood_batch_entropy"]
    assert entropy is not None and entropy["skipped"] is False
    assert entropy["global_ceiling"] == 1.0  # balanced 50/50 samples -> ceiling is log2(2)
    assert entropy["mixing_ratio"] <= 0.5
    assert entropy["entropy_per_broad_label"]  # per-cell-type breakdown present
    assert any(
        "batch-mixing entropy" in reason for reason in result["support_reasons"]
    )
    # Per-cell entropy is exposed on obs so a downstream UMAP can paint it.
    assert "batch_diagnostic_neighborhood_entropy" in adata.obs.columns
    assert (tmp_path / "batch_diagnostic_neighborhood_entropy.csv").exists()
    assert any(
        "Neighborhood batch-mixing entropy reflects mixing" in limit
        for limit in result["evidence_limits"]
    )


def test_neighborhood_entropy_quiet_when_samples_well_mixed():
    adata = _two_cluster_two_sample_adata()
    # X_pca: clusters separate (x = cluster) but samples interleave within each
    # cluster -> neighborhoods are sample-mixed -> high entropy, no support reason.
    cluster_offset = np.where(adata.obs["leiden"].values == "1", 10.0, 0.0)
    rng = np.random.default_rng(1)
    adata.obsm["X_pca"] = np.column_stack(
        [cluster_offset, rng.normal(scale=0.5, size=adata.n_obs)]
    )

    result = diagnose_batch_effect(
        adata,
        batch_key="sample",
        cluster_key="leiden",
        min_cells_per_cluster_sample=10,
        entropy_n_neighbors=10,
    )

    entropy = result["neighborhood_batch_entropy"]
    assert entropy is not None and entropy["skipped"] is False
    assert entropy["mixing_ratio"] >= 0.8  # well mixed
    assert not any(
        "batch-mixing entropy" in reason for reason in result["support_reasons"]
    )
    # Clusters here are cell types with both samples evenly present -> independent of
    # sample -> ARI/NMI ~0, no concordance support reason.
    concordance = result["cluster_batch_concordance"]
    assert concordance["tracks_sample"] is False
    assert concordance["ari"] < 0.2
    assert not any("ARI" in reason for reason in result["support_reasons"])


def test_cluster_batch_concordance_and_epithelial_caveat(tmp_path):
    # Each sample contributes its own clusters (sample-exclusive), and the s2 clusters
    # are epithelial -> clusters track sample (high ARI/NMI) AND the epithelial caveat
    # should name the driving markers (EPCAM/KRT).
    genes = ["CD3D", "CD3E", "TRAC", "IL7R", "EPCAM", "KRT8", "KRT18", "KRT19"]
    t_markers = ("CD3D", "CD3E", "TRAC", "IL7R")
    epi_markers = ("EPCAM", "KRT8", "KRT18", "KRT19")
    rows = []
    obs = []
    plan = [
        ("0", "s1", t_markers),
        ("1", "s1", t_markers),
        ("2", "s2", epi_markers),
        ("3", "s2", epi_markers),
    ]
    for cluster, sample, markers in plan:
        for _ in range(30):
            expr = np.ones(len(genes))
            for marker in markers:
                expr[genes.index(marker)] = 8
            rows.append(expr)
            obs.append({"sample": sample, "leiden": cluster})
    adata = ad.AnnData(
        np.asarray(rows, dtype=float),
        obs=pd.DataFrame(obs, index=[f"cell_{i}" for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    adata.raw = adata.copy()

    result = diagnose_batch_effect(
        adata,
        batch_key="sample",
        cluster_key="leiden",
        min_cells_per_cluster_sample=10,
        output_dir=str(tmp_path),
    )

    concordance = result["cluster_batch_concordance"]
    assert concordance["tracks_sample"] is True
    assert concordance["ari"] >= 0.2
    assert concordance["nmi"] >= 0.3
    assert any(
        "track sample" in reason and "ARI" in reason for reason in result["support_reasons"]
    )
    # Epithelial caveat names the markers driving the call and stays tissue-agnostic
    # (must not assume malignancy/tumor as the explanation).
    epi_caution = [r for r in result["caution_reasons"] if "Epithelial" in r]
    assert epi_caution
    assert "EPCAM" in epi_caution[0]
    assert "donor/patient-private" in epi_caution[0]
    assert "malignant epithelium" not in epi_caution[0]  # no tumor assumption
    # ARI/NMI appears in the terminal summary and the (tissue-agnostic) caveat in limits.
    assert any("ARI" in line for line in result["terminal_summary"])
    assert any("donor/patient-private" in limit for limit in result["evidence_limits"])
