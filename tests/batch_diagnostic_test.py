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
