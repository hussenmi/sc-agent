"""Phase 2/3: the model records its inspection judgment (column roles + species)
via record_inspection; the runtime validates it, stores it as a decision, and
overrides the heuristic in the data summary. The tool is exposed only under the
SCAGENT_MODEL_INSPECTION flag."""

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.agent.tools import get_openai_tools, get_tools
from scagent.agent.world_state import AgentWorldState


def _adata():
    n, g = 60, 6
    X = (np.arange(n * g) % 5).reshape(n, g).astype(np.int32)
    labels = ["T cell", "B cell", "NK cell"]
    obs = pd.DataFrame(
        {
            "cell_barcode": [f"AAACCTG{i:04d}-1" for i in range(n)],  # identifier
            "donor": [f"Donor_{i % 8:02d}" for i in range(n)],
            "labels": [labels[i % 3] for i in range(n)],  # a real label column
        },
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=["MT-ND1", "HLA-A", "CD3D", "CD19", "MS4A1", "GAPDH"])
    return AnnData(X=X, obs=obs, var=var)


def _synced_ws(adata):
    ws = AgentWorldState()
    ws.sync_from_adata(adata)
    return ws


def test_record_inspection_no_celltype_overrides_heuristic():
    adata = _adata()
    ws = _synced_ws(adata)
    # Model says: barcodes are not labels -> omit cell_type_col; donor is the batch.
    out = ws.record_inspection(
        {"batch_col": "donor", "species": "human", "rationale": "barcodes are unique per cell"},
        adata=adata,
    )
    assert out["status"] == "ok"
    assert ws.data_summary["processing"]["has_celltypes"] is False
    assert ws.data_summary["cell_type_key"] is None
    assert "external_or_manual" not in ws.annotation_sources
    # batch routed through the existing batch_key slot; species overridden.
    assert ws.get_confirmed_value("batch_key") == "donor"
    assert ws.data_summary["batch_key"] == "donor"
    assert ws.data_summary["biological_context"]["species"] == "human"
    assert ws.data_summary["biological_context"]["species_source"] == "model_inspection"
    # The decision is surfaced back to the model.
    assert ws.data_summary["inspection"]["batch_col"] == "donor"


def test_record_inspection_with_real_celltype_column():
    adata = _adata()
    ws = _synced_ws(adata)
    out = ws.record_inspection({"cell_type_col": "labels", "species": "human"}, adata=adata)
    assert out["status"] == "ok"
    assert ws.data_summary["processing"]["has_celltypes"] is True
    assert ws.data_summary["cell_type_key"] == "labels"
    assert "external_or_manual" in ws.annotation_sources


def test_record_inspection_overrides_all_judgments():
    # Every semantic judgment — cluster, tissue, condition — comes from the model.
    adata = _adata()
    adata.obs["leiden"] = pd.Categorical([str(i % 4) for i in range(adata.n_obs)])
    ws = _synced_ws(adata)
    out = ws.record_inspection(
        {
            "cell_type_col": "labels",
            "batch_col": "donor",
            "cluster_col": "leiden",
            "species": "human",
            "tissue": "lung",
            "condition": "IPF",
        },
        adata=adata,
    )
    assert out["status"] == "ok"
    assert ws.data_summary["cluster_key"] == "leiden"
    bc = ws.data_summary["biological_context"]
    assert bc["tissue"] == "lung" and bc["tissue_source"] == "model_inspection"
    assert bc["condition"] == "IPF" and bc["condition_source"] == "model_inspection"
    assert bc["species"] == "human"


def test_inspection_yields_to_post_annotation_celltype():
    # cell_type_col=null means "no PRE-EXISTING label column at load" (suppresses
    # the predicted_doublet false-positive). It must NOT clobber the cell_type
    # column the pipeline creates: once annotation is finalized, the real column
    # is reported. Regression for has_celltypes=False after a finished annotation.
    n = 60
    labels = ["T cell", "B cell", "NK cell"]
    obs = pd.DataFrame(
        {
            "cell_barcode": [f"BC{i}-1" for i in range(n)],
            "donor": [f"Donor_{i % 8:02d}" for i in range(n)],
            "cell_type": [labels[i % 3] for i in range(n)],  # produced by the pipeline
        },
        index=[f"c{i}" for i in range(n)],
    )
    obs["donor"] = obs["donor"].astype("category")
    obs["cell_type"] = obs["cell_type"].astype("category")
    adata = AnnData(
        X=(np.arange(n * 5) % 4).reshape(n, 5).astype(np.int32),
        obs=obs,
        var=pd.DataFrame(index=["CD3D", "CD19", "MS4A1", "GAPDH", "ACTB"]),
    )
    ws = AgentWorldState()
    ws.record_inspection({"batch_col": "donor", "species": "human"}, adata=adata)

    # Before annotation finalizes: override holds → no cell types reported.
    assert ws.data_summary["processing"]["has_celltypes"] is False
    assert ws.data_summary["cell_type_key"] is None

    # After annotation finalizes: the real cell_type column must be reported.
    ws.annotation_validation = {"required": True, "finalized": True, "status": "validated_and_finalized"}
    ws.sync_from_adata(adata)
    assert ws.data_summary["cell_type_key"] == "cell_type"
    assert ws.data_summary["processing"]["has_celltypes"] is True


def test_record_inspection_drops_missing_cluster_column():
    # Partial-accept: an invalid column is dropped with a warning, not rejected —
    # a single guessed field no longer forces a retry loop.
    adata = _adata()
    ws = _synced_ws(adata)
    out = ws.record_inspection({"cluster_col": "no_such_clusters"}, adata=adata)
    assert out["status"] == "ok"
    assert any("no_such_clusters" in w for w in out["warnings"])
    # The invalid field is dropped (recorded as omitted); the inspection is stored.
    assert out["inspection"]["cluster_col"] is None
    assert ws.get_confirmed_value("inspection") is not None


def test_record_inspection_drops_missing_column_but_keeps_valid():
    adata = _adata()
    ws = _synced_ws(adata)
    out = ws.record_inspection(
        {"cell_type_col": "does_not_exist", "species": "human"}, adata=adata
    )
    assert out["status"] == "ok"
    assert any("does_not_exist" in w for w in out["warnings"])
    # Invalid field dropped, valid field (species) still recorded.
    assert out["inspection"]["cell_type_col"] is None
    assert out["inspection"]["species"] == "human"
    assert ws.get_confirmed_value("inspection") is not None


def test_record_inspection_coerces_bad_species_to_unknown():
    adata = _adata()
    ws = _synced_ws(adata)
    out = ws.record_inspection({"species": "martian"}, adata=adata)
    assert out["status"] == "ok"
    assert out["inspection"]["species"] == "unknown"
    assert any("martian" in w for w in out["warnings"])


def test_tool_exposed_by_default_and_gated_off(monkeypatch):
    # Default ON (unset): record_inspection is exposed in both schema formats.
    monkeypatch.delenv("SCAGENT_MODEL_INSPECTION", raising=False)
    anthropic_names = {t["name"] for t in get_tools()}
    openai_names = {t["function"]["name"] for t in get_openai_tools()}
    assert "record_inspection" in anthropic_names
    assert "record_inspection" in openai_names

    # Explicit OFF removes it.
    monkeypatch.setenv("SCAGENT_MODEL_INSPECTION", "0")
    assert "record_inspection" not in {t["name"] for t in get_tools()}
