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


def test_record_inspection_rejects_missing_column():
    adata = _adata()
    ws = _synced_ws(adata)
    out = ws.record_inspection({"cell_type_col": "does_not_exist"}, adata=adata)
    assert out["status"] == "error"
    assert any("does_not_exist" in e for e in out["errors"])
    # Nothing stored on rejection.
    assert ws.get_confirmed_value("inspection") is None
    assert "inspection" not in ws.data_summary


def test_record_inspection_rejects_bad_species():
    adata = _adata()
    ws = _synced_ws(adata)
    out = ws.record_inspection({"species": "martian"}, adata=adata)
    assert out["status"] == "error"
    assert ws.get_confirmed_value("inspection") is None


def test_tool_exposed_only_under_flag(monkeypatch):
    monkeypatch.delenv("SCAGENT_MODEL_INSPECTION", raising=False)
    assert "record_inspection" not in {t["name"] for t in get_tools()}

    monkeypatch.setenv("SCAGENT_MODEL_INSPECTION", "1")
    anthropic_names = {t["name"] for t in get_tools()}
    openai_names = {t["function"]["name"] for t in get_openai_tools()}
    assert "record_inspection" in anthropic_names
    assert "record_inspection" in openai_names
