"""Biological context must distinguish text the USER provided from text a
tool/model supplied (e.g. the `context` arg of load_data). A model that guesses
"PBMC scRNA-seq" must not have that laundered into user_provided context — else
the model later mistakes its own guess for user ground truth (the PBMC bug)."""

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.analysis.context import infer_biological_context
from scagent.agent.world_state import AgentWorldState


def _adata():
    genes = ["HLA-A", "HLA-B", "CD3D", "CD19", "GAPDH", "ACTB"]  # human symbols
    return AnnData(
        X=np.ones((6, len(genes)), dtype=np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame(index=genes),
    )


# --- context.py: user vs model-supplied attribution ---------------------------

def test_model_supplied_context_is_not_user_provided():
    ctx = infer_biological_context(_adata(), text_context="", hint_context="PBMC scRNA-seq")
    assert ctx.tissue == "PBMC"
    assert ctx.provenance["tissue"] == "context_supplied"
    assert ctx.context_supplied.get("tissue") == "PBMC"
    assert "tissue" not in ctx.user_provided  # the key fix: not attributed to the user


def test_real_user_text_is_user_provided():
    ctx = infer_biological_context(_adata(), text_context="human lung biopsy", hint_context="")
    assert ctx.tissue == "lung"
    assert ctx.provenance["tissue"] == "user_provided"
    assert ctx.user_provided.get("tissue") == "lung"


def test_user_text_wins_over_model_hint():
    ctx = infer_biological_context(_adata(), text_context="lung", hint_context="PBMC")
    assert ctx.tissue == "lung"
    assert ctx.provenance["tissue"] == "user_provided"


def test_tissue_not_guessed_from_annotation_biology():
    # A PBMC-like annotation composition with NO explicit tissue context must NOT
    # be laundered into tissue="PBMC" by a hardcoded cell-type biology vocabulary.
    # Tissue stays unknown; the model reads the annotation values and decides.
    a = _adata()
    a.obs["cell_type"] = pd.Categorical(
        ["T cell", "NK cell", "B cell", "monocyte", "dendritic cell", "T cell"]
    )
    ctx = infer_biological_context(a, text_context="", hint_context="")
    assert ctx.tissue == "unknown"
    assert ctx.provenance.get("tissue") in (None, "unknown")
    assert "marker_inferred" not in ctx.provenance.get("tissue", "")


# --- end-to-end through world_state -------------------------------------------

def test_world_state_attributes_model_hint_as_context_supplied():
    ws = AgentWorldState()
    ws.set_active_request("analyze /data/Reyfman_all_raw.h5ad")  # no tissue mentioned
    ws.add_context_hint("PBMC scRNA-seq")  # model/tool-supplied (default source)
    ws.sync_from_adata(_adata())
    bc = ws.data_summary["biological_context"]
    assert bc["tissue"] == "PBMC"
    assert bc["provenance"]["tissue"] == "context_supplied"
    assert "tissue" not in bc.get("user_provided", {})


def test_world_state_user_hint_is_user_provided():
    ws = AgentWorldState()
    ws.set_active_request("analyze this")
    ws.add_context_hint("lung fibrosis tissue", source="user")
    ws.sync_from_adata(_adata())
    bc = ws.data_summary["biological_context"]
    assert bc["tissue"] == "lung"
    assert bc["provenance"]["tissue"] == "user_provided"
