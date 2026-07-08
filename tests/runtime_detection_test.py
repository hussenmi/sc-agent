"""Regression tests for two runtime false-positives a reasoning model caught:

1. A near-unique `cell_barcode` column was tagged as a cell-type annotation
   (it shares the generic token "cell" with the cell_type role aliases).
2. Human data was read as `conflicting_marker_gene_evidence` for species,
   because the mouse-MHC `H2` pattern also matched human histone genes.
"""

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.analysis.context import _infer_species
from scagent.core.inspector import rank_obs_semantic_candidates


def _adata(obs: pd.DataFrame, var_names) -> AnnData:
    # Plain-list columns in `obs` avoid pandas index-alignment turning values to NaN.
    X = np.zeros((len(obs), len(var_names)), dtype=np.float32)
    return AnnData(X=X, obs=obs, var=pd.DataFrame(index=list(var_names)))


# --- Bug 1: barcode column must not be a cell_type candidate -----------------

def test_cell_barcode_not_detected_as_celltype():
    n = 200
    barcodes = [f"AAACCTG{i:04d}-1" for i in range(n)]
    barcodes[0] = barcodes[1]  # 199 unique of 200 → identifier-like, but < n_obs
    obs = pd.DataFrame(
        {
            "cell_barcode": barcodes,
            "donor": [f"Donor_{i % 8:02d}" for i in range(n)],
        },
        index=[f"c{i}" for i in range(n)],
    )
    obs["donor"] = obs["donor"].astype("category")
    ranked = rank_obs_semantic_candidates(_adata(obs, ["GENE1", "GENE2"]))
    celltype_cols = {c.column for c in ranked.get("cell_type", [])}
    assert "cell_barcode" not in celltype_cols


def test_real_celltype_column_still_detected():
    n = 200
    labels = ["T cell", "B cell", "NK cell", "Monocyte", "Macrophage"]
    obs = pd.DataFrame(
        {"cell_type": [labels[i % len(labels)] for i in range(n)]},
        index=[f"c{i}" for i in range(n)],
    )
    obs["cell_type"] = obs["cell_type"].astype("category")
    ranked = rank_obs_semantic_candidates(_adata(obs, ["GENE1", "GENE2"]))
    celltype_cols = {c.column for c in ranked.get("cell_type", [])}
    assert "cell_type" in celltype_cols  # the fix must not suppress genuine labels


# --- Bug 2: human histones must not collide with mouse MHC detection ---------

def _species_adata(genes):
    obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
    return _adata(obs, genes)


def test_human_histones_not_conflicting():
    # Human data: histone genes (H2A*/H2B*) plus HLA — must read as human, not conflicting.
    genes = ["H2AFZ", "H2AC6", "H2BC12", "HLA-A", "HLA-B", "HLA-DRA", "FAM138A", "OR4F5"]
    species, source, _ = _infer_species(_species_adata(genes))
    assert species == "human"
    assert source == "marker_gene_evidence"


def test_mouse_mhc_still_detected():
    # Mouse MHC genes are hyphenated and must still resolve to mouse.
    genes = ["H2-K1", "H2-D1", "H2-Aa", "H2-Ab1", "Gapdh", "Actb"]
    species, source, _ = _infer_species(_species_adata(genes))
    assert species == "mouse"
    assert source == "marker_gene_evidence"
