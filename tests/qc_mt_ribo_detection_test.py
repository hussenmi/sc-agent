"""MT/ribo QC detection must not depend on var_names being gene symbols.

run_2026_07_02_150701: the dataset's var_names were Ensembl IDs, so matching
'MT-'/'RPS'/'RPL' against var_names found ZERO genes → pct_counts_mt and
pct_counts_ribo were 0 for every cell (the flat, empty % mitochondrial / %
ribosomal panels in the per-cluster QC figure), which the model misread as
"snRNA-seq excludes MT/ribo". calculate_qc_metrics now matches prefixes against
the dataset's symbol column when var_names are not symbols, and case-insensitively
so mouse (mt-/Rps/Rpl) is caught too.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scagent.core.qc import _gene_names_for_prefix_matching, calculate_qc_metrics, filter_genes


# scanpy's calculate_qc_metrics uses percent_top defaults up to 500, so fixtures
# need a comfortable number of genes to avoid an unrelated IndexError.
def _pad_symbols(core, total=60):
    return list(core) + [f"FILLER{i}" for i in range(total - len(core))]


def _adata_ensembl_index_with_symbols():
    # var.index = Ensembl IDs; symbols (incl. MT- and RPS/RPL) live in feature_name.
    symbols = _pad_symbols(["CD3D", "MT-CO1", "MT-ND2", "RPS6", "RPL13", "GAPDH", "ACTB", "MT-CYB"])
    ensembl = [f"ENSG{str(i).zfill(11)}" for i in range(len(symbols))]
    var = pd.DataFrame({"feature_name": pd.Categorical(symbols)}, index=ensembl)
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.integers(1, 10, size=(20, len(symbols))).astype("float32"))
    a = ad.AnnData(X=X, var=var)
    return a


def test_mt_ribo_detected_via_symbol_column_when_varnames_are_ensembl():
    a = _adata_ensembl_index_with_symbols()
    # names used for matching are the symbols, not the Ensembl index
    names = _gene_names_for_prefix_matching(a)
    assert "MT-CO1" in names and "RPS6" in names
    calculate_qc_metrics(a, inplace=True)
    assert int(a.var["mt"].sum()) == 3        # MT-CO1, MT-ND2, MT-CYB
    assert int(a.var["ribo"].sum()) == 2      # RPS6, RPL13
    # the actual bug symptom: pct_counts_mt must not be uniformly zero
    assert float(a.obs["pct_counts_mt"].max()) > 0.0
    assert float(a.obs["pct_counts_ribo"].max()) > 0.0


def test_symbol_indexed_data_still_works():
    # var_names already symbols → matched directly, unchanged behavior.
    symbols = _pad_symbols(["CD3D", "MT-CO1", "RPS6", "GAPDH"])
    a = ad.AnnData(
        X=sp.csr_matrix(np.random.default_rng(0).integers(1, 10, size=(20, len(symbols))).astype("float32")),
        var=pd.DataFrame(index=symbols),
    )
    calculate_qc_metrics(a, inplace=True)
    assert int(a.var["mt"].sum()) == 1
    assert int(a.var["ribo"].sum()) == 1


def test_mouse_lowercase_mt_ribo_detected():
    # Mouse symbols are title-case (mt-Co1, Rps6); case-insensitive matching catches them.
    symbols = _pad_symbols(["Cd3d", "mt-Co1", "mt-Nd2", "Rps6", "Rpl13", "Actb"])
    a = ad.AnnData(
        X=sp.csr_matrix(np.random.default_rng(1).integers(1, 10, size=(20, len(symbols))).astype("float32")),
        var=pd.DataFrame(index=symbols),
    )
    calculate_qc_metrics(a, inplace=True)
    assert int(a.var["mt"].sum()) == 2
    assert int(a.var["ribo"].sum()) == 2


def test_filter_genes_removes_ribo_via_symbol_column():
    a = _adata_ensembl_index_with_symbols()
    n_before = a.n_vars
    filter_genes(a, min_cells=None, remove_ribo=True, remove_mt=False, inplace=True)
    assert a.n_vars == n_before - 2  # the 2 ribo genes removed despite Ensembl index
