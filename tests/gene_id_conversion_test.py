"""Robust gene-identifier handling.

Regression cover for run_2026_07_02_120046 (CELLxGENE lung h5ad): var.index was
Ensembl IDs with symbols in var['feature_name'], and raw counts lived in
adata.raw stored as float32-but-integer-valued. That combination made SCimilarity
report "Gene overlap of 0" and made counts detection think no raw was available.

These tests pin the general fixes, not the one dataset:
  * ID-format classification and offline Ensembl->symbol conversion via any
    content-validated symbol column (not a hardcoded name list),
  * counts resolution from adata.raw / layers on integer VALUES not dtype,
  * inspector reporting of the resolvable symbol column,
  * the convert_gene_ids agent tool.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scagent.core import genes
from scagent.core.inspector import find_counts_matrix, _characterize_features
from scagent.agent.tools import get_tools, process_tool_call


# --- fixtures ------------------------------------------------------------------
def _cellxgene_like(n_obs=8, with_raw=True):
    """h5ad shaped like the failing run: Ensembl var.index + feature_name symbols,
    log-normalized X, integer-valued float32 counts in adata.raw."""
    rng = np.random.default_rng(0)
    ensembl = [f"ENSG{str(i).zfill(11)}" for i in range(6)]
    symbols = ["CD3D", "CD4", "MS4A1", "LYZ", "ENSG00000999999", "GAPDH"]
    # index 4 has no HGNC symbol -> feature_name keeps the Ensembl id
    var = pd.DataFrame({"feature_name": pd.Categorical(symbols)}, index=ensembl)

    counts = sp.csr_matrix(rng.integers(0, 5, size=(n_obs, 6)).astype("float32"))
    lognorm = counts.copy()
    lognorm.data = np.log1p(lognorm.data).astype("float32")  # non-integer

    a = ad.AnnData(X=lognorm, var=var.copy())
    a.uns["log1p"] = {"base": None}
    if with_raw:
        a.raw = ad.AnnData(X=counts.copy(), var=var.copy())
    return a


# --- classification ------------------------------------------------------------
def test_classify_and_infer_formats():
    assert genes.classify_gene_id("ENSG00000000003") == "ensembl"
    assert genes.classify_gene_id("ENSMUSG00000000001") == "ensembl"
    assert genes.classify_gene_id("ENSG00000000003.14") == "ensembl"  # versioned
    assert genes.classify_gene_id("7157") == "entrez"
    assert genes.classify_gene_id("CD3D") == "symbol"
    assert genes.infer_id_format(["ENSG00000000003", "ENSG00000000005"]) == "ensembl"
    assert genes.infer_id_format(["CD3D", "CD4", "MS4A1"]) == "symbol"
    assert genes.infer_id_format(["7157", "1234"]) == "entrez"


def test_strip_ensembl_version_and_genome_prefix():
    assert genes.strip_ensembl_version("ENSG00000000003.14") == "ENSG00000000003"
    assert genes.strip_ensembl_version("CD3D") == "CD3D"
    names = ["GRCh38_CD3D", "GRCh38_CD4", "GRCh38_MS4A1", "GRCh38_LYZ"]
    stripped, prefix = genes.strip_genome_prefix(names)
    assert prefix == "GRCh38_"
    assert stripped == ["CD3D", "CD4", "MS4A1", "LYZ"]
    # symbols already -> prefix inferred as symbol format
    assert genes.infer_id_format(names) == "symbol"


# --- column detection ----------------------------------------------------------
def test_find_symbol_column_content_validated():
    a = _cellxgene_like()
    assert genes.find_symbol_column(a) == "feature_name"
    # oddly-named column is still found by content
    a.var = a.var.rename(columns={"feature_name": "totally_custom_name"})
    assert genes.find_symbol_column(a) == "totally_custom_name"


def test_symbol_column_rejects_mislabeled_column():
    # a column literally named 'gene_symbols' but holding Ensembl IDs must NOT
    # be trusted as symbols (content validation, not name matching).
    ensembl = [f"ENSG{str(i).zfill(11)}" for i in range(6)]
    var = pd.DataFrame({"gene_symbols": ensembl}, index=ensembl)
    a = ad.AnnData(X=sp.csr_matrix((4, 6), dtype="float32"), var=var)
    assert genes.find_symbol_column(a) is None


# --- conversion ----------------------------------------------------------------
def test_convert_ensembl_to_symbols_via_column():
    a = _cellxgene_like()
    conv, rep = genes.convert_var_to_symbols(a, inplace=False)
    # input untouched (copy semantics)
    assert genes.infer_id_format(a.var_names) == "ensembl"
    assert rep.changed and rep.to_format == "symbol"
    assert rep.source == "column:feature_name"
    assert list(conv.var_names[:4]) == ["CD3D", "CD4", "MS4A1", "LYZ"]
    # the gene with no HGNC symbol keeps its Ensembl id (not dropped)
    assert "ENSG00000999999" in list(conv.var_names)
    assert rep.n_mapped == 5 and rep.n_unmapped == 1
    # original ids preserved
    assert rep.original_ids_saved_to == "ensembl_id"
    assert list(conv.var["ensembl_id"][:1]) == ["ENSG00000000000"]


def test_convert_is_noop_for_symbols():
    a = _cellxgene_like()
    a.var_names = ["CD3D", "CD4", "MS4A1", "LYZ", "NKG7", "GAPDH"]
    conv, rep = genes.convert_var_to_symbols(a, inplace=False)
    assert rep.changed is False
    assert rep.source == "already_symbols"
    assert list(conv.var_names) == list(a.var_names)


def test_convert_makes_duplicates_unique_without_dropping():
    ensembl = [f"ENSG{str(i).zfill(11)}" for i in range(4)]
    var = pd.DataFrame({"feature_name": ["CD3D", "CD3D", "CD4", "CD4"]}, index=ensembl)
    a = ad.AnnData(X=sp.csr_matrix((3, 4), dtype="float32"), var=var)
    conv, rep = genes.convert_var_to_symbols(a, inplace=False)
    assert conv.n_vars == 4  # no genes dropped
    assert conv.var_names.is_unique
    assert rep.n_duplicates_made_unique == 2


def test_convert_no_symbol_source_leaves_names():
    ensembl = [f"ENSG{str(i).zfill(11)}" for i in range(4)]
    a = ad.AnnData(X=sp.csr_matrix((3, 4), dtype="float32"),
                   var=pd.DataFrame(index=ensembl))
    conv, rep = genes.convert_var_to_symbols(a, inplace=False, use_mygene=False)
    assert rep.changed is False
    assert rep.source == "none"
    assert list(conv.var_names) == ensembl


# --- counts resolver -----------------------------------------------------------
def test_find_counts_matrix_reads_raw_integer_float():
    a = _cellxgene_like(with_raw=True)
    res = find_counts_matrix(a)
    assert res is not None
    assert res["source"] == "raw"
    # var returned matches the counts matrix (raw.var), carrying feature_name
    assert "feature_name" in res["var"].columns


def test_find_counts_matrix_prefers_layer():
    a = _cellxgene_like(with_raw=True)
    a.layers["counts"] = a.raw.X.copy()
    res = find_counts_matrix(a)
    assert res["source"] == "layer:counts"


def test_find_counts_matrix_none_when_only_lognorm():
    a = _cellxgene_like(with_raw=False)  # X is log-normalized, no raw, no layers
    assert find_counts_matrix(a) is None


# --- inspector reporting -------------------------------------------------------
def test_inspector_reports_symbol_column_and_convertibility():
    a = _cellxgene_like()
    info = _characterize_features(a)
    assert info["gene_id_format"] == "ensembl"
    assert info["symbol_column"] == "feature_name"
    assert info["convertible_to_symbols"] is True
    assert info["has_gene_symbols"] is True


# --- agent tool ----------------------------------------------------------------
def test_convert_gene_ids_tool_registered():
    names = {t["name"] for t in get_tools()}
    assert "convert_gene_ids" in names


def test_convert_gene_ids_tool_converts_in_place():
    a = _cellxgene_like()
    res, updated = process_tool_call("convert_gene_ids", {}, a)
    d = json.loads(res)
    assert d["status"] == "ok"
    assert d["changed"] is True
    assert d["before_format"] == "ensembl"
    assert d["after_format"] == "symbol"
    assert d["conversion"]["symbol_column"] == "feature_name"
    # the returned (persisted) adata now uses symbols
    assert "CD3D" in list(updated.var_names)
