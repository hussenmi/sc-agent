"""Raw-counts resolution and the '__raw__' sentinel leak.

run_2026_07_02_150701 (misharin CELLxGENE h5ad): X was log-normalized and the
only raw counts lived in adata.raw (float32 but integer-valued). Two harness
failures followed:
  * inspection reported "raw counts in layer '__raw__'" — a layer that does not
    exist — sending the model chasing it for ~5 iterations, and
  * normalize_and_hvg could only reset from a *named* layer, so the model had to
    hand-copy adata.raw.X into layers['raw_counts'].

These tests pin the general fixes: adata.raw is never reported as a fake layer,
its integer-ness is surfaced as a fact, and normalize_and_hvg materializes counts
from adata.raw automatically.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scagent.agent.tools import process_tool_call
from scagent.core.inspector import (
    _detect_raw_layer,
    _is_integer_matrix,
    _x_facts,
    dataset_facts,
    inspect_data,
    summarize_state,
)


def _lognorm_with_raw(n_obs=200, n_vars=50, raw_extra_genes=0, raw_integer=True):
    """X log-normalized; counts (or non-counts) only in adata.raw."""
    rng = np.random.default_rng(0)
    counts = sp.csr_matrix(rng.integers(0, 8, size=(n_obs, n_vars)).astype("float32"))
    lognorm = counts.copy()
    lognorm.data = np.log1p(lognorm.data).astype("float32")
    a = ad.AnnData(X=lognorm)
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.uns["log1p"] = {"base": None}

    n_raw = n_vars + raw_extra_genes
    raw_counts = sp.csr_matrix(rng.integers(0, 8, size=(n_obs, n_raw)).astype("float32"))
    if not raw_integer:
        raw_counts.data = raw_counts.data + 0.3  # decimals -> not counts
    raw = ad.AnnData(X=raw_counts)
    raw.var_names = [f"G{i}" for i in range(n_raw)]
    a.raw = raw
    return a


# --- __raw__ sentinel is gone -------------------------------------------------
def test_detect_raw_layer_ignores_adata_raw():
    a = _lognorm_with_raw()
    has_layer, name = _detect_raw_layer(a)
    assert has_layer is False and name == ""  # adata.raw is not a "layer"


def test_inspect_reports_raw_as_adata_raw_not_layer():
    a = _lognorm_with_raw()
    st = inspect_data(a)
    assert st.has_raw is True
    assert st.raw_is_counts is True
    assert st.has_raw_layer is False
    assert st.raw_layer_name == ""
    summary = summarize_state(st)
    assert "__raw__" not in summary
    assert "raw counts in adata.raw" in summary


def test_non_integer_raw_reported_as_not_counts():
    a = _lognorm_with_raw(raw_integer=False)
    st = inspect_data(a)
    assert st.has_raw is True
    assert st.raw_is_counts is False
    assert "not raw counts" in summarize_state(st)


# --- facts carry the decimal-point evidence -----------------------------------
def test_dataset_facts_raw_integer_evidence():
    a = _lognorm_with_raw()
    facts = dataset_facts(a)
    assert facts["raw"]["present"] is True
    # the model can see adata.raw.X is integer-valued from actual values
    assert facts["raw"]["X"]["all_integer_sample"] is True
    assert facts["raw"]["X"]["has_negative_sample"] is False
    # X itself is NOT integer (log-normalized)
    assert facts["X"]["all_integer_sample"] is False


def test_dataset_facts_layer_facts_present():
    a = _lognorm_with_raw()
    a.layers["counts"] = a.raw.X[:, : a.n_vars].copy()
    facts = dataset_facts(a)
    assert "counts" in facts["layer_facts"]
    assert facts["layer_facts"]["counts"]["all_integer_sample"] is True


# --- value-based, NOT dtype-based (the core of the fix) -----------------------
def test_float_dtype_but_integer_valued_is_counts():
    # float32 matrix whose values are all .0 (1.0, 20.0, 5643.0) IS raw counts.
    X = sp.csr_matrix(np.array([[1.0, 20.0, 3.0, 5643.0]], dtype="float32"))
    assert _is_integer_matrix(X) is True
    facts = _x_facts(X)
    assert facts["dtype"] == "float32"
    assert facts["fraction_integer_valued"] == 1.0
    assert facts["has_negative_sample"] is False


def test_fraction_integer_valued_surfaced_for_near_integer():
    # SoupX/CellBender-style corrected counts: mostly integer, some decimals.
    # The mechanical check is strict (not all integer), but the FACT exposes the
    # fraction so the model can still reason "near-integer corrected counts".
    X = sp.csr_matrix(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.3]], dtype="float32"))
    assert _is_integer_matrix(X) is False
    facts = _x_facts(X)
    assert 0.85 <= facts["fraction_integer_valued"] < 1.0
    assert facts["all_integer_sample"] is False


def test_facts_report_fraction_for_raw():
    a = _lognorm_with_raw()
    facts = dataset_facts(a)
    assert facts["raw"]["X"]["fraction_integer_valued"] == 1.0
    assert facts["X"]["fraction_integer_valued"] == 0.0


# --- normalize_and_hvg pulls counts from adata.raw ----------------------------
def test_normalize_materializes_counts_from_adata_raw():
    a = _lognorm_with_raw()
    res, _ = process_tool_call("normalize_and_hvg", {"n_hvg": 20, "normalization_source": "auto"}, a)
    d = json.loads(res)
    assert d["status"] == "ok"
    assert d["raw_counts_source_note"] and "adata.raw" in d["raw_counts_source_note"]
    assert d["reset_from_raw_counts"] is True
    assert d["resolved_source"] == "raw_counts"
    assert d["raw_counts_present"] and d["raw_counts_integer_like"]


def test_normalize_aligns_raw_superset_genes():
    # adata.raw carries more genes than adata.X (typical post-HVG); the counts
    # layer must be aligned to the current var_names, not fail.
    a = _lognorm_with_raw(n_vars=30, raw_extra_genes=20)
    assert a.raw.n_vars == 50 and a.n_vars == 30
    res, updated = process_tool_call("normalize_and_hvg", {"n_hvg": 15, "normalization_source": "auto"}, a)
    d = json.loads(res)
    assert d["status"] == "ok"
    assert "aligned 30 genes" in (d["raw_counts_source_note"] or "")
