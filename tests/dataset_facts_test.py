"""The facts layer must be comprehensive and judgment-free: it reports everything
observable (cardinality, missingness, distributions, X characteristics, gene
namespace) but makes no role/species/'is this cell types' decision — those live
in the judgment layer that consumes this sheet."""

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.core.inspector import _column_facts, dataset_facts


def _adata():
    n, g = 40, 6
    X = (np.arange(n * g) % 5).reshape(n, g).astype(np.int32)  # non-negative integer counts
    barcodes = [f"AAACCTG{i:04d}-1" for i in range(n)]
    obs = pd.DataFrame(
        {
            "cell_barcode": barcodes,  # unique per cell → identifier
            "donor": [f"Donor_{i % 8:02d}" for i in range(n)],  # 8 balanced groups
        },
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=["MT-ND1", "HLA-A", "H2AFZ", "FAM138A", "OR4F5", "RP11-34P13.3"])
    return AnnData(X=X, obs=obs, var=var)


def test_column_facts_reports_cardinality_and_distribution():
    ad = _adata()
    bc = _column_facts(ad.obs["cell_barcode"], ad.n_obs)
    assert bc["n_unique"] == 40
    assert bc["unique_fraction"] == 1.0  # the signal that says "identifier, not label"

    donor = _column_facts(ad.obs["donor"], ad.n_obs)
    assert donor["n_unique"] == 8
    assert donor["unique_fraction"] == 0.2
    # Low-cardinality columns report a value-count distribution.
    assert "top_values" in donor
    assert all("value" in tv and "count" in tv for tv in donor["top_values"])
    assert sum(tv["count"] for tv in donor["top_values"]) == 40


def test_dataset_facts_is_comprehensive():
    facts = dataset_facts(_adata())
    assert facts["shape"] == {"n_obs": 40, "n_vars": 6}
    # X characterized factually.
    assert facts["X"]["all_integer_sample"] is True
    assert facts["X"]["has_negative_sample"] is False
    # Both axes' columns are covered.
    assert set(facts["obs_columns"]) == {"cell_barcode", "donor"}
    assert "var_columns" in facts
    # Gene-namespace signals are raw counts, not a species call.
    gn = facts["gene_namespace"]
    assert gn["mt_prefixed"] == 1  # MT-ND1
    assert gn["n_checked"] == 6


def test_dataset_facts_has_no_judgment_fields():
    # The whole point: facts contain no interpretation.
    facts = dataset_facts(_adata())
    blob = repr(facts).lower()
    for judgment_key in ("cell_type_key", "recommended_batch_key", "species", "role", "confidence"):
        assert judgment_key not in facts
    # and no role-scored verdicts leaked into the structure
    assert "semantic_obs_roles" not in facts
    assert "species" not in blob.replace("var_names_examples", "")  # sanity: no species verdict
