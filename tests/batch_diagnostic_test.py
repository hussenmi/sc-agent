"""Gene-first batch diagnostic — verdict derivation on controlled scenarios.

The synthetic builder injects a known batch structure so we can assert the
two-axis verdict (gene evidence x design) rather than any specific biology.
Runs on the in-env Wilcoxon path (prefer_diffxpy=False) so it is fast and
deterministic in CI; one test exercises the real diffxpy path when available.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scagent.analysis.batch_diagnostic import diagnose_batch_effect
from scagent.batch.diffxpy import diffxpy_available

A, B, C, STRESS = range(0, 5), range(5, 10), range(10, 15), range(15, 20)
N_GENES = 30


def _lognorm(counts: np.ndarray) -> np.ndarray:
    return np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)


def _build(blocks, *, add_condition=False, seed=0):
    """blocks: list of (cluster, sample, n_cells, identity_range, stress_bool)."""
    rng = np.random.default_rng(seed)
    rows, cl, sm = [], [], []
    for cluster, sample, n, identity, stress in blocks:
        ct = rng.poisson(0.5, size=(n, N_GENES)).astype(np.float32)
        for g in identity:
            ct[:, g] += rng.poisson(6.0, size=n)
        if stress:
            for g in STRESS:
                ct[:, g] += rng.poisson(5.0, size=n)
        rows.append(ct)
        cl += [cluster] * n
        sm += [sample] * n
    X = _lognorm(np.vstack(rows))
    obs = pd.DataFrame({"leiden": cl, "sample": sm})
    if add_condition:
        # Condition perfectly tracks sample -> confounded with batch.
        obs["condition"] = ["treated" if s == "S2" else "control" for s in sm]
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)]))
    a.obsm["X_pca"] = np.asarray(X)[:, :10]
    return a, X


# Two populations each split S1/S2, with an S2 stress program in both -> recurrence.
RECURRING_BLOCKS = [
    ("cA_S1", "S1", 150, A, False), ("cA_S2", "S2", 150, A, True),
    ("cB_S1", "S1", 150, B, False), ("cB_S2", "S2", 150, B, True),
    ("cC_S1", "S1", 120, C, False),
]


def test_recurring_no_design_cannot_determine(tmp_path):
    a, _ = _build(RECURRING_BLOCKS)
    r = diagnose_batch_effect(
        a, batch_key="sample", cluster_key="leiden",
        min_cells_per_cluster_sample=30, min_enrichment=1.5,
        prefer_diffxpy=False, output_dir=str(tmp_path),
    )
    assert r["status"] == "ok"
    assert r["gene_evidence"] == "recurring_sample_associated"
    assert r["design_interpretation"] == "unknown"
    assert r["recommendation"] == "cannot_determine_technical_vs_biological"
    # A recurring program is reported, scoped to a sample with named pairs.
    assert r["recurrent_programs"]
    prog = r["recurrent_programs"][0]
    assert prog["n_populations"] >= 2 and prog["contributing_pairs"]
    # All five tables + README exist; design_check status is 'unknown' (no metadata).
    for name in [
        "batch_diagnostic_sample_enriched_regions", "batch_diagnostic_within_sample_degs",
        "batch_diagnostic_population_pairs", "batch_diagnostic_direct_pair_degs",
        "batch_diagnostic_design_check",
    ]:
        assert (tmp_path / f"{name}.csv").exists()
    design = pd.read_csv(tmp_path / "batch_diagnostic_design_check.csv")
    assert design.iloc[0]["status"] == "unknown"
    # The README's Interpretation is deterministic prose, not a "Pending" placeholder.
    from scagent.core import artifact_docs as _ad
    readme_text = (tmp_path / "README.md").read_text()
    assert not _ad.interpretation_is_empty(readme_text)
    assert "Pending" not in readme_text.split("## Interpretation")[1]
    assert "This diagnostic examined" in readme_text


def test_recurring_confounded_condition_cannot_determine(tmp_path):
    a, _ = _build(RECURRING_BLOCKS, add_condition=True)
    r = diagnose_batch_effect(
        a, batch_key="sample", cluster_key="leiden",
        condition_keys=["condition"], min_cells_per_cluster_sample=30,
        min_enrichment=1.5, prefer_diffxpy=False, output_dir=str(tmp_path),
    )
    assert r["gene_evidence"] == "recurring_sample_associated"
    assert r["design_interpretation"] == "confounded_with_biology"
    assert r["recommendation"] == "cannot_determine_technical_vs_biological"
    design = pd.read_csv(tmp_path / "batch_diagnostic_design_check.csv")
    assert design.iloc[0]["status"] == "confounded"


def test_technical_batch_keys_must_name_the_batch_key():
    a, _ = _build(RECURRING_BLOCKS)
    # A non-empty list that does NOT include the batch_key does not count as a
    # documented technical batch -> stays unknown -> not integration_supported.
    r_other = diagnose_batch_effect(
        a, batch_key="sample", cluster_key="leiden", min_cells_per_cluster_sample=30,
        min_enrichment=1.5, prefer_diffxpy=False, technical_batch_keys=["some_other_key"],
    )
    assert r_other["design_interpretation"] == "unknown"
    assert r_other["recommendation"] == "cannot_determine_technical_vs_biological"
    # Naming the batch_key itself documents it as technical -> integration_supported.
    r_named = diagnose_batch_effect(
        a, batch_key="sample", cluster_key="leiden", min_cells_per_cluster_sample=30,
        min_enrichment=1.5, prefer_diffxpy=False, technical_batch_keys=["sample"],
    )
    assert r_named["design_interpretation"] == "documented_technical_batch"
    assert r_named["recommendation"] == "integration_supported"


def test_localized_effect_do_not_integrate(tmp_path):
    # Only ONE population splits across samples -> a pair but no recurrence.
    blocks = [
        ("cA_S1", "S1", 150, A, False), ("cA_S2", "S2", 150, A, True),
        ("cB", "S1", 150, B, False), ("cB", "S2", 150, B, False),  # one mixed cluster
    ]
    a, _ = _build(blocks)
    r = diagnose_batch_effect(
        a, batch_key="sample", cluster_key="leiden",
        min_cells_per_cluster_sample=30, min_enrichment=1.5,
        prefer_diffxpy=False, output_dir=str(tmp_path),
    )
    assert r["gene_evidence"] == "localized"
    assert r["recommendation"] == "do_not_integrate_based_on_current_evidence"
    assert not r["recurrent_programs"]


def test_no_matchable_pairs_do_not_integrate(tmp_path):
    # Each sample's clusters carry DIFFERENT identities -> no cross-sample match.
    blocks = [
        ("cA_S1", "S1", 150, A, False),
        ("cB_S2", "S2", 150, B, False),
        ("cC_S1", "S1", 150, C, False),
    ]
    a, _ = _build(blocks)
    r = diagnose_batch_effect(
        a, batch_key="sample", cluster_key="leiden",
        min_cells_per_cluster_sample=30, min_enrichment=1.5,
        prefer_diffxpy=False, output_dir=str(tmp_path),
    )
    assert r["gene_evidence"] == "none"
    assert r["recommendation"] == "do_not_integrate_based_on_current_evidence"
    # Must NOT claim sample-associated differences were found.
    assert "did not find gene-level support" in r["recommendation_reason"]


def test_visible_wilcoxon_fallback_is_recorded(tmp_path):
    a, _ = _build(RECURRING_BLOCKS)
    r = diagnose_batch_effect(
        a, batch_key="sample", cluster_key="leiden",
        min_cells_per_cluster_sample=30, min_enrichment=1.5,
        prefer_diffxpy=False, output_dir=str(tmp_path),
    )
    # prefer_diffxpy=False forces the in-env engine; it must be visibly recorded.
    assert "scanpy_wilcoxon" in r["de_engines_used"]
    degs = pd.read_csv(tmp_path / "batch_diagnostic_within_sample_degs.csv")
    assert (degs["de_engine"] == "scanpy").all()


def test_real_diffxpy_engine_when_available(tmp_path):
    if not diffxpy_available():
        import pytest
        pytest.skip("diffxpy env not built on this host")
    a, _ = _build(RECURRING_BLOCKS)
    r = diagnose_batch_effect(
        a, batch_key="sample", cluster_key="leiden",
        min_cells_per_cluster_sample=30, min_enrichment=1.5,
        prefer_diffxpy=True, output_dir=str(tmp_path),
    )
    assert any(e.startswith("diffxpy_") for e in r["de_engines_used"])
    assert r["matrix_source"] == "lognorm"
