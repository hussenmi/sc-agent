"""Gene-first batch investigation on controlled synthetic data.

We build two samples where the same two cell populations each split into a
per-sample cluster (a batch-like separation), plus a sample-wide "stress" program
that is elevated in S2 across BOTH populations, plus one genuinely sample-private
population and one mixed-but-enriched cluster. The tests assert the *procedure*:
enrichment without purity, cross-sample identity matching, direct-comparison gene
orientation, and recurrence — never specific biology.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scagent.analysis.batch_gene_investigation import (
    build_pair_narrative,
    find_sample_enriched_regions,
    run_gene_investigation,
    two_group_deg,
)
from scagent.batch.diffxpy import diffxpy_available


def _synthetic():
    """Return (adata, matrix, genes) with a known batch structure."""
    rng = np.random.default_rng(0)
    # genes: A-identity (0-4), B-identity (5-9), C-identity (10-14),
    # S2 stress program (15-19), noise (20-29)
    n_genes = 30
    A, B, C, STRESS = range(0, 5), range(5, 10), range(10, 15), range(15, 20)

    # Three samples. S1/S2 are the balanced majors (so a per-sample-pure cluster of
    # either is enriched ~2x, not near-baseline), and S3 is a rare minority present
    # only inside the mixed cluster cD — so cD is strongly S3-enriched despite being
    # far from pure, the case an 80%-purity gate would miss.
    blocks = [
        # (cluster, sample, n_cells, identity_genes)
        ("cA_S1", "S1", 150, A),
        ("cA_S2", "S2", 150, A),
        ("cB_S1", "S1", 150, B),
        ("cB_S2", "S2", 150, B),
        ("cC_S1", "S1", 120, C),   # sample-private population, no cross-sample twin
        ("cD",    "S1", 60, C),    # mixed cluster, part 1
        ("cD",    "S2", 60, C),    # mixed cluster, part 2
        ("cD",    "S3", 45, C),    # mixed cluster, part 3 -> cD is S3-enriched, not pure
    ]
    rows = []
    obs_cluster, obs_sample = [], []
    for cluster, sample, n, identity in blocks:
        counts = rng.poisson(0.5, size=(n, n_genes)).astype(np.float32)
        for g in identity:
            counts[:, g] += rng.poisson(6.0, size=n)
        if sample == "S2":  # sample-wide stress program in S2
            for g in STRESS:
                counts[:, g] += rng.poisson(5.0, size=n)
        rows.append(counts)
        obs_cluster += [cluster] * n
        obs_sample += [sample] * n
    counts = np.vstack(rows)
    # Log-normalize as the real diagnostic does; DEGs run on this expression matrix.
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)
    genes = [f"g{i}" for i in range(n_genes)]
    a = ad.AnnData(
        X=lognorm,
        obs=pd.DataFrame({"leiden": obs_cluster, "sample": obs_sample}),
        var=pd.DataFrame(index=genes),
    )
    return a, lognorm, genes


def test_enrichment_finds_regions_without_requiring_purity():
    a, X, genes = _synthetic()
    regions = find_sample_enriched_regions(a, "sample", "leiden", min_cells=30, min_enrichment=1.5)
    keys = {(r["cluster"], r["sample"]) for r in regions}
    # The pure per-sample clusters are found...
    assert ("cA_S2", "S2") in keys
    # ...and so is the mixed cluster cD's S3 side, which is enriched but NOT pure.
    cd_s3 = next(r for r in regions if r["cluster"] == "cD" and r["sample"] == "S3")
    assert cd_s3["frac_of_cluster"] < 0.8  # would have been missed by an 80% gate
    assert cd_s3["enrichment"] >= 1.5


@pytest.mark.parametrize("prefer_diffxpy", [False, True])
def test_run_gene_investigation_matches_populations_and_finds_recurrence(prefer_diffxpy):
    if prefer_diffxpy and not diffxpy_available():
        pytest.skip("diffxpy env not built on this host")
    a, X, genes = _synthetic()
    inv = run_gene_investigation(
        a, X, genes, batch_key="sample", cluster_key="leiden",
        prefer_diffxpy=prefer_diffxpy, min_cells=30, min_enrichment=1.5,
        min_shared_top25=3, identity_min_cells=20,
    )

    # cA_S1<->cA_S2 and cB_S1<->cB_S2 are confirmed same-population pairs.
    confirmed = {
        tuple(sorted([(p["cluster_a"], p["sample_a"]), (p["cluster_b"], p["sample_b"])]))
        for p in inv["selected_pairs"]
    }
    assert (("cA_S1", "S1"), ("cA_S2", "S2")) in confirmed
    assert (("cB_S1", "S1"), ("cB_S2", "S2")) in confirmed
    assert inv["direct_results"]

    # Direct DEG: the S2 side is higher for the injected stress program.
    stress = {f"g{i}" for i in range(15, 20)}
    for res in inv["direct_results"]:
        higher_s2 = res["higher_in_a"] if res["sample_a"] == "S2" else res["higher_in_b"]
        assert stress & set(higher_s2), "S2-elevated stress program not detected"

    # Recurrence is scoped to the S2 direction, >= 2 populations, with named pairs.
    recurrent_s2 = {r["gene"] for r in inv["recurrent_programs"] if r["associated_batch_group"] == "S2"}
    assert stress & recurrent_s2, "recurrence across populations not detected"
    for r in inv["recurrent_programs"]:
        assert r["n_populations"] >= 2
        assert r["direction"] == f"higher in {r['associated_batch_group']}"
        assert len(r["contributing_populations"]) == r["n_populations"]
        assert r["contributing_pairs"]


def test_requested_but_unavailable_diffxpy_falls_back_visibly(monkeypatch):
    # prefer_diffxpy=True but the env is unavailable -> the engine tag makes the
    # fallback explicit, and the DEG still produces valid results.
    import scagent.batch.diffxpy as dx
    monkeypatch.setattr(dx, "diffxpy_available", lambda: False)
    a, X, genes = _synthetic()
    cluster = a.obs["leiden"].to_numpy()
    target = np.flatnonzero(cluster == "cA_S1")
    reference = np.flatnonzero(cluster == "cB_S1")
    deg, engine = two_group_deg(X, genes, target, reference, prefer_diffxpy=True)
    assert engine == "scanpy_wilcoxon_diffxpy_unavailable"
    assert {"expression_effect", "mean_target", "pct_target"}.issubset(deg.columns)


def test_pair_narrative_recurrence_is_scoped_to_the_pairs_samples():
    # A G5/G4 direct pair must NOT claim recurrence that belongs to a G8 program.
    pair = {"cluster_a": "24", "sample_a": "G5", "cluster_b": "20", "sample_b": "G4",
            "n_shared_top25": 8, "shared_top25_genes": ["Sst", "Rbp4"]}
    direct = {"cluster_a": "24", "sample_a": "G5", "cluster_b": "20", "sample_b": "G4",
              "higher_in_a": ["Hspa8", "Cela1"], "higher_in_b": ["Gap43"]}
    identity = {("24", "G5"): None, ("20", "G4"): None}
    # Hspa8 recurs in G8 (an unrelated sample), not in G5 or G4.
    recurrent = [{"associated_batch_group": "G8", "gene": "Hspa8",
                  "direction": "higher in G8", "n_populations": 2,
                  "contributing_populations": ["7", "12"], "contributing_pairs": []}]
    lines = "\n".join(build_pair_narrative(pair, identity, direct, recurrent))
    # The pair legitimately shows Hspa8 in its direct comparison, but must NOT claim
    # it recurs (that recurrence belongs to G8, not G5/G4).
    assert "did not recur across other populations of the same samples" in lines
    assert "recur across this pair's own samples" not in lines


def test_two_group_deg_orientation_self_consistent():
    a, X, genes = _synthetic()
    cluster = a.obs["leiden"].to_numpy()
    target = np.flatnonzero(cluster == "cA_S1")
    reference = np.flatnonzero(cluster == "cB_S1")
    deg, engine = two_group_deg(X, genes, target, reference, prefer_diffxpy=False)
    assert engine == "scanpy_wilcoxon"
    # Orientation invariant: expression_effect sign == which side's mean is higher.
    for _, row in deg.iterrows():
        assert (row["expression_effect"] > 0) == (row["mean_target"] > row["mean_reference"])
