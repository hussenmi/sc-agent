"""Cluster QC now adjudicates each problematic cluster on THREE independent axes —
metric quality, a one-vs-rest DEG identity check, and gene-gene covariance
coherence — and AUTO-APPLIES removal only when all three agree a cluster is junk
(DEG markers are technical/generic/too-few AND covariance is unstructured AND
metric QC flagged it). A cluster whose covariance looks coherent, or that has a
real identity program, is flagged and KEPT.
"""

import json

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.agent.tools import _classify_cluster_deg, process_tool_call


# ---------------------------------------------------------------------------
# Unit: the DEG marker classifier
# ---------------------------------------------------------------------------
def test_classifier_flags_mt_dominated_markers():
    markers = [
        {"gene": f"MT-{i}", "logfc": 5.0, "padj": 1e-6} for i in range(6)
    ] + [{"gene": "ACTB", "logfc": 3.0, "padj": 1e-4}]
    out = _classify_cluster_deg(markers)
    assert out["deg_verdict"] == "junk_markers"
    assert out["mt_fraction"] >= 0.4


def test_classifier_flags_too_few_markers():
    markers = [{"gene": "REALGENE1", "logfc": 4.0, "padj": 1e-5},
               {"gene": "REALGENE2", "logfc": 3.5, "padj": 1e-4}]
    out = _classify_cluster_deg(markers)
    # Only two significant markers -> too few to define a cluster.
    assert out["deg_verdict"] == "junk_markers"
    assert any("too few" in r or "few significant" in r for r in out["junk_reasons"])


def test_classifier_supports_real_identity():
    markers = [
        {"gene": g, "logfc": 4.0, "padj": 1e-6}
        for g in ("CD3D", "CD3E", "CD8A", "IL7R", "TRAC", "GZMK")
    ]
    out = _classify_cluster_deg(markers)
    assert out["deg_verdict"] == "identity_supported"
    assert out["n_specific_markers"] >= 5


# ---------------------------------------------------------------------------
# Integration: run_cluster_qc removes the confirmed-junk cluster, keeps the rest
# ---------------------------------------------------------------------------
def _build_qc_adata():
    """3 real clusters (A/B/C), each defined by a co-varying block of specific
    genes (strong DEG + structured covariance), plus one junk cluster: low
    library / few genes / high MT, MT-dominated markers, unstructured covariance.
    """
    # Junk kept well under the 20% auto-apply cap, but large enough that its
    # independent (unstructured) gene expression doesn't spuriously read as
    # structured from small-sample correlation noise.
    rng = np.random.RandomState(0)
    n_mt, n_spec = 10, 60
    G = n_mt + n_spec
    sizes = {"A": 140, "B": 140, "C": 140, "junk": 70}
    labels = [k for k, v in sizes.items() for _ in range(v)]
    N = len(labels)
    labels_arr = np.array(labels)

    X = np.zeros((N, G), dtype=np.float32)
    idx = {k: np.where(labels_arr == k)[0] for k in sizes}
    blocks = {
        "A": range(n_mt, n_mt + 20),
        "B": range(n_mt + 20, n_mt + 40),
        "C": range(n_mt + 40, n_mt + 60),
    }
    # Real clusters: a co-varying (shared-latent) block of specific genes.
    for k, block in blocks.items():
        cells = idx[k]
        latent = rng.normal(3.0, 0.4, size=len(cells))
        for g in block:
            X[cells, g] = np.clip(
                latent * rng.uniform(0.8, 1.2) + rng.normal(0, 0.15, len(cells)), 0, None
            )
    # Junk cluster: MT genes high (its only real up-markers vs rest), and a low,
    # independent (unstructured) smear across all specific genes.
    jc = idx["junk"]
    X[np.ix_(jc, list(range(n_mt)))] = np.abs(rng.normal(4.0, 0.5, (len(jc), n_mt)))
    X[np.ix_(jc, list(range(n_mt, G)))] = np.abs(rng.normal(0.3, 0.2, (len(jc), n_spec)))
    X = np.log1p(X).astype(np.float32)

    var = pd.DataFrame(index=[f"MT-{i}" for i in range(n_mt)] + [f"g{i}" for i in range(n_spec)])
    obs = pd.DataFrame(index=[f"c{i}" for i in range(N)])
    obs["leiden"] = pd.Categorical(labels)
    is_junk = labels_arr == "junk"
    obs["total_counts"] = np.where(is_junk, rng.normal(800, 40, N), rng.normal(5000, 200, N))
    obs["n_genes_by_counts"] = np.where(is_junk, rng.normal(300, 25, N), rng.normal(2000, 80, N))
    obs["pct_counts_mt"] = np.clip(np.where(is_junk, rng.normal(40, 2, N), rng.normal(5, 0.5, N)), 0, None)
    return AnnData(X=X, obs=obs, var=var)


def test_confirmed_junk_cluster_is_auto_removed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    adata = _build_qc_adata()
    rj, updated = process_tool_call(
        "run_cluster_qc",
        {"cluster_key": "leiden", "save_checkpoint": False},
        adata,
    )
    r = json.loads(rj)
    assert r["status"] == "ok"

    # The junk cluster is metric-flagged, DEG-junk, and unstructured -> removed.
    assert "junk" in r["auto_removed_clusters"]
    assert r["cells_auto_removed"] == 70
    assert updated.n_obs == 420  # 70 junk cells dropped
    assert "junk" not in set(updated.obs["leiden"].astype(str))

    # The real clusters survive.
    for c in ("A", "B", "C"):
        assert c not in r["auto_removed_clusters"]

    # The decision table carries the per-axis "why" for each cluster.
    sq = r["structure_qc"]
    table = {row["cluster"]: row for row in sq["qc_decision_table"]}
    assert table["junk"]["deg_verdict"] == "junk_markers"
    assert table["junk"]["decision"] == "remove"
    assert table["junk"]["covariance"] in ("unstructured", "weak")
    for c in ("A", "B", "C"):
        assert table[c]["deg_verdict"] == "identity_supported"
        assert table[c]["decision"] in ("keep", "review")


def test_auto_removal_re_arms_the_qc_obligation(monkeypatch, tmp_path):
    """Removing cells invalidates the embedding AND the clustering the QC ran on —
    the surviving labels were computed on the pre-removal cells. The registry must
    NOT read as fresh against the post-removal cell set, or the run could proceed
    to annotation on a stale clustering. The obligation re-fires until recluster.
    """
    from scagent.agent.world_state import AgentWorldState

    monkeypatch.chdir(tmp_path)
    ws = AgentWorldState()
    adata = _build_qc_adata()
    ws.apply_tool_result("run_clustering", {"status": "ok", "cluster_key": "leiden"}, adata=adata)
    ws.sync_from_adata(adata)

    res, adata = process_tool_call(
        "run_cluster_qc", {"cluster_key": "leiden", "save_checkpoint": False}, adata, world_state=ws
    )
    r = json.loads(res)
    ws.apply_tool_result("run_cluster_qc", r, adata=adata)
    assert r["cells_auto_removed"] == 70

    # Cells were dropped -> QC is NOT satisfied; a recluster is demanded.
    assert "cluster_qc" in [o["key"] for o in ws.unmet_obligations()]
    summary = ws.data_summary["cluster_qc"]
    assert summary["status"] == "needed"
    assert "stale" in summary["reason"]
    # ...and it stays unmet across a resync (not a transient blip).
    ws.sync_from_adata(adata)
    assert "cluster_qc" in [o["key"] for o in ws.unmet_obligations()]

    # After a genuine recluster + a clean QC pass, the obligation clears.
    adata.obs["leiden"] = pd.Categorical(
        np.random.RandomState(1).choice(["0", "1", "2"], adata.n_obs)
    )
    ws.apply_tool_result("run_clustering", {"status": "ok", "cluster_key": "leiden"}, adata=adata)
    res2, adata = process_tool_call(
        "run_cluster_qc", {"cluster_key": "leiden", "save_checkpoint": False}, adata, world_state=ws
    )
    ws.apply_tool_result("run_cluster_qc", json.loads(res2), adata=adata)
    assert "cluster_qc" not in [o["key"] for o in ws.unmet_obligations()]


def test_auto_removal_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    adata = _build_qc_adata()
    rj, updated = process_tool_call(
        "run_cluster_qc",
        {"cluster_key": "leiden", "save_checkpoint": False, "auto_apply_removal": False},
        adata,
    )
    r = json.loads(rj)
    assert r["status"] == "ok"
    # Proposed but NOT applied.
    assert r["auto_removed_clusters"] == []
    assert updated.n_obs == 490
    assert "junk" in (r["structure_qc"].get("synthesized_removal") or [])
