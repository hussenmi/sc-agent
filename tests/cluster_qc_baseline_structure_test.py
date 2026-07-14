"""When doublet detection was skipped, cluster QC can't flag doublet-enriched
clusters — and structure QC (gated on metric flags) would never run, leaving a
metrically-normal-but-incoherent cluster with no detection path. run_cluster_qc
must then nominate a baseline structure-QC pass over all clusters.
"""

import json

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.agent.tools import process_tool_call


def _clean_clustered_adata(with_doublets: bool):
    n, k, g = 120, 3, 60
    rng = np.random.RandomState(0)
    obs = pd.DataFrame(
        {
            # All clusters near the global median -> nothing metric-flagged.
            "total_counts": rng.normal(5000, 200, n),
            "n_genes_by_counts": rng.normal(2000, 80, n),
            "pct_counts_mt": rng.normal(5, 0.5, n),
            "leiden": pd.Categorical([str(i % k) for i in range(n)]),
        },
        index=[f"c{i}" for i in range(n)],
    )
    if with_doublets:
        obs["doublet_score"] = rng.uniform(0.02, 0.08, n)  # low -> no doublet flags
    # Enough genes/values for the auto-chained structure QC to run.
    X = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    a = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    return a


def _run(adata, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # any incidental figure write lands in tmp
    rj, _ = process_tool_call(
        "run_cluster_qc",
        {"cluster_key": "leiden", "save_checkpoint": False},
        adata,
    )
    return json.loads(rj)


def test_baseline_runs_even_when_doublets_present_and_clean(monkeypatch, tmp_path):
    # Structure QC must ALWAYS run: metric-clean is not "coherent". Even with
    # doublet scores present and nothing metric-flagged, a baseline coherence
    # check over all clusters is nominated and auto-run.
    r = _run(_clean_clustered_adata(with_doublets=True), monkeypatch, tmp_path)
    assert r["status"] == "ok"
    assert r["metric_flagged_clusters"] == []          # all clean
    assert r["doublet_signal_missing"] is False
    assert sorted(r["structure_qc_baseline_clusters"]) == ["0", "1", "2"]
    assert r.get("structure_qc_ran") is True
    assert r["structure_qc"].get("structure_qc_run_id")


def test_baseline_nominated_when_doublets_missing(monkeypatch, tmp_path):
    r = _run(_clean_clustered_adata(with_doublets=False), monkeypatch, tmp_path)
    assert r["status"] == "ok"
    assert r["metric_flagged_clusters"] == []          # still no metric flags
    assert r["doublet_signal_missing"] is True
    # ...but structure QC is nominated over ALL clusters as a baseline AND is now
    # auto-run within the same run_cluster_qc call (no separate step needed).
    assert sorted(r["structure_qc_baseline_clusters"]) == ["0", "1", "2"]
    assert r["structure_qc_recommended"] is True
    assert r.get("structure_qc_ran") is True
    assert "structure_qc" in r and r["structure_qc"].get("structure_qc_run_id")


def test_structure_qc_renders_one_heatmap_per_cluster(monkeypatch, tmp_path):
    # Coherence metrics AND a covariance heatmap are produced for every analyzed
    # cluster (coherent ones included) — the auto-run covers the whole clustering,
    # not just a flagged/least-coherent subset.
    import json
    monkeypatch.chdir(tmp_path)
    a = _clean_clustered_adata(with_doublets=True)
    rj, _ = process_tool_call(
        "run_cluster_qc",
        {"cluster_key": "leiden", "save_checkpoint": False},
        a,
    )
    r = json.loads(rj)
    sq = r.get("structure_qc") or {}
    assert sq.get("structure_qc_run_id")
    assert sq.get("n_clusters_analyzed") == 3           # all clusters assessed
    assert isinstance(sq.get("coherence_breakdown"), dict)
    # One heatmap per analyzed cluster — no coherent-cluster skipping, no cap.
    assert sq.get("n_heatmaps_rendered") == 3
    assert "coherence" in (sq.get("structure_summary") or "").lower()


def test_structure_qc_explicit_max_heatmaps_caps_rendering(monkeypatch, tmp_path):
    # An explicit max_heatmaps still bounds how many heatmaps are drawn, even
    # though every cluster is analyzed.
    import json
    monkeypatch.chdir(tmp_path)
    a = _clean_clustered_adata(with_doublets=True)
    rj, _ = process_tool_call(
        "run_cluster_structure_qc",
        {"cluster_key": "leiden", "clusters_to_analyze": ["0", "1", "2"], "max_heatmaps": 2},
        a,
    )
    r = json.loads(rj)
    assert r.get("status") in ("ok", "success")
    assert r.get("n_clusters_analyzed") == 3            # all still analyzed
    assert r.get("n_heatmaps_rendered") == 2            # but rendering capped
