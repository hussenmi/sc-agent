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


def test_no_baseline_when_doublets_present(monkeypatch, tmp_path):
    r = _run(_clean_clustered_adata(with_doublets=True), monkeypatch, tmp_path)
    assert r["status"] == "ok"
    assert r["metric_flagged_clusters"] == []          # all clean
    assert r["doublet_signal_missing"] is False
    assert r["structure_qc_baseline_clusters"] == []   # nothing to force
    assert "next_step" not in r


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
