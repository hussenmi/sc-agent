"""Cluster QC must be enforced and multi-metric.

Two gaps this pins:
  * Metric cluster QC (run_cluster_qc) was entirely voluntary — only the
    structure_qc floor existed, and it fires only AFTER metric QC ran. So a run
    that clustered but skipped cluster QC (e.g. nothing looked wrong because
    Scrublet wasn't run) bypassed the whole QC chain. A `cluster_qc` entry
    obligation now enforces metric QC on the active clustering, and re-fires per
    reclustering round (driving the iterative QC loop).
  * run_cluster_qc aggregated ribosomal% but never used it in a decision. It now
    contributes (elevated ribo → structure-QC review), and the decision draws on
    every metric present, not the doublet signal alone.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import process_tool_call
from scagent.agent.world_state import AgentWorldState


# --- cluster_qc obligation ----------------------------------------------------
def test_cluster_qc_unmet_when_clustering_ready_but_qc_not_run():
    ws = AgentWorldState()
    # _cluster_qc_summary returns status="needed" when clustering + QC metrics
    # exist but no fresh metric QC has run on the active clustering.
    ws.data_summary = {"cluster_qc": {"status": "needed", "cluster_key": "leiden"}}
    assert ws.cluster_qc_obligation_unmet() is True
    o = next(o for o in ws.unmet_obligations() if o["key"] == "cluster_qc")
    assert o["blocks_terminal"] is True
    assert o["kind"] == "entry"  # bounded nudge, lapses if user opted out


def test_cluster_qc_satisfied_when_fresh():
    ws = AgentWorldState()
    for status in ("fresh_clean", "fresh_review_required", "not_applicable", "not_ready"):
        ws.data_summary = {"cluster_qc": {"status": status}}
        assert ws.cluster_qc_obligation_unmet() is False, status
    ws.data_summary = {}
    assert ws.cluster_qc_obligation_unmet() is False


def test_metric_qc_obligation_listed_before_structure_qc():
    # The chain reads metric-first: run_cluster_qc, then structure QC.
    ws = AgentWorldState()
    ws.data_summary = {"cluster_qc": {"status": "needed"}}
    ws.cluster_qc_registry = {"leiden": {"cluster_key": "leiden", "checked_at": "t0"}}
    keys = [o["key"] for o in ws.unmet_obligations()]
    assert "cluster_qc" in keys and "structure_qc" in keys
    assert keys.index("cluster_qc") < keys.index("structure_qc")


# --- ribosomal signal in the decision ----------------------------------------
def _clustered_adata(with_ribo=True, high_ribo_cluster="2"):
    n = 30
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=np.abs(rng.normal(size=(n, 6))).astype("float32"))
    leiden = [str(i % 3) for i in range(n)]
    a.obs["leiden"] = pd.Categorical(leiden)
    a.obs["total_counts"] = rng.uniform(2000, 5000, n).astype("float32")
    a.obs["n_genes_by_counts"] = rng.uniform(1000, 2500, n).astype("float32")
    a.obs["pct_counts_mt"] = rng.uniform(2, 8, n).astype("float32")
    if with_ribo:
        ribo = np.where(np.array(leiden) == high_ribo_cluster, 65.0, 15.0).astype("float32")
        a.obs["pct_counts_ribo"] = ribo
    return a


def test_high_ribo_cluster_routed_to_review():
    a = _clustered_adata()
    res, _ = process_tool_call("run_cluster_qc", {"save_checkpoint": False}, a)
    d = json.loads(res)
    assert d["status"] in ("ok", "success")
    assert d["ribo_signal_available"] is True
    assert d["thresholds_used"]["ribo_threshold"] == 50.0
    dec = d["cluster_decisions"]["2"]
    assert dec["evidence"]["high_ribo"] is True
    assert dec["evidence"]["mean_ribo_pct"] == 65.0
    assert any("ribosomal" in r for r in dec["reasons"])
    assert dec["recommended_action"] == "review"
    assert "2" in d["ambiguous"]
    # a normal cluster is not ribo-flagged
    assert d["cluster_decisions"]["0"]["evidence"]["high_ribo"] is False


def test_missing_ribo_is_handled_gracefully():
    a = _clustered_adata(with_ribo=False)
    res, _ = process_tool_call("run_cluster_qc", {"save_checkpoint": False}, a)
    d = json.loads(res)
    assert d["status"] in ("ok", "success")
    assert d["ribo_signal_available"] is False
    for dec in d["cluster_decisions"].values():
        assert dec["evidence"]["high_ribo"] is False
        assert dec["evidence"]["mean_ribo_pct"] is None


def test_no_doublet_signal_nominates_baseline_structure_qc():
    # Scrublet not run (no doublet_score): metric QC flags nothing on clean data,
    # but a baseline structure-QC pass over all clusters is still nominated so
    # incoherent clusters aren't missed.
    a = _clustered_adata(with_ribo=False)
    res, _ = process_tool_call("run_cluster_qc", {"save_checkpoint": False}, a)
    d = json.loads(res)
    assert d["doublet_signal_missing"] is True
    assert sorted(d["structure_qc_baseline_clusters"]) == ["0", "1", "2"]
    assert d["structure_qc_recommended"] is True
