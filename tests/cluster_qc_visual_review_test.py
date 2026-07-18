"""Tests for holistic per-cluster QC flagging and the visual-review step.

Background (run_2026_07_15_121111): cluster 36 was the lowest library, lowest
genes, and highest %MT (~17%) of all 61 clusters — a textbook dying population —
yet it crossed no single absolute gate (MT < 25%, genes/lib just above the 0.5x
floor) so the old per-metric decision tree kept it. Two fixes:

1. Holistic, distribution-relative flagging: a cluster that is an outlier in the
   bad direction on >=2 metrics jointly is routed to review even if no single
   metric is individually extreme.
2. A recorded VISUAL review of the QC box plot, merged into the review set and
   persisted, plus a bounded nudge toward it before annotation.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.agent import SCAgent
from scagent.agent.tools import get_tools, process_tool_call
from scagent.agent.world_state import AgentWorldState


def _adata_with_bad_cluster(borderline: bool):
    """8 clusters; cluster '7' is worst on lib+genes+MT. borderline=True keeps it
    just above every absolute gate so only holistic flagging can catch it."""
    rng = np.random.default_rng(3)
    n = 150
    parts = []
    for c in range(8):
        if c == 7:
            if borderline:
                lib, genes, mt = 3000, 1000, 16.0   # above 0.5x gates, MT < 25
            else:
                lib, genes, mt = 1500, 600, 22.0
            libv = rng.normal(lib, 250, n)
            genesv = rng.normal(genes, 100, n)
            mtv = rng.normal(mt, 1.5, n)
        else:
            libv = rng.normal(6000, 1000, n)
            genesv = rng.normal(1900, 300, n)
            mtv = rng.normal(6, 2, n)
        parts.append(pd.DataFrame({
            "leiden": str(c),
            "total_counts": np.abs(libv),
            "n_genes_by_counts": np.abs(genesv),
            "pct_counts_mt": np.clip(mtv, 0, 100),
            "pct_counts_ribo": np.abs(rng.normal(15, 4, n)),
        }))
    obs = pd.concat(parts, ignore_index=True)
    a = ad.AnnData(X=np.abs(rng.normal(size=(len(obs), 5))).astype("float32"))
    a.obs = obs
    a.obs.index = [f"c{i}" for i in range(len(obs))]
    a.obs["leiden"] = pd.Categorical(a.obs["leiden"])
    a.var_names = [f"GENE{i}" for i in range(5)]
    return a


def _run_qc(a, ws=None):
    r, _ = process_tool_call(
        "run_cluster_qc",
        {"save_checkpoint": False, "auto_structure_qc": False},
        a,
        world_state=ws,
    )
    return json.loads(r)


# --- holistic flagging ------------------------------------------------------- #


def test_multi_metric_outlier_flagged_even_under_all_hard_gates():
    a = _adata_with_bad_cluster(borderline=True)
    d = _run_qc(a)
    dec = d["cluster_decisions"]["7"]
    ev = dec["evidence"]
    # crosses no single hard gate...
    assert ev["low_library"] is False
    assert ev["low_genes"] is False
    assert ev["high_mt"] is False
    # ...but is an outlier on multiple metrics jointly, so it's routed to review
    assert ev["n_concerning_metrics"] >= 2
    assert dec["recommended_action"] == "review"
    assert "7" in d["ambiguous"]
    assert any("outlier" in r for r in dec["reasons"])


def test_healthy_clusters_stay_clean():
    a = _adata_with_bad_cluster(borderline=True)
    d = _run_qc(a)
    for cid in ("0", "1", "2", "3"):
        assert d["cluster_decisions"][cid]["recommended_action"] == "keep"


def test_severe_cluster_still_proposed_for_removal():
    # When the cluster crosses the hard gates outright, the existing obvious-junk
    # path still fires (holistic did not weaken removal).
    a = _adata_with_bad_cluster(borderline=False)
    d = _run_qc(a)
    assert "7" in d["proposed_removal"]


def test_result_requests_visual_review():
    a = _adata_with_bad_cluster(borderline=True)
    d = _run_qc(a)
    vr = d.get("visual_review")
    assert vr and vr["required"] is True
    assert vr["next_tool"] == "record_cluster_qc_visual_review"


# --- visual review recording ------------------------------------------------- #


def test_visual_review_tool_is_advertised():
    assert "record_cluster_qc_visual_review" in {t["name"] for t in get_tools()}


def test_record_visual_review_merges_and_persists():
    a = _adata_with_bad_cluster(borderline=True)
    ws = AgentWorldState()
    d = _run_qc(a, ws)
    ws.apply_tool_result("run_cluster_qc", d, adata=a)
    assert ws.cluster_qc_registry["leiden"].get("visual_review_recorded_at") is None

    r, _ = process_tool_call(
        "record_cluster_qc_visual_review",
        {
            "cluster_key": "leiden",
            "suspicious_clusters": [
                {"cluster": "3", "concern": "low genes vs peers", "panels": ["genes"]},
                {"cluster": "404", "concern": "does not exist"},
            ],
            "overall_note": "cluster 7 clearly worst; 3 borderline",
        },
        a,
        world_state=ws,
    )
    out = json.loads(r)
    assert out["status"] == "ok"
    assert "3" in out["newly_flagged_for_review"]
    assert any("404" in w for w in out["warnings"])  # invalid id dropped
    # merged into the review set + freshness marker set
    entry = ws.cluster_qc_registry["leiden"]
    assert "3" in entry["ambiguous"]
    assert entry.get("visual_review_recorded_at")
    # decision for cluster 3 flipped to review and carries the visual flag
    dec3 = entry["cluster_decisions"]["3"]
    assert dec3["recommended_action"] == "review"
    assert dec3["evidence"].get("visual_flag") is True
    # persisted to uns for downstream (annotation) connection
    assert "leiden" in a.uns.get("cluster_qc_visual_review", {})


def test_record_visual_review_empty_is_valid():
    a = _adata_with_bad_cluster(borderline=True)
    ws = AgentWorldState()
    d = _run_qc(a, ws)
    ws.apply_tool_result("run_cluster_qc", d, adata=a)
    r, _ = process_tool_call(
        "record_cluster_qc_visual_review",
        {"cluster_key": "leiden", "suspicious_clusters": [], "overall_note": "all comparable"},
        a,
        world_state=ws,
    )
    out = json.loads(r)
    assert out["status"] == "ok"
    assert out["newly_flagged_for_review"] == []
    assert ws.cluster_qc_registry["leiden"].get("visual_review_recorded_at")


# --- the bounded nudge gate -------------------------------------------------- #


def _gate_agent(registry, active="leiden"):
    class _WS:
        def __init__(self):
            self.cluster_qc_registry = registry
            self.active_cluster_key = active

    agent = object.__new__(SCAgent)
    agent.adata = object()
    agent.world_state = _WS()
    agent._visual_review_nudged = False
    return agent


def test_gate_nudges_then_falls_back(monkeypatch):
    monkeypatch.delenv("SCAGENT_CLUSTER_QC_VISUAL_REVIEW", raising=False)
    agent = _gate_agent({"leiden": {"checked_at": "t"}})
    assert agent._cluster_qc_visual_review_gate_action("prepare_annotation") == "nudge"
    agent._visual_review_nudged = True
    assert agent._cluster_qc_visual_review_gate_action("prepare_annotation") == "fallback"


def test_gate_silent_when_reviewed_or_not_annotation(monkeypatch):
    monkeypatch.delenv("SCAGENT_CLUSTER_QC_VISUAL_REVIEW", raising=False)
    reviewed = {"leiden": {"checked_at": "t", "visual_review_recorded_at": "t"}}
    assert _gate_agent(reviewed)._cluster_qc_visual_review_gate_action("prepare_annotation") is None
    # non-annotation tool is never gated
    unreviewed = {"leiden": {"checked_at": "t"}}
    assert _gate_agent(unreviewed)._cluster_qc_visual_review_gate_action("run_deg") is None


def test_gate_silent_when_qc_not_run(monkeypatch):
    monkeypatch.delenv("SCAGENT_CLUSTER_QC_VISUAL_REVIEW", raising=False)
    assert _gate_agent({})._cluster_qc_visual_review_gate_action("prepare_annotation") is None


def test_gate_disabled_by_env(monkeypatch):
    monkeypatch.setenv("SCAGENT_CLUSTER_QC_VISUAL_REVIEW", "0")
    agent = _gate_agent({"leiden": {"checked_at": "t"}})
    assert agent._cluster_qc_visual_review_gate_action("prepare_annotation") is None
