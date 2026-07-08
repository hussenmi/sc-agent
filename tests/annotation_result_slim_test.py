"""finalize_annotation results must stay small.

run_2026_07_05_193415 finalized 60 clusters successfully, then died: the finalize
tool result echoed the full per_cluster_evidence (~238 KB) inline, which also
propagated into world_state.annotation_validation (world_state reads it from the
RESULT) and re-entered the LLM snapshot every turn. That single oversized message
could not be trimmed below the emergency budget → context overflow right after a
successful finalize. The full evidence stays on adata.uns + the saved JSON/MD; the
RESULT now carries only a slim summary.
"""

from __future__ import annotations

import json

from scagent.agent.tools import (
    _annotation_low_confidence_clusters,
    _slim_annotation_validation,
)


def _full_payload(n=60):
    per_cluster = {
        str(i): {
            "label": f"Label{i % 6}",
            "confidence": "high" if i % 3 else "low",
            "validation_tier": "cytopus_plus_deg",
            "supporting_genes": ["GENEA", "GENEB", "GENEC"],
            "reasoning": "detailed reasoning " * 20,
            "source_synthesis": {"final_decision_basis": "x" * 200},
            "competing_labels_considered": ["alt1", "alt2"],
            "panglaodb_queried": bool(i % 2),
        }
        for i in range(n)
    }
    return {
        "annotation_key": "cell_type",
        "cluster_key": "leiden",
        "panglaodb_validated": True,
        "external_validation_policy": "cytopus_local_primary_panglaodb_fallback",
        "validation_strategy": "reference_consensus_and_local_cytopus",
        "validation_tier_breakdown": {"cytopus_plus_deg": n},
        "n_cytopus_adjudicated": n,
        "n_panglaodb_adjudicated": 5,
        "panglaodb_required_clusters": ["7", "12"],
        "n_clusters_validated": n,
        "per_cluster_evidence": per_cluster,
        "label_counts": {"Label0": 1000},
        "auto_fixes": [f"auto fix {i}" for i in range(30)],
        "finalized": True,
    }


def test_slim_drops_per_cluster_evidence_and_shrinks():
    payload = _full_payload()
    slim = _slim_annotation_validation(payload)
    assert "per_cluster_evidence" not in slim
    # dramatically smaller
    assert len(json.dumps(slim)) < len(json.dumps(payload)) * 0.25


def test_slim_preserves_metadata_world_state_needs():
    # world_state._update_annotation_validation reads these off the result.
    payload = _full_payload()
    slim = _slim_annotation_validation(payload)
    for key in (
        "annotation_key", "cluster_key", "n_clusters_validated",
        "external_validation_policy", "validation_strategy",
        "validation_tier_breakdown", "n_cytopus_adjudicated",
        "n_panglaodb_adjudicated", "panglaodb_required_clusters", "label_counts",
    ):
        assert slim.get(key) == payload.get(key), key
    assert slim["n_auto_fixes"] == 30
    assert "per_cluster_evidence_note" in slim


def test_low_confidence_clusters_only():
    payload = _full_payload()
    low = _slim_annotation_validation(payload)["low_confidence_clusters"]
    # every entry is a non-high-confidence cluster with a compact shape
    assert low, "expected some low-confidence clusters"
    for e in low:
        assert set(e.keys()) == {"cluster", "label", "confidence"}
        assert str(e["confidence"]).lower() != "high"


def test_low_confidence_helper_empty_and_capped():
    assert _annotation_low_confidence_clusters({}) == []
    all_low = {str(i): {"label": "L", "confidence": "low"} for i in range(100)}
    assert len(_annotation_low_confidence_clusters(all_low, limit=40)) == 40


def test_slim_handles_missing_fields():
    # validate_only preview payloads have fewer keys; must not raise.
    minimal = {"annotation_key": "cell_type", "per_cluster_evidence": {"0": {"label": "T", "confidence": "low"}}}
    slim = _slim_annotation_validation(minimal)
    assert slim["n_auto_fixes"] == 0
    assert "per_cluster_evidence" not in slim
    assert slim["low_confidence_clusters"] == [{"cluster": "0", "label": "T", "confidence": "low"}]
