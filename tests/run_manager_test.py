"""Tests for RunManager manifest step recording."""

import json

from scagent.agent.run_manager import RunManager


def _normalize_result():
    """A trimmed normalize_and_hvg-style result payload."""
    return {
        "status": "ok",
        "tool": "normalize_and_hvg",
        "before": {"n_cells": 100, "n_genes": 200},
        "after": {"n_cells": 100, "n_genes": 190},
        "backend": "rapids_singlecell",
        "n_hvg": 40,
        "target_sum": 10000.0,
        "log_transform": True,
        "normalization": {"target_sum": 10000.0, "log_transform": True, "resolved_source": "raw_counts"},
        "resolved_source": "raw_counts",
        "normalization_source": "auto",
        # a large field that must NOT leak into the manifest metrics
        "state": {"huge": list(range(1000))},
    }


def test_log_step_records_normalization_provenance(tmp_path):
    """log1p / target_sum / normalization must survive into manifest metrics."""
    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    rm.log_step(tool="normalize_and_hvg", parameters={"n_hvg": 40}, result=_normalize_result())

    manifest = json.loads((rm.run_dir / "manifest.json").read_text())
    step = manifest["steps_completed"][-1]
    metrics = step["metrics"]

    # The whole point of the fix: whether log1p ran is now visible in the manifest.
    assert metrics["log_transform"] is True
    assert metrics["target_sum"] == 10000.0
    assert metrics["normalization"]["resolved_source"] == "raw_counts"
    assert metrics["resolved_source"] == "raw_counts"
    # Existing whitelist keys still captured.
    assert metrics["backend"] == "rapids_singlecell"
    assert metrics["n_hvg"] == 40
    assert metrics["status"] == "ok"


def test_log_step_does_not_leak_full_result(tmp_path):
    """Only whitelisted keys are kept — bulky/non-metric fields are dropped."""
    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    rm.log_step(tool="normalize_and_hvg", parameters={}, result=_normalize_result())

    metrics = json.loads((rm.run_dir / "manifest.json").read_text())["steps_completed"][-1]["metrics"]
    assert "state" not in metrics
    assert "tool" not in metrics  # not a metric key


def test_log_step_skips_absent_keys(tmp_path):
    """Tools that don't emit normalization keys record no such metrics (no crash)."""
    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    rm.log_step(
        tool="run_clustering",
        parameters={"resolution": 1.0},
        result={"status": "ok", "n_clusters": 12, "backend": "scanpy_cpu"},
    )

    metrics = json.loads((rm.run_dir / "manifest.json").read_text())["steps_completed"][-1]["metrics"]
    assert metrics == {"status": "ok", "n_clusters": 12, "backend": "scanpy_cpu"}
    assert "log_transform" not in metrics
