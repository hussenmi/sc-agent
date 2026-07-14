"""diffxpy runs in a separate environment (frozen batchglm/TensorFlow stack) and is
driven across a process boundary via SCAGENT_DIFFXPY, like CellBender.

These tests split in two:
- Pure parent-side logic (interpreter resolution, availability, input validation)
  that needs no diffxpy env.
- A real end-to-end handoff gated on ``diffxpy_available()`` — skipped when the
  diffxpy env is not built on this host.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from scagent.batch.diffxpy import (
    DiffxpyUnavailable,
    diffxpy_available,
    diffxpy_python,
    diffxpy_versions,
    run_two_group_de,
)


def test_diffxpy_python_unset(monkeypatch):
    monkeypatch.delenv("SCAGENT_DIFFXPY", raising=False)
    assert diffxpy_python() is None
    assert diffxpy_available() is False


def test_diffxpy_python_resolves_file(monkeypatch, tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("SCAGENT_DIFFXPY", str(fake))
    assert diffxpy_python() == str(fake)


def test_diffxpy_python_resolves_env_dir(monkeypatch, tmp_path):
    (tmp_path / "bin").mkdir()
    py = tmp_path / "bin" / "python"
    py.write_text("#!/bin/sh\n")
    monkeypatch.setenv("SCAGENT_DIFFXPY", str(tmp_path))
    assert diffxpy_python() == str(py)


def test_diffxpy_python_missing_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SCAGENT_DIFFXPY", str(tmp_path / "does_not_exist"))
    assert diffxpy_python() is None


def test_run_two_group_de_unavailable(monkeypatch):
    monkeypatch.delenv("SCAGENT_DIFFXPY", raising=False)
    with pytest.raises(DiffxpyUnavailable):
        run_two_group_de(
            np.zeros((4, 3), dtype=np.float32),
            np.array([1, 1, 0, 0]),
            ["a", "b", "c"],
        )


def test_run_two_group_de_validates_shapes():
    # python_exe is supplied so we get past the availability gate and hit the
    # shape checks, which must fire before any subprocess is launched.
    with pytest.raises(ValueError, match="group_mask length"):
        run_two_group_de(
            np.zeros((4, 3), dtype=np.float32),
            np.array([1, 1, 0]),  # wrong length
            ["a", "b", "c"],
            python_exe=sys.executable,
        )
    with pytest.raises(ValueError, match="gene_names length"):
        run_two_group_de(
            np.zeros((4, 3), dtype=np.float32),
            np.array([1, 1, 0, 0]),
            ["a", "b"],  # wrong length
            python_exe=sys.executable,
        )


def test_run_two_group_de_validates_group_sizes():
    # A group that is too small must be rejected before any subprocess launches.
    with pytest.raises(ValueError, match="both groups need"):
        run_two_group_de(
            np.zeros((10, 3), dtype=np.float32),
            np.array([1] + [0] * 9),  # group of interest has 1 cell
            ["a", "b", "c"],
            min_cells_per_group=3,
            python_exe=sys.executable,
        )


def test_run_two_group_de_rejects_noncount_for_nb_wald():
    # NB Wald on log/normalized (non-integer) input must fail loudly, not mis-model.
    lognorm = np.log1p(np.arange(30, dtype=np.float32).reshape(10, 3))
    with pytest.raises(ValueError, match="integer counts"):
        run_two_group_de(
            lognorm,
            np.array([1] * 5 + [0] * 5),
            ["a", "b", "c"],
            test="wald",
            noise_model="nb",
            python_exe=sys.executable,
        )


def test_run_two_group_de_rejects_negative_for_nb_wald():
    neg = np.full((10, 3), -1.0, dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        run_two_group_de(
            neg,
            np.array([1] * 5 + [0] * 5),
            ["a", "b", "c"],
            test="wald",
            noise_model="nb",
            python_exe=sys.executable,
        )


def _toy_counts(seed=0):
    """Counts where g0 is strongly up in the group of interest and g1 up in the rest."""
    rng = np.random.default_rng(seed)
    n, g = 150, 8
    counts = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    group = np.array([1] * (n // 2) + [0] * (n - n // 2))
    counts[: n // 2, 0] += rng.poisson(8.0, size=n // 2)
    counts[n // 2 :, 1] += rng.poisson(8.0, size=n - n // 2)
    genes = [f"g{i}" for i in range(g)]
    return counts, group, genes


@pytest.mark.skipif(not diffxpy_available(), reason="diffxpy env not built on this host")
@pytest.mark.parametrize(
    "test,noise_model,size_factors",
    [
        ("rank", None, None),
        ("t-test", None, None),
        ("wald", "nb", "total_count"),
    ],
)
def test_run_two_group_de_end_to_end(test, noise_model, size_factors):
    """Full handoff on real diffxpy across all supported tests: the group-of-interest
    gene gets a positive log2fc, the rest-gene a negative one, both significant, and
    the result carries the normalized columns. Exercises the group-vs-rest sign flip."""
    counts, group, genes = _toy_counts()
    res = run_two_group_de(
        counts, group, genes, test=test, noise_model=noise_model, size_factors=size_factors
    )
    assert {
        "gene", "pval", "qval", "expression_effect", "engine_log2fc", "mean",
        "mean_target", "mean_reference", "pct_target", "pct_reference",
    }.issubset(res.columns)
    g0 = res[res["gene"] == "g0"].iloc[0]
    g1 = res[res["gene"] == "g1"].iloc[0]
    # expression_effect (mean_target - mean_reference) is the primary oriented effect.
    assert g0["expression_effect"] > 0 and g0["pval"] < 0.05  # up in group of interest
    assert g1["expression_effect"] < 0 and g1["pval"] < 0.05  # up in the rest
    # Orientation invariant uses the mean difference, NOT the engine fold-change.
    for _, row in res.iterrows():
        assert (row["expression_effect"] > 0) == (row["mean_target"] > row["mean_reference"])


@pytest.mark.skipif(not diffxpy_available(), reason="diffxpy env not built on this host")
def test_diffxpy_versions_reports_stack():
    v = diffxpy_versions()
    assert v.get("diffxpy", "").lstrip("v").startswith("0.7.4")
    assert "numpy" in v and "anndata" in v


@pytest.mark.skipif(not diffxpy_available(), reason="diffxpy env not built on this host")
def test_run_two_group_de_unsupported_test_errors():
    counts, group, genes = _toy_counts()
    with pytest.raises(RuntimeError, match="unsupported diffxpy test"):
        run_two_group_de(counts, group, genes, test="lrt", noise_model="nb")
