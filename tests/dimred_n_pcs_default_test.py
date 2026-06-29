"""Tests for the variance-based default-PC selection used by run_neighbors."""

import numpy as np

from scagent.agent.tools import _default_n_pcs_from_variance


def test_stops_at_75pct_variance_when_below_cap():
    # Cumulative variance crosses 75% at PC3 (0.4+0.25+0.15=0.80).
    ratios = [0.40, 0.25, 0.15, 0.10, 0.05, 0.05]
    assert _default_n_pcs_from_variance(ratios) == 3


def test_caps_at_50_when_target_not_reached_early():
    # Flat variance: 75% is only reached well past PC50, so we cap at 50.
    ratios = np.full(80, 1.0 / 80)
    assert _default_n_pcs_from_variance(ratios) == 50


def test_falls_back_to_available_pcs_when_target_never_reached():
    # Only 10 PCs computed and they never reach 75% cumulative variance.
    ratios = np.full(10, 0.05)  # cumulative tops out at 0.50
    assert _default_n_pcs_from_variance(ratios) == 10


def test_respects_custom_target_and_cap():
    ratios = [0.30, 0.20, 0.20, 0.15, 0.15]  # 50% at PC2, 70% at PC3
    assert _default_n_pcs_from_variance(ratios, variance_target=0.50) == 2
    assert _default_n_pcs_from_variance(ratios, variance_target=0.70) == 3
    # Cap dominates when it's the tighter bound.
    assert _default_n_pcs_from_variance(ratios, variance_target=0.99, max_default_n_pcs=2) == 2


def test_empty_returns_cap():
    assert _default_n_pcs_from_variance([], max_default_n_pcs=50) == 50
