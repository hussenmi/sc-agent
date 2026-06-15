"""Regression tests: concat_datasets tolerates anndata.concat-style call forms.

Multi-dataset loading is a universal first step, and models across providers
routinely conflate concat_datasets() with anndata.concat() — passing fill_value=,
label=, keys=, etc. A single near-miss used to dead-end the load step and
cascade into hand-rolled concat fallbacks. concat_datasets now tolerates the
common aliases and only rejects genuinely unknown kwargs (with a clear message).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from scagent.core import concat_datasets


def _mk(n, genes):
    a = ad.AnnData(X=np.ones((n, len(genes)), dtype="float32"))
    a.var_names = list(genes)
    a.obs_names = [f"cell{i}" for i in range(n)]
    return a


def _pair():
    return _mk(3, ["A", "B", "C"]), _mk(2, ["B", "C", "D"])


def test_fill_value_kwarg_tolerated():
    # the exact failing call from the runs
    d1, d2 = _pair()
    r = concat_datasets([d1, d2], batch_key="replicate", batch_names=["R1", "R2"],
                        join="outer", fill_value=0)
    assert r.shape == (5, 4)
    assert dict(r.obs["replicate"].value_counts()) == {"R1": 3, "R2": 2}


def test_label_alias_maps_to_batch_key():
    d1, d2 = _pair()
    r = concat_datasets([d1, d2], label="sample", keys=["S1", "S2"], join="outer")
    assert "sample" in r.obs
    assert list(r.obs["sample"].cat.categories) == ["S1", "S2"]


def test_explicit_batch_key_beats_label_alias():
    # a real batch_key must not be silently overridden by a stray label=
    d1, d2 = _pair()
    r = concat_datasets([d1, d2], batch_key="donor", batch_names=["x", "y"], label="ignored")
    assert "donor" in r.obs
    assert "ignored" not in r.obs


def test_structural_kwargs_ignored():
    d1, d2 = _pair()
    r = concat_datasets([d1, d2], batch_names=["a", "b"], axis=0, index_unique="-", merge="same")
    assert r.shape == (5, 4)


def test_unknown_kwarg_raises_clear_error():
    d1, d2 = _pair()
    with pytest.raises(TypeError) as exc:
        concat_datasets([d1, d2], bogus_param=1)
    msg = str(exc.value)
    assert "bogus_param" in msg
    assert "batch_key" in msg and "join" in msg  # names the accepted params


def test_normal_call_unaffected():
    d1, d2 = _pair()
    r = concat_datasets([d1, d2], batch_key="batch_id", batch_names=["0", "1"])
    assert r.shape == (5, 4)
    assert "batch_id" in r.obs
