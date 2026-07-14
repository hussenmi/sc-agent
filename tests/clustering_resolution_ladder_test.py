"""Resolution ladder + canonical clustering-UMAP naming are harness-enforced.

Consistency contract (was prompt-only, drifted run-to-run):
  * The first leiden clustering of a run must be resolution 2.0; every
    subsequent one comes down through {1.5, 1.0} and never climbs back up.
  * run_clustering auto-saves the canonical cluster UMAP to
    figures/{pre,post}_integration/umap_leiden_res_<R>.png — resolution and
    stage always in the name, never overwritten, no reliance on the model.
  * prepare_annotation refuses unless the annotated clustering is at res 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scagent.agent.tools import (
    _leiden_ladder_violation,
    process_tool_call,
)
from scagent.core.inspector import (
    format_resolution_token,
    infer_cluster_key,
    register_clustering,
)


def test_resolution_token_is_uniform_across_ladder():
    # The whole point: res-1.0 must not spell inconsistently ("1" vs "1_0").
    # Every ladder rung carries one decimal, matching the unavoidable 1.5 -> "1_5".
    assert format_resolution_token(2.0) == "2_0"
    assert format_resolution_token(1.5) == "1_5"
    assert format_resolution_token(1.0) == "1_0"
    assert format_resolution_token(0.5) == "0_5"
    assert format_resolution_token(0.75) == "0_75"
    # so the harness auto-key for res 1.0 now matches what a model naturally picks
    assert infer_cluster_key("leiden", 1.0) == "leiden_res_1_0"
    assert infer_cluster_key("leiden", 2.0) == "leiden_res_2_0"


class _RM:
    """Minimal run_manager: figures land under run_dir, add_output is a no-op."""

    def __init__(self, run_dir):
        self.run_dir = str(run_dir)

    def add_output(self, *_a, **_k):
        pass


def _embedded_adata(n=80, g=40):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(g)]),
    )
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    sc.tl.umap(a)
    return a


def _cluster(a, res, rm):
    res_json, a2 = process_tool_call(
        "run_clustering", {"resolution": res}, a, run_manager=rm
    )
    return json.loads(res_json), a2


# --- pure ladder helper -------------------------------------------------------
def test_ladder_helper_first_pass_must_be_two():
    assert _leiden_ladder_violation(2.0, []) is None
    assert _leiden_ladder_violation(1.0, []) is not None
    assert _leiden_ladder_violation(0.5, []) is not None
    assert _leiden_ladder_violation(2.5, []) is not None


def test_ladder_helper_comes_down_never_up():
    # after 2.0, both coarser rungs are allowed
    assert _leiden_ladder_violation(1.5, [2.0]) is None
    assert _leiden_ladder_violation(1.0, [2.0]) is None
    # repeating 2.0 or an off-ladder value is refused
    assert _leiden_ladder_violation(2.0, [2.0]) is not None
    assert _leiden_ladder_violation(0.8, [2.0]) is not None
    # once at 1.0 you cannot climb back to 1.5
    assert _leiden_ladder_violation(1.5, [2.0, 1.0]) is not None
    assert _leiden_ladder_violation(1.0, [2.0, 1.5]) is None


# --- run_clustering enforcement + naming --------------------------------------
def test_first_pass_below_two_is_blocked(tmp_path):
    a = _embedded_adata()
    out, _ = _cluster(a, 1.0, _RM(tmp_path))
    assert out["status"] == "error"
    assert "resolution 2.0" in out["message"]
    # nothing was clustered / no ladder history recorded
    assert "leiden" not in a.obs.columns


def test_user_override_honors_nonstandard_resolution(tmp_path):
    # The ladder guards against model drift, not the user: an explicit request
    # (surfaced as allow_nonstandard_resolution=true) is honored even off-ladder.
    a = _embedded_adata()
    res_json, a = process_tool_call(
        "run_clustering",
        {"resolution": 0.5, "allow_nonstandard_resolution": True},
        a,
        run_manager=_RM(tmp_path),
    )
    out = json.loads(res_json)
    assert out["status"] == "ok"
    assert out["resolution"] == 0.5
    assert "leiden" in a.obs.columns


def test_ladder_and_canonical_umap_naming(tmp_path):
    rm = _RM(tmp_path)
    a = _embedded_adata()

    out2, a = _cluster(a, 2.0, rm)
    assert out2["status"] == "ok"
    pre = tmp_path / "figures" / "pre_integration"
    # uniform token: 2.0 -> "2_0" (not a bare "2"), matching 1.5 -> "1_5"
    assert (pre / "umap_leiden_res_2_0.png").exists()
    assert out2["cluster_umap_figure"].endswith("umap_leiden_res_2_0.png")

    # a second 2.0 is refused (must come down)
    out_again, a = _cluster(a, 2.0, rm)
    assert out_again["status"] == "error"

    # coarsen to 1.5 -> distinct file, pre-integration (no corrected embedding yet)
    out15, a = _cluster(a, 1.5, rm)
    assert out15["status"] == "ok"
    assert (pre / "umap_leiden_res_1_5.png").exists()

    # integration appears and the embedding is rebuilt on the corrected rep ->
    # res 1.0 lands in post_integration/, resolution in name.
    a.obsm["X_scVI"] = np.random.default_rng(1).normal(size=(a.n_obs, 8)).astype("float32")
    sc.pp.neighbors(a, use_rep="X_scVI", n_neighbors=10)
    sc.tl.umap(a)
    out10, a = _cluster(a, 1.0, rm)
    assert out10["status"] == "ok"
    assert (tmp_path / "figures" / "post_integration" / "umap_leiden_res_1_0.png").exists()


# --- prepare_annotation final-resolution floor --------------------------------
def _missing(result_json: str):
    return json.loads(result_json).get("missing_prerequisites") or []


def test_prepare_annotation_refuses_nonstandard_final_resolution():
    a = _embedded_adata()
    a.obs["leiden"] = pd.Categorical(
        np.random.default_rng(0).integers(0, 4, a.n_obs).astype(str)
    )
    register_clustering(a, cluster_key="leiden", method="leiden", resolution=1.5)
    res, _ = process_tool_call("prepare_annotation", {"cluster_key": "leiden"}, a)
    assert "final_resolution_1.0_clustering" in _missing(res)


def test_prepare_annotation_override_bypasses_final_resolution_floor():
    a = _embedded_adata()
    a.obs["leiden"] = pd.Categorical(
        np.random.default_rng(0).integers(0, 4, a.n_obs).astype(str)
    )
    register_clustering(a, cluster_key="leiden", method="leiden", resolution=1.5)
    res, _ = process_tool_call(
        "prepare_annotation",
        {"cluster_key": "leiden", "allow_nonstandard_final_resolution": True},
        a,
    )
    assert "final_resolution_1.0_clustering" not in _missing(res)


# --- harness-owned batch/donor UMAP (run_umap): stage-labeled, no resolution ---
def _with_donor(a, n_groups=4):
    a.obs["donor"] = pd.Categorical(
        np.random.default_rng(0).integers(0, n_groups, a.n_obs).astype(str)
    )
    return a


def test_batch_umap_pre_integration_has_no_resolution(tmp_path):
    a = _with_donor(_embedded_adata())  # X_pca neighbors, no corrected embedding
    res_json, _ = process_tool_call(
        "run_umap", {"min_dist": 0.1}, a, run_manager=_RM(tmp_path)
    )
    out = json.loads(res_json)
    assert out["status"] == "ok"
    fig = out["batch_umap_figure"]
    assert fig and Path(fig).name == "umap_donor.png"  # colored by donor, NO res tag
    assert (tmp_path / "figures" / "pre_integration" / "umap_donor.png").exists()
    # the coordinates/coloring do not depend on resolution -> name must not carry one
    assert "res" not in Path(fig).name


def test_batch_umap_post_integration_stage(tmp_path):
    a = _with_donor(_embedded_adata())
    a.obsm["X_scVI"] = np.random.default_rng(1).normal(size=(a.n_obs, 8)).astype("float32")
    sc.pp.neighbors(a, use_rep="X_scVI", n_neighbors=10)  # rebuild on corrected rep
    res_json, _ = process_tool_call("run_umap", {}, a, run_manager=_RM(tmp_path))
    out = json.loads(res_json)
    assert out["status"] == "ok"
    assert (tmp_path / "figures" / "post_integration" / "umap_donor.png").exists()
    # exactly one per stage: not duplicated into pre_integration/
    assert not (tmp_path / "figures" / "pre_integration" / "umap_donor.png").exists()


def test_batch_umap_skipped_without_multigroup_batch(tmp_path):
    a = _embedded_adata()  # single sample, no donor/batch column
    res_json, _ = process_tool_call("run_umap", {}, a, run_manager=_RM(tmp_path))
    out = json.loads(res_json)
    assert out["batch_umap_figure"] is None
    assert not (tmp_path / "figures" / "pre_integration").exists()
