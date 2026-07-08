"""Categorical UMAP/t-SNE figures must not be squished by scanpy's right-margin
legend (which shrinks the data axes to fit many long labels). generate_figure
draws the legend OUTSIDE the panel for discrete obs colors, keeping the embedding
in a proper landscape panel."""

import json

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.agent.tools import (
    _add_outside_categorical_legend,
    _is_discrete_obs_color,
    process_tool_call,
)


def _adata_umap(n_cat=18):
    n = 200
    labels = [f"Cell type {i:02d}" for i in range(n_cat)]
    obs = pd.DataFrame(
        {
            "cell_type": [labels[i % n_cat] for i in range(n)],
            "score": np.linspace(0, 1, n),  # continuous → colorbar, not a legend
        },
        index=[f"c{i}" for i in range(n)],
    )
    obs["cell_type"] = obs["cell_type"].astype("category")
    a = AnnData(X=np.zeros((n, 3), dtype=np.float32), obs=obs, var=pd.DataFrame(index=["A", "B", "C"]))
    a.obsm["X_umap"] = np.random.RandomState(0).randn(n, 2)
    return a


def test_discrete_color_detection():
    a = _adata_umap()
    assert _is_discrete_obs_color(a, "cell_type") is True
    assert _is_discrete_obs_color(a, "score") is False   # continuous numeric
    assert _is_discrete_obs_color(a, "A") is False        # gene (not in obs)
    assert _is_discrete_obs_color(a, None) is False


def test_outside_legend_adds_one_handle_per_category():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = _adata_umap(n_cat=12)
    fig, ax = plt.subplots()
    assert _add_outside_categorical_legend(ax, a, "cell_type") is True
    leg = ax.get_legend()
    assert leg is not None
    assert len(leg.get_texts()) == 12
    plt.close(fig)


def test_categorical_umap_is_landscape_not_squished(tmp_path):
    a = _adata_umap(n_cat=18)
    out = str(tmp_path / "umap.png")
    rj, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "umap", "color_by": "cell_type", "output_path": out, "include_image": False},
        a,
    )
    assert json.loads(rj)["status"] == "ok"
    from PIL import Image
    w, h = Image.open(out).size
    # Legend is outside → wide panel. The old right-margin path produced a tall,
    # narrow figure (w < h); guard against regressing to that.
    assert w > h
