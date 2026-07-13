"""dotplot/heatmap/violin require a categorical groupby (color_by) and genes.

run_2026_07_13_112751: the model called generate_figure(plot_type='dotplot',
genes=[...]) with NO color_by. scanpy received groupby=None and raised a cryptic
"'NoneType' object is not iterable", which the model read as a data/tool bug and
gave up on the plot. generate_figure now validates the grouping column and the
gene list up front and returns an actionable message (naming the valid grouping
columns) instead of letting scanpy crash.
"""

import json

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.agent.tools import process_tool_call


def _adata():
    return AnnData(
        X=np.random.RandomState(0).rand(30, 5).astype("float32"),
        obs=pd.DataFrame(
            {
                "leiden": pd.Categorical([str(i % 3) for i in range(30)]),
                "cell_type": pd.Categorical(["A", "B", "C"] * 10),
            },
            index=[f"c{i}" for i in range(30)],
        ),
        var=pd.DataFrame(index=["SFTPC", "SFTPA1", "C1QB", "CD14", "PECAM1"]),
    )


def test_dotplot_without_color_by_returns_actionable_error():
    a = _adata()
    rj, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "dotplot", "genes": ["SFTPC", "C1QB"], "output_path": "x.png"},
        a,
    )
    r = json.loads(rj)
    assert r["status"] != "ok"
    assert "groups genes by a categorical obs column" in r["message"]
    # the actual grouping columns are handed back so the model can retry in one round
    assert set(r["available_grouping_columns"]) == {"leiden", "cell_type"}


def test_heatmap_without_color_by_returns_actionable_error():
    a = _adata()
    rj, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "heatmap", "genes": ["SFTPC"], "output_path": "x.png"},
        a,
    )
    r = json.loads(rj)
    assert r["status"] != "ok"
    assert r["plot_type"] == "heatmap"


def test_dotplot_with_nonexistent_group_errors():
    a = _adata()
    rj, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "dotplot", "genes": ["SFTPC"], "color_by": "not_a_column",
         "output_path": "x.png"},
        a,
    )
    r = json.loads(rj)
    assert r["status"] != "ok"
    assert "not an obs column" in r["message"]


def test_dotplot_with_missing_genes_errors_before_scanpy():
    a = _adata()
    rj, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "dotplot", "genes": ["SFTPC", "SEPR"], "color_by": "cell_type",
         "output_path": "x.png"},
        a,
    )
    r = json.loads(rj)
    assert r["status"] != "ok"
    assert "SEPR" in r["message"]
    assert r["missing_genes"] == ["SEPR"]


def test_dotplot_with_empty_genes_errors():
    a = _adata()
    rj, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "dotplot", "genes": [], "color_by": "cell_type", "output_path": "x.png"},
        a,
    )
    r = json.loads(rj)
    assert r["status"] != "ok"
    assert "non-empty list of genes" in r["message"]


def test_valid_dotplot_still_succeeds(tmp_path):
    a = _adata()
    out = str(tmp_path / "dotplot_markers.png")
    rj, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "dotplot", "genes": ["SFTPC", "C1QB"], "color_by": "cell_type",
         "output_path": out, "include_image": False},
        a,
    )
    r = json.loads(rj)
    assert r["status"] == "ok"
    import os
    assert os.path.exists(r["output_path"])
