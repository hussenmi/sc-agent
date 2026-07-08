"""Figure saves must never silently overwrite a previous figure.

A reused name (multi-resolution clustering UMAPs, pre/post-integration UMAPs, or
the model simply passing the same output_path twice) previously clobbered the
earlier figure. unique_output_path guarantees the file written is preserved, and
generate_figure returns the actual path it used.
"""

import json

import numpy as np
import pandas as pd
from anndata import AnnData

from scagent.agent.tools import process_tool_call, unique_output_path


def _adata_umap():
    n = 120
    obs = pd.DataFrame(
        {"leiden": pd.Categorical([str(i % 4) for i in range(n)])},
        index=[f"c{i}" for i in range(n)],
    )
    a = AnnData(X=np.zeros((n, 3), dtype=np.float32), obs=obs,
                var=pd.DataFrame(index=["A", "B", "C"]))
    a.obsm["X_umap"] = np.random.RandomState(0).randn(n, 2)
    return a


# --- unit: the helper ---------------------------------------------------


def test_unique_output_path_returns_input_when_free(tmp_path):
    p = str(tmp_path / "umap_leiden.png")
    assert unique_output_path(p) == p  # nothing there yet


def test_unique_output_path_increments_on_collision(tmp_path):
    p = tmp_path / "umap_leiden.png"
    p.write_bytes(b"")  # occupy the name
    got = unique_output_path(str(p))
    assert got == str(tmp_path / "umap_leiden_2.png")

    (tmp_path / "umap_leiden_2.png").write_bytes(b"")
    got2 = unique_output_path(str(p))
    assert got2 == str(tmp_path / "umap_leiden_3.png")


def test_unique_output_path_passthrough_for_empty():
    assert unique_output_path("") == ""
    assert unique_output_path(None) is None


# --- end-to-end: generate_figure twice at the same path ------------------


def test_generate_figure_twice_preserves_both(tmp_path):
    adata = _adata_umap()
    out = str(tmp_path / "umap_leiden.png")

    rj1, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "umap", "color_by": "leiden", "output_path": out, "include_image": False},
        adata,
    )
    rj2, _ = process_tool_call(
        "generate_figure",
        {"plot_type": "umap", "color_by": "leiden", "output_path": out, "include_image": False},
        adata,
    )
    r1, r2 = json.loads(rj1), json.loads(rj2)
    assert r1["status"] == "ok" and r2["status"] == "ok"

    # Second call must NOT reuse the first path — both figures survive on disk.
    assert r1["output_path"] != r2["output_path"]
    assert r2["output_path"].endswith("umap_leiden_2.png")
    assert (tmp_path / "umap_leiden.png").exists()
    assert (tmp_path / "umap_leiden_2.png").exists()
