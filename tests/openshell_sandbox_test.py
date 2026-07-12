"""run_code executed inside an OpenShell sandbox must honor the same contract as
the in-process path: committed adata on success, discarded reassignment on error,
and artifacts_created pointing at real host files.

These tests need a working OpenShell gateway + the scagent sandbox image, so they
skip cleanly on hosts without them (e.g. the Iris HPC / CI). Build the image with:
    docker build -t scagent-sbx:cpu docker/scagent-sandbox
"""
import json

import numpy as np
import pytest
import anndata as ad

from scagent.agent.tools import process_tool_call
from scagent.agent.sandbox import OpenShellSandbox, capability

_cap = capability()
pytestmark = pytest.mark.skipif(
    not _cap["available"], reason=f"OpenShell unavailable: {_cap['reason']}"
)


@pytest.fixture(scope="module")
def sandbox():
    sbx = OpenShellSandbox()
    yield sbx
    sbx.delete()


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(60, 25)).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"g{i}" for i in range(25)]
    a.obs_names = [f"c{i}" for i in range(60)]
    return a


def test_success_commits_adata_and_artifacts(sandbox, adata, tmp_path):
    # A non-destructive reassignment (copy + add an obs column) — exercises the
    # adata write-back/commit without tripping the destructive-subsetting guard.
    code = (
        "print('cells:', adata.n_obs)\n"
        "sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)\n"
        "adata = adata.copy()\n"
        "adata.obs['flag'] = 1\n"
        "write_report('note', '# hi\\n')\n"
        "print('flagged')\n"
    )
    rj, new_adata = process_tool_call(
        "run_code",
        {"code": code, "output_dir": str(tmp_path)},
        adata=adata,
        sandbox=sandbox,
    )
    r = json.loads(rj)
    assert r["status"] == "ok", r
    # the reassigned adata (with the new column) was committed back to the session
    assert r["shape"]["n_cells"] == 60
    assert "flag" in new_adata.obs.columns and (new_adata.obs["flag"] == 1).all()
    assert "cells: 60" in r["output"] and "flagged" in r["output"]
    # artifact was downloaded to a real host path under the run dir
    arts = r.get("artifacts_created", [])
    assert any(a["role"] == "report" for a in arts), arts
    from pathlib import Path
    assert all(Path(a["path"]).exists() for a in arts), arts


def test_error_discards_reassignment(sandbox, adata, tmp_path):
    code = "adata = adata[:5].copy()\nraise ValueError('boom 42')"
    rj, new_adata = process_tool_call(
        "run_code",
        {"code": code, "output_dir": str(tmp_path)},
        adata=adata,
        sandbox=sandbox,
    )
    r = json.loads(rj)
    assert r["status"] == "error"
    assert r["error_type"] == "ValueError"
    assert "boom 42" in r["message"]
    assert r["reassigned_adata_discarded"] is True
    # session keeps the ORIGINAL adata, not the 5-cell reassignment
    assert new_adata is adata
    assert new_adata.n_obs == 60
