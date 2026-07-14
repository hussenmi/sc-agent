"""Subprocess worker for diffxpy differential-expression tests.

diffxpy (and its ``batchglm``/TensorFlow backend) is a frozen 2020-era stack that
conflicts with scagent's modern scanpy / anndata / RAPIDS dependencies, so it is
installed in a **separate environment** and driven across a process boundary —
the same isolation model as CellBender (a foreign interpreter), not scVI (which
shares this env and is subprocessed only to isolate its CUDA context). See
``scagent/batch/diffxpy.py`` for the parent side.

Not a public API. Invoked *by file path* (never ``-m``) so the foreign
interpreter does not import the ``scagent`` package::

    <diffxpy-env python> /path/to/scagent/batch/_diffxpy_worker.py <spec.json>

The spec is a JSON file naming its inputs and outputs. The worker writes the
per-gene result table to ``result_csv`` and prints ``DIFFXPY_WORKER_OK`` on
success; on failure it writes the traceback to ``error_out`` and exits non-zero.

The worker deliberately depends only on numpy / scipy / pandas / anndata /
diffxpy — nothing from the rest of scagent — so it can run under the foreign
interpreter, which does not have scagent installed.
"""

import json
import os
import sys

# Invoked by file path, so sys.path[0] is this script's own directory
# (scagent/batch/), which contains modules named diffxpy.py, scvi.py, harmony.py,
# entropy.py … Those would SHADOW the installed diffxpy package (and any similarly
# named dependency). Strip this directory from sys.path before importing anything
# third-party so imports resolve against the diffxpy env's site-packages.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]


def _load_matrix(spec: dict):
    import numpy as np

    npz = spec.get("matrix_npz")
    if npz and os.path.exists(npz):
        from scipy import sparse

        # diffxpy/batchglm are happiest with a dense array; these subsets are one
        # sample's cells, so densifying is cheap and avoids sparse-path surprises.
        return np.asarray(sparse.load_npz(npz).todense(), dtype=np.float64)
    return np.asarray(np.load(spec["matrix_npy"]), dtype=np.float64)


def _patch_dask_writable(np) -> None:
    """Make dask's ``.compute()`` return writable arrays.

    batchglm 0.7.4's numpy IRLS backend does in-place assignment into arrays it
    obtains via ``dask_array.compute()``. Modern numpy hands back read-only arrays
    there, so the Wald GLM crashes with ``assignment destination is read-only``.
    We wrap ``Array.compute`` to return a writable copy. Contained to
    this isolated worker process; the copy cost is negligible for the small
    per-sample matrices we test. No effect on the rank / t-test paths (they don't
    use dask), so it is always safe to install.
    """
    try:
        import dask.array.core as _dc
    except Exception:
        return
    if getattr(_dc.Array.compute, "_scagent_writable", False):
        return
    _orig = _dc.Array.compute

    def _writable_compute(self, **kw):
        r = _orig(self, **kw)
        if isinstance(r, np.ndarray) and not r.flags.writeable:
            r = np.array(r)  # writable copy
        return r

    _writable_compute._scagent_writable = True
    _dc.Array.compute = _writable_compute


def _run(spec: dict) -> None:
    # Keep TensorFlow quiet and off the GPU: these are small per-sample tests, and
    # a CUDA context here would only risk clashing with the parent's GPU work.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import anndata as ad
    import numpy as np
    import pandas as pd

    _patch_dask_writable(np)

    import diffxpy.api as de

    matrix = _load_matrix(spec)
    genes = list(spec["genes"]) if "genes" in spec else None
    if genes is None:
        with open(spec["genes_json"]) as f:
            genes = json.load(f)
    group = np.asarray(np.load(spec["group_npy"])).ravel()
    if group.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"group vector length {group.shape[0]} != n_cells {matrix.shape[0]}"
        )

    # Two levels only: the cluster/state of interest ("group") vs everything it is
    # compared against ("rest"). diffxpy reports log2fc as the effect of the
    # NON-reference level with "group" as the (alphabetical) reference, so its raw
    # log2fc is oriented rest-vs-group — negative when a gene is higher in the
    # group of interest. We negate below so the returned log2fc is group-vs-rest
    # (positive = higher in the group of interest), which is the contract the
    # parent driver documents.
    labels = np.where(group.astype(bool), "group", "rest")
    obs = pd.DataFrame({"group": pd.Categorical(labels, categories=["group", "rest"])})
    var = pd.DataFrame(index=[str(g) for g in genes])
    adata = ad.AnnData(X=matrix, obs=obs, var=var)

    test_name = str(spec.get("test", "rank")).lower()
    noise_model = spec.get("noise_model")

    # Library-size factors for the count GLMs. diffxpy wants an explicit per-cell
    # array here (its "total_count" string is not a recognised keyword); we build
    # the standard total-count factor normalized to mean 1.
    size_factors = None
    if spec.get("size_factors") and test_name == "wald":
        totals = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        mean_total = float(totals.mean()) or 1.0
        size_factors = totals / mean_total

    if test_name == "rank":
        test = de.test.rank_test(data=adata, grouping="group")
    elif test_name in ("t-test", "t_test", "ttest"):
        test = de.test.t_test(data=adata, grouping="group")
    elif test_name == "wald":
        test = de.test.wald(
            data=adata,
            formula_loc="~1+group",
            factor_loc_totest="group",
            noise_model=noise_model or "nb",
            size_factors=size_factors,
        )
    else:
        # LRT is intentionally unsupported: in this frozen diffxpy 0.7.4 stack the
        # intercept-only reduced model trips a batchglm design-parsing bug, and for
        # a single tested coefficient the Wald test checks the same hypothesis.
        raise ValueError(
            f"unsupported diffxpy test '{test_name}'; expected 'rank', 't-test' or 'wald'"
        )

    summary = test.summary()

    # Normalize the columns we hand back so the parent never has to know which
    # diffxpy test produced them. diffxpy always provides gene/pval/qval/log2fc/mean.
    keep = [c for c in ("gene", "pval", "qval", "log2fc", "mean") if c in summary.columns]
    out = summary[keep].copy()
    if "gene" not in out.columns:
        out.insert(0, "gene", [str(g) for g in genes][: len(out)])
    if "log2fc" in out.columns:
        # Flip to group-vs-rest orientation (see the label comment above).
        out["log2fc"] = -out["log2fc"].astype(float)

    # NOTE: per-side mean and detection-rate (pct) columns are intentionally NOT
    # computed here. They are the shared trust/evidence columns and are added by
    # the parent via scagent.batch.diffxpy.attach_mean_pct, so the SAME
    # implementation feeds both the diffxpy and the scanpy-Wilcoxon paths.
    out.to_csv(spec["result_csv"], index=False)


def main() -> int:
    spec_path = sys.argv[1]
    with open(spec_path) as f:
        spec = json.load(f)
    try:
        _run(spec)
    except Exception:
        import traceback

        tb = traceback.format_exc()
        error_out = spec.get("error_out")
        if error_out:
            try:
                with open(error_out, "w") as f:
                    f.write(tb)
            except OSError:
                pass
        sys.stderr.write(tb)
        return 1
    print("DIFFXPY_WORKER_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
