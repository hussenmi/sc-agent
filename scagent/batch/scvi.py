"""
scVI batch correction for scagent.

scVI (single-cell Variational Inference) is a deep generative model that
learns a low-dimensional latent representation of cells while accounting
for batch effects. Unlike Harmony (which corrects PCA embeddings) or
Scanorama (which uses MNN), scVI models the raw count generating process
directly using a variational autoencoder.

Advantages over Harmony/Scanorama:
- Models raw count noise explicitly (negative binomial likelihood)
- Can correct complex, non-linear batch effects
- Produces normalized expression estimates alongside the latent space

Requirement: raw integer counts must be available in adata.layers['raw_counts']
"""

import logging

from anndata import AnnData

logger = logging.getLogger(__name__)


def _preload_nvrtc_builtins() -> None:
    """
    Pre-load libnvrtc-builtins into the global symbol table before scVI GPU training.

    PyTorch's CUDA JIT compiler calls dlopen("libnvrtc-builtins.so.X") by bare name
    at compile time.  In HPC environments the library lives inside a pip-installed
    nvidia package (site-packages/nvidia/cu*/lib/) that is not on LD_LIBRARY_PATH,
    so dlopen fails with "failed to open libnvrtc-builtins.so.X".

    Loading the library via ctypes with RTLD_GLOBAL before training causes glibc's
    dynamic linker to satisfy the subsequent bare-name dlopen from the already-loaded
    handle, bypassing the LD_LIBRARY_PATH lookup entirely.
    """
    import ctypes
    import glob
    import os
    import sys

    searched = []
    for sp in sys.path:
        for pattern in [
            os.path.join(sp, "nvidia", "cu*", "lib", "libnvrtc-builtins.so.*"),
            os.path.join(sp, "nvidia", "cuda_nvrtc", "lib", "libnvrtc-builtins.so.*"),
        ]:
            for lib in sorted(glob.glob(pattern), reverse=True):  # prefer highest version
                searched.append(lib)
                if not os.path.exists(lib):
                    continue
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    logger.debug("Pre-loaded nvrtc builtins: %s", lib)
                    return
                except OSError:
                    pass

    logger.debug("libnvrtc-builtins not found in site-packages (searched %d paths); "
                 "GPU training may fail if the library is not on LD_LIBRARY_PATH.", len(searched))


def _select_gpu_device() -> int:
    """Pick the visible CUDA device with the most free memory.

    Lightning otherwise defaults scVI training to ``cuda:0``, which on a shared
    HPC node is frequently the busiest GPU (e.g. colocated with the vLLM server
    backing the agent's own LLM). Selecting by free memory steers training onto
    an idle GPU when one exists.

    Free memory is read via NVML (``pynvml``), which queries the driver directly
    and does *not* allocate a CUDA context — unlike ``torch.cuda.mem_get_info``,
    which leaves a ~0.5 GB context on every device it probes. NVML enumerates
    physical GPUs, so we map them onto the CUDA-visible ordering to honor
    ``CUDA_VISIBLE_DEVICES``; the returned index is what Lightning's
    ``devices=[idx]`` expects. Falls back to torch (and finally device 0) when
    NVML is unavailable or the visible set uses non-integer (UUID) tokens.
    """
    import os

    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if visible:
                # Map each visible CUDA ordinal to its physical NVML index.
                # UUID tokens aren't integer indices -> ValueError -> torch fallback.
                physical = [int(token) for token in visible.split(",") if token.strip()]
            else:
                physical = list(range(pynvml.nvmlDeviceGetCount()))

            best_index, best_free = 0, -1
            for cuda_index, physical_index in enumerate(physical):
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
                free = pynvml.nvmlDeviceGetMemoryInfo(handle).free
                if free > best_free:
                    best_index, best_free = cuda_index, free
            return best_index
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # Fallback: torch query (creates a CUDA context per probed device).
    import torch

    best_index, best_free = 0, -1
    for index in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:
            continue
        if free > best_free:
            best_index, best_free = index, free
    return best_index


# Default DDP strategy for multi-GPU scVI. ``find_unused_parameters_true`` is what
# scvi-tools documents for non-interactive (script) multi-GPU runs; our worker is a
# ``python -m`` subprocess, not a notebook, so this is the right variant. Override
# with SCAGENT_SCVI_STRATEGY (e.g. ``ddp_notebook_find_unused_parameters_true``).
_DEFAULT_SCVI_DDP_STRATEGY = "ddp_find_unused_parameters_true"


def _resolve_n_devices(n_devices: int | None) -> int:
    """Resolve how many GPUs scVI should train on.

    Multi-GPU is strictly opt-in. When ``n_devices`` is None we read the
    ``SCAGENT_SCVI_DEVICES`` environment variable (mirroring how ``SCAGENT_GPU``
    gates the RAPIDS path): unset/``1`` -> single GPU (the default, picks the
    least-busy device); ``-1`` or ``all`` -> every visible GPU; ``N`` -> up to N
    GPUs. An explicit ``n_devices`` argument always wins over the env var.
    """
    if n_devices is not None:
        return n_devices

    import os

    raw = (os.environ.get("SCAGENT_SCVI_DEVICES") or "").strip().lower()
    if not raw or raw == "1":
        return 1
    if raw in ("-1", "all"):
        return -1
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid SCAGENT_SCVI_DEVICES=%r; using a single GPU.", raw)
        return 1
    return value if value != 0 else 1


def run_scvi(
    adata: AnnData,
    batch_key: str,
    n_latent: int = 30,
    max_epochs: int | None = None,
    layer: str = "raw_counts",
    latent_key: str = "X_scVI",
    store_normalized: bool = False,
    use_gpu: bool = True,
    n_devices: int | None = None,
    use_hvg: bool = True,
    early_stopping: bool = True,
    diagnostics_dir: str | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """
    Run scVI batch correction.

    Trains a variational autoencoder on raw counts, conditioning on the
    batch covariate to learn a batch-corrected latent representation.
    After training, neighbors and UMAP should be recomputed using the
    latent representation (latent_key).

    Parameters
    ----------
    adata : AnnData
        AnnData object. Must contain raw integer counts in `layer`.
    batch_key : str
        Column in adata.obs containing batch labels.
    n_latent : int, default 30
        Dimensionality of the latent space. 30 matches PCA dims used
        for Harmony/Scanorama for comparability.
    max_epochs : int or None, default None
        Upper bound on training epochs. When None, scVI's own cell-count
        heuristic is used (``get_max_epochs_heuristic``: 400 for <=20k cells,
        decaying above) — the library's recommended default. Early stopping
        still halts sooner once the validation ELBO plateaus, so this is only a
        cap. Pass an explicit integer for a quick test (e.g. 10) or to override.
    layer : str, default 'raw_counts'
        Layer containing raw integer counts. scVI requires non-normalized
        counts — it models the count generating process directly.
    latent_key : str, default 'X_scVI'
        Key to store the latent representation in adata.obsm.
    store_normalized : bool, default False
        If True, store scVI-normalized expression in adata.layers['scvi_normalized'].
        Useful for downstream DEG but adds memory overhead.
    use_gpu : bool, default True
        Use GPU if available. Falls back to CPU automatically if no GPU found
        or if a CUDA JIT/NVRTC error occurs during training. When using a single
        GPU, the device with the most free memory is selected to avoid colliding
        with a busy GPU (e.g. one hosting an LLM inference server).
    n_devices : int or None, default None
        How many GPUs to train on. Multi-GPU training is opt-in. None reads the
        ``SCAGENT_SCVI_DEVICES`` env var (unset/``1`` -> single GPU; ``-1``/``all``
        -> every visible GPU; ``N`` -> up to N GPUs). ``1`` keeps the single-GPU
        path (least-busy device). Values >1 or -1 use scvi-tools' DDP backend
        (requires scvi-tools >=1.3). DDP shards each minibatch across GPUs; note
        it **cannot use early stopping** (the worker disables it and runs the full
        ``max_epochs`` cap) and adds coordination overhead, so it only pays off on
        large datasets and a dedicated multi-GPU allocation. Restrict which GPUs
        are eligible with ``CUDA_VISIBLE_DEVICES``.
    use_hvg : bool, default True
        Train only on highly variable genes (``adata.var['highly_variable']``)
        when that flag is present. scVI on the HVG subset is several-fold
        faster than on the full gene set and is standard practice. The latent
        representation is still stored back on the full ``adata``. Falls back to
        all genes if the flag is missing or selects too few genes.
    early_stopping : bool, default True
        Stop training once the validation ELBO plateaus instead of always
        running the full ``max_epochs``.
    diagnostics_dir : str or None, default None
        If given, the training worker writes a per-epoch loss history CSV
        (``scvi_training_history.csv``) and a train-vs-validation ELBO
        convergence plot (``scvi_training_loss.png``) here. Training metrics
        (resolved epoch cap, epochs actually trained, early-stop flag, final
        ELBOs, overfitting warning, artifact paths) are stored on
        ``adata.uns['scvi_training']`` regardless.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.

    Raises
    ------
    ImportError
        If scvi-tools is not installed.
    ValueError
        If required layer or batch_key is missing.
    """
    # Availability check only — do NOT `import scvi` here, as that imports torch
    # into the parent process. Training happens in the _scvi_worker subprocess.
    import importlib.util

    if importlib.util.find_spec("scvi") is None:
        raise ImportError(
            "scvi-tools is not installed. Install with: pip install scvi-tools"
        )

    if not inplace:
        adata = adata.copy()

    # Validate inputs
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    if layer not in adata.layers:
        raise ValueError(
            f"Layer '{layer}' not found. scVI requires raw integer counts. "
            f"Available layers: {list(adata.layers.keys())}. "
            f"If counts are in adata.X, copy them first: "
            f"adata.layers['raw_counts'] = adata.X.copy()"
        )

    import os

    n_batches = adata.obs[batch_key].nunique()
    epochs_label = max_epochs if max_epochs is not None else "auto (scVI heuristic)"
    resolved_n_devices = _resolve_n_devices(n_devices)
    scvi_strategy = (
        os.environ.get("SCAGENT_SCVI_STRATEGY") or _DEFAULT_SCVI_DDP_STRATEGY
    ).strip()
    logger.info(
        f"Running scVI batch correction: {n_batches} batches, "
        f"n_latent={n_latent}, max_epochs={epochs_label}"
    )
    if resolved_n_devices != 1:
        gpu_label = "all visible" if resolved_n_devices < 0 else str(resolved_n_devices)
        logger.info(
            "scVI multi-GPU training requested (n_devices=%s -> %s GPUs, strategy=%s). "
            "Early stopping is disabled under DDP; training will run the full epoch cap.",
            n_devices if n_devices is not None else f"env:{resolved_n_devices}",
            gpu_label,
            scvi_strategy,
        )

    # Train on the HVG subset when available — scVI on a few thousand HVGs is
    # several-fold faster than on the full ~30k-gene matrix, and is the standard
    # workflow. We train on a gene-subset copy but keep all cells in order, so
    # the per-cell latent maps straight back onto the full adata.
    train_adata = adata
    if use_hvg and "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
        n_hvg = int(hvg_mask.sum())
        if n_hvg >= 100:
            train_adata = adata[:, hvg_mask].copy()
            logger.info(
                "Training scVI on %d highly variable genes (of %d total)",
                n_hvg, adata.n_vars,
            )
        else:
            logger.info(
                "Only %d highly variable genes flagged; training scVI on all %d genes",
                n_hvg, adata.n_vars,
            )

    # Train scVI in a SEPARATE PROCESS. scVI/torch creates a CUDA context that, in
    # one process, poisons a subsequent RAPIDS/cuML (rapids_singlecell) session with
    # a sticky CUDA_ERROR_ILLEGAL_ADDRESS. Isolating torch in a child process means
    # its CUDA context is torn down on exit and never coexists with cuML here. We
    # therefore never import torch in this (parent) process — even a
    # torch.cuda.is_available() check would create a contaminating context.
    import json
    import shutil
    import subprocess
    import sys
    import tempfile

    import anndata as ad
    import numpy as np

    # Hand the worker a minimal AnnData: counts (HVG subset) in X + the batch column.
    counts = train_adata.layers[layer]
    child = ad.AnnData(X=counts.copy())
    child.obs[batch_key] = train_adata.obs[batch_key].to_numpy()
    child.var_names = train_adata.var_names.copy()

    work_dir = tempfile.mkdtemp(prefix="scagent_scvi_")
    try:
        input_h5ad = os.path.join(work_dir, "train.h5ad")
        latent_out = os.path.join(work_dir, "latent.npy")
        normalized_out = os.path.join(work_dir, "normalized.npy")
        columns_out = os.path.join(work_dir, "columns.json")
        error_out = os.path.join(work_dir, "error.txt")
        metrics_out = os.path.join(work_dir, "metrics.json")
        spec_path = os.path.join(work_dir, "spec.json")
        child.write(input_h5ad)
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "input_h5ad": input_h5ad,
                    "batch_key": batch_key,
                    "n_latent": n_latent,
                    "max_epochs": max_epochs,
                    "early_stopping": early_stopping,
                    "use_gpu": use_gpu,
                    "n_devices": resolved_n_devices,
                    "strategy": scvi_strategy,
                    "store_normalized": store_normalized,
                    "latent_out": latent_out,
                    "normalized_out": normalized_out,
                    "columns_out": columns_out,
                    "error_out": error_out,
                    "metrics_out": metrics_out,
                    "diagnostics_dir": diagnostics_dir,
                },
                f,
            )

        logger.info("Launching scVI training subprocess (isolated CUDA context)")
        # Inherit the parent's stdout/stderr (do NOT capture) so scVI's live
        # training progress bar renders in the terminal. The worker writes any
        # fatal traceback to error_out, which we surface on a non-zero exit.
        proc = subprocess.run([sys.executable, "-m", "scagent.batch._scvi_worker", spec_path])
        if proc.returncode != 0:
            detail = ""
            if os.path.exists(error_out):
                with open(error_out) as f:
                    detail = f.read()[-2000:]
            raise RuntimeError(
                f"scVI training subprocess failed (exit {proc.returncode}).\n{detail}"
            )

        # Latent maps straight onto the full adata (cell order unchanged).
        adata.obsm[latent_key] = np.load(latent_out)
        logger.info(f"scVI latent representation stored in adata.obsm['{latent_key}']")

        # Surface training diagnostics (epochs trained, convergence, artifact paths).
        if os.path.exists(metrics_out):
            with open(metrics_out) as f:
                training_metrics = json.load(f)
            adata.uns["scvi_training"] = training_metrics
            # Record the true backend the worker trained on (accounts for a
            # GPU->CPU retry) so the tool result reports it, not the request.
            from ..core.gpu import record_backend
            _accel = training_metrics.get("accelerator")
            if _accel:
                record_backend(f"scvi:{_accel}")
            if training_metrics.get("epochs_trained") is not None:
                logger.info(
                    "scVI trained %s/%s epochs (early_stopped=%s)%s",
                    training_metrics.get("epochs_trained"),
                    training_metrics.get("resolved_max_epochs"),
                    training_metrics.get("early_stopped"),
                    "; " + training_metrics["overfitting_warning"]
                    if training_metrics.get("overfitting_warning")
                    else "",
                )

        if store_normalized:
            norm = np.load(normalized_out)
            with open(columns_out) as f:
                norm_columns = json.load(f)
            # Map the modeled (HVG) genes back onto the full var axis; genes scVI
            # did not model stay zero so the layer aligns with adata.
            full = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
            col_idx = adata.var_names.get_indexer(norm_columns)
            full[:, col_idx] = norm
            adata.layers["scvi_normalized"] = full
            logger.info("scVI normalized expression stored in adata.layers['scvi_normalized']")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("scVI batch correction complete")

    if not inplace:
        return adata
