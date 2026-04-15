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

from typing import Optional
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
    import ctypes, glob, os, sys

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


def run_scvi(
    adata: AnnData,
    batch_key: str,
    n_latent: int = 30,
    max_epochs: int = 200,
    layer: str = "raw_counts",
    latent_key: str = "X_scVI",
    store_normalized: bool = False,
    use_gpu: bool = True,
    inplace: bool = True,
) -> Optional[AnnData]:
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
    max_epochs : int, default 200
        Training epochs. 200 is recommended for real data; use fewer
        only for quick tests (notebook demo used 10).
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
        or if a CUDA JIT/NVRTC error occurs during training.
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
    try:
        import scvi as scvi_tools
    except ImportError:
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

    n_batches = adata.obs[batch_key].nunique()
    logger.info(
        f"Running scVI batch correction: {n_batches} batches, "
        f"n_latent={n_latent}, max_epochs={max_epochs}"
    )

    # Determine accelerator
    accelerator = "cpu"
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                accelerator = "gpu"
                logger.info("GPU detected — using GPU acceleration for scVI training")
            else:
                logger.info("No GPU detected — using CPU for scVI training")
        except ImportError:
            logger.info("torch not importable — using CPU for scVI training")

    # Setup AnnData for scVI (registers the dataset with the model).
    # batch_key triggers scVI's proper batch integration (batch-specific decoder
    # parameters + batch-conditioned ELBO). Using categorical_covariate_keys instead
    # would treat batch as a weak covariate and produce worse correction.
    scvi_tools.model.SCVI.setup_anndata(
        adata,
        layer=layer,
        batch_key=batch_key,
    )

    # Pre-load nvrtc builtins so GPU JIT compilation can find them by name
    if accelerator == "gpu":
        _preload_nvrtc_builtins()
        import torch
        torch.set_float32_matmul_precision("medium")  # use Tensor Cores on A100/H100

    # Build and train the model
    model = scvi_tools.model.SCVI(adata, n_latent=n_latent)
    try:
        model.train(max_epochs=max_epochs, accelerator=accelerator, devices=1)
    except Exception as e:
        # NVRTC / CUDA JIT compilation errors happen when the PyTorch build's
        # bundled CUDA runtime is mismatched with the system's libnvrtc-builtins
        # library (e.g. "failed to open libnvrtc-builtins.so.X").  Retry on CPU.
        err_str = str(e)
        if accelerator == "gpu" and (
            "nvrtc" in err_str.lower()
            or "libnvrtc" in err_str.lower()
            or "cuda error" in err_str.lower()
        ):
            logger.warning(
                "GPU training failed due to CUDA JIT error (%s). "
                "Retrying on CPU — this will be slower.",
                err_str[:200],
            )
            # Re-build the model (training state is corrupt after a CUDA crash)
            scvi_tools.model.SCVI.setup_anndata(
                adata,
                layer=layer,
                categorical_covariate_keys=[batch_key],
            )
            model = scvi_tools.model.SCVI(adata, n_latent=n_latent)
            model.train(max_epochs=max_epochs, accelerator="cpu", devices=1)
        else:
            raise

    # Extract latent representation
    adata.obsm[latent_key] = model.get_latent_representation()
    logger.info(f"scVI latent representation stored in adata.obsm['{latent_key}']")

    # Optionally store normalized expression
    if store_normalized:
        adata.layers["scvi_normalized"] = model.get_normalized_expression(
            library_size=10_000
        )
        logger.info("scVI normalized expression stored in adata.layers['scvi_normalized']")

    logger.info("scVI batch correction complete")

    if not inplace:
        return adata
