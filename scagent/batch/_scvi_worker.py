"""Subprocess worker for scVI training.

Runs scVI/torch in a *separate process* so its CUDA context is created and torn
down entirely within this child. The parent scagent process therefore never
imports torch or initialises a torch CUDA context, which is what lets RAPIDS /
cuML (rapids_singlecell) run neighbors/UMAP/Leiden afterward without hitting the
sticky ``CUDA_ERROR_ILLEGAL_ADDRESS`` that torch+cuML coexistence triggers in one
process. See ``scagent/batch/scvi.py::run_scvi`` for the parent side.

Not a public API. Invoked as ``python -m scagent.batch._scvi_worker <spec.json>``.
The spec is a JSON file; outputs are written to the paths it names. The worker
prints ``SCVI_WORKER_OK`` on success and exits non-zero on failure (parent reads
stderr).
"""

import json
import sys


def _train(spec: dict) -> None:
    import logging

    import anndata as ad
    import numpy as np
    import scvi as scvi_tools

    # Reuse the parent's CUDA helpers (neither imports torch at module load).
    from scagent.batch.scvi import _preload_nvrtc_builtins, _select_gpu_device

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("scvi_worker")

    adata = ad.read_h5ad(spec["input_h5ad"])
    batch_key = spec["batch_key"]
    n_latent = int(spec["n_latent"])
    max_epochs = int(spec["max_epochs"])
    early_stopping = bool(spec["early_stopping"])
    use_gpu = bool(spec["use_gpu"])

    # Decide accelerator here (in the child) so the parent never touches torch.
    accelerator = "cpu"
    train_devices: object = 1
    if use_gpu:
        import torch

        if torch.cuda.is_available():
            accelerator = "gpu"
            device_index = _select_gpu_device()
            train_devices = [device_index]
            log.info("scVI worker: training on GPU %d", device_index)
        else:
            log.info("scVI worker: no GPU available, training on CPU")

    def _setup_and_build():
        # Counts live in X of the minimal AnnData the parent wrote (layer=None).
        scvi_tools.model.SCVI.setup_anndata(adata, batch_key=batch_key)
        return scvi_tools.model.SCVI(adata, n_latent=n_latent)

    if accelerator == "gpu":
        _preload_nvrtc_builtins()
        import torch

        torch.set_float32_matmul_precision("medium")

    model = _setup_and_build()
    try:
        model.train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=train_devices,
            early_stopping=early_stopping,
        )
    except Exception as e:  # noqa: BLE001 - inspect message to decide CPU retry
        err = str(e)
        if accelerator == "gpu" and (
            "nvrtc" in err.lower() or "libnvrtc" in err.lower() or "cuda error" in err.lower()
        ):
            log.warning("scVI GPU training failed (%s); retrying on CPU.", err[:200])
            model = _setup_and_build()  # training state is corrupt after a CUDA crash
            model.train(
                max_epochs=max_epochs,
                accelerator="cpu",
                devices=1,
                early_stopping=early_stopping,
            )
        else:
            raise

    np.save(spec["latent_out"], np.asarray(model.get_latent_representation(), dtype=np.float32))

    if spec.get("store_normalized"):
        norm = model.get_normalized_expression(library_size=10_000)
        np.save(spec["normalized_out"], np.asarray(norm, dtype=np.float32))
        with open(spec["columns_out"], "w") as f:
            json.dump([str(c) for c in norm.columns], f)


def main() -> int:
    spec_path = sys.argv[1]
    with open(spec_path) as f:
        spec = json.load(f)
    try:
        _train(spec)
    except Exception:
        # Stream the traceback to the (inherited) terminal AND persist it so the
        # parent can surface it on a non-zero exit.
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
    print("SCVI_WORKER_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
