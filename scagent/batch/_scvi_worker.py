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


def _plot_loss(train_curve, val_curve, metrics: dict, path: str) -> None:
    """Render the scVI train/validation ELBO convergence curve to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    if train_curve is not None:
        ax.plot(train_curve.index, train_curve.values, label="train ELBO", color="#1f77b4")
    if val_curve is not None:
        ax.plot(val_curve.index, val_curve.values, label="validation ELBO", color="#d62728")
        best = metrics.get("best_val_epoch")
        if best is not None:
            ax.axvline(best, color="gray", linestyle="--", linewidth=1, label=f"best val epoch {best}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss (lower = better)")
    if metrics.get("early_stopped"):
        title = f"scVI convergence — early-stopped at {metrics.get('epochs_trained')} epochs"
    else:
        title = f"scVI convergence — ran full {metrics.get('resolved_max_epochs')}-epoch cap"
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_diagnostics(model, resolved_max_epochs: int, spec: dict, log) -> dict:
    """Extract scVI's per-epoch loss history; save a CSV + convergence plot + metrics.

    scVI logs ``model.history`` as a dict of per-epoch DataFrames (``elbo_train``,
    ``elbo_validation``, reconstruction/KL terms). We persist the full table, plot the
    train-vs-validation ELBO so convergence and overfitting are visible, and record how
    many epochs actually ran (vs the cap) so a run that hit the cap — i.e. did *not*
    converge — is obvious rather than hidden behind the live progress bar.
    """
    import os

    metrics: dict = {"resolved_max_epochs": int(resolved_max_epochs)}
    try:
        history = model.history or {}

        def _series(key):
            df = history.get(key)
            if df is None or len(df) == 0:
                return None
            return df[df.columns[0]]

        train_curve = _series("elbo_train")
        if train_curve is None:
            train_curve = _series("train_loss_epoch")
        val_curve = _series("elbo_validation")
        if val_curve is None:
            val_curve = _series("validation_loss")

        epochs_trained = int(len(train_curve)) if train_curve is not None else None
        metrics["epochs_trained"] = epochs_trained
        metrics["early_stopped"] = bool(
            epochs_trained is not None and epochs_trained < resolved_max_epochs
        )
        if train_curve is not None:
            metrics["final_elbo_train"] = float(train_curve.iloc[-1])
        metrics["overfitting_warning"] = None
        if val_curve is not None:
            metrics["final_elbo_validation"] = float(val_curve.iloc[-1])
            best_idx = int(val_curve.values.argmin())
            best_val = float(val_curve.iloc[best_idx])
            last_val = float(val_curve.iloc[-1])
            metrics["best_val_epoch"] = best_idx
            metrics["best_elbo_validation"] = best_val
            # Validation ELBO that rises meaningfully after its best epoch => overfitting.
            if best_idx < len(val_curve) - 1 and best_val != 0 and (last_val - best_val) / abs(best_val) > 0.02:
                metrics["overfitting_warning"] = (
                    f"validation ELBO was best at epoch {best_idx} but rose "
                    f"{(last_val - best_val) / abs(best_val) * 100:.1f}% by the final epoch — "
                    "possible overfitting; consider fewer epochs or tighter early stopping"
                )

        diag_dir = spec.get("diagnostics_dir")
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)
            merged = None
            for key, df in history.items():
                col = df.rename(columns={df.columns[0]: key})
                merged = col if merged is None else merged.join(col, how="outer")
            if merged is not None:
                csv_path = os.path.join(diag_dir, "scvi_training_history.csv")
                merged.to_csv(csv_path, index_label="epoch")
                metrics["history_csv"] = csv_path
            if train_curve is not None or val_curve is not None:
                plot_path = os.path.join(diag_dir, "scvi_training_loss.png")
                _plot_loss(train_curve, val_curve, metrics, plot_path)
                metrics["loss_plot"] = plot_path
    except Exception as e:  # noqa: BLE001 - diagnostics must never fail the run
        log.warning("scVI diagnostics failed: %s", e)
        metrics.setdefault("diagnostics_error", str(e))

    metrics_out = spec.get("metrics_out")
    if metrics_out:
        try:
            with open(metrics_out, "w") as f:
                json.dump(metrics, f)
        except OSError as e:
            log.warning("Could not write scVI metrics file: %s", e)
    log.info(
        "scVI training summary: %s",
        {k: metrics.get(k) for k in ("epochs_trained", "resolved_max_epochs", "early_stopped", "final_elbo_validation")},
    )
    return metrics


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
    max_epochs_spec = spec.get("max_epochs")
    early_stopping = bool(spec["early_stopping"])
    use_gpu = bool(spec["use_gpu"])

    # When max_epochs is left unset, defer to scVI's own cell-count heuristic
    # (400 for <=20k cells, decaying above) instead of a fixed cap. Early stopping
    # still governs the actual run length; this only sets the upper bound.
    if max_epochs_spec is None:
        from scvi.model._utils import get_max_epochs_heuristic

        resolved_max_epochs = int(get_max_epochs_heuristic(adata.n_obs))
        log.info(
            "scVI worker: max_epochs not provided; using scVI heuristic = %d epochs for %d cells",
            resolved_max_epochs,
            adata.n_obs,
        )
    else:
        resolved_max_epochs = int(max_epochs_spec)

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
            max_epochs=resolved_max_epochs,
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
                max_epochs=resolved_max_epochs,
                accelerator="cpu",
                devices=1,
                early_stopping=early_stopping,
            )
        else:
            raise

    np.save(spec["latent_out"], np.asarray(model.get_latent_representation(), dtype=np.float32))

    # Persist training diagnostics (loss history + convergence plot + metrics). Never
    # let a diagnostics failure sink an otherwise-successful run.
    _write_diagnostics(model, resolved_max_epochs, spec, log)

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
