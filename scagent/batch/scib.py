"""
scib-metrics integration benchmarking for scagent.

Wraps the scib_metrics Benchmarker to produce a standardised table of
bio-conservation and batch-correction scores across multiple embeddings.

Workshop reference: Session 5 — the lab uses this to decide which batch
correction method to keep before moving to annotation.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Embeddings produced by each correction method — used for auto-detection
_METHOD_EMBEDDING_MAP = {
    "harmony": "X_pca_harmony",
    "scvi": "X_scVI",
    "scanorama": "X_scanorama",
}

# Metrics that are slow and can be skipped for a quick run
_SLOW_METRICS = {"kbet_per_label"}


def _auto_detect_embeddings(adata) -> List[str]:
    """Return obsm keys that are worth benchmarking, in a sensible order."""
    candidates = ["X_pca"]  # always include uncorrected baseline
    for key in ("X_pca_harmony", "X_scVI", "X_scanorama"):
        if key in adata.obsm:
            candidates.append(key)
    return candidates


def run_scib_benchmark(
    adata,
    batch_key: str,
    label_key: str,
    embedding_keys: Optional[List[str]] = None,
    run_bio_conservation: bool = True,
    run_batch_correction: bool = True,
    fast: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Benchmark batch integration methods using scib-metrics.

    Runs the same Benchmarker setup used in workshop session 5:
    - Bio conservation: NMI, ARI (k-means), silhouette (label), cLISI
    - Batch correction: silhouette (batch), iLISI, kBET, graph connectivity, PCR

    Parameters
    ----------
    adata : AnnData
        Must contain corrected embeddings in obsm (e.g. X_pca_harmony, X_scVI).
    batch_key : str
        obs column with batch labels.
    label_key : str
        obs column with cell type / cluster labels for bio-conservation metrics.
    embedding_keys : list of str, optional
        obsm keys to benchmark. Auto-detected from known correction outputs if None.
        Always includes X_pca as the uncorrected baseline.
    run_bio_conservation : bool, default True
        Include bio-conservation metrics (NMI, ARI, silhouette label, cLISI).
    run_batch_correction : bool, default True
        Include batch correction metrics (silhouette batch, iLISI, kBET, graph connectivity).
    fast : bool, default False
        Skip slow metrics (kBET) for a quicker result.
    output_dir : str, optional
        If provided, saves results CSV and results table PNG here.

    Returns
    -------
    dict with keys: results_table (dict), best_method, scores_by_embedding,
    embeddings_benchmarked, output_csv, output_figure.
    """
    try:
        from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
    except ImportError:
        raise ImportError(
            "scib-metrics is required for integration benchmarking. "
            "Install with: pip install scib-metrics"
        )

    import numpy as np
    from pathlib import Path

    if embedding_keys is None:
        embedding_keys = _auto_detect_embeddings(adata)

    # Filter to keys actually present
    missing = [k for k in embedding_keys if k not in adata.obsm]
    if missing:
        logger.warning("Embedding keys not found in adata.obsm, skipping: %s", missing)
    embedding_keys = [k for k in embedding_keys if k in adata.obsm]

    if not embedding_keys:
        raise ValueError(
            "No valid embedding keys found in adata.obsm. "
            "Run batch correction first, or specify embedding_keys explicitly."
        )

    # Build metric configs
    bio_metrics = BioConservation() if run_bio_conservation else None
    batch_metrics = BatchCorrection() if run_batch_correction else None

    if fast and batch_metrics is not None:
        # Disable kBET (slow) by passing a custom config
        try:
            batch_metrics = BatchCorrection(kbet_per_label=False)
        except TypeError:
            # Older scib-metrics versions may not support this kwarg
            logger.warning("Could not disable kBET via constructor — running full metrics.")

    logger.info(
        "Running scib-metrics benchmark on %d embeddings: %s",
        len(embedding_keys),
        embedding_keys,
    )

    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        bio_conservation_metrics=bio_metrics,
        batch_correction_metrics=batch_metrics,
        embedding_obsm_keys=embedding_keys,
    )
    bm.benchmark()

    results_df = bm.get_results(min_max_scale=False)

    # Convert to JSON-serialisable dict
    results_table = results_df.round(4).to_dict()

    # Identify best method by total score (last row if present, else mean)
    scores_by_embedding: Dict[str, float] = {}
    try:
        # scib_metrics adds a "Total" row — use it if present
        if "Total" in results_df.index:
            total_row = results_df.loc["Total"]
        else:
            total_row = results_df.mean(axis=0)
        for emb in embedding_keys:
            if emb in total_row.index:
                scores_by_embedding[emb] = round(float(total_row[emb]), 4)
        best_method = max(scores_by_embedding, key=scores_by_embedding.get) if scores_by_embedding else None
    except Exception:
        best_method = None

    output_csv = None
    output_figure = None

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "scib_benchmark_results.csv"
        results_df.to_csv(csv_path)
        output_csv = str(csv_path)
        logger.info("Saved scib results to %s", csv_path)

        try:
            # plot_results_table returns a great_tables GT object, not a Figure.
            # Save it via GT.as_raw_html() → selenium/webshot, or fall back to
            # a plain matplotlib heatmap from the results DataFrame.
            fig_path = out / "scib_benchmark_table.png"
            table_obj = bm.plot_results_table(min_max_scale=False)
            saved = False

            # Try great_tables / GT save path
            try:
                table_obj.save(str(fig_path))
                saved = True
            except Exception:
                pass

            if not saved:
                # Fallback: draw results_df as a matplotlib table
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(
                    figsize=(max(6, len(embedding_keys) * 2), max(4, len(results_df) * 0.5 + 1))
                )
                ax.axis("off")
                tbl = ax.table(
                    cellText=results_df.round(3).values,
                    rowLabels=results_df.index,
                    colLabels=results_df.columns,
                    cellLoc="center",
                    loc="center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.auto_set_column_width(col=list(range(len(results_df.columns))))
                ax.set_title("scib-metrics benchmark", pad=12)
                fig.tight_layout()
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved = True

            if saved:
                output_figure = str(fig_path)
                logger.info("Saved scib results figure to %s", fig_path)
        except Exception as e:
            logger.warning("Could not save scib results figure: %s", e)

    return {
        "embeddings_benchmarked": embedding_keys,
        "results_table": results_table,
        "scores_by_embedding": scores_by_embedding,
        "best_method": best_method,
        "output_csv": output_csv,
        "output_figure": output_figure,
    }
