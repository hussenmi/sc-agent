"""
Quality control pipeline for scagent.

Implements lab's best practices for single-cell QC:
- QC metrics calculation (library size, gene counts, MT/ribo content)
- Cell filtering based on QC metrics
- Gene filtering
- Doublet detection with Scrublet (batch-aware)
"""

from typing import Optional, Union, List
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import logging

from ..config.defaults import QC_DEFAULTS
from . import genes as _genes
from .gpu import gpu_available, on_gpu
from .inspector import resolve_batch_metadata

logger = logging.getLogger(__name__)


def _gene_names_for_prefix_matching(adata: AnnData) -> List[str]:
    """Gene names to match MT/ribo (and similar) symbol prefixes against.

    Returns ``var_names`` when they are already gene symbols; otherwise the
    dataset's content-validated symbol column (``feature_name``/``gene_symbols``/…)
    when one exists, so MT/ribo detection works on Ensembl- or Entrez-indexed
    objects instead of silently matching nothing. Falls back to ``var_names`` when
    no symbol column is found. A dominant multi-genome prefix (``GRCh38_``) is
    stripped so ``MT-``/``RPS`` still match. See core.genes.
    """
    if _genes.infer_id_format(adata.var_names) == "symbol":
        names = [str(n) for n in adata.var_names]
    else:
        col = _genes.find_symbol_column(adata)
        if col is not None:
            series = adata.var[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                series = series.astype(str)
            names = [str(v) for v in series.tolist()]
        else:
            names = [str(n) for n in adata.var_names]
    stripped, _ = _genes.strip_genome_prefix(names)
    return stripped


def _safe_scrublet_n_prin_comps(
    adata: AnnData,
    batch_key: Optional[str],
    requested_n_prin_comps: int,
) -> int:
    """
    Pick a Scrublet PCA dimension that is safe for the current dataset.

    Scrublet runs PCA internally and can fail on small datasets or small
    per-batch subsets when ``n_prin_comps`` is larger than the effective
    sample/feature limit. We bound the requested value conservatively so QC
    preview can degrade gracefully instead of erroring.
    """
    max_cells = adata.n_obs
    if batch_key and batch_key in adata.obs:
        batch_sizes = adata.obs[batch_key].value_counts()
        if len(batch_sizes) > 0:
            max_cells = int(batch_sizes.min())

    upper_bound = min(int(requested_n_prin_comps), int(adata.n_vars) - 1, int(max_cells) - 1)
    if upper_bound < 2:
        raise ValueError(
            "Scrublet requires at least 3 cells and 3 genes in the smallest analysis unit "
            f"(current limit: cells={max_cells}, genes={adata.n_vars})."
        )
    return upper_bound


def calculate_qc_metrics(
    adata: AnnData,
    mt_prefix: str = QC_DEFAULTS.mt_prefix,
    ribo_prefixes: tuple = QC_DEFAULTS.ribo_prefixes,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate QC metrics including MT and ribosomal content.

    Parameters
    ----------
    adata : AnnData
        AnnData object with count data.
    mt_prefix : str, default 'MT-'
        Prefix for mitochondrial genes.
    ribo_prefixes : tuple, default ('RPS', 'RPL')
        Prefixes for ribosomal genes.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    logger.info("Calculating QC metrics...")

    # Gene NAMES to match MT/ribo prefixes against. Critical: var_names are often
    # Ensembl IDs (ENSG…), against which 'MT-'/'RPS'/'RPL' match NOTHING — which
    # silently yields pct_counts_mt/pct_counts_ribo = 0 for every cell (the empty
    # QC panels in run_2026_07_02_150701, misread as "snRNA-seq excludes MT/ribo").
    # When var_names are not symbols, match against the dataset's symbol column
    # (feature_name/gene_symbols/…, content-validated). A dominant genome prefix
    # (e.g. 'GRCh38_MT-CO1') is stripped so the prefix still matches.
    match_names = _gene_names_for_prefix_matching(adata)

    def _prefix_mask(prefixes) -> pd.Series:
        # Case-insensitive so mouse ('mt-', 'Rps'/'Rpl') is caught as well as human.
        prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
        lower = tuple(p.lower() for p in prefixes)
        return pd.Series(
            [str(n).lower().startswith(lower) for n in match_names],
            index=adata.var_names,
        )

    # Mitochondrial genes
    adata.var['mt'] = _prefix_mask(mt_prefix)
    n_mt_genes = int(adata.var['mt'].sum())
    logger.info(f"Found {n_mt_genes} mitochondrial genes")

    # Ribosomal genes
    adata.var['ribo'] = _prefix_mask(ribo_prefixes)
    n_ribo_genes = int(adata.var['ribo'].sum())
    logger.info(f"Found {n_ribo_genes} ribosomal genes")

    # Calculate MT and ribo content; log1p=True adds log1p_total_counts and
    # log1p_n_genes_by_counts columns which give compact 0–10 values for violin plots.
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt', 'ribo'],
        percent_top=None,
        log1p=True,
        inplace=True,
    )

    logger.info("QC metrics calculated successfully")

    if not inplace:
        return adata


def filter_cells_by_mt(
    adata: AnnData,
    mt_threshold: Optional[float] = None,
    data_type: str = 'auto',
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Filter cells based on mitochondrial content.

    Parameters
    ----------
    adata : AnnData
        AnnData object with MT metrics calculated.
    mt_threshold : float, optional
        Maximum allowed MT percentage. If None, uses default based on data type.
    data_type : str, default 'auto'
        Type of data: 'cells', 'nuclei', or 'auto' (detect automatically).
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if 'pct_counts_mt' not in adata.obs.columns:
        raise ValueError("MT metrics not calculated. Run calculate_qc_metrics first.")

    if not inplace:
        adata = adata.copy()

    # Determine threshold
    if mt_threshold is None:
        if data_type == 'auto':
            # Auto-detect based on MT distribution
            median_mt = np.median(adata.obs['pct_counts_mt'])
            if median_mt < 2.0:
                data_type = 'nuclei'
            else:
                data_type = 'cells'
            logger.info(f"Auto-detected data type: {data_type}")

        if data_type == 'nuclei':
            mt_threshold = QC_DEFAULTS.mt_threshold_nuclei
        else:
            mt_threshold = QC_DEFAULTS.mt_threshold_cells

    n_before = adata.n_obs
    adata._inplace_subset_obs(adata.obs['pct_counts_mt'] < mt_threshold)
    n_after = adata.n_obs
    n_removed = n_before - n_after

    logger.info(
        f"Filtered cells by MT content (<{mt_threshold}%): "
        f"{n_before:,} -> {n_after:,} ({n_removed:,} removed)"
    )

    if not inplace:
        return adata


def filter_genes(
    adata: AnnData,
    min_cells: Optional[float] = None,
    remove_ribo: bool = True,
    remove_mt: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Filter genes based on expression and optionally remove ribo/MT genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    min_cells : float, optional
        Minimum cells a gene must be expressed in. If None, no min-cell filtering is applied.
    remove_ribo : bool, default True
        Remove ribosomal genes.
    remove_mt : bool, default False
        Remove mitochondrial genes.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    n_before = adata.n_vars

    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)
        logger.info(f"Filtered genes by min_cells={min_cells:.0f}: {n_before:,} -> {adata.n_vars:,}")

    # Names to match against — symbol column when var_names are Ensembl/Entrez, so
    # removal doesn't silently miss every gene (see _gene_names_for_prefix_matching).
    match_names = None

    def _prefix_mask(prefixes) -> pd.Series:
        nonlocal match_names
        if match_names is None:
            match_names = _gene_names_for_prefix_matching(adata)
        prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
        lower = tuple(p.lower() for p in prefixes)
        return pd.Series(
            [str(n).lower().startswith(lower) for n in match_names],
            index=adata.var_names,
        )

    # Remove ribosomal genes
    if remove_ribo:
        if 'ribo' not in adata.var.columns:
            adata.var['ribo'] = _prefix_mask(QC_DEFAULTS.ribo_prefixes)

        n_ribo = adata.var['ribo'].sum()
        adata._inplace_subset_var(~adata.var['ribo'])
        logger.info(f"Removed {n_ribo} ribosomal genes")

    # Remove MT genes
    if remove_mt:
        if 'mt' not in adata.var.columns:
            adata.var['mt'] = _prefix_mask(QC_DEFAULTS.mt_prefix)

        n_mt = adata.var['mt'].sum()
        adata._inplace_subset_var(~adata.var['mt'])
        logger.info(f"Removed {n_mt} mitochondrial genes")

    logger.info(f"Final gene count: {adata.n_vars:,}")

    if not inplace:
        return adata


def _run_scrublet_per_batch(
    adata: AnnData,
    batch_key: str,
    expected_doublet_rate: float,
    sim_doublet_ratio: float,
    n_prin_comps: int,
    scrublet_min_counts: int,
    scrublet_min_cells: int,
    scrublet_min_gene_variability_pctl: float,
    random_state: int,
) -> None:
    """
    Run Scrublet independently on each batch and write scores back by positional index.

    sc.pp.scrublet(batch_key=...) reassigns scores via adata.obs.loc[sub.obs_names],
    which raises KeyError when obs_names contain non-standard separators (e.g.
    'BARCODE-1-Sample_3'). This function avoids that by tracking batch positions
    as a boolean mask and writing results directly into pre-allocated arrays.
    """
    import scrublet as scr
    import scipy.sparse as sp

    doublet_scores = np.zeros(adata.n_obs, dtype=np.float64)
    predicted_doublets = np.zeros(adata.n_obs, dtype=bool)

    batches = adata.obs[batch_key].unique()
    logger.info(f"Running Scrublet on {len(batches)} batches via per-batch loop")

    for batch in batches:
        mask = (adata.obs[batch_key] == batch).values
        sub = adata[mask]

        # Scrublet requires raw counts — prefer raw_counts layer over X,
        # which may be log-normalized or contain NaNs from outer-join fill.
        if 'raw_counts' in sub.layers:
            X = sub.layers['raw_counts']
        else:
            X = sub.X
        if sp.issparse(X):
            X = X.tocsc().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)
        # Replace any NaNs (from outer-join zero-fill after normalization) with 0
        if np.isnan(X).any() if not sp.issparse(X) else False:
            X = np.nan_to_num(X, nan=0.0)

        safe_n = min(n_prin_comps, X.shape[1] - 1, X.shape[0] - 1)
        if safe_n < 2:
            logger.warning(
                "Batch '%s' too small for Scrublet (cells=%d, genes=%d) — skipping.",
                batch, X.shape[0], X.shape[1],
            )
            continue

        try:
            scrub = scr.Scrublet(
                X,
                expected_doublet_rate=expected_doublet_rate,
                sim_doublet_ratio=sim_doublet_ratio,
                random_state=random_state,
            )
            scores, doublets = scrub.scrub_doublets(
                min_counts=scrublet_min_counts,
                min_cells=scrublet_min_cells,
                min_gene_variability_pctl=scrublet_min_gene_variability_pctl,
                n_prin_comps=safe_n,
                log_transform=True,
                verbose=False,
            )
            doublet_scores[mask] = scores
            predicted_doublets[mask] = doublets
            n_batch_doublets = doublets.sum()
            logger.info(
                "Batch '%s': %d/%d cells flagged as doublets (%.1f%%)",
                batch, n_batch_doublets, mask.sum(), n_batch_doublets / mask.sum() * 100,
            )
        except Exception as e:
            logger.warning(
                "Scrublet failed for batch '%s' (%s) — scores set to 0 for this batch.",
                batch, e,
            )

    adata.obs['doublet_score'] = doublet_scores
    adata.obs['predicted_doublet'] = predicted_doublets


def detect_doublets(
    adata: AnnData,
    batch_key: Optional[str] = None,
    expected_doublet_rate: float = QC_DEFAULTS.scrublet_expected_doublet_rate,
    sim_doublet_ratio: float = QC_DEFAULTS.scrublet_sim_ratio,
    n_prin_comps: int = QC_DEFAULTS.scrublet_n_prin_comps,
    scrublet_min_counts: int = 2,
    scrublet_min_cells: int = 3,
    scrublet_min_gene_variability_pctl: float = 85.0,
    random_state: int = QC_DEFAULTS.scrublet_random_state,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Detect doublets using Scrublet (batch-aware).

    IMPORTANT: Must be run on raw counts, before normalization.
    If batch_key is provided, runs Scrublet per batch.

    GPU note: when ``SCAGENT_GPU`` is enabled, doublet detection runs through
    rapids_singlecell's Scrublet on the GPU (much faster on large data). This is
    an *approximation* of the CPU path, not a bit-for-bit match: rapids_singlecell
    does its own internal gene selection and does not expose
    ``min_counts``/``min_cells``/``min_gene_variability_pctl``, so it typically
    flags somewhat fewer doublets (per-cell score rank correlation ~0.45 in
    testing). It uses adata.X (assumed raw counts per the contract above) and
    handles ``batch_key`` internally without the obs-label fragility of
    ``sc.pp.scrublet``. If the GPU run fails it falls back to the CPU path.

    Parameters
    ----------
    adata : AnnData
        AnnData object with raw counts.
    batch_key : str, optional
        Column in adata.obs containing batch information.
        If provided, runs Scrublet per batch (recommended for multi-sample data).
    expected_doublet_rate : float, default 0.06
        Expected fraction of doublets.
    sim_doublet_ratio : float, default 2.0
        Number of simulated doublets relative to observed cells.
    n_prin_comps : int, default 30
        Number of principal components for KNN.
    scrublet_min_counts : int, default 2
        Scrublet preprocessing minimum counts per gene.
    scrublet_min_cells : int, default 3
        Scrublet preprocessing minimum cells per gene.
    scrublet_min_gene_variability_pctl : float, default 85.0
        Scrublet preprocessing gene variability percentile.
    random_state : int, default 0
        Random seed for reproducibility.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    logger.info("Running Scrublet doublet detection...")

    resolution = resolve_batch_metadata(adata, requested_column=batch_key)
    batch_key = resolution.applied_column
    if resolution.status == "auto_selected" and batch_key:
        logger.info(
            "Auto-selected '%s' for per-batch Scrublet (%s)",
            batch_key,
            resolution.reason,
        )
    elif resolution.status == "needs_confirmation":
        logger.info(
            "No confirmed batch_key for Scrublet; running without batch stratification. "
            "Recommended candidate: %s",
            resolution.recommended_column,
        )
    elif resolution.status == "invalid_requested":
        logger.warning(resolution.reason)

    safe_n_prin_comps = _safe_scrublet_n_prin_comps(adata, batch_key, n_prin_comps)
    if safe_n_prin_comps != n_prin_comps:
        logger.info(
            "Adjusted Scrublet n_prin_comps from %s to %s for dataset size/batch size constraints.",
            n_prin_comps,
            safe_n_prin_comps,
        )

    # GPU path (SCAGENT_GPU): rapids_singlecell Scrublet. Approximate vs CPU — see
    # the GPU note in this function's docstring. rsc handles batch_key internally
    # (no obs-label fragility) and does not accept the min_*/variability params.
    gpu_done = False
    if gpu_available():
        try:
            import rapids_singlecell as rsc

            logger.info("Running GPU Scrublet via rapids_singlecell (approximate; see docstring).")
            gpu_batch_key = batch_key if (batch_key and batch_key in adata.obs.columns) else None
            with on_gpu(adata):
                rsc.pp.scrublet(
                    adata,
                    batch_key=gpu_batch_key,
                    sim_doublet_ratio=sim_doublet_ratio,
                    expected_doublet_rate=expected_doublet_rate,
                    n_prin_comps=safe_n_prin_comps,
                    log_transform=True,
                    random_state=random_state,
                    verbose=False,
                )
            # Normalize dtypes to match the CPU path (float64 score, bool flag).
            adata.obs["doublet_score"] = np.asarray(adata.obs["doublet_score"], dtype=np.float64)
            adata.obs["predicted_doublet"] = np.asarray(adata.obs["predicted_doublet"], dtype=bool)
            gpu_done = True
        except Exception as e:
            logger.warning("GPU Scrublet failed (%s); falling back to CPU Scrublet.", e)

    if gpu_done:
        pass
    elif batch_key and batch_key in adata.obs.columns:
        # Run Scrublet per-batch using a manual loop instead of sc.pp.scrublet(batch_key=...).
        # sc.pp.scrublet's batch_key implementation reassigns scores via adata.obs.loc[sub.obs_names],
        # which fails with a KeyError when obs_names contain non-standard separators (e.g.
        # "BARCODE-1-Sample_3"). Running the loop ourselves sidesteps that fragility entirely.
        _run_scrublet_per_batch(
            adata,
            batch_key=batch_key,
            expected_doublet_rate=expected_doublet_rate,
            sim_doublet_ratio=sim_doublet_ratio,
            n_prin_comps=safe_n_prin_comps,
            scrublet_min_counts=scrublet_min_counts,
            scrublet_min_cells=scrublet_min_cells,
            scrublet_min_gene_variability_pctl=scrublet_min_gene_variability_pctl,
            random_state=random_state,
        )
    else:
        import scrublet as scr
        import scipy.sparse as sp

        X = adata.X
        if sp.issparse(X):
            X = X.tocsc().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        scrub = scr.Scrublet(
            X,
            expected_doublet_rate=expected_doublet_rate,
            sim_doublet_ratio=sim_doublet_ratio,
            random_state=random_state,
        )
        scores, doublets = scrub.scrub_doublets(
            min_counts=scrublet_min_counts,
            min_cells=scrublet_min_cells,
            min_gene_variability_pctl=scrublet_min_gene_variability_pctl,
            n_prin_comps=safe_n_prin_comps,
            log_transform=True,
            verbose=False,
        )
        adata.obs['doublet_score'] = scores
        adata.obs['predicted_doublet'] = doublets

    n_doublets = adata.obs['predicted_doublet'].sum()
    doublet_rate = n_doublets / adata.n_obs * 100
    adata.uns["scrublet_params"] = {
        "batch_key": batch_key,
        "expected_doublet_rate": float(expected_doublet_rate),
        "sim_doublet_ratio": float(sim_doublet_ratio),
        "n_prin_comps": int(n_prin_comps),
        "n_prin_comps_used": int(safe_n_prin_comps),
        "min_counts": int(scrublet_min_counts),
        "min_cells": int(scrublet_min_cells),
        "min_gene_variability_pctl": float(scrublet_min_gene_variability_pctl),
        "random_state": int(random_state),
    }
    logger.info(f"Detected {n_doublets:,} doublets ({doublet_rate:.1f}%)")

    if not inplace:
        return adata


def filter_cells_by_genes(
    adata: AnnData,
    min_genes: Optional[int] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Filter cells based on the number of detected genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with QC metrics calculated.
    min_genes : int, optional
        Minimum number of detected genes per cell. If None, no filtering is applied.
    inplace : bool, default True
        Modify adata in place.
    """
    if min_genes is None:
        return None if inplace else adata.copy()

    if 'n_genes_by_counts' not in adata.obs.columns:
        raise ValueError("Cell gene-count metrics not calculated. Run calculate_qc_metrics first.")

    if not inplace:
        adata = adata.copy()

    n_before = adata.n_obs
    adata._inplace_subset_obs(adata.obs['n_genes_by_counts'] >= min_genes)
    n_after = adata.n_obs
    logger.info(
        "Filtered cells by min_genes=%d: %d -> %d (%d removed)",
        min_genes,
        n_before,
        n_after,
        n_before - n_after,
    )

    if not inplace:
        return adata


def filter_doublets(
    adata: AnnData,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Remove cells marked as Scrublet predicted doublets."""
    if 'predicted_doublet' not in adata.obs.columns:
        raise ValueError("Doublet predictions not available. Run detect_doublets first.")

    if not inplace:
        adata = adata.copy()

    n_before = adata.n_obs
    adata._inplace_subset_obs(~adata.obs['predicted_doublet'].astype(bool))
    n_after = adata.n_obs
    logger.info(
        "Filtered predicted doublets: %d -> %d (%d removed)",
        n_before,
        n_after,
        n_before - n_after,
    )

    if not inplace:
        return adata


def run_qc_pipeline(
    adata: AnnData,
    mt_threshold: Optional[float] = None,
    filter_mt: bool = True,
    min_genes: Optional[int] = None,
    min_cells: Optional[float] = None,
    remove_ribo: bool = True,
    detect_doublets_flag: bool = True,
    remove_doublets: bool = False,
    batch_key: Optional[str] = None,
    scrublet_expected_doublet_rate: float = QC_DEFAULTS.scrublet_expected_doublet_rate,
    scrublet_sim_doublet_ratio: float = QC_DEFAULTS.scrublet_sim_ratio,
    scrublet_n_prin_comps: int = QC_DEFAULTS.scrublet_n_prin_comps,
    scrublet_min_counts: int = 2,
    scrublet_min_cells: int = 3,
    scrublet_min_gene_variability_pctl: float = 85.0,
    scrublet_random_state: int = QC_DEFAULTS.scrublet_random_state,
    force_doublet_recompute: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run the complete QC pipeline.

    This is a convenience function that runs:
    1. QC metrics calculation
    2. Doublet detection (on raw counts)
    3. Optional cell filtering by detected genes, MT content, and doublet calls
    4. Gene filtering

    Parameters
    ----------
    adata : AnnData
        AnnData object with raw counts.
    mt_threshold : float, optional
        Maximum MT percentage. Auto-detected based on data type if None.
    filter_mt : bool, default True
        If false, calculate/report MT metrics but do not remove cells by MT percentage.
    min_genes : int, optional
        Minimum detected genes per cell. If None, no cell-level gene-count filtering is applied.
    min_cells : float, optional
        Minimum cells per gene. If None, no gene-level cell-count filtering is applied.
    remove_ribo : bool, default True
        Remove ribosomal genes.
    detect_doublets_flag : bool, default True
        Run Scrublet doublet detection.
    remove_doublets : bool, default False
        Remove cells marked as predicted doublets after detection.
    batch_key : str, optional
        Batch key for batch-aware doublet detection.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    logger.info("Starting QC pipeline...")

    # Step 1: Calculate QC metrics (skip if already present — idempotent guard)
    if 'pct_counts_mt' not in adata.obs.columns:
        calculate_qc_metrics(adata, inplace=True)
    else:
        logger.info("QC metrics already present in adata.obs; skipping recalculation.")

    # Step 2: Doublet detection (skip if scores already present)
    if force_doublet_recompute:
        for col in ("doublet_score", "predicted_doublet"):
            if col in adata.obs.columns:
                del adata.obs[col]

    if detect_doublets_flag and 'predicted_doublet' not in adata.obs.columns:
        detect_doublets(
            adata,
            batch_key=batch_key,
            expected_doublet_rate=scrublet_expected_doublet_rate,
            sim_doublet_ratio=scrublet_sim_doublet_ratio,
            n_prin_comps=scrublet_n_prin_comps,
            scrublet_min_counts=scrublet_min_counts,
            scrublet_min_cells=scrublet_min_cells,
            scrublet_min_gene_variability_pctl=scrublet_min_gene_variability_pctl,
            random_state=scrublet_random_state,
            inplace=True,
        )
    elif detect_doublets_flag:
        logger.info("Doublet scores already present in adata.obs; skipping Scrublet re-run.")

    # Step 3: Optional cell filters
    filter_cells_by_genes(adata, min_genes=min_genes, inplace=True)
    if filter_mt:
        filter_cells_by_mt(adata, mt_threshold=mt_threshold, inplace=True)
    else:
        logger.info("Skipping hard MT% cell filtering; MT metrics remain available for QC review.")
    if remove_doublets:
        filter_doublets(adata, inplace=True)

    # Step 4: Filter genes (min_cells=None means skip cell-count filtering)
    filter_genes(adata, min_cells=min_cells, remove_ribo=remove_ribo, inplace=True)

    logger.info(f"QC pipeline complete. Final shape: {adata.n_obs:,} x {adata.n_vars:,}")

    if not inplace:
        return adata
