"""
CellTypist cell type annotation for scagent.

CRITICAL: CellTypist requires data normalized to target_sum=10000.
This module handles the special normalization requirements.
"""

from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import scanpy as sc
from anndata import AnnData
import logging

from ..config.defaults import CELLTYPIST_DEFAULTS

logger = logging.getLogger(__name__)


_KNOWN_HUMAN_MODELS = {
    "Immune_All_Low.pkl",
    "Immune_All_High.pkl",
    "Adult_COVID19_PBMC.pkl",
    "COVID19_HumanChallenge_Blood.pkl",
    "COVID19_Immune_Landscape.pkl",
}


def _model_basename(model: str) -> str:
    return Path(str(model or "")).name


def celltypist_models_description(force_update: bool = False):
    """Return the installed CellTypist model catalog, using the package API."""
    try:
        from celltypist import models
    except ImportError:
        raise ImportError("celltypist not installed. Install with: pip install celltypist")

    try:
        return models.models_description(on_the_fly=bool(force_update))
    except TypeError:
        return models.models_description()


def infer_celltypist_model_organism(
    model: str,
    *,
    force_update: bool = False,
    catalog=None,
) -> Dict[str, Any]:
    """Infer model organism from CellTypist model metadata and name."""
    basename = _model_basename(model)
    if not basename:
        return {"organism": "unknown", "source": "missing_model_name", "model": model}
    if str(model).startswith(("/", "./")):
        return {
            "organism": "unknown",
            "source": "explicit_model_path",
            "model": model,
            "model_name": basename,
            "reason": "Explicit model paths do not expose CellTypist catalog metadata.",
        }

    description = ""
    try:
        if catalog is None:
            catalog = celltypist_models_description(force_update=force_update)
        if "model" in catalog.columns:
            row = catalog[catalog["model"].astype(str) == basename]
            if not row.empty and "description" in row.columns:
                description = str(row.iloc[0]["description"])
    except Exception as exc:
        catalog = None
        catalog_error = str(exc)
    else:
        catalog_error = None

    text = f"{basename} {description}".lower()
    source = "celltypist_model_catalog" if description else "model_name"
    organism = "unknown"
    reason = ""

    if basename in _KNOWN_HUMAN_MODELS:
        organism = "human"
        source = "known_celltypist_model"
        reason = "Known CellTypist immune/blood model trained for human annotations."
    elif any(token in text for token in ("mouse", "murine", "mice")):
        organism = "mouse"
        reason = "Model name or description indicates mouse/murine data."
    elif any(token in text for token in ("human", "homo sapiens", "donor", "patient", "covid19", "covid-19")):
        organism = "human"
        reason = "Model name or description indicates human data."

    payload = {
        "organism": organism,
        "source": source,
        "model": model,
        "model_name": basename,
        "description": description,
        "reason": reason,
    }
    if catalog_error:
        payload["catalog_error"] = catalog_error
    return payload


def available_celltypist_models_for_organism(organism: str, *, force_update: bool = False) -> List[str]:
    """List locally known CellTypist models whose metadata matches organism."""
    target = str(organism or "").lower()
    if target not in {"human", "mouse"}:
        return []
    try:
        catalog = celltypist_models_description(force_update=force_update)
    except Exception:
        return []
    matches: List[str] = []
    for model in catalog.get("model", []):
        info = infer_celltypist_model_organism(str(model), force_update=False, catalog=catalog)
        if info.get("organism") == target:
            matches.append(str(model))
    return matches


def prepare_for_celltypist(
    adata: AnnData,
    raw_layer: Optional[str] = None,
    target_sum: int = CELLTYPIST_DEFAULTS.target_sum,
) -> AnnData:
    """
    Prepare data for CellTypist annotation.

    CRITICAL: CellTypist requires data normalized to target_sum=10000
    and log-transformed. This function creates a separate AnnData
    with the correct normalization.

    Parameters
    ----------
    adata : AnnData
        AnnData object (can be at any processing stage).
    raw_layer : str, optional
        Layer containing raw counts. If None, tries to auto-detect.
    target_sum : int, default 10000
        Target sum for normalization (CellTypist requirement).

    Returns
    -------
    AnnData
        New AnnData object prepared for CellTypist.
    """
    logger.info("Preparing data for CellTypist annotation...")

    # Find raw counts
    if raw_layer is None:
        for layer in ['raw_counts', 'raw_data', 'counts']:
            if layer in adata.layers:
                raw_layer = layer
                break

    if raw_layer is not None:
        logger.info(f"Using raw counts from layer '{raw_layer}'")
        X = adata.layers[raw_layer].copy()
    else:
        # Falling back to adata.X is dangerous: by the time CellTypist is
        # called, adata.X is almost always log-normalized. Running
        # normalize_total + log1p on it produces log(1 + log1p_X / Σ * 10000)
        # — a transformation CellTypist was never trained on. The model
        # still returns confident-looking labels, but they are systematically
        # wrong. Refuse rather than silently mis-annotate.
        if 'log1p' in adata.uns:
            raise ValueError(
                "CellTypist requires raw integer counts, but no raw-counts "
                "layer was found and adata.X is already log-normalized "
                "(adata.uns['log1p'] is set). Re-normalizing log1p data would "
                "produce wrong predictions. Place raw counts into "
                "adata.layers['raw_counts'] before calling run_celltypist, "
                "or pass raw_layer explicitly."
            )
        logger.warning(
            "No raw-counts layer found and adata.X is not flagged as "
            "log-normalized; assuming adata.X holds raw counts."
        )
        X = adata.X.copy()

    # Create new AnnData with raw counts
    adata_ct = AnnData(X, obs=adata.obs.copy(), var=adata.var.copy())

    # Ensure var_names are gene symbols, not Ensembl IDs.
    # CellTypist will silently fail ("no features overlap") if given Ensembl IDs.
    import re as _re
    # --- Strip genome-prefix from multi-genome CellRanger references ---
    # e.g. "GRCh38_CD3D" or "GRCh38___CD3D" → "CD3D"
    _genome_prefix_re = _re.compile(r'^[A-Za-z0-9]+_{1,10}([A-Z].+)$')
    _sample = list(adata_ct.var_names[:200])
    _n_prefixed = sum(1 for v in _sample if _genome_prefix_re.match(str(v)))
    if _n_prefixed > 100:
        _cleaned = [_genome_prefix_re.sub(r'\1', v) for v in adata_ct.var_names]
        adata_ct.var_names = _cleaned
        adata_ct.var_names_make_unique()
        logger.info(f"Stripped genome prefix from var_names (e.g. 'GRCh38_CD3D' → 'CD3D')")

    # --- Swap Ensembl IDs to gene symbols if needed ---
    _ensembl_re = _re.compile(r'^ENSG\d{5,}')
    _n_ensembl = sum(1 for v in list(adata_ct.var_names[:200]) if _ensembl_re.match(str(v)))
    if _n_ensembl > 100:
        for _col in ['gene_symbols', 'gene_names', 'feature_name', 'Gene', 'Symbol']:
            if _col in adata_ct.var.columns:
                adata_ct.var_names = adata_ct.var[_col].astype(str).values
                adata_ct.var_names_make_unique()
                logger.info(f"Swapped var_names from Ensembl IDs to gene symbols using var['{_col}']")
                break
        else:
            logger.warning("var_names look like Ensembl IDs but no gene-symbol column found in var; CellTypist may fail.")

    # --- Strip -N suffixes added by var_names_make_unique (e.g. CD3D-1 → CD3D) ---
    _suffixed = _re.compile(r'^(.+)-\d+$')
    _base_names = [_suffixed.sub(r'\1', v) for v in adata_ct.var_names]
    _seen: set = set()
    _keep = []
    for i, (orig, base) in enumerate(zip(adata_ct.var_names, _base_names)):
        if base not in _seen:
            _seen.add(base)
            _keep.append(i)
    if len(_keep) < len(adata_ct.var_names):
        n_dropped = len(adata_ct.var_names) - len(_keep)
        adata_ct = adata_ct[:, _keep].copy()
        new_names = [_suffixed.sub(r'\1', v) for v in adata_ct.var_names]
        adata_ct.var_names = new_names
        logger.info(f"Stripped var_names_make_unique suffixes: dropped {n_dropped} duplicate-suffix genes")

    # Normalize to target_sum=10000 (CellTypist requirement)
    sc.pp.normalize_total(adata_ct, target_sum=target_sum, inplace=True)

    # Log transform
    sc.pp.log1p(adata_ct)

    # Copy UMAP if available
    if 'X_umap' in adata.obsm:
        adata_ct.obsm['X_umap'] = adata.obsm['X_umap'].copy()

    logger.info(f"Data prepared for CellTypist: normalized to target_sum={target_sum}")

    return adata_ct


def run_celltypist(
    adata: AnnData,
    model: str = CELLTYPIST_DEFAULTS.model,
    organism: Optional[str] = None,
    allow_cross_species: bool = False,
    majority_voting: bool = CELLTYPIST_DEFAULTS.majority_voting,
    over_clustering: Optional[str] = "leiden",
    mode: str = 'best match',
    raw_layer: Optional[str] = None,
    transfer_results: bool = True,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run CellTypist cell type annotation.

    This function handles the special normalization requirements of CellTypist
    and optionally transfers results back to the original AnnData.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    model : str, default 'Immune_All_Low.pkl'
        CellTypist model to use.
    organism : {'human', 'mouse'}, optional
        Dataset organism. If provided, it is checked against model metadata.
    allow_cross_species : bool, default False
        Allow species/model mismatch as an explicit expert override.
    majority_voting : bool, default True
        Use majority voting for cluster-level annotation.
    over_clustering : str, optional
        Column in adata.obs for over-clustering in majority voting.
    mode : str, default 'best match'
        Prediction mode: 'best match' or 'prob match'.
    raw_layer : str, optional
        Layer containing raw counts.
    transfer_results : bool, default True
        Transfer results back to original adata.
    inplace : bool, default True
        Modify adata in place (when transfer_results=True).

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    try:
        import celltypist
        from celltypist import models
    except ImportError:
        raise ImportError("celltypist not installed. Install with: pip install celltypist")

    if not inplace:
        adata = adata.copy()

    logger.info(f"Running CellTypist with model '{model}'")

    requested_organism = str(organism or "").strip().lower()
    if requested_organism and requested_organism not in {"human", "mouse"}:
        raise ValueError(
            f"CellTypist organism must be 'human' or 'mouse' when provided; got {organism!r}."
        )
    model_info = infer_celltypist_model_organism(model)
    model_organism = model_info.get("organism")
    if (
        requested_organism
        and model_organism in {"human", "mouse"}
        and requested_organism != model_organism
        and not allow_cross_species
    ):
        raise ValueError(
            f"Refusing to run CellTypist model '{_model_basename(model)}' because "
            f"the dataset organism is '{requested_organism}' but the model appears "
            f"to be '{model_organism}'. Choose a {requested_organism}-compatible "
            "model, use Scimilarity/marker validation, or set allow_cross_species=true "
            "only if you intentionally want a non-definitive cross-species run."
        )

    if majority_voting and over_clustering and over_clustering not in adata.obs.columns:
        raise ValueError(
            f"CellTypist majority voting requested over '{over_clustering}', "
            "but that column is not present in adata.obs."
        )

    # Download model if needed
    if not (model.startswith('/') or model.startswith('./')):
        try:
            models.download_models(model=model)
        except Exception as e:
            logger.warning(f"Could not download model: {e}")

    # Prepare data with correct normalization
    adata_ct = prepare_for_celltypist(adata, raw_layer=raw_layer)

    # Run annotation
    predictions = celltypist.annotate(
        adata_ct,
        model=model,
        mode=mode,
        majority_voting=majority_voting,
        over_clustering=over_clustering if majority_voting else None,
    )

    # Get results
    adata_preds = predictions.to_adata()

    # Transfer results back to original adata
    if transfer_results:
        cols_to_transfer = ['predicted_labels', 'conf_score']
        if majority_voting:
            cols_to_transfer.extend(['majority_voting', 'over_clustering'])

        # CellTypist should preserve obs_names, but guard against silent
        # misalignment — reindexing .loc against a mismatched index returns
        # NaNs instead of raising, which would corrupt downstream labels.
        if set(adata_preds.obs_names) != set(adata.obs_names):
            raise ValueError(
                "CellTypist predictions obs_names do not match input adata "
                "obs_names; refusing to transfer labels to avoid silent "
                "misalignment."
            )

        for col in cols_to_transfer:
            if col in adata_preds.obs.columns:
                val = adata_preds.obs[col].loc[adata.obs_names]
                # Guard against CellTypist returning a DataFrame instead of a Series
                # (happens with some model/version combinations for majority_voting)
                if hasattr(val, 'squeeze'):
                    val = val.squeeze()
                adata.obs[f'celltypist_{col}'] = val

        logger.info("CellTypist results transferred to adata.obs")

    # Log summary
    n_celltypes = adata_preds.obs['predicted_labels'].nunique()
    logger.info(f"CellTypist complete: {n_celltypes} cell types annotated")

    adata.uns["celltypist"] = {
        "model": model,
        "model_name": _model_basename(model),
        "requested_organism": requested_organism or None,
        "model_organism": model_organism,
        "model_organism_source": model_info.get("source"),
        "model_description": model_info.get("description"),
        "allow_cross_species": bool(allow_cross_species),
        "majority_voting": bool(majority_voting),
        "over_clustering": over_clustering if majority_voting else None,
    }

    if not inplace:
        return adata


def list_celltypist_models() -> 'pd.DataFrame':
    """
    List available CellTypist models.

    Returns
    -------
    pd.DataFrame
        DataFrame with model descriptions.
    """
    try:
        from celltypist import models
        return models.models_description()
    except ImportError:
        raise ImportError("celltypist not installed. Install with: pip install celltypist")
