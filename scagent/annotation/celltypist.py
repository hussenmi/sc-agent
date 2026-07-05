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


def _celltypist_model_cache_path(model: str) -> Optional[Path]:
    """Return the expected local CellTypist cache path for a named model."""
    basename = _model_basename(model)
    if not basename or str(model).startswith(("/", "./")):
        return None
    try:
        from celltypist import models
    except ImportError:
        return None
    models_path = getattr(models, "models_path", None)
    if not models_path:
        return None
    return Path(models_path) / basename


def celltypist_model_records(
    *,
    organism: Optional[str] = None,
    query: Optional[str] = None,
    force_update: bool = False,
) -> List[Dict[str, Any]]:
    """Return CellTypist model catalog records with inferred organism/cache state."""
    catalog = celltypist_models_description(force_update=force_update)
    target = str(organism or "").strip().lower()
    query_text = str(query or "").strip().lower()
    records: List[Dict[str, Any]] = []

    for _, row in catalog.iterrows():
        model = str(row.get("model") or "")
        description = str(row.get("description") or "")
        info = infer_celltypist_model_organism(model, force_update=False, catalog=catalog)
        inferred = info.get("organism") or "unknown"
        if target in {"human", "mouse"} and inferred != target:
            continue
        haystack = f"{model} {description}".lower()
        if query_text and query_text not in haystack:
            continue
        cache_path = _celltypist_model_cache_path(model)
        records.append({
            "model": model,
            "description": description,
            "inferred_organism": inferred,
            "organism_source": info.get("source"),
            "organism_reason": info.get("reason"),
            "cached": bool(cache_path and cache_path.exists()),
            "cache_path": str(cache_path) if cache_path else None,
        })
    return records


def check_celltypist_model(
    model: str = CELLTYPIST_DEFAULTS.model,
    *,
    organism: Optional[str] = None,
    query: Optional[str] = None,
    force_update: bool = False,
) -> Dict[str, Any]:
    """Check whether a CellTypist model is suitable and suggest alternatives."""
    requested_organism = str(organism or "").strip().lower()
    if requested_organism not in {"human", "mouse"}:
        requested_organism = ""
    model_info = infer_celltypist_model_organism(model, force_update=force_update)
    model_organism = model_info.get("organism") or "unknown"
    cache_path = None
    cached = False
    explicit_path_exists = None
    if str(model or "").startswith(("/", "./")):
        explicit_path = Path(model)
        explicit_path_exists = explicit_path.exists()
        cache_path = str(explicit_path)
        cached = bool(explicit_path_exists)
    else:
        cached_path = _celltypist_model_cache_path(model)
        cache_path = str(cached_path) if cached_path else None
        cached = bool(cached_path and cached_path.exists())

    species_match = (
        not requested_organism
        or model_organism not in {"human", "mouse"}
        or requested_organism == model_organism
    )
    compatible = bool(species_match and (explicit_path_exists is not False))
    reasons: List[str] = []
    if not requested_organism:
        reasons.append("Dataset organism was not provided, so species compatibility is unresolved.")
    if requested_organism and model_organism in {"human", "mouse"} and requested_organism != model_organism:
        reasons.append(
            f"Requested organism is {requested_organism}, but model appears to be {model_organism}."
        )
    if explicit_path_exists is False:
        reasons.append(f"Explicit CellTypist model path does not exist: {model}")
    if model_organism == "unknown":
        reasons.append("Could not infer model organism from the CellTypist catalog or model name.")

    recommendation_target = requested_organism if requested_organism in {"human", "mouse"} else None
    recommended = (
        celltypist_model_records(organism=recommendation_target, query=query, force_update=force_update)
        if recommendation_target else []
    )
    return {
        "model": model,
        "model_info": model_info,
        "requested_organism": requested_organism or None,
        "model_organism": model_organism,
        "species_match": species_match,
        "compatible": compatible,
        "cached": cached,
        "cache_path": cache_path,
        "download_required": bool(not cached and not str(model or "").startswith(("/", "./"))),
        "reasons": reasons,
        "recommended_models": recommended[:20],
    }


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

    # Find raw counts — check named layers AND adata.raw (which carries its own
    # var), on integer VALUES not dtype (counts are commonly stored as float32).
    # See core.inspector.find_counts_matrix.
    from ..core.inspector import find_counts_matrix
    from ..core.genes import convert_var_to_symbols, infer_id_format

    counts = find_counts_matrix(adata, prefer_layer=raw_layer)
    if counts is not None:
        X = counts["X"].copy()
        counts_var = counts["var"]
        logger.info(
            f"Using raw counts from {counts['source']} ({counts['n_vars']} genes)"
        )
    else:
        # No integer-valued counts anywhere. Falling back to adata.X is
        # dangerous: by the time CellTypist is called, adata.X is almost always
        # log-normalized. Running normalize_total + log1p on it produces
        # log(1 + log1p_X / Σ * 10000) — a transformation CellTypist was never
        # trained on. The model still returns confident-looking labels, but they
        # are systematically wrong. Refuse rather than silently mis-annotate.
        if 'log1p' in adata.uns:
            raise ValueError(
                "CellTypist requires raw integer counts, but none were found in "
                "any layer, adata.raw, or adata.X, and adata.X is already "
                "log-normalized (adata.uns['log1p'] is set). Re-normalizing "
                "log1p data would produce wrong predictions. Place raw counts "
                "into adata.layers['raw_counts'] (or adata.raw) before calling "
                "run_celltypist, or pass raw_layer explicitly."
            )
        logger.warning(
            "No integer-valued counts found; assuming adata.X holds raw counts."
        )
        X = adata.X.copy()
        counts_var = adata.var

    # Create new AnnData with raw counts (var must match the counts matrix —
    # adata.raw has its own var, often more genes than adata.var).
    adata_ct = AnnData(X, obs=adata.obs.copy(), var=counts_var.copy())

    # Ensure var_names are gene symbols — CellTypist silently fails ("no features
    # overlap") when handed Ensembl IDs. Offline conversion via the dataset's own
    # symbol column (feature_name/gene_symbols/...) when present; genome prefixes
    # and duplicate symbols are handled by the shared helper.
    if infer_id_format(adata_ct.var_names) != "symbol":
        adata_ct, gene_report = convert_var_to_symbols(adata_ct, inplace=True)
        logger.info(f"Gene-id conversion for CellTypist: {gene_report.message}")

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

    # Download model if needed. Treat download failures as structured
    # unavailability instead of continuing and failing later with a vague
    # CellTypist loader error.
    if model.startswith('/') or model.startswith('./'):
        if not Path(model).exists():
            raise RuntimeError(f"CellTypist model path does not exist: {model}")
    else:
        cached_path = _celltypist_model_cache_path(model)
        if not (cached_path and cached_path.exists()):
            try:
                models.download_models(model=model)
            except Exception as e:
                raise RuntimeError(
                    f"CellTypist model download failed for '{model}': {e}"
                ) from e
            cached_path = _celltypist_model_cache_path(model)
            if cached_path and not cached_path.exists():
                raise RuntimeError(
                    f"CellTypist model download finished but '{model}' is still not available locally."
                )

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

    if model.startswith(("/", "./")):
        model_cache_path = model
        model_cached = Path(model).exists()
    else:
        cached_model_path = _celltypist_model_cache_path(model)
        model_cache_path = str(cached_model_path) if cached_model_path else None
        model_cached = bool(cached_model_path and cached_model_path.exists())

    adata.uns["celltypist"] = {
        "model": model,
        "model_name": _model_basename(model),
        "requested_organism": requested_organism or None,
        "model_organism": model_organism,
        "model_organism_source": model_info.get("source"),
        "model_description": model_info.get("description"),
        "model_cached": model_cached,
        "model_cache_path": model_cache_path,
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
