"""
Claude API tool definitions for scagent.

Tools are organized into two layers:
1. Action tools - mutate/generate analysis artifacts
2. Inspection tools - read-only queries for more detail

All tools return structured JSON for LLM reasoning.
"""

# Configure tqdm for cleaner progress bars (must be before any imports that use tqdm)
import os
os.environ.setdefault('TQDM_NCOLS', '60')
os.environ.setdefault('TQDM_MININTERVAL', '0.5')  # Update less frequently

from typing import List, Dict, Any
import json


def get_tools() -> List[Dict[str, Any]]:
    """
    Get Claude API tool definitions for single-cell analysis.

    Returns
    -------
    List[Dict]
        List of tool definitions in Claude API format.
    """
    # Action tools (mutate state)
    action_tools = [
        {
            "name": "run_qc",
            "description": "Run or preview the quality control pipeline: QC metrics, doublet detection, cell/gene filtering. Use preview_only=true first in collaborative workflows so the user can review proposed removals before filters are applied. Do not assume intermediate h5ad saving is desired.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad or 10X h5 file (required for initial load, optional if data already in memory)"},
                    "output_path": {"type": "string", "description": "Optional path to save a processed h5ad. Prefer saving only final outputs unless the user explicitly asks for checkpoints."},
                    "preview_only": {"type": "boolean", "description": "If true, do not filter. Instead compute full QC metrics, estimate removals, and generate pre-filter QC figures."},
                    "mt_threshold": {"type": "number", "description": "Max MT% (default: auto-detect)"},
                    "min_cells": {"type": "integer", "description": "Minimum cells per gene before gene removal (default: ~55)"},
                    "remove_ribo": {"type": "boolean", "description": "Remove ribosomal genes (default: true)"},
                    "remove_mt": {"type": "boolean", "description": "Remove mitochondrial genes from the feature set (default: false)"},
                    "detect_doublets_flag": {"type": "boolean", "description": "Run Scrublet doublet detection (default: true)"},
                    "figure_dir": {"type": "string", "description": "Directory for QC figures. Plots are generated from the full pre-filter data."},
                    "batch_key": {"type": "string", "description": "Batch column for per-batch doublet detection"}
                },
                "required": []
            }
        },
        {
            "name": "normalize_and_hvg",
            "description": "Normalize, log-transform, and select highly variable genes. Preserves raw counts in layer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_hvg": {"type": "integer", "description": "Number of HVGs (default: 4000)"}
                },
                "required": []
            }
        },
        {
            "name": "run_dimred",
            "description": "Run PCA, compute neighbor graph, and UMAP embedding.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_pcs": {"type": "integer", "description": "Number of PCs (default: 30)"},
                    "n_neighbors": {"type": "integer", "description": "Number of neighbors (default: 30)"}
                },
                "required": []
            }
        },
        {
            "name": "run_clustering",
            "description": "Run Leiden or PhenoGraph clustering.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "method": {"type": "string", "enum": ["leiden", "phenograph"], "description": "Method (default: leiden)"},
                    "resolution": {"type": "number", "description": "Resolution (default: 1.0)"}
                },
                "required": []
            }
        },
        {
            "name": "run_celltypist",
            "description": "Annotate cell types with CellTypist. Handles target_sum=10000 normalization automatically.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "model": {"type": "string", "description": "Model name (default: Immune_All_Low.pkl)"},
                    "majority_voting": {"type": "boolean", "description": "Use majority voting (default: true)"}
                },
                "required": []
            }
        },
        {
            "name": "run_scimilarity",
            "description": "Annotate cell types with Scimilarity (embedding-based). Uses pretrained embeddings and kNN to annotate cells. Model path is pre-configured via SCIMILARITY_MODEL_PATH env var - do NOT ask the user for a model path. Different from CellTypist - use when you want embedding-based annotation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data if already loaded)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"}
                },
                "required": []
            }
        },
        {
            "name": "run_batch_correction",
            "description": "Correct batch effects using Harmony or Scanorama.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional)"},
                    "batch_key": {"type": "string", "description": "Batch column name"},
                    "method": {"type": "string", "enum": ["harmony", "scanorama"], "description": "Method (default: harmony)"}
                },
                "required": ["batch_key"]
            }
        },
        {
            "name": "run_deg",
            "description": "Run validated differential expression analysis between groups. Validates input data (matrix type, cluster sizes, batch confounding) and attaches validity metadata for downstream GSEA interpretation. Returns validation warnings alongside DEG results.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional)"},
                    "groupby": {"type": "string", "description": "Group column (default: leiden)"},
                    "method": {"type": "string", "enum": ["wilcoxon", "t-test", "logreg"], "description": "Method (default: wilcoxon)"},
                    "layer": {"type": "string", "description": "Optional expression layer to use for DEG (for example scran_norm)"},
                    "target_geneset": {"type": "string", "description": "Target gene set database for compatibility check (default: MSigDB_Hallmark_2020)"}
                },
                "required": []
            }
        },
        {
            "name": "generate_figure",
            "description": "Generate and save a visualization (UMAP, violin, dotplot, etc.).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save PNG figure"},
                    "plot_type": {"type": "string", "enum": ["umap", "violin", "dotplot", "heatmap"], "description": "Plot type"},
                    "color_by": {"type": "string", "description": "Column or gene to color by"},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Genes for dotplot/heatmap"}
                },
                "required": ["output_path", "plot_type"]
            }
        },
        {
            "name": "run_gsea",
            "description": "Run Gene Set Enrichment Analysis on DEG results. Identifies enriched biological pathways/processes. Requires DEG to be run first. Returns top enriched pathways with NES scores and FDR values.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad with DEG results (optional - uses in-memory data)"},
                    "output_dir": {"type": "string", "description": "Directory to save GSEA results"},
                    "cluster": {"type": "string", "description": "Cluster to analyze (or 'all' for all clusters)"},
                    "gene_sets": {"type": "string", "description": "Gene set database: KEGG_2021_Human, GO_Biological_Process_2021, Reactome_2022, MSigDB_Hallmark_2020 (default: KEGG_2021_Human)"},
                    "min_size": {"type": "integer", "description": "Min genes in pathway (default: 5)"},
                    "max_size": {"type": "integer", "description": "Max genes in pathway (default: 500)"},
                    "permutation_num": {"type": "integer", "description": "Permutations for p-value (default: 1000)"}
                },
                "required": ["output_dir", "cluster"]
            }
        },
        {
            "name": "save_data",
            "description": "Save the current in-memory AnnData object without modifying it. Use this as the final save step after analysis and annotation are complete.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Path to save the current in-memory h5ad"}
                },
                "required": ["output_path"]
            }
        },
    ]

    # Meta tools (agent control)
    meta_tools = [
        {
            "name": "ask_user",
            "description": "Ask the user a question and wait for their response. Collaboration is the default style, but do not use this for trivial bookkeeping. Use it when a preprocessing or interpretation choice should stay under user control, including QC thresholds, filtering decisions, clustering resolution changes, annotation ambiguity, batch correction, DEG comparisons, or any other material fork in the analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the user"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "Optional list of choices"},
                    "default": {"type": "string", "description": "Default answer if user just presses enter"}
                },
                "required": ["question"]
            }
        },
        {
            "name": "run_code",
            "description": "Execute custom Python code on the AnnData object. Use this for operations not covered by other tools, like: gene ID conversion, custom filtering, subsetting, merging datasets, plotting, or any data manipulation. The code has access to 'adata', 'sc' (scanpy), 'plt' (matplotlib), 'np', 'pd'. For plots, save to a file path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Has access to adata, sc, plt, np, pd."},
                    "description": {"type": "string", "description": "Brief description of what the code does"},
                    "save_to": {"type": "string", "description": "Optional path to save adata after execution"}
                },
                "required": ["code", "description"]
            }
        },
        {
            "name": "web_search_docs",
            "description": "Search general web and documentation sources. Use for package docs, API references, troubleshooting, method pages, and implementation details. Prefer this for software/documentation questions, not for scientific evidence claims.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "site": {"type": "string", "description": "Optional site filter (e.g., 'scanpy.readthedocs.io', 'gseapy.readthedocs.io')"},
                    "max_results": {"type": "integer", "description": "Maximum results to return (default: 5)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_papers",
            "description": "Search scientific literature using PubMed. Use for papers, reviews, recent pathway/cell-type findings, and scientific evidence. Returns PMID, title, year, journal, abstract excerpt, and PubMed URL.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PubMed-style search query or plain text topic"},
                    "max_results": {"type": "integer", "description": "Maximum papers to return (default: 5)"},
                    "recent_years": {"type": "integer", "description": "Limit to the last N publication years (default: 5)"},
                    "reviews_only": {"type": "boolean", "description": "Restrict to review articles"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "fetch_url",
            "description": "Fetch and summarize a web page or article landing page. Use after a search step when you need the page contents rather than just search snippets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Maximum text characters to return (default: 4000)"}
                },
                "required": ["url"]
            }
        },
        {
            "name": "web_search",
            "description": "Backward-compatible alias for documentation/web search. Prefer web_search_docs for docs and search_papers for literature.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "site": {"type": "string", "description": "Optional site filter"},
                    "max_results": {"type": "integer", "description": "Maximum results to return (default: 5)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "research_findings",
            "description": "Conduct focused literature research on GSEA/pathway findings. Uses PubMed queries tailored to pathway, cell type, and leading-edge genes, and returns structured citations plus interpretation guidance. Use AFTER run_gsea to understand what enriched pathways mean biologically.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pathway": {"type": "string", "description": "Pathway name to research (e.g., 'Oxidative phosphorylation', 'TNF signaling')"},
                    "cell_type": {"type": "string", "description": "Cell type context (e.g., 'classical monocytes', 'CD8 T cells', 'B cells')"},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Leading edge genes from GSEA to include in search"},
                    "context": {"type": "string", "description": "Additional context (e.g., 'PBMC', 'tumor microenvironment', 'inflammation')"},
                    "cluster_confidence": {"type": "number", "description": "Optional cluster confidence score (0-1) from annotation sanity checks"},
                    "recent_years": {"type": "integer", "description": "Limit to papers from last N years (default: 3)"}
                },
                "required": ["pathway", "cell_type"]
            }
        },
        {
            "name": "install_package",
            "description": "Request installation of a Python package. Requires user approval. Use when you need a package that isn't installed (e.g., gseapy, mygene, biomart).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name (pip format)"},
                    "reason": {"type": "string", "description": "Why this package is needed"}
                },
                "required": ["package", "reason"]
            }
        },
    ]

    # Inspection tools (read-only)
    inspection_tools = [
        {
            "name": "inspect_data",
            "description": "Inspect data state: shape, processing status, available embeddings, what steps are done. Use this first to understand the data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"},
                    "goal": {"type": "string", "description": "Analysis goal to get recommendations (e.g., 'cluster', 'annotate')"},
                    "context": {"type": "string", "description": "Optional biological context hint from the user or file path (e.g., 'PBMC healthy human cells')"}
                },
                "required": []
            }
        },
        {
            "name": "get_cluster_sizes",
            "description": "Get cell counts per cluster.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"},
                    "cluster_key": {"type": "string", "description": "Cluster column (default: leiden)"}
                },
                "required": []
            }
        },
        {
            "name": "get_top_markers",
            "description": "Get top marker genes for a cluster (requires DEG analysis first).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad with DEG results (optional - uses in-memory data)"},
                    "cluster": {"type": "string", "description": "Cluster ID"},
                    "n_genes": {"type": "integer", "description": "Number of genes (default: 10)"}
                },
                "required": ["cluster"]
            }
        },
        {
            "name": "summarize_qc_metrics",
            "description": "Get summary statistics of QC metrics (library size, genes, MT%, doublet scores).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"}
                },
                "required": []
            }
        },
        {
            "name": "get_celltypes",
            "description": "Get cell type annotation summary (counts per type).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"},
                    "annotation_key": {"type": "string", "description": "Annotation column (default: auto-detect)"}
                },
                "required": []
            }
        },
        {
            "name": "list_obs_columns",
            "description": "List available columns in obs (cell metadata).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"}
                },
                "required": []
            }
        },
    ]

    return action_tools + meta_tools + inspection_tools


def get_openai_tools() -> List[Dict[str, Any]]:
    """
    Get OpenAI-format tool definitions.

    OpenAI uses a different schema format than Anthropic.
    """
    anthropic_tools = get_tools()
    openai_tools = []

    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
        })

    return openai_tools


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64 for vision API."""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type for image."""
    ext = image_path.lower().split(".")[-1]
    mime_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return mime_types.get(ext, "image/png")


def process_tool_call(tool_name: str, tool_input: Dict[str, Any], adata=None) -> tuple:
    """
    Process a tool call and return structured JSON result.

    Returns
    -------
    tuple
        (json_result_string, updated_adata)
    """
    import numpy as np

    from ..core import (
        inspect_data, load_data, run_qc_pipeline, normalize_data,
        run_pca, compute_neighbors, compute_umap,
        run_leiden, run_phenograph, recommend_next_steps,
        calculate_qc_metrics, detect_doublets
    )
    from ..core.normalization import select_hvg
    from ..core.clustering import run_differential_expression, get_top_markers
    from ..annotation import run_celltypist, run_scimilarity
    from ..batch import run_scanorama, run_harmony
    from ..analysis import infer_biological_context, build_literature_context, score_paper_relevance

    def make_state(adata):
        """Create compact state dict."""
        state = inspect_data(adata)
        return {
            "has_raw_counts": state.has_raw_layer,
            "has_qc_metrics": state.has_qc_metrics,
            "has_doublets": state.has_doublet_scores,
            "is_normalized": state.is_normalized,
            "has_hvg": state.has_hvg,
            "has_pca": state.has_pca,
            "has_neighbors": state.has_neighbors,
            "has_umap": state.has_umap,
            "has_clusters": state.has_clusters,
            "has_celltypes": state.has_celltypist or state.has_scimilarity,
        }

    def get_adata(tool_input, existing_adata, update_memory: bool = True, prefer_memory: bool = False):
        """Get adata from memory or load from disk.

        If ``update_memory`` is False, loading from disk is treated as read-only
        and does not replace the active in-memory AnnData tracked by the agent.
        """
        data_path = tool_input.get("data_path")
        if prefer_memory and existing_adata is not None:
            return existing_adata, existing_adata
        # If adata is already in memory and no specific path given, use it
        if existing_adata is not None and (data_path is None or data_path == "memory"):
            return existing_adata, existing_adata
        # Otherwise load from disk
        if data_path and data_path != "memory":
            loaded = load_data(data_path)
            if update_memory:
                return loaded, loaded
            return loaded, existing_adata
        raise ValueError("No data available. Provide data_path or load data first.")

    def fix_output_path(output_path: str, tool_name: str) -> str:
        """Normalize output_path values for h5ad-producing tools."""
        import os as os_module
        if output_path is None:
            return None
        if os_module.path.isdir(output_path):
            if tool_name == "save_data":
                return os_module.path.join(output_path, "final_result.h5ad")
            return None
        return output_path

    def _state_preservation_warning(tool_input, existing_adata):
        if existing_adata is not None and tool_input.get("data_path") not in (None, "memory"):
            return ["Ignored data_path and continued with the in-memory dataset to preserve prior analysis state."]
        return []

    def _validate_obs_column(adata_obj, column_name: str, warnings: List[str], *, required: bool = False, context: str = "parameter"):
        """Validate an obs column reference and either warn or raise a clean error."""
        if not column_name:
            return None
        if column_name not in adata_obj.obs.columns:
            available = list(adata_obj.obs.columns)
            if required:
                raise ValueError(
                    f"{context} '{column_name}' is not present in adata.obs. "
                    f"Available columns: {available}"
                )
            warnings.append(f"Ignored invalid {context} '{column_name}' because it is not present in adata.obs.")
            return None
        return column_name

    def _stringify_dataframe_columns(df):
        if df is None:
            return df
        for col in df.columns:
            try:
                dtype_str = str(df[col].dtype)
            except Exception:
                dtype_str = ""
            if dtype_str in {"object", "category"}:
                df[col] = df[col].astype(str)
        return df

    def _sanitize_uns_value(value):
        import numpy as _np
        import pandas as _pd
        try:
            import scipy.sparse as _sp
        except Exception:
            _sp = None

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (_np.integer, _np.floating, _np.bool_)):
            return value.item()
        if _sp is not None and _sp.issparse(value):
            return value.toarray().tolist()
        if isinstance(value, _np.ndarray):
            if value.dtype.names is not None:
                return {str(name): _sanitize_uns_value(value[name]) for name in value.dtype.names}
            if value.dtype.kind in "biufc":
                return value.tolist()
            if value.dtype.kind in "SU":
                return value.astype(str).tolist()
            return [_sanitize_uns_value(v) for v in value.tolist()]
        if isinstance(value, (_pd.Series, _pd.Index)):
            return [_sanitize_uns_value(v) for v in value.tolist()]
        if isinstance(value, _pd.DataFrame):
            safe_df = value.copy()
            _stringify_dataframe_columns(safe_df)
            return {str(col): [_sanitize_uns_value(v) for v in safe_df[col].tolist()] for col in safe_df.columns}
        if isinstance(value, dict):
            return {str(k): _sanitize_uns_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            sanitized_items = [_sanitize_uns_value(v) for v in value]
            normalized_items = []
            for item in sanitized_items:
                if isinstance(item, (dict, list, tuple, set)):
                    try:
                        normalized_items.append(json.dumps(item, default=str, sort_keys=True))
                    except Exception:
                        normalized_items.append(str(item))
                else:
                    normalized_items.append(item)
            return normalized_items
        return str(value)

    def _make_serializable_copy(current_adata, aggressive_uns: bool = False):
        sanitized = current_adata.copy()
        _stringify_dataframe_columns(sanitized.obs)
        _stringify_dataframe_columns(sanitized.var)
        if sanitized.raw is not None:
            _stringify_dataframe_columns(sanitized.raw.var)
        if aggressive_uns:
            sanitized.uns = {str(k): _sanitize_uns_value(v) for k, v in sanitized.uns.items()}
        return sanitized

    def write_h5ad_safe(current_adata, output_path: str) -> Dict[str, Any]:
        details = {"save_mode": "direct", "warnings": []}
        first_error_msg = None
        second_error_msg = None
        try:
            current_adata.write_h5ad(output_path)
            return details
        except Exception as first_error:
            first_error_msg = str(first_error)
            details["warnings"].append(f"Direct save failed; retrying with obs/var cleanup: {first_error_msg}")

        try:
            sanitized = _make_serializable_copy(current_adata, aggressive_uns=False)
            sanitized.write_h5ad(output_path)
            details["save_mode"] = "clean_obs_var"
            return details
        except Exception as second_error:
            second_error_msg = str(second_error)
            details["warnings"].append(f"Obs/var cleanup save failed; retrying with uns cleanup: {second_error_msg}")

        try:
            sanitized = _make_serializable_copy(current_adata, aggressive_uns=True)
            sanitized.write_h5ad(output_path)
            details["save_mode"] = "clean_obs_var_uns"
            return details
        except Exception as third_error:
            raise RuntimeError(
                "Unable to save AnnData after serialization cleanup. "
                f"Direct error: {first_error_msg}; obs/var cleanup error: {second_error_msg}; uns cleanup error: {third_error}"
            )

    def search_web(query: str, site: str = "", max_results: int = 5) -> Dict[str, Any]:
        """Search web/docs using Tavily (primary), DuckDuckGo (secondary), or Google CSE (last fallback)."""
        import os as os_module
        from urllib.parse import urlparse
        import requests

        scoped_query = f"site:{site} {query}" if site else query
        search_errors = []

        def normalize_domain(value: str) -> str:
            value = (value or "").strip().lower()
            if not value:
                return ""
            if "://" not in value:
                value = f"https://{value}"
            parsed = urlparse(value)
            return parsed.netloc.replace("www.", "")

        def extract_query_tokens(value: str) -> List[str]:
            raw_tokens = [tok.strip(" ,:;()[]{}").lower() for tok in value.split()]
            tokens = [tok for tok in raw_tokens if tok and tok not in {
                "documentation", "docs", "api", "function", "method", "tutorial", "guide",
                "site"
            }]
            return tokens

        query_tokens = extract_query_tokens(query)
        technical_tokens = [
            tok for tok in query_tokens
            if "_" in tok or "." in tok or any(ch.isdigit() for ch in tok) or len(tok) >= 8
        ]
        priority_tokens = technical_tokens or query_tokens[-2:]

        def score_result(item: Dict[str, Any], preferred_domain: str = "") -> tuple:
            domain = normalize_domain(item.get("url", ""))
            title = (item.get("title") or "").lower()
            snippet = (item.get("snippet") or "").lower()
            url = (item.get("url") or "").lower()
            domain_match = 1 if preferred_domain and preferred_domain in domain else 0
            exact_token_match = 0
            partial_token_match = 0
            api_page_bonus = 0

            for token in priority_tokens:
                if token and token in url:
                    exact_token_match += 1
                if token and token in title:
                    partial_token_match += 1

            api_indicators = [
                "/generated/",
                "/api/",
                "api.",
                "reference",
                "class",
                "function",
            ]
            if any(ind in url for ind in api_indicators):
                api_page_bonus = 1

            query_bonus = 0
            for token in query_tokens:
                if token in title:
                    query_bonus += 2
                elif token in snippet:
                    query_bonus += 1
            return (
                exact_token_match,
                domain_match,
                api_page_bonus,
                partial_token_match,
                query_bonus,
                len(snippet),
            )

        def run_ddg(search_query: str) -> List[Dict[str, Any]]:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=max_results))

            snippets = []
            for item in results:
                snippets.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", "") or item.get("url", ""),
                    "snippet": item.get("body", "")[:300],
                })
            return snippets

        def dedupe_and_rank(snippets: List[Dict[str, Any]], preferred_domain: str = "") -> List[Dict[str, Any]]:
            deduped: Dict[str, Dict[str, Any]] = {}
            for item in snippets:
                url = item.get("url", "")
                key = url or f"{item.get('title', '')}|{item.get('snippet', '')}"
                if key not in deduped:
                    deduped[key] = item
            ranked = sorted(
                deduped.values(),
                key=lambda item: score_result(item, preferred_domain=preferred_domain),
                reverse=True,
            )
            return ranked[:max_results]

        preferred_domain = normalize_domain(site)

        def run_tavily(search_query: str, include_domains: List[str]) -> List[Dict[str, Any]]:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": search_query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_domains": include_domains,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            snippets = []
            for item in data.get("results", []):
                snippets.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "")[:300],
                })
            return snippets

        # === TAVILY (Primary - best for AI agents) ===
        tavily_api_key = os_module.environ.get("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                snippets = []
                used_dual_query = False
                if site:
                    snippets.extend(run_tavily(query, [site]))
                    snippets.extend(run_tavily(query, []))
                    used_dual_query = True
                else:
                    snippets.extend(run_tavily(query, []))

                if snippets:
                    snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)
                    return {
                        "status": "ok",
                        "backend": "tavily",
                        "query": query,
                        "results": snippets,
                        "used_dual_query": used_dual_query,
                        "site_filter_requested": bool(site),
                    }
                search_errors.append({"backend": "tavily", "type": "no_results"})
            except requests.HTTPError as e:
                response = getattr(e, "response", None)
                search_errors.append({
                    "backend": "tavily",
                    "type": "http_error",
                    "status_code": response.status_code if response is not None else None,
                    "message": response.text[:200] if response is not None and response.text else str(e),
                })
            except Exception as e:
                search_errors.append({"backend": "tavily", "type": type(e).__name__, "message": str(e)})

        # === DUCKDUCKGO (Secondary) ===
        try:
            ddg_queries = [scoped_query]
            if site:
                ddg_queries.append(query)
                ddg_queries.append(f"{query} {site}")

            snippets = []
            retried_without_site = False
            for idx, ddg_query in enumerate(ddg_queries):
                current = run_ddg(ddg_query)
                if idx > 0 and current:
                    retried_without_site = True
                snippets.extend(current)
                if len(snippets) >= max_results * 2:
                    break

            snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)

            if snippets:
                return {
                    "status": "ok",
                    "backend": "duckduckgo",
                    "query": scoped_query,
                    "results": snippets,
                    "retried_without_site": retried_without_site,
                    "fallback_used": True,
                    "backends_tried": [e["backend"] for e in search_errors],
                    "errors": search_errors if search_errors else None,
                }
        except ImportError:
            search_errors.append({"backend": "duckduckgo", "type": "not_installed", "message": "duckduckgo-search not installed"})

        # === GOOGLE (Last fallback) ===
        google_api_key = os_module.environ.get("GOOGLE_API_KEY")
        google_cx = os_module.environ.get("GOOGLE_CX")

        if google_api_key and google_cx:
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": google_api_key,
                    "cx": google_cx,
                    "q": scoped_query,
                    "num": max(1, min(max_results, 10)),
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                snippets = []
                for item in data.get("items", []):
                    snippets.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")[:300],
                    })

                if snippets:
                    snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)
                    return {
                        "status": "ok",
                        "backend": "google",
                        "query": scoped_query,
                        "results": snippets,
                        "fallback_used": True,
                        "backends_tried": [e["backend"] for e in search_errors],
                        "errors": search_errors if search_errors else None,
                    }
                search_errors.append({"backend": "google", "type": "no_results"})
            except requests.HTTPError as e:
                response = getattr(e, "response", None)
                search_errors.append({
                    "backend": "google",
                    "type": "http_error",
                    "status_code": response.status_code if response is not None else None,
                    "message": response.text[:200] if response is not None and response.text else str(e),
                })
            except Exception as e:
                search_errors.append({"backend": "google", "type": type(e).__name__, "message": str(e)})

        if search_errors:
            return {
                "status": "warning",
                "query": scoped_query,
                "backend": "none",
                "results": [],
                "message": "No search backend returned results for this query.",
                "errors": search_errors,
            }

    def search_pubmed(query: str, max_results: int = 5, recent_years: int = 5, reviews_only: bool = False) -> List[Dict[str, Any]]:
        """Search PubMed and return structured article metadata."""
        import re
        import requests
        from datetime import datetime

        current_year = datetime.now().year
        min_date = f"{current_year - recent_years}/01/01"
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        search_term = query
        if reviews_only:
            search_term += " AND review[pt]"

        search_params = {
            "db": "pubmed",
            "term": search_term,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "mindate": min_date,
            "maxdate": f"{current_year}/12/31",
            "datetype": "pdat",
        }

        try:
            resp = requests.get(esearch_url, params=search_params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "rettype": "abstract",
            }
            resp = requests.get(efetch_url, params=fetch_params, timeout=10)
            resp.raise_for_status()
            xml = resp.text

            results = []
            articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
            for article in articles:
                title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
                abstract_match = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
                pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
                year_match = re.search(r"<PubDate>.*?<Year>(\d+)</Year>", article, re.DOTALL)
                journal_match = re.search(r"<Title>(.*?)</Title>", article)

                if title_match and pmid_match:
                    title = re.sub(r"<[^>]+>", "", title_match.group(1))
                    abstract = re.sub(r"<[^>]+>", "", abstract_match.group(1))[:700] if abstract_match else ""
                    pmid = pmid_match.group(1)
                    year = year_match.group(1) if year_match else "N/A"
                    journal = journal_match.group(1) if journal_match else "N/A"
                    results.append({
                        "pmid": pmid,
                        "title": title,
                        "year": year,
                        "journal": journal,
                        "abstract": abstract,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    })

            return results
        except Exception:
            return []

    def fetch_url_text(url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Fetch a URL and return a compact, structured text summary."""
        import html
        import io
        import re
        from urllib.parse import urlparse
        import requests

        headers = {
            "User-Agent": "scagent/0.1 (+single-cell analysis agent)"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        raw_bytes = resp.content
        text = resp.text
        title = ""
        meta_description = ""
        extracted_with = "plain_text"
        warning = None

        def clean_whitespace(value: str) -> str:
            return re.sub(r"\s+", " ", value).strip()

        def extract_html_text(html_text: str) -> Dict[str, Any]:
            nonlocal warning
            try:
                from bs4 import BeautifulSoup  # type: ignore
            except ImportError:
                BeautifulSoup = None

            if BeautifulSoup is not None:
                soup = BeautifulSoup(html_text, "html.parser")
                for tag in soup(["script", "style", "noscript", "svg"]):
                    tag.decompose()

                page_title = clean_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
                meta = soup.find("meta", attrs={"name": "description"})
                meta_desc = clean_whitespace(meta.get("content", "")) if meta else ""

                main = soup.find("main") or soup.find("article") or soup.body or soup
                parts = []
                for tag in main.find_all(["h1", "h2", "h3", "p", "li"]):
                    text_part = clean_whitespace(tag.get_text(" ", strip=True))
                    if text_part:
                        parts.append(text_part)

                if not parts:
                    parts = [clean_whitespace(main.get_text(" ", strip=True))]

                return {
                    "title": page_title,
                    "meta_description": meta_desc,
                    "text": clean_whitespace(" ".join(parts)),
                    "extracted_with": "beautifulsoup4",
                }

            warning = "beautifulsoup4 not installed; used regex-based HTML extraction."
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
            meta_match = re.search(
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
                html_text,
                re.IGNORECASE | re.DOTALL,
            )
            page_title = html.unescape(clean_whitespace(title_match.group(1))) if title_match else ""
            meta_desc = html.unescape(clean_whitespace(meta_match.group(1))) if meta_match else ""
            cleaned = re.sub(r"<script.*?</script>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<noscript.*?</noscript>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
            cleaned = html.unescape(cleaned)
            cleaned = clean_whitespace(cleaned)
            return {
                "title": page_title,
                "meta_description": meta_desc,
                "text": cleaned,
                "extracted_with": "regex_html",
            }

        def extract_pdf_text(data: bytes) -> Dict[str, Any]:
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError:
                try:
                    from PyPDF2 import PdfReader  # type: ignore
                except ImportError as e:
                    raise ImportError("PDF reader dependency not installed") from e

            reader = PdfReader(io.BytesIO(data))
            pages = []
            for page in reader.pages[:5]:
                page_text = page.extract_text() or ""
                if page_text:
                    pages.append(clean_whitespace(page_text))

            return {
                "title": "",
                "meta_description": "",
                "text": clean_whitespace(" ".join(pages)),
                "extracted_with": "pdf_reader",
            }

        lower_content_type = content_type.lower()
        if "html" in lower_content_type:
            extracted = extract_html_text(text)
            title = extracted["title"]
            meta_description = extracted["meta_description"]
            cleaned = extracted["text"]
            extracted_with = extracted["extracted_with"]
        elif "pdf" in lower_content_type or urlparse(str(resp.url)).path.lower().endswith(".pdf"):
            try:
                extracted = extract_pdf_text(raw_bytes)
                title = extracted["title"]
                meta_description = extracted["meta_description"]
                cleaned = extracted["text"]
                extracted_with = extracted["extracted_with"]
            except ImportError:
                cleaned = ""
                extracted_with = "pdf_unsupported"
                warning = "PDF content fetched but no PDF text extraction library is installed."
        else:
            cleaned = clean_whitespace(text)

        result = {
            "status": "ok",
            "url": url,
            "final_url": str(resp.url),
            "content_type": content_type,
            "domain": urlparse(str(resp.url)).netloc,
            "title": title,
            "meta_description": meta_description,
            "extracted_with": extracted_with,
            "text_excerpt": cleaned[:max_chars],
            "text_length": len(cleaned),
            "truncated": len(cleaned) > max_chars,
        }
        if warning:
            result["warning"] = warning
        if not cleaned:
            result["status"] = "warning"
            result["message"] = "Fetched the URL, but extracted little or no readable text."
        return result

    try:
        # ===== META TOOLS =====
        if tool_name == "ask_user":
            # This is handled specially by the agent loop - just return the question
            return json.dumps({
                "status": "needs_input",
                "tool": "ask_user",
                "question": tool_input["question"],
                "options": tool_input.get("options", []),
                "default": tool_input.get("default", ""),
            }, indent=2), adata

        elif tool_name == "run_code":
            # Execute custom Python code on adata
            import scanpy as sc
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            code = tool_input["code"]
            description = tool_input["description"]
            save_to = tool_input.get("save_to")
            save_warning = None

            # Security: basic checks (not foolproof, but helps)
            forbidden = ["import os", "import sys", "subprocess", "eval(",
                        "__import__", "rm -rf", "shutil.rmtree", "requests."]
            for f in forbidden:
                if f in code:
                    return json.dumps({
                        "status": "error",
                        "tool": "run_code",
                        "message": f"Forbidden operation: {f}"
                    }, indent=2), adata

            # Load data if needed
            if adata is None and "data_path" in tool_input:
                adata = get_adata(tool_input, adata)

            # Execute in controlled namespace
            namespace = {
                "adata": adata,
                "sc": sc,
                "np": np,
                "pd": pd,
                "plt": plt,
                "scanpy": sc,
                "matplotlib": matplotlib,
            }

            # Capture stdout so LLM can see print outputs
            import io
            import sys
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout

            # Capture any figures created
            plt.close('all')

            try:
                sys.stdout = stdout_capture
                exec(code, namespace)
            finally:
                sys.stdout = old_stdout

            captured_output = stdout_capture.getvalue()
            adata = namespace.get("adata", adata)

            # Check if any figures were created
            figures_saved = []
            if plt.get_fignums():
                # There are open figures - check if code saved them
                pass

            if save_to:
                save_warning = "run_code ignored save_to; use save_data to save AnnData after custom code modifications"

            # Save code to file if output directory exists
            code_file = None
            if "output_dir" in tool_input:
                import os
                code_dir = os.path.join(tool_input["output_dir"], "code")
                os.makedirs(code_dir, exist_ok=True)

                # Create filename from description
                safe_desc = "".join(c if c.isalnum() or c in "_ " else "_" for c in description)
                safe_desc = safe_desc.replace(" ", "_")[:50]
                code_file = os.path.join(code_dir, f"{safe_desc}.py")

                with open(code_file, "w") as f:
                    f.write(f'"""\n{description}\n\nAuto-generated by scagent\n"""\n\n')
                    f.write("import scanpy as sc\n")
                    f.write("import numpy as np\n")
                    f.write("import pandas as pd\n")
                    f.write("import matplotlib.pyplot as plt\n\n")
                    f.write("# Load data (adjust path as needed)\n")
                    f.write("# adata = sc.read_h5ad('path/to/data.h5ad')\n\n")
                    f.write("# Generated code:\n")
                    f.write(code)

            result = {
                "status": "ok",
                "tool": "run_code",
                "description": description,
            }
            if adata is not None:
                result["shape"] = {"n_cells": adata.n_obs, "n_genes": adata.n_vars}
            if save_to:
                result["ignored_save_to"] = save_to
            if save_warning:
                result.setdefault("warnings", []).append(save_warning)
            if code_file:
                result["code_file"] = code_file
            if captured_output:
                # Truncate if too long
                result["output"] = captured_output[:2000]
                if len(captured_output) > 2000:
                    result["output_truncated"] = True

            return json.dumps(result, indent=2), adata

        elif tool_name in {"web_search", "web_search_docs"}:
            query = tool_input["query"]
            site = tool_input.get("site", "")
            max_results = tool_input.get("max_results", 5)
            search_result = search_web(query, site=site, max_results=max_results)
            search_result["tool"] = tool_name
            search_result["search_kind"] = "docs"
            if tool_name == "web_search":
                search_result["deprecated_alias"] = True
                search_result["warning"] = (
                    "web_search is retained for backward compatibility. Prefer web_search_docs for documentation lookup."
                )
            search_result["note"] = (
                "Use this for documentation, troubleshooting, and implementation details. "
                "For scientific literature, prefer search_papers or research_findings."
            )
            return json.dumps(search_result, indent=2), adata

        elif tool_name == "search_papers":
            query = tool_input["query"]
            max_results = tool_input.get("max_results", 5)
            recent_years = tool_input.get("recent_years", 5)
            reviews_only = tool_input.get("reviews_only", False)

            results = search_pubmed(
                query=query,
                max_results=max_results,
                recent_years=recent_years,
                reviews_only=reviews_only,
            )

            return json.dumps({
                "status": "ok",
                "tool": "search_papers",
                "backend": "pubmed",
                "query": query,
                "reviews_only": reviews_only,
                "years_searched": f"last {recent_years} years",
                "results": results,
                "note": "Use fetch_url on a selected PubMed URL if you want the landing page text. Prefer PMID-backed evidence for scientific claims.",
            }, indent=2), adata

        elif tool_name == "fetch_url":
            url = tool_input["url"]
            max_chars = tool_input.get("max_chars", 4000)
            fetched = fetch_url_text(url, max_chars=max_chars)
            fetched["tool"] = "fetch_url"
            return json.dumps(fetched, indent=2), adata

        elif tool_name == "research_findings":
            import re

            pathway = tool_input["pathway"]
            cell_type = tool_input["cell_type"]
            genes = tool_input.get("genes", [])
            context = tool_input.get("context", "")
            cluster_confidence = tool_input.get("cluster_confidence")
            recent_years = tool_input.get("recent_years", 3)

            def normalize_pathway_term(term: str) -> str:
                cleaned = term.replace("HALLMARK_", "").replace("GO_", "").replace("REACTOME_", "")
                cleaned = cleaned.replace("_", " ")
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                replacements = {
                    "nf kb": "NF-kB",
                    "il 2": "IL-2",
                    "stat 5": "STAT5",
                    "tgf beta": "TGF-beta",
                }
                lowered = cleaned.lower()
                for old, new in replacements.items():
                    lowered = lowered.replace(old, new.lower())
                return lowered.title().replace("Nf-Kb", "NF-kB").replace("Il-2", "IL-2").replace("Stat5", "STAT5")

            def pathway_tokens(term: str) -> List[str]:
                tokens = [tok.lower() for tok in re.split(r"[^A-Za-z0-9]+", term) if tok]
                stop = {"pathway", "signaling", "response", "process", "hallmark", "up", "down", "v1", "cell", "cells", "immune"}
                return [tok for tok in tokens if tok not in stop and len(tok) > 2]

            def infer_cell_lineage(cell_type_term: str) -> str:
                lowered = re.sub(r"\s+", " ", cell_type_term.lower()).strip()
                if any(tok in lowered for tok in ["cytotoxic t", "helper t", "regulatory t", "treg", "mait", "trm", "tem", "tcm", "t cell"]):
                    return "t_cell"
                if "nk" in lowered or "natural killer" in lowered:
                    return "nk"
                if "pdc" in lowered or "plasmacytoid" in lowered:
                    return "pdc"
                if "dc" in lowered or "dendritic" in lowered:
                    return "dendritic"
                if "monocyte" in lowered or "myelo" in lowered or "myelocyte" in lowered:
                    return "monocyte"
                if "b cell" in lowered:
                    return "b_cell"
                if "plasma" in lowered or "plasmablast" in lowered:
                    return "plasma"
                return "unknown"

            def pathway_query_profile(term: str, cell_type_term: str) -> Dict[str, Any]:
                normalized = normalize_pathway_term(term)
                lineage = infer_cell_lineage(cell_type_term)
                profiles = {
                    "allograft rejection": {
                        "query_terms": [normalized, "immune activation", "cytotoxic lymphocyte", "antigen presentation"],
                        "scoring_terms": ["immune activation", "cytotoxic lymphocyte", "antigen presentation", "t cell activation", "nk cell activation"],
                        "penalty_terms": ["transplant", "allogeneic", "recipient", "donor"],
                    },
                    "interferon gamma response": {
                        "query_terms": [normalized, "interferon gamma signaling", "ifng response", "antigen presentation"],
                        "scoring_terms": ["interferon gamma", "interferon signaling", "antigen presentation"],
                    },
                    "il-2/stat5 signaling": {
                        "query_terms": [normalized, "IL-2 signaling", "STAT5 signaling", "cytokine signaling"],
                        "scoring_terms": ["il-2", "stat5", "cytokine signaling"],
                    },
                    "apical junction": {
                        "query_terms": [normalized, "cell adhesion", "junctional remodeling", "cytoskeletal remodeling"],
                        "scoring_terms": ["cell adhesion", "junction", "cytoskeletal remodeling"],
                    },
                    "kras signaling up": {
                        "query_terms": [normalized, "RAS signaling", "MAPK signaling"],
                        "scoring_terms": ["ras signaling", "mapk signaling", "kras"],
                    },
                    "coagulation": {
                        "query_terms": [normalized, "coagulation", "immunothrombosis", "thromboinflammation"],
                        "scoring_terms": ["coagulation", "immunothrombosis", "thromboinflammation"],
                    },
                    "myc targets v1": {
                        "query_terms": [normalized, "MYC signaling", "MYC target genes", "proliferation"],
                        "scoring_terms": ["myc", "proliferation"],
                    },
                    "p53 pathway": {
                        "query_terms": [normalized, "p53 signaling", "DNA damage response", "apoptosis"],
                        "scoring_terms": ["p53", "dna damage", "apoptosis"],
                    },
                }
                profile = dict(profiles.get(normalized.lower(), {"query_terms": [normalized], "scoring_terms": [normalized]}))

                if normalized.lower() == "allograft rejection":
                    if lineage in {"t_cell", "nk"}:
                        profile["query_terms"] = [normalized, "cytotoxic lymphocyte activation", "t cell activation", "natural killer cell activation"]
                        profile["scoring_terms"] = ["cytotoxic lymphocyte", "t cell activation", "natural killer cell activation", "immune activation"]
                    elif lineage in {"dendritic", "monocyte", "pdc"}:
                        profile["query_terms"] = [normalized, "antigen presentation", "myeloid activation", "interferon response"]
                        profile["scoring_terms"] = ["antigen presentation", "myeloid activation", "interferon response", "inflammatory activation"]
                    elif lineage in {"b_cell", "plasma"}:
                        profile["query_terms"] = [normalized, "antigen presentation", "lymphocyte activation", "b cell activation"]
                        profile["scoring_terms"] = ["antigen presentation", "lymphocyte activation", "b cell activation"]

                if normalized.lower() == "apical junction" and lineage in {"t_cell", "nk", "monocyte", "dendritic", "pdc"}:
                    profile["query_terms"] = [normalized, "cell adhesion", "immune cell migration", "cytoskeletal remodeling", "immune synapse"]
                    profile["scoring_terms"] = ["cell adhesion", "immune cell migration", "cytoskeletal remodeling", "immune synapse"]

                if normalized.lower() == "kras signaling up" and lineage in {"monocyte", "dendritic", "pdc"}:
                    profile["query_terms"] = [normalized, "MAPK signaling", "myeloid activation", "inflammatory signaling"]
                    profile["scoring_terms"] = ["mapk signaling", "myeloid activation", "inflammatory signaling"]

                if normalized.lower() == "myc targets v1" and lineage in {"t_cell", "b_cell", "plasma"}:
                    profile["query_terms"] = [normalized, "lymphocyte proliferation", "MYC signaling", "activation-induced proliferation"]
                    profile["scoring_terms"] = ["lymphocyte proliferation", "myc signaling", "proliferation"]

                profile["display_term"] = normalized
                profile["lineage"] = lineage
                return profile

            def build_cell_type_terms(cell_type_term: str) -> List[str]:
                lowered = re.sub(r"\s+", " ", cell_type_term.lower()).strip().rstrip("s")
                terms = [lowered]
                if any(tok in lowered for tok in ["cytotoxic t", "trm", "tem", "tcm", "helper t", "regulatory t", "treg", "mait", "t cell"]):
                    if "cytotoxic" in lowered or "trm" in lowered or "tem" in lowered:
                        terms.extend(["cytotoxic t cell", "t cell"])
                    elif "regulatory" in lowered or "treg" in lowered:
                        terms.extend(["regulatory t cell", "t cell"])
                    else:
                        terms.append("t cell")
                elif "nk" in lowered or "natural killer" in lowered:
                    terms.extend(["natural killer cell", "nk cell"])
                elif "pdc" in lowered or "plasmacytoid" in lowered:
                    terms.extend(["plasmacytoid dendritic cell", "dendritic cell"])
                elif "dc" in lowered or "dendritic" in lowered:
                    terms.extend(["dendritic cell", "conventional dendritic cell"])
                elif "monocyte" in lowered or "myelo" in lowered:
                    terms.append("monocyte")
                elif "b cell" in lowered:
                    terms.append("b cell")
                elif "plasma" in lowered:
                    terms.extend(["plasma cell", "antibody secreting cell"])

                seen = set()
                ordered = []
                for term in terms:
                    if term and term not in seen:
                        ordered.append(term)
                        seen.add(term)
                return ordered[:3]

            all_findings = {
                "pathway": pathway,
                "cell_type": cell_type,
                "normalized_pathway": normalize_pathway_term(pathway),
                "pubmed_results": [],
                "review_articles": [],
                "gene_specific": [],
                "selected_papers": [],
                "search_strategy": [],
            }

            cell_type_terms = build_cell_type_terms(cell_type)
            pathway_profile = pathway_query_profile(pathway, cell_type)
            context_profile = build_literature_context(
                context,
                cell_type=cell_type,
                cluster_confidence=cluster_confidence,
            )
            normalized_pathway = pathway_profile["display_term"]
            all_findings["normalized_pathway"] = normalized_pathway
            all_findings["pathway_query_terms"] = pathway_profile["query_terms"]
            all_findings["cell_type_query_terms"] = cell_type_terms
            all_findings["context_profile"] = context_profile.to_dict()

            primary_queries = [
                ("pathway_and_cell_type_strict", f'("{normalized_pathway}"[Title/Abstract]) AND ("{cell_type_terms[0]}"[Title/Abstract])'),
                ("pathway_only", f'("{normalized_pathway}"[Title/Abstract])'),
            ]

            for idx, alias_term in enumerate(pathway_profile.get("query_terms", [])[1:3], start=1):
                primary_queries.append(
                    (f"pathway_alias_{idx}_and_cell_type", f'("{alias_term}"[Title/Abstract]) AND ("{cell_type_terms[0]}"[Title/Abstract])')
                )

            if len(cell_type_terms) > 1:
                primary_queries.append(
                    ("pathway_and_broad_cell_type", f'("{normalized_pathway}"[Title/Abstract]) AND ("{cell_type_terms[1]}"[Title/Abstract])')
                )

            if genes and len(genes) >= 2:
                gene_str = " OR ".join([f'"{g}"[Title/Abstract]' for g in genes[:3]])
                primary_queries.append(
                    ("leading_edge_and_cell_type", f'({gene_str}) AND ("{cell_type_terms[0]}"[Title/Abstract])')
                )
                alias_for_genes = pathway_profile.get("query_terms", [normalized_pathway])[1] if len(pathway_profile.get("query_terms", [])) > 1 else normalized_pathway
                primary_queries.append(
                    ("pathway_alias_and_leading_edge", f'("{alias_for_genes}"[Title/Abstract]) AND ({gene_str})')
                )

            seen_pmids = set()
            aggregated_primary = []
            for label, query_str in primary_queries:
                results = search_pubmed(query_str, max_results=4, recent_years=recent_years)
                all_findings["search_strategy"].append({
                    "label": label,
                    "query": query_str,
                    "results_found": len(results),
                })
                for paper in results:
                    if paper["pmid"] not in seen_pmids:
                        seen_pmids.add(paper["pmid"])
                        aggregated_primary.append(paper)

            all_findings["pubmed_results"] = aggregated_primary

            # Review searches
            review_queries = [
                ("review_pathway_and_cell_type", f'("{normalized_pathway}"[Title/Abstract]) AND ("{cell_type_terms[0]}"[Title/Abstract])'),
                ("review_pathway_only", f'("{normalized_pathway}"[Title/Abstract])'),
            ]
            for idx, alias_term in enumerate(pathway_profile.get("query_terms", [])[1:3], start=1):
                review_queries.append(
                    (f"review_alias_{idx}", f'("{alias_term}"[Title/Abstract]) AND ("{cell_type_terms[0]}"[Title/Abstract])')
                )
            review_results = []
            seen_review_pmids = set()
            for label, query_str in review_queries:
                results = search_pubmed(
                    query_str,
                    max_results=3,
                    recent_years=recent_years,
                    reviews_only=True,
                )
                all_findings["search_strategy"].append({
                    "label": label,
                    "query": f"{query_str} AND review[pt]",
                    "results_found": len(results),
                })
                for paper in results:
                    if paper["pmid"] not in seen_review_pmids:
                        seen_review_pmids.add(paper["pmid"])
                        review_results.append(paper)
            scored_reviews = [
                score_paper_relevance(
                    paper,
                    pathway_profile=pathway_profile,
                    cell_type_terms=cell_type_terms,
                    genes_list=genes,
                    context_profile=context_profile,
                    pathway_tokens_fn=pathway_tokens,
                    prefer_reviews=True,
                )
                for paper in review_results
            ]
            scored_reviews.sort(
                key=lambda paper: (paper.get("relevance_score", 0), paper.get("year", "0")),
                reverse=True,
            )
            all_findings["review_articles"] = [paper for paper in scored_reviews if paper.get("relevance_score", 0) >= 2.0][:3]

            # Gene-specific subset for compatibility / inspection
            gene_specific = []
            if genes and len(genes) >= 2:
                for paper in aggregated_primary:
                    scored = score_paper_relevance(
                        paper,
                        pathway_profile=pathway_profile,
                        cell_type_terms=cell_type_terms,
                        genes_list=genes,
                        context_profile=context_profile,
                        pathway_tokens_fn=pathway_tokens,
                    )
                    if any("leading-edge genes" in reason for reason in scored["match_reasons"]):
                        gene_specific.append(scored)
            all_findings["gene_specific"] = gene_specific[:4]

            scored_primary = [
                score_paper_relevance(
                    paper,
                    pathway_profile=pathway_profile,
                    cell_type_terms=cell_type_terms,
                    genes_list=genes,
                    context_profile=context_profile,
                    pathway_tokens_fn=pathway_tokens,
                )
                for paper in aggregated_primary
            ]
            scored_primary.sort(
                key=lambda paper: (paper.get("relevance_score", 0), paper.get("year", "0")),
                reverse=True,
            )
            selected_primary = [paper for paper in scored_primary if paper.get("relevance_score", 0) >= 2.5]
            all_findings["selected_papers"] = selected_primary[:5]

            # Count findings
            total_results = (
                len(all_findings["pubmed_results"]) +
                len(all_findings["review_articles"]) +
                len(all_findings["gene_specific"])
            )

            return json.dumps({
                "status": "ok",
                "tool": "research_findings",
                "pathway": pathway,
                "normalized_pathway": normalized_pathway,
                "cell_type": cell_type,
                "genes_researched": genes[:5] if genes else [],
                "context": context,
                "cluster_confidence": cluster_confidence,
                "context_profile": context_profile.to_dict(),
                "years_searched": f"last {recent_years} years",
                "total_papers_found": total_results,
                "findings": all_findings,
                "recommended_next_steps": [
                    "Prioritize review articles to understand the pathway in this cell context.",
                    "Use primary studies to support concrete biological claims.",
                    "Cite PMIDs when summarizing pathway relevance."
                ],
                "note": "Papers are from PubMed. Review abstracts for biological function, disease context, and therapeutic implications."
            }, indent=2), adata

        elif tool_name == "install_package":
            # Request package installation - requires user approval
            package = tool_input["package"]
            reason = tool_input["reason"]

            return json.dumps({
                "status": "needs_approval",
                "tool": "install_package",
                "package": package,
                "reason": reason,
                "message": f"Agent wants to install '{package}': {reason}"
            }, indent=2), adata

        # ===== INSPECTION TOOLS =====
        elif tool_name == "inspect_data":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            state = inspect_data(working_adata)
            goal = tool_input.get("goal")
            context_hint = tool_input.get("context", "")
            if tool_input.get("data_path"):
                context_hint = " ".join(part for part in [context_hint, str(tool_input.get("data_path"))] if part)
            biological_context = infer_biological_context(working_adata, text_context=context_hint)

            result = {
                "status": "ok",
                "tool": "inspect_data",
                "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
                "data_type": state.data_type,
                "state": make_state(working_adata),
                "embeddings": [k for k in working_adata.obsm.keys()],
                "layers": list(working_adata.layers.keys()),
                "genes": {
                    "format": state.gene_id_format,
                    "has_symbols": state.has_gene_symbols,
                    "has_ensembl": state.has_ensembl_ids,
                    "sample": state.sample_gene_names,
                    "var_columns": list(working_adata.var.columns)[:10],
                },
                "clustering": {
                    "has_clusters": state.has_clusters,
                    "cluster_key": state.cluster_key,
                    "n_clusters": state.n_clusters
                },
                "batch": {
                    "batch_key": state.batch_key,
                    "n_batches": state.n_batches
                },
                "biological_context": biological_context.to_dict(),
            }
            if goal:
                result["recommended_steps"] = recommend_next_steps(state, goal)

            return json.dumps(result, indent=2), updated_adata

        elif tool_name == "get_cluster_sizes":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            key = tool_input.get("cluster_key", "leiden")
            if key not in working_adata.obs:
                return json.dumps({"status": "error", "message": f"No cluster column '{key}'"}), updated_adata

            sizes = working_adata.obs[key].value_counts().to_dict()
            return json.dumps({
                "status": "ok",
                "tool": "get_cluster_sizes",
                "cluster_key": key,
                "n_clusters": len(sizes),
                "sizes": {str(k): int(v) for k, v in sizes.items()}
            }, indent=2), updated_adata

        elif tool_name == "get_top_markers":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            cluster = tool_input["cluster"]
            n_genes = tool_input.get("n_genes", 10)

            markers_df = get_top_markers(working_adata, group=cluster, n_genes=n_genes)
            markers = markers_df[['names', 'scores', 'logfoldchanges', 'pvals_adj']].to_dict('records')

            return json.dumps({
                "status": "ok",
                "tool": "get_top_markers",
                "cluster": cluster,
                "markers": markers
            }, indent=2), updated_adata

        elif tool_name == "summarize_qc_metrics":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)

            metrics = {}
            for col in ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'doublet_score']:
                if col in working_adata.obs:
                    metrics[col] = {
                        "median": float(working_adata.obs[col].median()),
                        "mean": float(working_adata.obs[col].mean()),
                        "min": float(working_adata.obs[col].min()),
                        "max": float(working_adata.obs[col].max())
                    }

            doublet_info = {}
            if 'predicted_doublet' in working_adata.obs:
                doublet_info = {
                    "n_doublets": int(working_adata.obs['predicted_doublet'].sum()),
                    "doublet_rate": float(working_adata.obs['predicted_doublet'].mean())
                }

            return json.dumps({
                "status": "ok",
                "tool": "summarize_qc_metrics",
                "n_cells": working_adata.n_obs,
                "metrics": metrics,
                "doublets": doublet_info
            }, indent=2), updated_adata

        elif tool_name == "get_celltypes":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)

            # Find annotation column
            key = tool_input.get("annotation_key")
            if not key:
                for candidate in ['celltypist_majority_voting', 'celltypist_predicted_labels',
                                  'scimilarity_representative_prediction', 'cell_type', 'celltype']:
                    if candidate in working_adata.obs:
                        key = candidate
                        break

            if not key or key not in working_adata.obs:
                return json.dumps({"status": "error", "message": "No cell type annotations found"}), updated_adata

            counts = working_adata.obs[key].value_counts()
            total_cells = working_adata.n_obs

            # Build detailed breakdown with percentages
            breakdown = {}
            for ct, count in counts.items():
                breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }

            # Group by major categories if there are many types
            major_types = {}
            if len(counts) > 5:
                for ct in counts.head(10).index:
                    major_types[str(ct)] = {
                        "count": int(counts[ct]),
                        "percent": round(100.0 * counts[ct] / total_cells, 1)
                    }

            return json.dumps({
                "status": "ok",
                "tool": "get_celltypes",
                "annotation_key": key,
                "total_cells": total_cells,
                "n_types": len(counts),
                "top_10_types": major_types if major_types else breakdown,
                "all_types": breakdown
            }, indent=2), updated_adata

        elif tool_name == "list_obs_columns":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            return json.dumps({
                "status": "ok",
                "tool": "list_obs_columns",
                "columns": list(working_adata.obs.columns),
                "n_columns": len(working_adata.obs.columns)
            }, indent=2), updated_adata

        # ===== ACTION TOOLS =====
        elif tool_name == "run_qc":
            warnings = []
            if adata is not None and tool_input.get("data_path") not in (None, "memory"):
                warnings.append(
                    "Ignored data_path and continued with the in-memory dataset to preserve prior analysis state."
                )

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_before, g_before = adata.n_obs, adata.n_vars

            batch_key = _validate_obs_column(adata, tool_input.get("batch_key"), warnings, required=False, context="batch_key")
            if not batch_key:
                # Auto-detect
                for k in ['batch', 'batch_id', 'sample', 'sample_id']:
                    if k in adata.obs and adata.obs[k].nunique() > 1:
                        batch_key = k
                        warnings.append(f"batch_key auto-detected as '{k}'")
                        break
            elif adata.obs[batch_key].nunique() <= 1:
                warnings.append(f"Ignored batch_key '{batch_key}' because it has only one unique value.")
                batch_key = None

            detect_doublets_flag = tool_input.get("detect_doublets_flag", True)
            remove_ribo = tool_input.get("remove_ribo", True)
            remove_mt = tool_input.get("remove_mt", False)
            min_cells = tool_input.get("min_cells")
            requested_mt_threshold = tool_input.get("mt_threshold")
            preview_only = bool(tool_input.get("preview_only", False))

            qc_preview = adata.copy()
            try:
                calculate_qc_metrics(qc_preview, inplace=True)
                if detect_doublets_flag:
                    detect_doublets(qc_preview, batch_key=batch_key, inplace=True)
            except ValueError as e:
                if detect_doublets_flag and "skimage is not installed" in str(e):
                    warnings.append("Scrublet auto-threshold requires skimage; preview reran without doublet detection.")
                    detect_doublets_flag = False
                else:
                    raise

            if requested_mt_threshold is None:
                median_mt_preview = float(qc_preview.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in qc_preview.obs else 0.0
                auto_data_type = "nuclei" if median_mt_preview < 2.0 else "cells"
                mt_threshold = 5.0 if auto_data_type == "nuclei" else 25.0
                warnings.append(
                    f"mt_threshold auto-selected as {mt_threshold:.1f}% based on median pct_counts_mt={median_mt_preview:.2f} ({auto_data_type}-like)."
                )
            else:
                mt_threshold = float(requested_mt_threshold)
                auto_data_type = "user-specified"

            if min_cells is None:
                from ..config.defaults import QC_DEFAULTS
                min_cells = int(QC_DEFAULTS.min_cells_per_gene)

            cells_over_mt = int((qc_preview.obs['pct_counts_mt'] >= mt_threshold).sum()) if 'pct_counts_mt' in qc_preview.obs else 0
            predicted_doublets = int(qc_preview.obs['predicted_doublet'].sum()) if 'predicted_doublet' in qc_preview.obs else 0
            genes_low_cells = int((qc_preview.var['n_cells_by_counts'] < min_cells).sum()) if 'n_cells_by_counts' in qc_preview.var.columns else 0
            ribo_genes = int(qc_preview.var['ribo'].sum()) if 'ribo' in qc_preview.var.columns and remove_ribo else 0
            mt_genes = int(qc_preview.var['mt'].sum()) if 'mt' in qc_preview.var.columns and remove_mt else 0

            figure_outputs = []
            figure_dir = tool_input.get("figure_dir")
            if figure_dir:
                import os
                import scanpy as sc
                import matplotlib.pyplot as plt
                os.makedirs(figure_dir, exist_ok=True)
                try:
                    qc_plot_adata = qc_preview.copy()
                    violin_keys = []
                    if "total_counts" in qc_plot_adata.obs.columns:
                        qc_plot_adata.obs["log10_total_counts"] = np.log10(qc_plot_adata.obs["total_counts"] + 1)
                        violin_keys.append("log10_total_counts")
                    if "n_genes_by_counts" in qc_plot_adata.obs.columns:
                        qc_plot_adata.obs["log10_n_genes_by_counts"] = np.log10(qc_plot_adata.obs["n_genes_by_counts"] + 1)
                        violin_keys.append("log10_n_genes_by_counts")
                    violin_keys.extend(
                        key for key in ["pct_counts_mt", "pct_counts_ribo", "doublet_score"]
                        if key in qc_plot_adata.obs.columns
                    )
                    if violin_keys:
                        sc.pl.violin(qc_plot_adata, violin_keys, jitter=0.2, multi_panel=True, show=False)
                        violin_path = os.path.join(figure_dir, "qc_violin.png")
                        plt.savefig(violin_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        figure_outputs.append(violin_path)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    if 'n_genes_by_counts' in qc_plot_adata.obs.columns and 'total_counts' in qc_plot_adata.obs.columns:
                        sc.pl.scatter(qc_plot_adata, x='total_counts', y='n_genes_by_counts', ax=axes[0], show=False)
                        axes[0].set_xscale('log')
                        axes[0].set_yscale('log')
                    if 'pct_counts_mt' in qc_plot_adata.obs.columns and 'total_counts' in qc_plot_adata.obs.columns:
                        sc.pl.scatter(qc_plot_adata, x='total_counts', y='pct_counts_mt', ax=axes[1], show=False)
                        axes[1].set_xscale('log')
                        axes[1].axhline(mt_threshold, color='red', linestyle='--', linewidth=1)
                    scatter_path = os.path.join(figure_dir, "qc_scatter.png")
                    fig.tight_layout()
                    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    figure_outputs.append(scatter_path)
                except Exception as e:
                    warnings.append(f"QC figure generation failed: {e}")

            qc_decisions = {
                "mt_threshold": {
                    "value": mt_threshold,
                    "reason": (
                        f"Cells with pct_counts_mt >= {mt_threshold:.1f}% are usually stressed, damaged, or dying; "
                        "high mitochondrial RNA often indicates low-quality cells."
                    ),
                    "cells_flagged": cells_over_mt,
                },
                "min_cells_per_gene": {
                    "value": int(min_cells),
                    "reason": (
                        f"Genes detected in fewer than {int(min_cells)} cells add noise and little clustering signal."
                    ),
                    "genes_flagged": genes_low_cells,
                },
                "doublet_detection": {
                    "enabled": bool(detect_doublets_flag),
                    "reason": (
                        "Scrublet flags likely multiplets; these are reported so the user can decide whether to exclude them."
                    ),
                    "cells_flagged": predicted_doublets,
                    "removal_default": False,
                },
                "remove_ribo": {
                    "enabled": bool(remove_ribo),
                    "reason": "Ribosomal genes can dominate variance and dilute biologically informative structure.",
                    "genes_flagged": ribo_genes,
                },
                "remove_mt_genes": {
                    "enabled": bool(remove_mt),
                    "reason": "Mitochondrial genes are often excluded from downstream feature selection to reduce QC-driven signal.",
                    "genes_flagged": mt_genes,
                },
            }

            recommendation = (
                f"I recommend filtering {cells_over_mt} high-MT cells with pct_counts_mt >= {mt_threshold:.1f}%, "
                f"removing {genes_low_cells} low-detection genes, and "
                f"{'flagging' if detect_doublets_flag else 'skipping'} doublets ({predicted_doublets} cells)."
            )

            if preview_only:
                return json.dumps({
                    "status": "ok",
                    "tool": "run_qc",
                    "mode": "preview",
                    "before": {"n_cells": n_before, "n_genes": g_before},
                    "after": {"n_cells": n_before, "n_genes": g_before},
                    "recommended_data_type": auto_data_type,
                    "recommendation": recommendation,
                    "qc_decisions": qc_decisions,
                    "metrics": {
                        "median_pct_mt": float(qc_preview.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in qc_preview.obs else None,
                        "doublet_rate": float(qc_preview.obs['predicted_doublet'].mean()) if 'predicted_doublet' in qc_preview.obs else None,
                    },
                    "warnings": warnings,
                    "figures": figure_outputs,
                    "state": make_state(adata)
                }, indent=2), adata

            try:
                run_qc_pipeline(
                    adata,
                    mt_threshold=mt_threshold,
                    min_cells=min_cells,
                    remove_ribo=remove_ribo,
                    detect_doublets_flag=detect_doublets_flag,
                    batch_key=batch_key,
                )
            except ValueError as e:
                if detect_doublets_flag and "skimage is not installed" in str(e):
                    warnings.append("Scrublet auto-threshold requires skimage; reran QC without doublet detection.")
                    detect_doublets_flag = False
                    run_qc_pipeline(
                        adata,
                        mt_threshold=mt_threshold,
                        min_cells=min_cells,
                        remove_ribo=remove_ribo,
                        detect_doublets_flag=False,
                        batch_key=batch_key,
                    )
                else:
                    raise

            output_path = fix_output_path(tool_input.get("output_path"), "run_qc")
            if output_path:
                write_h5ad_safe(adata, output_path)

            return json.dumps({
                "status": "ok",
                "tool": "run_qc",
                "input_path": tool_input.get("data_path", "memory"),
                "output_path": output_path,
                "saved": output_path is not None,
                "before": {"n_cells": n_before, "n_genes": g_before},
                "after": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "recommendation": recommendation,
                "qc_decisions": qc_decisions,
                "metrics": {
                    "cells_removed": n_before - adata.n_obs,
                    "genes_removed": g_before - adata.n_vars,
                    "doublet_rate": float(adata.obs['predicted_doublet'].mean()) if 'predicted_doublet' in adata.obs else None,
                    "median_pct_mt": float(adata.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in adata.obs else None,
                },
                "warnings": warnings,
                "figures": figure_outputs,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "normalize_and_hvg":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            normalize_data(adata)
            n_hvg = tool_input.get("n_hvg", 4000)
            select_hvg(adata, n_top_genes=n_hvg)

            output_path = fix_output_path(tool_input.get("output_path"), "normalize_and_hvg")
            if output_path:
                write_h5ad_safe(adata, output_path)

            return json.dumps({
                "status": "ok",
                "tool": "normalize_and_hvg",
                "output_path": output_path,
                "saved": output_path is not None,
                "n_hvg": int(adata.var['highly_variable'].sum()),
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_dimred":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_pcs = tool_input.get("n_pcs", 30)
            n_neighbors = tool_input.get("n_neighbors", 30)

            run_pca(adata, n_comps=n_pcs)
            compute_neighbors(adata, n_neighbors=n_neighbors)
            compute_umap(adata)

            output_path = fix_output_path(tool_input.get("output_path"), "run_dimred")
            if output_path:
                write_h5ad_safe(adata, output_path)

            return json.dumps({
                "status": "ok",
                "tool": "run_dimred",
                "output_path": output_path,
                "saved": output_path is not None,
                "n_pcs": n_pcs,
                "n_neighbors": n_neighbors,
                "variance_explained": float(adata.uns['pca']['variance_ratio'].sum()),
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_clustering":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "leiden")
            resolution = tool_input.get("resolution", 1.0)

            if method == "leiden":
                run_leiden(adata, resolution=resolution)
                cluster_key = "leiden"
            else:
                run_phenograph(adata, resolution=resolution)
                cluster_key = "pheno_leiden"

            output_path = fix_output_path(tool_input.get("output_path"), "run_clustering")
            if output_path:
                write_h5ad_safe(adata, output_path)
            sizes = adata.obs[cluster_key].value_counts().to_dict()

            return json.dumps({
                "status": "ok",
                "tool": "run_clustering",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": method,
                "resolution": resolution,
                "n_clusters": len(sizes),
                "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_celltypist":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            model = tool_input.get("model", "Immune_All_Low.pkl")
            majority = tool_input.get("majority_voting", True)

            run_celltypist(adata, model=model, majority_voting=majority)

            output_path = fix_output_path(tool_input.get("output_path"), "run_celltypist")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get detailed type breakdown
            key = 'celltypist_majority_voting' if majority and 'celltypist_majority_voting' in adata.obs else 'celltypist_predicted_labels'
            all_counts = adata.obs[key].value_counts() if key in adata.obs else {}
            total_cells = adata.n_obs

            # Build detailed breakdown with counts and percentages
            type_breakdown = {}
            for ct, count in all_counts.items():
                type_breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }

            return json.dumps({
                "status": "ok",
                "tool": "run_celltypist",
                "output_path": output_path,
                "saved": output_path is not None,
                "model": model,
                "majority_voting": majority,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "cell_type_breakdown": type_breakdown,
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_scimilarity":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            model_path = tool_input.get("model_path")

            # Only pass model_path if specified, otherwise use default
            if model_path:
                run_scimilarity(adata, model_path=model_path)
            else:
                run_scimilarity(adata)

            output_path = fix_output_path(tool_input.get("output_path"), "run_scimilarity")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get detailed type breakdown
            key = 'scimilarity_predictions_unconstrained'
            if key not in adata.obs:
                key = 'scimilarity_representative_prediction'

            all_counts = adata.obs[key].value_counts() if key in adata.obs else {}
            total_cells = adata.n_obs

            # Build detailed breakdown with counts and percentages
            type_breakdown = {}
            for ct, count in all_counts.items():
                type_breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }

            return json.dumps({
                "status": "ok",
                "tool": "run_scimilarity",
                "output_path": output_path,
                "saved": output_path is not None,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "has_embeddings": "X_scimilarity" in adata.obsm,
                "cell_type_breakdown": type_breakdown,
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "save_data":
            if adata is None:
                return json.dumps({
                    "status": "error",
                    "tool": "save_data",
                    "message": "No in-memory data available to save. Run an analysis tool first."
                }, indent=2), adata

            output_path = fix_output_path(tool_input.get("output_path"), "save_data")
            if not output_path:
                return json.dumps({
                    "status": "error",
                    "tool": "save_data",
                    "message": "Provide an .h5ad output_path or a directory where the final_result.h5ad can be written."
                }, indent=2), adata

            save_details = write_h5ad_safe(adata, output_path)

            return json.dumps({
                "status": "ok",
                "tool": "save_data",
                "output_path": output_path,
                "saved": True,
                "save_mode": save_details.get("save_mode", "direct"),
                "warnings": save_details.get("warnings", []),
                "shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_batch_correction":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "harmony")
            batch_key = _validate_obs_column(adata, tool_input["batch_key"], warnings, required=True, context="batch_key")

            # Get batch sizes for output
            batch_sizes = adata.obs[batch_key].value_counts().to_dict()

            if method == "harmony":
                run_harmony(adata, batch_key=batch_key)
                corrected_rep = 'X_pca_harmony'
            else:
                run_scanorama(adata, batch_key=batch_key)
                corrected_rep = 'X_scanorama'

            # Recompute neighbors and UMAP on corrected embedding
            compute_neighbors(adata, n_neighbors=30, use_rep=corrected_rep)
            compute_umap(adata)

            output_path = fix_output_path(tool_input.get("output_path"), "run_batch_correction")
            if output_path:
                write_h5ad_safe(adata, output_path)

            return json.dumps({
                "status": "ok",
                "tool": "run_batch_correction",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": method,
                "batch_key": batch_key,
                "n_batches": len(batch_sizes),
                "batch_sizes": {str(k): int(v) for k, v in batch_sizes.items()},
                "corrected_embedding": corrected_rep,
                "umap_recomputed": True,
                "note": f"UMAP recomputed using corrected {corrected_rep} embedding",
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_deg":
            from ..analysis.deg import run_validated_deg, get_deg_caveats
            from ..config.defaults import DEG_DEFAULTS

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            groupby = _validate_obs_column(
                adata,
                tool_input.get("groupby", "leiden"),
                warnings,
                required=True,
                context="groupby"
            )
            method = tool_input.get("method", "wilcoxon")
            layer = tool_input.get("layer")
            target_geneset = tool_input.get("target_geneset", DEG_DEFAULTS.default_geneset)

            # Run validated DEG - this validates inputs, runs rank_genes_groups,
            # validates outputs, and attaches validity metadata to adata.uns
            try:
                _, validity_report = run_validated_deg(
                    adata,
                    groupby=groupby,
                    method=method,
                    layer=layer,
                    target_geneset=target_geneset,
                    min_cluster_size=DEG_DEFAULTS.min_cluster_size,
                    warn_cluster_size=DEG_DEFAULTS.warn_cluster_size,
                    imbalance_ratio=DEG_DEFAULTS.max_imbalance_ratio,
                    block_on_errors=False,  # Don't block, let agent see issues
                    batch_confound_threshold=DEG_DEFAULTS.batch_confound_threshold,
                    max_logfc=DEG_DEFAULTS.max_logfc_sanity,
                    inplace=True,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "tool": "run_deg",
                    "message": str(e),
                }, indent=2), adata

            output_path = fix_output_path(tool_input.get("output_path"), "run_deg")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get top 5 markers per cluster for immediate insight
            groups = list(adata.obs[groupby].unique())
            top_markers_summary = {}
            for group in groups[:15]:  # Limit to first 15 clusters for response size
                try:
                    markers_df = get_top_markers(adata, group=str(group), n_genes=5)
                    top_markers_summary[str(group)] = [
                        {
                            "gene": row['names'],
                            "logfc": round(row['logfoldchanges'], 2),
                            "pval_adj": float(f"{row['pvals_adj']:.2e}")
                        }
                        for _, row in markers_df.iterrows()
                    ]
                except Exception:
                    pass

            # Build validation summary for response
            validity_summary = {
                "is_valid": validity_report.is_valid,
                "has_warnings": validity_report.has_warnings,
                "n_errors": len(validity_report.errors),
                "n_warnings": len(validity_report.warnings),
                "matrix_type": validity_report.matrix_type,
                "data_species": validity_report.data_species,
                "gene_id_format": validity_report.gene_id_format,
            }

            # Include specific issues for agent awareness
            if validity_report.errors:
                validity_summary["errors"] = [e.message for e in validity_report.errors]
            if validity_report.warnings:
                validity_summary["warnings"] = [w.message for w in validity_report.warnings]

            # Get caveats that should propagate to GSEA
            deg_caveats = get_deg_caveats(adata)

            return json.dumps({
                "status": "ok",
                "tool": "run_deg",
                "output_path": output_path,
                "saved": output_path is not None,
                "groupby": groupby,
                "method": method,
                "n_groups": len(groups),
                "requested_layer": layer,
                "layer_used": validity_report.layer_used,
                "cluster_sizes": validity_report.cluster_sizes,
                "validity": validity_summary,
                "caveats_for_gsea": deg_caveats,
                "top_markers_per_cluster": top_markers_summary,
                "note": "Validity metadata stored in adata.uns['deg_validity'] - will propagate to GSEA",
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_gsea":
            import os
            import scanpy as sc
            from ..analysis.deg import get_deg_validity, get_cluster_caveats

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            output_dir = tool_input["output_dir"]
            cluster = tool_input["cluster"]
            gene_sets = tool_input.get("gene_sets", "KEGG_2021_Human")
            min_size = tool_input.get("min_size", 5)
            max_size = tool_input.get("max_size", 500)
            permutation_num = tool_input.get("permutation_num", 1000)

            # Check DEG results exist
            if 'rank_genes_groups' not in adata.uns:
                return json.dumps({
                    "status": "error",
                    "tool": "run_gsea",
                    "message": "No DEG results found. Run run_deg first.",
                    "warnings": warnings,
                }, indent=2), adata

            # Get DEG validity info for caveats
            deg_validity = get_deg_validity(adata)
            deg_caveats = adata.uns.get("deg_caveats", [])

            try:
                import gseapy
            except ImportError:
                return json.dumps({
                    "status": "error",
                    "tool": "run_gsea",
                    "message": "gseapy not installed. Use install_package tool first.",
                    "install_command": "pip install gseapy"
                }, indent=2), adata

            os.makedirs(output_dir, exist_ok=True)

            # Get clusters to analyze
            groupby = adata.uns['rank_genes_groups']['params']['groupby']
            max_clusters = tool_input.get("max_clusters")
            if cluster == 'all':
                clusters_to_analyze = list(adata.obs[groupby].unique())
                if max_clusters is not None:
                    clusters_to_analyze = clusters_to_analyze[: int(max_clusters)]
            else:
                clusters_to_analyze = [cluster]
            all_results = {}

            for clust in clusters_to_analyze:
                try:
                    # Get DEG results for this cluster
                    deg_df = sc.get.rank_genes_groups_df(adata, group=str(clust))

                    # Create ranked gene list (gene -> score)
                    # Use scores from DEG (stat values work well for GSEA)
                    df_rank = deg_df[['names', 'scores']].dropna()
                    df_rank = df_rank.set_index('names')['scores']

                    # Run GSEA prerank
                    gsea_outdir = os.path.join(output_dir, f"cluster_{clust}")
                    pre_res = gseapy.prerank(
                        rnk=df_rank,
                        gene_sets=gene_sets,
                        threads=1,
                        min_size=min_size,
                        max_size=max_size,
                        permutation_num=permutation_num,
                        outdir=gsea_outdir,
                        seed=42,
                        verbose=False,
                    )

                    # Get top results
                    res_df = pre_res.res2d
                    res_df = res_df.sort_values('NES', ascending=False)

                    # Top 5 upregulated and top 5 downregulated
                    top_up = res_df[res_df['NES'] > 0].head(5)
                    top_down = res_df[res_df['NES'] < 0].tail(5)

                    cluster_results = {
                        "upregulated_pathways": [
                            {
                                "term": row['Term'],
                                "nes": round(row['NES'], 2),
                                "fdr": float(f"{row['FDR q-val']:.2e}"),
                                "genes": row['Lead_genes'].split(';')[:5] if row['Lead_genes'] else []
                            }
                            for _, row in top_up.iterrows()
                        ],
                        "downregulated_pathways": [
                            {
                                "term": row['Term'],
                                "nes": round(row['NES'], 2),
                                "fdr": float(f"{row['FDR q-val']:.2e}"),
                                "genes": row['Lead_genes'].split(';')[:5] if row['Lead_genes'] else []
                            }
                            for _, row in top_down.iterrows()
                        ],
                        "total_significant": int((res_df['FDR q-val'] < 0.25).sum()),
                    }
                    all_results[str(clust)] = cluster_results

                except Exception as e:
                    all_results[str(clust)] = {"error": str(e)}

            # Add per-cluster caveats based on DEG validity
            cluster_caveats = {}
            for clust in clusters_to_analyze:
                caveats = get_cluster_caveats(adata, str(clust))
                if caveats:
                    cluster_caveats[str(clust)] = caveats

            # Build response with validity metadata
            response = {
                "status": "ok",
                "tool": "run_gsea",
                "output_dir": output_dir,
                "gene_sets": gene_sets,
                "clusters_analyzed": clusters_to_analyze,
                "results": all_results,
                "recommended_next_steps": [
                    "Use research_findings on the most significant pathway terms for biological interpretation.",
                    "Use search_papers for broader literature questions or recent reviews.",
                    "Use web_search_docs for pathway database or software documentation questions."
                ],
                "note": "NES > 0 means pathway upregulated in this cluster. FDR < 0.25 is typically significant.",
            }

            # Add DEG validity info if present
            if deg_validity:
                response["deg_validity"] = {
                    "is_valid": deg_validity.get("is_valid", True),
                    "has_warnings": deg_validity.get("has_warnings", False),
                    "matrix_type": deg_validity.get("matrix_type"),
                    "data_species": deg_validity.get("data_species"),
                    "gene_id_format": deg_validity.get("gene_id_format"),
                }
            if deg_caveats:
                response["deg_caveats"] = deg_caveats
            if cluster_caveats:
                response["cluster_caveats"] = cluster_caveats

            return json.dumps(response, indent=2), adata

        elif tool_name == "generate_figure":
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import scanpy as sc

            adata, _ = get_adata(tool_input, adata)
            plot_type = tool_input["plot_type"]
            output_path = tool_input.get("output_path")
            if not output_path:
                safe_color = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(tool_input.get("color_by", "plot")))
                output_path = f"{plot_type}_{safe_color or 'plot'}.png"
            color_by = tool_input.get("color_by", "leiden")
            genes = tool_input.get("genes", [])
            include_image = tool_input.get("include_image", True)

            fig, ax = plt.subplots(figsize=(10, 8))

            if plot_type == "umap":
                sc.pl.umap(adata, color=color_by, ax=ax, show=False)
            elif plot_type == "violin":
                sc.pl.violin(adata, keys=genes or [color_by], groupby=color_by, ax=ax, show=False)
            elif plot_type == "dotplot" and genes:
                sc.pl.dotplot(adata, var_names=genes, groupby=color_by, show=False)
            elif plot_type == "heatmap" and genes:
                sc.pl.heatmap(adata, var_names=genes, groupby=color_by, show=False)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            result = {
                "status": "ok",
                "tool": "generate_figure",
                "output_path": output_path,
                "plot_type": plot_type,
                "color_by": color_by,
            }

            # Include base64 image for vision models
            if include_image:
                result["image_base64"] = encode_image_base64(output_path)
                result["image_mime"] = get_image_mime_type(output_path)

            return json.dumps(result, indent=2), adata

        else:
            return json.dumps({
                "status": "error",
                "message": f"Unknown tool: {tool_name}"
            }), adata

    except Exception as e:
        return json.dumps({
            "status": "error",
            "tool": tool_name,
            "message": str(e),
            "error_type": type(e).__name__
        }, indent=2), adata
