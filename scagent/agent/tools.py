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
from pathlib import Path
import re


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
            "description": "Run Leiden or PhenoGraph clustering. Preserves alternative clustering results under explicit keys so comparisons do not overwrite the primary clustering by accident.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "method": {"type": "string", "enum": ["leiden", "phenograph"], "description": "Method (default: leiden)"},
                    "resolution": {"type": "number", "description": "Resolution (default: 1.0)"},
                    "cluster_key": {"type": "string", "description": "Optional explicit obs column to store this clustering result. If omitted, scagent will keep primary aliases like 'leiden' stable and store comparisons under deterministic keys like 'leiden_res_0_5'."},
                    "make_primary": {"type": "boolean", "description": "If true, promote this clustering to the default alias for the method (for example 'leiden') while preserving the explicit result key."}
                },
                "required": []
            }
        },
        {
            "name": "compare_clusterings",
            "description": "Run a safe clustering comparison across multiple resolutions without overwriting earlier results. Use this instead of chaining several run_clustering calls when the goal is to compare resolutions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "method": {"type": "string", "enum": ["leiden", "phenograph"], "description": "Method (default: leiden)"},
                    "resolutions": {"type": "array", "items": {"type": "number"}, "description": "List of resolutions to compare"},
                    "generate_figures": {"type": "boolean", "description": "If true and UMAP is present, save one figure per clustering"},
                    "figure_dir": {"type": "string", "description": "Optional directory for generated comparison figures"},
                    "include_images": {"type": "boolean", "description": "If true, include image data for generated figures"},
                    "promote_resolution": {"type": "number", "description": "Optional resolution to promote to the primary alias after comparison"}
                },
                "required": ["resolutions"]
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
                    "majority_voting": {"type": "boolean", "description": "Use majority voting (default: true)"},
                    "cluster_key": {"type": "string", "description": "Cluster column to use for CellTypist majority voting (default: leiden)"}
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
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "cluster_key": {"type": "string", "description": "Cluster column to use for cluster-level representative predictions (default: leiden)"}
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
                "required": []
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
            "description": "Generate and save a visualization (UMAP, violin, dotplot, etc.). For clustering comparisons, always use an explicit cluster key returned by run_clustering or compare_clusterings rather than a bare primary alias unless you intentionally want the promoted default.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save PNG figure"},
                    "plot_type": {"type": "string", "enum": ["umap", "violin", "dotplot", "heatmap"], "description": "Plot type"},
                    "color_by": {"type": "string", "description": "Column or gene to color by"},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Genes for dotplot/heatmap"},
                    "include_image": {"type": "boolean", "description": "If true, include image data for model review (default: true)"}
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
    # Note: ask_user is intentionally absent. The agent follows a turn-based
    # model (like Claude Code / Codex): run all tools to completion, produce a
    # final response with numbered options, then wait for the user's next message.
    # The user's reply comes back through the CLI loop as a normal analyze() call
    # so state, data, and conversation history are always fully maintained.
    meta_tools = [
        {
            "name": "run_code",
            "description": "FLEXIBLE FALLBACK: Execute custom Python code on the AnnData object. This is your most versatile tool - use it for ANY valid request not covered by specialized tools. Examples: custom plots (variance explained, gene correlations, histograms), data filtering (remove clusters, subset cells), calculations (cluster sizes, gene stats), or any scanpy/pandas operation. Access: adata, sc (scanpy), plt (matplotlib), np, pd, output_dir, Path, ensure_dir(path). Use ensure_dir() to create directories before saving. ALWAYS prefer this over saying 'I can't do that'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Has access to: adata, sc, plt, np, pd, output_dir, Path, ensure_dir(), write_report(). Key helpers: ensure_dir(path) creates the dir and returns a Path — use it for figures: fig_dir = ensure_dir(Path(output_dir) / 'figures'); out = fig_dir / 'plot.png'. write_report(name, content) saves a markdown report to reports/name.md and returns the path — always use this instead of open() when saving text results, never write .txt files. Do NOT import os."},
                    "description": {"type": "string", "description": "Brief description of what the code does"},
                    "save_to": {"type": "string", "description": "Optional path to save adata after execution"}
                },
                "required": ["code", "description"]
            }
        },
        {
            "name": "web_search",
            "description": (
                "Search the web for documentation, API references, package guides, troubleshooting, and tutorials. "
                "Use the `site` parameter to target specific documentation sources. "
                "Common bioinformatics doc sites: scanpy.readthedocs.io, anndata.readthedocs.io, "
                "celltypist.readthedocs.io, gseapy.readthedocs.io, harmonypy.readthedocs.io, "
                "scvi-tools.readthedocs.io, muon.readthedocs.io, squidpy.readthedocs.io. "
                "For troubleshooting use scverse.discourse.org or github.com. "
                "Use search_papers instead for peer-reviewed scientific evidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — be specific (e.g., 'scanpy normalize_total target_sum parameter' rather than 'scanpy normalize')"},
                    "site": {"type": "string", "description": "Optional domain to restrict results (e.g., 'scanpy.readthedocs.io')"},
                    "max_results": {"type": "integer", "description": "Maximum results (default: 5)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_papers",
            "description": (
                "Search PubMed for peer-reviewed scientific literature. Use for: cell type markers, "
                "pathway biology, disease mechanisms, method papers, and any claim that needs a citation. "
                "Automatically normalises GSEA/gene set names (strips HALLMARK_, REACTOME_, GO_ prefixes). "
                "Returns PMID, first author, year, journal, abstract, and PubMed URL. "
                "Use web_search for package documentation instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text query or PubMed search string. GSEA set names are normalised automatically (e.g. 'HALLMARK_TNFA_SIGNALING_VIA_NFKB' → 'TNF alpha signaling NF-kB'). Be specific: include cell type, disease, or gene names for better results."},
                    "max_results": {"type": "integer", "description": "Maximum papers (default: 5)"},
                    "recent_years": {"type": "integer", "description": "Restrict to last N years (default: 5). Use 10-15 for foundational method papers."},
                    "reviews_only": {"type": "boolean", "description": "Return review articles only — good for overviews of a topic"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "fetch_url",
            "description": (
                "Fetch the full text of a web page. Use after web_search when snippets are not enough — "
                "e.g. to read a function's full parameter list, a method's README, or a paper abstract. "
                "Works well for readthedocs, GitHub READMEs, and static HTML pages. "
                "JavaScript-heavy sites (Notion, some dashboards) may return little content."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default: 4000; increase to 8000 for long API pages)"}
                },
                "required": ["url"]
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
            "description": "Inspect data state: shape, processing status, available embeddings, what steps are done, likely metadata columns, and tracked clustering results. Use this first to understand the data.",
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
            "name": "inspect_session",
            "description": "Inspect the unified agent session state: active dataset summary, artifacts, recent actions, unresolved decisions, and latest verification.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "include_history": {"type": "boolean", "description": "Include recent events and resolved decisions (default: true)"}
                },
                "required": []
            }
        },
        {
            "name": "list_artifacts",
            "description": "List known artifacts from the current session ledger or a saved run manifest.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_path": {"type": "string", "description": "Optional run directory or manifest.json path when inspecting a saved run."},
                    "artifact_kind": {"type": "string", "description": "Optional artifact kind filter (for example figure, report, data, log)."},
                    "limit": {"type": "integer", "description": "Maximum artifacts to return (default: 20)"}
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
        {
            "name": "review_figure",
            "description": "Attach and review an existing saved figure with the LLM. Use this when the user wants the agent to interpret QC plots, UMAPs, or other already-generated artifacts.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "figure_path": {"type": "string", "description": "Path to an existing figure image file"},
                    "question": {"type": "string", "description": "Optional prompt to guide the review of the figure"},
                    "include_image": {"type": "boolean", "description": "If true, include the image data for model review (default: true)"}
                },
                "required": ["figure_path"]
            }
        },
        {
            "name": "review_artifact",
            "description": "Review an existing artifact from the session or workspace. Supports figures, text reports, JSON outputs, logs, and AnnData files in read-only mode.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "artifact_path": {"type": "string", "description": "Absolute or run-relative path to the artifact."},
                    "artifact_id": {"type": "string", "description": "Artifact id from list_artifacts or inspect_session."},
                    "question": {"type": "string", "description": "Optional prompt to guide the review."},
                    "include_image": {"type": "boolean", "description": "If true, include image data when reviewing image artifacts (default: true)."},
                    "max_chars": {"type": "integer", "description": "Maximum text characters to return for text-like artifacts (default: 4000)."}
                },
                "required": []
            }
        },
        {
            "name": "inspect_run_state",
            "description": "Inspect a saved run manifest or the active run ledger: status, steps, artifacts, decisions, and recent world-state snapshots.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_path": {"type": "string", "description": "Run directory or manifest.json path. Optional when an active run exists."},
                    "include_history": {"type": "boolean", "description": "Include recent events and snapshots (default: true)."}
                },
                "required": []
            }
        },
        {
            "name": "inspect_workspace",
            "description": "Read-only workspace inspection for the current project or run directory. Use this sparingly for awareness and recovery.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path to inspect (default: current working directory)."},
                    "max_depth": {"type": "integer", "description": "Maximum directory depth to traverse (default: 2)."},
                    "limit": {"type": "integer", "description": "Maximum entries to return (default: 50)."}
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


def process_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    adata=None,
    world_state=None,
    run_manager=None,
) -> tuple:
    """
    Process a tool call and return structured JSON result.

    Returns
    -------
    tuple
        (json_result_string, updated_adata)
    """
    import numpy as np

    from ..core import (
        clustering_record_to_dict,
        default_cluster_key_for_method,
        get_clustering_registry,
        infer_cluster_key,
        inspect_data,
        load_data,
        metadata_candidate_to_dict,
        metadata_resolution_to_dict,
        promote_clustering_to_primary,
        rank_obs_metadata_candidates,
        recommend_next_steps,
        register_clustering,
        resolve_batch_metadata,
        run_qc_pipeline,
        normalize_data,
        run_pca,
        compute_neighbors,
        compute_umap,
        run_leiden,
        run_phenograph,
        calculate_qc_metrics,
        detect_doublets,
    )
    from ..core.normalization import select_hvg
    from ..core.clustering import run_differential_expression, get_top_markers
    from ..annotation import run_celltypist, run_scimilarity
    from ..batch import run_scanorama, run_harmony
    from ..analysis import infer_biological_context
    from .decision_policy import (
        decision_for_batch_strategy,
        decision_for_clustering_selection,
    )
    from .world_state import ArtifactRecord, StateDelta, VerificationResult

    def make_state(adata):
        """Create compact state dict."""
        state = inspect_data(adata)
        return {
            "has_raw_counts": state.has_raw_layer or state.has_raw,
            "raw_in_adata_raw": state.has_raw,
            "raw_in_layer": state.has_raw_layer,
            "raw_layer_name": state.raw_layer_name if state.has_raw_layer else None,
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

    starting_state = make_state(adata) if adata is not None else {}

    def _stage_from_state(state_dict: Dict[str, Any]) -> str:
        if not state_dict:
            return "uninitialized"
        if state_dict.get("has_celltypes"):
            return "annotated"
        if state_dict.get("has_clusters"):
            return "clustered"
        if state_dict.get("has_umap") or state_dict.get("has_neighbors"):
            return "embedded"
        if state_dict.get("is_normalized") or state_dict.get("has_hvg"):
            return "normalized"
        if state_dict.get("has_qc_metrics") or state_dict.get("has_doublets"):
            return "qc"
        if state_dict.get("has_raw_counts"):
            return "loaded"
        return "unknown"

    def _artifact_kind_from_path(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return "figure"
        if suffix in {".h5ad", ".h5", ".loom"}:
            return "data"
        if suffix in {".json"}:
            return "json"
        if suffix in {".md", ".txt", ".csv", ".tsv"}:
            return "report"
        if suffix in {".log"}:
            return "log"
        return "artifact"

    def _artifact_payload(path: str, *, role: str = "artifact", metadata: Dict[str, Any] | None = None):
        if not path:
            return None
        artifact = ArtifactRecord.from_path(
            path,
            kind=_artifact_kind_from_path(path),
            role=role,
            source_tool=tool_name,
            metadata=metadata or {},
        )
        return artifact.to_dict()

    def _build_state_delta(
        current_adata,
        *,
        summary: str,
        dataset_changed: bool,
        notes: List[str] | None = None,
    ) -> Dict[str, Any]:
        after_state = make_state(current_adata) if current_adata is not None else {}
        changed_flags = {}
        all_keys = set(starting_state.keys()) | set(after_state.keys())
        for key in sorted(all_keys):
            before = starting_state.get(key)
            after = after_state.get(key)
            if before != after:
                changed_flags[key] = {"before": before, "after": after}
        return StateDelta(
            tool=tool_name,
            summary=summary,
            dataset_changed=dataset_changed,
            stage_before=_stage_from_state(starting_state),
            stage_after=_stage_from_state(after_state),
            changed_flags=changed_flags,
            notes=notes or [],
        ).to_dict()

    def _build_verification(
        status: str,
        summary: str,
        checks: List[Dict[str, Any]],
        recovery_options: List[str] | None = None,
    ) -> Dict[str, Any]:
        if status == "passed" and any(check.get("status") == "failed" for check in checks):
            status = "warning"
        return VerificationResult(
            status=status,
            summary=summary,
            checks=checks,
            recovery_options=recovery_options or [],
        ).to_dict()

    def _check(name: str, passed: bool, details: str) -> Dict[str, Any]:
        return {"name": name, "status": "passed" if passed else "failed", "details": details}

    def _finalize_result(
        result: Dict[str, Any],
        updated_adata,
        *,
        dataset_changed: bool,
        summary: str,
        artifacts_created: List[Dict[str, Any]] | None = None,
        decisions_raised: List[Dict[str, Any]] | None = None,
        verification: Dict[str, Any] | None = None,
        notes: List[str] | None = None,
    ):
        result.setdefault(
            "state_delta",
            _build_state_delta(
                updated_adata,
                summary=summary,
                dataset_changed=dataset_changed,
                notes=notes,
            ),
        )
        result.setdefault("artifacts_created", artifacts_created or [])
        result.setdefault("decisions_raised", decisions_raised or [])
        result.setdefault(
            "verification",
            verification
            or _build_verification(
                "passed",
                f"{tool_name} completed without verification issues.",
                [],
            ),
        )
        return json.dumps(result, indent=2), updated_adata

    def _confirmed_decision_value(key: str):
        if world_state is None:
            return None
        getter = getattr(world_state, "get_confirmed_value", None)
        if getter is None:
            return None
        return getter(key)

    def _clusterings_payload(adata_obj):
        return [
            clustering_record_to_dict(record)
            for record in get_clustering_registry(adata_obj)
        ]

    def _batch_relevance(goal: Any = None, context: str = "") -> bool:
        goal_text = str(goal or "").strip().lower()
        if goal_text == "batch_correct":
            return True
        context_text = str(context or "").lower()
        return bool(
            re.search(
                r"\b(batch|integration|integrate|harmony|scanorama|correct(?:ion)?|multi[- ]sample)\b",
                context_text,
            )
        )

    def _analysis_guidance(state, *, goal: Any = None, context: str = "") -> Dict[str, Any]:
        if not state.has_qc_metrics:
            next_priority = "qc_preview"
        elif not state.is_normalized:
            next_priority = "normalize_and_hvg"
        elif not (state.has_pca and state.has_neighbors and state.has_umap):
            next_priority = "run_dimred"
        elif not state.has_clusters:
            next_priority = "run_clustering"
        else:
            next_priority = "annotation_or_deg"

        batch_relevant_now = _batch_relevance(goal, context)
        notes = [
            "For a routine first pass, keep the workflow QC-first before moving into normalization, embedding, and clustering.",
        ]
        if batch_relevant_now:
            notes.append(
                "Batch handling is relevant for this request, so confirm the partition column before correction or other batch-sensitive steps."
            )
        else:
            notes.append(
                "Do not make batch correction a front-and-center decision right now; batch/sample metadata only matters later for explicit integration workflows or per-batch operations."
            )

        return {
            "next_priority": next_priority,
            "batch_relevant_now": batch_relevant_now,
            "notes": notes,
        }

    def _available_annotation_keys(adata_obj) -> List[str]:
        return [
            column
            for column in adata_obj.obs.columns
            if any(token in column.lower() for token in ("celltyp", "scimilar", "annotation", "label"))
        ]

    def _available_plot_colors(adata_obj) -> List[str]:
        preferred = [
            "leiden",
            "sample_id",
            "batch",
            "sample",
            "pct_counts_mt",
            "total_counts",
            "n_genes_by_counts",
        ]
        available: List[str] = []
        for candidate in preferred + list(adata_obj.obs.columns):
            if candidate in adata_obj.obs.columns and candidate not in available:
                available.append(candidate)
        return available[:20]

    def _smart_unavailable_result(
        *,
        tool: str,
        message: str,
        adata_obj,
        recovery_options: List[str],
        missing_prerequisites: List[str] | None = None,
        extra: Dict[str, Any] | None = None,
    ):
        payload = {
            "status": "warning",
            "tool": tool,
            "message": message,
            "missing_prerequisites": missing_prerequisites or [],
            "recovery_options": recovery_options,
            "available_clusterings": _clusterings_payload(adata_obj) if adata_obj is not None else [],
            "available_annotation_keys": _available_annotation_keys(adata_obj) if adata_obj is not None else [],
            "available_plot_colors": _available_plot_colors(adata_obj) if adata_obj is not None else [],
            "state": make_state(adata_obj) if adata_obj is not None else {},
        }
        if extra:
            payload.update(extra)
        return _finalize_result(
            payload,
            adata_obj,
            dataset_changed=False,
            summary=message,
            verification=_build_verification(
                "warning",
                message,
                [],
                recovery_options=recovery_options,
            ),
        )

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

    def _same_resolution(left, right) -> bool:
        if left is None or right is None:
            return False
        return abs(float(left) - float(right)) < 1e-9

    def _resolve_clustering_output_key(adata_obj, method: str, resolution: float, requested_key: str | None):
        normalized_method = "phenograph" if str(method).lower() == "phenograph" else "leiden"
        alias = default_cluster_key_for_method(normalized_method)
        if requested_key:
            return requested_key, requested_key == alias

        registry = {record["key"]: record for record in _clusterings_payload(adata_obj)}
        if alias not in adata_obj.obs:
            return alias, True

        alias_record = registry.get(alias)
        if alias_record and _same_resolution(alias_record.get("resolution"), resolution):
            return alias, True
        if alias_record is None and _same_resolution(resolution, 1.0):
            return alias, True
        return infer_cluster_key(normalized_method, resolution), False

    def _apply_clustering(
        adata_obj,
        *,
        method: str,
        resolution: float,
        cluster_key: str,
        make_primary: bool,
    ):
        normalized_method = "phenograph" if str(method).lower() == "phenograph" else "leiden"
        if normalized_method == "leiden":
            run_leiden(adata_obj, resolution=resolution, key_added=cluster_key)
        else:
            run_phenograph(adata_obj, resolution=resolution, key_added=cluster_key)

        register_clustering(
            adata_obj,
            cluster_key=cluster_key,
            method=normalized_method,
            resolution=resolution,
            created_by="tool",
        )
        primary_alias = default_cluster_key_for_method(normalized_method)
        if make_primary:
            primary_alias = promote_clustering_to_primary(
                adata_obj,
                cluster_key=cluster_key,
                method=normalized_method,
                resolution=resolution,
                created_by="tool",
            )

        sizes = adata_obj.obs[cluster_key].value_counts().to_dict()
        return {
            "cluster_key": cluster_key,
            "primary_cluster_key": primary_alias,
            "method": normalized_method,
            "resolution": float(resolution),
            "n_clusters": len(sizes),
            "cluster_sizes": {str(key): int(value) for key, value in sizes.items()},
            "clusterings": _clusterings_payload(adata_obj),
        }

    def _render_figure(
        adata_obj,
        *,
        plot_type: str,
        output_path: str,
        color_by: str | None = None,
        genes=None,
        include_image: bool = True,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc

        genes = genes or []
        if plot_type == "umap":
            if "X_umap" not in adata_obj.obsm:
                raise ValueError("UMAP embedding not found. Run run_dimred first.")
            if color_by in ("", None):
                color_by = None
            elif color_by not in adata_obj.obs.columns and color_by not in adata_obj.var_names:
                raise ValueError(f"'{color_by}' is not available for UMAP coloring.")

        # For large datasets, rasterized scatter is orders of magnitude faster than
        # vector rendering (the matplotlib default).  vector_friendly=False tells
        # scanpy to rasterize scatter points — identical PNG output, seconds not minutes.
        n_cells = adata_obj.n_obs
        large_dataset = n_cells > 50_000
        if large_dataset:
            sc.settings.set_figure_params(vector_friendly=False)
            dot_size = max(1, min(5, 120_000 // n_cells))
        else:
            dot_size = None

        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == "umap":
            kwargs = dict(ax=ax, show=False)
            if dot_size is not None:
                kwargs["size"] = dot_size
            if color_by is None:
                sc.pl.umap(adata_obj, **kwargs)
            else:
                sc.pl.umap(adata_obj, color=color_by, **kwargs)
        elif plot_type == "violin":
            sc.pl.violin(adata_obj, keys=genes or [color_by], groupby=color_by, ax=ax, show=False)
        elif plot_type == "dotplot" and genes:
            sc.pl.dotplot(adata_obj, var_names=genes, groupby=color_by, show=False)
        elif plot_type == "heatmap" and genes:
            sc.pl.heatmap(adata_obj, var_names=genes, groupby=color_by, show=False)
        else:
            raise ValueError(f"Unsupported plot configuration: plot_type={plot_type}")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Restore default figure params so subsequent plots in the same session
        # are not affected.
        if large_dataset:
            sc.settings.set_figure_params(vector_friendly=True)

        result = {
            "status": "ok",
            "tool": "generate_figure",
            "output_path": output_path,
            "plot_type": plot_type,
            "color_by": color_by,
        }
        if include_image:
            result["image_base64"] = encode_image_base64(output_path)
            result["image_mime"] = get_image_mime_type(output_path)
        return result

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

            def strip_tags(s: str) -> str:
                return re.sub(r"<[^>]+>", "", s).strip()

            results = []
            articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
            for article in articles:
                title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
                pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
                if not (title_match and pmid_match):
                    continue

                # Abstract: join all AbstractText sections (handles structured abstracts)
                abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
                abstract = " ".join(strip_tags(p) for p in abstract_parts)[:1000]

                pmid = pmid_match.group(1)
                title = strip_tags(title_match.group(1))

                year_match = re.search(r"<PubDate>.*?<Year>(\d+)</Year>", article, re.DOTALL)
                year = year_match.group(1) if year_match else "N/A"

                journal_match = re.search(r"<ISOAbbreviation>(.*?)</ISOAbbreviation>", article)
                if not journal_match:
                    journal_match = re.search(r"<Title>(.*?)</Title>", article)
                journal = strip_tags(journal_match.group(1)) if journal_match else "N/A"

                # First author
                last_name = re.search(r"<LastName>(.*?)</LastName>", article)
                first_name = re.search(r"<Initials>(.*?)</Initials>", article)
                first_author = ""
                if last_name:
                    first_author = strip_tags(last_name.group(1))
                    if first_name:
                        first_author += f" {strip_tags(first_name.group(1))}"

                # DOI
                doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
                doi = strip_tags(doi_match.group(1)) if doi_match else None

                results.append({
                    "pmid": pmid,
                    "title": title,
                    "first_author": first_author,
                    "year": year,
                    "journal": journal,
                    "abstract": abstract,
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })

            return results
        except Exception:
            return []

    def fetch_url_text(url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Fetch a URL and return a compact, structured text summary."""
        import html
        import io
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

                # Strip boilerplate before extraction
                for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                for tag in soup.find_all(attrs={"role": ["navigation", "banner", "complementary"]}):
                    tag.decompose()
                for tag in soup.find_all(class_=lambda c: c and any(
                    kw in (" ".join(c) if isinstance(c, list) else c)
                    for kw in ("sidebar", "toctree", "nav", "menu", "breadcrumb", "footer", "header")
                )):
                    tag.decompose()

                page_title = clean_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
                meta = soup.find("meta", attrs={"name": "description"})
                meta_desc = clean_whitespace(meta.get("content", "")) if meta else ""

                # Prefer semantic containers (readthedocs uses .rst-content, sphinx uses .document)
                main = (
                    soup.find("main") or
                    soup.find("article") or
                    soup.find(class_=lambda c: c and any(
                        kw in (" ".join(c) if isinstance(c, list) else c)
                        for kw in ("rst-content", "document", "content", "body-content")
                    )) or
                    soup.body or soup
                )
                parts = []
                for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "pre", "code"]):
                    text_part = clean_whitespace(tag.get_text(" ", strip=True))
                    if text_part and len(text_part) > 3:
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
            "text": cleaned[:max_chars],
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
                "option_actions": tool_input.get("option_actions", []),
                "default": tool_input.get("default", ""),
                "decision_key": tool_input.get("decision_key", ""),
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
            # Note: "import os" is blocked but pathlib.Path is allowed in namespace
            forbidden = ["import os", "import sys", "subprocess", "eval(",
                        "__import__", "rm -rf", "shutil.rmtree", "requests.",
                        "os.system", "os.popen", "os.exec"]
            for f in forbidden:
                if f in code:
                    return json.dumps({
                        "status": "error",
                        "tool": "run_code",
                        "message": f"Forbidden operation: {f}. Use Path from namespace for file operations."
                    }, indent=2), adata

            # Load data if needed
            if adata is None and "data_path" in tool_input:
                adata = get_adata(tool_input, adata)

            # Helper function for safe directory creation
            from pathlib import Path as _Path
            def ensure_dir(path):
                """Create directory if it doesn't exist and return it as a Path."""
                p = _Path(path)
                p.mkdir(parents=True, exist_ok=True)
                return p

            _run_dir = _Path(tool_input.get("output_dir", "."))

            def write_report(name: str, content: str) -> str:
                """Write a markdown report to reports/name.md and return the path.

                Always use this instead of open() when saving analysis results —
                it ensures reports land in the right directory as readable .md files.

                Example:
                    write_report('cluster_summary', '## Cluster Summary\\n\\n...')
                """
                reports_dir = ensure_dir(_run_dir / "reports")
                safe_name = name.replace(" ", "_").rstrip(".md")
                path = reports_dir / f"{safe_name}.md"
                path.write_text(content)
                return str(path)

            # Execute in controlled namespace
            # Note: Path and ensure_dir are provided - no need to import os
            namespace = {
                "adata": adata,
                "sc": sc,
                "np": np,
                "pd": pd,
                "plt": plt,
                "scanpy": sc,
                "matplotlib": matplotlib,
                "output_dir": tool_input.get("output_dir", "."),
                "Path": _Path,
                "ensure_dir": ensure_dir,
                "write_report": write_report,
            }

            # Capture stdout so LLM can see print outputs
            import io
            import sys
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout

            # Capture any figures created
            plt.close('all')

            exec_error = None
            try:
                sys.stdout = stdout_capture
                exec(code, namespace)
            except Exception as _exec_err:
                exec_error = _exec_err
            finally:
                sys.stdout = old_stdout

            captured_output = stdout_capture.getvalue()

            if exec_error is not None:
                err_type = type(exec_error).__name__
                err_msg = str(exec_error)

                # Give the LLM targeted guidance based on the error type
                if err_type in ("TypeError", "AttributeError") or "unexpected keyword" in err_msg or "got an unexpected" in err_msg:
                    hint = "This looks like an API mismatch. Look up the function's documentation with web_search + fetch_url before retrying — do not guess."
                elif err_type in ("KeyError", "IndexError") or "not in" in err_msg:
                    hint = "This looks like a missing column or key. Check adata.obs.columns, adata.var.columns, or adata.obsm with inspect_data or a quick run_code before retrying."
                elif err_type in ("ModuleNotFoundError", "ImportError"):
                    hint = f"Package not installed. Use the install_package tool to request installation of the missing package."
                elif err_type == "SyntaxError":
                    hint = "Syntax error in the generated code — fix it directly."
                elif err_type == "NameError":
                    hint = "NameError — check that all variables used are defined in the namespace (adata, sc, np, pd, plt, Path, ensure_dir, write_report, output_dir)."
                else:
                    hint = "Diagnose the error before retrying: if it's an API issue look up docs; if it's a data issue inspect adata state."

                return json.dumps({
                    "status": "error",
                    "tool": "run_code",
                    "description": description,
                    "error_type": err_type,
                    "error": err_msg,
                    "output": captured_output[:500] if captured_output else None,
                    "recovery_hint": hint,
                }, indent=2), adata

            adata = namespace.get("adata", adata)
            custom_output_path = namespace.get("output_path")

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
            if custom_output_path:
                result["output_path"] = str(custom_output_path)
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
            search_result["tool"] = "web_search"
            return json.dumps(search_result, indent=2), adata

        elif tool_name == "search_papers":
            raw_query = tool_input["query"]
            max_results = tool_input.get("max_results", 5)
            recent_years = tool_input.get("recent_years", 5)
            reviews_only = tool_input.get("reviews_only", False)

            # Normalise GSEA gene set names: HALLMARK_TNFA_SIGNALING_VIA_NFKB → TNF alpha signaling NF-kB
            import re as _re
            query = _re.sub(r"^(HALLMARK|REACTOME|KEGG|GO|WP|BIOCARTA|PID|NABA)_", "", raw_query, flags=_re.IGNORECASE)
            query = query.replace("_", " ").strip()

            results = search_pubmed(
                query=query,
                max_results=max_results,
                recent_years=recent_years,
                reviews_only=reviews_only,
            )

            return json.dumps({
                "status": "ok",
                "tool": "search_papers",
                "query": query,
                "original_query": raw_query if raw_query != query else None,
                "reviews_only": reviews_only,
                "years_searched": f"last {recent_years} years",
                "count": len(results),
                "results": results,
            }, indent=2), adata

        elif tool_name == "fetch_url":
            url = tool_input["url"]
            max_chars = tool_input.get("max_chars", 4000)
            fetched = fetch_url_text(url, max_chars=max_chars)
            fetched["tool"] = "fetch_url"
            return json.dumps(fetched, indent=2), adata

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
            # update_memory=True if loading from path (so data persists), False if just inspecting existing
            should_update = tool_input.get("data_path") is not None and adata is None
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=should_update)
            state = inspect_data(working_adata)
            batch_resolution = resolve_batch_metadata(working_adata)
            goal = tool_input.get("goal")
            context_hint = tool_input.get("context", "")
            guidance_context = context_hint
            if tool_input.get("data_path"):
                context_hint = " ".join(part for part in [context_hint, str(tool_input.get("data_path"))] if part)
            biological_context = infer_biological_context(working_adata, text_context=context_hint)
            confirmed_batch_key = _confirmed_decision_value("batch_key")
            guidance = _analysis_guidance(state, goal=goal, context=guidance_context)

            raw_info = {"adata_raw": None, "layers": []}
            if state.has_raw:
                raw_info["adata_raw"] = {
                    "n_vars": state.raw_n_vars,
                    "note": "full gene set before HVG subsetting" if state.raw_n_vars > state.n_genes else "same gene set as X",
                }
            if state.has_raw_layer:
                raw_info["layers"].append(state.raw_layer_name)

            result = {
                "status": "ok",
                "tool": "inspect_data",
                "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
                "data_type": state.data_type,
                "raw": raw_info,
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
                    "n_clusters": state.n_clusters,
                    "available_clusterings": _clusterings_payload(working_adata),
                },
                "batch": {
                    "confirmed_batch_key": confirmed_batch_key,
                    "inferred_batch_key": state.batch_key,
                    "n_batches": state.n_batches,
                    "status": batch_resolution.status,
                    "recommended_batch_key": batch_resolution.recommended_column,
                    "recommended_role": batch_resolution.recommended_role,
                    "needs_confirmation": batch_resolution.needs_user_confirmation,
                    "reason": batch_resolution.reason,
                    "relevance": "current" if guidance["batch_relevant_now"] else "later_optional",
                    "candidates": [
                        metadata_candidate_to_dict(candidate)
                        for candidate in state.metadata_candidates
                    ],
                },
                "metadata_candidates": [
                    metadata_candidate_to_dict(candidate)
                    for candidate in state.metadata_candidates
                ],
                "batch_key": confirmed_batch_key,
                "recommended_batch_key": batch_resolution.recommended_column,
                "available_clusterings": _clusterings_payload(working_adata),
                "biological_context": biological_context.to_dict(),
                "analysis_guidance": guidance,
            }
            if goal:
                result["recommended_steps"] = recommend_next_steps(state, goal)
            decisions = []
            batch_decision = decision_for_batch_strategy(
                metadata_resolution_to_dict(batch_resolution),
                context="inspect_data",
                source_tool="inspect_data",
                batch_relevant=guidance["batch_relevant_now"],
            )
            if batch_decision is not None:
                decisions.append(batch_decision)

            return _finalize_result(
                result,
                updated_adata,
                dataset_changed=False,
                summary="Inspected the active AnnData state and collaborative metadata candidates.",
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "inspect_data returned a coherent dataset summary.",
                    [
                        _check("shape_available", "shape" in result, "Dataset shape is present."),
                        _check(
                            "batch_section_present",
                            "batch" in result,
                            "Batch metadata summary is present.",
                        ),
                    ],
                ),
            )

        elif tool_name == "inspect_session":
            include_history = bool(tool_input.get("include_history", True))
            if world_state is None:
                session_payload = {
                    "status": "ok",
                    "tool": "inspect_session",
                    "message": "No active AgentWorldState was provided. Falling back to the current AnnData state only.",
                    "world_state": {
                        "analysis_stage": _stage_from_state(starting_state),
                        "data_summary": {"state": starting_state},
                    },
                }
                return _finalize_result(
                    session_payload,
                    adata,
                    dataset_changed=False,
                    summary="Inspected a minimal session fallback because no AgentWorldState was available.",
                )

            snapshot = world_state.snapshot()
            if not include_history:
                snapshot.pop("resolved_decisions", None)
                snapshot.pop("artifacts", None)

            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "inspect_session",
                    "world_state": snapshot,
                },
                adata,
                dataset_changed=False,
                summary="Inspected the unified agent session state.",
                verification=_build_verification(
                    "passed",
                    "Session state was available for inspection.",
                    [
                        _check(
                            "analysis_stage_present",
                            bool(snapshot.get("analysis_stage")),
                            "Session snapshot includes an analysis stage.",
                        ),
                    ],
                ),
            )

        elif tool_name == "list_artifacts":
            limit = int(tool_input.get("limit", 20))
            artifact_kind = tool_input.get("artifact_kind")
            artifacts: List[Dict[str, Any]] = []

            if world_state is not None:
                artifacts = [artifact.to_dict() for artifact in world_state.artifacts]
            elif run_manager is not None:
                artifacts = list(run_manager.manifest.artifact_registry)
            else:
                run_path = tool_input.get("run_path")
                if run_path:
                    manifest_path = Path(run_path)
                    if manifest_path.is_dir():
                        manifest_path = manifest_path / "manifest.json"
                    if not manifest_path.exists():
                        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
                    with open(manifest_path) as handle:
                        manifest_payload = json.load(handle)
                    artifacts = manifest_payload.get("artifact_registry", [])

            if artifact_kind:
                artifacts = [artifact for artifact in artifacts if artifact.get("kind") == artifact_kind]

            artifacts = artifacts[-limit:]
            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "list_artifacts",
                    "artifacts": artifacts,
                    "n_artifacts": len(artifacts),
                },
                adata,
                dataset_changed=False,
                summary="Listed known session or run artifacts.",
            )

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

            if "rank_genes_groups" not in working_adata.uns:
                return _smart_unavailable_result(
                    tool="get_top_markers",
                    message="Top markers are not available because differential expression has not been run yet.",
                    adata_obj=updated_adata,
                    missing_prerequisites=["deg"],
                    recovery_options=[
                        "Run differential expression on the current clustering first.",
                        "Inspect available clusterings before choosing a DEG grouping.",
                    ],
                    extra={"cluster": cluster},
                )

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
                return _smart_unavailable_result(
                    tool="get_celltypes",
                    message="No cell type annotations are available on the current in-memory dataset.",
                    adata_obj=updated_adata,
                    missing_prerequisites=["annotation"],
                    recovery_options=[
                        "Run cell type annotation on the current clustering.",
                        "Inspect available clustering keys before annotating.",
                    ],
                )

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
                "n_columns": len(working_adata.obs.columns),
                "metadata_candidates": [
                    metadata_candidate_to_dict(candidate)
                    for candidate in rank_obs_metadata_candidates(working_adata)
                ],
            }, indent=2), updated_adata

        elif tool_name in {"review_figure", "review_artifact"}:
            artifact_path = tool_input.get("artifact_path") or tool_input.get("figure_path")
            artifact_id = tool_input.get("artifact_id")
            include_image = bool(tool_input.get("include_image", True))
            question = tool_input.get("question", "")
            max_chars = int(tool_input.get("max_chars", 4000))

            if not artifact_path and artifact_id and world_state is not None:
                matched = next(
                    (artifact for artifact in world_state.artifacts if artifact.artifact_id == artifact_id),
                    None,
                )
                if matched is not None:
                    artifact_path = matched.path

            if not artifact_path:
                raise ValueError("Provide artifact_path/figure_path or artifact_id to review an artifact.")

            artifact_path = os.path.abspath(artifact_path)
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(f"Artifact not found: {artifact_path}")

            artifact_kind = _artifact_kind_from_path(artifact_path)
            result = {
                "status": "ok",
                "tool": tool_name,
                "artifact_path": artifact_path,
                "artifact_kind": artifact_kind,
                "question": question,
                "size_bytes": os.path.getsize(artifact_path),
            }
            if tool_name == "review_figure":
                result["figure_path"] = artifact_path

            if artifact_kind == "figure":
                if include_image:
                    result["image_base64"] = encode_image_base64(artifact_path)
                    result["image_mime"] = get_image_mime_type(artifact_path)
            elif artifact_kind == "json":
                with open(artifact_path) as handle:
                    payload = json.load(handle)
                excerpt = json.dumps(payload, indent=2)[:max_chars]
                result["content_excerpt"] = excerpt
                result["json_keys"] = list(payload.keys())[:25] if isinstance(payload, dict) else []
            elif artifact_kind in {"report", "log"}:
                with open(artifact_path) as handle:
                    content = handle.read()
                result["content_excerpt"] = content[:max_chars]
                result["truncated"] = len(content) > max_chars
            elif artifact_kind == "data":
                reviewed_adata = load_data(artifact_path)
                reviewed_state = inspect_data(reviewed_adata)
                result["data_summary"] = {
                    "shape": {"n_cells": reviewed_state.n_cells, "n_genes": reviewed_state.n_genes},
                    "data_type": reviewed_state.data_type,
                    "batch_key": reviewed_state.batch_key,
                    "cluster_key": reviewed_state.cluster_key,
                    "n_clusters": reviewed_state.n_clusters,
                }

            verification_checks = [
                _check("artifact_exists", os.path.exists(artifact_path), f"Artifact exists at {artifact_path}."),
            ]
            if artifact_kind == "figure" and include_image:
                verification_checks.append(
                    _check(
                        "image_payload_attached",
                        "image_base64" in result,
                        "Image payload attached for model review.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary=f"Reviewed existing {artifact_kind} artifact.",
                verification=_build_verification(
                    "passed",
                    f"{artifact_kind.title()} artifact was available for review.",
                    verification_checks,
                ),
            )

        elif tool_name == "inspect_run_state":
            include_history = bool(tool_input.get("include_history", True))
            manifest_payload = None
            if run_manager is not None:
                manifest_payload = run_manager.manifest.to_dict()
            else:
                run_path = tool_input.get("run_path")
                if not run_path:
                    raise ValueError("Provide run_path when no active run manager is available.")
                manifest_path = Path(run_path)
                if manifest_path.is_dir():
                    manifest_path = manifest_path / "manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
                with open(manifest_path) as handle:
                    manifest_payload = json.load(handle)

            result = {
                "status": "ok",
                "tool": "inspect_run_state",
                "run_id": manifest_payload.get("run_id"),
                "run_status": manifest_payload.get("status"),
                "request": manifest_payload.get("request"),
                "n_steps": len(manifest_payload.get("steps_completed", [])),
                "n_artifacts": len(manifest_payload.get("artifact_registry", [])),
                "n_decisions": len(manifest_payload.get("user_decisions", [])),
                "n_verifications": len(manifest_payload.get("verification_history", [])),
                "latest_world_state": (manifest_payload.get("world_state_snapshots", []) or [{}])[-1],
            }
            if include_history:
                result["recent_events"] = manifest_payload.get("session_events", [])[-10:]
                result["recent_steps"] = manifest_payload.get("steps_completed", [])[-10:]

            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary="Inspected the active or persisted run ledger.",
            )

        elif tool_name == "inspect_workspace":
            workspace_root = Path.cwd().resolve()
            allowed_roots = [workspace_root]
            if run_manager is not None:
                allowed_roots.append(Path(run_manager.run_dir).resolve())

            requested = tool_input.get("path", ".")
            requested_path = Path(requested)
            if not requested_path.is_absolute():
                requested_path = (workspace_root / requested_path).resolve()
            else:
                requested_path = requested_path.resolve()

            if not any(
                requested_path == root or root in requested_path.parents
                for root in allowed_roots
            ):
                raise ValueError(
                    f"inspect_workspace is limited to the project workspace and active run directory. "
                    f"Requested path: {requested_path}"
                )

            max_depth = int(tool_input.get("max_depth", 2))
            limit = int(tool_input.get("limit", 50))
            entries: List[Dict[str, Any]] = []

            def _walk(path_obj: Path, depth: int) -> None:
                if len(entries) >= limit:
                    return
                if path_obj.is_file():
                    entries.append(
                        {
                            "path": str(path_obj),
                            "kind": "file",
                            "size_bytes": path_obj.stat().st_size,
                        }
                    )
                    return
                if not path_obj.is_dir() or depth > max_depth:
                    return

                for child in sorted(path_obj.iterdir(), key=lambda candidate: candidate.name):
                    if len(entries) >= limit:
                        break
                    if child.name == "__pycache__":
                        continue
                    entries.append(
                        {
                            "path": str(child),
                            "kind": "directory" if child.is_dir() else "file",
                            "size_bytes": child.stat().st_size if child.is_file() else None,
                            "depth": depth,
                        }
                    )
                    if child.is_dir():
                        _walk(child, depth + 1)

            _walk(requested_path, 0)
            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "inspect_workspace",
                    "path": str(requested_path),
                    "entries": entries,
                    "n_entries": len(entries),
                },
                adata,
                dataset_changed=False,
                summary="Inspected the workspace in read-only mode.",
            )

        # ===== ACTION TOOLS =====
        elif tool_name == "run_qc":
            warnings = []
            if adata is not None and tool_input.get("data_path") not in (None, "memory"):
                warnings.append(
                    "Ignored data_path and continued with the in-memory dataset to preserve prior analysis state."
                )

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_before, g_before = adata.n_obs, adata.n_vars

            detect_doublets_flag = tool_input.get("detect_doublets_flag", True)
            remove_ribo = tool_input.get("remove_ribo", True)
            remove_mt = tool_input.get("remove_mt", False)
            min_cells = tool_input.get("min_cells")
            requested_mt_threshold = tool_input.get("mt_threshold")
            preview_only = bool(tool_input.get("preview_only", False))
            requested_batch_key = tool_input.get("batch_key") or _confirmed_decision_value("batch_key")
            if requested_batch_key and "batch_key" not in tool_input and _confirmed_decision_value("batch_key"):
                warnings.append(
                    f"Using previously confirmed batch_key '{requested_batch_key}' from session state."
                )

            if detect_doublets_flag:
                batch_resolution = resolve_batch_metadata(
                    adata,
                    requested_column=requested_batch_key,
                )
            else:
                batch_resolution = None

            batch_key = batch_resolution.applied_column if batch_resolution else None
            if batch_resolution:
                if batch_resolution.status == "auto_selected" and batch_key:
                    warnings.append(batch_resolution.reason)
                elif batch_resolution.status == "invalid_requested":
                    warnings.append(batch_resolution.reason)
                elif batch_resolution.status == "needs_confirmation":
                    warnings.append(
                        f"{batch_resolution.reason} "
                        f"{'Previewing' if preview_only else 'Running'} doublets without per-batch stratification for now."
                    )
                elif batch_resolution.status == "no_candidate":
                    warnings.append(batch_resolution.reason)

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
                import scanpy as sc
                import matplotlib.pyplot as plt
                os.makedirs(figure_dir, exist_ok=True)
                try:
                    qc_plot_adata = qc_preview.copy()

                    # Add log-transformed columns for visualization
                    if "total_counts" in qc_plot_adata.obs.columns:
                        qc_plot_adata.obs["log10_total_counts"] = np.log10(qc_plot_adata.obs["total_counts"] + 1)
                    if "n_genes_by_counts" in qc_plot_adata.obs.columns:
                        qc_plot_adata.obs["log10_n_genes"] = np.log10(qc_plot_adata.obs["n_genes_by_counts"] + 1)

                    # --- Figure 1: Main QC violin plots (log counts, log genes, MT%, ribo%) ---
                    main_violin_keys = []
                    if "log10_total_counts" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("log10_total_counts")
                    if "log10_n_genes" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("log10_n_genes")
                    if "pct_counts_mt" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("pct_counts_mt")
                    if "pct_counts_ribo" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("pct_counts_ribo")

                    if main_violin_keys:
                        sc.pl.violin(qc_plot_adata, main_violin_keys, jitter=0.2, multi_panel=True, show=False)
                        plt.suptitle("QC Metrics Distribution", y=1.02)
                        violin_path = os.path.join(figure_dir, "qc_violin_metrics.png")
                        plt.savefig(violin_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        figure_outputs.append(violin_path)

                    # --- Figure 2: Doublet scores (if available) ---
                    if "doublet_score" in qc_plot_adata.obs.columns:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        # Violin of doublet scores
                        sc.pl.violin(qc_plot_adata, "doublet_score", jitter=0.2, ax=axes[0], show=False)
                        axes[0].set_title("Doublet Score Distribution")
                        # Histogram with threshold
                        axes[1].hist(qc_plot_adata.obs["doublet_score"], bins=50, edgecolor='black', alpha=0.7)
                        axes[1].axvline(0.25, color='red', linestyle='--', label='Typical threshold (0.25)')
                        axes[1].set_xlabel("Doublet Score")
                        axes[1].set_ylabel("Cell Count")
                        axes[1].set_title("Doublet Score Histogram")
                        axes[1].legend()
                        fig.tight_layout()
                        doublet_path = os.path.join(figure_dir, "qc_doublet_scores.png")
                        fig.savefig(doublet_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(doublet_path)

                    # --- Figure 3: Histograms for counts and genes (log scale) ---
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    if "total_counts" in qc_plot_adata.obs.columns:
                        axes[0].hist(qc_plot_adata.obs["total_counts"], bins=100, edgecolor='black', alpha=0.7)
                        axes[0].set_xscale('log')
                        axes[0].set_xlabel("Total Counts (log scale)")
                        axes[0].set_ylabel("Cell Count")
                        axes[0].set_title("Library Size Distribution")
                    if "n_genes_by_counts" in qc_plot_adata.obs.columns:
                        axes[1].hist(qc_plot_adata.obs["n_genes_by_counts"], bins=100, edgecolor='black', alpha=0.7)
                        axes[1].set_xscale('log')
                        axes[1].set_xlabel("Genes Detected (log scale)")
                        axes[1].set_ylabel("Cell Count")
                        axes[1].set_title("Genes per Cell Distribution")
                    fig.tight_layout()
                    hist_path = os.path.join(figure_dir, "qc_histograms.png")
                    fig.savefig(hist_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    figure_outputs.append(hist_path)

                    # --- Figure 4: MT% histogram with threshold line ---
                    if "pct_counts_mt" in qc_plot_adata.obs.columns:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(qc_plot_adata.obs["pct_counts_mt"], bins=100, edgecolor='black', alpha=0.7)
                        ax.axvline(mt_threshold, color='red', linestyle='--', linewidth=2,
                                   label=f'Threshold: {mt_threshold:.1f}%')
                        ax.set_xlabel("Mitochondrial %")
                        ax.set_ylabel("Cell Count")
                        ax.set_title("Mitochondrial Content Distribution")
                        ax.legend()
                        fig.tight_layout()
                        mt_hist_path = os.path.join(figure_dir, "qc_mt_histogram.png")
                        fig.savefig(mt_hist_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(mt_hist_path)

                    # --- Figure 5: Scatter plots (genes vs counts, MT vs counts) ---
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    if 'n_genes_by_counts' in qc_plot_adata.obs.columns and 'total_counts' in qc_plot_adata.obs.columns:
                        sc.pl.scatter(qc_plot_adata, x='total_counts', y='n_genes_by_counts', ax=axes[0], show=False)
                        axes[0].set_xscale('log')
                        axes[0].set_yscale('log')
                        axes[0].set_title("Genes vs Counts (log-log)")
                    if 'pct_counts_mt' in qc_plot_adata.obs.columns and 'total_counts' in qc_plot_adata.obs.columns:
                        sc.pl.scatter(qc_plot_adata, x='total_counts', y='pct_counts_mt', ax=axes[1], show=False)
                        axes[1].set_xscale('log')
                        axes[1].axhline(mt_threshold, color='red', linestyle='--', linewidth=2,
                                        label=f'MT threshold: {mt_threshold:.1f}%')
                        axes[1].set_title("MT% vs Counts")
                        axes[1].legend()
                    fig.tight_layout()
                    scatter_path = os.path.join(figure_dir, "qc_scatter.png")
                    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    figure_outputs.append(scatter_path)

                    # --- Figure 6: Ribo vs MT scatter (if both available) ---
                    if 'pct_counts_mt' in qc_plot_adata.obs.columns and 'pct_counts_ribo' in qc_plot_adata.obs.columns:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sc.pl.scatter(qc_plot_adata, x='pct_counts_mt', y='pct_counts_ribo', ax=ax, show=False)
                        ax.axvline(mt_threshold, color='red', linestyle='--', linewidth=1, label=f'MT threshold')
                        ax.set_title("Ribosomal vs Mitochondrial Content")
                        ax.legend()
                        fig.tight_layout()
                        ribo_mt_path = os.path.join(figure_dir, "qc_ribo_vs_mt.png")
                        fig.savefig(ribo_mt_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(ribo_mt_path)

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
            if batch_resolution and batch_resolution.needs_user_confirmation and tool_input.get("batch_key"):
                recommendation += (
                    f" I could not confirm the requested per-batch column automatically; "
                    f"'{batch_resolution.recommended_column}' looks closest."
                )

            batch_strategy = (
                metadata_resolution_to_dict(batch_resolution)
                if batch_resolution
                else {
                    "status": "not_applicable",
                    "requested_column": requested_batch_key,
                    "applied_column": None,
                    "recommended_column": None,
                    "recommended_role": None,
                    "needs_user_confirmation": False,
                    "reason": "Doublet detection is disabled for this QC run.",
                    "candidates": [],
                }
            )
            batch_strategy["used_for_doublets"] = batch_key
            decisions = []
            batch_decision = decision_for_batch_strategy(
                batch_strategy,
                context="doublet_detection",
                source_tool="run_qc",
                batch_relevant=bool(tool_input.get("batch_key")),
            )
            if batch_decision is not None:
                decisions.append(batch_decision)
            artifact_payloads = [
                artifact
                for artifact in (
                    _artifact_payload(path, role="qc_figure", metadata={"mode": "preview" if preview_only else "applied"})
                    for path in figure_outputs
                )
                if artifact is not None
            ]

            if preview_only:
                preview_result = {
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
                    "batch_strategy": batch_strategy,
                    "state": make_state(adata)
                }
                verification_checks = [
                    _check("qc_metrics_computed", "pct_counts_mt" in qc_preview.obs.columns, "QC metrics were computed on the preview copy."),
                    _check(
                        "batch_strategy_reported",
                        bool(batch_strategy.get("status")),
                        "Batch strategy is attached to the QC preview.",
                    ),
                ]
                if figure_outputs:
                    verification_checks.append(
                        _check(
                            "preview_figures_exist",
                            all(os.path.exists(path) for path in figure_outputs),
                            "QC preview figures were written to disk.",
                        )
                    )
                return _finalize_result(
                    preview_result,
                    adata,
                    dataset_changed=False,
                    summary="Previewed QC metrics, thresholds, and doublet strategy without filtering cells.",
                    artifacts_created=artifact_payloads,
                    decisions_raised=decisions,
                    verification=_build_verification(
                        "passed",
                        "QC preview completed and reported a concrete batch strategy.",
                        verification_checks,
                    ),
                )

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
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifact_payloads.append(artifact)

            qc_result = {
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
                "batch_strategy": batch_strategy,
                "state": make_state(adata)
            }
            verification_checks = [
                _check("qc_metrics_present", "pct_counts_mt" in adata.obs.columns, "MT QC metric is present after QC."),
                _check(
                    "batch_key_valid",
                    batch_key is None or batch_key in adata.obs.columns,
                    f"Batch key '{batch_key}' is available on the filtered AnnData." if batch_key else "QC ran without per-batch stratification.",
                ),
            ]
            if detect_doublets_flag:
                verification_checks.append(
                    _check(
                        "doublet_scores_present",
                        "predicted_doublet" in adata.obs.columns,
                        "Doublet scores/predictions are available after QC.",
                    )
                )
            return _finalize_result(
                qc_result,
                adata,
                dataset_changed=True,
                summary="Applied QC filtering and recorded the batch-aware doublet strategy.",
                artifacts_created=artifact_payloads,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "QC completed and the resulting AnnData passed post-action checks.",
                    verification_checks,
                ),
            )

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
            resolution = float(tool_input.get("resolution", 1.0))
            requested_cluster_key = tool_input.get("cluster_key")
            cluster_key, default_make_primary = _resolve_clustering_output_key(
                adata,
                method,
                resolution,
                requested_cluster_key,
            )
            make_primary = tool_input.get("make_primary")
            if make_primary is None:
                make_primary = default_make_primary
            if requested_cluster_key and requested_cluster_key == default_cluster_key_for_method(method) and not make_primary:
                warnings.append(
                    f"cluster_key '{requested_cluster_key}' is the primary alias for {method}; forcing make_primary=true."
                )
                make_primary = True

            result_payload = _apply_clustering(
                adata,
                method=method,
                resolution=resolution,
                cluster_key=cluster_key,
                make_primary=bool(make_primary),
            )

            output_path = fix_output_path(tool_input.get("output_path"), "run_clustering")
            if output_path:
                write_h5ad_safe(adata, output_path)
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)

            clustering_result = {
                "status": "ok",
                "tool": "run_clustering",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": result_payload["method"],
                "resolution": result_payload["resolution"],
                "cluster_key": result_payload["cluster_key"],
                "primary_cluster_key": result_payload["primary_cluster_key"],
                "make_primary": bool(make_primary),
                "n_clusters": result_payload["n_clusters"],
                "cluster_sizes": result_payload["cluster_sizes"],
                "available_clusterings": result_payload["clusterings"],
                "warnings": warnings,
                "state": make_state(adata)
            }
            primary_key = result_payload["primary_cluster_key"]
            verification_checks = [
                _check(
                    "cluster_key_created",
                    result_payload["cluster_key"] in adata.obs.columns,
                    f"Clustering column '{result_payload['cluster_key']}' exists in adata.obs.",
                ),
                _check(
                    "cluster_count_matches",
                    adata.obs[result_payload["cluster_key"]].nunique() == result_payload["n_clusters"],
                    "Reported cluster count matches the stored clustering column.",
                ),
                _check(
                    "primary_alias_available",
                    primary_key in adata.obs.columns,
                    f"Primary clustering alias '{primary_key}' is available.",
                ),
            ]
            return _finalize_result(
                clustering_result,
                adata,
                dataset_changed=True,
                summary=(
                    f"Ran {result_payload['method']} clustering at resolution {result_payload['resolution']} "
                    f"and stored results in '{result_payload['cluster_key']}'."
                ),
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "Clustering results were created and verified against the AnnData state.",
                    verification_checks,
                ),
            )

        elif tool_name == "compare_clusterings":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "leiden")
            resolutions = [float(value) for value in tool_input.get("resolutions", [])]
            if not resolutions:
                raise ValueError("compare_clusterings requires at least one resolution.")

            compare_results = []
            figure_dir = tool_input.get("figure_dir")
            generate_figures = bool(tool_input.get("generate_figures", False))
            include_images = bool(tool_input.get("include_images", False))
            promote_resolution = tool_input.get("promote_resolution")
            image_payloads = []

            if figure_dir:
                os.makedirs(figure_dir, exist_ok=True)

            for resolution in resolutions:
                cluster_key = infer_cluster_key(method, resolution)
                result_payload = _apply_clustering(
                    adata,
                    method=method,
                    resolution=resolution,
                    cluster_key=cluster_key,
                    make_primary=False,
                )
                compare_entry = {
                    "resolution": result_payload["resolution"],
                    "cluster_key": result_payload["cluster_key"],
                    "n_clusters": result_payload["n_clusters"],
                    "cluster_sizes": result_payload["cluster_sizes"],
                }

                if generate_figures and "X_umap" in adata.obsm:
                    path_root = figure_dir or "."
                    figure_name = f"umap_{cluster_key}.png"
                    figure_path = os.path.join(path_root, figure_name)
                    figure_result = _render_figure(
                        adata,
                        plot_type="umap",
                        output_path=figure_path,
                        color_by=cluster_key,
                        include_image=include_images,
                    )
                    compare_entry["figure_path"] = figure_path
                    if include_images and "image_base64" in figure_result:
                        image_payloads.append({
                            "cluster_key": cluster_key,
                            "output_path": figure_path,
                            "image_base64": figure_result["image_base64"],
                            "image_mime": figure_result["image_mime"],
                        })

                compare_results.append(compare_entry)

            if promote_resolution is not None:
                promote_key = infer_cluster_key(method, float(promote_resolution))
                if promote_key not in adata.obs.columns:
                    raise ValueError(
                        f"Cannot promote resolution {promote_resolution}; "
                        f"expected clustering key '{promote_key}' was not generated."
                    )
                promote_clustering_to_primary(
                    adata,
                    cluster_key=promote_key,
                    method=method,
                    resolution=float(promote_resolution),
                    created_by="tool",
                )

            result = {
                "status": "ok",
                "tool": "compare_clusterings",
                "method": "phenograph" if str(method).lower() == "phenograph" else "leiden",
                "comparisons": compare_results,
                "available_clusterings": _clusterings_payload(adata),
                "warnings": warnings,
                "state": make_state(adata),
            }
            if image_payloads:
                first = image_payloads[0]
                result["image_base64"] = first["image_base64"]
                result["image_mime"] = first["image_mime"]
                result["image_context"] = {
                    "cluster_key": first["cluster_key"],
                    "output_path": first["output_path"],
                }
            artifacts_created = [
                artifact
                for artifact in (
                    _artifact_payload(
                        comparison.get("figure_path"),
                        role="comparison_figure",
                        metadata={"cluster_key": comparison.get("cluster_key")},
                    )
                    for comparison in compare_results
                    if comparison.get("figure_path")
                )
                if artifact is not None
            ]
            decisions = []
            if promote_resolution is None:
                clustering_decision = decision_for_clustering_selection(
                    compare_results,
                    source_tool="compare_clusterings",
                )
                if clustering_decision is not None:
                    decisions.append(clustering_decision)
            verification_checks = [
                _check(
                    "comparison_keys_created",
                    all(comparison["cluster_key"] in adata.obs.columns for comparison in compare_results),
                    "Every compared clustering key exists in adata.obs.",
                ),
            ]
            if artifacts_created:
                verification_checks.append(
                    _check(
                        "comparison_figures_exist",
                        all(os.path.exists(artifact["path"]) for artifact in artifacts_created),
                        "Generated comparison figures exist on disk.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary="Generated a safe multi-resolution clustering comparison without overwriting prior results.",
                artifacts_created=artifacts_created,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "All requested clustering comparisons were preserved and verified.",
                    verification_checks,
                ),
            )

        elif tool_name == "run_celltypist":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            model = tool_input.get("model", "Immune_All_Low.pkl")
            majority = tool_input.get("majority_voting", True)
            cluster_key = tool_input.get("cluster_key", "leiden")
            if majority and cluster_key not in adata.obs.columns:
                return _smart_unavailable_result(
                    tool="run_celltypist",
                    message=(
                        f"CellTypist majority voting needs a valid clustering column, but '{cluster_key}' "
                        "is not available on the current in-memory dataset."
                    ),
                    adata_obj=adata,
                    missing_prerequisites=["clustering"],
                    recovery_options=[
                        "Run clustering first, then rerun CellTypist.",
                        "Choose one of the available clustering keys for annotation.",
                    ],
                    extra={"requested_cluster_key": cluster_key},
                )
            if majority:
                cluster_key = _validate_obs_column(
                    adata,
                    cluster_key,
                    warnings,
                    required=True,
                    context="cluster_key",
                )

            run_celltypist(
                adata,
                model=model,
                majority_voting=majority,
                over_clustering=cluster_key if majority else None,
            )

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
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            celltypist_result = {
                "status": "ok",
                "tool": "run_celltypist",
                "output_path": output_path,
                "saved": output_path is not None,
                "model": model,
                "majority_voting": majority,
                "cluster_key_used": cluster_key if majority else None,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "cell_type_breakdown": type_breakdown,
                "warnings": warnings,
                "state": make_state(adata)
            }
            return _finalize_result(
                celltypist_result,
                adata,
                dataset_changed=True,
                summary="Ran CellTypist annotation and recorded the annotation source in session state.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "CellTypist annotations were written to AnnData.",
                    [
                        _check("annotation_key_present", key in adata.obs.columns, f"Annotation column '{key}' exists."),
                        _check(
                            "cluster_key_valid",
                            not majority or cluster_key in adata.obs.columns,
                            f"Cluster key '{cluster_key}' is valid for majority voting." if majority else "Majority voting was disabled.",
                        ),
                    ],
                ),
            )

        elif tool_name == "run_scimilarity":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            model_path = tool_input.get("model_path")
            cluster_key = tool_input.get("cluster_key", "leiden")
            cluster_key = _validate_obs_column(
                adata,
                cluster_key,
                warnings,
                required=False,
                context="cluster_key",
            ) or "leiden"

            # Only pass model_path if specified, otherwise use default
            if model_path:
                run_scimilarity(adata, model_path=model_path, cluster_key=cluster_key)
            else:
                run_scimilarity(adata, cluster_key=cluster_key)

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
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            scimilarity_result = {
                "status": "ok",
                "tool": "run_scimilarity",
                "output_path": output_path,
                "saved": output_path is not None,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "cluster_key_used": cluster_key,
                "has_embeddings": "X_scimilarity" in adata.obsm,
                "cell_type_breakdown": type_breakdown,
                "warnings": warnings,
                "state": make_state(adata)
            }
            return _finalize_result(
                scimilarity_result,
                adata,
                dataset_changed=True,
                summary="Ran Scimilarity annotation and recorded the representative predictions.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "Scimilarity outputs were written to AnnData.",
                    [
                        _check("annotation_key_present", key in adata.obs.columns, f"Annotation column '{key}' exists."),
                        _check(
                            "cluster_key_valid",
                            cluster_key in adata.obs.columns,
                            f"Cluster key '{cluster_key}' is available for cluster-level summaries.",
                        ),
                    ],
                ),
            )

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
            artifact = _artifact_payload(output_path, role="saved_dataset", metadata={"save_mode": save_details.get("save_mode", "direct")})
            save_result = {
                "status": "ok",
                "tool": "save_data",
                "output_path": output_path,
                "saved": True,
                "save_mode": save_details.get("save_mode", "direct"),
                "warnings": save_details.get("warnings", []),
                "shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "state": make_state(adata)
            }
            return _finalize_result(
                save_result,
                adata,
                dataset_changed=False,
                summary="Saved the current in-memory AnnData as a final dataset artifact.",
                artifacts_created=[artifact] if artifact is not None else [],
                verification=_build_verification(
                    "passed",
                    "The AnnData output file exists and was registered as an artifact.",
                    [
                        _check("output_exists", os.path.exists(output_path), f"Saved output exists at {output_path}."),
                    ],
                ),
            )

        elif tool_name == "run_batch_correction":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "harmony")
            requested_batch_key = tool_input.get("batch_key") or _confirmed_decision_value("batch_key")
            if not requested_batch_key:
                raise ValueError(
                    "No batch_key was provided and no confirmed batch_key exists in session state. "
                    "Confirm the correct batch column first."
                )
            batch_key = _validate_obs_column(adata, requested_batch_key, warnings, required=True, context="batch_key")

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
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            batch_strategy = {
                "status": "user_selected" if tool_input.get("batch_key") else "auto_selected",
                "requested_column": tool_input.get("batch_key"),
                "applied_column": batch_key,
                "recommended_column": batch_key,
                "recommended_role": "batch",
                "needs_user_confirmation": False,
                "reason": f"Using '{batch_key}' for batch correction.",
                "candidates": [{"column": batch_key}],
            }
            decisions = []
            batch_decision = decision_for_batch_strategy(
                batch_strategy,
                context="batch_correction",
                source_tool="run_batch_correction",
            )
            if batch_decision is not None:
                decisions.append(batch_decision)
            batch_result = {
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
            }
            return _finalize_result(
                batch_result,
                adata,
                dataset_changed=True,
                summary=f"Applied {method} batch correction using '{batch_key}' and recomputed the neighborhood graph.",
                artifacts_created=artifacts_created,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "Batch correction completed and the corrected embedding is available.",
                    [
                        _check("batch_key_present", batch_key in adata.obs.columns, f"Batch key '{batch_key}' exists in adata.obs."),
                        _check("corrected_embedding_present", corrected_rep in adata.obsm, f"Corrected embedding '{corrected_rep}' exists in adata.obsm."),
                        _check("umap_present", "X_umap" in adata.obsm, "UMAP was recomputed after correction."),
                    ],
                ),
            )

        elif tool_name == "run_deg":
            from ..analysis.deg import run_validated_deg, get_deg_caveats
            from ..config.defaults import DEG_DEFAULTS

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            requested_groupby = tool_input.get("groupby", "leiden")
            if requested_groupby not in adata.obs.columns:
                return _smart_unavailable_result(
                    tool="run_deg",
                    message=(
                        f"Differential expression needs a valid grouping column, but '{requested_groupby}' "
                        "is not available on the current in-memory dataset."
                    ),
                    adata_obj=adata,
                    missing_prerequisites=["grouping"],
                    recovery_options=[
                        "Run clustering first, then run DEG on the cluster key.",
                        "Use one of the available annotation or metadata columns as the DEG grouping.",
                    ],
                    extra={"requested_groupby": requested_groupby},
                )
            groupby = _validate_obs_column(
                adata,
                requested_groupby,
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
            adata, _ = get_adata(tool_input, adata)
            plot_type = tool_input["plot_type"]
            output_path = tool_input.get("output_path")
            if not output_path:
                safe_color = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(tool_input.get("color_by", "plot")))
                output_path = f"{plot_type}_{safe_color or 'plot'}.png"
            color_by = tool_input.get("color_by")
            genes = tool_input.get("genes", [])
            include_image = tool_input.get("include_image", True)
            if plot_type == "umap" and "X_umap" not in adata.obsm:
                return _smart_unavailable_result(
                    tool="generate_figure",
                    message="UMAP cannot be rendered because the embedding is not available on the current in-memory dataset.",
                    adata_obj=adata,
                    missing_prerequisites=["embedding"],
                    recovery_options=[
                        "Run PCA, neighbors, and UMAP first.",
                        "If you only need a summary of current state, inspect the session instead of plotting.",
                    ],
                    extra={"plot_type": plot_type, "color_by": color_by},
                )
            if plot_type == "umap" and color_by not in (None, "") and color_by not in adata.obs.columns and color_by not in adata.var_names:
                return _smart_unavailable_result(
                    tool="generate_figure",
                    message=f"UMAP coloring key '{color_by}' is not available on the current in-memory dataset.",
                    adata_obj=adata,
                    missing_prerequisites=["valid_color_key"],
                    recovery_options=[
                        "Use one of the available obs columns or genes for coloring.",
                        "Render a plain UMAP without coloring.",
                    ],
                    extra={"plot_type": plot_type, "requested_color_by": color_by},
                )
            result = _render_figure(
                adata,
                plot_type=plot_type,
                output_path=output_path,
                color_by=color_by,
                genes=genes,
                include_image=include_image,
            )
            result["available_clusterings"] = _clusterings_payload(adata)
            artifact = _artifact_payload(
                output_path,
                role="figure",
                metadata={"plot_type": plot_type, "color_by": color_by},
            )
            verification_checks = [
                _check("figure_exists", os.path.exists(output_path), f"Figure exists at {output_path}."),
            ]
            if plot_type == "umap":
                verification_checks.append(
                    _check(
                        "color_key_valid",
                        color_by in adata.obs.columns or color_by in adata.var_names,
                        f"Color key '{color_by}' exists in AnnData.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary=f"Generated a {plot_type} figure colored by '{color_by}'.",
                artifacts_created=[artifact] if artifact is not None else [],
                verification=_build_verification(
                    "passed",
                    "Figure output was created and verified.",
                    verification_checks,
                ),
            )

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
