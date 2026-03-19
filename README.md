# scagent

**Single-cell RNA-seq Analysis Agent** - An autonomous single-cell analysis toolkit that encapsulates lab best practices and can reason about your data.

## Features

### Core Analysis
- **Data Inspection**: Automatically detect data state, gene ID format, and recommend next steps
- **QC Pipeline**: Batch-aware doublet detection (Scrublet), MT/ribo filtering
- **Normalization**: Library size normalization with raw counts preservation
- **Dimensionality Reduction**: PCA, neighbors, UMAP with lab-validated parameters
- **Clustering**: Leiden and PhenoGraph (Jaccard-weighted)
- **Cell Type Annotation**: CellTypist and Scimilarity with proper target_sum=10000
- **Batch Correction**: Scanorama and Harmony
- **Differential Expression**: Wilcoxon, t-test, logistic regression

### Autonomous Agent
- **Multi-provider**: Works with OpenAI (GPT-4o, GPT-5.4) or Anthropic (Claude)
- **18 Built-in Tools**: QC, clustering, annotation, visualization, and more
- **Code Generation**: Dynamically writes and executes Python for custom analyses
- **Vision Support**: Agent can see and analyze generated plots
- **Web Search**: Look up gene functions, pathways, best practices
- **Interactive**: Can ask user questions when clarification needed
- **Package Installation**: Request new packages with user approval
- **Reproducibility**: All runs create manifests with full provenance

## Installation

```bash
cd /data1/peerd/ibrahih3/cs_agent
source setup.sh
```

Or with uv directly:
```bash
uv sync
uv pip install -e ".[agent]"
```

## Configuration

Create a `.env` file (copy from `.env.example`):

```bash
# API Keys (only one provider required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Defaults
SCAGENT_PROVIDER=openai
SCAGENT_MODEL=gpt-5.4-mini
```

## Quick Start

### Library Usage

```python
import scagent
from scagent.core import load_data, inspect_data, run_qc_pipeline

# Load and inspect
adata = load_data("your_data.h5")
state = inspect_data(adata)
print(f"Cells: {state.n_cells}, Genes: {state.n_genes}")
print(f"Gene format: {state.gene_id_format}")

# Run QC
run_qc_pipeline(adata)
```

### Autonomous Agent

```python
from scagent.agent import SCAgent

# Initialize (reads from .env)
agent = SCAgent()

# Simple analysis
result = agent.analyze(
    "QC and cluster this PBMC data, then identify cell types",
    data_path="pbmc.h5"
)

# Complex multi-step analysis
result = agent.analyze("""
    1. Run QC and clustering
    2. Find clusters with highest MT% - could indicate stressed cells
    3. Investigate marker genes for those clusters
    4. Generate UMAP colored by MT%
    5. Provide synthesis with quality concerns
""", data_path="pbmc.h5")
```

### Agent Capabilities

The agent can:
- **Use built-in tools**: 18 pre-defined analysis functions
- **Generate custom code**: For anything not covered by tools
- **See outputs**: Reads stdout from code, views generated plots
- **Reason and adapt**: Adjusts approach based on what it finds
- **Ask questions**: When clarification is needed
- **Install packages**: With user approval

Example of agent reasoning:
```
Agent: "Cluster 13 has highest MT% (mean 15.14%) and its markers
        are mitochondrial genes - this is likely a stressed/dying
        cell population, not a real immune cell type."
```

## Available Tools

| Category | Tools |
|----------|-------|
| **Analysis** | `run_qc`, `normalize_and_hvg`, `run_dimred`, `run_clustering`, `run_celltypist`, `run_batch_correction`, `run_deg` |
| **Visualization** | `generate_figure` (UMAP, violin, dotplot, heatmap) |
| **Inspection** | `inspect_data`, `get_cluster_sizes`, `get_top_markers`, `summarize_qc_metrics`, `get_celltypes`, `list_obs_columns` |
| **Meta** | `ask_user`, `run_code`, `web_search`, `install_package` |

## Lab Parameters (Best Practices)

| Parameter | Value | Notes |
|-----------|-------|-------|
| MT threshold (cells) | 25% | |
| MT threshold (nuclei) | 5% | |
| Min cells per gene | ~55 (np.exp(4)) | |
| Scrublet doublet rate | 0.06 | |
| HVG count | 4000 | seurat_v3 flavor |
| N PCs | 30 | |
| N neighbors | 30 | |
| Leiden resolution | 1.0 | |
| UMAP min_dist | 0.1 | |
| **CellTypist target_sum** | **10000** | CRITICAL |
| Scanorama dimred/knn | 30 | |

## Run Output Structure

Each agent run creates a structured directory:

```
run_2026_03_18_230927_full_analysis/
├── manifest.json           # Full reproducibility log
├── reports/
│   └── summary.md          # Analysis summary
├── code/
│   └── *.py                # Generated code saved for reuse
├── figures/
│   └── *.png               # Visualizations
├── intermediate/
│   └── *.h5ad              # Checkpoint files
├── pbmc_qc.h5ad
├── pbmc_clustered.h5ad
└── ...
```

## Testing

```bash
# Test tools only (no API call)
python test_agent.py --tools-only

# Test with OpenAI
python test_agent.py --provider openai

# Full analysis test
python test_agent.py --provider openai --full
```

## License

MIT
