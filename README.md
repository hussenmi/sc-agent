# scagent

**Single-cell RNA-seq Analysis Agent** - A standardized single-cell RNA-seq analysis package that encapsulates lab best practices.

## Features

- **Data Inspection**: Automatically detect data state and recommend next steps
- **QC Pipeline**: Batch-aware doublet detection, MT/ribo filtering
- **Normalization**: Library size normalization with raw counts preservation
- **Dimensionality Reduction**: PCA, neighbors, UMAP
- **Clustering**: Leiden and PhenoGraph (Jaccard-weighted)
- **Cell Type Annotation**: CellTypist and Scimilarity integration
- **Batch Correction**: Scanorama and Harmony
- **Autonomous Agent**: Claude API-powered analysis

## Installation

```bash
cd /data1/peerd/ibrahih3/cs_agent
source setup.sh
```

Or with uv directly:
```bash
uv sync
uv pip install -e .
```

## Quick Start

```python
import scagent
from scagent.core import load_data, inspect_data, run_qc_pipeline

# Load data
adata = load_data("your_data.h5")

# Inspect state
state = inspect_data(adata)
print(scagent.core.inspector.summarize_state(state))

# Run QC
run_qc_pipeline(adata)
```

## Autonomous Agent

```python
from scagent.agent import SCAgent

agent = SCAgent()  # Uses ANTHROPIC_API_KEY
result = agent.analyze("QC and cluster this PBMC data", data_path="pbmc.h5")
```

## Lab Parameters

| Parameter | Value |
|-----------|-------|
| MT threshold (cells) | 25% |
| MT threshold (nuclei) | 5% |
| HVG count | 4000 (seurat_v3) |
| N PCs | 30 |
| N neighbors | 30 |
| UMAP min_dist | 0.1 |
| Leiden resolution | 1.0 |
| CellTypist target_sum | 10000 |

## License

MIT
