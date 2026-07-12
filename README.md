# scagent

`scagent` is an LLM-powered single-cell RNA-seq analysis agent. It keeps one primary `AnnData` object in memory, runs analysis through structured tools, and records every run in a timestamped output directory.

The goal is a practical analysis partner: inspect the data, run QC, normalize, embed, cluster, annotate, validate annotations with external marker evidence, generate figures, and save a reproducible result.

## Core Components

- `scagent/agent/agent.py` — `SCAgent`, provider loops, tool routing, context management.
- `scagent/agent/tools.py` — native tool schemas and handlers.
- `scagent/agent/prompts.py` — workflow instructions and guardrails.
- `scagent/agent/world_state.py` — durable session state rendered back into the model.
- `scagent/core/` — QC, normalization, PCA/neighbors/UMAP, clustering, DEG primitives.
- `scagent/annotation/` — CellTypist and Scimilarity wrappers.
- `scagent/batch/` — Harmony, Scanorama, scVI, BBKNN helpers.
- `scagent/mcp/` — optional MCP tools for external databases such as PanglaoDB.
- `scagent/cli.py` — command-line entry point.

## Setup

### Environment

scagent runs on two hosts — the **Iris HPC** (x86_64) and the **DGX Spark** (aarch64/GB10). Activate the environment for your host/mode from the repo root **before** running anything:

| Host / mode | Activate | For |
|---|---|---|
| Iris — CPU / dev | `source setup.sh` | tests, lint, quick iteration (no GPU) |
| Iris — GPU | `source setup_gpu.sh` | RAPIDS-accelerated runs; scVI + CellBender + scimilarity |
| Spark — GPU | `pixi shell -e gpu` | RAPIDS on the GB10; scVI + CellTypist + scimilarity |

Confirm the GPU backend is actually live before a long run — a silent CPU fallback only shows up as `"backend": "scanpy_cpu"` in `manifest.json`:

```bash
python -c "from scagent.core.gpu import gpu_capability_report as g; print(g())"
```

See [`docs/environments.md`](docs/environments.md) for the full per-host reference: what each env provides, the `LD_PRELOAD` requirements, heavy-tool availability (scVI / CellBender / scimilarity), and serving.

### Provider / model

Configure provider/model settings in `.env`:

```bash
SCAGENT_PROVIDER=openai
SCAGENT_MODEL=...
SCAGENT_BASE_URL=http://host:port/v1

OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Supported provider paths include Anthropic, OpenAI-compatible/vLLM, Groq, Gemini, and Codex.

### MCP servers (optional)

The optional external-database tools (biocontext, PubMed) are launched from
`.mcp.json`, which is **not tracked** because it holds absolute paths that differ
per host. Set it up once per machine:

```bash
cp .mcp.json.example .mcp.json
# then replace each /ABSOLUTE/PATH/TO/... with the real path on this host, e.g.
which biocontext_kb   # -> paste into the "biocontext" command
which pubmedmcp       # -> paste into the "pubmed" command
```

Skip this if you are not using the MCP database tools.

## Basic Use

```bash
scagent start
scagent analyze --data path/to/data.h5ad
scagent inspect path/to/data.h5ad
```

Python:

```python
from scagent.agent import SCAgent

agent = SCAgent()
agent.analyze(
    "Analyze this dataset from scratch using raw counts; compare against existing annotations if present.",
    data_path="path/to/data.h5ad",
)
```

## Analysis Shape

The standard agent workflow is:

```text
inspect/load
  -> QC flagging
  -> normalize + HVG
  -> PCA
  -> neighbors
  -> UMAP
  -> clustering
  -> cluster-level QC and cleanup decision
  -> annotation
  -> DEG
  -> external marker validation
  -> final figures/report/save
```

Important defaults:

- QC is flag-first; cell removal is decided after clustering.
- Ribosomal genes are inspected during QC, then removed during `normalize_and_hvg` unless the user/source says to keep them.
- Existing annotations and metadata are preserved for comparison when doing a “from scratch” analysis.
- CellTypist/Scimilarity/DEG labels are candidates, not final labels. PanglaoDB or another external marker source should adjudicate final annotations.
- `run_code` is available for custom analysis, but unsafe mutations are guarded.

## Run Outputs

Each run creates a directory like:

```text
run_YYYY_MM_DD_HHMMSS/
├── manifest.json
├── logs/
├── code/
├── reports/
├── figures/
└── *.h5ad
```

`manifest.json` is the main provenance record. `code/` stores generated `run_code` snippets. Figures and reports are written into their respective folders.

## Quick Checks

```bash
ssh iris-hpc 'cd /data1/peerd/ibrahih3/cs_agent && source setup.sh && python -c "from scagent.agent.tools import get_tools; print(len(get_tools()))"'
```
