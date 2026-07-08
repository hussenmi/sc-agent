"""scagent_nat: NeMo Agent Toolkit (NAT) integration for the scagent scRNA-seq agent.

This package lives in a SEPARATE Python 3.11 venv (`nvidia-nat`) from scagent
(Python 3.10). It does NOT import scagent — it invokes scagent's CLI as a
subprocess — so the two dependency stacks never collide. It exposes:

  * a NAT *workflow function* (`scagent_analyze`) that runs scagent end-to-end
    against a dataset and returns the resulting run dir + annotated .h5ad path
  * a NAT *evaluator* (`annotation_ari`) that scores scagent's clustering
    against published ground-truth labels via Adjusted Rand Index

Components are registered with NAT through the `nat.components` entry point
(see pyproject.toml -> scagent_nat.register).
"""
