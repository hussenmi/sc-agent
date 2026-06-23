"""Entry-point module imported by NAT (group `nat.components`).

Importing this module triggers the @register_function / @register_evaluator
decorators in the submodules, making `scagent_analyze` and `annotation_ari`
available to NAT configs by their `_type`.
"""

from scagent_nat import annotation_eval  # noqa: F401  (registers `annotation_ari`)
from scagent_nat import luca_atlas_eval  # noqa: F401  (registers `luca_atlas`)
from scagent_nat import workflow  # noqa: F401  (registers `scagent_analyze`)
