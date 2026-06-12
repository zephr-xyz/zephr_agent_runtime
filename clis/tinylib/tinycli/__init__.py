"""tinycli — opinionated multi-stage CLI pipelines.

Stages are (plan, run) pairs.  Plan function signatures define CLI
arguments via ``Annotated``, ``Literal``, and default values.
Config priority: function defaults < TOML file < CLI flags.
"""

from clis.tinylib.tinycli.effects import PlanEffect, PlanEffectProvider
from clis.tinylib.tinycli.pipeline import DeriveError, Pipeline

__all__ = ["DeriveError", "Pipeline", "PlanEffect", "PlanEffectProvider"]
