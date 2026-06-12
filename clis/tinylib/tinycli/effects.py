"""Planned side effects for tinycli pipeline plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence


PlanEffectKind = Literal[
    "read_file",
    "read_dir",
    "write_file",
    "write_dir",
    "write_pattern",
    "run_command",
    "compute",
    "download",
    "hf_download_to_cache",
    "device_action",
    "network",
]


@dataclass(frozen=True)
class PlanEffect:
    """A filesystem or external effect expected from a planned stage."""

    kind: PlanEffectKind
    label: str
    path: Path | None = None
    pattern: str | None = None
    subject: str | None = None
    required: bool = True


class PlanEffectProvider(Protocol):
    """Protocol for plan objects that can describe their expected effects."""

    def plan_effects(self) -> Sequence[PlanEffect]:
        """Return effects expected when this plan executes."""
        ...
