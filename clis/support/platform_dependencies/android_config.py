"""Shared Android platform version configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from clis.support.paths import REPO_ROOT

PLATFORM_VERSIONS = REPO_ROOT / "platform_versions.json"


@dataclass(frozen=True)
class AndroidPlatformConfig:
    compile_sdk: str
    min_sdk: str
    build_tools: str
    cmake: str
    ndk: str


@cache
def android_platform_config() -> AndroidPlatformConfig:
    raw = json.loads(PLATFORM_VERSIONS.read_text(encoding="utf-8"))
    android = raw["android"]
    return AndroidPlatformConfig(
        compile_sdk=_required_string(android, "compile_sdk"),
        min_sdk=_required_string(android, "min_sdk"),
        build_tools=_required_string(android, "build_tools"),
        cmake=_required_string(android, "cmake"),
        ndk=_required_string(android, "ndk"),
    )


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"platform_versions.json android.{key} must be a non-empty string")
    return value
