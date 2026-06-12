"""Native dependency artifact locations and bridge generation."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from clis.support.paths import REPO_ROOT
from clis.tinylib.tinycli import DeriveError


GLOBAL_NATIVE_DEPS_BASE = Path("/opt/zephr/diagnosis/native_dependencies/zephragent")
BRIDGE_DIR = REPO_ROOT / "1_build" / "native_deps"
LEGACY_NATIVE_DEPS_DIR = REPO_ROOT / "0_native_deps"


HostPlatform = Literal["macos", "linux"]


@dataclass(frozen=True)
class NativeDepsRecipeInput:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class NativeDepsRecipeFingerprint:
    recipe_hash: str
    inputs: tuple[NativeDepsRecipeInput, ...]


@dataclass(frozen=True)
class NativeDepsAddress:
    host_platform: str
    recipe_hash: str
    root: Path
    addressing: str = "content_hash"
    recipe_alias: str | None = None
    recipe_inputs: tuple[NativeDepsRecipeInput, ...] = ()


def native_deps_host_platform() -> HostPlatform:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Linux":
        return "linux"
    raise DeriveError(f"unsupported native deps host platform: {system}")


def native_deps_recipe_inputs() -> list[Path]:
    # Keep Python dependency resolution out of this fingerprint. The native
    # artifact is keyed by source/config inputs that actually affect the C/C++
    # dependency build; lockfile churn from model conversion deps should not
    # force a native rebuild.
    paths: list[Path] = [
        REPO_ROOT / "clis" / "support" / "platform_dependencies" / "android_config.py",
        REPO_ROOT / "tinylib" / "zephr_agent_tools" / "proto" / "waypoints.proto",
        REPO_ROOT / "clis" / "support" / "native_dependencies" / "prepare_native_deps.py",
        REPO_ROOT / "clis" / "support" / "native_dependencies" / "process_runner.py",
    ]
    return [path for path in paths if path.exists()]


def native_deps_recipe_fingerprint() -> NativeDepsRecipeFingerprint:
    digest = hashlib.sha256()
    descriptors: list[NativeDepsRecipeInput] = []
    for path in native_deps_recipe_inputs():
        relative_path = str(path.relative_to(REPO_ROOT))
        content = path.read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        descriptors.append(
            NativeDepsRecipeInput(
                relative_path=relative_path,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        )
    return NativeDepsRecipeFingerprint(
        recipe_hash=digest.hexdigest(),
        inputs=tuple(descriptors),
    )


def native_deps_recipe_input_descriptors() -> list[NativeDepsRecipeInput]:
    return list(native_deps_recipe_fingerprint().inputs)


def native_deps_recipe_hash() -> str:
    return native_deps_recipe_fingerprint().recipe_hash


def native_deps_address(recipe_alias: str | None = None) -> NativeDepsAddress:
    host_platform = native_deps_host_platform()
    fingerprint = native_deps_recipe_fingerprint()
    recipe_hash = fingerprint.recipe_hash
    if recipe_alias:
        root = GLOBAL_NATIVE_DEPS_BASE / host_platform / "dev" / recipe_alias
        return NativeDepsAddress(
            host_platform=host_platform,
            recipe_hash=recipe_hash,
            root=root,
            addressing="dev_alias",
            recipe_alias=recipe_alias,
            recipe_inputs=fingerprint.inputs,
        )
    root = GLOBAL_NATIVE_DEPS_BASE / host_platform / "recipes" / recipe_hash
    return NativeDepsAddress(
        host_platform=host_platform,
        recipe_hash=recipe_hash,
        root=root,
        recipe_inputs=fingerprint.inputs,
    )


def global_build_lock_path() -> Path:
    return GLOBAL_NATIVE_DEPS_BASE / ".global_build.lock"


def ready_path(root: Path) -> Path:
    return root / ".ready"


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def build_status_path(root: Path) -> Path:
    return root / "build_status.json"


def build_log_path(root: Path) -> Path:
    return root / "build.log"


def ensure_no_legacy_native_deps() -> None:
    if LEGACY_NATIVE_DEPS_DIR.exists():
        raise DeriveError(
            "Legacy 0_native_deps/ detected. Native deps now live only under "
            f"{GLOBAL_NATIVE_DEPS_BASE}. Remove the old directory with: "
            "rm -rf 0_native_deps"
        )


def artifact_is_ready(root: Path, recipe_hash: str | None = None) -> bool:
    if not ready_path(root).is_file() or not manifest_path(root).is_file():
        return False
    try:
        manifest = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if recipe_hash is not None and manifest.get("recipe_hash") != recipe_hash:
        return False
    return manifest.get("schema") == 1


def bridge_paths() -> dict[str, Path]:
    return {
        "json": BRIDGE_DIR / "native_deps.json",
        "cmake": BRIDGE_DIR / "native_deps.cmake",
        "properties": BRIDGE_DIR / "native_deps.properties",
        "xcconfig": BRIDGE_DIR / "native_deps.xcconfig",
    }


def write_native_deps_bridges(address: NativeDepsAddress, *, dry_run: bool = False) -> None:
    root = address.root
    payload = {
        "schema": 1,
        "host_platform": address.host_platform,
        "recipe_hash": address.recipe_hash,
        "addressing": address.addressing,
        "recipe_alias": address.recipe_alias,
        "root": str(root),
        "ready": artifact_is_ready(root, address.recipe_hash),
    }
    paths = bridge_paths()
    if dry_run:
        for path in paths.values():
            print(f"write {path}")
        return

    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["cmake"].write_text(
        "\n".join(
            [
                "# Generated by prepare_dev. Do not edit.",
                f'set(ZEPHR_NATIVE_DEPS_ROOT "{root}")',
                f'set(ZEPHR_NATIVE_DEPS_RECIPE_HASH "{address.recipe_hash}")',
                f'set(ZEPHR_NATIVE_DEPS_HOST_PLATFORM "{address.host_platform}")',
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["properties"].write_text(
        "\n".join(
            [
                "# Generated by prepare_dev. Do not edit.",
                f"ZEPHR_NATIVE_DEPS_ROOT={root}",
                f"ZEPHR_NATIVE_DEPS_RECIPE_HASH={address.recipe_hash}",
                f"ZEPHR_NATIVE_DEPS_HOST_PLATFORM={address.host_platform}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["xcconfig"].write_text(
        "\n".join(
            [
                "// Generated by prepare_dev. Do not edit.",
                f"ZEPHR_NATIVE_DEPS_ROOT = {root}",
                f"ZEPHR_NATIVE_DEPS_RECIPE_HASH = {address.recipe_hash}",
                f"ZEPHR_NATIVE_DEPS_HOST_PLATFORM = {address.host_platform}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def require_native_deps_ready(address: NativeDepsAddress) -> None:
    if not artifact_is_ready(address.root, address.recipe_hash):
        status = build_status_path(address.root)
        extra = f" See status: {status}" if status.exists() else ""
        raise DeriveError(
            "Native deps are not ready for recipe "
            f"{address.recipe_hash[:12]} at {address.root}.{extra} "
            "Run: uv run prepare_native_deps"
        )


def read_bridge_root() -> Path:
    bridge = bridge_paths()["json"]
    if not bridge.is_file():
        raise RuntimeError("native deps bridge missing; run `uv run prepare_dev`")
    data = json.loads(bridge.read_text(encoding="utf-8"))
    return Path(data["root"])


def tool_version(cmd: list[str]) -> str | None:
    executable = shutil.which(cmd[0])
    if executable is None:
        return None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None
