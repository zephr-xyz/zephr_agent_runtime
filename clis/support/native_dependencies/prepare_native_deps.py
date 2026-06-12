"""Pre-build native dependencies for the analyze project.

Downloads, configures, and compiles C/C++ libraries that our native
extensions depend on. The output is a set of installed static libraries
under ./<platform>/2_installed/<build_type>/<arch>/<dep_name>/ with standard
layouts, ready to be consumed by downstream build targets.

Usage:
    prepare_native_deps
    prepare_native_deps --platform macos
    prepare_native_deps --platform ios
    prepare_native_deps --platform linux
    prepare_native_deps --CMAKE_BUILD_TYPE Release
    prepare_native_deps --dry_run
    prepare_native_deps --no_clean
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as platform_mod
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import anyio
import psutil

from clis.support.native_dependencies.locator import (
    NativeDepsAddress,
    artifact_is_ready,
    build_log_path,
    build_status_path,
    ensure_no_legacy_native_deps,
    global_build_lock_path,
    manifest_path,
    native_deps_address,
    ready_path,
    tool_version,
)
from clis.support.native_dependencies.process_runner import Runner
from clis.support.paths import REPO_ROOT
from clis.support.platform_dependencies.android_config import android_platform_config
from clis.tinylib.tinycli import DeriveError
from clis.tinylib.tinylog import log


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NATIVE_DEPS_DIR = native_deps_address().root
DOWNLOAD_DIR = NATIVE_DEPS_DIR / "downloads"
TARGET_STAMP_NAME = ".prepare_native_deps.json"

DEFAULT_ARCHES: dict[str, list[str]] = {
    "macos": [platform_mod.machine()],
    "linux": [platform_mod.machine()],
    "ios": ["arm64"],
    "android": ["arm64-v8a"],
}
DEFAULT_BUILD_TYPES = ["Release", "Debug"]

# LiteRT mainline pin. Keep the checkout pin as the commit SHA because readiness
# checks compare against `git rev-parse HEAD`.
LITERT_VERSION = "main"
LITERT_SOURCE_REF = "4b26ed7d2ae0a1fc8aebb876827841ac550eb22c"
LITERT_REPO = "https://github.com/google-ai-edge/LiteRT.git"
LITERT_LFS_BATCH_URL = f"{LITERT_REPO}/info/lfs/objects/batch"

# Must match sentencepiece's bundled protobuf-lite (SPM_USE_BUILTIN_PROTOBUF=ON).
PROTOBUF_VERSION = "v3.14.0"
PROTOBUF_REPO = "https://github.com/protocolbuffers/protobuf.git"

ZEPHR_AGENT_TOOLS_DIR = REPO_ROOT / "tinylib" / "zephr_agent_tools"
WAYPOINTS_PROTO = ZEPHR_AGENT_TOOLS_DIR / "proto" / "waypoints.proto"
WAYPOINTS_GENERATED_PROTO_DIR = ZEPHR_AGENT_TOOLS_DIR / "generated" / "proto"

# LiteRT accelerator prebuilts are committed as Git LFS pointer files in the
# LiteRT source tree. Keep our source-built libLiteRt pinned to the commit above
# and fetch only the matching repo-pinned accelerator objects.
LITERT_PREBUILT_DOWNLOAD_DIRNAME = "litert_repo_prebuilts"

IOS_DEPLOYMENT_TARGET = "16.0"
ANDROID_CONFIG = android_platform_config()
ANDROID_LINKER_PAGE_SIZE = "16384"

# Maps our platform/arch names to the directory names under litert/prebuilt.
LITERT_PREBUILT_DIRS: dict[tuple[str, str], str] = {
    ("macos", "arm64"): "macos_arm64",
    ("linux", "aarch64"): "linux_arm64",
    ("linux", "x86_64"): "linux_x86_64",
    ("ios", "arm64"): "ios_arm64",
    ("android", "arm64-v8a"): "android_arm64",
}

LITERT_PREBUILT_ACCELERATOR_FILES: dict[tuple[str, str], tuple[str, ...]] = {
    ("macos", "arm64"): ("libLiteRtMetalAccelerator.dylib",),
    ("ios", "arm64"): ("libLiteRtMetalAccelerator.dylib",),
    ("android", "arm64-v8a"): ("libLiteRtClGlAccelerator.so",),
    ("linux", "aarch64"): ("libLiteRtWebGpuAccelerator.so",),
    ("linux", "x86_64"): ("libLiteRtWebGpuAccelerator.so",),
}


def _platform_dirs(plat: str) -> tuple[Path, Path]:
    """Return (configure, install) dirs for a platform."""
    base = NATIVE_DEPS_DIR / plat
    return base / "1_configure", base / "2_installed"


def _target_root(install_dir: Path, build_type: str, arch: str) -> Path:
    return install_dir / build_type / arch


def _target_stamp_path(install_dir: Path, build_type: str, arch: str) -> Path:
    return _target_root(install_dir, build_type, arch) / TARGET_STAMP_NAME


def _target_stamp_payload(
    plan: "PreparePlan",
    plat: str,
    build_type: str,
    arch: str,
) -> dict[str, str | int | None]:
    return {
        "schema": 1,
        "inputs_hash": plan.address.recipe_hash,
        "platform": plat,
        "build_type": build_type,
        "arch": arch,
        "host_system": platform_mod.system(),
        "host_machine": platform_mod.machine(),
        "android_ndk_home": (
            str(plan.android_ndk_home)
            if plat == "android" and plan.android_ndk_home
            else None
        ),
    }


def _target_stamp_is_valid(
    plan: "PreparePlan",
    plat: str,
    install_dir: Path,
    build_type: str,
    arch: str,
) -> bool:
    stamp = _target_stamp_path(install_dir, build_type, arch)
    if not stamp.is_file():
        log.trace("native deps stamp missing", stamp=stamp)
        return False
    try:
        payload = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.trace("native deps stamp unreadable", stamp=stamp)
        return False
    expected = _target_stamp_payload(plan, plat, build_type, arch)
    if payload != expected:
        log.trace("native deps stamp mismatch", stamp=stamp)
        return False
    return True


def _write_target_stamp(
    plan: "PreparePlan",
    plat: str,
    install_dir: Path,
    build_type: str,
    arch: str,
) -> None:
    stamp = _target_stamp_path(install_dir, build_type, arch)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    payload = _target_stamp_payload(plan, plat, build_type, arch)
    tmp = stamp.with_suffix(stamp.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(stamp)


def _status(run: Runner, dry_run: bool, message: str) -> None:
    if not dry_run:
        log.debug(message)


def _platform_cmake_args(
    plat: str, arch: str, android_ndk_home: Path | None = None,
) -> list[str]:
    """Return cmake args specific to the target platform/arch."""
    if plat == "macos":
        return [
            f"-DCMAKE_OSX_ARCHITECTURES={arch}",
            "-DCMAKE_C_COMPILER=/usr/bin/clang",
            "-DCMAKE_CXX_COMPILER=/usr/bin/clang++",
        ]
    elif plat == "ios":
        return [
            "-DCMAKE_SYSTEM_NAME=iOS",
            f"-DCMAKE_OSX_ARCHITECTURES={arch}",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={IOS_DEPLOYMENT_TARGET}",
        ]
    elif plat == "android":
        if not android_ndk_home:
            raise RuntimeError("android_ndk_home not resolved — call _resolve_android_ndk() during plan")
        return [
            f"-DCMAKE_TOOLCHAIN_FILE={android_ndk_home}/build/cmake/android.toolchain.cmake",
            f"-DANDROID_ABI={arch}",
            f"-DANDROID_PLATFORM=android-{ANDROID_CONFIG.min_sdk}",
            f"-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size={ANDROID_LINKER_PAGE_SIZE}",
            f"-DCMAKE_MODULE_LINKER_FLAGS=-Wl,-z,max-page-size={ANDROID_LINKER_PAGE_SIZE}",
            f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,-z,max-page-size={ANDROID_LINKER_PAGE_SIZE}",
        ]
    elif plat == "linux":
        return []
    else:
        raise ValueError(f"Unknown platform: {plat}")


def _check_ios_sdk() -> None:
    """Verify the iOS SDK is available via Xcode."""
    import subprocess

    try:
        result = subprocess.run(
            ["xcrun", "--sdk", "iphoneos", "--show-sdk-path"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        raise DeriveError("xcrun not found. Install Xcode to build for iOS.")
    if result.returncode != 0:
        raise DeriveError(
            "iOS SDK not found. Open Xcode and install iOS platform tools."
        )


def _resolve_android_ndk() -> Path:
    """Resolve the Android NDK path, inferring from environment if needed.

    Resolution order:
      1. ANDROID_NDK_HOME (explicit)
      2. ANDROID_HOME/ndk/<pinned> or ANDROID_SDK_ROOT/ndk/<pinned>
      3. ANDROID_HOME/ndk-bundle (Nix androidenv layout)
      4. ANDROID_HOME/ndk/<latest> or ANDROID_SDK_ROOT/ndk/<latest>
      5. ~/Library/Android/sdk/ndk/<latest> (macOS) or ~/Android/Sdk/ndk/<latest>
    """
    toolchain = Path("build/cmake/android.toolchain.cmake")

    ndk_home = os.environ.get("ANDROID_NDK_HOME")
    if ndk_home:
        ndk_path = Path(ndk_home)
        if not (ndk_path / toolchain).is_file():
            raise DeriveError(
                f"ANDROID_NDK_HOME={ndk_home} does not contain {toolchain}"
            )
        return ndk_path

    sdk_root = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if not sdk_root:
        if platform_mod.system() == "Darwin":
            sdk_root = str(Path.home() / "Library" / "Android" / "sdk")
        else:
            sdk_root = str(Path.home() / "Android" / "Sdk")

    ndk_dir = Path(sdk_root) / "ndk"
    if not ndk_dir.is_dir():
        raise DeriveError(
            f"No NDK found. Searched ANDROID_NDK_HOME, ANDROID_HOME, "
            f"ANDROID_SDK_ROOT, and {ndk_dir}. "
            "Set ANDROID_NDK_HOME or install an NDK via Android Studio."
        )

    preferred_ndk = ndk_dir / ANDROID_CONFIG.ndk
    if (preferred_ndk / toolchain).is_file():
        return preferred_ndk

    ndk_bundle = Path(sdk_root) / "ndk-bundle"
    if (ndk_bundle / toolchain).is_file():
        return ndk_bundle

    # Pick the latest NDK that has a cmake toolchain
    ndks = sorted(
        [d for d in ndk_dir.iterdir() if (d / toolchain).is_file()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not ndks:
        raise DeriveError(
            f"No usable NDK in {ndk_dir} (none contain {toolchain})"
        )

    return ndks[0]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class PreparePlan:
    platforms: list[str]
    build_types: list[str]
    clean: bool
    verbose: bool
    dry_run: bool
    concurrency: int
    android_ndk_home: Path | None
    address: NativeDepsAddress
    status: bool
    retry_failed: bool
    break_lock: bool


def _default_platforms_for_host() -> list[str]:
    if platform_mod.system() == "Darwin":
        return ["macos", "ios", "android"]
    if platform_mod.system() == "Linux":
        return ["linux"]
    return []


def plan_prepare(
    platform: Annotated[list[Literal["macos", "ios", "android", "linux"]] | None, "target platforms (default: host native)"] = None,
    CMAKE_BUILD_TYPE: Annotated[Literal["Release", "Debug"] | None, "build type (default: both)"] = None,
    clean: Annotated[bool, "clean platform build dirs before building"] = False,
    verbose: Annotated[bool, "show build output"] = False,
    dry_run: Annotated[bool, "print commands without executing"] = False,
    status: Annotated[bool, "show global native deps build status and exit"] = False,
    retry_failed: Annotated[bool, "retry a recipe whose latest build_status.json is failed or interrupted"] = False,
    break_lock: Annotated[bool, "remove the global native deps build lock before building"] = False,
    recipe_alias: Annotated[str | None, "mutable dev alias for troubleshooting prepare_native_deps"] = None,
) -> PreparePlan:
    """Download and compile native C/C++ dependencies (sentencepiece, litert)."""
    ensure_no_legacy_native_deps()
    address = native_deps_address(recipe_alias=recipe_alias)
    platforms: list[str] = list(platform) if platform is not None else _default_platforms_for_host()
    build_types: list[str] = [CMAKE_BUILD_TYPE] if CMAKE_BUILD_TYPE else list(DEFAULT_BUILD_TYPES)
    android_ndk_home = _resolve_android_ndk() if "android" in platforms else None

    if status:
        return PreparePlan(
            platforms=platforms,
            build_types=build_types,
            clean=clean,
            verbose=verbose,
            dry_run=dry_run,
            concurrency=psutil.cpu_count(logical=False) or os.cpu_count() or 4,
            android_ndk_home=android_ndk_home,
            address=address,
            status=status,
            retry_failed=retry_failed,
            break_lock=break_lock,
        )

    if ("ios" in platforms or "macos" in platforms) and platform_mod.system() != "Darwin":
        apple_plats = [p for p in platforms if p in ("macos", "ios")]
        raise DeriveError(
            f"Cannot build for {', '.join(apple_plats)} on {platform_mod.system()}. "
            "Apple platform builds require macOS."
        )
    if "linux" in platforms and platform_mod.system() != "Linux":
        raise DeriveError(
            f"Cannot build for linux on {platform_mod.system()}. "
            "Linux host builds require Linux."
        )
    if "ios" in platforms:
        _check_ios_sdk()
    return PreparePlan(
        platforms=platforms,
        build_types=build_types,
        clean=clean,
        verbose=verbose,
        dry_run=dry_run,
        concurrency=psutil.cpu_count(logical=False) or os.cpu_count() or 4,
        android_ndk_home=android_ndk_home,
        address=address,
        status=status,
        retry_failed=retry_failed,
        break_lock=break_lock,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _set_native_deps_root(root: Path) -> None:
    global NATIVE_DEPS_DIR, DOWNLOAD_DIR
    NATIVE_DEPS_DIR = root
    DOWNLOAD_DIR = NATIVE_DEPS_DIR / "downloads"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _target_descriptors(plan: PreparePlan) -> list[dict[str, str]]:
    return [
        {"platform": plat, "build_type": bt, "arch": arch}
        for plat in plan.platforms
        for bt in plan.build_types
        for arch in DEFAULT_ARCHES[plat]
    ]


def _target_sort_key(target: dict[str, str]) -> tuple[int, int, str]:
    platform_order = {"macos": 0, "ios": 1, "android": 2, "linux": 3}
    build_type_order = {"Release": 0, "Debug": 1}
    return (
        platform_order.get(str(target.get("platform", "")), 99),
        build_type_order.get(str(target.get("build_type", "")), 99),
        str(target.get("arch", "")),
    )


def _target_key(target: dict[str, object]) -> tuple[str, str, str] | None:
    platform = target.get("platform")
    build_type = target.get("build_type")
    arch = target.get("arch")
    if not isinstance(platform, str) or not isinstance(build_type, str) or not isinstance(arch, str):
        return None
    return platform, build_type, arch


def _manifest_matches_recipe(plan: PreparePlan, manifest: dict[str, object] | None) -> bool:
    return bool(
        manifest
        and manifest.get("recipe_hash") == plan.address.recipe_hash
        and manifest.get("host_platform") == plan.address.host_platform
        and manifest.get("addressing") == plan.address.addressing
        and manifest.get("recipe_alias") == plan.address.recipe_alias
    )


def _manifest_covers_targets(plan: PreparePlan) -> bool:
    manifest = _read_json(manifest_path(plan.address.root))
    if not _manifest_matches_recipe(plan, manifest):
        return False
    targets = manifest.get("targets") if manifest else None
    if not isinstance(targets, list):
        return False
    existing = {
        key
        for target in targets
        if isinstance(target, dict)
        for key in [_target_key(target)]
        if key is not None
    }
    required = {
        key
        for target in _target_descriptors(plan)
        for key in [_target_key(target)]
        if key is not None
    }
    return required.issubset(existing)


def _stamped_target_descriptors(plan: PreparePlan) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for platform, arches in DEFAULT_ARCHES.items():
        _, install_dir = _platform_dirs(platform)
        for build_type in DEFAULT_BUILD_TYPES:
            for arch in arches:
                if _target_stamp_is_valid(plan, platform, install_dir, build_type, arch):
                    targets.append({
                        "platform": platform,
                        "build_type": build_type,
                        "arch": arch,
                    })
    return sorted(targets, key=_target_sort_key)


def _recipe_input_descriptors(plan: PreparePlan) -> list[dict[str, object]]:
    return [
        {
            "path": descriptor.relative_path,
            "sha256": descriptor.sha256,
            "size_bytes": descriptor.size_bytes,
        }
        for descriptor in plan.address.recipe_inputs
    ]


def _manifest_payload(plan: PreparePlan) -> dict[str, object]:
    return {
        "schema": 1,
        "addressing": plan.address.addressing,
        "recipe_alias": plan.address.recipe_alias,
        "recipe_hash": plan.address.recipe_hash,
        "host_platform": plan.address.host_platform,
        "host_details": {
            "system": platform_mod.system(),
            "machine": platform_mod.machine(),
            "platform": platform_mod.platform(),
            "python": platform_mod.python_version(),
            "cmake": tool_version(["cmake", "--version"]),
            "git": tool_version(["git", "--version"]),
            "xcodebuild": tool_version(["xcodebuild", "-version"]),
        },
        "targets": _stamped_target_descriptors(plan),
        "recipe_inputs": _recipe_input_descriptors(plan),
        "created_at": _utc_now(),
    }


def _write_build_status(
    plan: PreparePlan,
    *,
    state: str,
    stage: str | None = None,
    message: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": 1,
        "state": state,
        "recipe_hash": plan.address.recipe_hash,
        "host_platform": plan.address.host_platform,
        "addressing": plan.address.addressing,
        "recipe_alias": plan.address.recipe_alias,
        "root": str(plan.address.root),
        "targets": _target_descriptors(plan),
        "started_at": started_at,
        "finished_at": finished_at,
        "stage": stage,
        "message": message,
        "error": error,
        "log_path": str(build_log_path(plan.address.root)),
    }
    _write_json(build_status_path(plan.address.root), payload)


def _print_status(plan: PreparePlan) -> None:
    root = plan.address.root
    status_payload = _read_json(build_status_path(root))
    artifact_ready = artifact_is_ready(root, plan.address.recipe_hash)
    targets_current = _recipe_targets_are_current(plan)
    print(f"native deps root: {root}")
    print(f"host platform: {plan.address.host_platform}")
    print(f"recipe hash: {plan.address.recipe_hash}")
    print(f"ready: {targets_current}")
    if artifact_ready != targets_current:
        print(f"artifact ready: {artifact_ready}")
        print("current targets: False")
    for path in [build_status_path(root), manifest_path(root), build_log_path(root)]:
        print(f"{path.name}: {path if path.exists() else 'missing'}")
    lock = global_build_lock_path()
    lock_held = lock.exists()
    print(f"global lock: {lock if lock_held else 'not held'}")
    if status_payload and status_payload.get("state") == "building" and not lock_held:
        print("interrupted: build_status.json says building, but the global lock is not held")
        print("recovery: rerun with --retry_failed to continue, or --clean to rebuild target dirs")


def _print_recipe_hash_explanation(plan: PreparePlan) -> None:
    print("native deps recipe hash:")
    print("  algorithm: sha256(relative_path + NUL + file_bytes + NUL for each input)")
    print(f"  host platform: {plan.address.host_platform}")
    print(f"  recipe hash: {plan.address.recipe_hash}")
    print("  inputs:")
    for descriptor in plan.address.recipe_inputs:
        print(
            "    "
            f"{descriptor.relative_path} "
            f"sha256={descriptor.sha256} "
            f"size={descriptor.size_bytes}"
        )


def _print_skip_explanation(plan: PreparePlan) -> None:
    root = plan.address.root
    print("native deps current, skipping")
    _print_recipe_hash_explanation(plan)
    print("matched artifact:")
    print(f"  root: {root}")
    print(f"  ready: {ready_path(root)}")
    print(f"  manifest: {manifest_path(root)}")
    print(f"  build status: {build_status_path(root)}")
    print(f"  build log: {build_log_path(root)}")
    print("matched target stamps:")
    for plat in plan.platforms:
        _, install_dir = _platform_dirs(plat)
        for bt in plan.build_types:
            for arch in DEFAULT_ARCHES[plat]:
                print(f"  {_target_stamp_path(install_dir, bt, arch)}")


def _recipe_targets_are_current(plan: PreparePlan) -> bool:
    if not artifact_is_ready(plan.address.root, plan.address.recipe_hash):
        return False
    if not _manifest_covers_targets(plan):
        return False
    for plat in plan.platforms:
        _, install_dir = _platform_dirs(plat)
        for bt in plan.build_types:
            for arch in DEFAULT_ARCHES[plat]:
                if not _target_stamp_is_valid(plan, plat, install_dir, bt, arch):
                    return False
    return True


class _GlobalBuildLock:
    def __init__(self, plan: PreparePlan) -> None:
        self.plan = plan
        self.path = global_build_lock_path()

    def __enter__(self) -> "_GlobalBuildLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.plan.break_lock and self.path.exists():
            self.path.unlink()
        payload = {
            "schema": 1,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "started_at": _utc_now(),
            "host_platform": self.plan.address.host_platform,
            "recipe_hash": self.plan.address.recipe_hash,
            "root": str(self.plan.address.root),
            "status_path": str(build_status_path(self.plan.address.root)),
            "log_path": str(build_log_path(self.plan.address.root)),
        }
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_json(self.path) or {}
            raise DeriveError(
                "Native deps build already in progress.\n"
                f"Lock: {self.path}\n"
                f"Owner host: {existing.get('hostname', 'unknown')}\n"
                f"Owner pid: {existing.get('pid', 'unknown')}\n"
                f"Owner cwd: {existing.get('cwd', 'unknown')}\n"
                f"Started: {existing.get('started_at', 'unknown')}\n"
                f"Recipe: {existing.get('recipe_hash', 'unknown')}\n"
                f"Status: {existing.get('status_path', 'unknown')}\n"
                f"Log: {existing.get('log_path', 'unknown')}\n"
                "Coordinate with the owner, or rerun with --break_lock if the lock is stale."
            )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


async def run_prepare(plan: PreparePlan) -> None:
    _set_native_deps_root(plan.address.root)
    if plan.status:
        _print_status(plan)
        return
    if plan.dry_run:
        print(f"native deps root: {plan.address.root}")
        print(f"global lock: {global_build_lock_path()}")
        print(f"build status: {build_status_path(plan.address.root)}")
        print(f"build log: {build_log_path(plan.address.root)}")
        await _run_prepare_build(plan)
        return
    if _recipe_targets_are_current(plan) and not plan.clean:
        if plan.verbose:
            _print_skip_explanation(plan)
        else:
            _status(
                Runner(verbose=plan.verbose, dry_run=plan.dry_run),
                plan.dry_run,
                "native deps current, skipping",
            )
        return

    status = _read_json(build_status_path(plan.address.root))
    if (
        status
        and status.get("state") == "building"
        and not global_build_lock_path().exists()
        and not plan.retry_failed
        and not plan.clean
    ):
        raise DeriveError(
            "Native deps previous build was interrupted for this recipe.\n"
            f"Status: {build_status_path(plan.address.root)}\n"
            f"Log: {build_log_path(plan.address.root)}\n"
            "build_status.json says state=building, but the global lock is not held.\n"
            "Rerun with --retry_failed to continue using reusable partial outputs, "
            "or --clean to rebuild target dirs."
        )
    if (
        status
        and status.get("state") == "failed"
        and not plan.retry_failed
        and not plan.clean
    ):
        raise DeriveError(
            "Native deps previously failed for this recipe. "
            f"Status: {build_status_path(plan.address.root)} "
            f"Log: {build_log_path(plan.address.root)} "
            "Rerun with --retry_failed to try again."
        )

    with _GlobalBuildLock(plan):
        started_at = _utc_now()
        plan.address.root.mkdir(parents=True, exist_ok=True)
        ready_path(plan.address.root).unlink(missing_ok=True)
        build_log_path(plan.address.root).write_text(
            f"[{started_at}] prepare_native_deps start\n"
            f"root={plan.address.root}\n"
            f"host_platform={plan.address.host_platform}\n"
            f"recipe_hash={plan.address.recipe_hash}\n"
            f"targets={_target_descriptors(plan)}\n",
            encoding="utf-8",
        )
        _write_build_status(
            plan,
            state="building",
            stage="prepare",
            message="native deps build in progress",
            started_at=started_at,
        )
        try:
            await _run_prepare_build(plan)
        except BaseException as exc:
            _write_build_status(
                plan,
                state="failed",
                stage="prepare",
                message="native deps build failed",
                started_at=started_at,
                finished_at=_utc_now(),
                error=str(exc),
            )
            raise
        _write_json(manifest_path(plan.address.root), _manifest_payload(plan))
        _write_build_status(
            plan,
            state="ready",
            stage="prepare",
            message="native deps ready",
            started_at=started_at,
            finished_at=_utc_now(),
        )
        ready_path(plan.address.root).write_text(_utc_now() + "\n", encoding="utf-8")


async def _run_prepare_build(plan: PreparePlan) -> None:
    """Download and build native dependencies."""
    run = Runner(
        verbose=plan.verbose,
        dry_run=plan.dry_run,
        log_path=build_log_path(plan.address.root),
    )
    targets_to_stamp: list[tuple[str, Path, str, str]] = []

    if plan.clean:
        for plat in plan.platforms:
            configure_dir, install_dir = _platform_dirs(plat)
            for d in (configure_dir, install_dir):
                if d.exists():
                    if plan.dry_run:
                        print(f"rm -rf {d}")
                    else:
                        _status(run, plan.dry_run, f"Removing {d}")
                        shutil.rmtree(d)

    # Create directories (downloads are shared, configure/install are per-platform)
    dirs: list[str] = [str(DOWNLOAD_DIR)]
    for plat in plan.platforms:
        configure_dir, install_dir = _platform_dirs(plat)
        for bt in plan.build_types:
            for arch in DEFAULT_ARCHES[plat]:
                dirs += [
                    str(configure_dir / bt / arch),
                    str(install_dir / bt / arch),
                ]
    await run(["mkdir", "-p"] + dirs)

    label = (
        f"{', '.join(plan.build_types)}, "
        f"{'/'.join(plan.platforms)}, "
        f"{plan.concurrency} cores"
    )
    _status(run, plan.dry_run, f"Preparing native dependencies ({label})")

    # Phase 1: download sources + build host tools (once, shared across platforms)
    include_tools = ZEPHR_AGENT_TOOLS_DIR.is_dir()
    await _download_sentencepiece(run, plan.dry_run)
    if include_tools:
        await _build_host_protoc(plan.concurrency, run, plan.dry_run)
    else:
        _status(run, plan.dry_run, "  zephr_agent_tools absent, skipping protobuf tools")

    litert_platforms = [p for p in plan.platforms if p in ("macos", "ios", "android", "linux")]
    if litert_platforms:
        await _download_litert(run, plan.dry_run)
        prebuilt_targets = [
            (plat, arch)
            for plat in litert_platforms
            for arch in DEFAULT_ARCHES[plat]
            if (plat, arch) in LITERT_PREBUILT_ACCELERATOR_FILES
        ]
        await _download_litert_prebuilts(run, plan.dry_run, prebuilt_targets)
        await _build_litert_host_flatc(plan.concurrency, run, plan.dry_run)

    # Phase 2: configure + build per (platform, build_type, arch). Each target
    # already runs a parallel native build, so running targets sequentially keeps
    # logs readable and avoids oversubscribing the machine.
    for plat in plan.platforms:
        configure_dir, install_dir = _platform_dirs(plat)
        for bt in plan.build_types:
            for arch in DEFAULT_ARCHES[plat]:
                await _build_target(
                    plan, plat, configure_dir, install_dir,
                    bt, arch, plat in litert_platforms,
                    include_tools,
                    run, targets_to_stamp,
                )

    # Phase 3: proto codegen (needs host protoc from phase 1)
    await _run_proto_codegen(run, plan.dry_run)

    if not plan.dry_run:
        for plat, install_dir, bt, arch in targets_to_stamp:
            _write_target_stamp(plan, plat, install_dir, bt, arch)

    _status(run, plan.dry_run, "Done.")


async def _build_target(
    plan: PreparePlan,
    plat: str,
    configure_dir: Path,
    install_dir: Path,
    build_type: str,
    arch: str,
    include_litert: bool,
    include_tools: bool,
    run: Runner,
    targets_to_stamp: list[tuple[str, Path, str, str]],
) -> None:
    label = f"{plat}/{build_type}/{arch}"
    if not plan.clean and not plan.dry_run and _target_stamp_is_valid(
        plan, plat, install_dir, build_type, arch,
    ):
        _status(run, plan.dry_run, f"  native deps current ({label}), skipping")
        return
    if not plan.dry_run:
        log.trace(
            "native deps target needs build",
            platform=plat,
            build_type=build_type,
            arch=arch,
            clean=plan.clean,
        )

    await _build_sentencepiece(
        plat, configure_dir, install_dir,
        build_type, arch, plan.concurrency, plan.android_ndk_home,
        run, plan.dry_run,
    )
    if include_tools:
        await _build_protobuf_lite(
            plat, configure_dir, install_dir,
            build_type, arch, plan.concurrency, plan.android_ndk_home,
            run, plan.dry_run,
        )
    if include_litert:
        await _build_litert(
            plat, configure_dir, install_dir,
            build_type, arch, plan.concurrency, plan.android_ndk_home,
            run, plan.dry_run,
        )
    targets_to_stamp.append((plat, install_dir, build_type, arch))


# ---------------------------------------------------------------------------
# Protobuf (host protoc for .proto codegen)
# ---------------------------------------------------------------------------


async def _download_protobuf(run: Runner, dry_run: bool) -> None:
    dest = DOWNLOAD_DIR / "protobuf"
    if not dry_run and dest.exists():
        _status(run, dry_run, "  protobuf already downloaded, skipping")
        return
    _status(run, dry_run, "[protobuf] downloading...")
    await run([
        "git", "clone", PROTOBUF_REPO,
        "--branch", PROTOBUF_VERSION, "--depth", "1", str(dest),
    ])


async def _build_host_protoc(
    concurrency: int, run: Runner, dry_run: bool,
) -> Path:
    """Build protoc for the host. Must match sentencepiece's bundled protobuf-lite."""
    protoc_build = DOWNLOAD_DIR / "host_tools" / "protoc_build"
    protoc_bin = protoc_build / "protoc"
    if not dry_run and protoc_bin.exists():
        _status(run, dry_run, "  host protoc already built, skipping")
        return protoc_bin

    await _download_protobuf(run, dry_run)

    src = DOWNLOAD_DIR / "protobuf"
    _status(run, dry_run, "[protobuf] building host protoc...")

    await run(["mkdir", "-p", str(protoc_build)])
    await run([
        "cmake",
        "-S", str(src / "cmake"),
        "-B", str(protoc_build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-Dprotobuf_BUILD_TESTS=OFF",
        "-Dprotobuf_BUILD_EXAMPLES=OFF",
        "-Dprotobuf_BUILD_SHARED_LIBS=OFF",
        "-Dprotobuf_BUILD_PROTOC_BINARIES=ON",
        "-DCMAKE_CXX_STANDARD=17",
    ])
    await run([
        "cmake", "--build", str(protoc_build),
        "--target", "protoc", "-j", str(concurrency),
    ])

    _status(run, dry_run, f"[protobuf] host protoc built: {protoc_bin}")
    return protoc_bin


async def _run_proto_codegen(run: Runner, dry_run: bool) -> None:
    """Run protoc on tools-owned waypoints.proto into committed generated sources."""
    if not ZEPHR_AGENT_TOOLS_DIR.is_dir():
        _status(run, dry_run, "  zephr_agent_tools absent, skipping waypoints.proto codegen")
        return
    if not WAYPOINTS_PROTO.is_file():
        raise DeriveError(f"waypoints proto missing: {WAYPOINTS_PROTO}")
    protoc_bin = DOWNLOAD_DIR / "host_tools" / "protoc_build" / "protoc"
    proto_dir = WAYPOINTS_PROTO.parent
    generated_proto_dir = WAYPOINTS_GENERATED_PROTO_DIR
    output_cc = generated_proto_dir / "waypoints.pb.cc"

    if not dry_run and output_cc.exists():
        proto_mtime = WAYPOINTS_PROTO.stat().st_mtime
        cc_mtime = output_cc.stat().st_mtime
        if cc_mtime >= proto_mtime:
            _status(run, dry_run, "  waypoints.pb.cc up to date, skipping codegen")
            return

    _status(run, dry_run, "[protobuf] generating waypoints.pb.h/cc...")
    await run(["mkdir", "-p", str(generated_proto_dir)])

    await run([
        str(protoc_bin),
        f"--proto_path={proto_dir}",
        f"--cpp_out={generated_proto_dir}",
        str(WAYPOINTS_PROTO),
    ])


# ---------------------------------------------------------------------------
# Protobuf-lite (cross-compiled static lib for waypoints.pb.cc)
# ---------------------------------------------------------------------------


async def _build_protobuf_lite(
    plat: str,
    configure_dir: Path,
    install_dir: Path,
    build_type: str,
    arch: str,
    concurrency: int,
    android_ndk_home: Path | None,
    run: Runner,
    dry_run: bool,
) -> None:
    src = DOWNLOAD_DIR / "protobuf"
    build = configure_dir / build_type / arch / "protobuf-lite"
    install = install_dir / build_type / arch / "protobuf-lite"

    label = f"{plat}/{build_type}/{arch}"
    _status(run, dry_run, f"[protobuf-lite] configuring ({label})...")

    await run(["mkdir", "-p", str(build)])

    if plat == "ios":
        platform_args = [
            "-DCMAKE_SYSTEM_NAME=iOS",
            f"-DCMAKE_OSX_ARCHITECTURES={arch}",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=16.0",
        ]
    else:
        platform_args = _platform_cmake_args(plat, arch, android_ndk_home)

    await run([
        "cmake",
        "-S", str(src / "cmake"),
        "-B", str(build),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_INSTALL_PREFIX={install}",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-Dprotobuf_BUILD_TESTS=OFF",
        "-Dprotobuf_BUILD_EXAMPLES=OFF",
        "-Dprotobuf_BUILD_SHARED_LIBS=OFF",
        "-Dprotobuf_BUILD_PROTOC_BINARIES=OFF",
        "-Dprotobuf_WITH_ZLIB=OFF",
        "-DCMAKE_CXX_STANDARD=20",
        "-DCMAKE_C_FLAGS=-w",
        "-DCMAKE_CXX_FLAGS=-w",
    ] + platform_args)

    _status(run, dry_run, f"[protobuf-lite] building ({label})...")

    await run([
        "cmake", "--build", str(build),
        "--target", "libprotobuf-lite",
        "-j", str(concurrency),
    ])

    lib_dir = install / "lib"
    inc_dir = install / "include"
    await run(["mkdir", "-p", str(lib_dir), str(inc_dir)])
    # Protobuf cmake appends 'd' suffix for Debug builds
    src_lib = build / ("libprotobuf-lited.a" if build_type == "Debug" else "libprotobuf-lite.a")
    await run(["cp", str(src_lib), str(lib_dir / "libprotobuf-lite.a")])
    await run(["cp", "-r", str(src / "src" / "google"), str(inc_dir / "google")])

    _status(run, dry_run, f"[protobuf-lite] installed to {install}")


# ---------------------------------------------------------------------------
# Sentencepiece
# ---------------------------------------------------------------------------


async def _download_sentencepiece(run: Runner, dry_run: bool) -> None:
    dest = DOWNLOAD_DIR / "sentencepiece"
    if not dry_run and dest.exists():
        _status(run, dry_run, "  sentencepiece already downloaded, skipping")
        return
    _status(run, dry_run, "[sentencepiece] downloading...")
    await run([
        "git", "clone", "https://github.com/google/sentencepiece.git",
        "--branch", "v0.2.0", "--depth", "1", str(dest),
    ])


async def _build_sentencepiece(
    plat: str,
    configure_dir: Path,
    install_dir: Path,
    build_type: str,
    arch: str,
    concurrency: int,
    android_ndk_home: Path | None,
    run: Runner,
    dry_run: bool,
) -> None:
    src = DOWNLOAD_DIR / "sentencepiece"
    build = configure_dir / build_type / arch / "sentencepiece"
    install = install_dir / build_type / arch / "sentencepiece"

    label = f"{plat}/{build_type}/{arch}"
    _status(run, dry_run, f"[sentencepiece] configuring ({label})...")

    await run(["mkdir", "-p", str(build)])

    # iOS: use sentencepiece's bundled ios-cmake toolchain which defines
    # set_xcode_property (called unconditionally at configure time).
    if plat == "ios":
        platform_args = [
            f"-DCMAKE_TOOLCHAIN_FILE={src}/cmake/ios.toolchain.cmake",
            "-DPLATFORM=OS64",
            "-DDEPLOYMENT_TARGET=16.0",
        ]
    else:
        platform_args = _platform_cmake_args(plat, arch, android_ndk_home)

    await run([
        "cmake",
        "-S", str(src),
        "-B", str(build),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DCMAKE_INSTALL_PREFIX={install}",
    ] + platform_args + [
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DSPM_ENABLE_SHARED=OFF",
        "-DSPM_ENABLE_TCMALLOC=OFF",
        "-DSPM_USE_BUILTIN_PROTOBUF=ON",
        "-DSPM_NO_THREADLOCAL=OFF",
        "-DCMAKE_CXX_STANDARD=20",
        "-DCMAKE_C_FLAGS=-w",
        "-DCMAKE_CXX_FLAGS=-w",
    ])

    _status(run, dry_run, f"[sentencepiece] building ({label})...")

    # Build only the library targets — skip CLI tools (spm_encode, etc.)
    # which are useless on mobile and need -llog on android.
    await run([
        "cmake", "--build", str(build),
        "--target", "sentencepiece-static", "--target", "sentencepiece_train-static",
        "-j", str(concurrency),
    ])

    # Install libs + headers manually (cmake's install target builds everything)
    lib_dir = install / "lib"
    inc_dir = install / "include"
    await run(["mkdir", "-p", str(lib_dir), str(inc_dir)])
    for lib in ("libsentencepiece.a", "libsentencepiece_train.a"):
        await run(["cp", str(build / "src" / lib), str(lib_dir / lib)])
    for hdr in ("sentencepiece_processor.h", "sentencepiece_trainer.h"):
        await run(["cp", str(src / "src" / hdr), str(inc_dir / hdr)])

    _status(run, dry_run, f"[sentencepiece] installed to {install}")


# ---------------------------------------------------------------------------
# LiteRT
# ---------------------------------------------------------------------------


def _patch_litert(src: Path, *, verbose: bool = False) -> None:
    """Patch LiteRT source for the project build matrix.

    1. Remove -framework OpenGL (iOS has no OpenGL.framework; code
       already stubs out OpenGL on non-Android via LITERT_HAS_OPENGL_SUPPORT=0)
    2. Skip tools/ subdirectory (avoids building executables that fail install
       on iOS due to MACOSX_BUNDLE)
    3. Avoid compiling OpenCL runtime sources when OpenCL support is disabled
       (Linux WebGPU builds use the prebuilt accelerator and do not need CL/cl.h)
    4. Add a C API for attaching scoped external model weights. LiteRT already
       has this behavior in its C++ Options wrapper, but the Android shared
       library does not export the wrapper symbol. Keeping the field access
       inside LiteRT preserves the public boundary for our SDK code.
    5. Keep the runtime-context patch function around as a no-op for newer
       LiteRT sources and as a guard if this pin is temporarily moved back.
    6. Keep Abseil's command-line logging flags out of libLiteRt. Android apps
       load LiteRT as a shared library, not as a flag-owning executable, and
       Debug builds can abort during static flag initialization.
    7. Keep non-GoogleTensor vendor dispatches disabled unless explicitly
       enabled. LiteRT's vendor CMake currently treats a non-empty header-dir
       cache variable as permission to build Qualcomm/Samsung dispatches, but
       this script uses placeholder paths only to avoid SDK downloads.
    8. Mark the stock NPU dispatch accelerator as responsible for JIT
       compilation so mixed CPU/NPU environments do not apply the NPU delegate
       to CPU-only submodels.
    """
    # Remove OpenGL framework linkage
    for rel in ("litert/runtime/CMakeLists.txt", "litert/c/CMakeLists.txt"):
        path = src / rel
        text = path.read_text()
        new_text = text.replace(' "-framework OpenGL"', '')
        if new_text != text:
            path.write_text(new_text)
            if verbose:
                log.debug("patched litert cmake", file=rel, change="removed OpenGL framework")

    runtime_cmake = src / "litert" / "runtime" / "CMakeLists.txt"
    text = runtime_cmake.read_text()
    new_text = _normalize_litert_opencl_runtime_guard(text)
    if new_text != text:
        runtime_cmake.write_text(new_text)
        if verbose:
            log.debug(
                "patched litert cmake",
                file="litert/runtime/CMakeLists.txt",
                change="guarded OpenCL runtime sources",
            )

    # Keep vendors/ available so Android can build the open-source GoogleTensor
    # dispatch library. Skip tools/ only.
    main_cmake = src / "litert" / "CMakeLists.txt"
    original_text = main_cmake.read_text()
    text = _normalize_litert_enabled_subdirs(original_text, ("vendors",))
    new_text = _normalize_litert_skipped_subdirs(text, ("tools",))
    metal_snippet_start = new_text.find(
        "\nset(ZEPHR_LITERT_METAL_ACCELERATOR_SOURCE"
    )
    if metal_snippet_start >= 0:
        metal_snippet_end = new_text.find("\nendif()\n", metal_snippet_start)
        if metal_snippet_end >= 0:
            new_text = (
                new_text[:metal_snippet_start]
                + new_text[metal_snippet_end + len("\nendif()\n"):]
            )
    if new_text != original_text:
        main_cmake.write_text(new_text)
        if verbose:
            log.debug(
                "patched litert cmake",
                file="litert/CMakeLists.txt",
                change="enabled vendors/ and skipped tools",
            )

    vendors_cmake = src / "litert" / "vendors" / "CMakeLists.txt"
    text = vendors_cmake.read_text()
    replacements = {
        'option(LITERT_ENABLE_SAMSUNG "Enable Samsung NPU build" ON)':
            'option(LITERT_ENABLE_SAMSUNG "Enable Samsung NPU build" OFF)',
        "if(QAIRT_HEADERS_DIR AND NOT LITERT_ENABLE_QUALCOMM)":
            "if(FALSE AND QAIRT_HEADERS_DIR AND NOT LITERT_ENABLE_QUALCOMM)",
        "if(LITECORE_HEADERS_DIR AND NOT LITERT_ENABLE_SAMSUNG)":
            "if(FALSE AND LITECORE_HEADERS_DIR AND NOT LITERT_ENABLE_SAMSUNG)",
        'message(STATUS "Skipping Samsung dispatch: LITECORE_HEADERS_DIR not set")':
            'message(STATUS "Skipping Samsung dispatch: disabled")',
    }
    new_text = text
    for old, new in replacements.items():
        new_text = new_text.replace(old, new)
    if new_text != text:
        vendors_cmake.write_text(new_text)
        if verbose:
            log.debug(
                "patched litert cmake",
                file="litert/vendors/CMakeLists.txt",
                change="kept Qualcomm/Samsung dispatches opt-in",
            )

    c_cmake = src / "litert" / "c" / "CMakeLists.txt"
    text = c_cmake.read_text()
    new_text = text.replace("    absl::log_flags\n", "")
    if new_text != text:
        c_cmake.write_text(new_text)
        if verbose:
            log.debug(
                "patched litert cmake",
                file="litert/c/CMakeLists.txt",
                change="removed Abseil logging flag initializers from libLiteRt",
            )

    dispatch_accelerator = (
        src / "litert" / "runtime" / "accelerators" / "dispatch" / "dispatch_accelerator.cc"
    )
    text = dispatch_accelerator.read_text()
    method_anchor = """  // Stops collection of HW-specific metrics and report the collected metrics.
  static LiteRtStatus StopMetricsCollection(
      LiteRtRuntimeContext* runtime_context,
      LiteRtDelegateWrapper delegate_wrapper, LiteRtMetrics metrics) {
    LITERT_RETURN_IF_ERROR(delegate_wrapper != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Delegate pointer is null.";
    LITERT_RETURN_IF_ERROR(metrics != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Metrics pointer is null.";
    TfLiteOpaqueDelegate* delegate;
    runtime_context->unwrap_delegate(delegate_wrapper, &delegate);
    LITERT_LOG(LITERT_INFO, "Dispatch delegate stopped metrics collection.");
    return LiteRtDispatchDelegateStopMetricsCollection(delegate, metrics);
  }
"""
    method_replacement = method_anchor + """
  static LiteRtStatus IsTfLiteDelegateResponsibleForJitCompilation(
      LiteRtAccelerator accelerator, bool* does_jit_compilation) {
    LITERT_RETURN_IF_ERROR(accelerator != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "Accelerator handle is invalid.";
    LITERT_RETURN_IF_ERROR(does_jit_compilation != nullptr,
                           ErrorStatusBuilder::InvalidArgument())
        << "`does_jit_compilation` pointer is null.";
    *does_jit_compilation = true;
    return kLiteRtStatusOk;
  }
"""
    registration_anchor = """  LITERT_RETURN_IF_ERROR(litert::internal::SetAcceleratorBoilerplateFunctions<
                         litert::NpuAccelerator>(accelerator));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorStartMetricsCollection(
      accelerator.get(), litert::NpuAccelerator::StartMetricsCollection));
"""
    registration_replacement = """  LITERT_RETURN_IF_ERROR(litert::internal::SetAcceleratorBoilerplateFunctions<
                         litert::NpuAccelerator>(accelerator));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation(
          accelerator.get(),
          litert::NpuAccelerator::IsTfLiteDelegateResponsibleForJitCompilation));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorStartMetricsCollection(
      accelerator.get(), litert::NpuAccelerator::StartMetricsCollection));
"""
    new_text = text
    if "IsTfLiteDelegateResponsibleForJitCompilation" not in new_text:
        new_text = new_text.replace(method_anchor, method_replacement)
    if "LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation" not in new_text:
        new_text = new_text.replace(registration_anchor, registration_replacement)
    if new_text != text:
        dispatch_accelerator.write_text(new_text)
        if verbose:
            log.debug(
                "patched litert npu accelerator",
                file="litert/runtime/accelerators/dispatch/dispatch_accelerator.cc",
                change="marked NPU delegate responsible for JIT compilation",
            )


def _normalize_litert_opencl_runtime_guard(text: str) -> str:
    guard_line = 'if(NOT CMAKE_CXX_FLAGS MATCHES "LITERT_DISABLE_OPENCL_SUPPORT")'
    source_line = "target_sources(litert_runtime PRIVATE open_cl_memory.cc open_cl_sync.cc)"
    canonical = [
        f"{guard_line}\n",
        f"  {source_line}\n",
        "endif()\n",
    ]

    lines = text.splitlines(keepends=True)
    try:
        source_index = next(
            index for index, line in enumerate(lines)
            if line.strip() == source_line
        )
    except StopIteration:
        return text

    start = source_index
    guard_count = 0
    while start > 0 and lines[start - 1].strip() == guard_line:
        start -= 1
        guard_count += 1

    end = source_index + 1
    while guard_count > 0 and end < len(lines) and lines[end].strip() == "endif()":
        end += 1
        guard_count -= 1

    lines[start:end] = canonical
    return "".join(lines)


def _normalize_litert_skipped_subdirs(text: str, subdirs: tuple[str, ...]) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        for subdir in subdirs:
            if f"add_subdirectory({subdir})" in line:
                newline = "\n" if line.endswith("\n") else ""
                lines[index] = f"# add_subdirectory({subdir})  # patched out{newline}"
                break
    return "".join(lines)


def _normalize_litert_enabled_subdirs(text: str, subdirs: tuple[str, ...]) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        for subdir in subdirs:
            if stripped.startswith("# add_subdirectory(") and f"add_subdirectory({subdir})" in stripped:
                newline = "\n" if line.endswith("\n") else ""
                lines[index] = f"add_subdirectory({subdir}){newline}"
                break
    return "".join(lines)


@dataclass(frozen=True)
class LiteRtLfsPointer:
    oid: str
    size: int


def _parse_litert_lfs_pointer(text: str) -> LiteRtLfsPointer:
    oid: str | None = None
    size: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("oid sha256:"):
            oid = line.removeprefix("oid sha256:")
        elif line.startswith("size "):
            try:
                size = int(line.removeprefix("size "))
            except ValueError as exc:
                raise DeriveError(f"Invalid LiteRT LFS pointer size: {line}") from exc
    if not oid or size is None:
        raise DeriveError("Invalid LiteRT LFS pointer: missing oid or size")
    return LiteRtLfsPointer(oid=oid, size=size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _litert_prebuilt_pointer_path(prebuilt_dir: str, filename: str) -> Path:
    return (
        DOWNLOAD_DIR
        / "litert"
        / "litert"
        / "prebuilt"
        / prebuilt_dir
        / f"{filename}.lfs"
    )


def _litert_prebuilt_dest_path(prebuilt_dir: str, filename: str) -> Path:
    return DOWNLOAD_DIR / LITERT_PREBUILT_DOWNLOAD_DIRNAME / prebuilt_dir / filename


async def _download_litert_prebuilts(
    run: Runner, dry_run: bool, targets: list[tuple[str, str]],
) -> None:
    requested: dict[tuple[str, str], tuple[LiteRtLfsPointer, Path]] = {}
    for plat, arch in sorted(set(targets)):
        prebuilt_dir = LITERT_PREBUILT_DIRS.get((plat, arch))
        filenames = LITERT_PREBUILT_ACCELERATOR_FILES.get((plat, arch), ())
        if not prebuilt_dir or not filenames:
            continue
        for filename in filenames:
            pointer_path = _litert_prebuilt_pointer_path(prebuilt_dir, filename)
            dest_path = _litert_prebuilt_dest_path(prebuilt_dir, filename)
            if dry_run:
                print(f"fetch LiteRT LFS {pointer_path} -> {dest_path}")
                continue
            if not pointer_path.is_file():
                raise DeriveError(
                    f"LiteRT prebuilt pointer missing: {pointer_path}. "
                    f"Check LITERT_SOURCE_REF={LITERT_SOURCE_REF}."
                )
            pointer = _parse_litert_lfs_pointer(
                pointer_path.read_text(encoding="utf-8")
            )
            if (
                dest_path.is_file()
                and dest_path.stat().st_size == pointer.size
                and _sha256_file(dest_path) == pointer.oid
            ):
                continue
            requested[(pointer.oid, filename)] = (pointer, dest_path)

    if dry_run:
        return
    if not requested:
        _status(run, dry_run, "  litert repo-pinned prebuilts already downloaded, skipping")
        return

    _status(run, dry_run, "[litert] downloading repo-pinned accelerator prebuilts...")
    dest_root = DOWNLOAD_DIR / LITERT_PREBUILT_DOWNLOAD_DIRNAME
    dest_root.mkdir(parents=True, exist_ok=True)
    request_path = dest_root / "lfs_batch_request.json"
    response_path = dest_root / "lfs_batch_response.json"
    objects = [
        {"oid": pointer.oid, "size": pointer.size}
        for pointer, _ in requested.values()
    ]
    request_path.write_text(
        json.dumps(
            {
                "operation": "download",
                "transfers": ["basic"],
                "objects": objects,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    await run([
        "curl",
        "-s",
        "-X",
        "POST",
        "-H",
        "Accept: application/vnd.git-lfs+json",
        "-H",
        "Content-Type: application/vnd.git-lfs+json",
        "-d",
        f"@{request_path}",
        "-o",
        str(response_path),
        LITERT_LFS_BATCH_URL,
    ])
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeriveError(f"Could not read LiteRT LFS response at {response_path}") from exc
    response_objects = response.get("objects")
    if not isinstance(response_objects, list):
        raise DeriveError(f"Invalid LiteRT LFS response at {response_path}")
    download_urls: dict[str, str] = {}
    for response_object in response_objects:
        if not isinstance(response_object, dict):
            continue
        oid = response_object.get("oid")
        actions = response_object.get("actions")
        download = actions.get("download") if isinstance(actions, dict) else None
        href = download.get("href") if isinstance(download, dict) else None
        if isinstance(oid, str) and isinstance(href, str):
            download_urls[oid] = href
    for (oid, filename), (pointer, dest_path) in requested.items():
        url = download_urls.get(oid)
        if not url:
            raise DeriveError(f"LiteRT LFS response did not include download URL for {oid}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        await run(["curl", "-sL", "-o", str(dest_path), url])
        actual_size = dest_path.stat().st_size
        actual_oid = _sha256_file(dest_path)
        if actual_size != pointer.size or actual_oid != pointer.oid:
            dest_path.unlink(missing_ok=True)
            raise DeriveError(
                f"LiteRT LFS download verification failed for {filename}: "
                f"expected sha256={pointer.oid} size={pointer.size}, "
                f"got sha256={actual_oid} size={actual_size}"
            )


def _copy_replacing(src: Path, dst: Path) -> None:
    """Copy a file after removing stale install output that may be read-only."""
    try:
        dst.unlink()
    except FileNotFoundError:
        pass
    shutil.copy2(src, dst)


def _install_litert_prebuilts(
    plat: str, arch: str, install_dir: Path, *, verbose: bool = False,
) -> None:
    """Copy repo-pinned prebuilt accelerator libs into the install dir."""
    lib_dir = install_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)

    prebuilt_name = LITERT_PREBUILT_DIRS.get((plat, arch))
    if prebuilt_name:
        src_dir = DOWNLOAD_DIR / LITERT_PREBUILT_DOWNLOAD_DIRNAME / prebuilt_name
        if src_dir.exists():
            for f in src_dir.iterdir():
                if (
                    f.is_file()
                    and f.suffix in (".dylib", ".so")
                    and f.name not in ("libLiteRt.so", "libLiteRt.dylib")
                ):
                    _copy_replacing(f, lib_dir / f.name)
                    (lib_dir / f.name).chmod(0o755)
                    if verbose:
                        log.debug("installed prebuilt", file=f.name)
                elif f.is_dir() and f.suffix == ".framework":
                    dest = lib_dir / f.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(f, dest)
                    if verbose:
                        log.debug("installed prebuilt", file=f.name)


async def _install_apple_dsyms(
    run: Runner,
    dry_run: bool,
    lib_dir: Path,
    dylib_names: list[str],
) -> None:
    """Install matching dSYM companions for Apple dynamic libraries.

    For prebuilt dylibs this cannot manufacture source-level debug info; it only
    preserves whatever symbols/debug sections the binary already contains.
    """
    for dylib_name in dylib_names:
        dylib_path = lib_dir / dylib_name
        if not dry_run and not dylib_path.exists():
            continue
        await run([
            "dsymutil",
            str(dylib_path),
            "-o",
            str(lib_dir / f"{dylib_name}.dSYM"),
        ])


async def _set_apple_install_names(
    run: Runner,
    dry_run: bool,
    lib_dir: Path,
    install_names: dict[str, str],
) -> None:
    """Point Apple dylib IDs at their embedded framework locations."""
    for dylib_name, install_name in install_names.items():
        dylib_path = lib_dir / dylib_name
        if not dry_run and not dylib_path.exists():
            continue
        await run(["install_name_tool", "-id", install_name, str(dylib_path)])


async def _change_apple_dylib_dependencies(
    run: Runner,
    dry_run: bool,
    lib_dir: Path,
    changes: dict[str, dict[str, str]],
) -> None:
    """Rewrite Apple dylib load commands after moving libs into frameworks."""
    for dylib_name, dylib_changes in changes.items():
        dylib_path = lib_dir / dylib_name
        if not dry_run and not dylib_path.exists():
            continue
        for old_name, new_name in dylib_changes.items():
            await run([
                "install_name_tool",
                "-change",
                old_name,
                new_name,
                str(dylib_path),
            ])


def _apple_framework_info_plist(
    *,
    executable: str,
    bundle_id: str,
    bundle_name: str,
    minimum_os_version: str,
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>{executable}</string>
  <key>CFBundleIdentifier</key>
  <string>{bundle_id}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>{bundle_name}</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleSupportedPlatforms</key>
  <array>
    <string>iPhoneOS</string>
  </array>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>MinimumOSVersion</key>
  <string>{minimum_os_version}</string>
</dict>
</plist>
"""


def _install_apple_dynamic_framework(
    lib_dir: Path,
    *,
    dylib_name: str,
    framework_name: str,
    bundle_id: str,
    minimum_os_version: str,
) -> Path:
    frameworks_dir = lib_dir / "frameworks"
    framework_dir = frameworks_dir / framework_name
    if framework_dir.exists():
        shutil.rmtree(framework_dir)
    framework_dir.mkdir(parents=True, exist_ok=True)
    executable = framework_name.removesuffix(".framework")
    shutil.copy2(lib_dir / dylib_name, framework_dir / executable)
    if executable != dylib_name:
        os.symlink(executable, framework_dir / dylib_name)
    (framework_dir / "Info.plist").write_text(
        _apple_framework_info_plist(
            executable=executable,
            bundle_id=bundle_id,
            bundle_name=executable,
            minimum_os_version=minimum_os_version,
        ),
        encoding="utf-8",
    )
    return framework_dir


async def _create_apple_xcframework(
    run: Runner,
    dry_run: bool,
    *,
    framework_dir: Path,
    output_path: Path,
) -> None:
    if dry_run:
        print(f"rm -rf {output_path}")
    else:
        shutil.rmtree(output_path, ignore_errors=True)
    await run([
        "xcodebuild",
        "-create-xcframework",
        "-framework",
        str(framework_dir),
        "-output",
        str(output_path),
    ])


async def _install_litert_apple_frameworks(
    run: Runner,
    dry_run: bool,
    lib_dir: Path,
) -> None:
    """Create one-framework-per-dylib bundles and xcframework wrappers.

    iOS dynamic libraries need framework bundles for App Store packaging, while
    LiteRT's GPU registry still expects the dylib basename inside the runtime
    directory. These bundles use normal framework executable names and add
    `.dylib` symlink aliases for LiteRT's internal dlopen path.
    """
    frameworks = [
        (
            "libLiteRt.dylib",
            "libLiteRt.framework",
            "xyz.zephr.litert",
        ),
        (
            "libLiteRtMetalAccelerator.dylib",
            "libLiteRtMetalAccelerator.framework",
            "xyz.zephr.litert.metal",
        ),
    ]

    xcframework_dir = lib_dir / "xcframeworks"
    if dry_run:
        print(f"mkdir -p {xcframework_dir}")
    else:
        xcframework_dir.mkdir(parents=True, exist_ok=True)

    for dylib_name, framework_name, bundle_id in frameworks:
        dylib_path = lib_dir / dylib_name
        if not dry_run and not dylib_path.exists():
            continue
        if dry_run:
            framework_dir = lib_dir / "frameworks" / framework_name
            print(f"install framework {dylib_path} -> {framework_dir}")
        else:
            framework_dir = _install_apple_dynamic_framework(
                lib_dir,
                dylib_name=dylib_name,
                framework_name=framework_name,
                bundle_id=bundle_id,
                minimum_os_version=IOS_DEPLOYMENT_TARGET,
            )
        await _create_apple_xcframework(
            run,
            dry_run,
            framework_dir=framework_dir,
            output_path=xcframework_dir / f"{framework_name.removesuffix('.framework')}.xcframework",
        )


async def _download_litert(run: Runner, dry_run: bool) -> None:
    dest = DOWNLOAD_DIR / "litert"
    if not dry_run and dest.exists():
        result = await anyio.run_process(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=False,
        )
        current_ref = (
            result.stdout.decode(errors="replace").strip()
            if result.returncode == 0 and result.stdout
            else ""
        )
        if current_ref != LITERT_SOURCE_REF:
            status = await anyio.run_process(
                ["git", "-C", str(dest), "status", "--porcelain"],
                check=False,
            )
            dirty = status.returncode != 0 or bool(status.stdout.strip())
            if dirty:
                raise DeriveError(
                    f"Existing LiteRT checkout at {dest} is {current_ref or 'unknown'}, "
                    f"but this build is pinned to {LITERT_SOURCE_REF}. It also has local "
                    "changes, so prepare_native_deps will not replace it automatically. "
                    "Move it aside or clean it before rebuilding native deps."
                )
            shutil.rmtree(dest)
            _status(run, dry_run, f"[litert] replacing checkout with {LITERT_SOURCE_REF[:12]}...")
            await run(["git", "clone", "--no-checkout", LITERT_REPO, str(dest)])
            await run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", LITERT_SOURCE_REF])
            await run(["git", "-C", str(dest), "checkout", "--detach", LITERT_SOURCE_REF])
        else:
            _status(run, dry_run, "  litert already downloaded, skipping")
    else:
        _status(run, dry_run, f"[litert] downloading {LITERT_SOURCE_REF[:12]}...")
        await run(["git", "clone", "--no-checkout", LITERT_REPO, str(dest)])
        await run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", LITERT_SOURCE_REF])
        await run(["git", "-C", str(dest), "checkout", "--detach", LITERT_SOURCE_REF])
    if not dry_run:
        _patch_litert(dest, verbose=run.verbose)


async def _build_litert_host_flatc(
    concurrency: int, run: Runner, dry_run: bool,
) -> Path:
    """Build flatc for the host so cross-compilation doesn't need nested cmake."""
    flatc_build = DOWNLOAD_DIR / "host_tools" / "flatc_build"
    flatc_bin = flatc_build / "_deps" / "flatbuffers-build" / "flatc"
    if not dry_run and flatc_bin.exists():
        _status(run, dry_run, "  host flatc already built, skipping")
        return flatc_bin.parent

    _status(run, dry_run, "[litert] building host flatc...")
    await run(["mkdir", "-p", str(flatc_build)])

    cmake_lists = flatc_build / "CMakeLists.txt"
    if not dry_run:
        cmake_lists.write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(HostFlatc LANGUAGES C CXX)\n"
            "include(FetchContent)\n"
            "FetchContent_Declare(flatbuffers\n"
            "  GIT_REPOSITORY https://github.com/google/flatbuffers.git\n"
            "  GIT_TAG v25.9.23)\n"
            "set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)\n"
            "FetchContent_MakeAvailable(flatbuffers)\n"
        )

    await run([
        "cmake", "-S", str(flatc_build), "-B", str(flatc_build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ])
    await run([
        "cmake", "--build", str(flatc_build),
        "--target", "flatc", "-j", str(concurrency),
    ])
    return flatc_bin.parent



async def _build_litert(
    plat: str,
    configure_dir: Path,
    install_dir: Path,
    build_type: str,
    arch: str,
    concurrency: int,
    android_ndk_home: Path | None,
    run: Runner,
    dry_run: bool,
) -> None:
    src = DOWNLOAD_DIR / "litert"
    build = configure_dir / build_type / arch / "litert"
    install = install_dir / build_type / arch / "litert"

    label = f"{plat}/{build_type}/{arch}"
    _status(run, dry_run, f"[litert] configuring ({label})...")

    await run(["mkdir", "-p", str(build)])

    platform_args = _platform_cmake_args(plat, arch, android_ndk_home)
    host_tools_dir = DOWNLOAD_DIR / "host_tools" / "flatc_build" / "_deps" / "flatbuffers-build"
    empty_vendor_headers_dir = build / "empty_vendor_headers"
    await run(["mkdir", "-p", str(empty_vendor_headers_dir)])

    c_flags = ""
    cxx_flags = "-fpermissive"
    objcxx_flags = ""
    if plat in ("macos", "ios"):
        if build_type.lower() == "release":
            c_flags = "-gline-tables-only"
            cxx_flags += " -gline-tables-only"
            objcxx_flags = "-gline-tables-only"
        cxx_flags += " -DLITERT_DISABLE_OPENGL_SUPPORT"
    elif plat == "linux":
        cxx_flags += (
            " -DLITERT_DISABLE_OPENCL_SUPPORT"
            " -DLITERT_DISABLE_OPENGL_SUPPORT"
            " -DLITERT_DISABLE_VULKAN_SUPPORT"
        )

    cmake_args = [
        "cmake",
        "-S", str(src / "litert"),
        "-B", str(build),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_INSTALL_PREFIX={install}",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DCMAKE_CXX_STANDARD=20",
        f"-DCMAKE_C_FLAGS={c_flags}",
        f"-DCMAKE_CXX_FLAGS={cxx_flags}",
        f"-DCMAKE_OBJCXX_FLAGS={objcxx_flags}",
        "-DLITERT_AUTO_BUILD_TFLITE=ON",
        "-DLITERT_ENABLE_GPU=ON",
        f"-DTFLITE_ENABLE_GPU={'OFF' if plat == 'linux' else 'ON'}",
        "-DTFLITE_ENABLE_METAL=OFF",
        f"-DLITERT_ENABLE_NPU={'ON' if plat == 'android' else 'OFF'}",
        "-DLITERT_BUILD_TESTS=OFF",
        "-DLITERT_ENABLE_QUALCOMM=OFF",
        "-DLITERT_ENABLE_SAMSUNG=OFF",
        f"-DNEUROPILOT_HEADERS_DIR={empty_vendor_headers_dir}",
        f"-DQAIRT_HEADERS_DIR={empty_vendor_headers_dir}",
        f"-DLITECORE_HEADERS_DIR={empty_vendor_headers_dir}",
        "-DFLATBUFFERS_BUILD_FLATC=OFF",
        f"-DTFLITE_HOST_TOOLS_DIR={host_tools_dir}",
    ]
    if plat in ("macos", "ios"):
        cmake_args.append("-DCMAKE_MACOSX_BUNDLE=OFF")

    await run(cmake_args + platform_args)

    _status(run, dry_run, f"[litert] building ({label})...")

    await run([
        "cmake", "--build", str(build),
        "--target", "litert_runtime_c_api_shared_lib",
        "-j", str(concurrency),
    ])
    if plat == "android" and arch == "arm64-v8a":
        await run([
            "cmake", "--build", str(build),
            "--target", "dispatch_api_GoogleTensor_so",
            "-j", str(concurrency),
        ])

    # Install: copy the dylib and public headers
    lib_dir = install / "lib"
    inc_dir = install / "include" / "litert"
    await run(["mkdir", "-p", str(lib_dir), str(inc_dir)])

    # Find the built dylib/so (target is in litert/c/)
    if plat in ("macos", "ios"):
        lib_name = "libLiteRt.dylib"
    else:
        lib_name = "libLiteRt.so"

    await run(["cp", str(build / "c" / lib_name), str(lib_dir / lib_name)])
    if plat == "android" and arch == "arm64-v8a":
        await run([
            "cp",
            str(build / "vendors" / "libLiteRtDispatch_GoogleTensor.so"),
            str(lib_dir / "libLiteRtDispatch_GoogleTensor.so"),
        ])

    # Copy C API headers
    c_hdr_src = src / "litert" / "c"
    for hdr in c_hdr_src.glob("*.h"):
        if not dry_run:
            shutil.copy2(hdr, inc_dir / hdr.name)

    # Install prebuilt accelerator libs (Metal, OpenCL/GL, etc.)
    if not dry_run:
        _install_litert_prebuilts(plat, arch, install, verbose=run.verbose)

    if plat in ("macos", "ios"):
        if plat == "ios":
            install_names = {
                "libLiteRt.dylib": "@rpath/libLiteRt.framework/libLiteRt.dylib",
                "libLiteRtMetalAccelerator.dylib": "@rpath/libLiteRtMetalAccelerator.framework/libLiteRtMetalAccelerator.dylib",
            }
        else:
            install_names = {
                "libLiteRt.dylib": "@rpath/libLiteRt.dylib",
                "libLiteRtMetalAccelerator.dylib": "@rpath/libLiteRtMetalAccelerator.dylib",
            }
        await _set_apple_install_names(
            run,
            dry_run,
            lib_dir,
            install_names,
        )
        await _install_apple_dsyms(
            run,
            dry_run,
            lib_dir,
            [
                "libLiteRt.dylib",
                "libLiteRtMetalAccelerator.dylib",
            ],
        )
        if plat == "ios":
            await _install_litert_apple_frameworks(
                run,
                dry_run,
                lib_dir,
            )

    _status(run, dry_run, f"[litert] installed to {install}")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from clis.tinylib.tinycli import Pipeline

    pipeline = Pipeline("prepare_native_deps")
    pipeline.stage("prepare", plan_prepare, run_prepare)
    pipeline.main()
