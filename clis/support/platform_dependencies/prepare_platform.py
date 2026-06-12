"""Prepare platform-specific development dependencies.

Installs tooling and generates project files needed before building.

Usage:
    prepare_platform                          # all applicable platforms
    prepare_platform --platform ios           # iOS only
    prepare_platform --platform android       # Android only
    prepare_platform --dry_run                # show commands without executing
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from clis.support.native_dependencies.process_runner import Runner
from clis.support.paths import REPO_ROOT
from clis.support.platform_dependencies.android_config import android_platform_config
from clis.tinylib.tinycli import DeriveError
from clis.tinylib.tinylog import log


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

XCODE_PROJECTS: list[tuple[str, Path]] = [
    ("ZephrAgentRuntime", REPO_ROOT / "apple"),
]

ANDROID_CONFIG = android_platform_config()
ANDROID_SDK_PACKAGES = [
    "platform-tools",
    f"platforms;android-{ANDROID_CONFIG.compile_sdk}",
    f"build-tools;{ANDROID_CONFIG.build_tools}",
    f"cmake;{ANDROID_CONFIG.cmake}",
    f"ndk;{ANDROID_CONFIG.ndk}",
]
ANDROID_PROJECTS: list[tuple[str, Path]] = [
    ("Zephr Android", REPO_ROOT / "android"),
]
ANDROID_STUDIO_VOLATILE_STATE_PATHS = (
    REPO_ROOT / "android" / ".idea" / "workspace.xml",
    REPO_ROOT / "android" / ".idea" / "deploymentTargetSelector.xml",
    REPO_ROOT / "android" / ".idea" / "caches",
)
NANOBIND_GENERATED_ARTIFACT_PATTERNS = (
    "zephr_agent_runtime_nanobind*.so",
    "zephr_agent_tools_nanobind*.so",
    "tinyllm_nanobind*.so",
)
NANOBIND_GENERATED_ARTIFACT_ROOTS = (
    # Current generated extension locations.
    REPO_ROOT / "clis" / "zephr_agent",
    REPO_ROOT / "clis" / "zephr_agent_tools",
    # Legacy locations from before Python wrappers moved into public/private packages.
    REPO_ROOT / "clis" / "tinylib" / "tinyllm",
    REPO_ROOT / "tinylib" / "tinyllm",
)


# ---------------------------------------------------------------------------
# Generated local artifacts
# ---------------------------------------------------------------------------


def _cleanup_generated_nanobind_artifacts(dry_run: bool) -> None:
    for root in NANOBIND_GENERATED_ARTIFACT_ROOTS:
        if not root.exists():
            continue
        for pattern in NANOBIND_GENERATED_ARTIFACT_PATTERNS:
            for path in sorted(root.rglob(pattern)):
                if not path.is_file():
                    continue
                log.info("removing generated nanobind artifact", path=str(path))
                if dry_run:
                    print(f"remove {path}")
                else:
                    path.unlink()


def _cleanup_android_studio_volatile_state(dry_run: bool) -> None:
    for path in ANDROID_STUDIO_VOLATILE_STATE_PATHS:
        if not path.exists():
            continue
        log.info("removing Android Studio volatile state", path=str(path))
        if dry_run:
            print(f"remove {path}")
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


# ---------------------------------------------------------------------------
# iOS
# ---------------------------------------------------------------------------


def _check_macos() -> None:
    if platform.system() != "Darwin":
        raise DeriveError("iOS platform dependencies can only be prepared on macOS")


def _has_brew_package(name: str) -> bool:
    return shutil.which(name) is not None


async def _prepare_ios(run: Runner, dry_run: bool) -> None:
    _check_macos()

    # xcodegen
    if not dry_run and _has_brew_package("xcodegen"):
        log.info("xcodegen already installed")
    else:
        log.info("installing xcodegen")
        await run(["brew", "install", "xcodegen"])

    # Generate Xcode projects
    for name, project_dir in XCODE_PROJECTS:
        log.info(f"generating {name}", path=str(project_dir))
        await run(["xcodegen", "generate"], cwd=project_dir)


# ---------------------------------------------------------------------------
# Android
# ---------------------------------------------------------------------------


def _default_android_sdk_root() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Android" / "sdk"
    return Path.home() / "Android" / "Sdk"


def _android_sdk_root() -> Path:
    return Path(
        os.environ.get("ANDROID_HOME")
        or os.environ.get("ANDROID_SDK_ROOT")
        or _default_android_sdk_root()
    ).expanduser()


def _android_sdkmanager(sdk_root: Path) -> Path:
    candidates = [
        sdk_root / "cmdline-tools" / "latest" / "bin" / "sdkmanager",
        Path(shutil.which("sdkmanager") or ""),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise DeriveError(
        "Android sdkmanager not found. Install Android command line tools "
        f"under {sdk_root}/cmdline-tools/latest or put sdkmanager on PATH."
    )


def _android_package_path(sdk_root: Path, package: str) -> Path:
    return sdk_root.joinpath(*package.split(";"))


def _path_is_in_nix_store(path: Path) -> bool:
    return len(path.parts) >= 3 and path.parts[1:3] == ("nix", "store")


def _android_ndk_home(sdk_root: Path) -> Path:
    explicit = os.environ.get("ANDROID_NDK_HOME")
    if explicit:
        return Path(explicit).expanduser()

    versioned = _android_package_path(sdk_root, f"ndk;{ANDROID_CONFIG.ndk}")
    if (versioned / "build" / "cmake" / "android.toolchain.cmake").is_file():
        return versioned

    ndk_bundle = sdk_root / "ndk-bundle"
    if (ndk_bundle / "build" / "cmake" / "android.toolchain.cmake").is_file():
        return ndk_bundle

    return versioned


def _android_package_installed(sdk_root: Path, package: str) -> bool:
    path = _android_package_path(sdk_root, package)
    if package == "platform-tools":
        return (path / "adb").is_file()
    if package.startswith("cmake;"):
        return (path / "bin" / "cmake").is_file()
    if package.startswith("ndk;"):
        return (
            _android_ndk_home(sdk_root)
            / "build"
            / "cmake"
            / "android.toolchain.cmake"
        ).is_file()
    return path.exists()


def _write_android_local_properties(
    project_dir: Path,
    sdk_root: Path,
    dry_run: bool,
) -> None:
    local_properties = project_dir / "local.properties"
    contents = f"sdk.dir={sdk_root}\n"
    if dry_run:
        print(f"write {local_properties}")
        print(contents, end="")
        return
    local_properties.write_text(contents, encoding="utf-8")


async def _prepare_android(run: Runner, dry_run: bool) -> None:
    _cleanup_android_studio_volatile_state(dry_run)

    sdk_root = _android_sdk_root()
    ndk_home = _android_ndk_home(sdk_root)

    missing = [
        package
        for package in ANDROID_SDK_PACKAGES
        if not _android_package_installed(sdk_root, package)
    ]
    if missing:
        if _path_is_in_nix_store(sdk_root):
            raise DeriveError(
                "Android SDK is provided by Nix and cannot be mutated. "
                "Update flake.nix to include missing packages: "
                + ", ".join(missing)
            )
        sdkmanager = _android_sdkmanager(sdk_root)
        log.info(
            "installing android sdk packages",
            sdk_root=str(sdk_root),
            packages=", ".join(missing),
        )
        await run(
            [str(sdkmanager), f"--sdk_root={sdk_root}", *missing],
            env={"ANDROID_HOME": str(sdk_root), "ANDROID_SDK_ROOT": str(sdk_root)},
        )
    else:
        log.info("android sdk packages already installed", sdk_root=str(sdk_root))

    if not dry_run and not (ndk_home / "build" / "cmake" / "android.toolchain.cmake").is_file():
        raise DeriveError(
            f"Android NDK {ANDROID_CONFIG.ndk} did not install correctly: {ndk_home}"
        )

    for name, project_dir in ANDROID_PROJECTS:
        if not project_dir.is_dir():
            log.debug(f"skipping {name} local.properties; project does not exist yet",
                      path=str(project_dir))
            continue
        log.info(f"writing {name} local.properties", path=str(project_dir))
        _write_android_local_properties(project_dir, sdk_root, dry_run)


# ---------------------------------------------------------------------------
# Plan / Run
# ---------------------------------------------------------------------------

Platform = Literal["ios", "android"]

PLATFORM_RUNNERS: dict[str, object] = {
    "ios": _prepare_ios,
    "android": _prepare_android,
}


@dataclass
class PreparePlan:
    platforms: list[str]
    verbose: bool
    dry_run: bool


def plan_prepare(
    platform: Annotated[list[Platform] | None, "target platforms (default: all applicable)"] = None,
    verbose: Annotated[bool, "show command output"] = False,
    dry_run: Annotated[bool, "print commands without executing"] = False,
) -> PreparePlan:
    """Install tooling and generate project files for target platforms."""
    if platform is not None:
        platforms: list[str] = list(platform)
    else:
        # Default: all platforms applicable to this machine
        platforms = ["ios"] if _has_brew_package("brew") else []
        platforms.append("android")

    return PreparePlan(
        platforms=platforms,
        verbose=verbose,
        dry_run=dry_run,
    )


async def run_prepare(plan: PreparePlan) -> None:
    run = Runner(verbose=plan.verbose, dry_run=plan.dry_run)

    _cleanup_generated_nanobind_artifacts(plan.dry_run)

    for plat in plan.platforms:
        log.info(f"preparing {plat}")
        if plat == "ios":
            await _prepare_ios(run, plan.dry_run)
        elif plat == "android":
            await _prepare_android(run, plan.dry_run)
        else:
            log.warn(f"unknown platform: {plat}")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from clis.tinylib.tinycli import Pipeline

    pipeline = Pipeline("prepare_platform")
    pipeline.stage("prepare", plan_prepare, run_prepare)
    pipeline.main()
