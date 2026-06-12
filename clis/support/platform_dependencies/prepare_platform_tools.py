"""Prepare machine-level platform tooling.

Installs or verifies SDK/tool packages needed by platform and native builds.

Usage:
    prepare_platform_tools
    prepare_platform_tools --platform android
    prepare_platform_tools --platform ios
    prepare_platform_tools --dry_run
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from clis.support.native_dependencies.process_runner import Runner
from clis.support.platform_dependencies.prepare_platform import (
    ANDROID_CONFIG,
    ANDROID_SDK_PACKAGES,
    _android_ndk_home,
    _android_package_installed,
    _android_sdk_root,
    _android_sdkmanager,
    _check_macos,
    _has_brew_package,
    _path_is_in_nix_store,
)
from clis.tinylib.tinycli import DeriveError
from clis.tinylib.tinylog import log


Platform = Literal["ios", "android"]


@dataclass
class PreparePlatformToolsPlan:
    platforms: list[str]
    verbose: bool
    dry_run: bool
    android_sdk_root: Path | None
    android_ndk_home: Path | None
    android_packages: list[str]


def _default_platforms() -> list[str]:
    platforms = ["ios"] if _has_brew_package("brew") else []
    platforms.append("android")
    return platforms


def plan_prepare_platform_tools(
    platform: Annotated[list[Platform] | None, "platform tools to prepare (default: all applicable)"] = None,
    verbose: Annotated[bool, "show command output"] = False,
    dry_run: Annotated[bool, "print commands without executing"] = False,
) -> PreparePlatformToolsPlan:
    """Install or verify machine-level platform SDKs and tools."""
    platforms = list(platform) if platform is not None else _default_platforms()
    sdk_root: Path | None = None
    ndk_home: Path | None = None
    missing: list[str] = []
    if "android" in platforms:
        sdk_root = _android_sdk_root()
        ndk_home = _android_ndk_home(sdk_root)
        missing = [
            package
            for package in ANDROID_SDK_PACKAGES
            if not _android_package_installed(sdk_root, package)
        ]
    return PreparePlatformToolsPlan(
        platforms=platforms,
        verbose=verbose,
        dry_run=dry_run,
        android_sdk_root=sdk_root,
        android_ndk_home=ndk_home,
        android_packages=missing,
    )


async def run_prepare_platform_tools(plan: PreparePlatformToolsPlan) -> None:
    run = Runner(verbose=plan.verbose, dry_run=plan.dry_run)

    if "ios" in plan.platforms:
        _check_macos()
        if shutil.which("xcodegen"):
            log.info("xcodegen already installed")
        else:
            if platform.system() != "Darwin":
                raise DeriveError("xcodegen can only be installed automatically on macOS")
            log.info("installing xcodegen")
            await run(["brew", "install", "xcodegen"])

    if "android" in plan.platforms:
        sdk_root = plan.android_sdk_root or _android_sdk_root()
        missing = plan.android_packages
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

        ndk_home = plan.android_ndk_home or _android_ndk_home(sdk_root)
        if not plan.dry_run and not (ndk_home / "build" / "cmake" / "android.toolchain.cmake").is_file():
            raise DeriveError(
                f"Android NDK {ANDROID_CONFIG.ndk} is not ready at {ndk_home}"
            )


def main() -> None:
    from clis.tinylib.tinycli import Pipeline

    pipeline = Pipeline("prepare_platform_tools")
    pipeline.stage("tools", plan_prepare_platform_tools, run_prepare_platform_tools)
    pipeline.main()
