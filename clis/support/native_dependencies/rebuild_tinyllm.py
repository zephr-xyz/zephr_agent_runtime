"""Build the Zephr Agent Python native extensions and host dependencies.

Combined pipeline: prepare Python + host native deps -> rebuild nanobind extensions.

Usage:
    rebuild_tinyllm
    rebuild_tinyllm --CMAKE_BUILD_TYPE Debug
    rebuild_tinyllm --start_from rebuild
    rebuild_tinyllm --plan
    rebuild_tinyllm --dry_run
    rebuild_tinyllm --clean
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import psutil

from clis.support.native_dependencies.prepare_native_deps import (
    PreparePlan,
    plan_prepare,
    run_prepare,
)
from clis.support.native_dependencies.locator import (
    require_native_deps_ready,
    write_native_deps_bridges,
)
from clis.support.native_dependencies.process_runner import Runner
from clis.support.paths import REPO_ROOT
from clis.tinylib.tinylog import log


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CMAKELISTS_DIR = REPO_ROOT / "clis" / "support" / "nanobind"
BUILD_DIR = REPO_ROOT / "1_build" / "cmake"
INSTALL_DIRS = (
    REPO_ROOT / "clis" / "zephr_agent_runtime" / "nanobind",
    REPO_ROOT / "clis" / "zephr_agent_tools" / "nanobind",
)


# ---------------------------------------------------------------------------
# Prepare stage
# ---------------------------------------------------------------------------


def _host_native_platform() -> Literal["macos", "linux"]:
    if platform.system() == "Darwin":
        return "macos"
    if platform.system() == "Linux":
        return "linux"
    raise RuntimeError(f"Unsupported host platform: {platform.system()}")


def plan_prepare_host_native_deps(
    CMAKE_BUILD_TYPE: Annotated[Literal["Release", "Debug"] | None, "build type"] = None,
    clean: Annotated[bool, "clean platform build dirs before building"] = False,
    verbose: Annotated[bool, "show build output"] = False,
    dry_run: Annotated[bool, "print commands without executing"] = False,
) -> PreparePlan:
    """Prepare native dependencies for the host tinyllm extension."""
    build_type = CMAKE_BUILD_TYPE or "Release"
    return plan_prepare(
        platform=[_host_native_platform()],
        CMAKE_BUILD_TYPE=build_type,
        verbose=verbose,
        dry_run=dry_run,
        clean=clean,
    )


async def run_prepare_host_native_deps(plan: PreparePlan) -> None:
    """Prepare host native deps and refresh worktree bridge files."""
    await run_prepare(plan)
    require_native_deps_ready(plan.address)
    write_native_deps_bridges(plan.address, dry_run=plan.dry_run)


# ---------------------------------------------------------------------------
# Rebuild stage
# ---------------------------------------------------------------------------


@dataclass
class RebuildPlan:
    build_type: str
    clean: bool
    verbose: bool
    dry_run: bool
    concurrency: int
    build_dir: Path
    install_dirs: tuple[Path, ...]
    cmakelists_dir: Path
    python_executable: str


def _cmake_cache_source_dir(cache: Path) -> Path | None:
    if not cache.is_file():
        return None
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            return Path(line.split("=", 1)[1]).resolve()
    return None


def _current_native_deps_marker() -> str:
    bridge = REPO_ROOT / "1_build" / "native_deps" / "native_deps.cmake"
    if not bridge.is_file():
        return ""
    values: dict[str, str] = {}
    for line in bridge.read_text(errors="replace").splitlines():
        match = re.match(r'set\((ZEPHR_NATIVE_DEPS_[A-Z_]+) "([^"]*)"\)', line)
        if match:
            values[match.group(1)] = match.group(2)
    return "|".join(
        values.get(key, "")
        for key in (
            "ZEPHR_NATIVE_DEPS_ROOT",
            "ZEPHR_NATIVE_DEPS_RECIPE_HASH",
            "ZEPHR_NATIVE_DEPS_HOST_PLATFORM",
        )
    )


def _remove_install_artifacts(install_dirs: tuple[Path, ...], *, verbose: bool) -> None:
    for install_dir in install_dirs:
        for pattern in (
            "zephr_agent_runtime_nanobind*.so",
            "zephr_agent_runtime_nanobind*.pyi",
            "zephr_agent_tools_nanobind*.so",
            "zephr_agent_tools_nanobind*.pyi",
        ):
            for artifact in sorted(install_dir.glob(pattern)):
                if verbose:
                    log.debug("removing nanobind installed artifact", path=artifact)
                artifact.unlink()


def plan_rebuild(
    CMAKE_BUILD_TYPE: Annotated[Literal["Release", "Debug"] | None, "build type"] = None,
    clean: Annotated[bool, "clean build artifacts before rebuilding"] = False,
    verbose: Annotated[bool, "show build output"] = False,
    dry_run: Annotated[bool, "print commands without executing"] = False,
) -> RebuildPlan:
    """Compile the nanobind C++ extensions and install into their package dirs."""
    build_type = CMAKE_BUILD_TYPE or "Release"
    return RebuildPlan(
        build_type=build_type,
        clean=clean,
        verbose=verbose,
        dry_run=dry_run,
        concurrency=psutil.cpu_count(logical=False) or os.cpu_count() or 4,
        build_dir=BUILD_DIR / build_type / "nanobind",
        install_dirs=INSTALL_DIRS,
        cmakelists_dir=CMAKELISTS_DIR,
        python_executable=sys.executable,
    )


async def run_rebuild(plan: RebuildPlan) -> None:
    """Configure, build, and install the nanobind extension."""
    run = Runner(verbose=plan.verbose, dry_run=plan.dry_run)

    if plan.clean and not plan.dry_run:
        parent = plan.build_dir.parent
        if parent.exists():
            if plan.verbose:
                log.debug("removing nanobind build directory", path=parent)
            shutil.rmtree(parent)
        _remove_install_artifacts(plan.install_dirs, verbose=plan.verbose)

    if not plan.clean and not plan.dry_run:
        cache_source = _cmake_cache_source_dir(plan.build_dir / "CMakeCache.txt")
        expected_source = plan.cmakelists_dir.resolve()
        if cache_source is not None and cache_source != expected_source:
            log.debug(
                "removing stale nanobind CMake cache",
                build_dir=plan.build_dir,
                cached_source=cache_source,
                expected_source=expected_source,
            )
            shutil.rmtree(plan.build_dir)
        marker = plan.build_dir / ".native_deps_marker"
        expected_marker = _current_native_deps_marker()
        cached_marker = marker.read_text(errors="replace") if marker.is_file() else ""
        if plan.build_dir.exists() and expected_marker and cached_marker != expected_marker:
            log.debug(
                "removing nanobind build directory for native deps change",
                build_dir=plan.build_dir,
            )
            shutil.rmtree(plan.build_dir)
            _remove_install_artifacts(plan.install_dirs, verbose=plan.verbose)

    if not plan.dry_run:
        log.debug(
            "rebuilding Zephr Agent nanobind extensions",
            build_type=plan.build_type,
            concurrency=plan.concurrency,
        )

    await run(["mkdir", "-p", str(plan.build_dir)])

    cmd = [
        "cmake",
        "-S", str(plan.cmakelists_dir),
        "-B", str(plan.build_dir),
        f"-DCMAKE_BUILD_TYPE={plan.build_type}",
        f"-DPython_EXECUTABLE={plan.python_executable}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    if platform.system() == "Darwin":
        cmd += [
            "-DCMAKE_C_COMPILER=/usr/bin/clang",
            "-DCMAKE_CXX_COMPILER=/usr/bin/clang++",
        ]
    await run(cmd)

    await run([
        "cmake", "--build", str(plan.build_dir),
        "--target", "install",
        "-j", str(plan.concurrency),
    ])
    if not plan.dry_run:
        (plan.build_dir / ".native_deps_marker").write_text(
            _current_native_deps_marker(),
            encoding="utf-8",
        )

    compile_db = plan.build_dir / "compile_commands.json"
    link = REPO_ROOT / "compile_commands.json"
    if not plan.dry_run and compile_db.exists():
        link.unlink(missing_ok=True)
        try:
            link.symlink_to(compile_db)
        except FileExistsError:
            if not link.is_symlink() or link.resolve() != compile_db.resolve():
                link.unlink(missing_ok=True)
                link.symlink_to(compile_db)

    if not plan.dry_run:
        log.debug("Zephr Agent nanobind rebuild complete")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from clis.tinylib.tinycli import Pipeline

    pipeline = Pipeline("rebuild_tinyllm")
    pipeline.stage("prepare", plan_prepare_host_native_deps, run_prepare_host_native_deps)
    pipeline.stage("rebuild", plan_rebuild, run_rebuild)
    pipeline.main()
