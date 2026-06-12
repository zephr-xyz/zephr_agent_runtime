"""Build-time guard for the generated native dependency bridge."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from clis.support.native_dependencies.locator import (
    artifact_is_ready,
    bridge_paths,
    ensure_no_legacy_native_deps,
    native_deps_address,
)


def _parse_key_values(path: Path, pattern: re.Pattern[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group("key")] = match.group("value")
    return values


def _read_bridge(path: Path, kind: str) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"native deps bridge missing at {path}. Run: uv run prepare_dev")

    if kind == "properties":
        return _parse_key_values(
            path,
            re.compile(r"(?P<key>ZEPHR_NATIVE_DEPS_[A-Z_]+)=(?P<value>.*)"),
        )
    if kind == "xcconfig":
        return _parse_key_values(
            path,
            re.compile(r"(?P<key>ZEPHR_NATIVE_DEPS_[A-Z_]+)\s*=\s*(?P<value>.*)"),
        )
    if kind == "cmake":
        return _parse_key_values(
            path,
            re.compile(r'set\((?P<key>ZEPHR_NATIVE_DEPS_[A-Z_]+)\s+"(?P<value>.*)"\)'),
        )

    raise RuntimeError(f"unsupported bridge kind: {kind}")


def _validate_bridge(kind: str) -> None:
    ensure_no_legacy_native_deps()
    address = native_deps_address()
    if not artifact_is_ready(address.root, address.recipe_hash):
        raise RuntimeError(
            "native deps artifact is not ready for current recipe "
            f"{address.recipe_hash[:12]} at {address.root}. Run: uv run prepare_native_deps"
        )

    path = bridge_paths()[kind]
    values = _read_bridge(path, kind)
    expected = {
        "ZEPHR_NATIVE_DEPS_ROOT": str(address.root),
        "ZEPHR_NATIVE_DEPS_RECIPE_HASH": address.recipe_hash,
        "ZEPHR_NATIVE_DEPS_HOST_PLATFORM": address.host_platform,
    }
    mismatches = [
        f"{key}: expected {expected_value}, found {values.get(key) or '<missing>'}"
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    ]
    if mismatches:
        details = "\n".join(f"  - {line}" for line in mismatches)
        raise RuntimeError(
            f"native deps bridge {path} is stale for current checkout.\n"
            f"{details}\n"
            "Run: uv run prepare_dev"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify generated native dependency bridge files match the current recipe."
    )
    parser.add_argument(
        "--bridge",
        choices=["cmake", "properties", "xcconfig", "all"],
        default="all",
        help="bridge format to validate",
    )
    args = parser.parse_args(argv)

    kinds = ["cmake", "properties", "xcconfig"] if args.bridge == "all" else [args.bridge]
    try:
        for kind in kinds:
            _validate_bridge(kind)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
