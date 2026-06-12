"""Extract CLI argument definitions from function signatures.

A function's type hints and defaults become the CLI spec:
    - Annotated[T, "help text"] -> --flag with help
    - list[X] -> consumes multiple values
    - Literal["a", "b"] -> validated choices
    - bool -> --flag / --no_flag (no value)
    - X | None -> optional
"""

from __future__ import annotations

import inspect
import textwrap
import types
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints


# ---------------------------------------------------------------------------
# ParamInfo — one per function parameter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamInfo:
    name: str
    python_type: type  # base scalar type after unwrapping (str, int, bool, Path, ...)
    description: str
    default: Any  # inspect.Parameter.empty if no default
    choices: tuple[Any, ...] | None  # from Literal
    is_optional: bool  # X | None or has a default
    is_list: bool  # list[X]
    is_bool: bool  # bool


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_params(fn: Any) -> list[ParamInfo]:
    """Return a :class:`ParamInfo` for every annotated parameter of *fn*."""
    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)
    result: list[ParamInfo] = []
    for name, param in sig.parameters.items():
        ann = hints.get(name)
        if ann is None:
            continue
        result.append(_parse_annotation(name, ann, param.default))
    return result


def _parse_annotation(name: str, annotation: Any, default: Any) -> ParamInfo:
    description = ""
    inner = annotation

    # Annotated[T, "help text", ...]
    if get_origin(inner) is Annotated:
        args = get_args(inner)
        inner = args[0]
        for meta in args[1:]:
            if isinstance(meta, str):
                description = meta
                break

    # T | None  ->  optional
    is_optional = False
    origin = get_origin(inner)
    if origin is Union or origin is types.UnionType:
        union_args = get_args(inner)
        non_none = [a for a in union_args if a is not type(None)]
        if type(None) in union_args and len(non_none) == 1:
            is_optional = True
            inner = non_none[0]

    # Help text might be on inner type after unwrapping Optional
    # e.g.  Annotated[int, "help"] | None
    if not description and get_origin(inner) is Annotated:
        args = get_args(inner)
        inner = args[0]
        for meta in args[1:]:
            if isinstance(meta, str):
                description = meta
                break

    # list[T]  ->  multi-value (check before Literal so list[Literal[...]] works)
    is_list = False
    if get_origin(inner) is list:
        is_list = True
        la = get_args(inner)
        inner = la[0] if la else str

    # Literal["a", "b"]  ->  choices
    choices: tuple[Any, ...] | None = None
    if get_origin(inner) is Literal:
        choices = get_args(inner)
        inner = type(choices[0]) if choices else str

    is_bool = inner is bool

    return ParamInfo(
        name=name,
        python_type=inner,
        description=description,
        default=default,
        choices=choices,
        is_optional=is_optional or default is not inspect.Parameter.empty,
        is_list=is_list,
        is_bool=is_bool,
    )


# ---------------------------------------------------------------------------
# CLI parsing — no argparse
# ---------------------------------------------------------------------------


def parse_args(
    argv: list[str], schema: list[ParamInfo], context: str = ""
) -> dict[str, Any]:
    """Parse CLI args against a schema. Returns dict of name -> value.

    Only args explicitly passed on the CLI appear in the returned dict.
    *context* is used in error messages (e.g. pipeline name).
    """
    by_name = {p.name: p for p in schema}
    result: dict[str, Any] = {}
    prefix = f"[{context}] " if context else ""
    i = 0

    while i < len(argv):
        arg = argv[i]

        if not arg.startswith("--"):
            raise SystemExit(f"{prefix}unexpected positional arg: {arg}")

        # --no_flag  ->  negate a bool param (snake_case only)
        if arg.startswith("--no_"):
            candidate = arg[5:]
            if candidate in by_name and by_name[candidate].is_bool:
                result[candidate] = False
                i += 1
                continue
            # Fall through to regular parsing (handles params named no_*)

        name = arg[2:]

        if name not in by_name:
            raise SystemExit(f"{prefix}unknown option: {arg}")

        info = by_name[name]

        # Bool flag: --flag (no value needed)
        if info.is_bool:
            result[name] = True
            i += 1
            continue

        # List: consume all following non-flag args
        if info.is_list:
            values: list[str] = []
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                values.append(argv[i])
                i += 1
            if not values:
                raise SystemExit(f"{prefix}--{name} requires at least one value")
            coerced = [_coerce_and_validate(v, info, prefix) for v in values]
            if name in result:
                result[name].extend(coerced)
            else:
                result[name] = coerced
            continue

        # Scalar: consume one value
        if i + 1 >= len(argv):
            raise SystemExit(f"{prefix}--{name} requires a value")
        result[name] = _coerce_and_validate(argv[i + 1], info, prefix)
        i += 2

    return result


def _coerce_and_validate(value: str, info: ParamInfo, prefix: str) -> Any:
    """Validate choices and coerce a CLI string to the target type."""
    if info.choices and value not in [str(c) for c in info.choices]:
        raise SystemExit(
            f"{prefix}--{info.name}: invalid value '{value}'. "
            f"Choose from: {', '.join(str(c) for c in info.choices)}"
        )
    return _coerce_cli(value, info.python_type, info.name, prefix)


def _coerce_cli(value: str, target: type, param_name: str, prefix: str) -> Any:
    """Coerce a CLI string value to the target type."""
    try:
        if target is bool:
            return value.lower() in ("true", "1", "yes")
        if target is int:
            return int(value)
        if target is float:
            return float(value)
        if target is Path:
            return Path(value)
        return value
    except (ValueError, TypeError) as e:
        raise SystemExit(
            f"{prefix}--{param_name}: invalid value '{value}' for type "
            f"{getattr(target, '__name__', target)}: {e}"
        )


def coerce_toml(value: Any, info: ParamInfo) -> Any:
    """Coerce a TOML-parsed value to the target type."""
    if info.python_type is Path and isinstance(value, str):
        return Path(value)
    if info.is_list and isinstance(value, list):
        return [_coerce_toml_scalar(v, info.python_type) for v in value]
    return value


def _coerce_toml_scalar(value: Any, target: type) -> Any:
    if target is Path and isinstance(value, str):
        return Path(value)
    return value


# ---------------------------------------------------------------------------
# Resolution — defaults < TOML < CLI
# ---------------------------------------------------------------------------


def resolve_args(
    schema: list[ParamInfo],
    cli_values: dict[str, Any],
    toml_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Layer defaults < TOML < CLI into a final dict."""
    cfg = toml_config or {}
    out: dict[str, Any] = {}
    for p in schema:
        if p.name in cli_values:
            out[p.name] = cli_values[p.name]
        elif p.name in cfg:
            out[p.name] = coerce_toml(cfg[p.name], p)
        elif p.default is not inspect.Parameter.empty:
            out[p.name] = p.default
        elif p.is_optional:
            out[p.name] = None
        else:
            raise SystemExit(f"missing required argument: --{p.name}")
    return out


# ---------------------------------------------------------------------------
# Auto-generated args dataclass (for --plan display / fingerprinting)
# ---------------------------------------------------------------------------


def make_args_dataclass(
    class_name: str, params: list[ParamInfo], values: dict[str, Any]
) -> Any:
    """Return a frozen dataclass instance snapshotting resolved arg values."""
    fields = [(p.name, Any) for p in params if p.name in values]
    if not fields:
        return None
    cls = make_dataclass(class_name, fields, frozen=True)
    return cls(**{n: values[n] for n, _ in fields})


# ---------------------------------------------------------------------------
# Help formatting
# ---------------------------------------------------------------------------


def format_param_line(p: ParamInfo) -> str:
    """Format a single parameter as a help line."""
    flag = f"--{p.name}"
    if p.is_bool and not p.name.startswith("no_"):
        flag += f"  --no_{p.name}"
    parts = [flag]
    if p.choices:
        parts.append("{" + ", ".join(str(c) for c in p.choices) + "}")
    if p.description:
        parts.append(p.description)
    default = _default_str(p)
    if default and default != "required":
        parts.append(f"[default: {default}]")
    elif default == "required":
        parts.append("(required)")
    return "  ".join(parts)


def _default_str(p: ParamInfo) -> str:
    if p.default is inspect.Parameter.empty:
        return "required"
    if p.default is None:
        return ""
    return str(p.default)


def box(title: str, lines: list[str], width: int) -> str:
    """Draw a Unicode box around lines with a title."""
    width = max(width, len(title) + 8)
    inner = width - 2
    content_width = max(1, width - 4)
    parts = [f"\u256d\u2500 {title} " + "\u2500" * max(0, inner - len(title) - 3) + "\u256e"]
    for line in lines:
        wrapped = textwrap.wrap(
            line,
            width=content_width,
            subsequent_indent="  ",
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for wrapped_line in wrapped:
            padding = content_width - len(wrapped_line)
            parts.append(f"\u2502 {wrapped_line}{' ' * max(0, padding)} \u2502")
    parts.append("\u2570" + "\u2500" * inner + "\u256f")
    return "\n".join(parts)
