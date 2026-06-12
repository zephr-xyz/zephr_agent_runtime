"""Pipeline — opinionated multi-stage CLI runner.

Stages are (plan, run) pairs.  The plan function's signature defines
CLI arguments; the async run function receives the plan object.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, fields as dc_fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine

import anyio

from clis.tinylib.tinycli.effects import PlanEffect
from clis.tinylib.tinycli.signature import (
    ParamInfo,
    box,
    extract_params,
    format_param_line,
    make_args_dataclass,
    parse_args,
    resolve_args,
)
from clis.tinylib.tinycli.ui import (
    PipelineUiReport,
    UiBrowser,
    UiMode,
    default_ui_output_dir,
    open_ui_report,
)


_UI_CONSOLE_MAX_CHARS = 200_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DeriveError(Exception):
    """Raised by plan functions when filesystem state doesn't match expectations.

    The pipeline catches this and prints a clear message with no traceback.
    """


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _Stage:
    name: str
    plan_fn: Callable[..., Any]
    run_fn: Callable[..., Coroutine[Any, Any, None]]
    params: list[ParamInfo]


@dataclass
class _Planned:
    stage: _Stage
    plan: Any
    args: Any  # auto-generated frozen dataclass of resolved arg values
    effects: list[PlanEffect]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """An opinionated, multi-stage CLI pipeline.

    Each stage is a (plan, run) pair.  The plan function's signature
    defines CLI arguments.  The run function is async and receives the plan.

    Usage::

        pipeline = Pipeline("build_tinyllm")
        pipeline.stage("prepare", plan_prepare, run_prepare)
        pipeline.stage("rebuild", plan_rebuild, run_rebuild)
        pipeline.main()
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._stages: list[_Stage] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def stage(
        self,
        name: str,
        plan_fn: Callable[..., Any],
        run_fn: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        """Register a stage with a plan function and an async run function."""
        params = extract_params(plan_fn)
        # Detect type/choice conflicts with previously registered params
        existing: dict[str, ParamInfo] = {}
        for stg in self._stages:
            for p in stg.params:
                existing[p.name] = p
        for p in params:
            if p.name in existing:
                prev = existing[p.name]
                if (prev.python_type, prev.choices) != (p.python_type, p.choices):
                    raise ValueError(
                        f"Conflicting definitions for --{p.name} between stages: "
                        f"type={prev.python_type.__name__}/{p.python_type.__name__}, "
                        f"choices={prev.choices}/{p.choices}"
                    )
        self._stages.append(
            _Stage(name=name, plan_fn=plan_fn, run_fn=run_fn, params=params)
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def main(
        self,
        argv: list[str] | None = None,
        start_from: str | None = None,
        stop_after: str | None = None,
    ) -> None:
        """Sync entry point — parse args, plan, optionally run.

        *start_from* sets the first stage to run. Typically determined
        by which script entrypoint was invoked.
        *stop_after* sets the last stage to run unless the caller passes
        --continue.
        """
        if argv is None:
            argv = sys.argv[1:]

        stage_names = [s.name for s in self._stages]

        # --help intercept
        if "--help" in argv or "-h" in argv:
            self._print_help(start_from=start_from, stop_after=stop_after)
            return

        # Extract built-in flags before parsing user args
        flags = _extract_builtins(argv)
        if start_from is not None and flags.start_from is None:
            flags.start_from = start_from
        if stop_after is not None and flags.stop_after is None and not flags.continue_pipeline:
            flags.stop_after = stop_after

        # Apply logging verbosity.
        if flags.trace or flags.verbose:
            from clis.tinylib.tinylog import Level, log

            level_name = "TRACE" if flags.trace else "DEBUG"
            os.environ["TINYLOG_LEVEL"] = level_name
            log.set_level(Level.TRACE if flags.trace else Level.DEBUG)
            tinyllm = sys.modules.get("clis.zephr_agent_runtime")
            if tinyllm is not None and hasattr(tinyllm, "set_log_level"):
                tinyllm.set_log_level(level_name.lower())

        # Validate stage selection
        if flags.start_from and flags.start_from not in stage_names:
            raise SystemExit(
                f"unknown start stage: {flags.start_from}. "
                f"Available: {', '.join(stage_names)}"
            )
        if flags.stop_after and flags.stop_after not in stage_names:
            raise SystemExit(
                f"unknown stage for --stop_after: {flags.stop_after}. "
                f"Available: {', '.join(stage_names)}"
            )

        # Load TOML config
        toml_global: dict[str, Any] = {}
        toml_stages: dict[str, dict[str, Any]] = {}
        if flags.config_path:
            toml_global, toml_stages = _load_toml(flags.config_path, self.name)

        # Parse user args against merged schema
        merged = self._merged_schema()
        cli_values = parse_args(flags.remaining or [], merged, self.name)

        # Filter stages
        active = _filter_stages(self._stages, flags.start_from, flags.stop_after)

        # Plan phase
        planned: list[_Planned] = []
        for stg in active:
            if flags.trace or flags.verbose:
                from clis.tinylib.tinylog import log

                log.debug("planning stage", stage=stg.name)
            merged_cfg = {**toml_global, **toml_stages.get(stg.name, {})}
            args_dict = resolve_args(stg.params, cli_values, merged_cfg)
            if (flags.trace or flags.verbose) and "verbose" in args_dict:
                args_dict["verbose"] = True
            try:
                plan = stg.plan_fn(**args_dict)
            except DeriveError as exc:
                print(f"error ({stg.name}): {exc}", file=sys.stderr)
                sys.exit(1)
            args_dc = make_args_dataclass(
                f"{stg.name.title()}Args", stg.params, args_dict
            )
            planned.append(
                _Planned(
                    stage=stg,
                    plan=plan,
                    args=args_dc,
                    effects=_plan_effects(plan),
                )
            )

        if flags.plan_mode:
            ui_report = _maybe_write_ui_report(
                self.name,
                planned,
                flags,
                argv,
                all_stage_names=stage_names,
                mode="plan",
            )
            _display_plans(planned)
            if ui_report is not None:
                print(f"\nPipeline UI: {ui_report.url}")
                _maybe_open_ui_report(ui_report, flags)
            return

        ui_report = _maybe_write_ui_report(
            self.name,
            planned,
            flags,
            argv,
            all_stage_names=stage_names,
            mode="run",
        )
        if ui_report is not None:
            print(f"Pipeline UI: {ui_report.url}")
            _maybe_open_ui_report(ui_report, flags)

        # Run phase
        anyio.run(self._execute, planned, ui_report)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _merged_schema(self) -> list[ParamInfo]:
        """Collect unique params across all stages (first definition wins)."""
        seen: set[str] = set()
        result: list[ParamInfo] = []
        for stg in self._stages:
            for p in stg.params:
                if p.name not in seen:
                    seen.add(p.name)
                    result.append(p)
        return result

    def _print_help(
        self,
        *,
        start_from: str | None = None,
        stop_after: str | None = None,
    ) -> None:
        prog = Path(sys.argv[0]).name
        stages_str = " -> ".join(s.name for s in self._stages)
        print(f"\nUsage: {prog} [options]\n")
        if start_from is not None and stop_after == start_from:
            print(f"  {prog}                              run stage: {start_from}")
        elif start_from is not None:
            print(f"  {prog}                              run from {start_from}: {stages_str}")
        else:
            print(f"  {prog}                              run all: {stages_str}")
        is_stage_entrypoint = start_from is not None and stop_after == start_from
        if len(self._stages) > 1 and not is_stage_entrypoint:
            print(f"  {prog} --start_from X               run starting from X")
            print(f"  {prog} --stop_after X               run up to and including X")
        if len(self._stages) > 1 and is_stage_entrypoint:
            print(f"  {prog} --continue                   continue after the entrypoint stage")
        print(f"  {prog} --plan                        show plan without running")
        print()

        # Collect all sections
        all_sections: list[tuple[str, list[str]]] = []

        # Keep plan-function parameters in their stage sections. Some options
        # intentionally appear in multiple stages, but that does not make them
        # semantically global to the whole pipeline.
        merged = self._merged_schema()

        # Global Options: built-in pipeline flags only.
        option_lines: list[str] = []
        option_names = {p.name for p in merged}
        if len(self._stages) > 1 and not is_stage_entrypoint:
            option_lines.append("--start_from STAGE  start from STAGE")
            option_lines.append("--stop_after STAGE  stop after STAGE")
        if len(self._stages) > 1 and is_stage_entrypoint:
            option_lines.append("--continue  continue after the entrypoint stage")
        option_lines.append("--plan  show execution plan without running")
        option_lines.append("--ui, --no-ui  render standalone HTML pipeline UI")
        option_lines.append("--ui_browser {chrome, none}  UI browser to open  [default: chrome]")
        if "verbose" in option_names:
            option_lines.append("-v  alias for --verbose")
        else:
            option_lines.append("--verbose, -v  enable debug logging")
        option_lines.append("--trace  enable trace logging")
        option_lines.append("--config PATH  path to TOML config file")
        option_lines.append("--help  show this help")
        all_sections.append(("Global Options", option_lines))

        # Per-stage sections (source location, doc, stage-specific params).
        # Fixed stage entrypoints are task-focused; keep their help focused too.
        help_stages = self._stages
        if is_stage_entrypoint and start_from is not None:
            help_stages = [stg for stg in self._stages if stg.name == start_from]
        for stg in help_stages:
            try:
                src_file = inspect.getfile(stg.plan_fn)
                src_line = inspect.getsourcelines(stg.plan_fn)[1]
                loc = f"  {src_file}:{src_line}"
            except (TypeError, OSError):
                loc = ""
            doc = (stg.plan_fn.__doc__ or "").strip().split("\n")[0]
            lines: list[str] = []
            if loc:
                lines.append(loc)
            if doc:
                lines.append(f"  {doc}")
            lines += [format_param_line(p) for p in stg.params]
            if lines:
                all_sections.append((stg.name, lines))

        terminal_width = shutil.get_terminal_size(fallback=(100, 24)).columns
        width = max(80, min(terminal_width, 120))

        for title, lines in all_sections:
            print(box(title, lines, width))
            print()

    @staticmethod
    async def _execute(
        planned: list[_Planned],
        ui_report: PipelineUiReport | None = None,
    ) -> None:
        for p in planned:
            from clis.tinylib.tinylog import log

            log.debug("running stage", stage=p.stage.name)
            stage_start = time.monotonic()
            console = _TailBuffer(max_chars=_UI_CONSOLE_MAX_CHARS)
            last_console_update = 0.0

            def _console_text() -> str:
                return console.getvalue()

            def _update_console(*, force: bool = False) -> None:
                nonlocal last_console_update
                if ui_report is None:
                    return
                now = time.monotonic()
                if not force and now - last_console_update < 1.0:
                    return
                last_console_update = now
                ui_report.update_stage_console(p.stage.name, _console_text())

            original_stdout = sys.stdout
            original_stderr = sys.stderr
            if ui_report is not None:
                ui_report.mark_stage(p.stage.name, "running")
                sys.stdout = _TeeStream(original_stdout, console, _update_console)
                sys.stderr = _TeeStream(original_stderr, console, _update_console)
            try:
                await p.stage.run_fn(p.plan)
            except DeriveError as exc:
                if ui_report is not None:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    _update_console(force=True)
                    ui_report.mark_stage(
                        p.stage.name,
                        "error",
                        duration=time.monotonic() - stage_start,
                        error=str(exc),
                        console=_console_text(),
                    )
                    ui_report.complete(failed_stage=p.stage.name)
                print(f"error ({p.stage.name}): {exc}", file=sys.stderr)
                sys.exit(1)
            except SystemExit as exc:
                if ui_report is not None:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    _update_console(force=True)
                    error = str(exc) or f"exit code {exc.code}"
                    ui_report.mark_stage(
                        p.stage.name,
                        "error",
                        duration=time.monotonic() - stage_start,
                        error=error,
                        console=_console_text(),
                    )
                    ui_report.complete(failed_stage=p.stage.name)
                raise
            except Exception as exc:
                if ui_report is not None:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    _update_console(force=True)
                    ui_report.mark_stage(
                        p.stage.name,
                        "error",
                        duration=time.monotonic() - stage_start,
                        error=f"{type(exc).__name__}: {exc}",
                        console=_console_text(),
                    )
                    ui_report.complete(failed_stage=p.stage.name)
                raise
            finally:
                if ui_report is not None:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
            if ui_report is not None:
                _update_console(force=True)
                ui_report.mark_stage(
                    p.stage.name,
                    "success",
                    duration=time.monotonic() - stage_start,
                    console=_console_text(),
                )
        if ui_report is not None:
            ui_report.complete()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TailBuffer:
    def __init__(self, *, max_chars: int) -> None:
        self._max_chars = max_chars
        self._chunks: list[str] = []
        self._length = 0
        self._dropped = 0

    def write(self, text: str) -> int:
        if not text:
            return 0
        text_len = len(text)
        if text_len >= self._max_chars:
            self._dropped += self._length + text_len - self._max_chars
            self._chunks = [text[-self._max_chars:]]
            self._length = self._max_chars
            return text_len

        self._chunks.append(text)
        self._length += text_len
        while self._length > self._max_chars and self._chunks:
            overflow = self._length - self._max_chars
            first = self._chunks[0]
            if len(first) <= overflow:
                self._chunks.pop(0)
                self._length -= len(first)
                self._dropped += len(first)
            else:
                self._chunks[0] = first[overflow:]
                self._length -= overflow
                self._dropped += overflow
        return text_len

    def getvalue(self) -> str:
        text = "".join(self._chunks)
        if not self._dropped:
            return text
        return (
            f"[console truncated; showing last {self._max_chars} chars, "
            f"omitted {self._dropped} chars]\n"
            + text
        )


class _TeeStream:
    def __init__(
        self,
        original: Any,
        capture: _TailBuffer,
        on_write: Callable[[], None],
    ) -> None:
        self._original = original
        self._capture = capture
        self._on_write = on_write

    def write(self, text: str) -> int:
        written = self._original.write(text)
        self._capture.write(text)
        self._on_write()
        return written

    def flush(self) -> None:
        self._original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


@dataclass
class _BuiltinFlags:
    plan_mode: bool = False
    ui_mode: bool = False
    ui_browser: UiBrowser = "chrome"
    config_path: Path | None = None
    start_from: str | None = None
    stop_after: str | None = None
    continue_pipeline: bool = False
    verbose: bool = False
    trace: bool = False
    remaining: list[str] | None = None

    def __post_init__(self) -> None:
        if self.remaining is None:
            self.remaining = []


def _extract_builtins(argv: list[str]) -> _BuiltinFlags:
    """Strip built-in flags from argv."""
    flags = _BuiltinFlags()
    filtered: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--plan":
            flags.plan_mode = True
            i += 1
        elif a == "--ui":
            flags.ui_mode = True
            i += 1
        elif a == "--no-ui":
            flags.ui_mode = False
            i += 1
        elif a == "--ui_browser":
            if i + 1 >= len(argv):
                raise SystemExit("--ui_browser requires a value: chrome or none")
            value = argv[i + 1]
            if value not in ("chrome", "none"):
                raise SystemExit("--ui_browser must be one of: chrome, none")
            flags.ui_browser = value  # type: ignore[assignment]
            if value != "none":
                flags.ui_mode = True
            i += 2
        elif a in ("--verbose", "-v"):
            flags.verbose = True
            i += 1
        elif a == "--trace":
            flags.trace = True
            i += 1
        elif a == "--config" and i + 1 < len(argv):
            flags.config_path = Path(argv[i + 1])
            i += 2
        elif a == "--start_from" and i + 1 < len(argv):
            flags.start_from = argv[i + 1]
            i += 2
        elif a == "--stop_after" and i + 1 < len(argv):
            flags.stop_after = argv[i + 1]
            i += 2
        elif a == "--continue":
            flags.continue_pipeline = True
            i += 1
        else:
            filtered.append(a)
            i += 1
    flags.remaining = filtered
    return flags


def _filter_stages(
    stages: list[_Stage],
    start_from: str | None,
    stop_after: str | None,
) -> list[_Stage]:
    names = [s.name for s in stages]
    lo = names.index(start_from) if start_from else 0
    hi = names.index(stop_after) + 1 if stop_after else len(stages)
    return stages[lo:hi]


def _load_toml(
    path: Path, pipeline_name: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return ``(global_args, {stage: stage_args})`` from a TOML file.

    Expected layout::

        [pipeline_name]
        shared_arg = "value"

        [pipeline_name.stage_name]
        stage_specific_arg = "value"
    """
    import tomllib

    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)

    section = data.get(pipeline_name, {})
    global_args: dict[str, Any] = {}
    stage_args: dict[str, dict[str, Any]] = {}
    for key, val in section.items():
        if isinstance(val, dict):
            stage_args[key] = val
        else:
            global_args[key] = val

    return global_args, stage_args


def _display_plans(planned: list[_Planned]) -> None:
    for i, p in enumerate(planned):
        if i > 0:
            print()
        print(f"=== {p.stage.name} ===")

        # Resolved CLI/config args
        if p.args is not None:
            print("  args:")
            for f in dc_fields(p.args):
                _print_plan_value(f.name, getattr(p.args, f.name), indent=4)

        # Plan object fields
        if p.plan is not None and hasattr(p.plan, "__dataclass_fields__"):
            print("  plan:")
            for f in dc_fields(p.plan):
                _print_plan_value(f.name, getattr(p.plan, f.name), indent=4)

        if p.effects:
            print("  effects:")
            for effect in p.effects:
                print(f"    - {_format_effect(effect)}")


def _maybe_write_ui_report(
    pipeline_name: str,
    planned: list[_Planned],
    flags: _BuiltinFlags,
    argv: list[str],
    *,
    all_stage_names: list[str],
    mode: UiMode,
) -> PipelineUiReport | None:
    if not flags.ui_mode:
        return None
    command = " ".join([Path(sys.argv[0]).name, *argv])
    output_dir = default_ui_output_dir(pipeline_name, planned)
    previous_state = _read_previous_ui_state(output_dir)
    _reset_ui_report_dir(output_dir)
    report = PipelineUiReport(
        pipeline_name=pipeline_name,
        planned=planned,
        all_stage_names=all_stage_names,
        previous_state=previous_state,
        output_dir=output_dir,
        command=command,
        mode=mode,
    )
    report.write()
    return report


def _read_previous_ui_state(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "pipeline.state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _reset_ui_report_dir(output_dir: Path) -> None:
    for path in (
        output_dir / "pipeline.html",
        output_dir / "pipeline.state.json",
    ):
        path.unlink(missing_ok=True)

    # Clean up the previous multi-file tinycli UI layout when present. This
    # directory only belongs to tinycli if it contains the old generated state.
    old_reports_dir = output_dir / "reports"
    if (old_reports_dir / "000_pipeline.state.json").is_file():
        shutil.rmtree(old_reports_dir)


def _maybe_open_ui_report(report: PipelineUiReport, flags: _BuiltinFlags) -> None:
    if flags.ui_browser == "none":
        return
    try:
        anyio.run(open_ui_report, report, flags.ui_browser)
    except Exception as exc:
        print(
            f"warning: failed to open pipeline UI in {flags.ui_browser}: {exc}",
            file=sys.stderr,
        )


def _plan_effects(plan: Any) -> list[PlanEffect]:
    method = getattr(plan, "plan_effects", None)
    if method is None:
        return []
    effects = method()
    return list(effects)


def _format_effect(effect: PlanEffect) -> str:
    target = ""
    if effect.path is not None:
        target = str(effect.path)
    elif effect.pattern is not None:
        target = effect.pattern
    elif effect.subject is not None:
        target = effect.subject
    optional = "" if effect.required else " (optional)"
    if target:
        return f"{effect.kind} {target} - {effect.label}{optional}"
    return f"{effect.kind} - {effect.label}{optional}"


def _print_plan_value(name: str, value: Any, *, indent: int) -> None:
    prefix = " " * indent
    if is_dataclass(value) and not isinstance(value, type):
        print(f"{prefix}{name}:")
        for f in dc_fields(value):
            _print_plan_value(f.name, getattr(value, f.name), indent=indent + 2)
        return

    if isinstance(value, dict):
        if not value:
            print(f"{prefix}{name}: {{}}")
            return
        print(f"{prefix}{name}:")
        for key, item in value.items():
            _print_plan_value(str(key), item, indent=indent + 2)
        return

    if isinstance(value, (list, tuple)):
        if not value:
            print(f"{prefix}{name}: []")
            return
        print(f"{prefix}{name}:")
        for item in value:
            if is_dataclass(item) and not isinstance(item, type):
                print(f"{prefix}  -")
                for f in dc_fields(item):
                    _print_plan_value(f.name, getattr(item, f.name), indent=indent + 4)
            elif isinstance(item, dict):
                print(f"{prefix}  -")
                for key, nested in item.items():
                    _print_plan_value(str(key), nested, indent=indent + 4)
            elif isinstance(item, list):
                print(f"{prefix}  -")
                for nested in item:
                    print(f"{prefix}    - {nested}")
            else:
                print(f"{prefix}  - {item}")
        return

    print(f"{prefix}{name}: {value!r}")
