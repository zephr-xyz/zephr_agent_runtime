"""tinylog — structured logging, metrics, and snapshot testing.

Zero dependencies. Four output formats (json, logfmt, human, gcp).
Metrics and logs share one event type and one pipeline.
Built for snapshot-based testing from day one.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from collections.abc import Callable
from typing import Any, TextIO
from uuid import uuid1


# ---------------------------------------------------------------------------
# Level
# ---------------------------------------------------------------------------

class Level(IntEnum):
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40

    def __str__(self) -> str:
        return self.name.lower()


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Caller:
    file: str
    line: int
    function: str
    chain: list[str] | None = None

    def short(self) -> str:
        """filename:line without full path."""
        name = self.file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return f"{name}:{self.line}"

    def __str__(self) -> str:
        if self.chain:
            return "->".join(self.chain)
        if self.function and self.function != "<module>":
            return self.function
        name = self.file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return f"{name}:{self.line}"


@dataclass(slots=True)
class Event:
    ts: float
    level: Level
    msg: str
    fields: dict[str, Any]
    caller: Caller
    kind: str = "log"  # "log" or "metric"
    metric_name: str | None = None
    metric_value: float | None = None
    metric_unit: str | None = None
    exception: str | None = None
    seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for serialization. Merges fields into top level."""
        d: dict[str, Any] = {}
        if self.seq is not None:
            d["seq"] = self.seq
        else:
            d["ts"] = _format_ts(self.ts)
        d["level"] = str(self.level)
        d["msg"] = self.msg
        d["caller"] = str(self.caller)
        d["kind"] = self.kind
        if self.kind == "metric":
            d["metric_name"] = self.metric_name
            d["metric_value"] = self.metric_value
            if self.metric_unit is not None:
                d["metric_unit"] = self.metric_unit
        if self.exception is not None:
            d["exception"] = self.exception
        d.update(self.fields)
        return d


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_ts(ts: float) -> str:
    """ISO 8601 UTC with milliseconds."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _json_encode(value: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_encode(v) for k, v in value.items()}
    if isinstance(value, bytes):
        import base64
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def format_json(event: Event) -> str:
    """One JSON object per line."""
    d = event.to_dict()
    return json.dumps({k: _json_encode(v) for k, v in d.items()}, separators=(",", ":"))


def format_logfmt(event: Event) -> str:
    """key=value pairs, one line. LLM and grep friendly."""
    d = event.to_dict()
    # For logfmt, shorten exception to just the error line (no full traceback)
    if "exception" in d and d["exception"] is not None:
        lines = d["exception"].strip().splitlines()
        d["exception"] = lines[-1] if lines else d["exception"]
    parts: list[str] = []
    for k, v in d.items():
        if v is None:
            continue
        sv = str(v)
        if " " in sv or '"' in sv or "=" in sv or not sv:
            sv = '"' + sv.replace("\\", "\\\\").replace('"', '\\"') + '"'
        parts.append(f"{k}={sv}")
    return " ".join(parts)


# ANSI colors
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_COLORS = {
    Level.TRACE: "\033[37m",     # white/gray
    Level.DEBUG: "\033[36m",     # cyan
    Level.INFO: "\033[32m",      # green
    Level.WARN: "\033[33m",      # yellow
    Level.ERROR: "\033[31m",     # red
}


def format_human(event: Event) -> str:
    """Colorized multi-line format for terminals."""
    c = _COLORS.get(event.level, "")
    if event.seq is not None:
        ts = f"#{event.seq}"
    else:
        ts = _format_ts(event.ts).split("T")[1].rstrip("Z")  # just HH:MM:SS.mmmZ
    level = f"{event.level.name:<5}"
    if event.caller.file:
        loc = f"{event.caller.file}:{event.caller.line}"
        if event.caller.chain:
            loc += ":" + "->".join(event.caller.chain)
        elif event.caller.function and event.caller.function != "<module>":
            loc += f":{event.caller.function}"
    else:
        loc = str(event.caller)
    header = f"{_DIM}{ts}{_RESET} {c}{level}{_RESET} {_DIM}{loc}{_RESET}"
    lines = [f"{header}\n  {c}{event.msg}{_RESET}"]

    if event.kind == "metric":
        lines.append(f"  {_DIM}metric:{_RESET} {event.metric_name}={event.metric_value}"
                      + (f" {event.metric_unit}" if event.metric_unit else ""))

    if event.fields:
        kvs = " ".join(f"{k}={v}" for k, v in event.fields.items())
        lines.append(f"  {_DIM}{kvs}{_RESET}")

    if event.exception:
        for exc_line in event.exception.splitlines():
            lines.append(f"  {_DIM}{exc_line}{_RESET}")

    return "\n".join(lines)


def format_gcp(event: Event) -> str:
    """Google Cloud Logging structured JSON envelope."""
    severity = event.level.name
    if severity == "TRACE":
        severity = "DEBUG"
    elif severity == "WARN":
        severity = "WARNING"

    seconds, partial = divmod(event.ts, 1)
    d: dict[str, Any] = {
        "severity": severity,
        "timestamp": {
            "seconds": int(seconds),
            "nanos": int(partial * 1e9),
        },
        "logging.googleapis.com/insertId": uuid1().hex,
        "logging.googleapis.com/sourceLocation": {
            "file": event.caller.file,
            "line": str(event.caller.line),
            "function": event.caller.function,
        },
        "message": event.msg,
    }
    if event.fields:
        d["extra"] = {k: _json_encode(v) for k, v in event.fields.items()}
    if event.kind == "metric":
        d["metric"] = {
            "name": event.metric_name,
            "value": event.metric_value,
        }
        if event.metric_unit is not None:
            d["metric"]["unit"] = event.metric_unit
    if event.exception is not None:
        d["exception"] = event.exception

    return json.dumps(d, separators=(",", ":"))


Formatter = Callable[[Event], str]

_FORMATTERS: dict[str, Formatter] = {
    "json": format_json,
    "logfmt": format_logfmt,
    "human": format_human,
    "gcp": format_gcp,
}


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

@dataclass
class _Sink:
    write: Any  # callable(str) or file-like
    formatter: str | None  # None = raw events (Recorder)
    level: Level

    def emit(self, event: Event):
        if event.level < self.level:
            return
        if self.formatter is None:
            # Raw sink (Recorder) — pass Event directly
            self.write(event)
        else:
            fmt = _FORMATTERS[self.formatter]
            line = fmt(event) + "\n"
            self.write(line)


# ---------------------------------------------------------------------------
# Recorder — snapshot capture and comparison
# ---------------------------------------------------------------------------

class Recorder:
    """Sink that captures Events for snapshot testing.

    Optionally takes a path at construction for incremental commit().
    Can be attached/detached from a Logger via add_sink()/remove_sink().

    deterministic=True normalizes volatile fields on ingest:
      - ts → sequence integer (0, 1, 2, ...)
      - caller → function/chain only (no file path or line number)
      - exception → just the exception line (no traceback)
    """

    def __init__(self, path: str | Path | None = None,
                 deterministic: bool = False) -> None:
        self.events: list[Event] = []
        self._path: Path | None = Path(path) if path is not None else None
        self._committed: int = 0  # index of first uncommitted event
        self._deterministic: bool = deterministic
        self._seq: int = 0

    def __call__(self, event: Event):
        if self._deterministic:
            event = self._normalize(event)
        self.events.append(event)

    def _normalize(self, event: Event) -> Event:
        """Return a copy with volatile fields replaced by stable values."""
        seq = self._seq
        self._seq += 1
        caller = Caller(
            file="",
            line=0,
            function=event.caller.function,
            chain=event.caller.chain,
        )
        exception = None
        if event.exception is not None:
            lines = event.exception.strip().splitlines()
            exception = lines[-1] if lines else event.exception
        return Event(
            ts=0.0,
            level=event.level,
            msg=event.msg,
            fields=event.fields,
            caller=caller,
            kind=event.kind,
            metric_name=event.metric_name,
            metric_value=event.metric_value,
            metric_unit=event.metric_unit,
            exception=exception,
            seq=seq,
        )

    def clear(self):
        self.events.clear()
        self._committed = 0
        self._seq = 0

    def commit(self, path: str | Path | None = None):
        """Append uncommitted events to the JSONL file.

        Uses the path from construction, or an explicit override.
        Only writes events added since the last commit().
        """
        p = Path(path) if path is not None else self._path
        if p is None:
            raise ValueError("No path specified — pass path to commit() or Recorder(path=...)")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for event in self.events[self._committed:]:
                f.write(format_json(event) + "\n")
        self._committed = len(self.events)

    def save(self, path: str | Path):
        """Write all captured events as JSONL (overwrites)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for event in self.events:
                f.write(format_json(event) + "\n")
        self._committed = len(self.events)

    def assert_matches(
        self,
        path: str | Path,
        mask: list[str] | None = None,
        approx: dict[str, float] | None = None,
    ):
        """Compare captured events against a snapshot file.

        mask: field names to ignore entirely (e.g. ["ts"])
        approx: field names with relative tolerance (e.g. {"latency_ms": 0.5} = within 50%)
        """
        mask_set = set(mask or [])
        approx = approx or {}
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Snapshot not found: {p}  (run with save() first)")

        saved = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    saved.append(json.loads(line))

        actual = [e.to_dict() for e in self.events]

        if len(actual) != len(saved):
            raise AssertionError(
                f"Event count mismatch: got {len(actual)}, snapshot has {len(saved)}"
            )

        for i, (act, exp) in enumerate(zip(actual, saved)):
            diffs = _diff_dicts(act, exp, mask_set, approx)
            if diffs:
                diff_str = "\n".join(f"  {d}" for d in diffs)
                raise AssertionError(f"Event {i} mismatch:\n{diff_str}")


def _diff_dicts(
    actual: dict, expected: dict,
    mask: set[str], approx: dict[str, float],
) -> list[str]:
    """Compare two flat dicts, returning a list of human-readable diffs."""
    diffs: list[str] = []
    all_keys = set(actual) | set(expected)
    for k in sorted(all_keys):
        if k in mask:
            continue
        a = actual.get(k)
        e = expected.get(k)
        if k in approx and isinstance(a, (int, float)) and isinstance(e, (int, float)):
            if e == 0:
                if a != 0:
                    diffs.append(f"{k}: {a} != {e} (expected zero)")
            elif abs(a - e) / abs(e) > approx[k]:
                diffs.append(f"{k}: {a} != {e} (outside {approx[k]:.0%} tolerance)")
        elif a != e:
            diffs.append(f"{k}: {a!r} != {e!r}")
    return diffs


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

# Context var for bound fields (supports `with log.bind(...)`)
_bound_fields: ContextVar[dict[str, Any]] = ContextVar("_bound_fields", default={})


class _BindContext:
    """Context manager that temporarily adds fields to all log calls."""

    def __init__(self, fields: dict[str, Any]):
        self._fields = fields
        self._token = None

    def __enter__(self):
        current = _bound_fields.get()
        merged = {**current, **self._fields}
        self._token = _bound_fields.set(merged)
        return self

    def __exit__(self, *exc):
        _bound_fields.reset(self._token)

    # Also usable as a child logger
    def info(self, msg: str, **fields): _module_log(Level.INFO, msg, fields, _stack_offset=2)
    def debug(self, msg: str, **fields): _module_log(Level.DEBUG, msg, fields, _stack_offset=2)
    def warn(self, msg: str, **fields): _module_log(Level.WARN, msg, fields, _stack_offset=2)
    def error(self, msg: str, **fields): _module_log(Level.ERROR, msg, fields, _stack_offset=2)
    def trace(self, msg: str, **fields): _module_log(Level.TRACE, msg, fields, _stack_offset=2)
    def metric(self, name: str, value: float, unit: str | None = None, **fields):
        _module_metric(name, value, unit, fields, _stack_offset=2)


class _OptLogger:
    """Logger wrapper returned by Logger.opt() — carries capture options."""

    __slots__ = ("_logger", "_depth", "_exception")

    def __init__(self, logger: Logger, depth: int = 1, exception: bool = False):
        self._logger = logger
        self._depth = depth
        self._exception = exception

    def trace(self, msg: str, **fields):
        self._logger._log(Level.TRACE, msg, fields, _stack_offset=2, depth=self._depth, exception=self._exception)

    def debug(self, msg: str, **fields):
        self._logger._log(Level.DEBUG, msg, fields, _stack_offset=2, depth=self._depth, exception=self._exception)

    def info(self, msg: str, **fields):
        self._logger._log(Level.INFO, msg, fields, _stack_offset=2, depth=self._depth, exception=self._exception)

    def warn(self, msg: str, **fields):
        self._logger._log(Level.WARN, msg, fields, _stack_offset=2, depth=self._depth, exception=self._exception)

    def error(self, msg: str, **fields):
        self._logger._log(Level.ERROR, msg, fields, _stack_offset=2, depth=self._depth, exception=self._exception)

    def metric(self, name: str, value: float, unit: str | None = None, **fields):
        self._logger._metric(name, value, unit, fields, _stack_offset=2, depth=self._depth, exception=self._exception)


class Logger:
    """Configurable logger with sinks."""

    def __init__(self) -> None:
        self._sinks: list[_Sink] = []
        self._lock = threading.Lock()

    def add_sink(
        self,
        sink: Any,  # file-like, callable, or Recorder
        format: str = "human",
        level: Level = Level.TRACE,
    ) -> int:
        """Add a sink. Returns an ID for later removal.

        sink: file-like with .write(), a callable(str), or a Recorder
        format: "json", "logfmt", "human", "gcp", or None for raw Events
        """
        if isinstance(sink, Recorder):
            s = _Sink(write=sink, formatter=None, level=level)
        elif hasattr(sink, "write") and callable(sink.write):
            s = _Sink(write=sink.write, formatter=format, level=level)
        elif callable(sink):
            s = _Sink(write=sink, formatter=format, level=level)
        else:
            raise TypeError(f"Sink must be file-like, callable, or Recorder, got {type(sink)}")

        with self._lock:
            self._sinks.append(s)
            return len(self._sinks) - 1

    def remove_sink(self, sink: int | Recorder):
        """Remove a sink by ID (from add_sink return value) or by Recorder instance."""
        with self._lock:
            if isinstance(sink, int):
                if 0 <= sink < len(self._sinks):
                    self._sinks[sink] = None  # type: ignore
            else:
                for i, s in enumerate(self._sinks):
                    if s is not None and s.write is sink:
                        self._sinks[i] = None  # type: ignore
                        break

    def _emit(self, event: Event):
        with self._lock:
            sinks = [s for s in self._sinks if s is not None]
        for s in sinks:
            try:
                s.emit(event)
            except Exception:
                pass  # never let logging crash the app

    def _log(self, level: Level, msg: str, fields: dict[str, Any],
             _stack_offset: int = 1, depth: int = 1, exception: bool = False):
        # Short-circuit: any sink at this level?
        with self._lock:
            min_level = min((s.level for s in self._sinks if s is not None), default=Level.ERROR)
        if level < min_level:
            return

        caller = _get_caller(_stack_offset + 1, depth=depth)
        merged = {**_bound_fields.get(), **fields}
        exc_text = _capture_exception() if exception else None
        event = Event(
            ts=time.time(),
            level=level,
            msg=msg,
            fields=merged,
            caller=caller,
            exception=exc_text,
        )
        self._emit(event)

    def _metric(self, name: str, value: float, unit: str | None,
                fields: dict[str, Any], _stack_offset: int = 1,
                depth: int = 1, exception: bool = False):
        caller = _get_caller(_stack_offset + 1, depth=depth)
        merged = {**_bound_fields.get(), **fields}
        exc_text = _capture_exception() if exception else None
        event = Event(
            ts=time.time(),
            level=Level.INFO,
            msg=name,
            fields=merged,
            caller=caller,
            kind="metric",
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            exception=exc_text,
        )
        self._emit(event)

    def opt(self, depth: int = 1, exception: bool = False) -> _OptLogger:
        """Return a logger wrapper with capture options.

        depth: number of stack frames to capture (>1 shows call chain)
        exception: if True, capture current exception traceback
        """
        return _OptLogger(self, depth=depth, exception=exception)

    def set_level(self, level: Level) -> None:
        """Set the minimum level on all sinks."""
        with self._lock:
            for s in self._sinks:
                if s is not None:
                    s.level = level

    def bind(self, **fields) -> _BindContext:
        """Bind fields to all log calls within a `with` block, or use as a child logger."""
        return _BindContext(fields)

    def trace(self, msg: str, **fields): self._log(Level.TRACE, msg, fields, _stack_offset=2)
    def debug(self, msg: str, **fields): self._log(Level.DEBUG, msg, fields, _stack_offset=2)
    def info(self, msg: str, **fields): self._log(Level.INFO, msg, fields, _stack_offset=2)
    def warn(self, msg: str, **fields): self._log(Level.WARN, msg, fields, _stack_offset=2)
    def error(self, msg: str, **fields): self._log(Level.ERROR, msg, fields, _stack_offset=2)
    def metric(self, name: str, value: float, unit: str | None = None, **fields):
        self._metric(name, value, unit, fields, _stack_offset=2)


# ---------------------------------------------------------------------------
# Caller capture
# ---------------------------------------------------------------------------

def _capture_exception() -> str | None:
    """Capture the current exception as a formatted traceback string."""
    exc_info = sys.exc_info()
    if exc_info[0] is None:
        return None
    return "".join(traceback.format_exception(*exc_info)).rstrip()


def _get_caller(offset: int, depth: int = 1) -> Caller:
    """Capture call site info via sys._getframe.

    depth > 1 walks up the stack and populates caller.chain with
    the function names from outermost to innermost.
    """
    try:
        frame = sys._getframe(offset)
        caller = Caller(
            file=frame.f_code.co_filename,
            line=frame.f_lineno,
            function=frame.f_code.co_name,
        )
        if depth > 1:
            chain = [frame.f_code.co_name]
            f = frame.f_back
            for _ in range(depth - 1):
                if f is None:
                    break
                name = f.f_code.co_name
                if name == "<module>":
                    break
                chain.append(name)
                f = f.f_back
            chain.reverse()
            caller.chain = chain
        return caller
    except (AttributeError, ValueError):
        return Caller(file="<unknown>", line=0, function="<unknown>")


# ---------------------------------------------------------------------------
# Module-level default logger
# ---------------------------------------------------------------------------

def _level_from_env() -> Level:
    """Read TINYLOG_LEVEL env var, defaulting to INFO."""
    name = os.environ.get("TINYLOG_LEVEL", "").upper()
    for member in Level:
        if member.name == name:
            return member
    return Level.INFO


log = Logger()

# Wire up stderr with human format by default
log.add_sink(sys.stderr, format="human", level=_level_from_env())


def _module_log(level: Level, msg: str, fields: dict[str, Any], _stack_offset: int = 1):
    log._log(level, msg, fields, _stack_offset=_stack_offset + 1)


def _module_metric(name: str, value: float, unit: str | None,
                   fields: dict[str, Any], _stack_offset: int = 1):
    log._metric(name, value, unit, fields, _stack_offset=_stack_offset + 1)
