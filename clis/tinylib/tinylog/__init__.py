"""tinylog — structured logging, metrics, and snapshot testing.

Zero dependencies. Four output formats (json, logfmt, human, gcp).
Metrics and logs share one event type and one pipeline.
Built for snapshot-based testing from day one.
"""

from clis.tinylib.tinylog.tinylog import (
    Level,
    Caller,
    Event,
    Logger,
    Recorder,
    format_json,
    format_logfmt,
    format_human,
    format_gcp,
    log,
)

__all__ = [
    "Level",
    "Caller",
    "Event",
    "Logger",
    "Recorder",
    "format_json",
    "format_logfmt",
    "format_human",
    "format_gcp",
    "log",
]
