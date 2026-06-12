"""Shared async subprocess runner for build scripts."""

from __future__ import annotations

import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anyio

from clis.tinylib.tinylog import log


def _trace_command(cmd: list[str], cwd: Path | None = None) -> str:
    shell_line = shlex.join(cmd)
    if cwd:
        shell_line = f"cd {shlex.quote(str(cwd))} && {shell_line}"
    return shell_line


@dataclass
class Runner:
    """Async subprocess runner with verbose/dry-run support."""

    verbose: bool = False
    dry_run: bool = False
    log_path: Path | None = None
    stage: str | None = None

    def _append_log(self, text: str) -> None:
        if self.log_path is None or self.dry_run:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(text)

    async def __call__(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        trace_command = _trace_command(cmd, cwd)
        if self.dry_run:
            print(trace_command)
            return
        merged_env = {**os.environ, **(env or {})}
        log.trace("run command", cmd=trace_command)
        started = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()
        self._append_log(
            f"\n[{timestamp}] stage={self.stage or 'command'}\n"
            f"cwd={cwd or Path.cwd()}\n"
            f"cmd={trace_command}\n"
        )
        result = await anyio.run_process(
            cmd,
            cwd=cwd,
            env=merged_env,
            check=False,
        )
        duration = time.monotonic() - started
        stdout = result.stdout.decode(errors="replace") if result.stdout else ""
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        if self.verbose:
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
        if stdout:
            self._append_log("\n--- stdout ---\n" + stdout)
        if stderr:
            self._append_log("\n--- stderr ---\n" + stderr)
        self._append_log(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            f"exit_code={result.returncode} duration_s={duration:.1f}\n"
        )
        if result.returncode != 0:
            detail = ""
            if not self.verbose and stderr:
                detail = f"\n{stderr}"
            log.error(
                "command failed",
                exit_code=result.returncode,
                cmd=trace_command,
                stderr=stderr,
            )
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): {trace_command}{detail}"
            )
        log.trace("command complete", cmd=trace_command)
