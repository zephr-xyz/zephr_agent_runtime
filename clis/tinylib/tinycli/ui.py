"""Standalone HTML UI rendering for tinycli pipelines."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
import zipfile
from glob import glob
from dataclasses import dataclass, fields as dc_fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

import anyio

from clis.tinylib.tinycli.effects import PlanEffect
from clis.tinylib.tinycli.plotly_visualization import (
    PLOTLY_CDN_URL,
    PLOTLY_RENDER_JS,
    read_legacy_plotly_loss_html,
    training_loss_figure,
)


UiMode = Literal["plan", "run"]
UiBrowser = Literal["none", "chrome"]
ZEPHR_AGENT_SDK_JS_PATH = Path(__file__).resolve().parents[2] / "support" / "zephr_agent_runtime.js"


class PipelineUiReport:
    """A standalone HTML report with a live-polling JSON sidecar."""

    def __init__(
        self,
        *,
        pipeline_name: str,
        planned: list[Any],
        all_stage_names: list[str],
        previous_state: dict[str, Any] | None,
        output_dir: Path,
        command: str,
        mode: UiMode,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.planned = planned
        self.output_dir = output_dir
        self.html_path = output_dir / "pipeline.html"
        self.state_path = output_dir / "pipeline.state.json"
        self._started_at = time.monotonic()
        now = datetime.now(timezone.utc).isoformat()
        stages = _initial_stage_states(
            planned=planned,
            all_stage_names=all_stage_names,
            previous_state=previous_state,
            output_dir=output_dir,
            mode=mode,
            run_started_at=now,
        )
        self._state = {
            "pipeline": pipeline_name,
            "mode": mode,
            "command": command,
            "planned_stage_names": [item.stage.name for item in planned],
            "all_stage_names": all_stage_names,
            "generated_at": now,
            "updated_at": now,
            "elapsed": 0.0,
            "done": mode == "plan",
            "failed_stage": None,
            "ui_outputs": [
                {
                    "label": "Pipeline UI",
                    "path": str(self.html_path),
                    "state_path": str(self.state_path),
                    "live": mode == "run",
                }
            ],
            "external_dependencies": _external_dependencies(planned),
            "stages": stages,
        }
        for stage in self._state["stages"]:
            if stage.get("execution") == "ran_now":
                _reset_stage_outputs(stage, self.output_dir)
                _delete_stage_state(stage)

    @property
    def url(self) -> str:
        return f"file://{self.html_path.resolve()}"

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_files()

    def mark_stage(
        self,
        stage_name: str,
        status: str,
        *,
        duration: float | None = None,
        error: str | None = None,
        console: str | None = None,
    ) -> None:
        for stage in self._state["stages"]:
            if stage["name"] != stage_name:
                continue
            stage["status"] = status
            if duration is not None:
                stage["duration"] = duration
            if error is not None:
                stage["error"] = error
            if console is not None:
                stage["console"] = console
            break
        self._write_files()

    def update_stage_console(self, stage_name: str, console: str) -> None:
        for stage in self._state["stages"]:
            if stage["name"] != stage_name:
                continue
            stage["console"] = console
            break
        self._write_files()

    def complete(self, *, failed_stage: str | None = None) -> None:
        self._state["done"] = True
        self._state["failed_stage"] = failed_stage
        self._write_files()

    def _write_files(self) -> None:
        self._state["elapsed"] = time.monotonic() - self._started_at
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _refresh_stage_outputs(self._state["stages"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stage in self._state["stages"]:
            _write_stage_state(stage, updated_at=self._state["updated_at"])
        self.state_path.write_text(
            json.dumps(_jsonable(_pipeline_index_state(self._state)), indent=2) + "\n",
            encoding="utf-8",
        )
        self.html_path.write_text(_render_html(self._state, self.state_path.name), encoding="utf-8")


def default_ui_output_dir(pipeline_name: str, planned: list[Any]) -> Path:
    """Choose a current-run UI directory from common plan output fields."""
    for item in planned:
        for obj in (getattr(item, "plan", None), getattr(item, "args", None)):
            if obj is None:
                continue
            output_base = getattr(obj, "output_base", None)
            if output_base is not None:
                return Path(output_base)
            output_dir = getattr(obj, "output_dir", None)
            if output_dir is not None:
                return Path(output_dir)
    return Path(".tinycli") / pipeline_name


async def open_ui_report(report: PipelineUiReport, browser: UiBrowser) -> None:
    """Open a UI report in the requested browser surface."""
    if browser == "none":
        return
    if browser == "chrome":
        await _open_files_in_chrome_for_testing([report.html_path], report.output_dir)
        return
    raise ValueError(f"unsupported UI browser: {browser}")

def _planned_stage_state(
    item: Any,
    *,
    status: str,
    output_dir: Path,
    run_started_at: str,
) -> dict[str, Any]:
    state_path = _stage_state_path(item, output_dir)
    return {
        "name": item.stage.name,
        "status": status,
        "execution": "ran_now",
        "run_started_at": run_started_at,
        "duration": 0.0,
        "error": "",
        "console": "",
        "state_path": str(state_path),
        "state_href": _relative_href(state_path, output_dir),
        "outputs": [],
        "analysis": {},
        "args": _object_fields(getattr(item, "args", None)),
        "plan": _object_fields(getattr(item, "plan", None)),
        "effects": [_effect_state(effect) for effect in getattr(item, "effects", [])],
    }


def _initial_stage_states(
    *,
    planned: list[Any],
    all_stage_names: list[str],
    previous_state: dict[str, Any] | None,
    output_dir: Path,
    mode: UiMode,
    run_started_at: str,
) -> list[dict[str, Any]]:
    planned_by_name = {item.stage.name: item for item in planned}
    planned_names = [item.stage.name for item in planned]
    if planned_names:
        last_planned_index = max(all_stage_names.index(name) for name in planned_names)
        flow_names = all_stage_names[: last_planned_index + 1]
    else:
        flow_names = []

    previous_by_name = {
        str(stage.get("name")): stage
        for stage in (previous_state or {}).get("stages", [])
        if isinstance(stage, dict) and stage.get("name")
    }
    stages: list[dict[str, Any]] = []
    for name in flow_names:
        planned_item = planned_by_name.get(name)
        if planned_item is not None:
            stages.append(
                _planned_stage_state(
                    planned_item,
                    status="planned" if mode == "plan" else "pending",
                    output_dir=output_dir,
                    run_started_at=run_started_at,
                )
            )
            continue
        previous = previous_by_name.get(name)
        if previous is not None:
            stages.append(_reused_stage_state(previous))
        else:
            stages.append(_missing_stage_state(name))
    return stages


def _reused_stage_state(previous: dict[str, Any]) -> dict[str, Any]:
    stage = _load_stage_state_from_index(previous) or dict(previous)
    stage.update({
        "name": previous.get("name") or stage.get("name"),
        "status": "reused",
        "execution": "reused",
        "state_path": previous.get("state_path") or stage.get("state_path", ""),
        "state_href": previous.get("state_href") or stage.get("state_href", ""),
        "reused_status": stage.get("status"),
    })
    return stage


def _missing_stage_state(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "missing",
        "execution": "missing",
        "duration": 0.0,
        "error": "No previous stage state was found",
        "console": "",
        "state_path": "",
        "state_href": "",
        "outputs": [],
        "analysis": {},
        "args": {},
        "plan": {},
        "effects": [],
    }


def _load_stage_state_from_index(stage_index: dict[str, Any]) -> dict[str, Any] | None:
    path_text = str(stage_index.get("state_path") or "")
    if not path_text:
        return None
    try:
        value = json.loads(Path(path_text).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _stage_state_path(item: Any, output_dir: Path) -> Path:
    stage_name = item.stage.name
    plan = getattr(item, "plan", None)
    explicit = getattr(plan, "stage_state_path", None)
    if explicit is not None:
        return Path(explicit)

    output_base = getattr(plan, "output_base", None)
    if output_base is not None:
        output_base = Path(output_base)
        stage_root = _stage_root_from_effects(output_base, stage_name, getattr(item, "effects", []))
        if stage_root is not None:
            return stage_root / "stage_state.json"
        return output_base / ".tinycli" / "stages" / f"{stage_name}.stage_state.json"

    return output_dir / ".tinycli" / "stages" / f"{stage_name}.stage_state.json"


def _stage_root_from_effects(
    output_base: Path,
    stage_name: str,
    effects: list[PlanEffect],
) -> Path | None:
    candidates: list[Path] = []
    for effect in effects:
        path = effect.path
        if path is None or effect.kind not in {"write_dir", "write_file"}:
            continue
        candidate = path if effect.kind == "write_dir" else path.parent
        try:
            relative = candidate.relative_to(output_base)
        except ValueError:
            continue
        if not relative.parts:
            continue
        root = output_base / relative.parts[0]
        root_name = root.name.lower()
        normalized_stage = stage_name.lower().replace("_", "")
        normalized_root = root_name.replace("_", "")
        if re.match(r"^\d+_", root_name) or normalized_stage in normalized_root:
            candidates.append(root)
    return candidates[0] if candidates else None


def _relative_href(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def _delete_stage_state(stage: dict[str, Any]) -> None:
    path_text = str(stage.get("state_path") or "")
    if not path_text:
        return
    path = Path(path_text)
    path.unlink(missing_ok=True)


def _reset_stage_outputs(stage: dict[str, Any], output_dir: Path) -> None:
    if stage.get("name") == "analyze":
        return
    for root in _stage_cleanup_roots(stage, output_dir):
        if root.is_symlink() or root.is_file():
            root.unlink(missing_ok=True)
        elif root.is_dir():
            shutil.rmtree(root)


def _stage_cleanup_roots(stage: dict[str, Any], output_dir: Path) -> list[Path]:
    roots: list[Path] = []
    output_base = _stage_output_base(stage)
    state_path_text = str(stage.get("state_path") or "")
    if state_path_text:
        state_path = Path(state_path_text)
        parent = state_path.parent
        if parent != output_dir and ".tinycli" not in parent.parts:
            roots.append(parent)

    for effect in stage.get("effects", []):
        if not isinstance(effect, dict) or effect.get("target_kind") != "path":
            continue
        kind = effect.get("kind")
        if kind not in {"write_dir", "write_file"}:
            continue
        target = str(effect.get("target") or "")
        if not target:
            continue
        path = Path(target)
        candidate = path if kind == "write_dir" else path.parent
        if output_base is not None:
            try:
                relative = candidate.relative_to(output_base)
            except ValueError:
                continue
            if not relative.parts:
                continue
            candidate = output_base / relative.parts[0]
        elif candidate == output_dir:
            continue
        if candidate == output_dir or ".tinycli" in candidate.parts:
            continue
        roots.append(candidate)

    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            result.append(root)
    return result


def _write_stage_state(stage: dict[str, Any], *, updated_at: str) -> None:
    if stage.get("execution") != "ran_now":
        return
    path_text = str(stage.get("state_path") or "")
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_base = {
        key: value
        for key, value in stage.items()
        if key not in {"stage_revision", "stage_updated_at", "updated_at", "_stage_payload_fingerprint"}
    }
    fingerprint = json.dumps(
        _stage_revision_payload(payload_base),
        sort_keys=True,
        separators=(",", ":"),
    )
    if path.exists() and stage.get("_stage_payload_fingerprint") == fingerprint:
        return
    stage["_stage_payload_fingerprint"] = fingerprint
    stage["stage_revision"] = int(stage.get("stage_revision") or 0) + 1
    stage["stage_updated_at"] = updated_at
    payload = dict(stage)
    payload.pop("_stage_payload_fingerprint", None)
    payload["updated_at"] = updated_at
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _stage_revision_payload(stage: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(_jsonable(stage)))
    outputs = payload.get("outputs")
    if isinstance(outputs, list):
        for output in outputs:
            if isinstance(output, dict):
                output.pop("size_bytes", None)
                output.pop("mtime", None)
    return payload


def _pipeline_index_state(state: dict[str, Any]) -> dict[str, Any]:
    index = dict(state)
    index["stages"] = [_stage_index_state(stage) for stage in state.get("stages", [])]
    index["embedded_stage_count"] = len(state.get("stages", []))
    return index


def _stage_index_state(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": stage.get("name"),
        "status": stage.get("status"),
        "execution": stage.get("execution", "ran_now"),
        "run_started_at": stage.get("run_started_at", ""),
        "duration": stage.get("duration", 0.0),
        "error": stage.get("error", ""),
        "state_path": stage.get("state_path", ""),
        "state_href": stage.get("state_href", ""),
        "stage_revision": stage.get("stage_revision", 0),
        "stage_updated_at": stage.get("stage_updated_at", ""),
    }


def _external_dependencies(planned: list[Any]) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in planned:
        for effect in getattr(item, "effects", []):
            if getattr(effect, "kind", None) != "network":
                continue
            target = effect.subject or str(effect.path or effect.pattern or "")
            key = (item.stage.name, effect.label, target)
            if key in seen:
                continue
            seen.add(key)
            dependencies.append(
                {
                    "stage": item.stage.name,
                    "label": effect.label,
                    "target": target,
                    "required": effect.required,
                }
            )
    return dependencies


def _refresh_stage_outputs(stages: list[dict[str, Any]]) -> None:
    for stage in stages:
        stage["outputs"] = _stage_outputs(stage)
        stage["analysis"] = _stage_analysis(stage)


def _stage_outputs(stage: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for effect in stage.get("effects", []):
        if not _is_report_output_effect(effect):
            continue
        target = str(effect.get("target") or "")
        if not target:
            continue
        paths: list[Path] = []
        if effect.get("target_kind") == "pattern":
            paths = [Path(item) for item in sorted(glob(target))]
        else:
            path = Path(target)
            if path.exists():
                paths = [path]
                if path.is_dir():
                    paths.extend(sorted(path.glob("*.html")))
        for path in paths:
            if path.is_file() and not _path_is_current_for_stage(path, stage):
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            outputs.append(_output_state(path, str(effect.get("label") or "")))
            if len(outputs) >= 80:
                return outputs
    return outputs


def _is_report_output_effect(effect: dict[str, Any]) -> bool:
    kind = effect.get("kind")
    if kind not in {"write_file", "write_dir", "write_pattern"}:
        return False
    if kind != "write_dir":
        return True
    label = str(effect.get("label") or "").lower()
    return "work" not in label


def _output_state(path: Path, label: str) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "label": label, "exists": False}
    return {
        "path": str(path),
        "href": str(path.resolve()),
        "label": label,
        "exists": True,
        "kind": "dir" if path.is_dir() else "file",
        "size_bytes": stat.st_size if path.is_file() else None,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "html": path.suffix.lower() == ".html",
    }


def _stage_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    name = stage.get("name")
    if name == "convert":
        return _convert_analysis(stage)
    if name == "assemble_models":
        return _assemble_models_analysis(stage)
    if name == "train":
        return _train_analysis(stage)
    if name == "collect":
        return _collect_analysis(stage)
    if name == "analyze":
        return _analyze_stage_analysis(stage)
    return {}


def _stage_output_base(stage: dict[str, Any]) -> Path | None:
    for source in (stage.get("plan"), stage.get("args")):
        if isinstance(source, dict) and source.get("output_base"):
            return Path(str(source["output_base"]))
    return None


def _path_is_current_for_stage(path: Path, stage: dict[str, Any]) -> bool:
    if stage.get("execution") != "ran_now":
        return True
    started_at = str(stage.get("run_started_at") or "")
    if not started_at:
        return True
    try:
        started = datetime.fromisoformat(started_at).timestamp()
        mtime = path.stat().st_mtime
    except (OSError, ValueError):
        return False
    return mtime >= started - 1.0


def _train_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    output_base = _stage_output_base(stage)
    if output_base is None:
        return {"kind": "train"}
    train_dir = output_base / "2_train" / "model"
    losses = _read_loss_series(train_dir)
    eval_summary = _read_eval_summary(train_dir / "eval_results.txt")
    return {
        "kind": "train",
        "losses": losses,
        "figures": {
            "loss": training_loss_figure(losses),
        },
        "eval": eval_summary,
    }


def _convert_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    output_base = _stage_output_base(stage)
    if output_base is None:
        return {"kind": "convert", "artifacts": []}
    manifest_path = output_base / "3_convert" / "converted_model_manifest.json"
    if not _path_is_current_for_stage(manifest_path, stage):
        return {"kind": "convert", "artifacts": []}
    manifest = _read_json_object(manifest_path)
    artifacts = _manifest_artifact_rows(manifest.get("artifacts"), base_dir=manifest_path.parent / "models")
    by_role: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        role = str(artifact.get("role") or "unknown")
        row = by_role.setdefault(role, {"role": role, "count": 0, "size_bytes": 0})
        row["count"] += 1
        row["size_bytes"] += int(artifact.get("size_bytes") or 0)
    npu_targets = sorted({
        str((artifact.get("metadata") or {}).get("hardware_target"))
        for artifact in artifacts
        if isinstance(artifact.get("metadata"), dict)
        and (artifact.get("metadata") or {}).get("hardware_target")
    })
    return {
        "kind": "convert",
        "manifest_path": str(manifest_path),
        "manifest_href": str(manifest_path.resolve()) if manifest_path.is_file() else "",
        "schema_version": manifest.get("schema_version"),
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(int(artifact.get("size_bytes") or 0) for artifact in artifacts),
        "by_role": sorted(by_role.values(), key=lambda row: str(row.get("role") or "")),
        "npu_targets": npu_targets,
        "artifacts": artifacts,
    }


def _assemble_models_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    output_base = _stage_output_base(stage)
    if output_base is None:
        return {"kind": "assemble_models", "artifacts": [], "roles": {}}
    manifest_path = output_base / "4_assemble_models" / "models" / "local.json"
    if not _path_is_current_for_stage(manifest_path, stage):
        return {"kind": "assemble_models", "artifacts": [], "roles": {}}
    manifest = _read_json_object(manifest_path)
    artifacts = _manifest_artifact_rows(manifest.get("artifacts"), base_dir=manifest_path.parent)
    roles = manifest.get("roles") if isinstance(manifest.get("roles"), dict) else {}
    return {
        "kind": "assemble_models",
        "manifest_path": str(manifest_path),
        "manifest_href": str(manifest_path.resolve()) if manifest_path.is_file() else "",
        "manifest_id": manifest.get("id"),
        "channel": manifest.get("channel"),
        "created_at": manifest.get("created_at"),
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(int(artifact.get("size_bytes") or 0) for artifact in artifacts),
        "artifacts": artifacts,
        "roles": roles,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_artifact_rows(value: Any, *, base_dir: Path) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows: list[dict[str, Any]] = []
    for artifact_id, artifact in sorted(value.items()):
        if not isinstance(artifact, dict):
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        capabilities = artifact.get("capabilities") if isinstance(artifact.get("capabilities"), list) else []
        filename = str(artifact.get("filename") or "")
        path = base_dir / filename if filename else None
        rows.append({
            "id": str(artifact_id),
            "title": str(artifact.get("title") or artifact_id),
            "role": str(artifact.get("role") or ""),
            "family": str(artifact.get("family") or ""),
            "capabilities": [str(item) for item in capabilities],
            "filename": filename,
            "size_bytes": int(artifact.get("size_bytes") or 0),
            "sha256": str(artifact.get("sha256") or ""),
            "metadata": metadata,
            "variant": str(metadata.get("variant") or ""),
            "hardware_target": str(metadata.get("hardware_target") or ""),
            "source_artifact": str(metadata.get("source_artifact") or ""),
            "href": str(path.resolve()) if path is not None and path.exists() else "",
        })
    return rows


def _read_loss_series(train_dir: Path) -> dict[str, list[dict[str, float]]]:
    json_path = train_dir / "losses.json"
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            return {
                "train": _normalize_loss_points(payload.get("train")),
                "eval": _normalize_loss_points(payload.get("eval")),
            }
        except (OSError, json.JSONDecodeError):
            pass
    html_path = train_dir / "losses.html"
    if html_path.is_file():
        return read_legacy_plotly_loss_html(html_path)
    return {"train": [], "eval": []}


def _normalize_loss_points(value: Any) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    if not isinstance(value, list):
        return points
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            points.append({"epoch": float(item["epoch"]), "loss": float(item["loss"])})
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _read_eval_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    summary: dict[str, Any] = {}
    match = re.search(r"Success:\s*(\d+)\s*/\s*(\d+)\s*\(([^)]+)\)", text)
    if match is not None:
        passed = int(match.group(1))
        total = int(match.group(2))
        summary.update({
            "passed": passed,
            "total": total,
            "accuracy": (passed / total) if total else 0,
            "label": match.group(0),
        })
    summary["failures"] = text.count("-> FAIL")
    summary["passes"] = text.count("-> PASS")
    return summary


def _collect_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    output_base = _stage_output_base(stage)
    if output_base is None:
        return {"kind": "collect", "runs": []}
    return {
        "kind": "collect",
        "runs": _collect_run_summaries(output_base / "5_collect", stage=stage),
    }


def _collect_run_summaries(
    collect_dir: Path,
    *,
    stage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for results_path in sorted(collect_dir.glob("*/*/results.jsonl")):
        if stage is not None and not _path_is_current_for_stage(results_path, stage):
            continue
        platform = results_path.parent.parent.name
        profile = results_path.parent.name
        records = _read_jsonl_records(results_path)
        rows.append(_summarize_collect_records(
            platform=platform,
            profile=profile,
            results_path=results_path,
            records=records,
        ))
    for results_path in sorted((collect_dir / "runs").glob("*/*/*/results.jsonl")):
        if stage is not None and not _path_is_current_for_stage(results_path, stage):
            continue
        suite_run = results_path.parents[2].name
        platform = f"{suite_run}/{results_path.parent.parent.name}"
        profile = results_path.parent.name
        records = _read_jsonl_records(results_path)
        rows.append(_summarize_collect_records(
            platform=platform,
            profile=profile,
            results_path=results_path,
            records=records,
        ))
    for results_path in sorted((collect_dir / "dataset_replay").glob("*/*.results.jsonl")):
        if stage is not None and not _path_is_current_for_stage(results_path, stage):
            continue
        platform = results_path.parent.name
        profile = results_path.name.removesuffix(".results.jsonl")
        records = _read_jsonl_records(results_path)
        rows.append(_summarize_dataset_replay_records(
            platform=platform,
            profile=profile,
            results_path=results_path,
            records=records,
        ))
    for results_path in sorted((collect_dir / "runs").glob("*/dataset_replay/*/*.results.jsonl")):
        if stage is not None and not _path_is_current_for_stage(results_path, stage):
            continue
        suite_run = results_path.parents[2].name
        platform = f"{suite_run}/dataset_replay/{results_path.parent.name}"
        profile = results_path.name.removesuffix(".results.jsonl")
        records = _read_jsonl_records(results_path)
        rows.append(_summarize_dataset_replay_records(
            platform=platform,
            profile=profile,
            results_path=results_path,
            records=records,
        ))
    return rows


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _summarize_collect_records(
    *,
    platform: str,
    profile: str,
    results_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    turns = [
        record for record in records
        if record.get("recordType") == "conversation_turn"
    ]
    scored = turns
    summaries = [
        record for record in records
        if record.get("recordType") == "run_summary"
    ]
    summary_failed_turns = sum(int(record.get("failedTurns") or 0) for record in summaries)
    summary_generation_failures = sum(int(record.get("generationFailures") or 0) for record in summaries)
    passed = sum(1 for record in scored if _collect_record_passed(record))
    durations = [
        int(record.get(key) or 0)
        for record in scored
        for key in ("elapsedMs", "durationMs", "prefillMs", "decodeMs")
        if record.get(key) is not None
    ]
    lifecycle_failures = [
        record for record in records
        if record.get("recordType") == "lifecycle" and record.get("status") == "failed"
    ]
    return {
        "platform": platform,
        "profile": profile,
        "path": str(results_path),
        "href": str(results_path.resolve()),
        "records": len(records),
        "validation_records": len(scored),
        "passed": passed,
        "failed": max(0, len(scored) - passed, summary_failed_turns + summary_generation_failures) + len(lifecycle_failures),
        "accuracy": (passed / len(scored)) if scored else None,
        "avg_ms": (sum(durations) / len(durations)) if durations else None,
        "p95_ms": _percentile(durations, 0.95),
        "lifecycle_failures": len(lifecycle_failures),
    }


def _summarize_dataset_replay_records(
    *,
    platform: str,
    profile: str,
    results_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    turns = [
        record for record in records
        if record.get("recordType") == "conversation_turn"
    ]
    failure_events = [
        record for record in records
        if record.get("recordType") == "sdk_event"
        and record.get("eventType") == "generation_failure"
    ]
    summaries = [
        record for record in records
        if record.get("recordType") == "run_summary"
    ]
    failed_turns = sum(1 for record in turns if record.get("error"))
    replay_mismatch_turns = sum(
        1 for record in turns
        if record.get("toolReplayMismatches")
    )
    summary_failed_turns = sum(int(record.get("failedTurns") or 0) for record in summaries)
    summary_generation_failures = sum(int(record.get("generationFailures") or 0) for record in summaries)
    summary_replay_mismatches = sum(int(record.get("toolReplayMismatchTurns") or 0) for record in summaries)
    failed = max(failed_turns, summary_failed_turns) + max(len(failure_events), summary_generation_failures) + max(
        replay_mismatch_turns,
        summary_replay_mismatches,
    )
    passed = max(0, len(turns) - failed_turns - replay_mismatch_turns)
    durations = [
        int((record.get("prefillMs") or 0)) + int((record.get("decodeMs") or 0))
        for record in turns
        if record.get("prefillMs") is not None or record.get("decodeMs") is not None
    ]
    return {
        "kind": "dataset_replay",
        "platform": f"dataset_replay/{platform}",
        "profile": profile,
        "path": str(results_path),
        "href": str(results_path.resolve()),
        "records": len(records),
        "validation_records": len(turns),
        "passed": passed,
        "failed": failed,
        "accuracy": (passed / len(turns)) if turns else None,
        "avg_ms": (sum(durations) / len(durations)) if durations else None,
        "p95_ms": _percentile(durations, 0.95),
        "lifecycle_failures": len(failure_events),
        "timeline": _dataset_replay_timeline(records),
    }


def _dataset_replay_timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        record_type = str(record.get("recordType") or "")
        if record_type == "conversation_setup":
            timeline.append({
                "index": index,
                "kind": "setup",
                "title": str(record.get("runId") or "conversation setup"),
                "detail": ", ".join(
                    part for part in [
                        str(record.get("backend") or ""),
                        str(record.get("platform") or ""),
                        f"channel={record.get('modelChannel')}" if record.get("modelChannel") else "",
                        f"llm={record.get('llmExecutionChoiceID')}" if record.get("llmExecutionChoiceID") else "",
                        str(record.get("conversationStrategy") or record.get("conversation_strategy") or ""),
                        str(record.get("execution") or ""),
                    ]
                    if part
                ),
            })
        elif record_type == "sdk_event":
            event_type = str(record.get("eventType") or "")
            title_parts = [
                event_type,
                str(record.get("kind") or ""),
                str(record.get("label") or ""),
            ]
            timeline.append({
                "index": index,
                "kind": "sdk_event",
                "title": " / ".join(part for part in title_parts if part),
                "detail": _short_text(
                    record.get("reason")
                    or record.get("name")
                    or record.get("response")
                    or record.get("prompt")
                    or ""
                ),
            })
        elif record_type == "conversation_turn":
            dataset_user_text_index = record.get("datasetUserTextIndex")
            title = (
                f"dataset user text {dataset_user_text_index}"
                if dataset_user_text_index is not None
                else "turn"
            )
            error = record.get("error")
            tool_calls = record.get("toolCalls")
            detail_parts = [
                _short_text(record.get("userText") or ""),
                _short_text(record.get("queryText") or ""),
                f"tools={tool_calls}" if tool_calls is not None else "",
                f"error={_short_text(error)}" if error else "",
            ]
            timeline.append({
                "index": index,
                "kind": "conversation_turn",
                "title": title,
                "detail": " | ".join(part for part in detail_parts if part),
            })
        elif record_type == "run_summary":
            timeline.append({
                "index": index,
                "kind": "summary",
                "title": "run summary",
                "detail": (
                    f"queries={record.get('queries', '')} "
                    f"failed={record.get('failedTurns', 0)} "
                    f"generationFailures={record.get('generationFailures', 0)} "
                    f"replayMismatches={record.get('toolReplayMismatchTurns', 0)}"
                ),
            })
    return timeline[:80]


def _short_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _collect_record_passed(record: dict[str, Any]) -> bool:
    if "passed" in record:
        return bool(record.get("passed"))
    if record.get("recordType") == "conversation_turn":
        diagnostic = record.get("diagnostic")
        if isinstance(diagnostic, dict) and "passed" in diagnostic:
            return bool(diagnostic.get("passed"))
        return not bool(record.get("error"))
    return record.get("status") in {"passed", "skipped"}


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _analyze_stage_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    output_base = _stage_output_base(stage)
    if output_base is None:
        return {"kind": "analyze", "runs": []}
    runs_dir = output_base / "6_analyze" / "runs"
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir() if runs_dir.is_dir() else [], reverse=True):
        if not run_dir.is_dir() or run_dir.is_symlink():
            continue
        if not _path_is_current_for_stage(run_dir, stage):
            continue
        info_path = run_dir / "run_info.json"
        run: dict[str, Any] = {
            "id": run_dir.name,
            "path": str(run_dir),
            "href": str(run_dir.resolve()),
            "diagnostics_href": str((run_dir / "diagnostics.html").resolve())
            if (run_dir / "diagnostics.html").is_file() else "",
            "collect": _collect_run_summaries(run_dir),
        }
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                run["timestamp"] = info.get("timestamp")
                run["model_channel"] = info.get("model_channel")
                run["diagnostic_dataset_id"] = info.get("diagnostic_dataset_id")
            except (OSError, json.JSONDecodeError):
                pass
        runs.append(run)
    return {"kind": "analyze", "runs": runs}


def _object_fields(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dc_fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return {"value": _jsonable(value)}


def _effect_state(effect: PlanEffect) -> dict[str, Any]:
    if effect.path is not None:
        target = str(effect.path)
        target_kind = "path"
    elif effect.pattern is not None:
        target = effect.pattern
        target_kind = "pattern"
    elif effect.subject is not None:
        target = effect.subject
        target_kind = "subject"
    else:
        target = ""
        target_kind = ""
    return {
        "kind": effect.kind,
        "label": effect.label,
        "target": target,
        "target_kind": target_kind,
        "required": effect.required,
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _object_fields(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class UiPage:
    title: str
    state: dict[str, Any]
    state_file_name: str | None
    css: str
    js: str
    init_js: str
    mount_id: str = "app"
    include_plotly: bool = False


def _render_html(state: dict[str, Any], state_file_name: str) -> str:
    include_plotly = _state_uses_plotly(state)
    return _render_standalone_page(
        UiPage(
            title=f"{state['pipeline']} pipeline",
            state=state,
            state_file_name=state_file_name,
            css=_PIPELINE_PAGE_CSS + "\n" + _STAGE_PAGE_CSS,
            js=(
                (PLOTLY_RENDER_JS + "\n" if include_plotly else "")
                + _zephr_agent_runtime_js()
                + "\n"
                + _PIPELINE_PAGE_JS
                + "\n"
                + _STAGE_PAGE_JS
            ),
            init_js=(
                "const app = createPipelinePage("
                "document.getElementById('app'), EMBEDDED_STATE);\n"
                "startStatePolling({\n"
                "  stateFile: STATE_FILE,\n"
                "  initialState: EMBEDDED_STATE,\n"
                "  onState: state => app.update(state),\n"
                "  onTick: () => app.tick(),\n"
                "});"
            ),
            include_plotly=include_plotly,
        )
    )


def _zephr_agent_runtime_js() -> str:
    return ZEPHR_AGENT_SDK_JS_PATH.read_text(encoding="utf-8")


def _state_uses_plotly(state: dict[str, Any]) -> bool:
    return any(
        bool((stage.get("analysis") or {}).get("figures"))
        for stage in state.get("stages", [])
    )


def _render_standalone_page(page: UiPage) -> str:
    state_json = json.dumps(_jsonable(page.state), indent=2).replace("</", "<\\/")
    state_file_json = json.dumps(page.state_file_name)
    plotly_script = (
        f'<script src="{_html_escape(PLOTLY_CDN_URL)}"></script>\n'
        if page.include_plotly else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html_escape(page.title)}</title>
{plotly_script}<script>
window.TINYCLI_EXTERNAL_DEPENDENCIES = {json.dumps({"plotly": PLOTLY_CDN_URL}) if page.include_plotly else "{}"};
</script>
<style>
{_COMMON_PAGE_CSS}
{page.css}
</style>
</head>
<body>
<main id="{_html_escape(page.mount_id)}"></main>
<script type="module">
const EMBEDDED_STATE = {state_json};
const STATE_FILE = {state_file_json};
{_COMMON_PAGE_JS}
{page.js}
{page.init_js}
</script>
</body>
</html>
"""


_COMMON_PAGE_CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f8fafc;
  --panel: #ffffff;
  --text: #18202f;
  --muted: #667085;
  --line: #d9e2ec;
  --accent: #0f766e;
  --danger: #c2410c;
  --ok: #15803d;
  --pending: #64748b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101418;
    --panel: #171d23;
    --text: #ecf2f8;
    --muted: #9aa8b7;
    --line: #2f3a45;
    --accent: #2dd4bf;
    --danger: #fb923c;
    --ok: #4ade80;
    --pending: #94a3b8;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  width: 100%;
  max-width: none;
  margin: 0 auto;
  padding: 24px;
}
header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 18px;
}
h1 {
  margin: 0 0 6px;
  font-size: 24px;
  letter-spacing: 0;
}
.subtle, .meta { color: var(--muted); }
.meta { text-align: right; white-space: nowrap; }
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  margin: 10px 0 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}
@media (max-width: 700px) {
  main { padding: 16px; }
  header { display: block; }
  .meta { text-align: left; margin-top: 8px; white-space: normal; }
}
"""


_PIPELINE_PAGE_CSS = r"""
.bar {
  height: 10px;
  background: color-mix(in srgb, var(--line) 70%, transparent);
  border: 1px solid var(--line);
  border-radius: 5px;
  overflow: hidden;
  margin: 16px 0 22px;
}
.bar-fill {
  height: 100%;
  width: 0;
  background: var(--accent);
  transition: width 160ms ease-out;
}
.stage {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 12px 0;
  overflow: hidden;
}
.ui-output {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: baseline;
  margin: 0 0 8px;
  color: var(--muted);
}
.ui-output code {
  color: var(--text);
  overflow-wrap: anywhere;
}
.stage-head {
  display: grid;
  grid-template-columns: 18px minmax(140px, 1fr) auto auto;
  gap: 12px;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
}
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--pending);
}
.spinner {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 2px solid color-mix(in srgb, var(--accent) 25%, transparent);
  border-top-color: var(--accent);
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
.status-success .dot { background: var(--ok); }
.status-error .dot { background: var(--danger); }
.stage-name { font-weight: 700; }
.badge {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 8px;
  color: var(--muted);
  font-size: 12px;
}
.report-link {
  color: var(--accent);
  font-size: 12px;
  text-decoration: none;
}
.report-link:hover { text-decoration: underline; }
.output-link {
  color: var(--accent);
  overflow-wrap: anywhere;
}
.stage-body { padding: 14px 16px 16px; }
.error {
  color: var(--danger);
  margin-bottom: 12px;
  font-weight: 600;
}
.grid {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
details {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  background: color-mix(in srgb, var(--panel) 88%, var(--bg));
}
summary { cursor: pointer; font-weight: 650; }
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}
td, th {
  padding: 6px 8px;
  border-top: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
td.kind {
  width: 120px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--accent);
}
td.target {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  overflow-wrap: anywhere;
}
.console-pre {
  max-height: 360px;
  overflow: auto;
  background: color-mix(in srgb, var(--bg) 80%, #000);
  border-radius: 6px;
  padding: 10px;
}
.timeline-list {
  display: grid;
  gap: 8px;
  margin-top: 10px;
}
.timeline-item {
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr);
  gap: 10px;
  padding: 8px 0;
  border-top: 1px solid var(--line);
}
.timeline-kind {
  color: var(--accent);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.timeline-title {
  font-weight: 650;
  overflow-wrap: anywhere;
}
.timeline-detail {
  color: var(--muted);
  margin-top: 2px;
  overflow-wrap: anywhere;
}
@media (max-width: 700px) {
  .stage-head { grid-template-columns: 12px 1fr; }
  .badge, .duration { grid-column: 2; justify-self: start; }
}
"""


_STAGE_PAGE_CSS = r"""
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin: 14px 0 18px;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--panel);
}
.metric-label {
  color: var(--muted);
  font-size: 12px;
}
.metric-value {
  font-weight: 700;
  margin-top: 4px;
}
.output-link {
  color: var(--accent);
  overflow-wrap: anywhere;
}
.analysis-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: var(--panel);
}
.analysis-panel:empty { display: none; }
.plotly-figure {
  width: 100%;
  height: 320px;
}
.compact-table td, .compact-table th {
  font-size: 12px;
}
.timeline-list {
  display: grid;
  gap: 8px;
  margin-top: 10px;
}
.timeline-item {
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr);
  gap: 10px;
  padding: 8px 0;
  border-top: 1px solid var(--line);
}
.timeline-kind {
  color: var(--accent);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.timeline-title {
  font-weight: 650;
  overflow-wrap: anywhere;
}
.timeline-detail {
  color: var(--muted);
  margin-top: 2px;
  overflow-wrap: anywhere;
}
"""


_COMMON_PAGE_JS = r"""
let lastStateText = null;
let lastPollAt = null;
let lastChangeAt = Date.parse(EMBEDDED_STATE.updated_at || EMBEDDED_STATE.generated_at || "") || Date.now();
let liveFailed = false;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value == null ? "" : String(value);
    else node.setAttribute(name, value);
  }
  for (const child of children) node.append(child);
  return node;
}

function setText(node, value) {
  const text = value == null ? "" : String(value);
  if (node.textContent !== text) node.textContent = text;
}

function setClassName(node, value) {
  if (node.className !== value) node.className = value;
}

function fmtDuration(seconds) {
  seconds = Number(seconds || 0);
  if (seconds < 60) return seconds.toFixed(1) + "s";
  const minutes = Math.floor(seconds / 60);
  return minutes + "m " + Math.round(seconds % 60) + "s";
}

function fmtAge(isoString) {
  const then = Date.parse(isoString || "");
  if (!Number.isFinite(then)) return "";
  return fmtAgeMs(Date.now() - then);
}

function fmtAgeMs(ms) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return seconds + "s ago";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return minutes + "m ago";
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return hours + "h ago";
  const days = Math.floor(hours / 24);
  return days + "d ago";
}

function fmtGeneratedAt(isoString) {
  const then = new Date(isoString || "");
  if (!Number.isFinite(then.getTime())) return "";
  return then.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function stateUrl(stateFile) {
  return location.href.replace(/[^/]*$/, stateFile);
}

function fetchJsonFile(href, callback) {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", stateUrl(href), true);
  xhr.onload = function() {
    if (xhr.status && xhr.status >= 400) {
      callback(new Error("status " + xhr.status), null);
      return;
    }
    try {
      callback(null, JSON.parse(xhr.responseText));
    } catch (err) {
      callback(err, null);
    }
  };
  xhr.onerror = function() {
    callback(new Error("request failed"), null);
  };
  xhr.send();
}

function stageSidecarIsExpected(stageRef) {
  const status = stageRef && stageRef.status;
  return status !== "planned" && status !== "pending" && status !== "running";
}

function stageSidecarRevision(stage) {
  if (!stage) return "";
  if (stage.stage_revision != null) return String(stage.stage_revision);
  return stage.stage_updated_at || stage.updated_at || "";
}

function composePipelineState(indexState, fallbackState, callback) {
  const stageIndex = indexState.stages || [];
  const needsExternalStages = stageIndex.some(stage => stage && stage.state_href);
  if (!needsExternalStages) {
    callback(indexState);
    return;
  }

  const fallbackByName = new Map((fallbackState.stages || []).map(stage => [stage.name, stage]));
  const stages = new Array(stageIndex.length);
  let pending = stageIndex.length;
  let failed = false;

  if (!pending) {
    callback({ ...fallbackState, ...indexState, stages: [], external_refresh_failed: false });
    return;
  }

  stageIndex.forEach((stageRef, index) => {
    const fallbackStage = fallbackByName.get(stageRef.name) || {};
    if (!stageRef.state_href) {
      stages[index] = { ...fallbackStage, ...stageRef };
      pending -= 1;
      if (!pending) callback({ ...fallbackState, ...indexState, stages, external_refresh_failed: failed });
      return;
    }
    if (
      fallbackStage.state_href === stageRef.state_href
      && stageSidecarRevision(fallbackStage) === stageSidecarRevision(stageRef)
    ) {
      stages[index] = { ...fallbackStage, ...stageRef };
      pending -= 1;
      if (!pending) callback({ ...fallbackState, ...indexState, stages, external_refresh_failed: failed });
      return;
    }
    fetchJsonFile(stageRef.state_href, (err, stageState) => {
      if (err || !stageState) {
        if (stageSidecarIsExpected(stageRef)) failed = true;
        stages[index] = { ...fallbackStage, ...stageRef };
      } else {
        stages[index] = { ...stageState, ...stageRef };
      }
      pending -= 1;
      if (!pending) callback({ ...fallbackState, ...indexState, stages, external_refresh_failed: failed });
    });
  });
}

function startStatePolling({ stateFile, initialState, onState, onTick }) {
  let currentState = initialState;
  function applyState(nextState) {
    currentState = nextState;
    onState(nextState);
  }
  function poll() {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", stateUrl(stateFile), true);
    xhr.onload = function() {
      lastPollAt = Date.now();
      liveFailed = false;
      const text = xhr.responseText;
      if (text && text !== lastStateText) {
        lastStateText = text;
        lastChangeAt = lastPollAt;
        try {
          composePipelineState(JSON.parse(text), currentState, applyState);
        } catch (_err) {}
      } else {
        onTick();
      }
      if (!currentState.done) setTimeout(poll, 1500);
    };
    xhr.onerror = function() {
      liveFailed = true;
      onState({ ...currentState, external_refresh_failed: true });
      if (!currentState.done) setTimeout(poll, 4000);
    };
    xhr.send();
  }

  onState(initialState);
  setInterval(onTick, 1000);
  if (!initialState.done && stateFile) setTimeout(poll, 1500);
}
"""


_PIPELINE_PAGE_JS = r"""
function statusClass(status) {
  return status === "failed" ? "error" : status;
}

function renderModeLabel(state) {
  if (state.mode === "plan") return "Plan preview";
  if (!state.done) return "Running";
  return state.failed_stage ? "Run complete with errors" : "Run complete";
}

function valueText(value) {
  return typeof value === "object" && value !== null
    ? JSON.stringify(value, null, 2)
    : String(value);
}

function countConsoleLines(text) {
  if (!text) return 0;
  return text.split("\n").filter(line => line.length > 0).length;
}

function consoleDisplayText(text) {
  const maxChars = 50000;
  if (!text || text.length <= maxChars) return text || "";
  return "[showing last " + maxChars + " of " + text.length + " chars]\n" + text.slice(-maxChars);
}

function patchObjectRows(container, obj) {
  const nextKey = JSON.stringify(obj || {});
  if (container.dataset.renderKey === nextKey) return;
  container.dataset.renderKey = nextKey;
  container.replaceChildren();
  const keys = Object.keys(obj || {}).sort();
  if (!keys.length) {
    container.append(el("div", { class: "subtle", text: "No fields" }));
    return;
  }
  const table = el("table");
  const tbody = el("tbody");
  for (const key of keys) {
    const row = el("tr");
    row.append(el("td", { text: key }));
    row.append(el("td", {}, [el("pre", { text: valueText(obj[key]) })]));
    tbody.append(row);
  }
  table.append(tbody);
  container.append(table);
}

function patchEffects(container, effects) {
  const nextKey = JSON.stringify(effects || []);
  if (container.dataset.renderKey === nextKey) return;
  container.dataset.renderKey = nextKey;
  container.replaceChildren();
  if (!effects || !effects.length) {
    container.append(el("div", { class: "subtle", text: "No declared effects" }));
    return;
  }
  const table = el("table");
  const thead = el("thead", {}, [
    el("tr", {}, [
      el("th", { text: "Kind" }),
      el("th", { text: "Target" }),
      el("th", { text: "Description" }),
    ]),
  ]);
  const tbody = el("tbody");
  for (const effect of effects) {
    tbody.append(el("tr", {}, [
      el("td", { class: "kind", text: (effect.kind || "") + (effect.required ? "" : " optional") }),
      el("td", { class: "target", text: effect.target || "" }),
      el("td", { text: effect.label || "" }),
    ]));
  }
  table.append(thead, tbody);
  container.append(table);
}

function fmtBytes(bytes) {
  if (bytes == null) return "";
  const value = Number(bytes);
  if (!Number.isFinite(value)) return "";
  if (value < 1024) return value + " B";
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let scaled = value / 1024;
  for (const unit of units) {
    if (scaled < 1024) return scaled.toFixed(scaled < 10 ? 1 : 0) + " " + unit;
    scaled /= 1024;
  }
  return scaled.toFixed(0) + " PiB";
}

function patchOutputs(container, outputs) {
  const nextKey = JSON.stringify(outputs || []);
  if (container.dataset.renderKey === nextKey) return;
  container.dataset.renderKey = nextKey;
  container.replaceChildren();
  if (!outputs || !outputs.length) {
    container.append(el("div", { class: "subtle", text: "No discovered outputs yet" }));
    return;
  }
  const table = el("table");
  const thead = el("thead", {}, [
    el("tr", {}, [
      el("th", { text: "Kind" }),
      el("th", { text: "Path" }),
      el("th", { text: "Size" }),
      el("th", { text: "Description" }),
    ]),
  ]);
  const tbody = el("tbody");
  for (const output of outputs) {
    const pathNode = output.html
      ? el("a", { class: "output-link", href: output.href || output.path || "", text: output.path || "" })
      : el("span", { text: output.path || "" });
    tbody.append(el("tr", {}, [
      el("td", { class: "kind", text: output.kind || "" }),
      el("td", { class: "target" }, [pathNode]),
      el("td", { text: fmtBytes(output.size_bytes) }),
      el("td", { text: output.label || "" }),
    ]));
  }
  table.append(thead, tbody);
  container.append(table);
}

function createDetails(summaryText) {
  const body = el("div");
  const summary = el("summary", { text: summaryText });
  const details = el("details", {}, [summary, body]);
  return { details, summary, body };
}

function createStageNode(stage) {
  const article = el("article", { class: "stage" });
  const icon = el("span", { class: "dot" });
  const name = el("span", { class: "stage-name" });
  const badge = el("span", { class: "badge" });
  const duration = el("span", { class: "duration" });
  const error = el("div", { class: "error" });
  const analysis = el("section", { class: "analysis-panel" });
  const args = createDetails("Arguments");
  const plan = createDetails("Plan Data");
  const effects = createDetails("Effects");
  const outputs = createDetails("Outputs");
  const consoleBox = createDetails("Console (0 lines)");
  const consolePre = el("pre", { class: "console-pre" });

  consoleBox.body.append(consolePre);
  consoleBox.details.addEventListener("toggle", () => {
    if (consoleBox.details.open) {
      patchConsoleText(consolePre, consoleBox.details.dataset.consoleText || "");
    }
  });

  article.append(
    el("div", { class: "stage-head" }, [icon, name, badge, duration]),
    el("div", { class: "stage-body" }, [
      error,
      analysis,
      el("div", { class: "grid" }, [
        args.details,
        plan.details,
        effects.details,
        outputs.details,
        consoleBox.details,
      ]),
    ]),
  );

  return {
    article,
    icon,
    name,
    badge,
    duration,
    error,
    analysis,
    args,
    plan,
    effects,
    outputs,
    consoleBox,
    consolePre,
  };
}

function patchConsoleText(pre, text) {
  const displayText = consoleDisplayText(text);
  if (pre.dataset.renderKey === displayText) return;
  const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 8;
  pre.dataset.renderKey = displayText;
  pre.textContent = displayText;
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

function patchStageNode(nodes, stage) {
  const status = statusClass(stage.status || "pending");
  setClassName(nodes.article, "stage status-" + status);
  setClassName(nodes.icon, status === "running" ? "spinner" : "dot");
  setText(nodes.name, stage.name || "");
  setText(nodes.badge, stage.status || "pending");
  setText(nodes.duration, stage.duration ? fmtDuration(stage.duration) : "");

  setText(nodes.error, stage.error || "");
  nodes.error.hidden = !stage.error;
  nodes.analysis.hidden = !Object.keys(stage.analysis || {}).length;
  patchAnalysis(nodes.analysis, stage);
  patchObjectRows(nodes.args.body, stage.args);
  patchObjectRows(nodes.plan.body, stage.plan);
  patchEffects(nodes.effects.body, stage.effects);
  patchOutputs(nodes.outputs.body, stage.outputs);

  const consoleText = stage.console || "";
  const lines = countConsoleLines(consoleText);
  nodes.consoleBox.details.hidden = !consoleText;
  nodes.consoleBox.details.dataset.consoleText = consoleText;
  setText(nodes.consoleBox.summary, "Console (" + lines + " line" + (lines === 1 ? "" : "s") + ")");
  if (nodes.consoleBox.details.open) patchConsoleText(nodes.consolePre, consoleText);
}

function createPipelinePage(root, initialState) {
  root.replaceChildren(
    el("header", {}, [
      el("div", {}, [
        el("h1", { id: "title" }),
        el("div", { class: "subtle", id: "command" }),
      ]),
      el("div", { class: "meta" }, [
        el("div", { id: "mode" }),
        el("div", { id: "elapsed" }),
        el("div", { id: "live" }),
      ]),
    ]),
    el("div", { class: "bar", "aria-hidden": "true" }, [
      el("div", { class: "bar-fill", id: "bar-fill" }),
    ]),
    el("section", { id: "stages" }),
  );

  const refs = {
    title: root.querySelector("#title"),
    command: root.querySelector("#command"),
    mode: root.querySelector("#mode"),
    elapsed: root.querySelector("#elapsed"),
    live: root.querySelector("#live"),
    barFill: root.querySelector("#bar-fill"),
    stages: root.querySelector("#stages"),
  };
  const stageNodes = new Map();
  let currentState = initialState;

  function renderLiveStatus(state) {
    if (state.external_refresh_failed || liveFailed) {
      setText(refs.live, "External file refresh unavailable - showing embedded/latest available data");
      return;
    }
    if (state.mode === "run" && !state.done) {
      const pollAge = lastPollAt === null ? "last sync pending" : "last sync " + fmtAgeMs(Date.now() - lastPollAt);
      const changeAge = "last new data " + fmtAgeMs(Date.now() - lastChangeAt);
      setText(refs.live, "Live updates - " + pollAge + " - " + changeAge);
      return;
    }
    setText(refs.live, "Static view - generated " + fmtGeneratedAt(state.generated_at) + " (" + fmtAge(state.generated_at) + ")");
  }

  function update(state) {
    currentState = state;
    setText(refs.title, (state.pipeline || "") + " pipeline");
    setText(refs.command, state.command || "");
    setText(refs.mode, renderModeLabel(state));
    setText(refs.elapsed, state.mode === "run" ? fmtDuration(state.elapsed) : "");
    renderLiveStatus(state);

    const stages = state.stages || [];
    const complete = stages.filter(stage => ["success", "error", "failed", "skipped", "planned"].includes(stage.status)).length;
    refs.barFill.style.width = stages.length ? Math.round(complete / stages.length * 100) + "%" : "0%";

    const seen = new Set();
    for (const stage of stages) {
      const key = stage.name || "";
      seen.add(key);
      let nodes = stageNodes.get(key);
      if (!nodes) {
        nodes = createStageNode(stage);
        stageNodes.set(key, nodes);
        refs.stages.append(nodes.article);
      }
      patchStageNode(nodes, stage);
    }
    for (const [key, nodes] of stageNodes) {
      if (!seen.has(key)) {
        nodes.article.remove();
        stageNodes.delete(key);
      }
    }
  }

  return {
    update,
    tick() { renderLiveStatus(currentState); },
  };
}
"""


_STAGE_PAGE_JS = r"""
function createDependencySection(dependencies) {
  const box = createDetails("External Dependencies");
  patchDependenciesList(box.body, dependencies);
  return box;
}

function patchDependenciesList(container, dependencies) {
  const nextKey = JSON.stringify(dependencies || []);
  if (container.dataset.renderKey === nextKey) return;
  container.dataset.renderKey = nextKey;
  container.replaceChildren();
  if (!dependencies || !dependencies.length) {
    container.append(el("div", { class: "subtle", text: "No declared external dependencies" }));
    return;
  }
  const table = el("table");
  const tbody = el("tbody");
  for (const dep of dependencies) {
    tbody.append(el("tr", {}, [
      el("td", { text: dep.stage || "" }),
      el("td", { text: dep.label || "" }),
      el("td", { class: "target", text: dep.target || "" }),
    ]));
  }
  table.append(tbody);
  container.append(table);
}

function fmtPct(value) {
  if (value == null || !Number.isFinite(Number(value))) return "";
  return (Number(value) * 100).toFixed(1) + "%";
}

function fmtMs(value) {
  if (value == null || !Number.isFinite(Number(value))) return "";
  return Math.round(Number(value)) + " ms";
}

function appendSimpleTable(container, headers, rows) {
  const table = el("table", { class: "compact-table" });
  table.append(el("thead", {}, [
    el("tr", {}, headers.map(header => el("th", { text: header }))),
  ]));
  const tbody = el("tbody");
  for (const row of rows) {
    tbody.append(el("tr", {}, row.map(cell => {
      if (cell && cell.nodeType) return el("td", {}, [cell]);
      return el("td", { text: cell == null ? "" : String(cell) });
    })));
  }
  table.append(tbody);
  container.append(table);
}

function patchAnalysis(container, stage) {
  const analysis = stage.analysis || {};
  const nextKey = JSON.stringify(analysis);
  if (container.dataset.renderKey === nextKey) return;
  container.dataset.renderKey = nextKey;
  container.replaceChildren();
  if (analysis.kind === "convert") patchConvertAnalysis(container, analysis);
  else if (analysis.kind === "assemble_models") patchAssembleModelsAnalysis(container, analysis);
  else if (analysis.kind === "train") patchTrainAnalysis(container, analysis);
  else if (analysis.kind === "collect") patchCollectAnalysis(container, analysis);
  else if (analysis.kind === "analyze") patchAnalyzeAnalysis(container, analysis);
}

function patchConvertAnalysis(container, analysis) {
  const artifacts = analysis.artifacts || [];
  container.append(el("section", { class: "summary-grid" }, [
    metricCard("Artifacts", String(analysis.artifact_count || artifacts.length || 0)),
    metricCard("Total Size", fmtBytes(analysis.total_size_bytes)),
    metricCard("Schema", analysis.schema_version == null ? "" : "v" + String(analysis.schema_version)),
    metricCard("NPU Targets", (analysis.npu_targets || []).join(", ") || "none"),
  ]));
  if (analysis.manifest_href) {
    container.append(el("div", { class: "subtle" }, [
      document.createTextNode("Manifest: "),
      el("a", { class: "output-link", href: analysis.manifest_href, text: analysis.manifest_path || "converted_model_manifest.json" }),
    ]));
  }
  appendSimpleTable(container, ["Role", "Artifacts", "Size"], (analysis.by_role || []).map(row => [
    row.role,
    row.count,
    fmtBytes(row.size_bytes),
  ]));
  appendSimpleTable(container, ["Artifact", "Role", "Family", "Variant", "Hardware", "Capabilities", "Size", "File"], artifacts.map(row => [
    row.title || row.id,
    row.role || "",
    row.family || "",
    row.variant || "",
    row.hardware_target || "",
    (row.capabilities || []).join(", "),
    fmtBytes(row.size_bytes),
    row.href ? el("a", { class: "output-link", href: row.href, text: row.filename || row.id }) : (row.filename || ""),
  ]));
}

function assembleModelsSdkManifest(analysis) {
  return {
    id: analysis.manifest_id || "",
    channel: analysis.channel || "",
    created_at: analysis.created_at || "",
    roles: analysis.roles || {},
  };
}

function assembleModelsSettingsRows(analysis) {
  const sdk = globalThis.ZephrAgentRuntime;
  if (sdk && analysis.roles) return sdk.settingsChoiceRows(assembleModelsSdkManifest(analysis));
  return analysis.settings || [];
}

function assembleModelsComponentRows(analysis) {
  const sdk = globalThis.ZephrAgentRuntime;
  if (sdk && analysis.roles) return sdk.componentRows(assembleModelsSdkManifest(analysis));
  return analysis.component_rows || [];
}

function patchAssembleModelsAnalysis(container, analysis) {
  const artifacts = analysis.artifacts || [];
  const settings = assembleModelsSettingsRows(analysis);
  const componentRows = assembleModelsComponentRows(analysis);
  const defaults = settings.filter(row => row.default);
  container.append(el("section", { class: "summary-grid" }, [
    metricCard("Channel", analysis.channel || "local"),
    metricCard("Artifacts", String(analysis.artifact_count || artifacts.length || 0)),
    metricCard("Settings Choices", String(settings.length)),
    metricCard("Total Size", fmtBytes(analysis.total_size_bytes)),
  ]));
  if (analysis.manifest_href) {
    container.append(el("div", { class: "subtle" }, [
      document.createTextNode("Manifest: "),
      el("a", { class: "output-link", href: analysis.manifest_href, text: analysis.manifest_id || analysis.manifest_path || "local.json" }),
    ]));
  }
  appendSimpleTable(container, ["Platform", "Role", "Default Choice", "Plan", "Components"], defaults.map(row => [
    row.platform,
    row.role,
    row.label || row.choice_id,
    row.requested_plan,
    row.component_count,
  ]));
  const choices = createDetails("SDK Settings Choices");
  appendSimpleTable(choices.body, ["Platform", "Role", "Choice", "Plan", "Default", "Components", "Variants"], settings.map(row => [
    row.platform,
    row.role,
    row.label || row.choice_id,
    row.requested_plan,
    row.default ? "yes" : "",
    row.component_count,
    row.variant_count,
  ]));
  container.append(choices.details);
  const components = createDetails("Planned Components");
  appendSimpleTable(components.body, ["Platform", "Role", "Choice", "Component", "Target", "Requested", "Signature", "Hardware", "Artifact", "Why"], componentRows.map(row => [
    row.platform,
    row.role,
    row.choice_id,
    row.component,
    row.target,
    row.requested_target,
    row.signature,
    row.hardware_targets,
    row.artifact,
    row.reason,
  ]));
  container.append(components.details);
  appendSimpleTable(container, ["Artifact", "Role", "Family", "Variant", "Hardware", "Size"], artifacts.map(row => [
    row.title || row.id,
    row.role || "",
    row.family || "",
    row.variant || "",
    row.hardware_target || "",
    fmtBytes(row.size_bytes),
  ]));
}

function patchTrainAnalysis(container, analysis) {
  const evalSummary = analysis.eval || {};
  const losses = analysis.losses || {};
  const train = losses.train || [];
  const validation = losses.eval || [];
  const cards = el("section", { class: "summary-grid" });
  cards.append(
    metricCard("Eval Accuracy", evalSummary.accuracy == null ? "" : fmtPct(evalSummary.accuracy)),
    metricCard("Eval Passed", evalSummary.total ? String(evalSummary.passed || 0) + " / " + String(evalSummary.total) : ""),
    metricCard("Loss Points", String(train.length + validation.length)),
  );
  container.append(cards);
  const figureBox = el("div");
  container.append(figureBox);
  patchInteractiveFigure(figureBox, (analysis.figures || {}).loss);
}

function metricCard(label, value) {
  return el("div", { class: "metric" }, [
    el("div", { class: "metric-label", text: label }),
    el("div", { class: "metric-value", text: value || "pending" }),
  ]);
}

function patchCollectAnalysis(container, analysis) {
  const rows = analysis.runs || [];
  if (!rows.length) {
    container.append(el("div", { class: "subtle", text: "No collect results available yet" }));
    return;
  }
  const totalPassed = rows.reduce((sum, row) => sum + Number(row.passed || 0), 0);
  const totalValidation = rows.reduce((sum, row) => sum + Number(row.validation_records || 0), 0);
  const avgValues = rows.map(row => Number(row.avg_ms)).filter(Number.isFinite);
  container.append(el("section", { class: "summary-grid" }, [
    metricCard("Overall Accuracy", totalValidation ? fmtPct(totalPassed / totalValidation) : ""),
    metricCard("Runs", String(rows.length)),
    metricCard("Mean E2E", avgValues.length ? fmtMs(avgValues.reduce((a, b) => a + b, 0) / avgValues.length) : ""),
  ]));
  appendSimpleTable(container, ["Platform", "Profile", "Accuracy", "Avg", "P95", "Records", "Results"], rows.map(row => [
    row.platform,
    row.profile,
    row.accuracy == null ? "" : fmtPct(row.accuracy),
    fmtMs(row.avg_ms),
    fmtMs(row.p95_ms),
    row.records,
    el("a", { class: "output-link", href: row.href || row.path || "", text: "results.jsonl" }),
  ]));
  const timelineRows = rows.filter(row => Array.isArray(row.timeline) && row.timeline.length);
  if (timelineRows.length) {
    const box = createDetails("Dataset Replay Timelines");
    box.body.append(el("div", { class: "grid" }, timelineRows.map(row => createDatasetReplayTimeline(row))));
    container.append(box.details);
  }
}

function createDatasetReplayTimeline(row) {
  const items = row.timeline || [];
  const title = `${row.platform || ""} / ${row.profile || ""}`.trim();
  const wrapper = el("section", {}, [
    el("div", { class: "stage-name", text: title }),
    el("div", { class: "subtle", text: row.path || "" }),
  ]);
  wrapper.append(el("div", { class: "timeline-list" }, items.map(item => (
    el("div", { class: "timeline-item" }, [
      el("div", { class: "timeline-kind", text: item.kind || "" }),
      el("div", {}, [
        el("div", { class: "timeline-title", text: item.title || "" }),
        el("div", { class: "timeline-detail", text: item.detail || "" }),
      ]),
    ])
  ))));
  return wrapper;
}

function patchAnalyzeAnalysis(container, analysis) {
  const runs = analysis.runs || [];
  if (!runs.length) {
    container.append(el("div", { class: "subtle", text: "No archived runs yet" }));
    return;
  }
  appendSimpleTable(container, ["Run", "Channel", "Dataset", "Collect Runs", "Reports"], runs.map(run => [
    run.timestamp || run.id,
    run.model_channel || "",
    run.diagnostic_dataset_id || "",
    (run.collect || []).length,
    run.diagnostics_href ? el("a", { class: "output-link", href: run.diagnostics_href, text: "diagnostics" }) : "",
  ]));
}
"""


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_CFT_JSON_URL = (
    "https://googlechromelabs.github.io/chrome-for-testing/"
    "last-known-good-versions-with-downloads.json"
)


def _platform_key() -> str:
    system = platform.system()
    if system == "Darwin":
        return "mac-arm64" if platform.machine() == "arm64" else "mac-x64"
    if system == "Linux":
        return "linux64"
    raise RuntimeError(f"Chrome for Testing is not supported on {system}")


def _chrome_root(ui_base_dir: Path) -> Path:
    return ui_base_dir / ".tinycli-ui" / "chrome-for-testing"


def _chrome_profile_dir(ui_base_dir: Path) -> Path:
    return ui_base_dir / ".tinycli-ui" / "chrome-profile"


def _chrome_launch_log(ui_base_dir: Path) -> Path:
    return ui_base_dir / ".tinycli-ui" / "chrome-launch.log"


def _chrome_preferences_path(ui_base_dir: Path) -> Path:
    return _chrome_profile_dir(ui_base_dir) / "Default" / "Preferences"


def _chrome_binary(ui_base_dir: Path) -> Path:
    key = _platform_key()
    root = _chrome_root(ui_base_dir)
    if platform.system() == "Darwin":
        return (
            root
            / f"chrome-{key}"
            / "Google Chrome for Testing.app"
            / "Contents"
            / "MacOS"
            / "Google Chrome for Testing"
        )
    return root / f"chrome-{key}" / "chrome"


async def _open_files_in_chrome_for_testing(file_paths: list[Path], ui_base_dir: Path) -> None:
    await _ensure_chrome_for_testing(ui_base_dir)
    await anyio.to_thread.run_sync(_terminate_existing_chrome_for_testing, ui_base_dir)
    window_placement = _read_chrome_window_placement(ui_base_dir)
    _reset_chrome_profile(ui_base_dir)
    _ensure_chrome_profile(ui_base_dir, window_placement=window_placement)
    urls = [f"file://{path.resolve()}" for path in file_paths]
    process = await anyio.to_thread.run_sync(
        _launch_chrome_for_testing,
        ui_base_dir,
        urls,
        window_placement,
    )
    await anyio.sleep(1.0)
    returncode = process.poll()
    if returncode not in (None, 0):
        log_tail = _read_log_tail(_chrome_launch_log(ui_base_dir))
        raise RuntimeError(
            f"Chrome for Testing exited with code {returncode}. "
            f"Log: {_chrome_launch_log(ui_base_dir).resolve()}\n{log_tail}"
        )


def _launch_chrome_for_testing(
    ui_base_dir: Path,
    urls: list[str],
    window_placement: dict[str, Any] | None = None,
) -> subprocess.Popen[bytes]:
    ui_base_dir = ui_base_dir.resolve()
    log_path = _chrome_launch_log(ui_base_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] launching Chrome for Testing\n".encode())
        log.write(("urls: " + " ".join(urls) + "\n").encode())
        return subprocess.Popen(
            [
                str(_chrome_binary(ui_base_dir).resolve()),
                f"--user-data-dir={_chrome_profile_dir(ui_base_dir).resolve()}",
                "--new-window",
                "--disable-background-mode",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-gpu",
                "--disable-3d-apis",
                "--disable-webgl",
                "--disable-webgl2",
                "--disable-webgpu",
                "--disable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE,UseSkiaRenderer,CanvasOopRasterization",
                "--disk-cache-size=10485760",
                "--media-cache-size=1048576",
                "--js-flags=--max-old-space-size=256",
                "--allow-file-access-from-files",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-infobars",
                "--disable-session-crashed-bubble",
                "--hide-crash-restore-bubble",
                *_chrome_window_args(window_placement),
                *urls,
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )


def _chrome_window_args(window_placement: dict[str, Any] | None) -> list[str]:
    if not window_placement:
        return []
    if window_placement.get("maximized") is True:
        return ["--start-maximized"]
    try:
        left = int(window_placement["left"])
        top = int(window_placement["top"])
        width = int(window_placement["right"]) - left
        height = int(window_placement["bottom"]) - top
    except (KeyError, TypeError, ValueError):
        return []
    if width <= 0 or height <= 0:
        return []
    return [
        f"--window-position={left},{top}",
        f"--window-size={width},{height}",
    ]


def _read_log_tail(path: Path, *, max_bytes: int = 4000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-max_bytes:].decode("utf-8", errors="replace")


def _terminate_existing_chrome_for_testing(ui_base_dir: Path) -> None:
    """Stop the dedicated Chrome for Testing instance before opening fresh UI."""
    import psutil

    chrome_root = str(_chrome_root(ui_base_dir).resolve())
    profile_arg = f"--user-data-dir={_chrome_profile_dir(ui_base_dir).resolve()}"
    current_pid = psutil.Process().pid
    matched: list[psutil.Process] = []

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] == current_pid:
                continue
            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue
            if any(str(part).startswith(profile_arg) for part in cmdline) or any(
                chrome_root in str(part) for part in cmdline
            ):
                matched.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not matched:
        return
    matched.sort(key=lambda proc: len(proc.parents()), reverse=True)

    for proc in matched:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    gone, alive = psutil.wait_procs(matched, timeout=5)
    del gone
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=3)


def _reset_chrome_profile(ui_base_dir: Path) -> None:
    profile_dir = _chrome_profile_dir(ui_base_dir)
    try:
        shutil.rmtree(profile_dir)
    except FileNotFoundError:
        pass


async def _ensure_chrome_for_testing(ui_base_dir: Path) -> None:
    if _chrome_binary(ui_base_dir).exists():
        _repair_chrome_permissions(ui_base_dir)
        return
    await anyio.to_thread.run_sync(_download_chrome_for_testing, ui_base_dir)
    _repair_chrome_permissions(ui_base_dir)


def _download_chrome_for_testing(ui_base_dir: Path) -> None:
    key = _platform_key()
    print("Downloading Chrome for Testing for tinycli UI...")
    with urlopen(_CFT_JSON_URL) as response:
        metadata = json.loads(response.read())

    channel = metadata["channels"]["Stable"]
    download_url = None
    for item in channel["downloads"]["chrome"]:
        if item["platform"] == key:
            download_url = item["url"]
            break
    if download_url is None:
        raise RuntimeError(f"No Chrome for Testing download for platform {key}")

    root = _chrome_root(ui_base_dir)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "chrome.zip"
    with urlopen(download_url) as response:
        with open(zip_path, "wb") as file:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                file.write(chunk)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    zip_path.unlink(missing_ok=True)


def _repair_chrome_permissions(ui_base_dir: Path) -> None:
    """Ensure macOS Chrome bundle helper binaries are executable after unzip."""
    binary = _chrome_binary(ui_base_dir)
    if not binary.exists():
        return
    if platform.system() != "Darwin":
        binary.chmod(0o755)
        return

    app_dir = binary.parents[3]
    executable_dirs = {"MacOS"}
    executable_names = {"chrome_crashpad_handler"}
    for path in app_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.parent.name in executable_dirs or path.name in executable_names:
            path.chmod(0o755)


def _read_chrome_window_placement(ui_base_dir: Path) -> dict[str, Any] | None:
    preferences_path = _chrome_preferences_path(ui_base_dir)
    try:
        preferences = json.loads(preferences_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    browser = preferences.get("browser")
    if not isinstance(browser, dict):
        return None
    window_placement = browser.get("window_placement")
    if isinstance(window_placement, dict):
        return window_placement
    return None


def _ensure_chrome_profile(
    ui_base_dir: Path,
    *,
    window_placement: dict[str, Any] | None = None,
) -> None:
    profile_dir = _chrome_profile_dir(ui_base_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    local_state = profile_dir / "Local State"
    if not local_state.exists():
        local_state.write_text(
            json.dumps({"profile": {"info_cache": {"Default": {"name": "tinycli UI"}}}}),
            encoding="utf-8",
        )
    if window_placement is None:
        return

    preferences_path = _chrome_preferences_path(ui_base_dir)
    preferences_path.parent.mkdir(parents=True, exist_ok=True)
    preferences: dict[str, Any] = {}
    try:
        preferences = json.loads(preferences_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    browser = preferences.setdefault("browser", {})
    if isinstance(browser, dict):
        browser["window_placement"] = window_placement
    preferences_path.write_text(
        json.dumps(preferences, indent=2, sort_keys=True),
        encoding="utf-8",
    )
