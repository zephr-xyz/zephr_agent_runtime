"""Plotly figure specs and browser rendering helpers for tinycli reports.

All Plotly-specific Python and JavaScript should live in this module. Callers
pass plain data in and receive plain figure dictionaries that the report UI can
embed without importing Plotly elsewhere.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-3.5.0.min.js"

PLOTLY_RENDER_JS = r"""
function patchInteractiveFigure(container, figure) {
  const nextKey = JSON.stringify(figure || {});
  if (container.dataset.renderKey === nextKey) return;
  container.dataset.renderKey = nextKey;
  container.replaceChildren();
  if (!figure || !figure.data || !figure.data.length) {
    container.append(el("div", { class: "subtle", text: "No plot data available yet" }));
    return;
  }
  const plot = el("div", { class: "plotly-figure" });
  container.append(plot);
  if (!window.Plotly) {
    container.replaceChildren(el("div", { class: "subtle", text: "Plotly.js did not load" }));
    return;
  }
  Plotly.react(plot, figure.data || [], figure.layout || {}, { responsive: true, displaylogo: false });
}
"""


def training_loss_figure(losses: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """Return a Plotly-compatible figure for training and validation loss."""
    train = losses.get("train") or []
    eval_points = losses.get("eval") or []
    traces: list[dict[str, Any]] = []
    if train:
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": "Training Loss",
            "x": [point["epoch"] for point in train],
            "y": [point["loss"] for point in train],
            "line": {"color": "#0f766e", "width": 2},
        })
    if eval_points:
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Validation Loss",
            "x": [point["epoch"] for point in eval_points],
            "y": [point["loss"] for point in eval_points],
            "line": {"color": "#c2410c", "width": 2},
            "marker": {"color": "#c2410c", "size": 7},
        })
    return {
        "data": traces,
        "layout": {
            "height": 320,
            "margin": {"l": 48, "r": 20, "t": 18, "b": 44},
            "xaxis": {"title": {"text": "epoch"}},
            "yaxis": {"title": {"text": "loss"}, "rangemode": "tozero"},
            "hovermode": "x unified",
            "legend": {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.12},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#18202f"},
        },
    }


def read_legacy_plotly_loss_html(path: Path) -> dict[str, list[dict[str, float]]]:
    """Extract loss series from the older standalone Plotly losses.html file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"train": [], "eval": []}
    return {
        "train": _legacy_plotly_trace_points(text, "Training Loss"),
        "eval": _legacy_plotly_trace_points(text, "Validation Loss"),
    }


def _legacy_plotly_trace_points(text: str, trace_name: str) -> list[dict[str, float]]:
    match = re.search(
        rf'"name"\s*:\s*"{re.escape(trace_name)}".*?'
        r'"x"\s*:\s*(\[[^\]]*\]).*?'
        r'"y"\s*:\s*(\[[^\]]*\])',
        text,
        re.S,
    )
    if match is None:
        return []
    try:
        xs = json.loads(match.group(1))
        ys = json.loads(match.group(2))
    except json.JSONDecodeError:
        return []
    points: list[dict[str, float]] = []
    for x_value, y_value in zip(xs, ys, strict=False):
        try:
            points.append({"epoch": float(x_value), "loss": float(y_value)})
        except (TypeError, ValueError):
            continue
    return points
