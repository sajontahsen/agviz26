"""Deterministic fallback HTML for generation failures.

This module is intentionally small: if the LLM coder did not leave a usable
dist/index.html, write a public-safe page for the arena preview.
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any


def ensure_fallback_dist(workdir: Path, *, findings_available: bool) -> dict[str, Any]:
    """Ensure dist/index.html exists; return fallback telemetry for generation.json.

    Cases:
    - coder already wrote a usable artifact: fallback not used.
    - findings exist but visualization build failed: fallback writes findings.
    - findings do not exist: fallback writes a generic agent-failed page.
    """
    workdir = Path(workdir)
    dist = workdir / "dist" / "index.html"
    if _artifact_ready(dist):
        return {
            "used": False,
            "write_attempted": False,
            "write_succeeded": None,
            "reason": "artifact already exists",
            "path": str(dist),
        }

    reason = "coder failed" if findings_available else "analyst failed"
    try:
        html = _render_findings_page(workdir) if findings_available else _render_failure_page(workdir)
        dist.parent.mkdir(parents=True, exist_ok=True)
        dist.write_text(html, encoding="utf-8")
        print(f"[pipeline] wrote fallback dist/index.html ({reason})", file=sys.stderr)
        return {
            "used": True,
            "write_attempted": True,
            "write_succeeded": True,
            "reason": reason,
            "path": str(dist),
        }
    except Exception as exc:  # noqa: BLE001 - last-resort; must not crash the run
        print(f"[pipeline] FAILED to write fallback dist: {exc}", file=sys.stderr)
        return {
            "used": True,
            "write_attempted": True,
            "write_succeeded": False,
            "reason": f"fallback write failed: {exc}",
            "path": str(dist),
        }


def _artifact_ready(dist: Path) -> bool:
    try:
        return dist.exists() and dist.stat().st_size > 200
    except OSError:
        return False


def _render_findings_page(workdir: Path) -> str:
    return _page(
        title=_task_title(workdir),
        note="Visualization generation failed; showing computed findings.",
        body=_findings_html(workdir),
    )


def _render_failure_page(workdir: Path) -> str:
    return _page(
        title=_task_title(workdir),
        note="The agent failed to generate a visualization for this task.",
        body="<p>Reliable findings were not available.</p>",
    )


def _page(*, title: str, note: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem;
         background: #0f1420; color: #e8eaed; line-height: 1.5; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .5rem; }}
  .note {{ color: #f0a; margin-bottom: 1.5rem; }}
  .summary {{ max-width: 70rem; margin-bottom: 1rem; }}
  ul {{ max-width: 70rem; }} li {{ margin: .5rem 0; }}
  .k {{ color: #8ab4f8; font-weight: 600; }}
  .muted {{ color: #aeb4c0; }}
</style></head>
<body>
  <h1>{escape(title)}</h1>
  <p class="note">{escape(note)}</p>
  {body}
</body></html>
"""


def _task_title(workdir: Path) -> str:
    try:
        text = (workdir / "task.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "Visualization"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("title:"):
            return s.split(":", 1)[1].strip().strip('"').strip("'") or "Visualization"
        if s.startswith("# "):
            return s[2:].strip() or "Visualization"
    return "Visualization"


def _findings_html(workdir: Path) -> str:
    try:
        data = json.loads((workdir / "analysis_manifest.json").read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return "<p>Reliable findings were not available.</p>"

    parts: list[str] = []
    summary = str(data.get("task_summary") or "").strip()
    if summary:
        parts.append(f'<p class="summary">{escape(summary)}</p>')

    visualizations = data.get("visualizations")
    if not isinstance(visualizations, list) or not visualizations:
        parts.append("<p>Reliable findings were not available.</p>")
        return "\n".join(parts)

    rows = []
    for viz in visualizations[:40]:
        if not isinstance(viz, dict):
            continue
        title = escape(str(viz.get("title") or viz.get("id") or "Finding"))
        answer = escape(str(viz.get("answer") or ""))[:600]
        data_file = escape(str(viz.get("data_file") or ""))
        caveats = viz.get("caveats") or []
        caveat_text = ""
        if isinstance(caveats, list) and caveats:
            caveat_text = f'<br><span class="muted">Caveats: {escape("; ".join(map(str, caveats[:3])))}</span>'
        data_text = f'<br><span class="muted">Data: {data_file}</span>' if data_file else ""
        rows.append(f'<li><span class="k">{title}</span>: {answer}{data_text}{caveat_text}</li>')
    parts.append("<ul>" + "".join(rows) + "</ul>" if rows else "<p>Reliable findings were not available.</p>")
    return "\n".join(parts)
