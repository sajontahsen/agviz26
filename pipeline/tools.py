"""Agent tools (schema + executor pairs) operating on the workdir via ToolContext.

Executors return an error string rather than raising, so one bad call never crashes a stage.
"""
from __future__ import annotations

import base64
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Shared state handed to every tool executor within a stage."""

    tool_root: Path
    artifacts_dir: Path = field(init=False)
    _verify_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.tool_root = Path(self.tool_root).resolve()
        self.artifacts_dir = self.tool_root / ".artifacts"

    def resolve(self, path: str | None) -> Path:
        """Resolve a path under tool_root, stripping a literal ``WORKDIR/`` prefix models sometimes copy from prompts."""
        if not path:
            return self.tool_root
        if path == "WORKDIR":
            return self.tool_root
        for prefix in ("WORKDIR/", "WORKDIR\\", "./WORKDIR/"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        p = Path(path)
        return p if p.is_absolute() else (self.tool_root / p)


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 12000  # hard clip on any single tool result fed back to the model
_MAX_IMAGE_B64_CHARS = 6_800_000
_MAX_VISION_DIMENSION = 7_800
_VISION_VIEWPORT_WIDTH = 1280
_VISION_VIEWPORT_HEIGHT = 900
_VISION_SLICE_HEIGHT = 1500
_VISION_OVERLAP = 150
_VISION_MAX_IMAGES = 5
_VISION_JPEG_QUALITY = 72

ToolContent = str | list[dict[str, Any]]


def _clip(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    return f"{head}\n... [truncated {len(text) - limit + 200} chars]"


def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = ctx.resolve(args.get("path"))
    if not path.exists():
        return f"Error: no such file: {path}"
    if path.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return f"{path} (directory):\n" + "\n".join(entries)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Error reading {path}: {exc}"
    lines = text.splitlines()
    offset = max(1, int(args.get("offset", 1)))  # 1-indexed
    limit = int(args.get("limit", 2000))
    chunk = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{offset + i}\t{ln}" for i, ln in enumerate(chunk))
    footer = ""
    if offset - 1 + limit < len(lines):
        footer = f"\n... [{len(lines) - (offset - 1 + limit)} more lines; use offset to continue]"
    return _clip(numbered + footer)


def write_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = ctx.resolve(args.get("path"))
    content = args.get("content")
    if content is None:
        return "Error: write_file requires 'content'."
    append = bool(args.get("append"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if append else "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        return f"Error writing {path}: {exc}"
    size = path.stat().st_size
    verb = "Appended" if append else "Wrote"
    return f"{verb} {len(content)} chars to {path} (now {size} bytes)."


def str_replace(args: dict[str, Any], ctx: ToolContext) -> str:
    path = ctx.resolve(args.get("path"))
    old = args.get("old_str")
    new = args.get("new_str", "")
    if old is None:
        return "Error: str_replace requires 'old_str'."
    if not path.exists():
        return f"Error: no such file: {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    if count == 0:
        return "Error: old_str not found. Read the file to copy an exact, unique snippet."
    if count > 1:
        return f"Error: old_str matched {count} times; make it unique (add surrounding context)."
    path.write_text(text.replace(old, new), encoding="utf-8")
    return f"Edited {path} (1 replacement)."


def apply_patch(args: dict[str, Any], ctx: ToolContext) -> str:
    patch_text = args.get("patch")
    if not isinstance(patch_text, str) or not patch_text.strip():
        return "Error: apply_patch requires a non-empty unified diff in 'patch'."
    try:
        changed = _apply_unified_diff(patch_text, ctx)
    except ValueError as exc:
        return f"Patch error: {exc}"
    except OSError as exc:
        return f"Patch file error: {exc}"
    if not changed:
        return "Patch error: no file changes found in patch."
    return "Applied patch:\n" + "\n".join(f"- {path} ({count} hunks)" for path, count in changed)


def search(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = args.get("pattern")
    if not pattern:
        return "Error: search requires 'pattern'."
    root = ctx.resolve(args.get("path"))
    glob = args.get("glob")
    max_hits = int(args.get("max", 100))
    # Prefer ripgrep.
    rg = _which("rg")
    if rg:
        cmd = [rg, "-n", "--no-heading", "-m", str(max_hits)]
        if glob:
            cmd += ["-g", glob]
        cmd += [pattern, str(root)]
        try:
            out = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
            result = out.stdout or out.stderr or "(no matches)"
            return _clip(result)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"search error (rg): {exc}"
    # Pure-Python fallback.
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"Error: bad regex: {exc}"
    hits: list[str] = []
    files = root.rglob(glob) if glob else root.rglob("*")
    for fp in files:
        if not fp.is_file():
            continue
        try:
            for i, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{fp}:{i}:{line}")
                    if len(hits) >= max_hits:
                        return _clip("\n".join(hits))
        except OSError:
            continue
    return _clip("\n".join(hits) or "(no matches)")


def bash(args: dict[str, Any], ctx: ToolContext) -> str:
    command = args.get("command")
    if not command:
        return "Error: bash requires 'command'."
    cwd = ctx.resolve(args.get("cwd"))
    return _clip(_run_bash(command, cwd))


def verify(args: dict[str, Any], ctx: ToolContext) -> str:
    """Render an artifact and return a health summary with automatic chart/control diagnostics."""
    url = args.get("url")
    actions = args.get("actions") or []
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        return f"verify unavailable: Playwright not importable ({exc}). Skipping render check."

    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    ctx._verify_count += 1
    shot = ctx.artifacts_dir / f"verify_{ctx._verify_count}.png"

    with _maybe_serve(url, ctx.tool_root / "dist") as resolved_url:
        return _clip(_playwright_probe(resolved_url, actions, shot))


def vision_check(args: dict[str, Any], ctx: ToolContext) -> ToolContent:
    """Return a bounded visual sanity check as text plus page-slice screenshots."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        return f"vision_check unavailable: Playwright not importable ({exc}). Skipping visual sanity check."

    dist = ctx.tool_root / "dist" / "index.html"
    if not dist.exists():
        return "vision_check error: dist/index.html does not exist. Build the artifact and run verify first."

    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    ctx._verify_count += 1

    with _maybe_serve(None, ctx.tool_root / "dist") as resolved_url:
        return _capture_vision_slices(resolved_url, ctx)


# ---------------------------------------------------------------------------
# Playwright helpers (kept out of the model's hands so scripts are reliable)
# ---------------------------------------------------------------------------

_CHART_HEALTH_JS = """() => {
    const plots = [...document.querySelectorAll('.js-plotly-plot')];
    // Points, not traces: a trace with x:[]/y:[] draws nothing.
    const detail = plots.map(el => {
        const d = el.data || [];
        const pts = d.reduce((a, t) => a + ((t.y || t.values || t.x || []).length), 0);
        const r = el.getBoundingClientRect();
        return {id: el.id || '(anon)', traces: d.length, points: pts,
                h: Math.round(r.height), w: Math.round(r.width)};
    });
    return {
        plotly_total: plots.length,
        svg: document.querySelectorAll('svg').length,
        canvas: document.querySelectorAll('canvas').length,
        detail: detail,
        no_points: detail.filter(p => p.points === 0).map(p => p.id),
        zero_size: detail.filter(p => p.h < 5 || p.w < 5).map(p => p.id),
    };
}"""

_CONTROLS_JS = """() => {
    const sels = 'button, select, input, [role=tab], [role=button], [onclick]';
    const els = document.querySelectorAll(sels);
    const counts = {};
    els.forEach(el => {
        const tag = el.tagName.toLowerCase();
        const key = el.type ? tag + ':' + el.type : tag;
        counts[key] = (counts[key] || 0) + 1;
    });
    return {total: els.length, breakdown: counts};
}"""

_AXIS_HEALTH_JS = """() => {
    const plots = [...document.querySelectorAll('.js-plotly-plot')];
    const parseNum = (raw) => {
        const s = String(raw || '').trim()
            .replace(/,/g, '')
            .replace(/%$/, '')
            .replace(/[−–—]/g, '-');
        if (!/^[-+]?\\d*\\.?\\d+(e[-+]?\\d+)?$/i.test(s)) return null;
        const n = Number(s);
        return Number.isFinite(n) ? n : null;
    };
    const monotonic = (vals) => {
        if (vals.length < 4) return true;
        let inc = true, dec = true;
        for (let i = 1; i < vals.length; i++) {
            if (vals[i] < vals[i - 1]) inc = false;
            if (vals[i] > vals[i - 1]) dec = false;
        }
        return inc || dec;
    };
    const numericShareOf = (vals) => {
        const flat = vals.flatMap(v => Array.isArray(v) ? v : [v]).filter(v => v !== null && v !== undefined && v !== '');
        if (!flat.length) return 0;
        return flat.filter(v => typeof v === 'number' || parseNum(v) !== null).length / flat.length;
    };
    const rect = (el) => {
        const r = el.getBoundingClientRect();
        return {x: r.x, y: r.y, w: r.width, h: r.height};
    };
    const overlaps = (a, b) => (
        a.x < b.x + b.w && a.x + a.w > b.x &&
        a.y < b.y + b.h && a.y + a.h > b.y
    );
    const detail = [];
    for (const plot of plots) {
        const layout = plot._fullLayout || plot.layout || {};
        const plotId = plot.id || '(anon)';
        const axisKeys = Object.keys(layout).filter(k => /^yaxis\\d*$/.test(k));
        for (const axisKey of axisKeys.length ? axisKeys : ['yaxis']) {
            const axis = layout[axisKey] || {};
            const axisRef = axisKey === 'yaxis' ? 'y' : 'y' + axisKey.slice('yaxis'.length);
            const traceY = (plot.data || [])
                .filter(t => (t.yaxis || 'y') === axisRef)
                .map(t => t.y || []);
            const dataNumericShare = numericShareOf(traceY);
            const issues = [];
            if ((axis.type === 'category' || axis.type === 'multicategory') && dataNumericShare >= 0.8) {
                issues.push(`axis resolved as ${axis.type} even though trace y-data is mostly numeric`);
            }
            if (axis.type === 'category' || axis.type === 'multicategory') {
                if (issues.length) {
                    detail.push({
                        plot: plotId,
                        axis: axisKey,
                        type: axis.type || 'auto',
                        labels: [],
                        issues,
                    });
                }
                continue;
            }
            const suffix = axisKey === 'yaxis' ? '' : axisKey.slice('yaxis'.length);
            const tickClass = suffix ? `.y${suffix}tick text` : '.ytick text';
            const els = [...plot.querySelectorAll(tickClass)].filter(el => {
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
            });
            const labels = els.map(el => el.textContent.trim()).filter(Boolean);
            const nums = labels.map(parseNum).filter(v => v !== null);
            const numericShare = labels.length ? nums.length / labels.length : 0;
            if (labels.length >= 4 && numericShare >= 0.7 && !monotonic(nums)) {
                issues.push('numeric tick labels are non-monotonic');
            }
            let overlapCount = 0;
            const boxes = els.map(rect);
            for (let i = 0; i < boxes.length; i++) {
                for (let j = i + 1; j < boxes.length; j++) {
                    if (overlaps(boxes[i], boxes[j])) overlapCount++;
                }
            }
            if (overlapCount > 0) issues.push(`${overlapCount} tick-label overlaps`);
            if (issues.length) {
                detail.push({
                    plot: plotId,
                    axis: axisKey,
                    type: axis.type || 'auto',
                    labels: labels.slice(0, 16),
                    issues,
                });
            }
        }
    }
    return {
        checked_plots: plots.length,
        issues: detail,
    };
}"""

# Page prose only — inner_text('body') scrapes every SVG axis tick label too.
_BODY_TEXT_JS = """() => {
    const skip = new Set(['SVG', 'SCRIPT', 'STYLE', 'NOSCRIPT']);
    const out = [];
    const walk = (n) => {
        for (const c of n.childNodes) {
            if (c.nodeType === 3) {
                const t = c.textContent.trim();
                if (t) out.push(t);
            } else if (c.nodeType === 1 && !skip.has(c.tagName.toUpperCase())) {
                walk(c);
            }
        }
    };
    walk(document.body);
    return out.join('\\n');
}"""

_PAGE_HEIGHT_JS = """() => Math.ceil(Math.max(
    document.body ? document.body.scrollHeight : 0,
    document.documentElement ? document.documentElement.scrollHeight : 0,
    document.body ? document.body.offsetHeight : 0,
    document.documentElement ? document.documentElement.offsetHeight : 0,
    document.documentElement ? document.documentElement.clientHeight : 0
))"""


def _playwright_probe(url: str, actions: list[dict[str, Any]], shot: Path) -> str:
    from playwright.sync_api import sync_playwright

    console_errors: list[str] = []
    page_errors: list[str] = []
    action_log: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": _VISION_VIEWPORT_WIDTH, "height": _VISION_VIEWPORT_HEIGHT})
            page.on("console", lambda m: console_errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.goto(url, wait_until="networkidle", timeout=45000)
            title = page.title()
            text = page.evaluate(_BODY_TEXT_JS)[:1500]
            charts = page.evaluate(_CHART_HEALTH_JS)
            axes = page.evaluate(_AXIS_HEALTH_JS)
            controls = page.evaluate(_CONTROLS_JS)
            for act in actions:
                action_log.append(_run_action(page, act))
            page.screenshot(path=str(shot), full_page=False)
            browser.close()
    except Exception as exc:
        return f"verify error while rendering {url}: {exc}"

    ctrl_str = ", ".join(f"{k}={v}" for k, v in sorted(controls["breakdown"].items())) if controls["breakdown"] else "none"
    chart_str = " ".join(f"{p['id']}(t={p['traces']},pts={p['points']},{p['w']}x{p['h']})" for p in charts["detail"])
    parts = [
        f"url: {url}",
        f"viewport: {_VISION_VIEWPORT_WIDTH}x{_VISION_VIEWPORT_HEIGHT}",
        f"title: {title!r}",
        f"console_errors ({len(console_errors)}): " + ("; ".join(console_errors[:10]) or "none"),
        f"page_errors ({len(page_errors)}): " + ("; ".join(page_errors[:10]) or "none"),
        f"charts: plotly={charts['plotly_total']} svg={charts['svg']} canvas={charts['canvas']}",
    ]
    if chart_str:
        parts.append(f"  {chart_str}")
    if charts["no_points"]:
        parts.append(f"  EMPTY (no data points): {', '.join(charts['no_points'])}")
    if charts["zero_size"]:
        parts.append(f"  ZERO-SIZE (not visible): {', '.join(charts['zero_size'])}")
    if axes["issues"]:
        parts.append(f"axis_warnings ({len(axes['issues'])}):")
        for issue in axes["issues"][:12]:
            labels = ", ".join(issue.get("labels") or [])
            problems = "; ".join(issue.get("issues") or [])
            parts.append(
                f"  {issue.get('plot')} {issue.get('axis')} ({issue.get('type')}): "
                f"{problems}; labels=[{labels}]"
            )
    else:
        parts.append(f"axis_warnings (0): none across {axes['checked_plots']} Plotly plots")
    parts += [
        f"controls ({controls['total']}): {ctrl_str}",
        f"screenshot: {shot}",
    ]
    if action_log:
        parts.append("actions:\n  " + "\n  ".join(action_log))
    parts.append(f"body_text_sample:\n{text}")
    return "\n".join(parts)


def _capture_vision_slices(url: str, ctx: ToolContext) -> ToolContent:
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": _VISION_VIEWPORT_WIDTH, "height": _VISION_VIEWPORT_HEIGHT})
            page.goto(url, wait_until="networkidle", timeout=45000)
            page.wait_for_timeout(1200)
            page_height = max(_VISION_VIEWPORT_HEIGHT, int(page.evaluate(_PAGE_HEIGHT_JS) or _VISION_VIEWPORT_HEIGHT))
            plan = _vision_slice_plan(page_height)
            selected = plan["selected"]
            content: list[dict[str, Any]] = [_vision_intro_text(
                url=url,
                page_height=page_height,
                total_needed=len(plan["all"]),
                selected_count=len(selected),
                sampled=plan["sampled"],
            )]

            for idx, segment in enumerate(selected, 1):
                shot = ctx.artifacts_dir / f"vision_check_{ctx._verify_count}_{idx}.jpg"
                content.append({"type": "text", "text": _vision_slice_label(idx, len(selected), segment, plan["sampled"])})
                try:
                    _capture_vision_slice(page, shot, segment["y"], segment["height"])
                    content.extend(_image_parts(shot))
                except Exception as exc:
                    content.append({"type": "text", "text": f"[slice capture failed: {_error_reason(exc)}]"})
            browser.close()
            return content
    except Exception as exc:
        return f"vision_check error while rendering {url}: {exc}"


def _vision_intro_text(
    *,
    url: str,
    page_height: int,
    total_needed: int,
    selected_count: int,
    sampled: bool,
) -> dict[str, str]:
    gap_note = (
        "Coverage: sampled because the page needs more than 5 vertical chunks; unobserved vertical gaps exist between some returned slices."
        if sampled
        else "Coverage: exhaustive; adjacent slices overlap vertically."
    )
    text = "\n".join([
        "VISION_CHECK screenshots.",
        f"url: {url}",
        f"baseline_viewport: {_VISION_VIEWPORT_WIDTH}x{_VISION_VIEWPORT_HEIGHT}; screenshot_width: {_VISION_VIEWPORT_WIDTH}; slice_height: {_VISION_SLICE_HEIGHT}; overlap: {_VISION_OVERLAP}; max_images: {_VISION_MAX_IMAGES}",
        f"page_height_px: {page_height}; mechanical_chunks_needed: {total_needed}; returned_slices: {selected_count}; sampled: {sampled}",
        gap_note,
        "The following labeled slices are ordered top-to-bottom.",
    ])
    return {"type": "text", "text": _clip(text)}


def _vision_slice_plan(page_height: int) -> dict[str, Any]:
    slice_height = min(_VISION_SLICE_HEIGHT, _MAX_VISION_DIMENSION)
    overlap = min(_VISION_OVERLAP, slice_height - 1)
    step = slice_height - overlap
    max_y = max(0, page_height - slice_height)
    positions: list[int] = [0]
    while positions[-1] < max_y:
        next_y = positions[-1] + step
        if page_height - (next_y + slice_height) <= overlap:
            next_y = max_y
        if next_y <= positions[-1]:
            break
        positions.append(next_y)
    segments = [{"y": y, "height": min(slice_height, page_height - y)} for y in sorted(set(positions))]
    if len(segments) <= _VISION_MAX_IMAGES:
        return {"all": segments, "selected": segments, "sampled": False}
    indices = _sample_indices(len(segments), _VISION_MAX_IMAGES)
    return {"all": segments, "selected": [segments[i] for i in indices], "sampled": True}


def _sample_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    indices = {0, total - 1}
    if count > 2:
        for i in range(1, count - 1):
            indices.add(round(i * (total - 1) / (count - 1)))
    out = sorted(indices)
    cursor = 0
    while len(out) < count and cursor < total:
        if cursor not in out:
            out.append(cursor)
        cursor += 1
    return sorted(out[:count])


def _capture_vision_slice(page: Any, path: Path, y: int, height: int) -> None:
    width = min(_VISION_VIEWPORT_WIDTH, _MAX_VISION_DIMENSION)
    height = max(1, min(height, _MAX_VISION_DIMENSION))
    page.screenshot(
        path=str(path),
        type="jpeg",
        quality=_VISION_JPEG_QUALITY,
        full_page=True,
        scale="css",
        clip={"x": 0, "y": y, "width": width, "height": height},
    )


def _vision_slice_label(index: int, total: int, segment: dict[str, int], sampled: bool) -> str:
    y = segment["y"]
    h = segment["height"]
    lines = [
        f"VISION_CHECK SLICE {index} OF {total}: vertical range y={y}..{y + h}px.",
    ]
    if sampled:
        lines.append("Sampling note: some vertical ranges between returned slices are not shown.")
    return "\n".join(lines)


def _image_parts(path: Path) -> list[dict[str, Any]]:
    dims = _jpeg_dimensions(path)
    if dims and (dims[0] > _MAX_VISION_DIMENSION or dims[1] > _MAX_VISION_DIMENSION):
        return [{
            "type": "text",
            "text": f"[screenshot image omitted because dimensions exceed vision API limits: {path} is {dims[0]}x{dims[1]}]",
        }]
    data_url = _image_data_url(path)
    if len(data_url) > _MAX_IMAGE_B64_CHARS:
        return [{
            "type": "text",
            "text": f"[screenshot image omitted because it is too large for the vision API: {path}]",
        }]
    return [{"type": "image_url", "image_url": {"url": data_url}}]


def _image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _error_reason(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0][:240] or exc.__class__.__name__


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return (int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big"))
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        size = int.from_bytes(data[i + 2:i + 4], "big")
        if size < 2:
            return None
        i += 2 + size
    return None


def _run_action(page: Any, act: dict[str, Any]) -> str:
    """Run one small interaction probe; report outcome without throwing."""
    kind = act.get("do")
    sel = act.get("selector")
    try:
        if kind == "click" and sel:
            page.click(sel, timeout=5000)
            return f"click {sel!r}: ok"
        if kind == "hover" and sel:
            page.hover(sel, timeout=5000)
            return f"hover {sel!r}: ok"
        if kind == "wait":
            page.wait_for_timeout(int(act.get("ms", 500)))
            return f"wait {act.get('ms', 500)}ms: ok"
        if kind == "count" and sel:
            return f"count {sel!r}: {page.locator(sel).count()}"
        if kind == "text" and sel:
            return f"text {sel!r}: {page.locator(sel).first.inner_text()[:200]!r}"
        return f"unknown action {act!r}"
    except Exception as exc:
        # Drop Playwright's retry call log (~1.5k chars) — keep the reason.
        reason = str(exc).strip().splitlines()[0][:200]
        return f"{kind} {sel!r}: FAILED ({reason})"


@contextmanager
def _maybe_serve(url: str | None, dist_dir: Path) -> Iterator[str]:
    """Yield ``url`` if given, else serve ``dist_dir`` on a random localhost port."""
    if url:
        yield url
        return
    import http.server
    import socketserver
    import threading

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # noqa: A003
            pass

    handler = lambda *a, **kw: _Quiet(*a, directory=str(dist_dir), **kw)  # noqa: E731

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Shared bash runner
# ---------------------------------------------------------------------------

def _run_bash(command: str, cwd: Path, timeout_s: int = 300) -> str:
    cwd.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        return (
            f"$ {command}\nexit=timeout\nCommand exceeded {timeout_s}s and was killed. "
            f"Use a faster approach (avoid nested loops over the whole dataset; prefer "
            f"pandas/networkx).\n{out or ''}"
        )
    return f"$ {command}\nexit={completed.returncode}\n{completed.stdout}"


def _apply_unified_diff(patch_text: str, ctx: ToolContext) -> list[tuple[str, int]]:
    lines = patch_text.splitlines(keepends=True)
    i = 0
    changed: list[tuple[str, int]] = []
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1
            continue
        old_name = _parse_diff_path(lines[i][4:])
        i += 1
        if i >= len(lines) or not lines[i].startswith("+++ "):
            raise ValueError(f"expected +++ header after --- {old_name}")
        new_name = _parse_diff_path(lines[i][4:])
        i += 1

        target_name = new_name if new_name != "/dev/null" else old_name
        if target_name == "/dev/null":
            raise ValueError("delete-only patches are not supported")
        target = _resolve_patch_target(target_name, ctx)
        file_lines = [] if old_name == "/dev/null" else target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

        hunks = 0
        line_offset = 0
        while i < len(lines) and lines[i].startswith("@@"):
            header = lines[i]
            old_start = _parse_hunk_start(header)
            i += 1
            old_chunk: list[str] = []
            new_chunk: list[str] = []
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("--- "):
                line = lines[i]
                i += 1
                if line.startswith("\\"):
                    continue
                if not line:
                    raise ValueError("empty patch line without a diff prefix")
                prefix, text = line[0], line[1:]
                if prefix == " ":
                    old_chunk.append(text)
                    new_chunk.append(text)
                elif prefix == "-":
                    old_chunk.append(text)
                elif prefix == "+":
                    new_chunk.append(text)
                else:
                    raise ValueError(f"bad patch line prefix {prefix!r} in {target_name}")

            expected = max(0, old_start - 1 + line_offset)
            file_lines, delta = _apply_hunk(file_lines, old_chunk, new_chunk, expected, target_name)
            line_offset += delta
            hunks += 1

        if hunks == 0:
            raise ValueError(f"no hunks found for {target_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(file_lines), encoding="utf-8")
        changed.append((str(target), hunks))
    return changed


def _parse_diff_path(raw: str) -> str:
    path = raw.strip().split("\t", 1)[0].split(" ", 1)[0]
    if path in {"/dev/null", "dev/null"}:
        return "/dev/null"
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _parse_hunk_start(header: str) -> int:
    match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", header)
    if not match:
        raise ValueError(f"bad hunk header: {header.strip()}")
    return int(match.group(1))


def _resolve_patch_target(path: str, ctx: ToolContext) -> Path:
    target = ctx.resolve(path).resolve()
    root = ctx.tool_root.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"patch target points outside workdir: {path}") from exc
    return target


def _apply_hunk(
    file_lines: list[str],
    old_chunk: list[str],
    new_chunk: list[str],
    expected: int,
    target_name: str,
) -> tuple[list[str], int]:
    old_len = len(old_chunk)
    if old_len == 0:
        if expected > len(file_lines):
            raise ValueError(f"hunk insertion point beyond end of {target_name}")
        return file_lines[:expected] + new_chunk + file_lines[expected:], len(new_chunk)

    if _matches_at(file_lines, old_chunk, expected):
        start = expected
    else:
        matches = [
            idx for idx in range(0, len(file_lines) - old_len + 1)
            if _matches_at(file_lines, old_chunk, idx)
        ]
        if not matches:
            raise ValueError(f"hunk context not found in {target_name}; read the file and regenerate the patch")
        if len(matches) > 1:
            raise ValueError(f"hunk context matched {len(matches)} places in {target_name}; add more context")
        start = matches[0]

    return (
        file_lines[:start] + new_chunk + file_lines[start + old_len:],
        len(new_chunk) - old_len,
    )


def _matches_at(file_lines: list[str], old_chunk: list[str], start: int) -> bool:
    end = start + len(old_chunk)
    return start >= 0 and end <= len(file_lines) and file_lines[start:end] == old_chunk


def _which(name: str) -> str | None:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


# ---------------------------------------------------------------------------
# Tool registry — schema + executor, selectable per stage by name
# ---------------------------------------------------------------------------

Executor = Callable[[dict[str, Any], ToolContext], ToolContent]


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": _fn(
        "read_file",
        "Read a text file (or list a directory). Use offset/limit for line-range reads instead of reading the whole file.",
        {
            "path": {"type": "string", "description": "Path relative to the workdir."},
            "offset": {"type": "integer", "description": "1-indexed start line (default 1)."},
            "limit": {"type": "integer", "description": "Max lines to return (default 2000)."},
        },
        ["path"],
    ),
    "write_file": _fn(
        "write_file",
        "Write text to a file. Use append=true to build a large file across several calls "
        "instead of rewriting it. Prefer this over bash heredocs.",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "append": {"type": "boolean", "description": "Append instead of overwrite. Default false."},
        },
        ["path", "content"],
    ),
    "str_replace": _fn(
        "str_replace",
        "Replace one exact, unique occurrence of old_str with new_str in a file. Cheaper than rewriting the whole file.",
        {
            "path": {"type": "string"},
            "old_str": {"type": "string", "description": "Exact snippet to replace; must be unique in the file."},
            "new_str": {"type": "string", "description": "Replacement text (may be empty to delete)."},
        },
        ["path", "old_str", "new_str"],
    ),
    "apply_patch": _fn(
        "apply_patch",
        "Apply a standard unified diff patch to files under the workdir. Use this for localized multi-line edits "
        "when str_replace would require copying a large exact snippet. Patches must have ---/+++ file headers and @@ hunks.",
        {
            "patch": {
                "type": "string",
                "description": "Unified diff text. Paths may use a/ and b/ prefixes and must resolve under the workdir.",
            },
        },
        ["patch"],
    ),
    "search": _fn(
        "search",
        "Search files for a regex (ripgrep). Returns file:line:match.",
        {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Dir or file to search (default workdir)."},
            "glob": {"type": "string", "description": "Optional filename glob, e.g. '*.py'."},
            "max": {"type": "integer", "description": "Max matches (default 100)."},
        },
        ["pattern"],
    ),
    "bash": _fn(
        "bash",
        "Run a shell command (data inspection, running build scripts, pip-free tooling). 300s timeout.",
        {
            "command": {"type": "string"},
            "cwd": {"type": "string", "description": "Optional working directory."},
        },
        ["command"],
    ),
    "verify": _fn(
        "verify",
        "Render an artifact with a headless browser and return a compact health summary (console errors, chart health, "
        "numeric-axis warnings, title, text sample, screenshot) plus results of optional interaction probes. With no "
        "'url', serves dist/ locally. You MUST call this before finishing a build.",
        {
            "url": {"type": "string", "description": "Artifact URL; omit to serve the local dist/ folder."},
            "actions": {
                "type": "array",
                "description": "Optional interaction probes, each like {\"do\":\"click\",\"selector\":\"#btn\"}. "
                "Supported 'do': click, hover, wait(ms), count(selector), text(selector).",
                "items": {"type": "object"},
            },
        },
        [],
    ),
    "vision_check": _fn(
        "vision_check",
        "Capture labeled top-to-bottom screenshots of dist/index.html at 1280px width. Uses 1500px vertical "
        "slices with 150px overlap, capped at 5 images; long pages are sampled and marked with gap notes.",
        {},
        [],
    ),
}


EXECUTORS: dict[str, Executor] = {
    "read_file": read_file,
    "write_file": write_file,
    "str_replace": str_replace,
    "apply_patch": apply_patch,
    "search": search,
    "bash": bash,
    "verify": verify,
    "vision_check": vision_check,
}


def build_toolset(names: list[str]) -> tuple[list[dict[str, Any]], dict[str, Executor]]:
    """Return (schemas, executors) for the named subset, in the given order."""
    schemas = [SCHEMAS[n] for n in names]
    execs = {n: EXECUTORS[n] for n in names}
    return schemas, execs
