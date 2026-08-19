"""Vision-first browser evaluation tools.

The evaluator should judge visualizations from visual evidence first. The main
tool therefore returns page telemetry and a screenshot image together, so text
claims cannot quietly substitute for broken or absent charts.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ToolContent = str | list[dict[str, Any]]
Executor = Callable[[dict[str, Any], "EvalContext"], ToolContent]

_MAX_OUTPUT = 12000
_MAX_IMAGE_B64_CHARS = 6_800_000
_MAX_VISION_DIMENSION = 7_800


def _clip(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit - 200]}\n... [truncated {len(text) - limit + 200} chars]"


def _reason(exc: Exception) -> str:
    """Drop Playwright's retry call log; keep the actionable reason."""
    return str(exc).strip().splitlines()[0][:240]


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


# Page prose only. inner_text("body") scrapes every SVG axis tick label too.
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


_PAGE_STATS_JS = """() => {
    const rectOf = (el) => {
        const r = el.getBoundingClientRect();
        return {w: Math.round(r.width), h: Math.round(r.height), x: Math.round(r.x), y: Math.round(r.y)};
    };
    const plotly = [...document.querySelectorAll('.js-plotly-plot')].map((el) => {
        const traces = el.data || [];
        const points = traces.reduce((a, t) => a + ((t.y || t.values || t.x || []).length), 0);
        let title = '';
        try { title = (el.layout && el.layout.title && (el.layout.title.text || el.layout.title)) || ''; } catch(e) {}
        return {id: el.id || '(anon)', traces: traces.length, points, title: String(title).slice(0, 120), rect: rectOf(el)};
    });
    const svgs = [...document.querySelectorAll('svg')].map((el) => ({children: el.children.length, rect: rectOf(el)}));
    const canvases = [...document.querySelectorAll('canvas')].map((el) => {
        let painted = 'unknown';
        try {
            const ctx = el.getContext('2d');
            const W = el.width || 1, H = el.height || 1, S = 48;
            const spots = [[0, 0], [Math.max(0, (W - S) >> 1), Math.max(0, (H - S) >> 1)], [Math.max(0, W - S), Math.max(0, H - S)]];
            painted = spots.some(([px, py]) => {
                const d = ctx.getImageData(px, py, Math.min(S, W), Math.min(S, H)).data;
                return Array.from(d).some(v => v !== 0);
            });
        } catch(e) {}
        return {painted, rect: rectOf(el)};
    });
    const controls = [...document.querySelectorAll('button, select, input, [role=tab], [role=button], [onclick], .dropdown-toggle, [data-toggle]')]
        .slice(0, 35)
        .map((el) => {
            const text = (el.innerText || el.value || el.title || el.getAttribute('aria-label') || '').trim().slice(0, 80);
            const id = el.id ? '#' + el.id : '';
            const cls = el.className && el.className.toString ? el.className.toString().split(' ')[0] : '';
            const tag = el.tagName.toLowerCase();
            const hint = id || (cls ? tag + '.' + cls : tag);
            return {tag, type: el.type || '', text, selector_hint: hint, disabled: !!el.disabled, rect: rectOf(el)};
        });
    const external = [...document.querySelectorAll('script[src],link[href],img[src]')]
        .map(e => e.src || e.href)
        .filter(u => u && !u.includes(location.host))
        .slice(0, 20);
    const visibleText = document.body ? document.body.innerText.trim().length : 0;
    return {plotly, svgs, canvases, controls, external, visibleText, bodyRect: rectOf(document.body)};
}"""


@dataclass
class EvalContext:
    workdir: Path
    artifact_url: str
    _page: Any = field(default=None, init=False, repr=False)
    _browser: Any = field(default=None, init=False, repr=False)
    _pw: Any = field(default=None, init=False, repr=False)
    _console_errors: list[str] = field(default_factory=list, init=False)
    _page_errors: list[str] = field(default_factory=list, init=False)
    _loaded: bool = field(default=False, init=False)
    _observation_count: int = field(default=0, init=False)
    artifacts_dir: Path = field(init=False)
    observations: list[dict[str, Any]] = field(default_factory=list, init=False)
    actions: list[dict[str, Any]] = field(default_factory=list, init=False)
    controls_seen: int = field(default=0, init=False)
    load_failed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir).resolve()
        self.artifacts_dir = self.workdir / ".artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def ensure_browser(self) -> None:
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._page = self._browser.new_page(viewport={"width": 1440, "height": 900})
        self._page.on("console", self._on_console)
        self._page.on("pageerror", lambda err: self._page_errors.append(str(err)))

    def ensure_loaded(self) -> None:
        self.ensure_browser()
        if self._loaded:
            return
        self._page.goto(self.artifact_url, wait_until="networkidle", timeout=45000)
        self._loaded = True

    def _on_console(self, msg: Any) -> None:
        if msg.type == "error":
            self._console_errors.append(msg.text)

    def drain_errors(self) -> tuple[list[str], list[str]]:
        console, page = list(self._console_errors), list(self._page_errors)
        self._console_errors.clear()
        self._page_errors.clear()
        return console, page

    def close(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass

    def resolve(self, path: str | None) -> Path:
        if not path:
            return self.workdir
        for prefix in ("WORKDIR/", "WORKDIR\\", "./WORKDIR/"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        p = Path(path)
        return p if p.is_absolute() else (self.workdir / p)


def read_file(args: dict[str, Any], ctx: EvalContext) -> str:
    """Read a bounded text file or list a directory."""
    path = ctx.resolve(args.get("path"))
    if not path.exists():
        return f"Error: no such file: {path}"
    if path.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return _clip(f"{path} (directory):\n" + "\n".join(entries[:200]))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Error reading {path}: {exc}"
    lines = text.splitlines()
    offset = max(1, int(args.get("offset", 1)))
    limit = max(1, min(int(args.get("limit", 400)), 1200))
    chunk = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{offset + i}\t{ln}" for i, ln in enumerate(chunk))
    footer = ""
    if offset - 1 + limit < len(lines):
        footer = f"\n... [{len(lines) - (offset - 1 + limit)} more lines; use offset to continue]"
    return _clip(numbered + footer)


def observe(args: dict[str, Any], ctx: EvalContext) -> ToolContent:
    """Return a visual observation: telemetry plus a screenshot image."""
    full_page = bool(args.get("full_page", False))
    wait_ms = int(args.get("wait_ms") or 1000)
    label = str(args.get("label") or "").strip()[:80]
    selector = str(args.get("selector") or "").strip()
    try:
        ctx.ensure_loaded()
    except Exception as exc:
        ctx.load_failed = True
        return f"observe error: {_reason(exc)}"
    try:
        if selector:
            ctx._page.locator(selector).first.scroll_into_view_if_needed(timeout=5000)
        ctx._page.wait_for_timeout(wait_ms)
        return _make_observation(ctx, label=label, full_page=full_page, selector=selector or None)
    except Exception as exc:
        return f"observe error: {_reason(exc)}"


def inspect(args: dict[str, Any], ctx: EvalContext) -> str:
    if not ctx._loaded:
        return "Error: call observe first."
    selector = args.get("selector")
    if not selector:
        return "Error: requires 'selector'."

    page = ctx._page
    try:
        loc = page.locator(selector)
        count = loc.count()
        if count == 0:
            return f"inspect {selector!r}: 0 matches"

        max_show = min(count, 10)
        items: list[str] = []
        for i in range(max_show):
            try:
                el = loc.nth(i)
                text = el.inner_text(timeout=1500)[:300]
                visible = el.is_visible(timeout=1500)
                tag = el.evaluate("el => el.tagName.toLowerCase()")
                items.append(f"  [{i}] <{tag}> visible={visible} text={text!r}")
            except Exception as exc:
                items.append(f"  [{i}] error: {_reason(exc)}")

        more = f"\n  ... +{count - max_show} more" if count > max_show else ""
        console, page_errors = ctx.drain_errors()
        errors = _format_errors(console, page_errors)
        return _clip(f"inspect {selector!r}: {count} matches\n" + "\n".join(items) + more + f"\n{errors}")
    except Exception as exc:
        return f"inspect error: {_reason(exc)}"


def act(args: dict[str, Any], ctx: EvalContext) -> ToolContent:
    """Interact, then immediately return a visual observation of the new state."""
    if not ctx._loaded:
        return "Error: call observe first."
    action = args.get("action")
    selector = args.get("selector")
    if not action or not selector:
        return "Error: requires 'action' and 'selector'."

    page = ctx._page
    before_text = _safe_body_text(page, limit=800)
    status = "ok"
    detail = ""
    try:
        if action == "click":
            page.click(selector, timeout=5000)
            page.wait_for_timeout(700)
        elif action == "hover":
            page.hover(selector, timeout=5000)
            page.wait_for_timeout(500)
        elif action == "select":
            value = args.get("value", "")
            if not value:
                return "Error: select requires 'value'."
            page.select_option(selector, value, timeout=5000)
            page.wait_for_timeout(700)
        elif action == "type":
            page.fill(selector, args.get("value", ""), timeout=5000)
            page.wait_for_timeout(700)
        elif action == "scroll_to":
            page.evaluate("sel => document.querySelector(sel)?.scrollIntoView({behavior:'instant',block:'center'})", selector)
            page.wait_for_timeout(500)
        else:
            return f"Error: unknown action {action!r}. Use: click, hover, select, type, scroll_to."
    except Exception as exc:
        status = "FAILED"
        detail = _reason(exc)

    after_text = _safe_body_text(page, limit=800)
    changed = before_text != after_text
    ctx.actions.append({"action": action, "selector": selector, "status": status, "changed_text": changed})
    prefix = f"ACTION {action} {selector!r}: {status}; text_changed={changed}"
    if detail:
        prefix += f"; reason={detail}"
    return _make_observation(ctx, label=f"after {action} {selector}", full_page=False, selector=None, prefix=prefix)


def _make_observation(
    ctx: EvalContext,
    *,
    label: str,
    full_page: bool,
    selector: str | None,
    prefix: str = "",
) -> ToolContent:
    page = ctx._page
    title = page.title()
    body_text = page.evaluate(_BODY_TEXT_JS)[:2200]
    stats = page.evaluate(_PAGE_STATS_JS)
    console, page_errors = ctx.drain_errors()

    ctx._observation_count += 1
    obs_id = f"obs_{ctx._observation_count}"
    shot = ctx.artifacts_dir / f"eval_{ctx._observation_count}.jpg"
    capture_note = _capture_screenshot(page, shot, full_page=full_page, selector=selector, stats=stats)

    ctx.controls_seen = max(ctx.controls_seen, len(stats.get("controls") or []))
    observation = {
        "id": obs_id,
        "label": label,
        "screenshot": str(shot),
        "full_page": full_page,
        "selector": selector,
        "title": title,
        "stats": _summarize_stats(stats),
        "console_errors": console[:10],
        "page_errors": page_errors[:10],
        "capture_note": capture_note,
    }
    ctx.observations.append(observation)

    text = _observation_text(
        obs_id=obs_id,
        label=label,
        title=title,
        url=ctx.artifact_url,
        shot=shot,
        stats=stats,
        console=console,
        page_errors=page_errors,
        body_text=body_text,
        prefix=prefix,
        capture_note=capture_note,
    )
    return [{"type": "text", "text": text}, *_image_parts(shot)]


def _observation_text(
    *,
    obs_id: str,
    label: str,
    title: str,
    url: str,
    shot: Path,
    stats: dict[str, Any],
    console: list[str],
    page_errors: list[str],
    body_text: str,
    prefix: str,
    capture_note: str,
) -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts += [
        f"VISUAL OBSERVATION {obs_id}" + (f" ({label})" if label else ""),
        f"url: {url}",
        f"title: {title!r}",
        _format_errors(console, page_errors),
        _stats_text(stats),
        f"screenshot: {shot}",
        f"capture_note: {capture_note}",
        "Use the screenshot as primary evidence for visual craft, rendered charts, layout, and whether controls changed the view.",
        f"body_text_sample:\n{body_text}",
    ]
    return _clip("\n".join(parts))


def _stats_text(stats: dict[str, Any]) -> str:
    plotly = stats.get("plotly") or []
    svgs = stats.get("svgs") or []
    canvases = stats.get("canvases") or []
    controls = stats.get("controls") or []
    drawn_svg = sum(1 for s in svgs if (s.get("children") or 0) > 0 and _visible_rect(s.get("rect")))
    painted_canvas = sum(1 for c in canvases if c.get("painted") is True and _visible_rect(c.get("rect")))
    plotly_details = "; ".join(
        f"{p.get('id')} traces={p.get('traces')} points={p.get('points')} rect={p.get('rect')} title={p.get('title')!r}"
        for p in plotly[:10]
    ) or "none"
    controls_text = "\n".join(
        f"  {c.get('selector_hint')} ({c.get('tag')}{':' + c.get('type') if c.get('type') else ''}) "
        f"text={c.get('text')!r} disabled={c.get('disabled')}"
        for c in controls[:20]
    ) or "  none"
    return "\n".join([
        f"render_stats: plotly={len(plotly)} svg={len(svgs)} drawn_svg={drawn_svg} "
        f"canvas={len(canvases)} painted_canvas={painted_canvas} visible_text_chars={stats.get('visibleText')}",
        f"plotly_details: {plotly_details}",
        f"external_resource_refs: {stats.get('external') or []}",
        f"interactive_elements ({len(controls)}):",
        controls_text,
    ])


def _summarize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    svgs = stats.get("svgs") or []
    canvases = stats.get("canvases") or []
    plotly = stats.get("plotly") or []
    return {
        "plotly": len(plotly),
        "plotly_with_points": sum(1 for p in plotly if (p.get("points") or 0) > 0 and _visible_rect(p.get("rect"))),
        "svg": len(svgs),
        "drawn_svg": sum(1 for s in svgs if (s.get("children") or 0) > 0 and _visible_rect(s.get("rect"))),
        "canvas": len(canvases),
        "painted_canvas": sum(1 for c in canvases if c.get("painted") is True and _visible_rect(c.get("rect"))),
        "controls": len(stats.get("controls") or []),
        "external_resources": stats.get("external") or [],
        "visible_text_chars": stats.get("visibleText") or 0,
    }


def _visible_rect(rect: Any) -> bool:
    return isinstance(rect, dict) and (rect.get("w") or 0) >= 5 and (rect.get("h") or 0) >= 5


def _format_errors(console: list[str], page_errors: list[str]) -> str:
    return "\n".join([
        "console_errors: " + ("; ".join(console[:10]) if console else "none"),
        "page_errors: " + ("; ".join(page_errors[:10]) if page_errors else "none"),
    ])


def _safe_body_text(page: Any, limit: int) -> str:
    try:
        return page.evaluate(_BODY_TEXT_JS)[:limit]
    except Exception:
        return ""


def _capture_screenshot(page: Any, path: Path, *, full_page: bool, selector: str | None, stats: dict[str, Any]) -> str:
    """Capture a JPEG that Bedrock vision will accept.

    Claude rejects images with either dimension over 8000px. Long pages are
    therefore clipped and should be inspected through scroll_to + observe
    section captures instead of one giant full-page image.
    """
    if selector:
        loc = page.locator(selector).first
        box = loc.bounding_box(timeout=5000)
        if box and (box.get("width", 0) > _MAX_VISION_DIMENSION or box.get("height", 0) > _MAX_VISION_DIMENSION):
            clip = {
                "x": max(0, box.get("x", 0)),
                "y": max(0, box.get("y", 0)),
                "width": max(1, min(box.get("width", 1), _MAX_VISION_DIMENSION)),
                "height": max(1, min(box.get("height", 1), _MAX_VISION_DIMENSION)),
            }
            page.screenshot(path=str(path), type="jpeg", quality=72, clip=clip)
            return f"selector capture clipped to {int(clip['width'])}x{int(clip['height'])} for vision API limits"
        loc.screenshot(path=str(path), type="jpeg", quality=72)
        return "selector capture"

    body = stats.get("bodyRect") if isinstance(stats, dict) else {}
    body_h = int((body or {}).get("h") or 0)
    viewport = page.viewport_size or {"width": 1440, "height": 900}
    width = min(int(viewport.get("width") or 1440), _MAX_VISION_DIMENSION)
    if full_page and body_h > _MAX_VISION_DIMENSION:
        page.screenshot(
            path=str(path),
            type="jpeg",
            quality=72,
            clip={"x": 0, "y": 0, "width": width, "height": _MAX_VISION_DIMENSION},
        )
        return f"full_page requested but page height {body_h}px exceeds vision limit; captured top {width}x{_MAX_VISION_DIMENSION}px"
    page.screenshot(path=str(path), type="jpeg", quality=72, full_page=full_page)
    return "full page capture" if full_page else "viewport capture"


def _image_parts(path: Path) -> list[dict[str, Any]]:
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


SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": _fn(
        "read_file",
        "Read a text file or list a directory. Use offset/limit for line-range reads instead of reading the whole file.",
        {
            "path": {"type": "string", "description": "Path relative to the workdir."},
            "offset": {"type": "integer", "description": "1-indexed start line; default 1."},
            "limit": {"type": "integer", "description": "Max lines to return, capped at 1200; default 400."},
        },
        ["path"],
    ),
    "observe": _fn(
        "observe",
        "Load or re-observe the artifact and return BOTH visual screenshot evidence and telemetry: title, body prose, "
        "console/page errors, chart/render stats, external resources, and discovered controls. This is your primary evidence.",
        {
            "label": {"type": "string", "description": "Short reason for this observation."},
            "full_page": {"type": "boolean", "description": "Capture the full scrollable page instead of the viewport."},
            "selector": {"type": "string", "description": "Optional CSS selector to scroll to and screenshot."},
            "wait_ms": {"type": "integer", "description": "Extra wait before capture; default 1000."},
        },
        [],
    ),
    "inspect": _fn(
        "inspect",
        "Query DOM elements by CSS selector for precise text such as chart titles, axis labels, legends, table rows, or tooltip text. "
        "Use it to support, not replace, visual observations.",
        {"selector": {"type": "string", "description": "CSS selector."}},
        ["selector"],
    ),
    "act": _fn(
        "act",
        "Perform a UI action, then automatically return a fresh visual observation of the resulting page state. "
        "Use this for filters, tabs, dropdowns, search boxes, hover tooltips, and scrolling to sections.",
        {
            "action": {
                "type": "string",
                "enum": ["click", "hover", "select", "type", "scroll_to"],
                "description": "Interaction type.",
            },
            "selector": {"type": "string", "description": "CSS selector for the target element."},
            "value": {"type": "string", "description": "Value for select/type actions."},
        },
        ["action", "selector"],
    ),
}

EXECUTORS: dict[str, Executor] = {
    "read_file": read_file,
    "observe": observe,
    "inspect": inspect,
    "act": act,
}


def build_eval_toolset() -> tuple[list[dict[str, Any]], dict[str, Executor]]:
    return list(SCHEMAS.values()), dict(EXECUTORS)
