"""Browser-based evaluation tools with a shared Playwright session.

Structured tools (render, interact, inspect, screenshot, read_file) that operate
on a persistent browser session, replacing the raw playwright-script tool.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

Executor = Callable[[dict[str, Any], "EvalContext"], str]

_MAX_OUTPUT = 12000


def _clip(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit - 200]}\n... [truncated {len(text) - limit + 200} chars]"


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


# ---------------------------------------------------------------------------
# EvalContext — persistent browser state across tool calls within one eval run
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    workdir: Path
    artifact_url: str
    _page: Any = field(default=None, init=False, repr=False)
    _browser: Any = field(default=None, init=False, repr=False)
    _pw: Any = field(default=None, init=False, repr=False)
    _console_errors: list[str] = field(default_factory=list, init=False)
    _page_errors: list[str] = field(default_factory=list, init=False)
    _rendered: bool = field(default=False, init=False)
    _screenshot_count: int = field(default=0, init=False)
    artifacts_dir: Path = field(init=False)

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

    def _on_console(self, msg: Any) -> None:
        if msg.type == "error":
            self._console_errors.append(msg.text)

    def drain_errors(self) -> str:
        parts: list[str] = []
        if self._console_errors:
            parts.append(f"console_errors ({len(self._console_errors)}): " + "; ".join(self._console_errors[:10]))
            self._console_errors.clear()
        if self._page_errors:
            parts.append(f"page_errors ({len(self._page_errors)}): " + "; ".join(self._page_errors[:10]))
            self._page_errors.clear()
        return "\n".join(parts) if parts else "errors: none"

    def take_screenshot(self, **kwargs: Any) -> Path:
        self._screenshot_count += 1
        path = self.artifacts_dir / f"eval_{self._screenshot_count}.png"
        self._page.screenshot(path=str(path), **kwargs)
        return path

    def close(self) -> None:
        for obj in (self._browser, self._pw):
            try:
                if obj is not None:
                    obj.close() if hasattr(obj, "close") else obj.stop()
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


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

def read_file(args: dict[str, Any], ctx: EvalContext) -> str:
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
    return _clip(text)


def render(args: dict[str, Any], ctx: EvalContext) -> str:
    try:
        ctx.ensure_browser()
        page = ctx._page
        page.goto(ctx.artifact_url, wait_until="networkidle", timeout=45000)
        ctx._rendered = True

        title = page.title()
        body_text = page.inner_text("body")[:2000]

        charts = page.evaluate("""() => {
            const plotly = document.querySelectorAll('.plotly, .js-plotly-plot');
            const svgs = document.querySelectorAll('svg');
            const canvases = document.querySelectorAll('canvas');
            return {plotly: plotly.length, svg: svgs.length, canvas: canvases.length};
        }""")

        controls = page.evaluate("""() => {
            const sels = 'button, select, input, [role=tab], [role=button], [onclick], .dropdown-toggle, [data-toggle]';
            return Array.from(document.querySelectorAll(sels)).slice(0, 25).map(el => {
                const text = (el.innerText || el.value || el.title || el.getAttribute('aria-label') || '').trim().slice(0, 60);
                const id = el.id ? '#' + el.id : '';
                const cls = el.className?.toString?.()?.split(' ')[0] || '';
                const tag = el.tagName.toLowerCase();
                const hint = id || (cls ? tag + '.' + cls : tag);
                return {tag, type: el.type || '', text, selector_hint: hint};
            });
        }""")

        shot = ctx.take_screenshot(full_page=False)
        errors = ctx.drain_errors()

        parts = [
            f"url: {ctx.artifact_url}",
            f"title: {title!r}",
            errors,
            f"charts: plotly={charts['plotly']} svg={charts['svg']} canvas={charts['canvas']}",
            f"interactive_elements ({len(controls)}):",
        ]
        for c in controls:
            typ = f":{c['type']}" if c["type"] else ""
            parts.append(f"  {c['selector_hint']} ({c['tag']}{typ}) text={c['text']!r}")
        parts.append(f"screenshot: {shot}")
        parts.append(f"body_text:\n{body_text}")
        return _clip("\n".join(parts))
    except Exception as exc:
        return f"render error: {exc}"


def interact(args: dict[str, Any], ctx: EvalContext) -> str:
    if not ctx._rendered:
        return "Error: call render first."
    action = args.get("action")
    selector = args.get("selector")
    if not action or not selector:
        return "Error: requires 'action' and 'selector'."

    page = ctx._page
    try:
        if action == "click":
            page.click(selector, timeout=5000)
            page.wait_for_timeout(500)
        elif action == "hover":
            page.hover(selector, timeout=5000)
            page.wait_for_timeout(300)
        elif action == "select":
            value = args.get("value", "")
            if not value:
                return "Error: select requires 'value'."
            page.select_option(selector, value, timeout=5000)
            page.wait_for_timeout(500)
        elif action == "type":
            page.fill(selector, args.get("value", ""), timeout=5000)
        elif action == "scroll_to":
            page.evaluate(f"document.querySelector({selector!r})?.scrollIntoView({{behavior:'instant',block:'center'}})")
            page.wait_for_timeout(300)
        else:
            return f"Error: unknown action {action!r}. Use: click, hover, select, type, scroll_to."

        try:
            el_text = page.locator(selector).first.inner_text()[:300]
        except Exception:
            el_text = "(unavailable)"

        errors = ctx.drain_errors()
        return f"{action} {selector!r}: ok\n{errors}\nelement_text: {el_text}"
    except Exception as exc:
        return f"{action} {selector!r}: FAILED — {exc}"


def inspect(args: dict[str, Any], ctx: EvalContext) -> str:
    if not ctx._rendered:
        return "Error: call render first."
    selector = args.get("selector")
    if not selector:
        return "Error: requires 'selector'."

    page = ctx._page
    try:
        loc = page.locator(selector)
        count = loc.count()
        if count == 0:
            return f"inspect {selector!r}: 0 matches"

        max_show = min(count, 8)
        items: list[str] = []
        for i in range(max_show):
            try:
                el = loc.nth(i)
                text = el.inner_text()[:200]
                visible = el.is_visible()
                tag = el.evaluate("el => el.tagName.toLowerCase()")
                items.append(f"  [{i}] <{tag}> visible={visible} text={text!r}")
            except Exception as exc:
                items.append(f"  [{i}] error: {exc}")

        more = f"\n  ... +{count - max_show} more" if count > max_show else ""
        errors = ctx.drain_errors()
        return _clip(f"inspect {selector!r}: {count} matches\n" + "\n".join(items) + more + f"\n{errors}")
    except Exception as exc:
        return f"inspect error: {exc}"


def screenshot(args: dict[str, Any], ctx: EvalContext) -> str:
    if not ctx._rendered:
        return "Error: call render first."
    selector = args.get("selector")
    try:
        if selector:
            ctx._screenshot_count += 1
            path = ctx.artifacts_dir / f"eval_{ctx._screenshot_count}.png"
            ctx._page.locator(selector).first.screenshot(path=str(path))
        else:
            path = ctx.take_screenshot(full_page=bool(args.get("full_page", False)))
        return f"screenshot saved: {path}"
    except Exception as exc:
        return f"screenshot error: {exc}"


# ---------------------------------------------------------------------------
# Schema + executor registry
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": _fn(
        "read_file",
        "Read a text file in the workdir (e.g. task.md).",
        {"path": {"type": "string", "description": "File path relative to workdir."}},
        ["path"],
    ),
    "render": _fn(
        "render",
        "Load the artifact URL in a headless browser. Returns page title, body text, "
        "console errors, chart element counts, discovered interactive controls, and a "
        "screenshot. Must be called before interact/inspect/screenshot.",
        {},
        [],
    ),
    "interact": _fn(
        "interact",
        "Perform a UI interaction on the rendered page. Returns result, any new "
        "console errors, and the element text after the action.",
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
    "inspect": _fn(
        "inspect",
        "Query DOM elements by CSS selector. Returns match count, tag, visibility, "
        "and text content for each match (up to 8).",
        {"selector": {"type": "string", "description": "CSS selector."}},
        ["selector"],
    ),
    "screenshot": _fn(
        "screenshot",
        "Capture a screenshot of the current page state for evidence.",
        {
            "selector": {"type": "string", "description": "CSS selector to screenshot a specific element."},
            "full_page": {"type": "boolean", "description": "Capture the full scrollable page (default false)."},
        },
        [],
    ),
}

EXECUTORS: dict[str, Executor] = {
    "read_file": read_file,
    "render": render,
    "interact": interact,
    "inspect": inspect,
    "screenshot": screenshot,
}


def build_eval_toolset() -> tuple[list[dict[str, Any]], dict[str, Executor]]:
    return list(SCHEMAS.values()), dict(EXECUTORS)
