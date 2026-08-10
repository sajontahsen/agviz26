"""Agent tools (schema + executor pairs) operating on the workdir via ToolContext.

Executors return an error string rather than raising, so one bad call never crashes a stage.
"""
from __future__ import annotations

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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Error writing {path}: {exc}"
    n_lines = content.count("\n") + 1
    return f"Wrote {path} ({len(content)} bytes, {n_lines} lines)."


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


def _playwright_probe(url: str, actions: list[dict[str, Any]], shot: Path) -> str:
    from playwright.sync_api import sync_playwright

    console_errors: list[str] = []
    page_errors: list[str] = []
    action_log: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("console", lambda m: console_errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.goto(url, wait_until="networkidle", timeout=45000)
            title = page.title()
            text = page.evaluate(_BODY_TEXT_JS)[:1500]
            charts = page.evaluate(_CHART_HEALTH_JS)
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
    parts += [
        f"controls ({controls['total']}): {ctrl_str}",
        f"screenshot: {shot}",
    ]
    if action_log:
        parts.append("actions:\n  " + "\n  ".join(action_log))
    parts.append(f"body_text_sample:\n{text}")
    return "\n".join(parts)


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


def _which(name: str) -> str | None:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


# ---------------------------------------------------------------------------
# Tool registry — schema + executor, selectable per stage by name
# ---------------------------------------------------------------------------

Executor = Callable[[dict[str, Any], ToolContext], str]


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
        "Create or overwrite a file with exact content. Prefer this over bash heredocs.",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
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
        "Render an artifact with a headless browser and return a compact health summary (console errors, title, "
        "text sample, screenshot) plus results of optional interaction probes. With no 'url', serves dist/ locally. "
        "You MUST call this before finishing a build.",
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
}


EXECUTORS: dict[str, Executor] = {
    "read_file": read_file,
    "write_file": write_file,
    "str_replace": str_replace,
    "search": search,
    "bash": bash,
    "verify": verify,
}


def build_toolset(names: list[str]) -> tuple[list[dict[str, Any]], dict[str, Executor]]:
    """Return (schemas, executors) for the named subset, in the given order."""
    schemas = [SCHEMAS[n] for n in names]
    execs = {n: EXECUTORS[n] for n in names}
    return schemas, execs
