"""The v1 generation pipeline: profile -> planner -> analyst -> coder.

Each stage is a short LLM session (see runner.run_stage) that reads compact
files produced by earlier stages and writes one file forward:

    profile   task.md + data/            -> profile.json     (data structure)
    planner   task.md + profile.json     -> questions.json   (analytical asks)
    analyst   + questions.json + data/    -> findings.json    (verified answers)
    coder     task.md + findings.json     -> dist/index.html  (the artifact)

Models are routed by role: on the cloud, Sonnet for mechanical stages and Opus
for planning/analysis; locally everything uses LOCAL_MODEL (a cheap OpenAI model
for smoke tests). The raw dataset is only ever touched by Python scripts the
stages write and run — never slurped into the model context.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from html import escape
from pathlib import Path
from typing import Any

from llm_client import make_llm_client

from .runner import StageResult, run_stage
from .tools import ToolContext


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

CLOUD_SONNET = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
CLOUD_OPUS = "global.anthropic.claude-opus-4-8"
LOCAL_MODEL = os.environ.get("VIS_ARENA_LOCAL_MODEL", "gpt-5-nano")

_CLOUD_ROLES = {
    "profile": CLOUD_SONNET,
    "planner": CLOUD_OPUS,
    "analyst": CLOUD_OPUS,
    "coder": CLOUD_SONNET,
}


def pick_model(role: str) -> str:
    """Cloud jobs get role-appropriate Claude models; local runs use LOCAL_MODEL."""
    if os.environ.get("VIS_ARENA_JOB_ID"):
        return _CLOUD_ROLES.get(role, CLOUD_SONNET)
    return LOCAL_MODEL


# ---------------------------------------------------------------------------
# Stage prompts
# ---------------------------------------------------------------------------

PROFILE_SYSTEM = """You are a data-profiling agent. Produce a compact, accurate structural profile of a dataset so later stages can reason about it WITHOUT loading the raw data.

Rules:
- Never dump raw data into your reasoning. Inspect only schema/headers/samples, then write a Python script and let IT compute the profile.
- Data may be tabular (CSV) OR non-tabular (e.g. a JSON knowledge graph). Detect the shape and profile accordingly.

Workflow:
1. read_file task.md and list data/ to see the files and their formats.
2. Write source/profile.py that emits profile.json (at the workdir root). Run it with bash. Iterate until it runs cleanly.
3. Call finish with a one-line summary.

profile.json must be compact (aim < 40KB) and include:
- files: name, format, size.
- For tabular data: per-column {name, dtype, null_fraction, distinct_count, min, max, and up to 5 example values}; row_count.
- For graph data: node_count, edge_count; node types with counts; edge types with counts; per node-type attribute coverage (which attrs, % present); degree stats; any temporal fields with min/max year.
Prefer pandas/networkx. Keep example lists short."""

PLANNER_SYSTEM = """You are a visualization PLANNING agent. You do NOT write code and you do NOT compute answers. You decompose the task into the analytical questions whose answers will drive an insightful, well-structured visualization.

Inputs (read them): task.md (what was asked) and profile.json (the data's structure).

Break the task down yourself: cover every explicit ask in task.md (if it enumerates sub-questions, each must be represented) AND the higher-level narrative/insight the artifact should deliver.

Write questions.json (at the workdir root): an object {"questions": [ ... ]}. Each question:
  - id: short snake_case id
  - question: the analytical question in plain language
  - rationale: why it matters for the task and the story
  - computation_hint: concretely how to compute it from the data (which columns/edge-types/aggregation), grounded in profile.json field names
  - expected_form: one of scalar | series | table | ranking | breakdown
  - supports: which task requirement or narrative beat it serves

Aim for a focused set (roughly 6-12 questions) that together give the coder enough verified material for a complete, non-trivial story. Then call finish."""

ANALYST_SYSTEM = """You are a data ANALYST agent. Compute correct, verified answers to a set of questions directly from the raw dataset. These numbers become the ground truth the visualization displays, so correctness is paramount.

Inputs: task.md, profile.json, and questions.json (read all three). Raw data is in data/.

Workflow:
1. Read questions.json and profile.json.
2. Write source/analyze.py that loads data/ and computes an answer for EACH question id. Run it with bash; iterate until clean.
3. Emit findings.json (at the workdir root): {"findings": [ {id, answer, values, method, caveats} ... ]} where
   - answer: concise plain-language answer
   - values: the supporting numbers as a SMALL, plot-ready structure (e.g. a series of {x,y}, a top-N ranking, a small table). Keep it compact — top-N and aggregates only, never raw dumps.
   - method: one line on how it was computed
   - caveats: any data limitations (optional)
4. Call finish.

Keep findings.json well under ~100KB. If a result would be huge, aggregate or cap to top-N."""

CODER_SYSTEM = """You are a web data-visualization engineer. Build ONE cohesive, self-contained, interactive artifact at dist/index.html that tells a clear story and lets a reviewer inspect the answers to the task.

Inputs: task.md (what was asked) and findings.json (VERIFIED facts — display these; do not invent or alter numbers). profile.json is available for structure. Raw data is in data/ if you need finer aggregates for a chart.

How to build (this avoids the per-message output-token limit):
- Write source/build.py that (a) reads findings.json (and data/ if needed), (b) computes compact plot-ready aggregates, (c) writes dist/index.html with the data EMBEDDED as inline JSON. Run it with bash. Build the HTML in the script; use str_replace for edits rather than re-emitting whole files.
- Rendering libraries (pin these exact versions, load from CDN):
    Plotly.js  -> https://cdn.plot.ly/plotly-2.35.2.min.js   (statistical/comparative charts)
    Cytoscape.js -> https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js   (network/graph panels)
  Use Plotly for most charts; use Cytoscape only for network/graph views. Never render tens of thousands of nodes raw — show ego-graphs / aggregates.
- The page MUST render from dist/index.html with no dev server (data inline; only the two CDN libs are external).

Quality bar (you are judged on these):
- data_fidelity: numbers on screen match findings.json.
- insightfulness: call out trends, exceptions, comparisons — not just raw charts.
- narrative_coherence: a hook -> build -> payoff arc; consistent encodings across panels.
- visual_craft: right chart types, clear titles/axes/labels, disclosed filters/timeframes/scope, readable color.
- functionality: interactions (filters, tooltips, selection) actually work.

Before finishing you MUST call verify (it renders dist/ and returns console errors + a screenshot). Fix any console/page errors and re-verify until clean. Then call finish with a short summary of the panels you built."""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def orchestrate(workdir: Path) -> dict[str, Any]:
    workdir = Path(workdir).resolve()
    ctx = ToolContext(workdir)
    client = make_llm_client("generation")
    wd = str(workdir)

    # Stage specs run in order. Each is isolated: a crash in one is caught and
    # the pipeline continues best-effort (later stages degrade gracefully off
    # whatever files exist). This matters because cloud jobs give us no stderr
    # or traceback — an uncaught crash would surface only as an opaque "failed".
    specs = [
        dict(
            name="profile", system_prompt=PROFILE_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nProfile the dataset. Read task.md and data/, then write and run source/profile.py to emit profile.json.",
            tool_names=["read_file", "write_file", "str_replace", "bash"],
            model=pick_model("profile"), max_steps=20,
        ),
        dict(
            name="planner", system_prompt=PLANNER_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead task.md and profile.json, then write questions.json.",
            tool_names=["read_file", "write_file"],
            model=pick_model("planner"), max_steps=14,
        ),
        dict(
            name="analyst", system_prompt=ANALYST_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead questions.json/profile.json/task.md, then write and run source/analyze.py to emit findings.json.",
            tool_names=["read_file", "write_file", "str_replace", "bash", "search"],
            model=pick_model("analyst"), max_steps=30,
        ),
        dict(
            name="coder", system_prompt=CODER_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead task.md and findings.json, then write and run source/build.py to produce dist/index.html. Verify it renders before finishing.",
            tool_names=["read_file", "write_file", "str_replace", "bash", "search", "verify"],
            model=pick_model("coder"), max_steps=50, prune_keep=6,
        ),
    ]

    results = [_safe_run(ctx=ctx, client=client, **spec) for spec in specs]

    # Guarantee a renderable artifact so the job always yields a scorable
    # preview to inspect, rather than a hard "dist/index.html was not created".
    _ensure_fallback_dist(workdir)

    return _summarize(workdir, results)


def _safe_run(*, ctx: ToolContext, client: Any, name: str, **kwargs: Any) -> StageResult:
    """Run one stage, converting any crash into an unfinished result + stderr trace."""
    try:
        return run_stage(name=name, ctx=ctx, client=client, **kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately total: one stage must not abort the run
        print(f"[stage:{name}] CRASHED: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        return StageResult(name=name, result={"error": str(exc)}, finished=False)


def _summarize(workdir: Path, results: list[StageResult]) -> dict[str, Any]:
    """Report generation telemetry to stderr and to the returned dict's
    ``notes`` (which agent.py writes into generation.json). Note: in cloud the
    authoritative token spend comes from the arena's usage API
    (`vis-arena submissions usage <id>`), which already splits generation vs
    evaluation per job; this per-stage breakdown is a local-dev aid. The
    workspace is discarded after the job, so nothing is written for later reading.
    """
    total = sum(r.usage["total_tokens"] for r in results)
    stages_meta = [
        {"stage": r.name, "finished": r.finished, "steps": r.steps, **r.usage}
        for r in results
    ]
    dist_ready = (workdir / "dist" / "index.html").exists()
    # Compact per-stage line, embedded in notes so it persists via generation.json.
    per_stage = " ".join(f"{s['stage']}={s['total_tokens']}({'ok' if s['finished'] else 'unfinished'})" for s in stages_meta)
    notes = f"staged pipeline (profile->planner->analyst->coder); tokens total={total} [{per_stage}]; dist_ready={dist_ready}"

    print("[pipeline] token spend by stage:", file=sys.stderr)
    for s in stages_meta:
        print(f"  {s['stage']:<9} finished={s['finished']} steps={s['steps']:<3} tokens={s['total_tokens']}", file=sys.stderr)
    print(f"[pipeline] TOTAL tokens={total}  dist_ready={dist_ready}", file=sys.stderr)

    return {
        "notes": notes,
        "total_tokens": total,
        "stages": stages_meta,
        "dist_ready": dist_ready,
    }


# ---------------------------------------------------------------------------
# Fallback artifact — ensures the job always yields a renderable dist/index.html
# ---------------------------------------------------------------------------

_FALLBACK_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem;
         background: #0f1420; color: #e8eaed; line-height: 1.5; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .5rem; }}
  .note {{ color: #f0a; margin-bottom: 1.5rem; }}
  ul {{ max-width: 60rem; }} li {{ margin: .35rem 0; }}
  .k {{ color: #8ab4f8; font-weight: 600; }}
</style></head>
<body>
  <h1>{title}</h1>
  <p class="note">Generation did not complete a full interactive artifact; showing available findings.</p>
  {findings}
</body></html>
"""


def _ensure_fallback_dist(workdir: Path) -> None:
    """Write a minimal, valid dist/index.html only if the pipeline produced none.

    Must never raise. A tiny/empty file (truncated write) counts as "no artifact".
    """
    dist = workdir / "dist" / "index.html"
    try:
        if dist.exists() and dist.stat().st_size > 200:
            return
    except OSError:
        pass
    try:
        html = _FALLBACK_TEMPLATE.format(
            title=escape(_task_title(workdir)),
            findings=_fallback_findings_html(workdir),
        )
        dist.parent.mkdir(parents=True, exist_ok=True)
        dist.write_text(html, encoding="utf-8")
        print("[pipeline] wrote fallback dist/index.html (no artifact produced)", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - last-resort; must not crash the run
        print(f"[pipeline] FAILED to write fallback dist: {exc}", file=sys.stderr)


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


def _fallback_findings_html(workdir: Path) -> str:
    """Render whatever findings.json exists as a simple list; empty string if none."""
    try:
        data = json.loads((workdir / "findings.json").read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return "<p>No findings were available.</p>"
    items = data.get("findings", data) if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        return "<p>No findings were available.</p>"
    rows = []
    for it in items[:40]:
        if not isinstance(it, dict):
            continue
        fid = escape(str(it.get("id", "")))
        ans = escape(str(it.get("answer", "")))[:400]
        rows.append(f'<li><span class="k">{fid}:</span> {ans}</li>')
    return "<ul>" + "".join(rows) + "</ul>" if rows else "<p>No findings were available.</p>"
