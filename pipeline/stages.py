"""Generation pipeline: profile -> planner -> analyst -> coder, each a short LLM stage passing files forward."""
from __future__ import annotations

import json
import os
import sys
import traceback
from html import escape
from pathlib import Path
from typing import Any

from llm_client import make_llm_client

from .runner import BudgetTracker, StageResult, run_stage
from .tools import ToolContext

# Cap generation so the shared per-job budget (~1M cloud) leaves room for evaluation.
GEN_TOKEN_CEILING = 650_000


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

CLOUD_SONNET = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
CLOUD_OPUS = "global.anthropic.claude-opus-4-8"
LOCAL_MODEL = os.environ.get("VIS_ARENA_LOCAL_MODEL", "gpt-5-nano")

_CLOUD_ROLES = {
    "profile": CLOUD_OPUS,
    "planner": CLOUD_OPUS,
    "analyst": CLOUD_OPUS,
    "coder": CLOUD_OPUS,
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
3. Emit findings.json (at the workdir root): {"findings": [ {id, answer, data_profile, method, caveats} ... ]} where
   - id: Must exactly match the id from questions.json
   - answer: concise plain-language answer
   - data_profile: A schema definition containing:
        - "filepath": Local path where analyze.py saved the dataset (e.g., "outputs/<id>.csv" or "outputs/<id>.json"). Choose the best format for the data shape.
        - "format": "csv" or "json"
        - "schema": For tabular data, map columns to types. For graphs/nested data, describe the structure.
        - "sample": A minimal snippet (e.g., 2 rows, or 1 node/edge) showing exact structure.
   - method: one line on how it was computed
   - caveats: any data limitations (optional)
4. Call finish.

CRITICAL INSTRUCTION: Your analyze.py script MUST save the calculated, plot-ready data to separate files in an `outputs/` directory. NEVER inline full data arrays into findings.json. The `data_profile` must be generated programmatically by analyze.py to guarantee accuracy. Do not copy the original questions or rationales into findings.json; the system will merge those later."""

CODER_SYSTEM = """You are a web data-visualization engineer. Build ONE cohesive, self-contained, interactive artifact at dist/index.html that tells a clear story and lets a reviewer inspect the answers to the task.

Read these (workdir root):
- task.md — what was asked.
- viz_context.json — the visualizations to build, under "visualizations_to_build". Each item has: id, question, rationale, expected_form, answer, method, caveats, and a `data_profile` = {filepath, format, schema, sample}. The `data_profile` tells you where the plot-ready data lives and its exact structure. Use the exact fields shown in `schema`/`sample` — never rename or invent keys (mismatched keys silently break charts).

Build:
- Write source/build.py that reads each finding's data from its data_profile.filepath (relative to the workdir), embeds what it needs inline as JSON (use `json.dumps()` to safely inject into `<script>` tags), and writes dist/index.html. Run it with bash. You do NOT need to read the data files into your own context — build.py reads them.
- One panel per relevant finding, chart type matched to its expected_form / schema. Display the computed numbers as-is. Write defensive JavaScript to handle potential nulls or missing keys smoothly.
- Rendering libraries (pin these exact versions; load from CDN):
 Plotly.js https://cdn.plot.ly/plotly-2.35.2.min.js
 Cytoscape.js https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js
Plotly for statistical/comparative charts, Cytoscape only for network/graph findings. Never render tens of thousands of nodes raw.
- The page must render from dist/index.html with no dev server;
- Token Economy: Work economically — the job has a shared token budget, and long transcripts spend it fast.

You are judged on: data_fidelity (numbers match the findings), insightfulness (call out trends/exceptions/comparisons), narrative_coherence (hook->build->payoff, consistent encodings), visual_craft (right chart types, clear titles/axes/labels, disclosed scope), functionality (working interactions).

Validation Loop:
When the page is written, you MUST call the `verify` tool (with no arguments, to serve dist/ locally). It captures a screenshot and returns a health summary.
If `verify` reports ANY `console_errors` or `page_errors`:
1. Use `str_replace` or rewrite to fix the bug in build.py/HTML.
2. Re-run `build.py` via bash to generate the new dist/index.html.
3. Call `verify` again.
Iterate until `verify` passes with 0 errors and non-empty charts. Then call finish with a short summary of the panels."""

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def orchestrate(workdir: Path) -> dict[str, Any]:
    workdir = Path(workdir).resolve()
    ctx = ToolContext(workdir)
    client = make_llm_client("generation")
    wd = str(workdir)

    # Run in order; each stage is isolated so a crash doesn't abort the rest (cloud gives no traceback).
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
            user_prompt=f"WORKDIR={wd}\nRead task.md and viz_context.json, then write and run source/build.py to produce dist/index.html. Verify it renders before finishing.",
            tool_names=["read_file", "write_file", "str_replace", "bash", "search", "verify"],
            model=pick_model("coder"), max_steps=50,
        ),
    ]

    budget = BudgetTracker(ceiling=GEN_TOKEN_CEILING)
    results: list[StageResult] = []
    for spec in specs:
        results.append(_safe_run(ctx=ctx, client=client, budget=budget, **spec))
        if spec["name"] == "analyst":
            # Deterministic join of questions x findings -> viz_context.json.
            _safe_merge_context(workdir)

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
    """Log per-stage token telemetry to stderr and into the returned ``notes`` (goes to generation.json)."""
    total = sum(r.usage["total_tokens"] for r in results)
    stages_meta = [
        {"stage": r.name, "finished": r.finished, "steps": r.steps, **r.usage}
        for r in results
    ]
    dist_ready = (workdir / "dist" / "index.html").exists()
    # Compact per-stage line, embedded in notes so it persists via generation.json.
    per_stage = " ".join(f"{s['stage']}={s['total_tokens']}({'ok' if s['finished'] else 'unfinished'})" for s in stages_meta)
    notes = (f"staged pipeline (profile->planner->analyst->coder); tokens total={total}/{GEN_TOKEN_CEILING} "
             f"[{per_stage}]; dist_ready={dist_ready}")

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
# Deterministic merge (post-analyst): join questions x findings -> viz_context.json.
# ---------------------------------------------------------------------------

def _safe_merge_context(workdir: Path) -> None:
    try:
        _merge_context(workdir)
    except Exception as exc:  # noqa: BLE001 - glue must not abort the run
        print(f"[pipeline] merge_context skipped: {exc}", file=sys.stderr)


def _merge_context(workdir: Path) -> None:
    questions = json.loads((workdir / "questions.json").read_text(encoding="utf-8"))["questions"]
    findings = json.loads((workdir / "findings.json").read_text(encoding="utf-8"))["findings"]
    findings_dict = {f["id"]: f for f in findings if isinstance(f, dict) and f.get("id")}

    enriched = []
    for q in questions:
        if not isinstance(q, dict) or not q.get("id"):
            continue
        finding = findings_dict.get(q["id"], {})
        enriched.append({
            "id": q["id"],
            "question": q.get("question"),
            "rationale": q.get("rationale"),
            "expected_form": q.get("expected_form"),
            "answer": finding.get("answer"),
            "data_profile": finding.get("data_profile"),
            "method": finding.get("method"),
            "caveats": finding.get("caveats"),
        })

    (workdir / "viz_context.json").write_text(
        json.dumps({"visualizations_to_build": enriched}, indent=2) + "\n", encoding="utf-8")


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
    """Write a minimal findings-listing dist/index.html if the pipeline produced none (<=200 bytes counts as none). Never raises."""
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
