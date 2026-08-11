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
GEN_TOKEN_CEILING = 800_000


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
    "storyboard": CLOUD_OPUS,
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
1. read_file task.md, every documentation file mentioned, and list data/ to see the files and their formats. 
2. Write source/profile.py that emits profile.json (at the workdir root). Run it with bash. Iterate until it runs cleanly.
3. Call finish with a one-line summary.

profile.json exists so the ANALYST agent can write correct code on its first attempt
WITHOUT opening the raw data. Judge every field by that test. Whatever the shape
— tabular, graph, nested JSON, geospatial, time series, or something you have not
seen before — it MUST include:

- load: the literal Python expression that loads the file(s).
- accessors: the exact top-level keys / column names / field names / datatypes, copied
  verbatim including capitalisation and spaces.
- samples: one RAW record per record type, straight from the file, long string
  values truncated to ~80 chars. For a graph: one node per node type and one edge
  per edge type, showing real field names and how endpoints are referenced.
  Emit each sample as a JSON OBJECT, never as a pre-serialized string. Every key
  the record has must be present — a sample missing keys is worse than no sample,
  because the analyst will code against what it sees.
  Shorten samples by truncating VALUES, never the structure:
  - string value longer than 80 chars -> first 80 chars + "…"
  - array longer than 3 items         -> first 3 items + "…(N total)"
  - nested object deeper than 2 levels -> keep keys, replace the value with "{…}"
  Never slice the serialized JSON text, and never drop trailing keys to save space.
  Preserve the JSON type of every value. Do not stringify numbers, booleans, or
  nulls. The analyst needs to see that `"id": 17255` is an integer and not "17255",
  and that `"notable": true` is a boolean — guessing these wrong is a debug loop.
- conventions: anything ambiguous a reader would otherwise guess wrong — edge
  direction semantics, units, date formats, null encodings, id types.
- entities: for every specific person/place/thing named in task.md, its resolved
  id and type.
- If the ANLAYST would need to read additional metadata/documentation, note that in a relevant section

Prefer pandas/networkx. Keep example lists short."""

PLANNER_SYSTEM = """You are a visualization PLANNING agent. You do NOT write code and you do NOT compute answers. You decompose the task into the analytical questions whose answers will drive an insightful, well-structured visualization.

Inputs (read them): task.md (what was asked) and profile.json (the data's structure).

INSTRUCTIONS:
- Break the task down yourself: Design 5-6 questions that cover every explicit ask in task.md. Focus on the core narrative/insight the artifact should deliver, rather than trying to blindly cover every secondary sub-question in task.md.
- Balance deep statistical insights with broad token-efficient exploration. 3-4 questions should be simple, targeted data aggregations that explore the data and are fairly easy to compute (e.g., "volume over time" or "top 10 authors"), while 1-2 can be more multifaceted to provide deep insights.
- Envision the final artifact as a story dashboard driven by interactive UI controls (like tabs or dropdowns). 
- Ensure all data views remain meaningful and purposeful. Then call finish.

Output:
Write questions.json (at the workdir root): an object {"questions": [ ... ]}. Each question:
  - id: short snake_case id
  - question: the analytical question in plain language
  - rationale: why it matters for the task and the story
  - computation_hint: concretely how to compute it from the data (which columns/edge-types/aggregation), grounded in profile.json field names
  - expected_form: one of scalar | series | table | ranking | breakdown | interactive_dashboard
  - supports: which task requirement or narrative beat it serves"""

ANALYST_SYSTEM = """You are a data ANALYST agent. Compute the correct, verified answers for the analytical questions directly from the raw dataset. These numbers become the ground truth the visualization displays.

Inputs: task.md, profile.json, and questions.json. Raw data is in data/.

Workflow:
1. Read profile.json (gives you the data schema and sample values) and questions.json.
2. Write a single python script (source/analyze.py) that loads data/ ONCE, and then iterates through all the questions.
3. For EACH question it processes, the script must compute the answer and independently emit a findings_{id}.json file (at the workdir root) with this exact structure: {"findings": [ {id, answer, data_profile, method, caveats} ]} where:
   - id: Must exactly match the question id from questions.json
   - answer: concise plain-language answer
   - data_profile: A schema definition containing:
        - "filepath": Local path where analyze.py saved the plot-ready dataset for this question (e.g., "outputs/{id}.csv"). Choose the best format for the data shape.
        - "format": "csv" or "json"
        - "schema": For tabular data, map columns to types. For graphs/nested data, describe the structure.
        - "sample": A minimal snippet (e.g., 2 rows, or 1 node/edge) showing exact structure.
   - method: one line on how it was computed
   - caveats: any data limitations (optional)
4. Run analyze.py with bash; iterate until all feasible questions are computed. Then call finish.

CRITICAL INSTRUCTIONS:
- File-Level Atomicity: Wrap the computation for EACH question inside its own `try/except` block and print the error traceback if it fails! This ensures that if question 2 crashes, your script will simply log it and continue on to successfully compute questions 3, 4, and 5 in the very first run. The `findings_{id}.json` files for the successful questions are safely saved to disk! When you retry your script, you can either skip the questions that already have findings files, or just fix the logic for question 2 and re-run. This gives you partial fault tolerance.
- Your analyze.py script MUST save the calculated, plot-ready data to files in the `outputs/` directory. NEVER inline full data arrays into findings_{id}.json. NEVER print raw dataframes or large arrays to standard output. Use `.head(5)` or `.info()` if you must inspect data to preserve the token context window. The `data_profile` must be generated programmatically to guarantee accuracy. Do not copy the original question or rationale into findings_{id}.json.
- Here's your implementation ladder for getting a script to work:
Run 1: your full intended analysis.
Run 2 (if errored): try to fix the error, without unnecessary redesign.
Run 3 (if errored): rewrite the errored question the simplest way that works to answer one core point

There is no run 4 for a single bug. If you are stuck on a specific question, comment it out, ensure the other questions save their findings successfully, and call finish.
- Token Economy: Work economically — the job has a shared token budget."""

STORYBOARD_SYSTEM = """You are a STORYBOARD & LAYOUT agent. You design the structural flow and narrative of the final dashboard.

Inputs: task.md and viz_context.json (read them).

Workflow:
1. Read the inputs to understand what was asked and what findings were computed.
2. Figure out the most coherent way to arrange these findings into a unified dashboard. Strictly follow data visualization best practices. 
3. Write storyboard.json (at the workdir root): an object {"storyboard": { "hook": "...", "layout": [ ... ], "payoff": "..." }}.
   - hook: The overarching introductory narrative setting up the dashboard.
   - layout: An ordered array mapping the flow of the page. Each item dictates a section:
     - ids: Array of finding IDs to include in this section.
     - layout_hint: How to render it (e.g., "full-width interactive dashboard with tabs", "side-by-side static comparison", "single focused chart").
     - narrative_build: The transitional text explaining this section and guiding the user.
   - payoff: The concluding actionable takeaways or insights.
4. Call finish.

Keep the narrative tight, professional, and directly tied to the actual computed findings."""

CODER_SYSTEM = """You are a web data-visualization engineer. Build ONE cohesive, self-contained, interactive artifact at dist/index.html that tells a clear story and lets a reviewer inspect the answers to the task.

Read these (workdir root):
- task.md — what was asked.
- storyboard.json — the structural layout and narrative text (hook, layout, payoff).
- viz_context.json — the visualizations to build, under "visualizations_to_build". Each item has: id, question, rationale, expected_form, answer, method, caveats, and a `data_profile` = {filepath, format, schema, sample}. The `data_profile` tells you where the plot-ready data lives and its exact structure. Use the exact fields shown in `schema`/`sample` — never rename or invent keys (mismatched keys silently break charts).

Build:
- Write source/build.py that reads each finding's data from its data_profile.filepath (relative to the workdir), embeds what it needs inline as JSON (use `json.dumps()` to safely inject into `<script>` tags), and writes dist/index.html. Run it with bash. You do NOT need to read the data files into your own context — build.py reads them.
- Strictly follow the structural flow defined in the `layout` array of storyboard.json. Weave the `hook`, `narrative_build`, and `payoff` text directly into the HTML to create a coherent story. Render the specified finding IDs in the suggested `layout_hint` styles.
- One panel per relevant finding, chart type matched to its expected_form / schema. Display the computed numbers as-is. Write defensive JavaScript to handle potential nulls or missing keys smoothly.
- For `expected_form: interactive_dashboard` or other exploratory questions, you MUST implement working UI controls (dropdowns, tabs, sliders, etc.) using vanilla HTML/JS/CSS to filter or transform the plotted data dynamically. Do NOT rely purely on Plotly defaults.
- Apply a premium, minimalist design system using vanilla CSS. Use modern typography, a cohesive color palette, subtle borders for UI controls, and flexbox/grid for crisp layouts. Do not leave the page with unstyled browser defaults.
- Static charts (like deep statistical cuts) are perfectly fine as long as they implement good practices (e.g., tooltips, clear labels). Strive for a coherent balance between static insights and interactive exploration.
- Follow data visualization best practices. For Plotly layout: always place legends below the chart area (`legend: {orientation:'h', y:-0.15}`), set adequate `margin: {t, b}` so titles and legends never overlap chart content.
- For every chart, attach a `plotly_hover` listener that writes the hovered data point's details into a visible DOM element (e.g. a `<div class="detail-panel">`). This makes hover data accessible to automated evaluators that scan page text — do not rely solely on Plotly's native SVG tooltips.
- Rendering libraries (pin these exact versions; load from CDN):
  Plotly.js https://cdn.plot.ly/plotly-2.35.2.min.js
- Never render tens of thousands of nodes raw.
- The page must render from dist/index.html with no dev server;
- Token Economy: Work economically — the job has a shared token budget. 

You are judged on these:
- functionality: interactions (filters, tooltips, selection) actually work.
- visual_craft: right chart types, clear titles/axes/labels, disclosed filters/timeframes/scope, readable color.

- data_fidelity: numbers on screen match the actual data.
- insightfulness: call out trends, exceptions, comparisons — not just raw charts.
- narrative_coherence: a hook -> build -> payoff arc; consistent encodings across panels.

Validation Loop:
When the page is written, you MUST call the `verify` tool (with no arguments, to serve dist/ locally). It captures a screenshot and returns a health summary.
If `verify` reports any errors/discrepency:
1. Use `str_replace` or rewrite to fix the bug in build.py/HTML.
2. Re-run `build.py` via bash to generate the new dist/index.html.
3. Call `verify` again.

Iterate until verify passes with 0 errors and working charts. Then call finish with a short summary of the panels."""

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
            model=pick_model("profile"), max_steps=15, 
        ),
        dict(
            name="planner", system_prompt=PLANNER_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead task.md and profile.json, then write questions.json.",
            tool_names=["read_file", "write_file"],
            model=pick_model("planner"), max_steps=15,
        ),
        dict(
            name="analyst", system_prompt=ANALYST_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead task.md, profile.json, and questions.json. Write and run a single source/analyze.py script to process all questions. For each question, save findings_{{id}}.json to the workdir. Iterate until all questions are answered or you hit a blocker you cannot pass.",
            tool_names=["read_file", "write_file", "str_replace", "bash", "search"],
            model=pick_model("analyst"), max_steps=50, max_history_tokens=20_000, prune_keep=4, low_water=100_000
        ),
        dict(
            name="storyboard", system_prompt=STORYBOARD_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead task.md and viz_context.json, then write storyboard.json.",
            tool_names=["read_file", "write_file"],
            model=pick_model("storyboard"), max_steps=15,
        ),
        dict(
            name="coder", system_prompt=CODER_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead task.md, storyboard.json and viz_context.json, then write and run source/build.py to produce dist/index.html. Verify it renders before finishing.",
            tool_names=["read_file", "write_file", "str_replace", "bash", "search", "verify"],
            model=pick_model("coder"), max_steps=80, max_history_tokens=20_000, prune_keep=4,
        ),
    ]

    budget = BudgetTracker(ceiling=GEN_TOKEN_CEILING)
    results: list[StageResult] = []
    for spec in specs:
        if spec["name"] == "analyst":
            a_ceiling = min(300_000, budget.remaining())
            a_budget = BudgetTracker(ceiling=a_ceiling)
            r = _safe_run(ctx=ctx, client=client, budget=a_budget, **spec)
            budget.spent += r.usage.get("total_tokens", 0)
            results.append(r)
            _safe_merge_context(workdir)
        else:
            results.append(_safe_run(ctx=ctx, client=client, budget=budget, **spec))

    # Guarantee a renderable artifact so the job always yields a scorable
    # preview to inspect, rather than a hard "dist/index.html was not created".
    _ensure_fallback_dist(workdir)

    # Move intermediate JSON files to source/state/ for clean root directory
    _cleanup_state_files(workdir)

    return _summarize(workdir, results)


def _safe_run(*, ctx: ToolContext, client: Any, name: str, **kwargs: Any) -> StageResult:
    """Run one stage, converting any crash into an unfinished result + stderr trace."""
    try:
        return run_stage(name=name, ctx=ctx, client=client, **kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately total: one stage must not abort the run
        print(f"[stage:{name}] CRASHED: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        return StageResult(name=name, result={"error": str(exc)}, finished=False)


def _summarize(workdir: Path, results: list[StageResult]) -> dict[str, Any]:
    """Build a rich telemetry dict for generation.json (the framework reads the notes key)."""
    total = sum(r.usage.get("total_tokens", 0) for r in results)
    dist_ready = (workdir / "dist" / "index.html").exists()

    stages_meta = []
    for r in results:
        stages_meta.append({
            "stage": r.name,
            "finished": r.finished,
            "steps": r.steps,
            "usage": r.usage,
            "tool_counts": r.tool_counts,
        })
        
    notes_data = {
        "pipeline": {
            "total_tokens": total,
            "ceiling": GEN_TOKEN_CEILING,
            "dist_ready": dist_ready,
        },
        "stages": stages_meta
    }
    
    notes_str = json.dumps(notes_data, indent=2)
    print(f"[pipeline]\n{notes_str}", file=sys.stderr)
    return {"notes": notes_data}



# ---------------------------------------------------------------------------
# Deterministic merge (post-analyst): join questions x findings -> viz_context.json.
# ---------------------------------------------------------------------------

def _safe_merge_context(workdir: Path) -> None:
    try:
        _merge_context(workdir)
    except Exception as exc:  # noqa: BLE001 - glue must not abort the run
        print(f"[pipeline] merge_context skipped: {exc}", file=sys.stderr)


def _merge_context(workdir: Path) -> None:
    q_file = workdir / "questions.json"
    if not q_file.exists():
        return
        
    questions = json.loads(q_file.read_text(encoding="utf-8")).get("questions", [])

    enriched = []
    for q in questions:
        if not isinstance(q, dict) or not q.get("id"):
            continue
            
        qid = q["id"]
        f_file = workdir / f"findings_{qid}.json"
        
        if not f_file.exists():
            continue
            
        try:
            finding_data = json.loads(f_file.read_text(encoding="utf-8"))
            findings_list = finding_data.get("findings", [])
            if not findings_list:
                continue
            finding = findings_list[0]
        except Exception:
            continue
            
        enriched.append({
            "id": qid,
            "question": q.get("question"),
            "rationale": q.get("rationale"),
            # "expected_form": q.get("expected_form"),
            "answer": finding.get("answer"),
            "data_profile": finding.get("data_profile"),
            "method": finding.get("method"),
            "caveats": finding.get("caveats"),
        })

    (workdir / "viz_context.json").write_text(
        json.dumps({"visualization_context": enriched}, indent=2) + "\n", encoding="utf-8")


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


def _cleanup_state_files(workdir: Path) -> None:
    import shutil
    state_dir = workdir / "source" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_move = ["profile.json", "questions.json", "findings.json", "viz_context.json", "storyboard.json"]
    for f in workdir.glob("findings_*.json"):
        files_to_move.append(f.name)
        
    for fname in files_to_move:
        src = workdir / fname
        if src.exists():
            try:
                shutil.move(str(src), str(state_dir / fname))
            except OSError:
                pass
