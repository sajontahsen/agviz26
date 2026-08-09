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
  - expected_form: one of scalar | series | table | ranking | breakdown | interactive_dashboard
  - supports: which task requirement or narrative beat it serves

A single "question" (e.g., expected_form: interactive_dashboard) can encompass multiple charts driven by interactive UI controls (like tabs or dropdowns). However, do NOT pack multiple distinct queries or dimensions into every single question (e.g., asking for volume AND citations AND authors in one question). Instead, spread them out: 3-4 questions should be simple, highly targeted data aggregations (e.g., just "volume over time" or just "top 10 authors"), while 1-2 can be more multifaceted to provide deep insights. This ensures downstream agents can reliably compute the data without crashing. Balance deep statistical insights with broad token-efficient exploration, envision the final artifact as a story dashboard. Ensure all data views remain meaningful and purposeful. 
Organize the questions with this strict mechanical reality in mind: later, one agent has to compute the pandas logic then calculate answers for these questions, and another agent has to code the UI dashboard for them. Both operate under strict token budgets. If you create queries that are too complex, the downstream agents will crash and the question will remain entirely unanswered. Be safe and gentle. Prioritize the core narrative. It is better to have clean, working charts than to crash trying to cover every secondary metric. 
Then call finish."""

ANALYST_SYSTEM = """You are a data ANALYST agent. Compute the correct, verified answer for a specific target question directly from the raw dataset. These numbers become the ground truth the visualization displays, so correctness is paramount.

Inputs: task.md, profile.json, and the TARGET QUESTION (provided in your prompt). Raw data is in data/.

Workflow:
1. Read profile.json and understand your TARGET QUESTION.
2. Write source/analyze_{id}.py that loads data/ and computes the answer for your assigned question id. Run it with bash; iterate until clean.
3. Emit findings_{id}.json (at the workdir root): {"findings": [ {id, answer, data_profile, method, caveats} ]} where
   - id: Must exactly match your assigned id
   - answer: concise plain-language answer
   - data_profile: A schema definition containing:
        - "filepath": Local path where analyze_{id}.py saved the dataset (e.g., "outputs/{id}.csv" or "outputs/{id}.json"). Choose the best format for the data shape.
        - "format": "csv" or "json"
        - "schema": For tabular data, map columns to types. For graphs/nested data, describe the structure.
        - "sample": A minimal snippet (e.g., 2 rows, or 1 node/edge) showing exact structure.
   - method: one line on how it was computed
   - caveats: any data limitations (optional)
4. Call finish.

CRITICAL INSTRUCTION: Your analyze_{id}.py script MUST save the calculated, plot-ready data to a file in the `outputs/` directory. NEVER inline full data arrays into findings_{id}.json. NEVER print raw dataframes or large arrays to standard output. Use `.head(5)` or `.info()` if you must inspect data to preserve the token context window. The `data_profile` must be generated programmatically by analyze_{id}.py to guarantee accuracy. Do not copy the original question or rationale into findings_{id}.json; the system will merge those later."""

STORYBOARD_SYSTEM = """You are a STORYBOARD & LAYOUT agent. You design the structural flow and narrative of the final dashboard.

Inputs: task.md and viz_context.json (read them).

Workflow:
1. Read the inputs to understand what was asked and what findings were computed.
2. Figure out the most coherent way to arrange these findings into a unified dashboard.
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
- viz_context.json — the visualizations to build, under "visualizations_to_build". Each item has: id, question, rationale, expected_form, answer, method, caveats, and a `data_profile` = {filepath, format, schema, sample}. The `data_profile` tells you where the plot-ready data lives and its exact structure. Use the exact fields shown in `schema`/`sample` — never rename or invent keys (mismatched keys silently break charts).
- storyboard.json — the structural layout and narrative text (hook, layout, payoff).

Build:
- Write source/build.py that reads each finding's data from its data_profile.filepath (relative to the workdir), embeds what it needs inline as JSON (use `json.dumps()` to safely inject into `<script>` tags), and writes dist/index.html. Run it with bash. You do NOT need to read the data files into your own context — build.py reads them.
- Strictly follow the structural flow defined in the `layout` array of storyboard.json. Weave the `hook`, `narrative_build`, and `payoff` text directly into the HTML to create a coherent story. Render the specified finding IDs in the suggested `layout_hint` styles.
- One panel per relevant finding, chart type matched to its expected_form / schema. Display the computed numbers as-is. Write defensive JavaScript to handle potential nulls or missing keys smoothly.
- For `expected_form: interactive_dashboard` or other exploratory questions, you MUST implement working UI controls (dropdowns, tabs, sliders, etc.) using vanilla HTML/JS/CSS to filter or transform the plotted data dynamically. Do NOT rely purely on Plotly defaults.
- Apply a premium, minimalist design system using vanilla CSS. Use modern typography, a cohesive color palette, subtle borders for UI controls, and flexbox/grid for crisp layouts. Do not leave the page with unstyled browser defaults.
- Static charts (like deep statistical cuts) are perfectly fine as long as they implement good practices (e.g., tooltips, clear labels). Strive for a coherent balance between static insights and interactive exploration.
- Rendering libraries (pin these exact versions; load from CDN):
  Plotly.js https://cdn.plot.ly/plotly-2.35.2.min.js
  Cytoscape.js https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js
Plotly for statistical/comparative charts, Cytoscape only for network/graph findings. Never render tens of thousands of nodes raw.
- The page must render from dist/index.html with no dev server;
- Token Economy: Work economically — the job has a shared token budget, and long transcripts spend it fast.

You are judged on these:
- functionality: interactions (filters, tooltips, selection) actually work.
- data_fidelity: numbers on screen match the actual data.
- visual_craft: right chart types, clear titles/axes/labels, disclosed filters/timeframes/scope, readable color.
- insightfulness: call out trends, exceptions, comparisons — not just raw charts.
- narrative_coherence: a hook -> build -> payoff arc; consistent encodings across panels.

Validation Loop:
When the page is written, you MUST call the `verify` tool (with no arguments, to serve dist/ locally). It captures a screenshot and returns a health summary.
If `verify` reports ANY `console_errors` or `page_errors`:
1. CRITICAL: Use `str_replace` or `bash` (sed) to fix the bug. DO NOT use `write_file` to rewrite the entire script for minor edits. Rewriting entire files wastes output tokens and will cause you to fail.
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
            user_prompt="", # Injected dynamically
            tool_names=["read_file", "write_file", "str_replace", "bash", "search"],
            model=pick_model("analyst"), max_steps=20, max_history_tokens=20_000, prune_keep=6,
        ),
        dict(
            name="storyboard", system_prompt=STORYBOARD_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead task.md and viz_context.json, then write storyboard.json.",
            tool_names=["read_file", "write_file"],
            model=pick_model("storyboard"), max_steps=10,
        ),
        dict(
            name="coder", system_prompt=CODER_SYSTEM,
            user_prompt=f"WORKDIR={wd}\nRead task.md and viz_context.json, then write and run source/build.py to produce dist/index.html. Verify it renders before finishing.",
            tool_names=["read_file", "write_file", "str_replace", "bash", "search", "verify"],
            model=pick_model("coder"), max_steps=40, max_history_tokens=20_000, prune_keep=6,
        ),
    ]

    budget = BudgetTracker(ceiling=GEN_TOKEN_CEILING)
    results: list[StageResult] = []
    for spec in specs:
        if spec["name"] == "analyst":
            (workdir / "findings.json").write_text(json.dumps({"findings": []}))
            agg_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            agg_steps = 0
            agg_finished = True
            agg_tool_counts: dict[str, int] = {}
            
            q_file = workdir / "questions.json"
            questions = []
            if q_file.exists():
                try:
                    questions = json.loads(q_file.read_text()).get("questions", [])
                except Exception:
                    pass
            
            analyst_spent = 0
            for q in questions:
                if analyst_spent >= 200_000:
                    break
                    
                qid = q.get("id", "unknown")
                q_prompt = f"WORKDIR={wd}\nTARGET QUESTION:\n{json.dumps(q, indent=2)}\nRead task.md and profile.json. Write and run source/analyze_{qid}.py to emit findings_{qid}.json."
                
                sub_spec = dict(spec)
                sub_spec["name"] = f"analyst_{qid}"
                sub_spec["user_prompt"] = q_prompt
                sub_spec["prune_keep"] = 2
                sub_spec["max_history_tokens"] = 6_000
                sub_spec["low_water"] = 5_000
                
                q_ceiling = min(30_000, 200_000 - analyst_spent, budget.remaining())
                if q_ceiling <= 0:
                    break
                q_budget = BudgetTracker(ceiling=q_ceiling)
                
                r = _safe_run(ctx=ctx, client=client, budget=q_budget, **sub_spec)
                
                spent_here = r.usage.get("total_tokens", 0)
                analyst_spent += spent_here
                budget.spent += spent_here
                
                for k in agg_usage:
                    agg_usage[k] += r.usage.get(k, 0)
                agg_steps += r.steps
                if not r.finished:
                    agg_finished = False
                for k, v in r.tool_counts.items():
                    agg_tool_counts[k] = agg_tool_counts.get(k, 0) + v
                    
                if r.finished:
                    f_file = workdir / f"findings_{qid}.json"
                    if f_file.exists():
                        try:
                            f_data = json.loads(f_file.read_text())
                            main_data = json.loads((workdir / "findings.json").read_text())
                            main_data["findings"].extend(f_data.get("findings", []))
                            (workdir / "findings.json").write_text(json.dumps(main_data, indent=2))
                        except Exception:
                            pass
                        
            results.append(StageResult(
                name="analyst", result={}, usage=agg_usage, steps=agg_steps, 
                finished=agg_finished, tool_counts=agg_tool_counts
            ))
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
    """Build a ``notes`` string that carries all telemetry into generation.json (the only field agent.py reads)."""
    total = sum(r.usage["total_tokens"] for r in results)
    dist_ready = (workdir / "dist" / "index.html").exists()

    lines = [f"pipeline: tokens={total}/{GEN_TOKEN_CEILING} dist_ready={dist_ready}"]
    for r in results:
        tc = r.tool_counts
        tc_str = " ".join(f"{k}={v}" for k, v in sorted(tc.items())) if tc else "-"
        status = "ok" if r.finished else "FAIL"
        lines.append(
            f"  {r.name:<11} {status:<4} steps={r.steps:<3} "
            f"in={r.usage.get('input_tokens',0)} out={r.usage.get('output_tokens',0)} "
            f"total={r.usage['total_tokens']} tools=[{tc_str}]"
        )

    notes = "\n".join(lines)
    print(f"[pipeline]\n{notes}", file=sys.stderr)
    return {"notes": notes}


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
        if q["id"] not in findings_dict:
            continue
        finding = findings_dict[q["id"]]
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
