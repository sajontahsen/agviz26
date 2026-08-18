"""Generation pipeline: analysis_builder -> coder, passing a compact manifest forward."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from llm_client import make_llm_client

from .fallback import ensure_fallback_dist
from .runner import BudgetTracker, StageResult, run_stage
from .tools import ToolContext

# Cap generation so the shared per-job budget (~1M cloud) leaves room for evaluation.
GEN_TOKEN_CEILING = 800_000
ANALYSIS_TOKEN_CEILING = 300_000


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

CLOUD_SONNET = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
CLOUD_OPUS = "global.anthropic.claude-opus-4-8"
LOCAL_MODEL = os.environ.get("VIS_ARENA_LOCAL_MODEL", "gpt-5-nano")

_CLOUD_ROLES = {
    "analysis_builder": CLOUD_OPUS,
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

ANALYSIS_BUILDER_SYSTEM = """You are the analysis-builder stage for a web data visualization agent.

Your job is
- to understand the task, inspect the raw data only as much as needed,
- choose feasible analytical views, 
- compute the answers with Python, 
- and emit a compact build contract for the visualization coder.

Inputs:
- WORKDIR/task.md — read fully. It is the task and may include additional metadata.
- WORKDIR/data/ — inspect file names, schemas, headers, and small samples only.

Required output:
- source/analyze.py
- outputs/* plot-ready data files
- analysis_manifest.json at the workdir root

analysis_manifest.json MUST be written programmatically by source/analyze.py.
Do not hand-write final analytical numbers after eyeballing output. The script
is the source of truth for findings, file paths, schemas, samples, and caveats.

Manifest shape:
{
  "task_summary": "concise summary of the task and intended visualization",
  "data_scope": {
    "files_used": ["data/..."],
    "filters": ["..."],
    "time_range": "...",
    "units": "...",
    "known_limitations": ["..."]
  },
  "visualizations": [
    {
      "id": "short_snake_case",
      "title": "insightful chart title",
      "question": "atomic analytical question",
      "answer": "computed answer in plain language",
      "chart_type": "line_chart|bar_chart|stacked_bar|scatter|heatmap|network_summary|table|...",
      "data_file": "outputs/example.csv",
      "data_format": "csv|json",
      "schema": {"field": "type"},
      "sample": [{"field": "value"}],
      "encoding": {"x": "...", "y": "...", "color": null, "tooltip": ["..."]},
      "interaction": {"type": "hover_only|filter|select|tabs|static", "details": "..."},
      "display_notes": "units, aggregations, filters, scope",
      "caveats": ["..."]
    }
  ],
  "narrative": {
    "hook": "...",
    "sections": [
      {"title": "...", "visualization_ids": ["..."], "message": "..."}
    ],
    "payoff": "..."
  }
}

Workflow:
1. Read task.md and list data/.
2. Inspect small data samples/headers/schemas. Do not dump full datasets.
3. Pick 3-6 useful, feasible visualization views that directly answer the task.
4. Write source/analyze.py. It must:
   - load raw data once where practical;
   - create outputs/ if needed;
   - compute each view inside its own try/except block;
   - print concise traceback summaries for failed views;
   - write one plot-ready output file per successful visualization;
   - build analysis_manifest.json from successful visualizations only;
   - include generated schema, sample rows/items, caveats, filters, units, and file paths.
5. Run source/analyze.py with bash.
6. If a view fails twice, simplify or drop that view. Preserve the working views.
7. Finish only after analysis_manifest.json exists and references real output files.

Use pandas/networkx/json/csv as appropriate. Keep stdout concise. Prefer robust,
simple calculations over elaborate fragile ones.

Output limit: Each response is capped at ~8192 tokens. If source/analyze.py is
too large for a single write_file call, write the first chunk with append=false,
then continue with append=true for following chunks. For small edits, use
str_replace instead of rewriting.

Token Economy: Work economically — generation and evaluation share a fixed
competition token budget."""

CODER_SYSTEM = """You are the visualization-coder stage for a web data visualization agent. 

Your job is presentation engineering. The analytical source of truth is
analysis_manifest.json plus the output files it references. Build one cohesive, self-contained, interactive artifact at dist/index.html.

Read:
- analysis_manifest.json at the workdir root.
- task.md — what was asked.

Avoid reading raw data/ unless absolutely necessary to recover from a
manifest defect. Do not re-analyze, recompute, or second-guess findings. Build
the dashboard from the manifest's narrative, visualization specs, answers,
caveats, and referenced outputs.

Build:
- Write source/build.py that reads analysis_manifest.json and every referenced
  outputs/* file, embeds what it needs inline as JSON (use json.dumps() to safely
  inject into <script> tags), and writes dist/index.html. Run it with bash. You do NOT need to read the data files into your own context — build.py reads them.
- Use the manifest's narrative.hook, narrative.sections, visualization titles,
  answers, display_notes, caveats, and narrative.payoff to create a clear story.
- Follow each visualization's chart_type and encoding. Map exact data fields
  from schema/sample/output files; do not invent renamed keys.
- For interactive elements or other exploratory questions, you MUST implement working UI controls (dropdowns, tabs, sliders, etc.) using vanilla HTML/JS/CSS to filter or transform the plotted data dynamically. Do NOT rely purely on Plotly defaults.
- Apply a cohesive, minimalist design system using vanilla CSS. Use modern typography, a cohesive color palette, subtle borders for UI controls, and flexbox/grid for crisp layouts. Do not leave the page with unstyled browser defaults.
- Static charts (like deep statistical cuts) are perfectly fine as long as they implement good practices (e.g., tooltips, clear labels). Strive for a coherent balance between static insights and interactive exploration.
- Rendering libraries (pin these exact versions; load from CDN):
  Plotly.js https://cdn.plot.ly/plotly-2.35.2.min.js
- Never render tens of thousands of nodes raw.
- The page must render from dist/index.html with no dev server.
- Token Economy: Work economically — the job has a shared token budget.
- Output limit: Each response is capped at ~8192 tokens. If a file is too large
  for a single write_file call, write the first chunk with append=false, then
  continue with append=true for each following chunk. For small edits to an
  existing file, use str_replace instead of rewriting.

You are judged on these:
- functionality: interactions (filters, tooltips, selection) actually work.
- visual_craft: right chart types, clear titles/axes/labels, disclosed filters/timeframes/scope, readable color.
- data_fidelity: numbers on screen match the actual data.
- insightfulness: call out trends, exceptions, comparisons — not just raw charts.
- narrative_coherence: a hook -> build -> payoff arc; consistent encodings across panels.

Validation Loop:
When the page is written, you MUST call the verify tool (with no arguments, to serve dist/ locally). It captures a screenshot and returns a health summary.
If verify reports any errors/discrepancy:
1. Use str_replace or rewrite to fix the bug in build.py/HTML.
2. Re-run build.py via bash to generate the new dist/index.html.
3. Call verify again.

Iterate until verify passes with 0 errors and working charts. Then call finish with a short summary of the panels."""

CODER_FALLBACK_SYSTEM = """You are a web data visualization agent.

Build a useful, self-contained dist/index.html directly from the task and data.
Keep the result simple, honest.

Read:
- task.md
- data/ file listing
- small headers/samples only as needed

Build:
- Write source/build.py and run it to produce dist/index.html.
- Prefer a simple but honest visualization over an ambitious fragile one.
- If a clear tabular file exists, compute 1-3 lightweight summaries in build.py
  or a small helper inside build.py: row counts, categorical top values,
  numeric/date distributions, or a compact table of representative records.
- If the data is graph/nested/unknown, show a concise data overview, task
  summary, entity/type counts when practical, and a small readable table/list.
- Do not fabricate analytical claims. Label the output as a limited task-aware
  overview when deeper computation was not possible.
- Apply clean CSS, useful headings, caveats, and at least one simple Plotly chart
  when the data shape supports it.
- Rendering libraries (pin this exact version; load from CDN):
  Plotly.js https://cdn.plot.ly/plotly-2.35.2.min.js
- The page must render from dist/index.html with no dev server.
- Token Economy: Work economically — this is a recovery path.
- Output limit: Each response is capped at ~8192 tokens. If a file is too large
  for a single write_file call, write the first chunk with append=false, then
  continue with append=true for each following chunk.

Validation Loop:
When the page is written, you MUST call the verify tool. If verify reports
errors, fix source/build.py or the generated HTML, rerun build.py, and verify
again. Then call finish with a short summary."""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def orchestrate(workdir: Path) -> dict[str, Any]:
    workdir = Path(workdir).resolve()
    ctx = ToolContext(workdir)
    client = make_llm_client("generation")
    wd = str(workdir)

    budget = BudgetTracker(ceiling=GEN_TOKEN_CEILING)
    results: list[StageResult] = []

    analysis_budget = BudgetTracker(ceiling=min(ANALYSIS_TOKEN_CEILING, budget.remaining()))
    analysis = _safe_run(
        ctx=ctx,
        client=client,
        name="analysis_builder",
        system_prompt=ANALYSIS_BUILDER_SYSTEM,
        user_prompt=(
            f"WORKDIR={wd}\n"
            "Read task.md, inspect data/ as needed, write and run source/analyze.py, "
            "then emit analysis_manifest.json and outputs/*."
        ),
        tool_names=["read_file", "write_file", "str_replace", "bash", "search"],
        model=pick_model("analysis_builder"),
        max_steps=60,
        prune_keep=12,
        budget=analysis_budget,
        low_water=100_000,
    )
    budget.spent += analysis.usage.get("total_tokens", 0)
    results.append(analysis)

    manifest_ok, manifest_errors = _validate_analysis_manifest(workdir)
    if manifest_errors:
        for err in manifest_errors:
            print(f"[pipeline] analysis_manifest validation: {err}", file=sys.stderr)

    if manifest_ok:
        coder_system = CODER_SYSTEM
        coder_user = (
            f"WORKDIR={wd}\n"
            "Read analysis_manifest.json, then write and run source/build.py to produce "
            "dist/index.html. Verify it renders before finishing."
        )
    else:
        print("[pipeline] running fallback coder because analysis_manifest.json is not build-ready", file=sys.stderr)
        coder_system = CODER_FALLBACK_SYSTEM
        coder_user = (
            f"WORKDIR={wd}\n"
            "Read task.md and inspect data/ lightly, then write and run source/build.py "
            "to produce a basic but valid dist/index.html. Verify it renders before finishing."
        )

    results.append(_safe_run(
        ctx=ctx,
        client=client,
        name="coder",
        system_prompt=coder_system,
        user_prompt=coder_user,
        tool_names=["read_file", "write_file", "str_replace", "bash", "search", "verify"],
        model=pick_model("coder"),
        max_steps=70,
        prune_keep=12,
        budget=budget,
    ))

    # Guarantee a renderable artifact so the job always yields a scorable
    # preview to inspect, rather than a hard "dist/index.html was not created".
    fallback_info = ensure_fallback_dist(workdir, findings_available=manifest_ok)

    # Move intermediate JSON files to source/state/ for clean root directory.
    _cleanup_state_files(workdir)

    return _summarize(
        workdir,
        results,
        manifest_ok=manifest_ok,
        manifest_errors=manifest_errors,
        fallback_info=fallback_info,
    )


def _safe_run(*, ctx: ToolContext, client: Any, name: str, **kwargs: Any) -> StageResult:
    """Run one stage, converting any crash into an unfinished result + stderr trace."""
    try:
        return run_stage(name=name, ctx=ctx, client=client, **kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately total: one stage must not abort the run
        print(f"[stage:{name}] CRASHED: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        return StageResult(name=name, result={"error": str(exc)}, finished=False)


def _summarize(
    workdir: Path,
    results: list[StageResult],
    *,
    manifest_ok: bool,
    manifest_errors: list[str],
    fallback_info: dict[str, Any],
) -> dict[str, Any]:
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
            "result": r.result,
        })

    notes_data = {
        "pipeline": {
            "total_tokens": total,
            "ceiling": GEN_TOKEN_CEILING,
            "dist_ready": dist_ready,
            "analysis_manifest_valid": manifest_ok,
            "analysis_manifest_errors": manifest_errors,
            "fallback": fallback_info,
        },
        "stages": stages_meta,
    }

    notes_str = json.dumps(notes_data, indent=2)
    print(f"[pipeline]\n{notes_str}", file=sys.stderr)
    return {"notes": notes_data}


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

def _validate_analysis_manifest(workdir: Path) -> tuple[bool, list[str]]:
    """Minimal build-readiness check for analysis_manifest.json."""
    manifest_path = workdir / "analysis_manifest.json"
    errors: list[str] = []
    if not manifest_path.exists():
        return False, ["analysis_manifest.json missing"]

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return False, [f"analysis_manifest.json is invalid JSON: {exc}"]
    except OSError as exc:
        return False, [f"analysis_manifest.json could not be read: {exc}"]

    visualizations = data.get("visualizations")
    if not isinstance(visualizations, list) or not visualizations:
        errors.append("visualizations must be a non-empty list")
        return False, errors

    workdir_resolved = workdir.resolve()
    for idx, viz in enumerate(visualizations):
        if not isinstance(viz, dict):
            errors.append(f"visualizations[{idx}] is not an object")
            continue
        data_file = viz.get("data_file")
        if not data_file:
            continue
        path = Path(str(data_file))
        resolved = (path if path.is_absolute() else (workdir_resolved / path)).resolve()
        try:
            resolved.relative_to(workdir_resolved)
        except ValueError:
            errors.append(f"visualizations[{idx}].data_file points outside workdir: {data_file}")
            continue
        if not resolved.exists():
            errors.append(f"visualizations[{idx}].data_file missing: {data_file}")

    return not errors, errors


def _cleanup_state_files(workdir: Path) -> None:
    import shutil

    state_dir = workdir / "source" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    files_to_move = [
        "analysis_manifest.json",
        # Clean up legacy leftovers if present from interrupted/local runs.
        "profile.json",
        "questions.json",
        "findings.json",
        "viz_context.json",
        "storyboard.json",
    ]
    for f in workdir.glob("findings_*.json"):
        files_to_move.append(f.name)
    for f in workdir.glob("messages_*.json"):
        files_to_move.append(f.name)

    for fname in files_to_move:
        src = workdir / fname
        if src.exists():
            try:
                shutil.move(str(src), str(state_dir / fname))
            except OSError:
                pass
