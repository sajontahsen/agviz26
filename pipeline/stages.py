"""Generation pipeline: analysis_builder -> storyboard -> coder, passing compact artifacts forward."""
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

ANALYSIS_BUILDER_SYSTEM = """You are the analysis-builder stage for a web data visualization agent.

Your job is
- to understand the task, inspect the raw data only as much as needed,
- choose feasible analytical views, 
- compute the answers with Python, 
- and emit a compact analysis skeleton for the storyboard stage.


Workflow:
1. read_file task.md, every documentation file mentioned, and list data/ to see the files and their formats. .
2. Inspect small data samples/headers/schemas. Do not dump full datasets.
3. Pick useful, feasible analytical views that directly answer the task.
- Prioritize questions that answer the core tasks. Drop secondary sub-tasks to keep scope manageable. If open-ended, extrapolate strictly 5-6 questions.
4. Write source/analyze.py. It must:
   - load raw data once where practical;
   - create outputs/ if needed;
   - compute each view inside its own try/except block;
   - print concise traceback summaries for failed views;
   - write one plot-ready output file per successful view;
   - build analysis_skeleton.json from successful views only;
   - include generated schema, sample rows/items, caveats, assumptions, and file paths.
5. Run source/analyze.py with bash.
6. If a view fails twice, simplify or drop that view. Preserve the working views.
7. Finish only after analysis_skeleton.json exists and references real output files.

Skeleton shape:
{
  "task_summary": "concise summary of what was asked",
  "data_context": {
    "files_used": ["data/..."],
    "record_shapes": {"optional concise schema/profile notes": "..."},
    "assumptions": ["only include filters, units, date ranges, or joins that actually apply"],
    "limitations": ["..."]
  },
  "views": [
    {
      "id": "short_snake_case",
      "question": "atomic analytical question",
      "answer_data": {
        "file": "outputs/example.csv",
        "format": "csv|json",
        "schema": {"field": "type"} - For tabular data, map columns to types. For nested/graph data, describe the structure.,
        "sample": [{"field": "value"} - A minimal snippet (e.g., 2 rows, or 1 node/edge) showing the EXACT structure.],
        "grain": "one row/item means ..."
      },
      "method": "one concise sentence describing the computation",
      "rationale": ["why it matters for the task and the story"],
      "caveats": ["..."]
    }
  ]
}

Use pandas/networkx/json/csv as appropriate. Keep stdout concise. Prefer robust,
simple calculations over elaborate fragile ones.

Required output:
- source/analyze.py
- outputs/* plot-ready data files
- analysis_skeleton.json at the workdir root

Output limit: Each response is capped at ~8192 tokens. If source/analyze.py is
too large for a single write_file call, write the first chunk with append=false,
then continue with append=true for following chunks. For small edits, use
str_replace instead of rewriting.

analysis_skeleton.json MUST be written programmatically by source/analyze.py.
Do not hand-write final analytical numbers after eyeballing output. The script
is the source of truth for findings, file paths, schemas, samples, and caveats.

Token Economy: Work economically — generation and evaluation share a fixed
competition token budget."""

STORYBOARD_SYSTEM = """You are the storyboard stage for a web data visualization agent.

Your job is to turn computed analysis into an authoritative, compact
visualization manifest for the coder. You do NOT read raw data, recompute
metrics, or invent new analytical results.

Inputs:
- task.md — what was asked.
- analysis_skeleton.json — computed views, answers, output files, schemas,
  samples, methods, and caveats.

Required output:
- analysis_manifest.json at the workdir root.

The final manifest is the coder's source of truth. It should enforce
insightfulness, narrative coherence, visual craft, while
preserving data fidelity from the analysis skeleton.

Manifest shape:
{
  "task_summary": "...",
  "data_context": {
    "files_used": ["data/..."],
    "assumptions": ["only facts that apply"],
    "limitations": ["..."]
  },
  "views": [
    {
      "id": "same id from analysis_skeleton",
      "title": "reader-facing chart/panel title",
      "question": "...",
      "answer": "...",
      "data": {
        "file": "outputs/example.csv",
        "format": "csv|json",
        "schema": {"field": "type"},
        "sample": [{"field": "value"}],
        "grain": "one row/item means ..."
      },
      "analysis": {
        "method": "...",
        "supports": ["..."],
        "caveats": ["..."]
      },
      "presentation": {
        "role": "setup|evidence|comparison|exception|payoff|reference",
        "insight": "specific trend, contrast, exception, or implication this panel should make clear",
        "recommended_view": "line chart|bar chart|stacked bar|scatterplot|network summary|table|...",
        "plot_guidance": "high-level directions using exact field names where useful; avoid brittle low-level encoding specs unless necessary",
        "interaction": "static|hover|filter|select|tabs|search|linked_highlight, with a short reason",
        "annotations": ["specific labels/callouts the chart should include"],
        "disclosures": ["filters, aggregations, scope, exclusions, or units that apply to this view"]
      }
    }
  ],
  "story": {
    "hook": "...",
    "sections": [
      {
        "title": "...",
        "purpose": "why this section exists in the story",
        "view_ids": ["..."],
        "message": "the narrative beat this section should communicate"
      }
    ],
    "payoff": "concise concluding takeaway"
  },
  "evaluation_targets": {
    "insightfulness": ["trends/exceptions/comparisons the artifact should foreground"],
    "narrative_coherence": ["how the sections build from hook to payoff"],
    "visual_craft": ["labeling, chart-type, disclosure, and readability requirements"],
    "functionality": ["interactions that must be implemented and tested"]
  }
}

Workflow:
1. Read task.md and analysis_skeleton.json.
2. Preserve every computed view unless it is redundant or too weak to support the task.
3. Add reader-facing titles, section flow, insights, plot guidance, interactions,
   annotations, and disclosures. Use exact data field names from the skeleton
   when telling the coder what fields matter, but do not over-specify low-level
   encodings such as x/y/color unless they are central to correctness.
4. Keep the manifest compact. Do not inline full datasets or long samples.
5. Write valid JSON to analysis_manifest.json and call finish.

Token Economy: The analyst and storyboard share a 300k generation sub-budget.
Spend most of your effort on coherent story structure and evaluable insight."""

CODER_SYSTEM = """You are the visualization-coder stage for a web data visualization agent. 

Your job is presentation engineering. The analytical source of truth is
analysis_manifest.json plus the output files it references. Build one cohesive, self-contained, interactive artifact at dist/index.html.

Read:
- analysis_manifest.json at the workdir root.

Avoid reading task.md or raw data/ unless absolutely necessary to recover from a
manifest defect. Do not re-analyze, recompute, or second-guess findings. Build
the dashboard from the manifest's story, view presentation guidance, answers,
caveats, disclosures, and referenced outputs.

Build:
- Write source/build.py that reads analysis_manifest.json and every referenced
  outputs/* file, embeds what it needs inline as JSON (use json.dumps() to safely
  inject into <script> tags), and writes dist/index.html. Run it with bash. You do NOT need to read the data files into your own context — build.py reads them.
- Use the manifest's story.hook, story.sections, view titles, answers,
  presentation.insight, presentation.disclosures, caveats, and story.payoff to
  create a clear story.
- Follow each view's presentation.recommended_view, plot_guidance, interaction,
  annotations, and disclosures. Map exact data fields from schema/sample/output
  files; do not invent renamed keys.
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
            "then emit analysis_skeleton.json and outputs/*."
        ),
        tool_names=["read_file", "write_file", "str_replace", "bash", "search"],
        model=pick_model("analysis_builder"),
        max_steps=60,
        prune_keep=12,
        budget=analysis_budget,
        low_water=100_000,
    )
    results.append(analysis)

    skeleton_ok, skeleton_errors = _validate_analysis_skeleton(workdir)
    if skeleton_errors:
        for err in skeleton_errors:
            print(f"[pipeline] analysis_skeleton validation: {err}", file=sys.stderr)

    if skeleton_ok and not analysis_budget.exhausted():
        storyboard = _safe_run(
            ctx=ctx,
            client=client,
            name="storyboard",
            system_prompt=STORYBOARD_SYSTEM,
            user_prompt=(
                f"WORKDIR={wd}\n"
                "Read task.md and analysis_skeleton.json, then write analysis_manifest.json. "
                "Preserve computed answers and output file references; add story, insight, "
                "presentation guidance, and evaluation targets."
            ),
            tool_names=["read_file", "write_file", "str_replace"],
            model=pick_model("storyboard"),
            max_steps=24,
            prune_keep=10,
            budget=analysis_budget,
            low_water=50_000,
        )
        results.append(storyboard)
    elif skeleton_ok:
        print("[pipeline] skipping storyboard because shared analysis budget is exhausted", file=sys.stderr)
    else:
        print("[pipeline] skipping storyboard because analysis_skeleton.json is not build-ready", file=sys.stderr)

    budget.spent += analysis_budget.spent

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
    fallback_info = ensure_fallback_dist(workdir, findings_available=manifest_ok or skeleton_ok)

    # Move intermediate JSON files to source/state/ for clean root directory.
    _cleanup_state_files(workdir)

    return _summarize(
        workdir,
        results,
        skeleton_ok=skeleton_ok,
        skeleton_errors=skeleton_errors,
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
    skeleton_ok: bool,
    skeleton_errors: list[str],
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
            "analysis_ceiling": ANALYSIS_TOKEN_CEILING,
            "dist_ready": dist_ready,
            "analysis_skeleton_valid": skeleton_ok,
            "analysis_skeleton_errors": skeleton_errors,
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

def _validate_analysis_skeleton(workdir: Path) -> tuple[bool, list[str]]:
    """Minimal storyboard-readiness check for analysis_skeleton.json."""
    return _validate_view_artifact(workdir, "analysis_skeleton.json")


def _validate_analysis_manifest(workdir: Path) -> tuple[bool, list[str]]:
    """Minimal build-readiness check for analysis_manifest.json."""
    return _validate_view_artifact(workdir, "analysis_manifest.json")


def _validate_view_artifact(workdir: Path, filename: str) -> tuple[bool, list[str]]:
    """Minimal check that a JSON artifact has views pointing at real output files."""
    manifest_path = workdir / filename
    errors: list[str] = []
    if not manifest_path.exists():
        return False, [f"{filename} missing"]

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return False, [f"{filename} is invalid JSON: {exc}"]
    except OSError as exc:
        return False, [f"{filename} could not be read: {exc}"]

    views = data.get("views")
    if views is None:
        # Backward compatibility for artifacts made by the first manifest draft.
        views = data.get("visualizations")
    if not isinstance(views, list) or not views:
        errors.append("views must be a non-empty list")
        return False, errors

    workdir_resolved = workdir.resolve()
    for idx, view in enumerate(views):
        if not isinstance(view, dict):
            errors.append(f"views[{idx}] is not an object")
            continue
        data_block = view.get("data") if isinstance(view.get("data"), dict) else {}
        data_file = data_block.get("file") or view.get("data_file")
        if not data_file:
            errors.append(f"views[{idx}] is missing data.file")
            continue
        path = Path(str(data_file))
        resolved = (path if path.is_absolute() else (workdir_resolved / path)).resolve()
        try:
            resolved.relative_to(workdir_resolved)
        except ValueError:
            errors.append(f"views[{idx}].data.file points outside workdir: {data_file}")
            continue
        if not resolved.exists():
            errors.append(f"views[{idx}].data.file missing: {data_file}")

    return not errors, errors


def _cleanup_state_files(workdir: Path) -> None:
    import shutil

    state_dir = workdir / "source" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    files_to_move = [
        "analysis_manifest.json",
        "analysis_skeleton.json",
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
