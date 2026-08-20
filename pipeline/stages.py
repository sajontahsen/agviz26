"""Generation pipeline: analysis_builder -> narrative_coder, passing one compact artifact forward."""
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
GEN_TOKEN_CEILING = 700_000
ANALYSIS_TOKEN_CEILING = 300_000


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

CLOUD_SONNET = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
CLOUD_OPUS = "global.anthropic.claude-opus-4-8"
LOCAL_MODEL = os.environ.get("VIS_ARENA_LOCAL_MODEL", "gpt-5-nano")

_CLOUD_ROLES = {
    "analysis_builder": CLOUD_OPUS,
    "narrative_coder": CLOUD_OPUS,
}


def pick_model(role: str) -> str:
    """Cloud jobs get role-appropriate Claude models; local runs use LOCAL_MODEL."""
    if os.environ.get("VIS_ARENA_JOB_ID"):
        return _CLOUD_ROLES.get(role, CLOUD_OPUS)
    return LOCAL_MODEL


# ---------------------------------------------------------------------------
# Stage prompts
# ---------------------------------------------------------------------------

ANALYSIS_BUILDER_SYSTEM = """You are the analysis-builder stage for a web data visualization agent.

Your job is
- to understand the task, inspect the raw data only as much as needed,
- choose feasible analytical views, 
- compute the answers with Python, 
- and emit a compact analysis skeleton for the coder stage.

Workflow:
1. read_file task.md, every documentation file mentioned, and list data/ to see the files and their formats.
2. Inspect small data samples/headers/schemas. Do not dump full datasets into your reasoning. Instead use `.info()`, `.head(5)`, or print dict/graph samples. If you need a overall profile of the data, write a Python script and let IT compute the profile.
3. Decompose the task into the analytical questions whose answers will drive an insightful, well-structured visualization.
4. Write source/analyze.py that computes the answers to the questions. It must:
   - load raw data once where practical;
   - create outputs/ if needed;
   - compute each answer inside its own try/except block;
  - write one plot-ready output data file per successful view, never inline full data arrays in the skeleton;
   - build analysis_skeleton.json from successful views only;
   - print concise traceback summaries for failed views;
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

analysis_skeleton.json MUST be written programmatically by source/analyze.py.
Do not hand-write final analytical numbers after eyeballing output. The script
is the source of truth for findings, file paths, schemas, samples, and caveats.

Required output:
- source/analyze.py
- outputs/* plot-ready data files
- analysis_skeleton.json at the workdir root

Output limit: Each response is capped at ~8192 tokens. If source/analyze.py is
too large for a single write_file call, write the first chunk with append=false,
then continue with append=true for following chunks. For small exact edits, use
str_replace; for localized multi-line edits, use apply_patch instead of rewriting.

Token Economy: Work economically — generation and evaluation share a fixed
competition token budget."""

NARRATIVE_CODER_SYSTEM = """You are the narrative-coder stage for a web data visualization agent.

Your job is to turn computed analysis into one cohesive, self-contained,
interactive data story at dist/index.html. You own insightfulness,
narrative coherence, visual craft, and functionality. 

Inputs::
- analysis_skeleton.json — the authoritative inventory of computed views,
  output files, schemas, samples, methods, rationales, caveats, assumptions,
  and limitations.

Do NOT read task.md or raw data/ unless absolutely necessary to recover from a skeleton
defect. Do NOT re-analyze or second-guess the analyst. Use only
analysis_skeleton.json and the referenced outputs/* files as analytical truth.

Workflow:
1. Read the inputs to understand what was asked and what findings were computed.
2. Plan the story arc before coding. Choose a hook, coherent sections and 
   a payoff that directly answer the task. Group related views; 
   drop only views that are redundant or too weak.
3. Decide chart forms and interactions while planning the page, not in a
   separate manifest. Prefer familiar, inspectable charts over clever forms.
4. Write source/build.py that reads analysis_skeleton.json and every referenced
  outputs/* file, embeds what it needs inline as JSON (use json.dumps() to safely
   inject into <script> tags), and writes dist/index.html. Run it with bash.
   You do NOT need to read the data files into your own context; build.py
   reads them.
5. Verify, fix, rerun, and verify again until the artifact renders cleanly.

Narrative and insight requirements:
- The page must feel like an analytical answer, not a chart dump.
- Start with a specific hook that frames the central question and why the
  findings matter.
- Each section needs a clear claim, supporting visualization, and short
  interpretation. Call out trends, exceptions, comparisons, turning points,
  and implications.
- Make the payoff decision-pointing: what should the intended reader conclude,
  inspect next, or believe based on the evidence?
- Preserve caveats and data limitations near the relevant charts, especially
  synthetic/future years, incomplete influence edges, filters, units, and
  aggregation choices. Mention only caveats that apply.
- If analysis_skeleton lacks prose answers, derive concise claims from the
  plot-ready outputs inside build.py, not from raw data.
- Answers must be supported by visual evidence, not prose or text alone.  

Visual craft and functionality requirements:
- Choose chart types based on the output data grain and task: line/area for
  time trends, ranked bars for top categories, scatter/small multiples for
  comparisons, compact tables for sparse evidence, and summarized networks for
  graph relations.
- Use exact field names from answer_data.schema/sample/output files. Do not
  invent renamed keys. Be defensive about nulls and about samples represented
  as arrays rather than objects.
- Implement a number of meaningful controls yourself with vanilla JS
  when they help the story: tabs, filters, search, toggles, or linked
  highlighting. Do not rely purely on Plotly defaults for interactivity.
- Use Plotly hover/tooltips, readable axes, legends, labels, units, annotations,
  and chart captions. Never render tens of thousands of nodes raw.
- Apply a cohesive, minimalist design system using vanilla CSS. 
  Choose an accessible color palette. Avoid unstyled browser defaults. 
  Keep layout readable at desktop and mobile widths.
- Rendering libraries (pin these exact versions; load from CDN):
  Plotly.js https://cdn.plot.ly/plotly-2.35.2.min.js
- The page must render from dist/index.html with no dev server.

- Token Economy: Work economically — the job has a shared token budget.
- Output limit: Each response is capped at ~8192 tokens. When writing large files 
  with write_file, write the first chunk with append=false, then
  continue with append=true for each following chunk. For small edits to an
  existing file, use str_replace or apply_patch instead of rewriting.

You are judged on these:
- functionality: interactions (filters, tooltips, selection) actually work.
- visual_craft: right chart types, clear titles/axes/labels, disclosed filters/timeframes/scope, readable color.
- insightfulness: call out trends, exceptions, comparisons — not just raw charts.
- narrative_coherence: an explicit hook -> build -> payoff arc; consistent encodings across panels.
- data_fidelity: numbers on screen match the actual data.

Validation Loop:
When the page is written, you MUST call the verify tool (with no arguments, to serve dist/ locally). It captures a screenshot and returns a health summary.
If verify reports any errors/discrepancy:
1. Use str_replace or apply_patch to fix the bug in build.py/HTML.
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
  continue with append=true for each following chunk. For localized edits, use
  str_replace or apply_patch instead of rewriting.

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
        tool_names=["read_file", "write_file", "str_replace", "apply_patch", "bash", "search"],
        model=pick_model("analysis_builder"),
        max_steps=60,
        prune_keep=16,
        budget=analysis_budget,
        low_water=100_000,
    )
    results.append(analysis)

    skeleton_ok, skeleton_errors = _validate_analysis_skeleton(workdir)
    if skeleton_errors:
        for err in skeleton_errors:
            print(f"[pipeline] analysis_skeleton validation: {err}", file=sys.stderr)

    budget.spent += analysis_budget.spent

    if skeleton_ok:
        coder_system = NARRATIVE_CODER_SYSTEM
        coder_user = (
            f"WORKDIR={wd}\n"
            "Read analysis_skeleton.json. Design the narrative, " 
            "then write and run source/build.py to produce "
            "dist/index.html. Verify it renders before finishing."
        )
    else:
        print("[pipeline] running fallback coder because analysis_skeleton.json is not build-ready", file=sys.stderr)
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
        tool_names=["read_file", "write_file", "str_replace", "apply_patch", "bash", "search", "verify"],
        model=pick_model("narrative_coder"),
        max_steps=70,
        prune_keep=16,
        budget=budget,
    ))

    # Guarantee a renderable artifact so the job always yields a scorable
    # preview to inspect, rather than a hard "dist/index.html was not created".
    fallback_info = ensure_fallback_dist(workdir, findings_available=skeleton_ok)

    # Move intermediate JSON files to source/state/ for clean root directory.
    _cleanup_state_files(workdir)

    return _summarize(
        workdir,
        results,
        skeleton_ok=skeleton_ok,
        skeleton_errors=skeleton_errors,
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
            "model_errors": r.model_errors,
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
            "fallback": fallback_info,
        },
        "stages": stages_meta,
    }

    notes_str = json.dumps(notes_data, indent=2)
    print(f"[pipeline]\n{notes_str}", file=sys.stderr)
    return {"notes": notes_data}


# ---------------------------------------------------------------------------
# Analysis artifact validation
# ---------------------------------------------------------------------------

def _validate_analysis_skeleton(workdir: Path) -> tuple[bool, list[str]]:
    """Minimal narrative-coder-readiness check for analysis_skeleton.json."""
    return _validate_view_artifact(workdir, "analysis_skeleton.json")


def _validate_view_artifact(workdir: Path, filename: str) -> tuple[bool, list[str]]:
    """Minimal check that a JSON artifact has views pointing at real output files."""
    manifest_path = workdir / filename
    errors: list[str] = []
    if not manifest_path.exists():
        return False, [f"{filename} missing"]

    ## the following are brittle validations. will be replaced once schema is finalized or dropped altogether in favor of having the analyst stage fix its own errors via similar deterministic checks
    # try:
    #     data = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    # except json.JSONDecodeError as exc:
    #     return False, [f"{filename} is invalid JSON: {exc}"]
    # except OSError as exc:
    #     return False, [f"{filename} could not be read: {exc}"]

    # views = data.get("views")
    # if views is None:
    #     # Backward compatibility for artifacts made by the first manifest draft.
    #     views = data.get("visualizations")
    # if not isinstance(views, list) or not views:
    #     errors.append("views must be a non-empty list")
    #     return False, errors

    # workdir_resolved = workdir.resolve()
    # for idx, view in enumerate(views):
    #     if not isinstance(view, dict):
    #         errors.append(f"views[{idx}] is not an object")
    #         continue
    #     data_block = view.get("data") if isinstance(view.get("data"), dict) else {}
    #     data_file = data_block.get("file") or view.get("data_file")
    #     if not data_file:
    #         errors.append(f"views[{idx}] is missing data.file")
    #         continue
    #     path = Path(str(data_file))
    #     resolved = (path if path.is_absolute() else (workdir_resolved / path)).resolve()
    #     try:
    #         resolved.relative_to(workdir_resolved)
    #     except ValueError:
    #         errors.append(f"views[{idx}].data.file points outside workdir: {data_file}")
    #         continue
    #     if not resolved.exists():
    #         errors.append(f"views[{idx}].data.file missing: {data_file}")

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
