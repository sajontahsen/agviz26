"""Evaluation pipeline: structured browser-based assessment with deterministic scoring."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from llm_client import make_llm_client

from .eval_tools import EvalContext, build_eval_toolset
from .runner import run_stage

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

CLOUD_MODEL = "global.anthropic.claude-opus-4-8"
LOCAL_MODEL = os.environ.get("VIS_ARENA_LOCAL_MODEL", "gpt-5-nano")


def _pick_model() -> str:
    if os.environ.get("VIS_ARENA_JOB_ID"):
        return CLOUD_MODEL
    return LOCAL_MODEL


# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

EVAL_SYSTEM = """\
You are an impartial visualization evaluator with structured browser tools.

Workflow:
1. read_file("task.md") — understand what the task asked for.
2. render() — load the artifact in a headless browser. Review the returned page
   title, body text, console errors, chart counts, and discovered interactive
   elements.
3. Use inspect() to examine specific elements: chart titles, axis labels, legend
   entries, data values visible in the visualization.
4. Use interact() to test interactive controls discovered by render (tabs,
   dropdowns, filters, buttons). Test each distinct control type at least once.
   Verify that interactions actually change the displayed content.
5. Call finish() with your scoring JSON.

Rate each of the five criteria 1-5 using the anchors below. For each criterion
set "score" to that 1-5 integer and "max_score" to 5. Do NOT compute an overall
score — the system calculates it deterministically from your per-criterion scores.

Required criteria (use these exact ids):

1. data_fidelity — do displayed values, totals, and trends look internally
   consistent and match what the task asks for?
     1 fabricated or contradicts data · 2 major mismatch in key values ·
     3 mostly faithful, minor discrepancies · 4 faithful, all spot-checks pass ·
     5 fully faithful incl. aggregations, units, and edge cases.

2. insightfulness — does the artifact go beyond plotting to identify trends,
   exceptions, and implications?
     1 raw chart · 2 basic observation · 3 headline pattern called out ·
     4 trends + exceptions + comparison · 5 rich, actionable, decision-pointing.

3. narrative_coherence — story arc (hook -> build -> payoff) AND internal
   consistency across panels (no contradictions; encodings hold)?
     1 contradictory or no story · 2 disconnected panels, no setup or takeaway ·
     3 mostly coherent, implicit takeaway · 4 clear sections, consistent
     encodings, reasoned transitions · 5 tight arc; every panel reinforces a
     distinct hook and decisive payoff.

4. visual_craft — chart type + encodings + axes + labels/captions + legibility,
   including disclosure of filters / time frames / aggregations / scope?
     1 misrepresents or illegible · 2 suboptimal type or major label gaps ·
     3 readable, basic labels cover main filters · 4 well-matched type +
     encodings + captions name assumptions · 5 optimal, accessible color,
     comprehensive disclosure (filters + time + scope + exclusions).

5. functionality — do interactive controls (filters, tooltips, selection,
   resize) work? SCORE ONLY FROM INTERACTIONS YOU ACTUALLY PERFORMED, not
   from reading source code.
     1 broken / console errors · 2 some controls dead · 3 core interactions
     work · 4 all controls work as implied · 5 all work and meaningfully aid
     analysis.

Attach short evidence strings per criterion — what you observed via render,
inspect, or interact. Cite DOM text, error messages, and interaction results.

Call finish with this JSON shape:
{
  "summary": "2-3 sentence overall assessment.",
  "criteria": [
    {"id": "<criterion_id>", "score": <1-5>, "max_score": 5, "anchor": "<matched level description>", "evidence": ["observation 1", "..."]}
  ],
  "metadata": {"evaluator": "pipeline-evaluator"}
}"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

REQUIRED_CRITERIA = ["data_fidelity", "insightfulness", "narrative_coherence", "visual_craft", "functionality"]


def run_evaluation(workdir: Path, artifact_url: str) -> dict[str, Any]:
    workdir = Path(workdir).resolve()
    ctx = EvalContext(workdir=workdir, artifact_url=artifact_url)
    client = make_llm_client("evaluation")
    model = _pick_model()
    schemas, executors = build_eval_toolset()

    print(f"[evaluator] model={model}", file=sys.stderr)
    try:
        stage = run_stage(
            name="evaluate",
            system_prompt=EVAL_SYSTEM,
            user_prompt=f"WORKDIR={workdir}\nARTIFACT_URL={artifact_url}\n\nRead task.md, then render and evaluate the artifact.",
            model=model,
            ctx=ctx,
            client=client,
            tool_schemas=schemas,
            tool_executors=executors,
            max_steps=15,
            max_history_tokens=30_000,
        )
    finally:
        ctx.close()

    result = stage.result
    result = _compute_score(result)

    screenshots = sorted(str(p) for p in ctx.artifacts_dir.glob("eval_*.png"))
    result.setdefault("browser", {}).update({
        "tool": "playwright",
        "entrypoint_url": artifact_url,
        "viewports": [{"width": 1440, "height": 900}],
    })
    result.setdefault("artifacts", {}).update({"screenshots": screenshots})
    meta = result.setdefault("metadata", {})
    meta["tokens"] = stage.usage
    meta["steps"] = stage.steps
    meta["tool_counts"] = stage.tool_counts
    meta["finished"] = stage.finished

    tc = stage.tool_counts
    tc_str = " ".join(f"{k}={v}" for k, v in sorted(tc.items())) if tc else "-"
    print(
        f"[evaluator] finished={stage.finished} steps={stage.steps} "
        f"tokens={stage.usage['total_tokens']} score={result.get('score')} tools=[{tc_str}]",
        file=sys.stderr,
    )
    return result


def _compute_score(result: dict[str, Any]) -> dict[str, Any]:
    criteria = result.get("criteria", [])
    by_id = {c["id"]: c for c in criteria if isinstance(c, dict) and c.get("id")}

    total = 0
    for cid in REQUIRED_CRITERIA:
        c = by_id.get(cid)
        if not c:
            continue
        s = c.get("score", 0)
        s = max(0, min(5, int(s))) if isinstance(s, (int, float)) else 0
        c["score"] = s
        c["max_score"] = 5
        total += s

    result["score"] = total * 4
    result["max_score"] = 100
    return result
