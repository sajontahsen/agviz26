"""Vision-first evaluation pipeline."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from llm_client import make_llm_client

from .eval_tools import EvalContext, ToolContent, build_eval_toolset

CLOUD_MODEL = "global.anthropic.claude-opus-4-8"
LOCAL_MODEL = os.environ.get("VIS_ARENA_LOCAL_MODEL", "gpt-5-nano")

MAX_MODEL_CALLS = 30
KEEP_IMAGE_MESSAGES = 2
MIN_REMAINING_TOKENS = 60_000
TRACE_CONTENT_PREVIEW = 300
TRACE_ARGS_PREVIEW = 300

REQUIRED_CRITERIA = ["data_fidelity", "insightfulness", "narrative_coherence", "visual_craft", "functionality"]


def _pick_model() -> str:
    if os.environ.get("VIS_ARENA_JOB_ID"):
        return CLOUD_MODEL
    return LOCAL_MODEL


EVAL_SYSTEM = """\
You are an impartial visualization evaluator. Your goal is not to reward slick
text: your goal is to judge whether the rendered artifact genuinely answers the
task as a visualization.

Core principle:
- Visual evidence is primary. First observe the page and LOOK at the screenshot.
- Text claims, headings, and captions can explain a visualization, but they do
  not compensate for blank, broken, unreadable, or contradictory visual marks.
- DOM inspection is supporting evidence. It can confirm labels, values, axes,
  legends, and tooltips, but it must not replace visual judgment.
- Functionality is evidence-based: only credit controls that you actually use
  and visually confirm after the action.

Workflow:
1. Read task.md in the workdir to understand what the artifact was asked to
   communicate. If the held-out workdir includes additional task docs, you may
   read them, but do not read source code or raw data to replace visual judging.
2. observe() the initial page. Use the screenshot as your eyes, and use the
   returned telemetry for errors, chart counts, visible controls, and text.
3. inspect() targeted selectors when you need exact labels, legends, values,
   tables, or tooltip text that are visible in the rendered page.
4. If observe() reports meaningful analytic controls, use act() on each
   distinct control type you can reasonably test. act() returns a new
   screenshot; compare before/after visually and with telemetry.
5. observe(full_page=true) or observe(selector=...) if important sections are
   below the fold or need closer inspection.
6. finish only after your scores are grounded in visual observations.

Rate each criterion 1-5 using these anchors. Use exact ids. Award 5 for
excellent work under the available rendered-page evidence; do not reserve 5
for impossible full-data verification or flawlessness. Minor caveats can still
be compatible with 5 if they do not materially weaken the criterion.

1. data_fidelity - visible values, totals, trends, and encodings are internally
   consistent and answer the task.
   1 fabricated/contradictory or no visual data; 2 major mismatch or broken
   marks; 3 mostly plausible but weak support/disclosure; 4 strong with a
   material unresolved concern; 5 all visible claims, values, encodings, units,
   and disclosed transformations appear faithful and well-supported.

2. insightfulness - the artifact helps a reader learn something beyond raw
   plotting.
   1 raw/uninterpretable chart; 2 basic observation; 3 headline pattern; 4
   trends plus comparisons/exceptions; 5 rich, actionable, decision-pointing.

3. narrative_coherence - the story arc and internal consistency.
   1 contradictory/no story; 2 disconnected panels; 3 mostly coherent with an
   implicit takeaway; 4 clear sections with a material gap; 5 tight
   hook-build-payoff arc with consistent encodings and purposeful panels.

4. visual_craft - chart choice, encoding, layout, labels, legends, axes,
   legibility, accessibility, and disclosure of scope/filters/time/aggregation.
   1 illegible/misrepresenting/broken; 2 major label or chart-choice problems;
   3 readable but basic; 4 well-matched with a material limitation; 5 polished,
   accessible, legible, and well-disclosed.

5. functionality - working interaction that aids analysis.
   1 broken or unusable; 2 controls mostly dead/confusing; 3 core interactions
   work; 4 tested interactions work but add limited value or leave a material
   concern; 5 meaningful analytic interactions work and deepen the analysis.

Evidence requirements:
- Cite observation ids such as obs_1 / obs_2 in every criterion.
- Mention what you saw in screenshots, not just what body text claimed.
- If charts are blank/broken or no visual marks are visible, score data_fidelity
  and visual_craft low even if the prose is persuasive.
- If controls exist but you do not test them, functionality cannot exceed 3.

Call finish with:
{
  "summary": "2-3 sentence overall assessment grounded in visual evidence.",
  "criteria": [
    {"id": "<criterion_id>", "score": <1-5>, "max_score": 5, "anchor": "<matched level>", "evidence": ["obs_1: ...", "obs_2: ..."]}
  ],
  "metadata": {"evaluator": "evaluator-v2"}
}"""


FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Submit the final visual-evidence-based evaluation.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "enum": REQUIRED_CRITERIA},
                            "score": {"type": "integer", "minimum": 1, "maximum": 5},
                            "max_score": {"type": "integer"},
                            "anchor": {"type": "string"},
                            "evidence": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "score", "evidence"],
                    },
                },
                "metadata": {"type": "object"},
            },
            "required": ["summary", "criteria"],
        },
    },
}


def run_evaluation(workdir: Path, artifact_url: str) -> dict[str, Any]:
    workdir = Path(workdir).resolve()
    ctx = EvalContext(workdir=workdir, artifact_url=artifact_url)
    client = make_llm_client("evaluation")
    model = _pick_model()
    schemas, executors = build_eval_toolset()
    tools = [*schemas, FINISH_SCHEMA]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "user", "content": f"WORKDIR={workdir}\nARTIFACT_URL={artifact_url}\n\nJudge the rendered artifact visually, then score the five criteria."},
    ]
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    tool_counts: dict[str, int] = {}
    finish_rejections = 0
    finished = False
    result: dict[str, Any] = {}
    assistant_steps = 0
    model_errors: list[str] = []
    image_send_failures = 0

    print(f"[evaluator:v2] model={model}", file=sys.stderr)
    try:
        for step in range(1, MAX_MODEL_CALLS + 1):
            remaining = getattr(client, "remaining_tokens", None)
            tool_choice: str | dict[str, Any] = "auto"
            if step == MAX_MODEL_CALLS or (remaining is not None and remaining < MIN_REMAINING_TOKENS):
                messages.append({"role": "user", "content": "FINAL CALL: call finish now using the visual evidence you gathered. Use low scores where evidence is missing."})
                tool_choice = {"type": "function", "function": {"name": "finish"}}

            try:
                message = client.create(
                    model=model,
                    messages=_compact_for_model(messages),
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=8192,
                )
            except Exception as exc:  # noqa: BLE001 - evaluation must return a report if at all possible
                error_text = str(exc)[:500]
                model_errors.append(error_text)
                if _messages_have_images(messages):
                    image_send_failures += 1
                    messages = _strip_all_message_images(messages)
                    if image_send_failures >= 2:
                        guidance = (
                            "MODEL REQUEST ERROR: request failed again, error: "
                            f"{error_text}\n"
                            "Errors are persisting. Fall back to the text telemetry, "
                            "saved screenshot paths, DOM inspection results, and interaction logs you already have. "
                            "Call finish with conservative scores where visual evidence is incomplete."
                        )
                    else:
                        guidance = (
                            "MODEL REQUEST ERROR: the previous request failed before you could respond, "
                            f"error: {error_text}\n"
                            "Decide the recovery step. You may try a smaller observe(selector=...) or viewport "
                            "observation, use existing text telemetry and saved screenshot paths, inspect the DOM, "
                            "or finish conservatively if there is enough evidence."
                        )
                    messages.append({
                        "role": "user",
                        "content": guidance,
                    })
                    continue
                if len(model_errors) >= 5:
                    result = _fallback_result(ctx, reason="repeated model request failures")
                    break
                messages.append({
                    "role": "user",
                    "content": (
                        "MODEL REQUEST ERROR: the previous request failed before you could respond: "
                        f"{error_text}\nContinue if you can recover; if errors persist, call finish with "
                        "a conservative evaluation from the evidence already gathered."
                    ),
                })
                continue
            assistant_steps += 1
            _accumulate(usage, getattr(client, "last_usage", {}))
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                messages.append({"role": "user", "content": "Read the task file and observe the artifact, then inspect or act as needed. If you have enough visual evidence, call finish."})
                continue

            for call in calls:
                function = call.get("function") or {}
                fname = function.get("name")
                if fname:
                    tool_counts[fname] = tool_counts.get(fname, 0) + 1
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    messages.append(_tool_msg(call, f"Tool argument JSON error: {exc}"))
                    continue

                if fname == "finish":
                    candidate = {"summary": args.get("summary", ""), "criteria": args.get("criteria", []), "metadata": args.get("metadata") or {}}
                    problems = _finish_problems(candidate, ctx)
                    if problems and finish_rejections < 2:
                        finish_rejections += 1
                        messages.append(_tool_msg(call, "finish REJECTED: " + "; ".join(problems) + ". Gather the missing visual evidence or fix the report, then call finish again."))
                        continue
                    result = candidate
                    finished = True
                    break

                executor = executors.get(fname)
                try:
                    output: ToolContent = executor(args, ctx) if executor else f"Unknown tool: {fname}"
                except Exception as exc:  # noqa: BLE001 - report tool failures to the model, do not crash eval
                    output = f"{fname or 'tool'} error: {exc}"
                messages.append(_tool_msg(call, output))

            if finished:
                break
        else:
            result = _fallback_result(ctx, reason="maximum evaluator calls reached")
    finally:
        _write_message_trace(workdir, messages)
        ctx.close()

    result = _compute_score(result, ctx)
    screenshots = sorted(str(p) for p in ctx.artifacts_dir.glob("eval_*.jpg"))
    result.setdefault("browser", {}).update({
        "tool": "playwright",
        "entrypoint_url": artifact_url,
        "viewports": [{"width": 1440, "height": 900}],
        # "observations": ctx.observations,
        # "actions": ctx.actions,
    })
    result.setdefault("artifacts", {}).update({"screenshots": screenshots})
    meta = result.setdefault("metadata", {})
    meta["evaluator"] = meta.get("evaluator") or "evaluator-v2"
    meta["tokens"] = usage
    meta["steps"] = assistant_steps
    # meta["tool_counts"] = tool_counts
    meta["finished"] = finished
    # meta["finish_rejections"] = finish_rejections
    # meta["model_errors"] = model_errors
    # meta["image_send_failures"] = image_send_failures
    # meta["message_trace"] = str(workdir / "messages_evaluate.json")
    # meta["message_trace_written"] = (workdir / "messages_evaluate.json").exists()
    # meta["messages"] = _messages_for_trace(messages)

    tc_str = " ".join(f"{k}={v}" for k, v in sorted(tool_counts.items())) if tool_counts else "-"
    print(
        f"[evaluator:v2] finished={finished} tokens={usage['total_tokens']} "
        f"score={result.get('score')} observations={len(ctx.observations)} tools=[{tc_str}]",
        file=sys.stderr,
    )
    return result


def _compute_score(result: dict[str, Any], ctx: EvalContext) -> dict[str, Any]:
    criteria = _normalize_criteria(result.get("criteria", []))
    for criterion in criteria:
        criterion["max_score"] = 5

    result["criteria"] = criteria
    result["score"] = sum(c["score"] for c in criteria) * 4
    result["max_score"] = 100
    meta = result.setdefault("metadata", {})
    meta["observation_count"] = len(ctx.observations)
    meta["actions_count"] = len(ctx.actions)
    return result


def _normalize_criteria(raw: Any) -> list[dict[str, Any]]:
    by_id = {str(c.get("id")): c for c in raw or [] if isinstance(c, dict)}
    out: list[dict[str, Any]] = []
    for cid in REQUIRED_CRITERIA:
        c = by_id.get(cid, {})
        try:
            score = int(c.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        out.append({
            "id": cid,
            "score": max(1, min(5, score)),
            "max_score": 5,
            "anchor": str(c.get("anchor") or ""),
            "evidence": c.get("evidence") if isinstance(c.get("evidence"), list) else [],
        })
    return out


def _finish_problems(result: dict[str, Any], ctx: EvalContext) -> list[str]:
    problems: list[str] = []
    if not ctx.observations:
        problems.append("no observe() call has produced visual evidence")
    if ctx.controls_seen > 0 and not ctx.actions:
        problems.append("interactive controls were seen but no act() call tested them")
    ids = {str(c.get("id")) for c in result.get("criteria", []) if isinstance(c, dict)}
    if ids != set(REQUIRED_CRITERIA):
        problems.append("criteria must contain exactly: " + ", ".join(REQUIRED_CRITERIA))
    for c in result.get("criteria", []) or []:
        if not isinstance(c, dict):
            continue
        evidence = c.get("evidence") or []
        if not isinstance(evidence, list) or not any("obs_" in str(item) for item in evidence):
            problems.append(f"{c.get('id', 'criterion')} evidence must cite visual observation ids")
            break
    return problems


def _fallback_result(ctx: EvalContext, reason: str = "insufficient completed evaluation") -> dict[str, Any]:
    if ctx.observations:
        summary = "Evaluation did not finish cleanly, so scores are conservative and based on captured visual observations."
        score = 2
    else:
        summary = "Evaluation did not capture visual evidence, so the artifact cannot be credited beyond a minimal score."
        score = 1
    return {
        "summary": summary,
        "criteria": [
            {"id": cid, "score": score, "max_score": 5, "anchor": "fallback conservative score", "evidence": [f"fallback: {reason}"]}
            for cid in REQUIRED_CRITERIA
        ],
        "metadata": {"evaluator": "evaluator-v2", "fallback": True, "fallback_reason": reason},
    }


def _tool_msg(call: dict[str, Any], content: ToolContent) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.get("id"), "content": content}


def _accumulate(total: dict[str, int], last: dict[str, Any]) -> None:
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        total[k] += int((last or {}).get(k, 0) or 0)


def _compact_for_model(messages: list[dict[str, Any]], keep_image_messages: int = KEEP_IMAGE_MESSAGES) -> list[dict[str, Any]]:
    image_indices = [
        i for i, msg in enumerate(messages)
        if isinstance(msg.get("content"), list) and any(_is_image_part(part) for part in msg["content"])
    ]
    keep_images = set(image_indices[-keep_image_messages:]) if keep_image_messages > 0 else set()
    compacted: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list) and i not in keep_images:
            compacted.append({**msg, "content": _strip_image_parts(content)})
        elif isinstance(content, str) and len(content) > 14000:
            compacted.append({**msg, "content": content[:13800] + f"\n... [{len(content) - 13800} chars trimmed]"})
        else:
            compacted.append(msg)
    return compacted


def _strip_image_parts(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in content:
        if _is_image_part(part):
            parts.append({"type": "text", "text": "[old screenshot image omitted; use the observation text and latest images]"})
        else:
            text = str(part.get("text") if isinstance(part, dict) else part)
            if len(text) > 4000:
                text = text[:3900] + f"\n... [{len(text) - 3900} chars trimmed]"
            parts.append({"type": "text", "text": text})
    return parts


def _strip_all_message_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**msg, "content": _strip_image_parts(msg["content"])}
        if isinstance(msg.get("content"), list)
        else msg
        for msg in messages
    ]


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in {"image_url", "image"}


def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(msg.get("content"), list) and any(_is_image_part(part) for part in msg["content"])
        for msg in messages
    )


def _write_message_trace(workdir: Path, messages: list[dict[str, Any]]) -> None:
    try:
        (workdir / "messages_evaluate.json").write_text(json.dumps(_messages_for_trace(messages), indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[evaluator:v2] failed to write messages: {exc}", file=sys.stderr)


def _messages_for_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_message_for_trace(msg) for msg in messages]


def _message_for_trace(msg: dict[str, Any]) -> dict[str, Any]:
    traced: dict[str, Any] = {"role": msg.get("role")}
    if msg.get("tool_call_id"):
        traced["tool_call_id"] = msg.get("tool_call_id")
    if "content" in msg:
        traced["content"] = _content_for_trace(msg.get("content"))
    if msg.get("tool_calls"):
        traced["tool_calls"] = _tool_calls_for_trace(msg.get("tool_calls") or [])
    return traced


def _content_for_trace(content: Any) -> str:
    if isinstance(content, str):
        return _preview(content, TRACE_CONTENT_PREVIEW)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if _is_image_part(part):
                parts.append("[screenshot image omitted]")
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or part))
            else:
                parts.append(str(part))
        return _preview("\n".join(parts), TRACE_CONTENT_PREVIEW)
    return _preview(str(content), TRACE_CONTENT_PREVIEW)


def _tool_calls_for_trace(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traced = []
    for call in calls:
        function = call.get("function") or {}
        traced.append({
            "id": call.get("id"),
            "name": function.get("name"),
            "arguments": _preview(function.get("arguments") or "", TRACE_ARGS_PREVIEW),
        })
    return traced


def _preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} chars trimmed]"
