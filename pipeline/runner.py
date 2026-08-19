"""Runs one LLM tool-loop per pipeline stage, with context pruning and a shared token budget."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import ToolContext, build_toolset


_FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "End the stage. Return a small JSON summary of what you produced (paths written, key notes).",
        "parameters": {"type": "object", "properties": {"result": {"type": "object"}}, "required": ["result"]},
    },
}

_HEAVY_ARG_KEYS = ("content", "new_str", "old_str", "script")
_STALE_PREVIEW = 240
_MAX_MODEL_ERRORS = 5


@dataclass
class StageResult:
    name: str
    result: dict[str, Any]
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    steps: int = 0
    finished: bool = False
    tool_counts: dict[str, int] = field(default_factory=dict)
    model_errors: list[str] = field(default_factory=list)


@dataclass
class BudgetTracker:
    """Cumulative token ceiling shared across stages; caps generation so the shared per-job budget leaves room for evaluation."""

    ceiling: int
    spent: int = 0

    def add(self, usage: dict[str, Any] | None) -> None:
        self.spent += int((usage or {}).get("total_tokens", 0) or 0)

    def remaining(self) -> int:
        return max(0, self.ceiling - self.spent)

    def exhausted(self) -> bool:
        return self.spent >= self.ceiling


def run_stage(
    *,
    name: str,
    system_prompt: str,
    user_prompt: str,
    tool_names: list[str] | None = None,
    model: str,
    ctx: Any,
    client: Any,
    tool_schemas: list[dict[str, Any]] | None = None,
    tool_executors: dict[str, Any] | None = None,
    purpose: str = "generation",
    max_steps: int = 24,
    max_history_tokens: int = 40_000, # no longer used in compaction strategy
    prune_keep: int = 12,
    max_tokens: int = 8192,
    budget: BudgetTracker | None = None,
    low_water: int = 60_000,
) -> StageResult:
    if tool_schemas is not None and tool_executors is not None:
        schemas, execs = tool_schemas, tool_executors
    else:
        schemas, execs = build_toolset(tool_names or [])
    tools = [*schemas, _FINISH_SCHEMA]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    tool_counts: dict[str, int] = {}
    model_errors: list[str] = []
    idle_nudges = 0
    warned_budget = False
    warned_steps = False

    try:
        for step in range(1, max_steps + 1):
            # Budget gate: stop before making a call that would overrun the shared
            # generation ceiling; nudge the model to finalize as it nears the limit.
            if budget is not None:
                if budget.exhausted():
                    _log(name, f"budget ceiling hit (spent={budget.spent}); stopping at step {step}")
                    return StageResult(name, {}, usage, step - 1, finished=False, tool_counts=tool_counts, model_errors=model_errors)
                if not warned_budget and budget.remaining() < low_water:
                    warned_budget = True
                    messages.append({"role": "user", "content": (
                        "Your token budget is nearly exhausted. "
                        "Do NOT explore/debug any further. Remove or disable any broken features immediately. "
                        "Finalize the working parts and ship now."
                    )})
                    
                if not warned_steps and step >= max_steps - 3:
                    warned_steps = True
                    messages.append({"role": "user", "content": (
                        "Only 2 steps remaining. "
                        "Do NOT explore/debug any further. Remove or disable any broken features immediately. "
                        "Finalize the working parts and ship now."
                    )})
    
            try:
                message = client.create(
                    model=model,
                    messages=_compact(messages, prune_keep),
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - let stages recover from transient/provider failures
                error_text = _short_error(exc)
                model_errors.append(error_text)
                _log(name, f"model request error {len(model_errors)}/{_MAX_MODEL_ERRORS}: {error_text}")
                if len(model_errors) >= _MAX_MODEL_ERRORS:
                    return StageResult(
                        name,
                        {"error": "repeated model request failures", "model_errors": model_errors},
                        usage,
                        step - 1,
                        finished=False,
                        tool_counts=tool_counts,
                        model_errors=model_errors,
                    )
                messages.append({"role": "user", "content": (
                    "MODEL REQUEST ERROR: the previous request failed before you could respond: "
                    f"{error_text}\nTry to recover using the current files and concise tool calls. "
                    "If the stage has enough working output, call finish with a short JSON summary."
                )})
                continue
            _accumulate(usage, getattr(client, "last_usage", {}))
            if budget is not None:
                budget.add(getattr(client, "last_usage", {}))
            messages.append(message)
    
            calls = message.get("tool_calls") or []
            if not calls:
                idle_nudges += 1
                if idle_nudges >= 3:
                    _log(name, f"no tool call twice; ending stage at step {step}")
                    return StageResult(name, {}, usage, step, finished=False, tool_counts=tool_counts, model_errors=model_errors)
                messages.append({"role": "user", "content": "No tool calls made indicates stalling. Use the tools to make progress, eg write the script then run it with bash. Once the task is done, call finish with a short JSON summary."})
                continue
            idle_nudges = 0
    
            for call in calls:
                fn = call.get("function") or {}
                fname = fn.get("name")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    messages.append(_tool_msg(call, f"Tool argument JSON error: {exc}"))
                    if fname:
                        tool_counts[fname] = tool_counts.get(fname, 0) + 1
                    continue
                if fname:
                    tool_counts[fname] = tool_counts.get(fname, 0) + 1
                if fname == "finish":
                    _log(name, f"finished at step {step}; tokens={usage['total_tokens']} tools={tool_counts}")
                    return StageResult(name, args.get("result") or {}, usage, step, finished=True, tool_counts=tool_counts, model_errors=model_errors)
                executor = execs.get(fname)
                try:
                    output = executor(args, ctx) if executor else f"Unknown tool: {fname}"
                except Exception as exc:  # noqa: BLE001 - tool failures should be recoverable by the model
                    output = f"{fname or 'tool'} error: {_short_error(exc)}"
                messages.append(_tool_msg(call, output))
    
        _log(name, f"hit max_steps={max_steps} without finish; tokens={usage['total_tokens']} tools={tool_counts}")
        return StageResult(name, {}, usage, max_steps, finished=False, tool_counts=tool_counts, model_errors=model_errors)
    finally:
        root = getattr(ctx, "tool_root", None) or getattr(ctx, "workdir", None)
        if root:
            try:
                (Path(root) / f"messages_{name}.json").write_text(json.dumps(messages, indent=2), encoding="utf-8")
            except Exception as exc:
                _log(name, f"failed to write messages: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_msg(call: dict[str, Any], content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.get("id"), "content": content}


def _accumulate(total: dict[str, int], last: dict[str, Any]) -> None:
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        total[k] += int(last.get(k, 0) or 0)


def _short_error(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0][:500] or exc.__class__.__name__


def _compact(messages: list[dict[str, Any]], prune_keep: int) -> list[dict[str, Any]]:
    """Return a compacted copy for sending to the LLM."""
    if len(messages) <= prune_keep + 2:
        return messages
    tail_start = len(messages) - prune_keep
    compacted: list[dict[str, Any]] = list(messages[:2])
    for i in range(2, len(messages)):
        msg = messages[i]
        if i >= tail_start:
            compacted.append(msg)
            continue
        role = msg.get("role")
        if role == "tool":
            content = msg.get("content") or ""
            if len(content) > _STALE_PREVIEW:
                compacted.append({**msg, "content": content[:_STALE_PREVIEW] + f"\n... [{len(content)} chars, trimmed]"})
            else:
                compacted.append(msg)
        elif role == "assistant" and msg.get("tool_calls"):
            compacted.append(_shrink_assistant(msg))
        else:
            compacted.append(msg)
    return compacted


def _shrink_assistant(msg: dict[str, Any]) -> dict[str, Any]:
    """Return msg with heavy tool-call argument payloads preview-truncated."""
    changed = False
    new_calls = []
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        shrunk, did = _shrink_args(fn.get("arguments") or "")
        if did:
            changed = True
            call = {**call, "function": {**fn, "arguments": shrunk}}
        new_calls.append(call)
    return {**msg, "tool_calls": new_calls} if changed else msg


def _shrink_args(raw: str) -> tuple[str, bool]:
    """Truncate large string values in a tool-call arguments JSON, keeping a preview."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        if len(raw) > _STALE_PREVIEW:
            return (raw[:_STALE_PREVIEW] + f"... [{len(raw)} chars, trimmed]", True)
        return (raw, False)
    changed = False
    for k in _HEAVY_ARG_KEYS:
        v = args.get(k)
        if isinstance(v, str) and len(v) > _STALE_PREVIEW:
            args[k] = v[:_STALE_PREVIEW] + f"... [{len(v)} chars, trimmed]"
            changed = True
    return (json.dumps(args), True) if changed else (raw, False)


def _log(stage: str, msg: str) -> None:
    print(f"[stage:{stage}] {msg}", file=sys.stderr)
