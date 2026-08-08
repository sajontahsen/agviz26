"""Runs one LLM tool-loop per pipeline stage, with context pruning and a shared token budget."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
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

_ELIDED = "[older tool output elided to save context]"
# Argument keys whose values can be large (file/edit/script payloads).
_HEAVY_ARG_KEYS = ("content", "new_str", "old_str", "script")
_HEAVY_ARG_THRESHOLD = 200  # chars; below this, leave the value alone


@dataclass
class StageResult:
    name: str
    result: dict[str, Any]
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    steps: int = 0
    finished: bool = False
    tool_counts: dict[str, int] = field(default_factory=dict)


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
    max_history_tokens: int = 40_000,
    prune_keep: int | None = None,
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
    idle_nudges = 0
    warned_budget = False

    for step in range(1, max_steps + 1):
        # Budget gate: stop before making a call that would overrun the shared
        # generation ceiling; nudge the model to finalize as it nears the limit.
        if budget is not None:
            if budget.exhausted():
                _log(name, f"budget ceiling hit (spent={budget.spent}); stopping at step {step}")
                return StageResult(name, {}, usage, step - 1, finished=False, tool_counts=tool_counts)
            if not warned_budget and budget.remaining() < low_water:
                warned_budget = True
                messages.append({"role": "user", "content": (
                    f"Token budget is nearly exhausted (~{budget.remaining()} left). Stop exploring now: "
                    "save the best artifact you can, run verify once, and call finish immediately."
                )})

        message = client.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=max_tokens,
        )
        _accumulate(usage, getattr(client, "last_usage", {}))
        if budget is not None:
            budget.add(getattr(client, "last_usage", {}))
        messages.append(message)

        calls = message.get("tool_calls") or []
        if not calls:
            idle_nudges += 1
            if idle_nudges >= 2:
                _log(name, f"no tool call twice; ending stage at step {step}")
                return StageResult(name, {}, usage, step, finished=False, tool_counts=tool_counts)
            messages.append({"role": "user", "content": "Use the tools to make progress, then call finish with a short JSON summary."})
            _prune(messages, max_history_tokens, prune_keep)
            continue
        idle_nudges = 0

        for call in calls:
            fn = call.get("function") or {}
            fname = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                messages.append(_tool_msg(call, f"Tool argument JSON error: {exc}"))
                continue
            tool_counts[fname] = tool_counts.get(fname, 0) + 1
            if fname == "finish":
                _log(name, f"finished at step {step}; tokens={usage['total_tokens']} tools={tool_counts}")
                return StageResult(name, args.get("result") or {}, usage, step, finished=True, tool_counts=tool_counts)
            executor = execs.get(fname)
            output = executor(args, ctx) if executor else f"Unknown tool: {fname}"
            messages.append(_tool_msg(call, output))

        _prune(messages, max_history_tokens, prune_keep)

    _log(name, f"hit max_steps={max_steps} without finish; tokens={usage['total_tokens']} tools={tool_counts}")
    return StageResult(name, {}, usage, max_steps, finished=False, tool_counts=tool_counts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_msg(call: dict[str, Any], content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.get("id"), "content": content}


def _accumulate(total: dict[str, int], last: dict[str, Any]) -> None:
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        total[k] += int(last.get(k, 0) or 0)


def _prune(messages: list[dict[str, Any]], max_history_tokens: int, prune_keep: int | None = None) -> None:
    """Elide tool results and assistant tool-call payloads by turn count and/or token estimation. Idempotent."""
    # 1. Prune by hard turn count if specified
    if prune_keep is not None:
        tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_idxs) > prune_keep:
            cutoff = tool_idxs[-prune_keep]
            for i in range(2, cutoff):
                m = messages[i]
                role = m.get("role")
                if role == "tool":
                    if m.get("content") != _ELIDED:
                        messages[i] = {**m, "content": _ELIDED}
                elif role == "assistant" and m.get("tool_calls"):
                    messages[i] = _shrink_assistant(m)

    # 2. Prune by estimated token size
    def _est_tokens(msgs: list[dict[str, Any]]) -> int:
        return sum(len(str(m)) // 4 for m in msgs)
    
    if _est_tokens(messages) <= max_history_tokens:
        return

    # Skip 0 and 1 (system and first user prompt)
    for i in range(2, len(messages)):
        if _est_tokens(messages) <= max_history_tokens:
            break
            
        m = messages[i]
        role = m.get("role")
        if role == "tool":
            if m.get("content") != _ELIDED:
                messages[i] = {**m, "content": _ELIDED}
        elif role == "assistant" and m.get("tool_calls"):
            messages[i] = _shrink_assistant(m)


def _shrink_assistant(msg: dict[str, Any]) -> dict[str, Any]:
    """Return msg with heavy tool-call argument payloads elided (or unchanged)."""
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
    """Elide large string values in a tool-call arguments JSON string."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        # Unparseable but possibly huge — stub it if long.
        return ('{"_elided": true}', True) if len(raw) > 400 else (raw, False)
    changed = False
    for k in _HEAVY_ARG_KEYS:
        v = args.get(k)
        if isinstance(v, str) and len(v) > _HEAVY_ARG_THRESHOLD:
            args[k] = f"[elided {len(v)} chars]"
            changed = True
    return (json.dumps(args), True) if changed else (raw, False)


def _log(stage: str, msg: str) -> None:
    print(f"[stage:{stage}] {msg}", file=sys.stderr)
