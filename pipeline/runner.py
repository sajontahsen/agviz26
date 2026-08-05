"""Stage runner: one short, self-contained LLM session per pipeline stage.

Each stage gets a fresh message list (system + user), a chosen subset of tools,
and a model. History never crosses stage boundaries — stages hand work forward
through *files* in the workdir, not through conversation. That is what keeps the
multi-stage pipeline inside the token budget.

Within a stage we still prune: only the last few tool results are kept verbatim;
older ones are elided to short stubs so a long build/repair loop doesn't balloon
the input context. Assistant tool-call messages are always kept intact (the API
requires each tool result to follow its call), we only shrink tool *content*.
"""
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


@dataclass
class StageResult:
    name: str
    result: dict[str, Any]
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    steps: int = 0
    finished: bool = False


def run_stage(
    *,
    name: str,
    system_prompt: str,
    user_prompt: str,
    tool_names: list[str],
    model: str,
    ctx: ToolContext,
    client: Any,
    purpose: str = "generation",
    max_steps: int = 24,
    prune_keep: int = 4,
    max_tokens: int = 8192,
) -> StageResult:
    schemas, execs = build_toolset(tool_names)
    tools = [*schemas, _FINISH_SCHEMA]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    idle_nudges = 0

    for step in range(1, max_steps + 1):
        message = client.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=max_tokens,
        )
        _accumulate(usage, getattr(client, "last_usage", {}))
        messages.append(message)

        calls = message.get("tool_calls") or []
        if not calls:
            idle_nudges += 1
            if idle_nudges >= 2:
                _log(name, f"no tool call twice; ending stage at step {step}")
                return StageResult(name, {}, usage, step, finished=False)
            messages.append({"role": "user", "content": "Use the tools to make progress, then call finish with a short JSON summary."})
            _prune(messages, prune_keep)
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
            if fname == "finish":
                _log(name, f"finished at step {step}; tokens={usage['total_tokens']}")
                return StageResult(name, args.get("result") or {}, usage, step, finished=True)
            executor = execs.get(fname)
            output = executor(args, ctx) if executor else f"Unknown tool: {fname}"
            messages.append(_tool_msg(call, output))

        _prune(messages, prune_keep)

    _log(name, f"hit max_steps={max_steps} without finish; tokens={usage['total_tokens']}")
    return StageResult(name, {}, usage, max_steps, finished=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_msg(call: dict[str, Any], content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.get("id"), "content": content}


def _accumulate(total: dict[str, int], last: dict[str, Any]) -> None:
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        total[k] += int(last.get(k, 0) or 0)


def _prune(messages: list[dict[str, Any]], keep: int) -> None:
    """Elide the content of all but the last `keep` tool messages in place."""
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idxs[:-keep] if keep else tool_idxs:
        if messages[i].get("content") != _ELIDED:
            messages[i] = {**messages[i], "content": _ELIDED}


def _log(stage: str, msg: str) -> None:
    print(f"[stage:{stage}] {msg}", file=sys.stderr)
