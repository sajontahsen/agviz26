"""LLM call routing for the example agent.

How model calls are sent: locally via the OpenAI API (your OPENAI_API_KEY), or
in an arena cloud job via the arena broker (no provider key needed). Token usage
and per-job budgets are recorded and enforced server-side by the arena — editing
this file changes how requests are *sent*, not what the arena records.

You usually don't need to touch this; it lives here (separate from your agent
logic in example_agent.py) so you can see and customize the transport — retries,
streaming, timeouts — if you want. example_agent.py imports make_llm_client().
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def _usage_dict(usage: Any) -> dict[str, Any]:
    """Normalize a provider usage object into {input,output,total}_tokens (best effort across backends)."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        raw = usage
    elif hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    else:
        raw = {k: getattr(usage, k) for k in dir(usage) if not k.startswith("_") and isinstance(getattr(usage, k, None), int)}
    prompt = raw.get("prompt_tokens") or raw.get("input_tokens") or 0
    completion = raw.get("completion_tokens") or raw.get("output_tokens") or 0
    total = raw.get("total_tokens") or (prompt + completion)
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


class OpenAIChatClient:
    def __init__(self) -> None:
        self.client = OpenAI()
        # Populated after each create() so callers can meter token spend.
        self.last_usage: dict[str, Any] = {}

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        # gpt-5 / o-series reasoning models reject `max_tokens` and require
        # `max_completion_tokens`; classic chat models take `max_tokens`.
        token_param = "max_completion_tokens" if model.startswith(("gpt-5", "o1", "o3", "o4")) else "max_tokens"
        kwargs: dict[str, Any] = {"model": model, "messages": messages, token_param: max_tokens}
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        response = self.client.chat.completions.create(**kwargs)
        self.last_usage = _usage_dict(getattr(response, "usage", None))
        return response.choices[0].message.model_dump(exclude_none=True)


class ArenaChatClient:
    def __init__(self, purpose: str) -> None:
        from vis_arena_sdk import VisArenaClient

        self.job_id = os.environ["VIS_ARENA_JOB_ID"]
        self.purpose = purpose
        self.client = VisArenaClient(
            base_url=os.environ.get("VIS_ARENA_SERVER_URL", "http://host.docker.internal:8000"),
            token=os.environ["VIS_ARENA_API_TOKEN"],
            # Bedrock can take several minutes when returning a large
            # tool call that writes the final HTML artifact.
            timeout=600.0,
        )
        # Populated after each create() so callers can meter token spend.
        self.last_usage: dict[str, Any] = {}

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        response = self.client.create_llm_message(
            job_id=self.job_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            model=model,
            purpose=self.purpose,
            max_tokens=max_tokens,
        )
        self.last_usage = _usage_dict(getattr(response, "usage", None))
        return response.message


def make_llm_client(purpose: str) -> OpenAIChatClient | ArenaChatClient:
    # Cloud: the arena worker injects VIS_ARENA_API_TOKEN + VIS_ARENA_JOB_ID and
    # the agent routes model calls through the arena backend (no provider key
    # needed). Local: you set OPENAI_API_KEY.
    if os.environ.get("VIS_ARENA_API_TOKEN") and os.environ.get("VIS_ARENA_JOB_ID") and not os.environ.get("OPENAI_API_KEY"):
        return ArenaChatClient(purpose)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "  Local testing: export OPENAI_API_KEY=sk-... and re-run.\n"
            "  Submitting:   run `vis-arena submit .`; the arena provides cloud models."
        )
    return OpenAIChatClient()
