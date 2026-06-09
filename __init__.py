"""
turn_timestamp — pre_llm_call plugin that injects the current time
into each turn's user message.

Uses the hermes_time module for timezone-aware timestamps consistent
with Hermes's own timestamp formatting.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def on_pre_llm_call(
    session_id: str,
    user_message: str,
    conversation_history: list,
    is_first_turn: bool,
    model: str,
    platform: str,
    **kwargs: Any,
) -> dict:
    """Return current time as context to inject into the user message.

    Fires once per turn, before the LLM API call.  The returned
    ``{"context": ...}`` dict is appended to the user message so the
    model sees the time on every request — without touching the cached
    system prompt (prefix cache stays stable).
    """
    try:
        from hermes_time import now as _hermes_now

        now = _hermes_now()
        timestamp_line = now.strftime("%A, %B %d, %Y %H:%M:%S %Z")

        return {
            "context": f"[Current time: {timestamp_line}]"
        }
    except Exception as exc:
        logger.warning("turn_timestamp plugin failed: %s", exc)
        return {"context": "[time unavailable]"}


def register(ctx) -> None:
    """Register the pre_llm_call hook."""
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
