"""Parsing of agent outputs. Only text after the reasoning block is ever parsed."""

from __future__ import annotations

import re

_ANSWER = re.compile(r"answer\s*:\s*[*_`\"'\s]*([^\s*_`\"'.,;]+)", re.IGNORECASE)


def parse_answer(content: str, allowed: set[int]) -> int | None:
    """Value of the last `ANSWER: x` in `content`, or None if absent or not an allowed int."""
    matches = _ANSWER.findall(content)
    if not matches:
        return None
    try:
        value = int(matches[-1])
    except ValueError:
        return None
    return value if value in allowed else None


_MESSAGE = re.compile(r"message\s*:\s*[*_`\"'\s]*([a-z]+)", re.IGNORECASE)


def parse_message_token(content: str, tokens: dict[str, str]) -> str | None:
    """Canonical text for the last `MESSAGE: x` in `content`, or None if absent or not a token."""
    matches = _MESSAGE.findall(content)
    return tokens.get(matches[-1].lower()) if matches else None
