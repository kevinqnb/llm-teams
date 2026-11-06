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


_BARE = re.compile(r"[\s*_`\"'.,;:!]*([a-z]+)[\s*_`\"'.,;:!]*", re.IGNORECASE)


def parse_message_token(content: str, tokens: dict[str, str]) -> str | None:
    """Canonical text for a closed-vocabulary reply, or None.

    Accepts a reply that is exactly one token (any case, surrounding punctuation or markdown
    ignored), or else the last `MESSAGE: x` line. Anything longer than that is rejected: a token
    buried in prose is not a valid message.
    """
    bare = _BARE.fullmatch(content.strip())
    if bare and bare.group(1).lower() in tokens:
        return tokens[bare.group(1).lower()]
    matches = _MESSAGE.findall(content)
    return tokens.get(matches[-1].lower()) if matches else None
