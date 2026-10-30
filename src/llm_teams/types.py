"""Shared dataclasses that cross module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Action = int


@dataclass(frozen=True)
class ActionSpec:
    """Discrete action space; here `values` is the set of legal actions."""

    values: tuple[int, ...]


@dataclass
class StepResult:
    feedback: dict[int, dict[str, Any]]  # per-agent feedback
    success: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """A chat-style request. Agents fill `messages`; the episode fills the rest."""

    messages: list[dict[str, str]]
    request_id: str = ""
    seed: int = 0
    max_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    request_id: str
    text: str  # raw output, including any reasoning block
    content: str = ""  # text after reasoning is stripped
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    finish_reason: str | None = None  # 'stop' | 'length' | ...


@dataclass(frozen=True)
class DirectAction:
    """Returned by scripted agents in place of a GenerationRequest."""

    value: Action


@dataclass
class ParsedAction:
    value: Action | None
    valid: bool
    reasoning: str = ""
