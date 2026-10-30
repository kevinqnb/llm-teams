from __future__ import annotations

from collections.abc import Callable

from llm_teams.backends.base import Backend, split_reasoning
from llm_teams.types import GenerationRequest, GenerationResult


class MockBackend(Backend):
    name = "mock"

    """Canned outputs: `fn(request) -> text` (default: always 'ANSWER: 0'). Records calls."""

    def __init__(self, fn: Callable[[GenerationRequest], str] | None = None) -> None:
        self.fn = fn or (lambda r: "ANSWER: 0")
        self.calls: list[list[GenerationRequest]] = []

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        self.calls.append(list(requests))
        out = []
        for r in requests:
            text = self.fn(r)
            reasoning, content = split_reasoning(text)
            out.append(GenerationResult(r.request_id, text, content, reasoning,
                                        completion_tokens=len(text.split()),
                                        finish_reason="stop"))
        return out
