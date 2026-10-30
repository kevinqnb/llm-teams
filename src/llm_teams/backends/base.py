from __future__ import annotations

from abc import ABC, abstractmethod

from llm_teams.types import GenerationRequest, GenerationResult


class Backend(ABC):
    name = "unknown"

    @abstractmethod
    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Order-preserving batched generation."""


def split_reasoning(text: str, thinking: bool = False) -> tuple[str, str]:
    """Split raw output into (reasoning, content).

    Qwen's chat template may put the opening <think> in the prompt, so only the closing tag is
    reliable: everything before the last </think> is reasoning. In thinking mode an output with
    no </think> was truncated mid-reasoning, so it has no content.
    """
    if "</think>" in text:
        reasoning, content = text.rsplit("</think>", 1)
        return reasoning.replace("<think>", "").strip(), content.strip()
    if thinking or "<think>" in text:
        return text.replace("<think>", "").strip(), ""
    return "", text.strip()
