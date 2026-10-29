"""LLM-backed agent: a separate conversation per agent, rendered fresh from its view each turn."""

from __future__ import annotations

import numpy as np

from llm_teams.agents.base import AGENTS, Agent
from llm_teams.agents.parse import parse_answer
from llm_teams.agents.prompts import PromptTemplates, load_prompts
from llm_teams.agents.view import AgentView
from llm_teams.config import ExperimentConfig
from llm_teams.types import GenerationRequest, GenerationResult, ParsedAction

ALLOWED = {0, 1}  # sum_to_target's action space; generalize when a second task is added


class LLMAgent(Agent):
    def __init__(self, agent_id: int, prompts: PromptTemplates) -> None:
        super().__init__(agent_id)
        self.prompts = prompts

    def _messages(self, field: str, view: AgentView) -> list[dict[str, str]]:
        v = view.to_dict()
        return [
            {"role": "system", "content": self.prompts.render("system", v)},
            {"role": "user", "content": self.prompts.render(field, v)},
        ]

    def action_request(self, view: AgentView) -> GenerationRequest:
        return GenerationRequest(self._messages("action", view))

    def parse_action(self, view: AgentView, result: GenerationResult) -> ParsedAction:
        value = parse_answer(result.content, ALLOWED)
        return ParsedAction(value, value is not None, result.reasoning)

    def retry_request(self, view: AgentView, result: GenerationResult) -> GenerationRequest:
        # the earlier answer is replayed without its reasoning trace
        msgs = self._messages("action", view)
        msgs.append({"role": "assistant", "content": result.content or "(no answer)"})
        msgs.append({"role": "user", "content": self.prompts.render("retry", view.to_dict())})
        return GenerationRequest(msgs)

    def discussion_request(self, view: AgentView) -> GenerationRequest:
        return GenerationRequest(self._messages("discussion", view))

    def parse_message(self, view: AgentView, result: GenerationResult) -> str:
        return result.content.strip() or "(no message)"  # reasoning is never posted


@AGENTS.register("llm")
def make_llm(n: int, cfg: ExperimentConfig, rng: np.random.Generator) -> list[Agent]:
    if cfg.prompt is None:
        raise ValueError("llm agents need a `prompt:` in the experiment config")
    if cfg.agents.memory != "structured":
        raise NotImplementedError("agents.memory=transcript is not implemented yet")
    prompts = load_prompts(cfg.prompt)
    return [LLMAgent(i, prompts) for i in range(n)]
