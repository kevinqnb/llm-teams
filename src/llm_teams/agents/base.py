from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from llm_teams.agents.view import AgentView
from llm_teams.config import ExperimentConfig
from llm_teams.registry import Registry
from llm_teams.types import DirectAction, GenerationRequest, GenerationResult, ParsedAction


class Agent(ABC):
    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def action_request(self, view: AgentView) -> GenerationRequest | DirectAction:
        """Either a request for the backend or (scripted agents) the action itself."""

    def parse_action(self, view: AgentView, result: GenerationResult) -> ParsedAction:
        raise NotImplementedError(f"{type(self).__name__} produces direct actions only")

    def retry_request(self, view: AgentView, result: GenerationResult) -> GenerationRequest | None:
        """Corrective follow-up after an invalid output; None = no retry."""
        return None

    def discussion_request(self, view: AgentView) -> GenerationRequest | None:
        """Message-writing request; None = this agent never posts."""
        return None

    def parse_message(self, view: AgentView, result: GenerationResult) -> str:
        return result.content


# A team factory builds all n agents of one episode from the full config. rng is the
# episode's agent-team rng.
TeamFactory = Callable[[int, ExperimentConfig, np.random.Generator], list[Agent]]
AGENTS: Registry[TeamFactory] = Registry("agent type")
