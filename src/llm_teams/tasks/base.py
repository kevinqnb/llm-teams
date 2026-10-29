from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from llm_teams.registry import Registry
from llm_teams.types import Action, ActionSpec, StepResult

TASKS: Registry[type[Task]] = Registry("task")


class Task(ABC):
    name: str

    @classmethod
    @abstractmethod
    def make_instances(cls, dataset: dict[str, Any]) -> list[dict[str, Any]]:
        """Deterministic list of instances from a dataset config."""

    @staticmethod
    @abstractmethod
    def n_agents(instance: dict[str, Any]) -> int: ...

    @abstractmethod
    def reset(self, instance: dict[str, Any], rng: np.random.Generator) -> None: ...

    @abstractmethod
    def describe(self, agent_id: int) -> dict[str, Any]:
        """Static task info for prompts. Must not reveal hidden state."""

    @abstractmethod
    def action_space(self) -> ActionSpec: ...

    @abstractmethod
    def step(self, actions: dict[int, Action]) -> StepResult: ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Full state for logging (may include hidden state)."""

    @abstractmethod
    def reference_solution(self) -> dict[str, Any] | None: ...
