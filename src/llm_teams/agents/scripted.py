"""Scripted (non-LLM) agents. They use the normal Agent interface."""

from __future__ import annotations

import numpy as np

from llm_teams.agents.base import AGENTS, Agent
from llm_teams.agents.view import AgentView
from llm_teams.config import ExperimentConfig
from llm_teams.seeding import derive_rng
from llm_teams.tasks.sum_to_target import bisect_guess, update_interval
from llm_teams.types import DirectAction


def next_guess(n: int, feedback_history: list[dict]) -> int:
    """Bisection guess given past direction feedback, assuming past guesses were bisection too."""
    lo, hi = 0, n
    for fb in feedback_history:
        lo, hi = update_interval(lo, hi, bisect_guess(lo, hi), fb["result"])
    return bisect_guess(lo, hi)


class RandomAgent(Agent):
    def __init__(self, agent_id: int, rng: np.random.Generator, p: float = 0.5) -> None:
        super().__init__(agent_id)
        self.rng, self.p = rng, p

    def action_request(self, view: AgentView) -> DirectAction:
        return DirectAction(int(self.rng.random() < self.p))


class OracleAgent(Agent):
    """Centralized optimal bisection: a controller (knowing n and each agent's index by
    construction, not through the view) assigns 1 to the first `guess` agents."""

    def __init__(self, agent_id: int, n: int) -> None:
        super().__init__(agent_id)
        self.n = n

    def action_request(self, view: AgentView) -> DirectAction:
        return DirectAction(int(self.agent_id < next_guess(self.n, view.feedback_history)))


class IdBisectionAgent(Agent):
    """Decentralized, zero-communication bisection. Needs to know n and its own id.

    Every agent reconstructs the shared interval from the direction feedback alone (the team's
    sums are deterministic), then plays 1 iff id < guess.
    """

    def action_request(self, view: AgentView) -> DirectAction:
        if view.n is None or view.self_id is None:
            raise ValueError("id_bisection requires know_n and know_id")
        return DirectAction(int(view.self_id < next_guess(view.n, view.feedback_history)))


@AGENTS.register("random")
def make_random(n: int, cfg: ExperimentConfig, rng: np.random.Generator) -> list[Agent]:
    p = float((cfg.agents.model_extra or {}).get("p", 0.5))
    seeds = rng.integers(0, 2**32, size=n)
    return [RandomAgent(i, np.random.default_rng(int(seeds[i])), p) for i in range(n)]


@AGENTS.register("oracle")
def make_oracle(n: int, cfg: ExperimentConfig, rng: np.random.Generator) -> list[Agent]:
    return [OracleAgent(i, n) for i in range(n)]


@AGENTS.register("id_bisection")
def make_id_bisection(n: int, cfg: ExperimentConfig, rng: np.random.Generator) -> list[Agent]:
    return [IdBisectionAgent(i) for i in range(n)]


class CliqueControllerAgent(Agent):
    """Reference for collaboration: the clique coordinates perfectly, outsiders act randomly.

    Upper-bound baseline, not a realistic agent. Outsiders play i.i.d. Bernoulli(outsider_p) from a
    shared deterministic stream, and clique members are assumed to know it (oracle access to the
    outsiders' actions). Members pick how many of them play 1 so the team sum is as close as
    possible to the bisection guess, clamped to [0, m]. Everyone in the clique recomputes the same
    plan from the group (view.group, so reveal_group must be on) and the shared feedback.
    """

    def __init__(self, agent_id: int, n: int, seed: int, outsider_p: float) -> None:
        super().__init__(agent_id)
        self.n, self.seed, self.outsider_p = n, seed, outsider_p
        self._cache: dict[tuple[int, int], int] = {}

    def outsider_action(self, agent: int, turn: int) -> int:
        key = (agent, turn)
        if key not in self._cache:
            self._cache[key] = int(derive_rng(self.seed, "outsider", agent, turn).random() < self.outsider_p)
        return self._cache[key]

    def action_request(self, view: AgentView) -> DirectAction:
        members = view.group["members"] if view.group else None
        if members is None:
            raise ValueError("clique_controller requires comms.reveal_group: true")
        outsiders = [i for i in range(self.n) if i not in members]
        if not members or self.agent_id in outsiders:
            return DirectAction(self.outsider_action(self.agent_id, view.turn))
        m, lo, hi = len(members), 0, self.n
        for turn, fb in enumerate(view.feedback_history, start=1):
            o = sum(self.outsider_action(i, turn) for i in outsiders)
            c = min(max(bisect_guess(lo, hi) - o, 0), m)
            lo, hi = update_interval(lo, hi, c + o, fb["result"])
        o = sum(self.outsider_action(i, view.turn) for i in outsiders)
        c = min(max(bisect_guess(lo, hi) - o, 0), m)
        return DirectAction(int(members.index(self.agent_id) < c))


@AGENTS.register("clique_controller")
def make_clique_controller(n: int, cfg: ExperimentConfig, rng: np.random.Generator) -> list[Agent]:
    extra = cfg.agents.model_extra or {}
    seed = int(rng.integers(0, 2**62))
    return [CliqueControllerAgent(i, n, seed, float(extra.get("outsider_p", 0.5))) for i in range(n)]
