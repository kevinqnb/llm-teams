"""Test helpers: an LLM-style agent whose prompt is the JSON dump of its view."""

from __future__ import annotations

import json
from typing import Any

from llm_teams.agents.base import Agent
from llm_teams.agents.view import AgentView
from llm_teams.config import ExperimentConfig
from llm_teams.runner.episode import EpisodeOutput, run_episode
from llm_teams.runner.run import build_spec, drive
from llm_teams.types import DirectAction, GenerationRequest, GenerationResult, ParsedAction


class ProbeAgent(Agent):
    """Prompt = JSON of the view, so tests can inspect exactly what the agent was shown."""

    def __init__(self, agent_id: int, talk: bool = True) -> None:
        super().__init__(agent_id)
        self.talk = talk

    def action_request(self, view: AgentView) -> GenerationRequest | DirectAction:
        return GenerationRequest([{"role": "user", "content": json.dumps(view.to_dict())}])

    def parse_action(self, view: AgentView, result: GenerationResult) -> ParsedAction:
        t = result.content.strip()
        if t in ("0", "1"):
            return ParsedAction(int(t), True)
        return ParsedAction(None, False)

    def retry_request(self, view, result):
        return GenerationRequest([{"role": "user", "content": "retry: answer 0 or 1"}])

    def discussion_request(self, view: AgentView) -> GenerationRequest | None:
        if not self.talk:
            return None
        return GenerationRequest([{"role": "user", "content": json.dumps(view.to_dict())}])

    def parse_message(self, view, result):
        return f"msg from {self.agent_id} t{view.turn} r{view.round}"


def make_cfg(**kw: Any) -> ExperimentConfig:
    base: dict[str, Any] = {
        "name": "t", "task": "sum_to_target",
        "dataset": {"name": "d", "n": 4, "num_episodes": 4, "seed": 0},
        "agents": {"type": "random"}, "episode": {"max_turns": 5},
    }
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return ExperimentConfig.model_validate(base)


def run_with_probes(cfg, instance, backend, eid: int = 0, talk: bool = True) -> EpisodeOutput:
    """Run one episode with ProbeAgents against `backend`."""
    spec = build_spec(cfg, eid, instance, "h")
    spec.agents = [ProbeAgent(i, talk) for i in range(instance["n"])]
    box: list[EpisodeOutput] = []
    drive({eid: run_episode(spec)}, backend, box.append)
    return box[0]
