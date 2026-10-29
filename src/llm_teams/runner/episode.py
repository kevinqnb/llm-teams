"""One episode as a generator that yields batches of generation requests.

    results = yield [requests]   # the runner batches these across episodes

Protocol per turn: act -> feedback -> discuss (connected agents only) -> act ...
"""

from __future__ import annotations

import dataclasses
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_teams.agents.base import Agent
from llm_teams.agents.view import AgentView, EpisodeState, ViewConfig, build_view
from llm_teams.comms.generators import build_graph
from llm_teams.comms.graph import CommGraph
from llm_teams.config import ExperimentConfig
from llm_teams.seeding import derive_rng, derive_seed
from llm_teams.tasks.base import Task
from llm_teams.types import DirectAction, GenerationRequest, GenerationResult, ParsedAction

EpisodeGen = Generator[list[GenerationRequest], list[GenerationResult], "EpisodeOutput"]


@dataclass
class EpisodeSpec:
    episode_id: int
    instance: dict[str, Any]
    task: Task
    agents: list[Agent]
    cfg: ExperimentConfig
    cond_hash: str


@dataclass
class EpisodeOutput:
    episode: dict[str, Any]
    turns: list[dict[str, Any]] = field(default_factory=list)
    generations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    graphs: dict[str, CommGraph] = field(default_factory=dict)  # last graphs, for notes files


def _gather(calls: dict[int, GenerationRequest | DirectAction]):
    """Send the LLM requests (if any) as one batch; scripted agents pass straight through."""
    llm = [a for a, c in calls.items() if isinstance(c, GenerationRequest)]
    results: dict[int, GenerationResult] = {}
    if llm:
        res = yield [calls[a] for a in llm]  # type: ignore[misc]
        results = dict(zip(llm, res, strict=True))
    return results


def run_episode(spec: EpisodeSpec) -> EpisodeGen:
    cfg, task, agents = spec.cfg, spec.task, spec.agents
    ep, seed, n = spec.episode_id, cfg.seed, len(agents)
    comms = cfg.comms
    vcfg = ViewConfig(
        know_n=cfg.agents.know_n,
        know_id=cfg.agents.know_id,
        reveal_group=comms.reveal_group,
        action_history=comms.action_graph.action_history,
        notes_history=comms.message_graph.notes_history,
        max_turns=cfg.episode.max_turns,
    )
    task.reset(spec.instance, derive_rng(seed, ep, "task"))
    state = EpisodeState(n=n, task=task)
    out = EpisodeOutput(episode={})
    tokens = invalid = n_messages = 0
    ref = task.reference_solution() or {}

    def make_graphs(turn: int) -> None:
        for attr, spec_g, tag in (
            ("action_graph", comms.action_graph, "act"),
            ("message_graph", comms.message_graph, "msg"),
        ):
            key = turn if spec_g.resample == "per_turn" else 0
            g = build_graph(spec_g.generator, n, spec_g.params, derive_rng(seed, ep, "graph", tag, key))
            setattr(state, attr, g)

    def talkers() -> list[int]:
        assert state.message_graph is not None
        return [i for i in range(n) if not state.message_graph.is_isolated(i)]

    def finalize(req: GenerationRequest, kind: str, agent: int, turn: int, rnd: int, tag: str):
        return dataclasses.replace(
            req,
            request_id=f"{spec.cond_hash}:{ep}:{kind}:{turn}:{rnd}:{agent}{tag}",
            seed=derive_seed(seed, ep, agent, turn, rnd, kind, tag),
            meta={"episode_id": ep, "turn": turn, "round": rnd, "agent": agent, "phase": kind},
        )

    def log_gen(req: GenerationRequest, res: GenerationResult, parsed: Any, valid: bool | None):
        nonlocal tokens
        tokens += res.prompt_tokens + res.completion_tokens
        out.generations.append({
            "request_id": req.request_id, "condition_hash": spec.cond_hash, "episode_id": ep,
            **req.meta, "messages": req.messages, "raw_output": res.text,
            "reasoning": res.reasoning, "content": res.content, "parsed": parsed, "valid": valid,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "latency": res.latency,
        })

    def discuss(phase: str, turn: int, rnd: int):
        nonlocal n_messages
        group = talkers()
        calls: dict[int, GenerationRequest | DirectAction] = {}
        views: dict[int, AgentView] = {}
        for i in group:
            views[i] = build_view(i, state, vcfg, phase, turn, rnd)
            req = agents[i].discussion_request(views[i])
            if req is not None:
                req = finalize(req, "discussion" if phase == "discuss" else "planning", i, turn, rnd, "")
                req.max_tokens = comms.max_message_tokens
                calls[i] = req
        results = yield from _gather(calls)
        for i, res in results.items():  # simultaneous: posted only after all are generated
            text = agents[i].parse_message(views[i], res).strip()
            state.noteboard.post(turn, rnd, i, text)
            n_messages += 1
            log_gen(calls[i], res, text, None)  # type: ignore[arg-type]

    # ---- optional planning discussion before turn 1
    make_graphs(0)
    for r in range(1, comms.planning_rounds + 1):
        yield from discuss("planning", 0, r)

    success, turns_done = False, 0
    ag_per_turn = comms.action_graph.resample == "per_turn"
    mg_per_turn = comms.message_graph.resample == "per_turn"
    graphs_turn1: dict[str, Any] = {}
    for turn in range(1, cfg.episode.max_turns + 1):
        make_graphs(turn)
        if turn == 1:
            graphs_turn1 = {"action": state.action_graph.to_dict(),  # type: ignore[union-attr]
                            "message": state.message_graph.to_dict()}  # type: ignore[union-attr]
        # ---- 1. act
        views = {i: build_view(i, state, vcfg, "act", turn) for i in range(n)}
        calls = {i: agents[i].action_request(views[i]) for i in range(n)}
        for i, c in calls.items():
            if isinstance(c, GenerationRequest):
                calls[i] = finalize(c, "action", i, turn, 0, "")
        results = yield from _gather(calls)
        parsed: dict[int, ParsedAction] = {}
        for i, c in calls.items():
            if isinstance(c, DirectAction):
                parsed[i] = ParsedAction(c.value, True)
            else:
                parsed[i] = agents[i].parse_action(views[i], results[i])
        # retry once, with a corrective message
        retries = {}
        for i, p in parsed.items():
            if not p.valid and i in results:
                rq = agents[i].retry_request(views[i], results[i])
                if rq is not None:
                    retries[i] = finalize(rq, "action", i, turn, 0, ":retry")
        retry_results = yield from _gather(retries)
        for i, rq in retries.items():
            log_gen(calls[i], results[i], None, False)  # type: ignore[arg-type]
            p2 = agents[i].parse_action(views[i], retry_results[i])
            parsed[i] = p2
            calls[i], results[i] = rq, retry_results[i]
        space = set(task.action_space().values)
        actions: dict[int, int] = {}
        for i in range(n):
            p = parsed[i]
            valid = p.valid and p.value in space
            if i in results:
                log_gen(calls[i], results[i], p.value, valid)  # type: ignore[arg-type]
            if valid:
                actions[i] = int(p.value)  # type: ignore[arg-type]
            else:
                invalid += 1
                actions[i] = _fallback(cfg.episode.invalid_policy, i, state, task,
                                       derive_rng(seed, ep, "fallback", i, turn))

        # ---- 2. feedback
        step = task.step(actions)
        state.actions[turn] = actions
        state.feedback[turn] = step.feedback
        turns_done = turn
        success = step.success
        row = {
            "condition_hash": spec.cond_hash, "episode_id": ep, "turn": turn,
            "actions": [actions[i] for i in range(n)], **step.info,
            "feedback": step.feedback[0] if step.feedback else None, "success": success,
            "action_graph": state.action_graph.to_dict() if ag_per_turn else None,
            "message_graph": state.message_graph.to_dict() if mg_per_turn else None,
            "n_messages_after": 0,
        }
        out.turns.append(row)
        if success or turn == cfg.episode.max_turns:
            break

        # ---- 3. discuss (connected agents only; feedback of this turn is already in state)
        before = n_messages
        for r in range(1, comms.discussion_rounds + 1):
            yield from discuss("discuss", turn, r)
        row["n_messages_after"] = n_messages - before

    ag, mg = comms.action_graph, comms.message_graph
    out.graphs = {"message_graph": state.message_graph}  # type: ignore[dict-item]
    out.notes = [nt.to_dict() for nt in state.noteboard.notes]
    out.episode = {
        "condition_hash": spec.cond_hash, "episode_id": ep, "task": cfg.task, "n": n,
        **{k: v for k, v in spec.instance.items() if k != "n"},
        "action_graph": {"generator": ag.generator, "resample": ag.resample, **ag.params},
        "message_graph": {"generator": mg.generator, "resample": mg.resample, **mg.params},
        "realized_action_graph": _graph_meta(state.action_graph),
        "realized_message_graph": _graph_meta(state.message_graph),
        "success": success, "turns": turns_done, "censored": not success,
        **ref, "total_tokens": tokens, "total_messages": n_messages, "invalid_count": invalid,
        "clique_members": (state.message_graph.meta.get("members") or []),
        "graphs_turn1": graphs_turn1, "agents_type": cfg.agents.type,
    }
    return out


def _graph_meta(g: CommGraph | None) -> dict[str, Any]:
    assert g is not None
    return {**g.meta, "n_edges": len(g.edges()), "components": [c for c in g.components() if len(c) > 1]}


def _fallback(policy: str, agent: int, state: EpisodeState, task: Task, rng: np.random.Generator) -> int:
    values = task.action_space().values
    if policy == "random":
        return int(rng.choice(values))
    if policy == "repeat_last":
        last = [state.actions[t][agent] for t in sorted(state.actions)]
        return last[-1] if last else int(rng.choice(values))
    return int(values[0])  # zero
