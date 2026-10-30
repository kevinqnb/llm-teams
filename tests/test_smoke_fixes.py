"""Regressions from the first real-model smoke run."""

import json

import pytest

from llm_teams.backends.mock import MockBackend
from llm_teams.config import repo_root
from llm_teams.runner.episode import run_episode
from llm_teams.runner.run import build_spec, drive, run_experiment
from tests.helpers import make_cfg

INST = {"n": 4, "target": 3, "feedback": "direction"}


def run(cfg, instance, fn):
    spec = build_spec(cfg, 0, instance, "h")
    backend, box = MockBackend(fn), []
    drive({0: run_episode(spec)}, backend, box.append)
    return box[0], [r for b in backend.calls for r in b]


def llm_cfg(**kw):
    return make_cfg(prompt="sum_to_target/default", agents={"type": "llm"}, **kw)


def test_action_visibility_line_is_unambiguous():
    cfg = llm_cfg(**{"episode": {"max_turns": 3},
                     "comms": {"action_graph": {"generator": "complete"}}})
    _, reqs = run(cfg, INST, lambda r: "ANSWER: 1")
    acts = [r for r in reqs if r.meta["phase"] == "action" and r.meta["agent"] == 0]
    first, second = acts[0].messages[-1]["content"], acts[1].messages[-1]["content"]
    # turn 1: no choices exist yet, so the prompt must not claim any are visible
    assert "After each turn you will be able to see what agents 1, 2, 3 chose" in first
    assert "agent 1 chose" not in first
    assert "Choices you observed" in second and "Turn 1: agent 1 chose 1" in second
    assert "a few sentences at most" in first


def test_generations_log_finish_reason():
    cfg = llm_cfg(**{"episode": {"max_turns": 1}})
    out, _ = run(cfg, INST, lambda r: "ANSWER: 1")
    assert {g["finish_reason"] for g in out.generations} == {"stop"}


def test_resume_refuses_other_backend(tmp_path):
    cfg_path = str(repo_root() / "experiments/experiment-configs/sum_smoke.yaml")
    ov = {"sweep": {}, "comms.action_graph.p": 0.0}
    (rd,) = run_experiment(cfg_path, ov, dry_run=True, results_root=tmp_path)
    manifest = json.loads((rd / "manifest.json").read_text())
    assert manifest["backend"] == "mock"
    manifest["backend"] = "vllm"
    (rd / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="written by backend 'vllm'"):
        run_experiment(cfg_path, ov, dry_run=True, results_root=tmp_path)
