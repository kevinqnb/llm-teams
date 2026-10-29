import json
from pathlib import Path

import pytest

from llm_teams.agents.parse import parse_answer
from llm_teams.backends.base import split_reasoning
from llm_teams.backends.mock import MockBackend
from llm_teams.config import repo_root
from llm_teams.runner.episode import run_episode
from llm_teams.runner.run import build_spec, drive, run_experiment
from tests.helpers import make_cfg

OK = {0, 1}


@pytest.mark.parametrize("text,expected", [
    ("ANSWER: 1", 1),
    ("blah\nANSWER: 0\n", 0),
    ("answer:   1  ", 1),
    ("Answer : **1**", 1),
    ("ANSWER: 0\nactually\nANSWER: 1", 1),  # last match wins
    ("ANSWER: 2", None),
    ("ANSWER: maybe", None),
    ("no final line", None),
    ("", None),
])
def test_parse_answer(text, expected):
    assert parse_answer(text, OK) == expected


def test_reasoning_split_ignores_answers_inside_think():
    r, c = split_reasoning("hmm ANSWER: 1 </think>\nANSWER: 0")
    assert "ANSWER: 1" in r and parse_answer(c, OK) == 0
    r, c = split_reasoning("<think>ANSWER: 1</think>no answer here")
    assert parse_answer(c, OK) is None
    # truncated thinking has no content, so a stray ANSWER inside the reasoning is not used
    assert split_reasoning("ANSWER: 1 but wait", thinking=True) == ("ANSWER: 1 but wait", "")
    assert split_reasoning("ANSWER: 1") == ("", "ANSWER: 1")


def run(cfg, instance, fn):
    spec = build_spec(cfg, 0, instance, "h")
    backend, box = MockBackend(fn), []
    drive({0: run_episode(spec)}, backend, box.append)
    return box[0], [r for b in backend.calls for r in b]


def llm_cfg(**kw):
    return make_cfg(prompt="sum_to_target/default", agents={"type": "llm"}, **kw)


INST = {"n": 4, "target": 3, "feedback": "direction"}
CLIQUE = {"generator": "clique", "m": 3}


def test_llm_agents_end_to_end_with_thinking_stripped():
    cfg = llm_cfg(**{"episode": {"max_turns": 3}, "comms": {"message_graph": CLIQUE}})

    def fn(r):
        if r.meta["phase"] == "discussion":
            return "<think>SECRET</think>Let's all pick 1."
        return "<think>SECRET ANSWER: 0</think>ANSWER: 1"

    out, reqs = run(cfg, INST, fn)
    assert out.episode["success"] is False or out.episode["turns"] <= 3
    assert out.episode["invalid_count"] == 0
    assert all("SECRET" not in m["content"] for r in reqs for m in r.messages)
    assert out.generations[0]["reasoning"].startswith("SECRET")
    assert all(n["text"] == "Let's all pick 1." for n in out.notes)
    assert out.turns[0]["actions"] == [1, 1, 1, 1]


def test_prompts_reflect_group_membership_and_hide_target():
    cfg = llm_cfg(**{"episode": {"max_turns": 2}, "comms": {"message_graph": CLIQUE}})
    out, reqs = run(cfg, INST, lambda r: "ANSWER: 0")
    members = set(out.episode["clique_members"])
    first = {r.meta["agent"]: r for r in reqs if r.meta["turn"] == 1 and r.meta["phase"] == "action"}
    for a, r in first.items():
        sys_prompt = r.messages[0]["content"]
        if a in members:
            assert "exchange short messages" in sys_prompt and "1 agent is outside" in sys_prompt
            assert "group with" in sys_prompt
        else:
            assert "you cannot send or receive messages" in sys_prompt
        assert "target is 3" not in sys_prompt and f"ID is {a}" in sys_prompt


def test_retry_then_policy_and_invalid_logging():
    cfg = llm_cfg(**{"episode": {"max_turns": 1, "invalid_policy": "zero"}})
    seen = []

    def fn(r):
        seen.append(r.request_id)
        return "I refuse"

    out, reqs = run(cfg, INST, fn)
    assert out.episode["invalid_count"] == 4
    assert sum(r.request_id.endswith(":retry") for r in reqs) == 4
    retry = next(r for r in reqs if r.request_id.endswith(":retry"))
    assert [m["role"] for m in retry.messages] == ["system", "user", "assistant", "user"]
    assert all(g["valid"] is False for g in out.generations)


def test_retry_recovers():
    cfg = llm_cfg(**{"episode": {"max_turns": 1}})
    fn = lambda r: "ANSWER: 1" if r.request_id.endswith(":retry") else "hmm"  # noqa: E731
    out, _ = run(cfg, INST, fn)
    assert out.episode["invalid_count"] == 0 and out.turns[0]["actions"] == [1] * 4


def test_missing_prompt_is_an_error():
    cfg = make_cfg(agents={"type": "llm"})
    with pytest.raises(ValueError, match="prompt"):
        build_spec(cfg, 0, INST, "h")


def test_smoke_config_dry_run(tmp_path: Path):
    cfg = repo_root() / "experiments/experiment-configs/sum_smoke.yaml"
    dirs = run_experiment(str(cfg), {}, dry_run=True, results_root=tmp_path)
    assert len(dirs) == 2
    for d in dirs:
        gens = [json.loads(line) for line in (d / "generations.jsonl").read_text().splitlines()]
        assert gens and {"request_id", "messages", "raw_output", "parsed", "valid", "phase",
                         "prompt_tokens", "latency"} <= set(gens[0])
    ps = sorted(json.loads((d / "episodes.jsonl").read_text().splitlines()[0])["realized_action_graph"]["k"]
                for d in dirs)
    assert ps == [0, 3]  # p=0 -> no visibility, p=1 -> everyone sees all 3 others
