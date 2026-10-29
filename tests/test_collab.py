import json

import pytest

from llm_teams.backends.mock import MockBackend
from llm_teams.config import repo_root
from llm_teams.runner.episode import run_episode
from llm_teams.runner.run import build_spec, drive, run_experiment
from tests.helpers import make_cfg


def run(cfg, instance, fn=lambda r: "ANSWER: 0", eid=0):
    spec = build_spec(cfg, eid, instance, "h")
    backend, box = MockBackend(fn), []
    drive({eid: run_episode(spec)}, backend, box.append)
    return box[0], [r for b in backend.calls for r in b]


def ctrl_cfg(p, **kw):
    return make_cfg(**{"agents": {"type": "clique_controller"}, "episode": {"max_turns": 30},
                       "comms": {"message_graph": {"generator": "clique", "p": p}}, **kw})


def inst(n, t):
    return {"n": n, "target": t, "feedback": "direction"}


def test_full_clique_controller_matches_oracle():
    for n in (2, 5, 8, 16):  # n=1 has no clique (m <= 1 means no communication)
        for t in range(n + 1):
            ep = run(ctrl_cfg(1.0), inst(n, t))[0].episode
            assert ep["success"] and ep["turns"] == ep["oracle_turns"]


def test_controller_gets_worse_as_clique_shrinks():
    def mean_turns(p):
        total = 0
        for t in range(17):
            for e in range(4):
                total += run(ctrl_cfg(p), inst(16, t), eid=e)[0].episode["turns"]
        return total / (17 * 4)

    m_full, m_half, m_none = mean_turns(1.0), mean_turns(0.5), mean_turns(0.0)
    assert m_full < m_half < m_none


def test_controller_requires_reveal_group_and_sends_no_messages():
    ep = run(ctrl_cfg(0.5), inst(8, 3))[0].episode
    assert ep["total_messages"] == 0 and len(ep["clique_members"]) == 4
    with pytest.raises(ValueError, match="reveal_group"):
        run(ctrl_cfg(0.5, comms={"reveal_group": False,
                                 "message_graph": {"generator": "clique", "p": 0.5}}), inst(8, 3))


def llm_clique_cfg(p, **kw):
    return make_cfg(prompt="sum_to_target/default", agents={"type": "llm"},
                    **{"episode": {"max_turns": 3},
                       "comms": {"message_graph": {"generator": "clique", "p": p}}, **kw})


def test_llm_discussion_prompts_have_feedback_and_isolated_agents_are_silent():
    out, reqs = run(llm_clique_cfg(0.5), inst(4, 3),
                    lambda r: "We should all pick 1." if r.meta["phase"] == "discussion" else "ANSWER: 0")
    members = set(out.episode["clique_members"])
    disc = [r for r in reqs if r.meta["phase"] == "discussion"]
    assert disc and {r.meta["agent"] for r in disc} == members
    assert all("Too low" in r.messages[1]["content"] for r in disc)
    assert all("Do not write an ANSWER line" in r.messages[1]["content"] for r in disc)
    assert all(r.max_tokens == 200 for r in disc)
    # a member's later action prompt contains its clique-mates' messages, an outsider's does not
    late = [r for r in reqs if r.meta["phase"] == "action" and r.meta["turn"] == 2]
    for r in late:
        has = "We should all pick 1." in r.messages[1]["content"]
        assert has == (r.meta["agent"] in members)


def test_no_discussion_after_success_with_llm_agents():
    out, reqs = run(llm_clique_cfg(1.0), inst(4, 0))  # all-zero answers solve t=0 on turn 1
    assert out.episode["success"] and not [r for r in reqs if r.meta["phase"] == "discussion"]


def test_clique_smoke_dry_run_writes_readable_notes(tmp_path):
    cfg = repo_root() / "experiments/experiment-configs/sum_clique_smoke.yaml"
    dirs = run_experiment(str(cfg), {"dataset.seed": 5}, dry_run=True, results_root=tmp_path)
    assert len(dirs) == 2
    for d in dirs:
        eps = [json.loads(line) for line in (d / "episodes.jsonl").read_text().splitlines()]
        with_notes = [e for e in eps if e["total_messages"]]
        assert with_notes, "expected at least one unsolved episode with discussion"
        e = with_notes[0]
        f = d / "notes" / f"episode_{e['episode_id']}" / "after_turn_1_round_1.md"
        text = f.read_text()
        assert "# After turn 1, round 1" in text and "## Agent" in text and "read by" in text
        gens = [json.loads(line) for line in (d / "generations.jsonl").read_text().splitlines()]
        disc_agents = {g["agent"] for g in gens if g["phase"] == "discussion"}
        assert disc_agents <= {m for ep in eps for m in ep["clique_members"]}
