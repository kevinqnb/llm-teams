import json

import pytest

from llm_teams.backends.mock import MockBackend
from llm_teams.runner.episode import run_episode
from llm_teams.runner.run import build_spec, drive
from tests.helpers import make_cfg, run_with_probes

CLIQUE = {"generator": "clique", "m": 2, "resample": "fixed"}


def inst(n=4, target=2):
    return {"n": n, "target": target, "feedback": "direction"}


def action_fn(r):
    if r.meta["phase"] in ("discussion", "planning"):
        return "<think>SECRET-REASONING</think>I will play 1"
    return f"<think>SECRET-REASONING</think>{1 if r.meta['agent'] % 2 == 0 else 0}"


def run(cfg_kw, instance, fn=action_fn, **kw):
    backend = MockBackend(fn)
    out = run_with_probes(make_cfg(**cfg_kw), instance, backend, **kw)
    reqs = [r for batch in backend.calls for r in batch]
    return out, reqs


def view_of(req):
    return json.loads(req.messages[0]["content"])


# ------------------------------------------------------------- turn ordering


def test_discussion_only_after_feedback_not_after_success_or_final_turn():
    cfg = {"episode": {"max_turns": 3}, "comms": {"message_graph": CLIQUE}}
    out, reqs = run(cfg, inst(target=3))  # sum is 2 every turn -> never solved
    phases = [(r.meta["phase"], r.meta["turn"]) for r in reqs]
    order = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert order == [("action", 1), ("discussion", 1), ("action", 2), ("discussion", 2), ("action", 3)]
    assert out.episode["turns"] == 3 and out.episode["censored"]

    out, reqs = run(cfg, inst(target=2))  # solved on turn 1
    assert out.episode["success"] and out.episode["turns"] == 1
    assert {r.meta["phase"] for r in reqs} == {"action"}


def test_isolated_agents_never_discuss_and_prompts_have_latest_feedback():
    cfg = {"episode": {"max_turns": 3}, "comms": {"message_graph": CLIQUE}}
    out, reqs = run(cfg, inst(target=3))
    members = set(out.episode["clique_members"])
    assert len(members) == 2
    disc = [r for r in reqs if r.meta["phase"] == "discussion"]
    assert disc and {r.meta["agent"] for r in disc} == members
    for r in disc:
        v = view_of(r)
        assert len(v["feedback_history"]) == r.meta["turn"]
        assert v["feedback_history"][-1]["result"] == "too_low"
    assert out.episode["total_messages"] == len(disc)
    assert [t["n_messages_after"] for t in out.turns] == [2, 2, 0]


def test_no_comms_means_no_discussion_requests():
    _, reqs = run({"episode": {"max_turns": 3}}, inst(target=3))
    assert {r.meta["phase"] for r in reqs} == {"action"}


def test_planning_rounds_happen_before_turn_one():
    cfg = {"comms": {"message_graph": CLIQUE, "planning_rounds": 1}}
    _, reqs = run(cfg, inst(target=2))
    assert reqs[0].meta["phase"] == "planning" and reqs[0].meta["turn"] == 0


# --------------------------------------------------------- information access


def test_views_never_leak():
    cfg = {
        "episode": {"max_turns": 4},
        "comms": {
            "action_graph": {"generator": "random_k", "k": 1, "resample": "fixed"},
            "message_graph": CLIQUE,
            "discussion_rounds": 2,
        },
    }
    out, reqs = run(cfg, inst(n=6, target=5))
    actions = {t["turn"]: t["actions"] for t in out.turns}
    members = set(out.episode["clique_members"])
    for r in reqs:
        text = r.messages[0]["content"]
        assert "SECRET-REASONING" not in text  # reasoning never shown to anyone
        v = view_of(r)
        assert "target" not in v and "target" not in v["task_info"]
        me, turn, phase = r.meta["agent"], r.meta["turn"], r.meta["phase"]
        done = turn - 1 if phase == "action" else turn
        assert len(v["own_actions"]) == done and len(v["feedback_history"]) == done
        # only action-neighbours' past (never same-turn) actions
        for t, seen in v["visible_actions"].items():
            assert int(t) <= done
            assert set(map(int, seen)) <= set(v["action_neighbors"])
            assert all(actions[int(t)][int(j)] == a for j, a in seen.items())
        assert len(v["action_neighbors"]) == 1
        # notes: only from message neighbours (or self), never from outsiders
        for note in v["notes"]:
            assert note["author"] in members and me in members
            assert note["author"] == me or note["author"] in v["message_neighbors"]
            same_turn_ok = phase == "discussion" and note["after_turn"] == turn
            assert note["after_turn"] < turn or same_turn_ok
            if phase == "discussion" and note["after_turn"] == turn:
                assert note["round"] < r.meta["round"]  # never same-round messages
        if me not in members:
            assert v["notes"] == [] and v["message_neighbors"] == []
    # round 1 sees no same-turn notes; round 2 does
    r1 = [view_of(r) for r in reqs if r.meta["phase"] == "discussion" and r.meta["round"] == 1]
    r2 = [view_of(r) for r in reqs if r.meta["phase"] == "discussion" and r.meta["round"] == 2]
    assert all(all(n["after_turn"] < v["turn"] for n in v["notes"]) for v in r1)
    assert any(any(n["after_turn"] == v["turn"] for n in v["notes"]) for v in r2)


def test_know_flags_hide_n_and_id():
    cfg = {"agents": {"type": "random", "know_n": False, "know_id": False}}
    _, reqs = run(cfg, inst())
    v = view_of(reqs[0])
    assert v["n"] is None and v["self_id"] is None


def test_notes_history_last_turn():
    cfg = {
        "episode": {"max_turns": 4},
        "comms": {"message_graph": {**CLIQUE, "notes_history": "last_turn"}},
    }
    _, reqs = run(cfg, inst(target=3))
    late = [view_of(r) for r in reqs if r.meta["phase"] == "action" and r.meta["turn"] == 4]
    assert all({n["after_turn"] for n in v["notes"]} <= {3} for v in late)


def test_invalid_retry_then_fallback():
    calls = {"n": 0}

    def fn(r):
        calls["n"] += 1
        return "gibberish"

    cfg = {"episode": {"max_turns": 2, "invalid_policy": "zero"}}
    out, _ = run(cfg, inst(target=1), fn=fn)
    assert out.episode["invalid_count"] == 8  # 4 agents x 2 turns
    assert [t["actions"] for t in out.turns] == [[0] * 4] * 2
    valid_flags = [g["valid"] for g in out.generations]
    assert valid_flags and not any(valid_flags)
    assert any(g["request_id"].endswith(":retry") for g in out.generations)


def test_notes_files_and_graph_logged(tmp_path):
    from llm_teams.runner.run import RunDir

    cfg = make_cfg(**{"episode": {"max_turns": 3}, "comms": {"message_graph": CLIQUE}})
    out = run_with_probes(cfg, inst(target=3), MockBackend(action_fn))
    rd = RunDir(cfg, "h", tmp_path)
    rd.write(out)
    f = rd.path / "notes" / "episode_0" / "after_turn_1_round_1.md"
    assert f.exists() and "Agent" in f.read_text()
    assert out.episode["graphs_turn1"]["message"]["edges"]


# ------------------------------------------------------------- lockstep == serial


def test_lockstep_equals_serial():
    cfg = make_cfg(**{"episode": {"max_turns": 4}, "comms": {"message_graph": CLIQUE,
                    "action_graph": {"generator": "random_k", "p": 0.5, "resample": "per_turn"}}})
    fn = lambda r: str(r.seed % 2) if r.meta["phase"] == "action" else "hi"  # noqa: E731
    instances = [inst(target=t) for t in (0, 1, 3, 4, 2)]

    def specs():
        from tests.helpers import ProbeAgent

        out = []
        for e, i in enumerate(instances):
            s = build_spec(cfg, e, i, "h")
            s.agents = [ProbeAgent(a) for a in range(4)]
            out.append(s)
        return out

    serial, lock = [], []
    for s in specs():
        drive({s.episode_id: run_episode(s)}, MockBackend(fn), serial.append)
    backend = MockBackend(fn)
    drive({s.episode_id: run_episode(s) for s in specs()}, backend, lock.append)
    key = lambda o: o.episode["episode_id"]  # noqa: E731
    assert [o.turns for o in sorted(serial, key=key)] == [o.turns for o in sorted(lock, key=key)]
    assert [o.episode for o in sorted(serial, key=key)] == [o.episode for o in sorted(lock, key=key)]
    assert max(len(b) for b in backend.calls) > 4  # requests really were batched across episodes


@pytest.mark.parametrize("resample", ["fixed", "per_turn"])
def test_graph_resampling(resample):
    cfg = {"episode": {"max_turns": 4},
           "comms": {"message_graph": {"generator": "clique", "m": 2, "resample": resample}}}
    out, _ = run(cfg, inst(target=3))
    graphs = [t["message_graph"] for t in out.turns]
    if resample == "fixed":
        assert all(g is None for g in graphs)
    else:
        assert all(g is not None for g in graphs)
        assert len({json.dumps(g["edges"]) for g in graphs}) > 1
