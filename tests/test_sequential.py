"""Sequential discussion (one speaker at a time) and the aggregate board view."""

import pytest
from pydantic import ValidationError

from llm_teams.backends.mock import MockBackend
from llm_teams.comms.notes import Note, NoteBoard, tally
from llm_teams.config import condition_hash
from llm_teams.runner.episode import run_episode
from llm_teams.runner.run import build_spec, drive
from tests.helpers import ProbeAgent, make_cfg
from tests.test_llm_agents import CLIQUE, INST, llm_cfg, run

SEQ = {"message_graph": CLIQUE, "discussion_order": "sequential"}
AGG = {**SEQ, "message_format": "structured", "board_view": "aggregate"}


def disc(reqs):
    return [r for r in reqs if r.meta["phase"] == "discussion"]


def say(token_by_agent):
    """Mock model: always plays 1; discussion reply is token_by_agent[agent] (default 'raise')."""
    return lambda r: ("ANSWER: 1" if r.meta["phase"] == "action"
                      else token_by_agent.get(r.meta["agent"], "raise"))


# ------------------------------------------------------------------ config


def test_defaults_do_not_change_condition_hash():
    base = make_cfg()
    both = make_cfg(comms={"discussion_order": "simultaneous", "board_view": "named"})
    assert condition_hash(base) == condition_hash(both)
    assert condition_hash(base) != condition_hash(make_cfg(comms={"discussion_order": "sequential"}))
    structured = {"message_format": "structured"}
    assert (condition_hash(make_cfg(comms=structured))
            != condition_hash(make_cfg(comms={**structured, "board_view": "aggregate"})))


def test_aggregate_requires_structured_messages():
    with pytest.raises(ValidationError, match="requires comms.message_format=structured"):
        make_cfg(comms={"board_view": "aggregate"})
    make_cfg(comms={"board_view": "aggregate", "message_format": "structured"})


# ------------------------------------------------------------------ the board itself


def test_visible_to_orders_by_slot_and_hides_later_slots():
    from llm_teams.comms.graph import CommGraph

    g = CommGraph(3)
    for a in range(3):
        for b in range(a + 1, 3):
            g.add_undirected(a, b)
    board = NoteBoard()
    board.post(1, 1, 2, "first", slot=0)
    board.post(1, 1, 0, "second", slot=1)
    board.post(1, 1, 1, "third", slot=2)
    assert [n.text for n in board.visible_to(1, g, before=(1, 1, 2))] == ["first", "second"]
    assert [n.text for n in board.visible_to(2, g, before=(1, 1, 0))] == []
    assert [n.text for n in board.visible_to(0, g, before=(1, 2, 0))] == ["first", "second", "third"]
    assert Note(0, 1, 1, "positional construction still works").slot == 0


def test_tally_counts_structured_texts_only():
    notes = [Note(0, 1, 1, "I will raise"), Note(1, 1, 2, "I will lower"),
             Note(2, 1, 1, "I will raise"), Note(0, 2, 1, "I will hold"), Note(1, 2, 1, "free text")]
    assert tally(notes) == [
        {"after_turn": 1, "raised": 2, "lowered": 1, "held": 0},
        {"after_turn": 2, "raised": 0, "lowered": 0, "held": 1},
    ]


# ------------------------------------------------------------------ sequential discussion


def test_sequential_calls_one_speaker_at_a_time_in_a_deterministic_shuffled_order():
    cfg = llm_cfg(episode={"max_turns": 6}, comms=SEQ)
    out, reqs = run(cfg, {**INST, "target": 0}, say({}))
    members = set(out.episode["clique_members"])
    orders = [t["speaking_order"][0] for t in out.turns if "speaking_order" in t]
    assert len(orders) == 5 and all(set(o) == members and len(o) == 3 for o in orders)
    assert len({tuple(o) for o in orders}) > 1  # reshuffled across turns
    for t in out.turns[:-1]:  # requests of turn k follow exactly that order, slot by slot
        d = [r for r in disc(reqs) if r.meta["turn"] == t["turn"]]
        assert [r.meta["agent"] for r in d] == t["speaking_order"][0]
        assert [r.meta["slot"] for r in d] == [0, 1, 2]
    out2, _ = run(cfg, {**INST, "target": 0}, say({}))
    assert [t.get("speaking_order") for t in out2.turns] == [t.get("speaking_order") for t in out.turns]
    assert [n["slot"] for n in out.notes[:3]] == [0, 1, 2]


def test_each_speaker_sees_exactly_the_earlier_speakers_of_the_round():
    backend = MockBackend(say({}))
    cfg = llm_cfg(episode={"max_turns": 2}, comms=SEQ)
    spec = build_spec(cfg, 0, {**INST, "target": 0}, "h")
    box = []
    drive({0: run_episode(spec)}, backend, box.append)
    out, reqs = box[0], [r for b in backend.calls for r in b]
    assert all(len(b) == 1 for b in backend.calls if b[0].meta["phase"] == "discussion")
    order = out.turns[0]["speaking_order"][0]
    prompts = [r.messages[1]["content"] for r in disc(reqs)]
    assert "no one has spoken yet" in prompts[0] and "Agent" not in prompts[0].split("Turn 1 is over")[0]
    assert "speaker 2 of 3" in prompts[1] and f"Agent {order[0]}" in prompts[1]
    assert f"Agent {order[1]}" not in prompts[1] and f"Agent {order[2]}" not in prompts[1]
    assert "speaker 3 of 3" in prompts[2] and "2 of the 2 group members" in prompts[2]
    assert all(f"Agent {a}" in prompts[2] for a in order[:2])


def test_simultaneous_run_is_unchanged_by_the_new_options():
    kw = dict(episode={"max_turns": 3})
    a, _ = run(llm_cfg(**kw, comms={"message_graph": CLIQUE}), INST, say({}))
    b, reqs = run(llm_cfg(**kw, comms={"message_graph": CLIQUE, "discussion_order": "simultaneous"}),
                  INST, say({}))
    assert a.notes == b.notes and {n["slot"] for n in a.notes} == {0}
    assert all("speaking_order" not in t for t in a.turns)
    assert all("speaker" not in r.messages[1]["content"] for r in disc(reqs))


def test_sequential_free_text_is_visible_to_later_speakers():
    cfg = llm_cfg(episode={"max_turns": 2}, comms=SEQ)
    out, reqs = run(cfg, {**INST, "target": 0},
                    lambda r: "ANSWER: 1" if r.meta["phase"] == "action" else f"hello from {r.meta['agent']}")
    last = disc(reqs)[2].messages[1]["content"]
    first, second = out.turns[0]["speaking_order"][0][:2]
    assert f"hello from {first}" in last and f"hello from {second}" in last


# ------------------------------------------------------------------ aggregate board view


def test_aggregate_shows_counts_not_authors_and_excludes_dropped_messages():
    cfg = llm_cfg(episode={"max_turns": 3}, comms=AGG)
    inst = {**INST, "target": 0}
    members = run(cfg, inst, say({}))[0].episode["clique_members"]
    raiser, prose, lowerer = members
    out, reqs = run(cfg, inst, lambda r: "ANSWER: 1" if r.meta["phase"] == "action" else
                    {raiser: "raise", prose: "I dunno, maybe lower?", lowerer: "lower"}[r.meta["agent"]])
    assert out.episode["board_view"] == "aggregate"
    order = out.turns[0]["speaking_order"][0]
    prompts = {r.meta["agent"]: r.messages[1]["content"] for r in disc(reqs) if r.meta["turn"] == 1}
    # each speaker sees counts of the valid signals before them; the prose was dropped, not counted
    for pos, agent in enumerate(order):
        before = [a for a in order[:pos]]
        raised, lowered = int(raiser in before), int(lowerer in before)
        if before:
            assert f"After turn 1 so far: {raised} raise, {lowered} lower, 0 hold" in prompts[agent]
        else:
            assert "Signal counts" not in prompts[agent]
        assert not any(f"Agent {a}" in prompts[agent].split("Turn 1 is over")[0] for a in members)
    # act at turn 2: everyone's valid turn-1 signals, own included, with no authors
    act2 = [r for r in reqs if r.meta["phase"] == "action" and r.meta["turn"] == 2
            and r.meta["agent"] in members]
    assert act2 and all("After turn 1: 1 raise, 1 lower, 0 hold" in r.messages[1]["content"] for r in act2)
    assert out.episode["invalid_messages"] == 2  # the prose reply, once per discussed turn


def test_aggregate_view_never_contains_authors():
    cfg = make_cfg(episode={"max_turns": 3}, comms=AGG)
    import json

    class Signaller(ProbeAgent):
        def parse_message(self, view, result):
            return "I will raise"

    backend = MockBackend(lambda r: "1" if r.meta["phase"] == "action" else "raise")
    spec = build_spec(cfg, 0, {**INST, "target": 0}, "h")
    spec.agents = [Signaller(i) for i in range(4)]
    drive({0: run_episode(spec)}, backend, lambda o: None)
    views = [json.loads(r.messages[0]["content"]) for b in backend.calls for r in b]
    assert views and all(v["notes"] == [] for v in views)
    assert any(v["tallies"] for v in views)


# ------------------------------------------------------------------ analysis


def test_performance_table_keeps_discussion_orders_apart():
    import pandas as pd
    from analysis.report import performance_table

    base = {"agents_type": "llm", "n": 4, "p_message": 1.0, "p_action": None, "success": True,
            "turns": 3, "normalized_turns": 1.5, "oracle_turns": 2, "total_messages": 2,
            "total_tokens": 10, "invalid_count": 0, "consistency": 1.0, "bisection_score": 1.0,
            "invalid_messages": 0, "intent_coherent_rate": float("nan"),
            "intent_follow_rate": float("nan"), "message_format": "structured", "board_view": "named"}
    eps = pd.DataFrame([{**base, "discussion_order": "simultaneous"},
                        {**base, "discussion_order": "sequential"}])
    assert sorted(performance_table(eps)["discussion_order"]) == ["sequential", "simultaneous"]
