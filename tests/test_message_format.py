import pytest

from llm_teams.agents.parse import parse_message_token
from llm_teams.comms.notes import MESSAGE_TOKENS, STRUCTURED_TEXTS
from llm_teams.config import condition_hash
from tests.helpers import make_cfg
from tests.test_llm_agents import CLIQUE, INST, llm_cfg, run


@pytest.mark.parametrize("text,expected", [
    ("MESSAGE: raise", "I will raise"),
    ("hold", "I will hold"),
    ("  **Lower**.\n", "I will lower"),
    ("RAISE!", "I will raise"),
    ("I think we should hold this turn", None),  # token buried in prose is rejected
    ("hold on, let me think\nraise", None),
    ("raise lower", None),
    ("hmm\nmessage: **Lower**.", "I will lower"),
    ("MESSAGE: hold\nMESSAGE: raise", "I will raise"),  # last wins
    ("MESSAGE: jump", None),
    ("I will raise", None),
    ("", None),
])
def test_parse_message_token(text, expected):
    assert parse_message_token(text, MESSAGE_TOKENS) == expected


def comms(fmt):
    return {"message_graph": CLIQUE, "message_format": fmt}


def respond(msg):
    return lambda r: "ANSWER: 1" if r.meta["phase"] == "action" else msg


def test_structured_posts_only_canonical_text():
    cfg = llm_cfg(episode={"max_turns": 3}, comms=comms("structured"))
    out, _ = run(cfg, INST, respond("MESSAGE: raise"))
    assert out.notes and {n["text"] for n in out.notes} <= STRUCTURED_TEXTS
    assert out.episode["invalid_messages"] == 0


def test_structured_drops_free_text():
    cfg = llm_cfg(episode={"max_turns": 3}, comms=comms("structured"))
    out, reqs = run(cfg, INST, respond("Everyone pick 1, trust me"))
    assert out.notes == []
    assert out.episode["total_messages"] == 0
    assert out.episode["invalid_messages"] > 0
    disc = [r for r in reqs if r.meta["phase"] == "discussion"]
    assert any("Reply with exactly one word" in m["content"] for r in disc for m in r.messages)


def test_free_mode_unchanged():
    cfg = llm_cfg(episode={"max_turns": 3}, comms=comms("free"))
    out, reqs = run(cfg, INST, respond("Everyone pick 1"))
    assert {n["text"] for n in out.notes} == {"Everyone pick 1"}
    assert out.episode["invalid_messages"] == 0
    assert not any("Reply with exactly one word" in m["content"] for r in reqs for m in r.messages)


def test_default_condition_hash_ignores_new_field():
    a = make_cfg()
    assert condition_hash(a) == condition_hash(make_cfg(comms={"message_format": "free"}))
    assert condition_hash(a) != condition_hash(make_cfg(comms={"message_format": "structured"}))


def test_free_mode_drops_and_counts_empty_replies():
    cfg = llm_cfg(episode={"max_turns": 3}, comms=comms("free"))
    out, _ = run(cfg, INST, respond(""))
    assert out.notes == []
    assert out.episode["total_messages"] == 0
    assert out.episode["invalid_messages"] > 0


def test_truncated_thinking_is_dropped_in_both_formats():
    for fmt in ("free", "structured"):
        cfg = llm_cfg(episode={"max_turns": 3}, comms=comms(fmt))
        out, _ = run(cfg, INST, respond("<think>still thinking, never closed"))
        assert out.notes == [], fmt


@pytest.mark.parametrize("intent,prev,nxt,expected", [
    ("raise", 0, 1, (True, True)),
    ("raise", 0, 0, (True, False)),
    ("raise", 1, 1, (False, True)),   # cannot raise from 1, even though the agent ended on 1
    ("lower", 1, 0, (True, True)),
    ("lower", 0, 0, (False, True)),
    ("hold", 1, 1, (True, True)),
    ("hold", 0, 1, (True, False)),
])
def test_intent_outcome(intent, prev, nxt, expected):
    from analysis import metrics as M
    assert M.intent_outcome(intent, prev, nxt) == expected


def test_intent_stats_scores_hold_messages(tmp_path):
    import pandas as pd
    from analysis.features import intent_stats
    from analysis.load import Run

    cfg = llm_cfg(episode={"max_turns": 4}, comms=comms("structured"))
    # target 3 of 4 and everyone plays 1 (sum 4), so the game runs on and "hold" is always followed
    out, _ = run(cfg, INST, lambda r: "ANSWER: 1" if r.meta["phase"] == "action" else "hold")
    assert {n["text"] for n in out.notes} == {"I will hold"}
    turns, gens = pd.DataFrame(out.turns), pd.DataFrame(out.generations)
    turns["condition_hash"] = gens["condition_hash"] = "h"
    st = intent_stats(Run(tmp_path, {}, pd.DataFrame(), turns, gens))
    assert len(st) == 1 and st.loc[0, "n_intents"] > 0
    assert st.loc[0, "intent_coherent_rate"] == 1.0 and st.loc[0, "intent_follow_rate"] == 1.0


def test_performance_table_splits_formats():
    import pandas as pd
    from analysis.report import performance_table

    base = {"agents_type": "llm", "n": 4, "p_message": 0.5, "p_action": None, "success": True,
            "turns": 3, "normalized_turns": 1.5, "oracle_turns": 2, "total_messages": 2,
            "total_tokens": 10, "invalid_count": 0, "consistency": 1.0, "bisection_score": 1.0,
            "invalid_messages": 0, "intent_coherent_rate": float("nan"),
            "intent_follow_rate": float("nan")}
    eps = pd.DataFrame([{**base, "message_format": "free"}, {**base, "message_format": "structured"}])
    assert sorted(performance_table(eps)["message_format"]) == ["free", "structured"]
