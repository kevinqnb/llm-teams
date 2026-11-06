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
