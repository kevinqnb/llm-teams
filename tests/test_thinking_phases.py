from types import SimpleNamespace

import pytest

from llm_teams.config import ModelConfig
from llm_teams.types import GenerationRequest
from tests.helpers import make_cfg
from tests.test_llm_agents import CLIQUE, INST, run


def cfg_with(phases, **comms):
    agents = {"type": "llm"} if phases == "unset" else {"type": "llm", "thinking_phases": phases}
    return make_cfg(prompt="sum_to_target/default", agents=agents, episode={"max_turns": 3},
                    comms={"message_graph": CLIQUE, "message_format": "structured", **comms})


def reply(r):
    return "ANSWER: 1" if r.meta["phase"] == "action" else "hold"


def test_only_listed_phases_think():
    out, reqs = run(cfg_with(["discussion"]), INST, reply)
    by_phase = {r.meta["phase"]: {x.thinking for x in reqs if x.meta["phase"] == r.meta["phase"]}
                for r in reqs}
    assert by_phase == {"action": {False}, "discussion": {True}}
    assert {g["thinking"] for g in out.generations if g["phase"] == "discussion"} == {True}


def test_retry_follows_the_action_setting():
    def invalid_first(r):
        if r.meta["phase"] == "action":
            return "ANSWER: 1" if r.request_id.endswith(":retry") else "no answer"
        return "hold"

    _, reqs = run(cfg_with(["discussion"]), INST, invalid_first)
    retries = [r for r in reqs if r.request_id.endswith(":retry")]
    assert retries and all(r.thinking is False for r in retries)


def test_unset_leaves_backend_default_and_unknown_phase_is_rejected():
    _, reqs = run(cfg_with("unset"), INST, reply)
    assert {r.thinking for r in reqs} == {None}
    with pytest.raises(ValueError, match="thinking_phases"):
        run(cfg_with(["planning"]), INST, reply)


def fake_vllm_backend(enable_thinking):
    pytest.importorskip("vllm")
    from llm_teams.backends.vllm_backend import VLLMBackend

    calls = []

    def chat(msgs, params, chat_template_kwargs, use_tqdm):
        calls.append((len(msgs), chat_template_kwargs["enable_thinking"]))
        think = chat_template_kwargs["enable_thinking"]
        return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
            text=("t</think>raise" if think else "lower"), token_ids=[1, 2], finish_reason="stop")])
            for _ in msgs]

    b = VLLMBackend.__new__(VLLMBackend)
    b.cfg = ModelConfig(enable_thinking=enable_thinking,
                        thinking_sampling={"max_tokens": 64, "thinking_token_budget": 32})
    b.llm = SimpleNamespace(chat=chat)
    return b, calls


def test_backend_splits_calls_by_thinking_and_keeps_order():
    b, calls = fake_vllm_backend(enable_thinking=False)
    reqs = [GenerationRequest([], request_id="a", thinking=True),
            GenerationRequest([], request_id="b"),                  # default = global (off)
            GenerationRequest([], request_id="c", thinking=True)]
    res = b.generate(reqs)
    assert sorted(calls) == [(1, False), (2, True)]
    assert [r.content for r in res] == ["raise", "lower", "raise"]
    assert [r.reasoning for r in res] == ["t", "", "t"]


def test_backend_per_request_flag_overrides_global_thinking():
    b, calls = fake_vllm_backend(enable_thinking=True)
    res = b.generate([GenerationRequest([], thinking=False), GenerationRequest([])])
    assert sorted(calls) == [(1, False), (1, True)]
    assert [r.content for r in res] == ["lower", "raise"]
