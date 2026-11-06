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


def test_engine_kwargs_builds_reasoning_config_from_yaml_mapping():
    pytest.importorskip("vllm")
    from vllm.config.reasoning import ReasoningConfig

    from llm_teams.backends.vllm_backend import engine_kwargs
    import yaml

    from llm_teams.config import experiments_dir

    path = experiments_dir() / "model-configs" / "qwen3.5-4b-think2k.yaml"
    cfg = ModelConfig.model_validate(yaml.safe_load(path.read_text()))
    kw = engine_kwargs(cfg)
    rc = kw["reasoning_config"]
    assert isinstance(rc, ReasoningConfig) and kw["reasoning_parser"] == "qwen3"
    assert rc.reasoning_end_str.rstrip().endswith("</think>")  # the answer follows the closing tag
    assert "thinking time is up" in rc.reasoning_end_str
    assert isinstance(cfg.engine_kwargs["reasoning_config"], dict)  # input config is untouched


def test_brief_prompt_asks_for_answer_only_and_warns_about_the_cutoff():
    cfg = make_cfg(prompt="sum_to_target/brief", agents={"type": "llm", "thinking_phases": ["discussion"]},
                   episode={"max_turns": 3},
                   comms={"message_graph": CLIQUE, "message_format": "structured"})
    _, reqs = run(cfg, INST, reply)
    action = next(r for r in reqs if r.meta["phase"] == "action").messages[-1]["content"]
    disc = next(r for r in reqs if r.meta["phase"] == "discussion").messages[-1]["content"]
    assert "Do not explain or reason out loud" in action and "ANSWER: 1" in action
    assert "thinking time is limited" in disc and "exactly one word" in disc
