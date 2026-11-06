"""Offline vLLM backend: one batched `LLM.chat` call per `generate`."""

from __future__ import annotations

import time

from llm_teams.backends.base import Backend, split_reasoning
from llm_teams.config import ModelConfig
from llm_teams.types import GenerationRequest, GenerationResult


VISIBLE_TOKENS = 512  # visible-answer allowance for requests without their own max_tokens
FORCED_END_SLACK = 32  # room for the forced end-of-thinking tokens


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, cfg: ModelConfig) -> None:
        from vllm import LLM  # imported lazily: vllm is an optional extra, needs a GPU node

        if not cfg.model_id:
            raise ValueError("vllm backend needs model_id")
        if cfg.thinking_sampling.get("thinking_token_budget") is not None and cfg.enable_thinking \
                and not cfg.engine_kwargs.get("reasoning_parser"):
            raise ValueError("thinking_token_budget needs engine_kwargs.reasoning_parser (e.g. qwen3)")
        self.cfg = cfg
        self.llm = LLM(model=cfg.model_id, **cfg.engine_kwargs)

    def _sampling(self, req: GenerationRequest):
        from vllm import SamplingParams

        base = dict(self.cfg.thinking_sampling if self.cfg.enable_thinking else self.cfg.sampling)
        max_tokens = int(base.pop("max_tokens", 1024))
        budget = base.pop("thinking_token_budget", None)
        if budget is not None and self.cfg.enable_thinking:
            # soft budget: vLLM force-closes the thinking block after `budget` tokens, so the model
            # still writes an answer instead of being cut off mid-thought (needs reasoning_parser)
            visible = req.max_tokens if req.max_tokens is not None else VISIBLE_TOKENS
            return SamplingParams(seed=req.seed, max_tokens=int(budget) + visible + FORCED_END_SLACK,
                                  thinking_token_budget=int(budget), **base)
        if req.max_tokens is not None:
            # thinking mode: req.max_tokens caps the visible answer, on top of the thinking budget
            max_tokens = max_tokens + req.max_tokens if self.cfg.enable_thinking else req.max_tokens
        return SamplingParams(seed=req.seed, max_tokens=max_tokens, **base)

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        t0 = time.time()
        outs = self.llm.chat(
            [r.messages for r in requests],
            [self._sampling(r) for r in requests],
            chat_template_kwargs={"enable_thinking": self.cfg.enable_thinking},
            use_tqdm=False,
        )
        per_call = (time.time() - t0) / len(requests)  # batch time shared evenly (no per-request clock)
        results = []
        for req, out in zip(requests, outs, strict=True):
            text = out.outputs[0].text
            reasoning, content = split_reasoning(text, self.cfg.enable_thinking)
            results.append(GenerationResult(
                req.request_id, text, content, reasoning,
                prompt_tokens=len(out.prompt_token_ids or []),
                completion_tokens=len(out.outputs[0].token_ids), latency=per_call,
                finish_reason=out.outputs[0].finish_reason,
            ))
        return results
