"""Optional LLM-judge labelling of each episode's notes into a fixed strategy taxonomy.

Usage: uv run python -m analysis.judge_strategies <run_dir> [--model qwen3.5-4b | --mock] [--out f.csv]
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from analysis.load import load_notes
from llm_teams.backends.base import Backend
from llm_teams.backends.mock import MockBackend
from llm_teams.types import GenerationRequest

TAXONOMY = {
    "binary_search": "halving the range of possible targets using the Too high / Too low feedback",
    "linear_scan": "changing the sum by one unit per turn, scanning up or down",
    "id_based": "deciding who plays 1 by agent ID, e.g. 'IDs below k play 1'",
    "assignment": "explicitly assigning specific agents to play 0 or 1 to hit a planned sum",
    "random_or_none": "guessing at random or no coherent strategy",
}
PROMPT = """You will read messages that agents in a team exchanged while playing a game where each \
agent picks 0 or 1 and the team tries to hit a hidden target sum, learning only 'Too high' or \
'Too low' after each turn.

Label the team's *stated* strategy with exactly one of:
{taxonomy}

Messages:
{notes}

Reply with a single line: LABEL: <one label from the list>"""


def build_prompt(notes: list[str]) -> str:
    tax = "\n".join(f"- {k}: {v}" for k, v in TAXONOMY.items())
    return PROMPT.format(taxonomy=tax, notes="\n".join(f"- {n}" for n in notes))


def parse_label(content: str) -> str | None:
    m = re.findall(r"LABEL:\s*([a-z_]+)", content, re.I)
    return m[-1].lower() if m and m[-1].lower() in TAXONOMY else None


def judge(notes: pd.DataFrame, backend: Backend) -> pd.DataFrame:
    groups = list(notes.groupby("episode_id"))
    reqs = [GenerationRequest([{"role": "user", "content": build_prompt(list(g.sort_values(
        ["turn", "round", "agent"])["text"]))}], request_id=str(e), max_tokens=64) for e, g in groups]
    res = backend.generate(reqs)
    return pd.DataFrame({"episode_id": [e for e, _ in groups],
                         "stated_strategy": [parse_label(r.content) for r in res],
                         "n_notes": [len(g) for _, g in groups]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default="qwen3.5-4b", help="model config name")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.mock:
        backend: Backend = MockBackend(lambda r: "LABEL: random_or_none")
    else:
        import yaml

        from llm_teams.backends.vllm_backend import VLLMBackend
        from llm_teams.config import ModelConfig, experiments_dir

        cfg = ModelConfig.model_validate(yaml.safe_load(
            (experiments_dir() / "model-configs" / f"{a.model}.yaml").read_text()))
        backend = VLLMBackend(cfg)
    out = judge(load_notes(a.run_dir), backend)
    out.to_csv(a.out or f"{a.run_dir}/judged_strategies.csv", index=False)
    print(out["stated_strategy"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
