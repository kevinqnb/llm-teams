"""Prompt templates (Jinja2, stored as YAML in experiments/prompt-configs/<task>/<name>.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import jinja2
import yaml

from llm_teams.config import experiments_dir

FIELDS = ("system", "discussion", "action", "retry")


@dataclass(frozen=True)
class PromptTemplates:
    name: str
    templates: dict[str, jinja2.Template]

    def render(self, field: str, view: dict[str, Any]) -> str:
        return self.templates[field].render(view=view).strip()


@cache
def load_prompts(name: str) -> PromptTemplates:
    path = experiments_dir() / "prompt-configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"prompt config '{name}' not found at {path}")
    raw = yaml.safe_load(path.read_text())
    # optional `macros` field is prepended to every template
    missing = [f for f in FIELDS if f not in raw]
    if missing:
        raise ValueError(f"prompt config {name}: missing fields {missing}")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    return PromptTemplates(name, {f: env.from_string(raw.get("macros", "") + raw[f]) for f in FIELDS})
