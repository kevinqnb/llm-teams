"""YAML configs -> pydantic models, name resolution, overrides and sweep expansion."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------- paths


def repo_root() -> Path:
    env = os.environ.get("LLM_TEAMS_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "experiments").is_dir():
            return parent
    raise RuntimeError("cannot locate repo root (no experiments/ directory); set LLM_TEAMS_ROOT")


def experiments_dir() -> Path:
    return repo_root() / "experiments"


# ---------------------------------------------------------------- models


class GraphSpec(BaseModel):
    """A named graph generator; all keys other than the fields below are generator params."""

    generator: str = "empty"
    resample: Literal["fixed", "per_turn"] = "fixed"
    action_history: Literal["last", "all"] = "last"
    notes_history: Literal["last_turn", "all"] = "all"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_params(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        own = set(cls.model_fields)
        out = {k: x for k, x in v.items() if k in own}
        out["params"] = {**v.get("params", {}), **{k: x for k, x in v.items() if k not in own}}
        return out


class CommsConfig(BaseModel):
    action_graph: GraphSpec = GraphSpec()
    message_graph: GraphSpec = GraphSpec()
    discussion_rounds: int = 1
    planning_rounds: int = 0
    max_message_tokens: int = 200
    reveal_group: bool = True


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "random"
    know_n: bool = True
    know_id: bool = True
    memory: Literal["structured", "transcript"] = "structured"


class EpisodeConfig(BaseModel):
    max_turns: int = 20
    invalid_policy: Literal["random", "repeat_last", "zero"] = "random"


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""
    num_episodes: int = 10
    seed: int = 0


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    name: str = "mock"
    backend: str = "mock"
    model_id: str | None = None
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    enable_thinking: bool = False
    sampling: dict[str, Any] = Field(default_factory=dict)
    thinking_sampling: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str
    task: str
    dataset: DatasetConfig
    model: ModelConfig | None = None
    prompt: str | None = None
    agents: AgentsConfig = AgentsConfig()
    episode: EpisodeConfig = EpisodeConfig()
    comms: CommsConfig = CommsConfig()
    seed: int = 0
    episodes_per_batch: int = 64
    sweep: dict[str, list[Any]] = Field(default_factory=dict)


# ---------------------------------------------------------------- loading


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _resolve_named(raw: dict[str, Any], key: str, subdir: str) -> None:
    """Replace a name reference like `dataset: foo` with the loaded config dict."""
    val = raw.get(key)
    if isinstance(val, str):
        path = experiments_dir() / subdir / f"{val}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"{key} config '{val}' not found at {path}")
        loaded = _load_yaml(path)
        loaded.setdefault("name", val)
        raw[key] = loaded


NAMED = (("dataset", "dataset-configs"), ("model", "model-configs"))


def set_path(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
        if not isinstance(d, dict):
            raise ValueError(f"cannot set '{dotted}': '{k}' is not a mapping")
    d[keys[-1]] = value


def parse_override(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"override '{s}' must look like key=value")
    k, v = s.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def load_raw(path: str | Path) -> dict[str, Any]:
    """Experiment YAML as a dict with dataset/model inlined (still contains `sweep`)."""
    raw = _load_yaml(Path(path))
    for key, sub in NAMED:
        _resolve_named(raw, key, sub)
    return raw


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    for key, sub in NAMED:  # whole-config swaps first, so dotted overrides apply on top
        if isinstance(overrides.get(key), str):
            raw[key] = overrides[key]
            _resolve_named(raw, key, sub)
    for k, v in overrides.items():
        if k in dict(NAMED) and isinstance(v, str):
            continue
        set_path(raw, k, v)
    return raw


def expand_sweep(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of `sweep` as a list of {dotted.key: value} dicts ({} if no sweep)."""
    sweep = raw.get("sweep") or {}
    if not sweep:
        return [{}]
    keys = list(sweep)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(sweep[k] for k in keys))]


def condition_hash(cfg: ExperimentConfig) -> str:
    blob = json.dumps(cfg.model_dump(mode="json", exclude={"sweep"}), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def build_conditions(
    path: str | Path, overrides: dict[str, Any] | None = None
) -> list[tuple[dict[str, Any], ExperimentConfig]]:
    """All sweep conditions as (sweep point, validated config with sweep removed)."""
    raw = apply_overrides(load_raw(path), overrides or {})
    out = []
    for point in expand_sweep(raw):
        cond = apply_overrides(raw, point)
        cond["sweep"] = {}
        out.append((point, ExperimentConfig.model_validate(cond)))
    return out


def validate_all(root: Path | None = None) -> list[str]:
    """Load every config under experiments/; return a list of error strings."""
    exp = (root or repo_root()) / "experiments"
    errors: list[str] = []
    for f in sorted((exp / "experiment-configs").glob("*.yaml")):
        try:
            build_conditions(f)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f.name}: {e}")
    for sub in ("dataset-configs", "model-configs"):
        for f in sorted((exp / sub).glob("*.yaml")):
            try:
                _load_yaml(f)
                (DatasetConfig if sub == "dataset-configs" else ModelConfig).model_validate(
                    _load_yaml(f)
                )
            except Exception as e:  # noqa: BLE001
                errors.append(f"{sub}/{f.name}: {e}")
    return errors
