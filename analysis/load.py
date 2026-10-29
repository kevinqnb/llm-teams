"""Read run outputs (experiments/results/<experiment>/<run>/) into DataFrames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@dataclass
class Run:
    path: Path
    config: dict[str, Any]
    episodes: pd.DataFrame
    turns: pd.DataFrame
    generations: pd.DataFrame


def _condition_columns(cfg: dict[str, Any]) -> dict[str, Any]:
    comms = cfg.get("comms", {})
    ag, mg = comms.get("action_graph", {}), comms.get("message_graph", {})
    return {
        "experiment": cfg.get("name"),
        "agents_type": cfg.get("agents", {}).get("type"),
        "p_action": ag.get("params", {}).get("p"),
        "p_message": mg.get("params", {}).get("p"),
        "know_id": cfg.get("agents", {}).get("know_id"),
        "planning_rounds": comms.get("planning_rounds"),
        "discussion_rounds": comms.get("discussion_rounds"),
        "max_turns": cfg.get("episode", {}).get("max_turns"),
    }


def load_run(run_dir: str | Path) -> Run:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    cols = _condition_columns(cfg)
    eps = pd.DataFrame(_jsonl(run_dir / "episodes.jsonl"))
    if not eps.empty:
        eps["m"] = eps["realized_message_graph"].map(lambda g: g.get("m", 0))
        eps["k"] = eps["realized_action_graph"].map(lambda g: g.get("k", 0))
        for k, v in cols.items():
            eps[k] = v
    return Run(run_dir, cfg, eps, pd.DataFrame(_jsonl(run_dir / "turns.jsonl")),
               pd.DataFrame(_jsonl(run_dir / "generations.jsonl")))


def load_experiment(path: str | Path) -> Run:
    """Load every run directory under an experiment directory (or a single run directory)."""
    path = Path(path)
    dirs = [path] if (path / "episodes.jsonl").exists() else sorted(
        d for d in path.iterdir() if (d / "episodes.jsonl").exists())
    if not dirs:
        raise FileNotFoundError(f"no runs with episodes.jsonl under {path}")
    runs = [load_run(d) for d in dirs]
    cat = lambda name: pd.concat([getattr(r, name) for r in runs], ignore_index=True)  # noqa: E731
    return Run(path, runs[0].config, cat("episodes"), cat("turns"), cat("generations"))


def load_notes(run_dir: str | Path) -> pd.DataFrame:
    """Notes of a run, parsed back from generations.jsonl discussion rows."""
    g = load_run(run_dir).generations
    if g.empty:
        return pd.DataFrame(columns=["episode_id", "turn", "round", "agent", "text"])
    d = g[g["phase"].isin(["discussion", "planning"])]
    return d.rename(columns={"parsed": "text"})[["episode_id", "turn", "round", "agent", "text"]]
