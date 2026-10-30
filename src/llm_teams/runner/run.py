"""Lockstep runner, result writing, resume."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from llm_teams.agents import llm, scripted  # noqa: F401  (registers agent types)
from llm_teams.agents.base import AGENTS
from llm_teams.backends.base import Backend
from llm_teams.backends.mock import MockBackend
from llm_teams.config import ExperimentConfig, build_conditions, condition_hash, experiments_dir, repo_root
from llm_teams.runner.episode import EpisodeGen, EpisodeOutput, EpisodeSpec, run_episode
from llm_teams.seeding import derive_rng
from llm_teams.tasks.base import TASKS


def results_dir() -> Path:
    """experiments/results, or $LLM_TEAMS_RESULTS_DIR if set."""
    return Path(os.environ.get("LLM_TEAMS_RESULTS_DIR") or experiments_dir() / "results")


def make_backend(cfg: ExperimentConfig, dry_run: bool) -> Backend:
    backend = cfg.model.backend if cfg.model else "mock"
    if dry_run or backend == "mock":
        return MockBackend()
    if backend == "vllm":
        from llm_teams.backends.vllm_backend import VLLMBackend

        assert cfg.model is not None
        return VLLMBackend(cfg.model)
    raise NotImplementedError(f"backend '{backend}' is not implemented")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _jsonl_append(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if rows:
        with open(path, "a") as f:
            f.write("".join(json.dumps(r, default=str) + "\n" for r in rows))


class RunDir:
    """experiments/results/<experiment>/<timestamp>_<hash>/ ; reuses an existing dir (resume)."""

    def __init__(self, cfg: ExperimentConfig, cond_hash: str, results_root: Path | None = None):
        base = (results_root or results_dir()) / cfg.name
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted(base.glob(f"*_{cond_hash}"))
        self.path = existing[-1] if existing else base / f"{datetime.now():%Y%m%d-%H%M%S}_{cond_hash}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.cond_hash = cond_hash

    def done_episodes(self) -> set[int]:
        f = self.path / "episodes.jsonl"
        if not f.exists():
            return set()
        return {json.loads(line)["episode_id"] for line in f.read_text().splitlines() if line}

    def write(self, out: EpisodeOutput) -> None:
        eid = out.episode["episode_id"]
        _jsonl_append(self.path / "turns.jsonl", out.turns)
        _jsonl_append(self.path / "generations.jsonl", out.generations)
        if out.notes:
            nd = self.path / "notes" / f"episode_{eid}"
            nd.mkdir(parents=True, exist_ok=True)
            from llm_teams.comms.notes import NoteBoard

            board = NoteBoard()
            for n in out.notes:
                board.post(n["after_turn"], n["round"], n["author"], n["text"])
            graph = out.graphs["message_graph"]
            for t, r in sorted({(n["after_turn"], n["round"]) for n in out.notes}):
                (nd / f"after_turn_{t}_round_{r}.md").write_text(board.render_markdown(t, r, graph))
        _jsonl_append(self.path / "episodes.jsonl", [out.episode])  # last: marks completion

    def check_backend(self, backend: Backend) -> None:
        """Refuse to resume a run directory written by a different backend (e.g. mock vs vllm)."""
        f = self.path / "manifest.json"
        prev = json.loads(f.read_text()).get("backend") if f.exists() else None
        if prev is not None and prev != backend.name:
            raise RuntimeError(
                f"{self.path} was written by backend '{prev}', not '{backend.name}'; "
                "delete it or move it aside to start fresh")

    def write_manifest(self, cfg: ExperimentConfig, argv: list[str], start: float, end: float | None,
                       backend: Backend | None = None):
        (self.path / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
        manifest = {
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "versions": {p: _version(p) for p in ("llm-teams", "vllm", "torch", "transformers")},
            "backend": backend.name if backend else None,
            "model_id": cfg.model.model_id if cfg.model else None,
            "python": sys.version.split()[0], "hostname": platform.node(),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "start": start, "end": end, "argv": argv,
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2))


def build_spec(cfg: ExperimentConfig, eid: int, instance: dict[str, Any], cond_hash: str) -> EpisodeSpec:
    task = TASKS.get(cfg.task)()
    n = task.n_agents(instance)
    agents = AGENTS.get(cfg.agents.type)(n, cfg, derive_rng(cfg.seed, eid, "agents"))
    return EpisodeSpec(eid, instance, task, agents, cfg, cond_hash)


def drive(gens: dict[int, EpisodeGen], backend: Backend, on_done) -> None:
    """Advance episodes in lockstep, sending every pending request in one backend call."""
    waiting: dict[int, list] = {}

    def advance(eid: int, value: Any) -> None:
        try:
            waiting[eid] = gens[eid].send(value)
        except StopIteration as stop:
            waiting.pop(eid, None)
            on_done(stop.value)

    for eid in gens:
        advance(eid, None)
    while waiting:
        order = list(waiting)
        flat = [r for eid in order for r in waiting[eid]]
        results = backend.generate(flat)
        pos = 0
        for eid in order:
            k = len(waiting[eid])
            advance(eid, results[pos:pos + k])
            pos += k


def run_condition(
    cfg: ExperimentConfig,
    run_dir: RunDir,
    backend: Backend,
    max_episodes: int | None = None,
    argv: list[str] | None = None,
) -> int:
    """Run all pending episodes of one condition. Returns number of episodes run."""
    start = time.time()
    run_dir.check_backend(backend)
    run_dir.write_manifest(cfg, argv or [], start, None, backend)
    instances = TASKS.get(cfg.task).make_instances(cfg.dataset.model_dump())
    if max_episodes is not None:
        instances = instances[:max_episodes]
    done = run_dir.done_episodes()
    pending = [i for i in range(len(instances)) if i not in done]
    for lo in range(0, len(pending), cfg.episodes_per_batch):
        batch = pending[lo:lo + cfg.episodes_per_batch]
        gens = {e: run_episode(build_spec(cfg, e, instances[e], run_dir.cond_hash)) for e in batch}
        drive(gens, backend, run_dir.write)
    run_dir.write_manifest(cfg, argv or [], start, time.time(), backend)
    return len(pending)


def run_experiment(
    path: str, overrides: dict[str, Any], dry_run: bool = False, shard: tuple[int, int] | None = None,
    results_root: Path | None = None, argv: list[str] | None = None,
) -> list[Path]:
    conditions = build_conditions(path, overrides)
    dirs = []
    for idx, (_, cfg) in enumerate(conditions):
        if shard and idx % shard[1] != shard[0] - 1:
            continue
        h = condition_hash(cfg)
        rd = RunDir(cfg, h, results_root)
        n = run_condition(cfg, rd, make_backend(cfg, dry_run), 3 if dry_run else None, argv)
        print(f"[{idx + 1}/{len(conditions)}] {cfg.name} {h}: ran {n} episodes -> {rd.path}")
        dirs.append(rd.path)
    return dirs


def summarize(run_dir: Path) -> dict[str, Any]:
    eps = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines() if line]
    if not eps:
        return {"episodes": 0}
    succ = [e for e in eps if e["success"]]
    n_actions = sum(e["turns"] * e["n"] for e in eps)
    return {
        "episodes": len(eps),
        "success_rate": len(succ) / len(eps),
        "mean_turns_success": sum(e["turns"] for e in succ) / len(succ) if succ else None,
        "mean_turns_all": sum(e["turns"] for e in eps) / len(eps),
        "invalid_rate": sum(e["invalid_count"] for e in eps) / n_actions,
        "mean_messages": sum(e["total_messages"] for e in eps) / len(eps),
    }
