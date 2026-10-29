import json
import math
import shutil

import pytest

from llm_teams.backends.mock import MockBackend
from llm_teams.config import build_conditions, repo_root, validate_all
from llm_teams.runner.episode import run_episode
from llm_teams.runner.run import build_spec, drive, run_experiment, summarize
from llm_teams.tasks.sum_to_target import optimal_turn_bound
from tests.helpers import make_cfg

CFG = repo_root() / "experiments/experiment-configs/sum_scripted_baselines.yaml"


def solve(agent_type, n, target, **agents):
    cfg = make_cfg(**{"agents": {"type": agent_type, **agents}, "episode": {"max_turns": 30}})
    spec = build_spec(cfg, 0, {"n": n, "target": target, "feedback": "direction"}, "h")
    box = []
    drive({0: run_episode(spec)}, MockBackend(), box.append)
    return box[0]


@pytest.mark.parametrize("agent_type", ["oracle", "id_bisection"])
def test_scripted_bisection_hits_bound(agent_type):
    for n in range(1, 65):
        targets = range(n + 1) if n <= 20 else {0, 1, n // 3, n // 2, n - 1, n}
        for t in targets:
            out = solve(agent_type, n, t)
            ep = out.episode
            assert ep["success"], (n, t)
            assert ep["turns"] <= optimal_turn_bound(n) <= math.ceil(math.log2(n + 2))
            assert ep["turns"] == ep["oracle_turns"]
            assert ep["total_messages"] == 0  # zero communication


def test_id_bisection_needs_id_and_n():
    with pytest.raises(ValueError):
        solve("id_bisection", 4, 2, know_id=False)


def test_random_agents_log_censoring():
    ep = solve("random", 16, 3).episode
    assert ep["censored"] == (not ep["success"])


def test_validate_configs_ships_clean():
    assert validate_all() == []


def test_sweep_expansion():
    conds = build_conditions(CFG)
    assert len(conds) == 12
    assert {(p["dataset.n"], p["agents.type"]) for p, _ in conds} == {
        (n, a) for n in (4, 8, 16, 32) for a in ("oracle", "id_bisection", "random")}
    assert all(c.dataset.n == p["dataset.n"] for p, c in conds)  # type: ignore[attr-defined]


def test_run_writes_schema_and_resumes(tmp_path):
    ov = {"sweep": {}, "dataset.n": 8, "dataset.num_episodes": 6}
    (rd,) = run_experiment(str(CFG), ov, results_root=tmp_path)
    for f in ("config.resolved.yaml", "manifest.json", "episodes.jsonl", "turns.jsonl",
              "generations.jsonl"):
        assert (rd / f).exists() or f == "generations.jsonl"
    eps = [json.loads(line) for line in (rd / "episodes.jsonl").read_text().splitlines()]
    assert [e["episode_id"] for e in eps] == list(range(6))
    for key in ("n", "target", "success", "turns", "optimal_turns", "censored", "total_tokens",
                "total_messages", "invalid_count", "clique_members", "action_graph",
                "message_graph", "realized_message_graph", "condition_hash"):
        assert key in eps[0], key
    turns = [json.loads(line) for line in (rd / "turns.jsonl").read_text().splitlines()]
    assert {"actions", "sum", "feedback", "interval_before", "interval_after",
            "n_messages_after"} <= set(turns[0])
    assert json.loads((rd / "manifest.json").read_text())["git_commit"]

    # resume: nothing left to do, then drop the last episode and re-run just that one
    assert run_experiment(str(CFG), ov, results_root=tmp_path) == [rd]
    assert len((rd / "episodes.jsonl").read_text().splitlines()) == 6
    lines = (rd / "episodes.jsonl").read_text().splitlines()
    (rd / "episodes.jsonl").write_text("\n".join(lines[:5]) + "\n")
    run_experiment(str(CFG), ov, results_root=tmp_path)
    assert len((rd / "episodes.jsonl").read_text().splitlines()) == 6
    assert summarize(rd)["success_rate"] == 1.0
    shutil.rmtree(tmp_path)


def test_shards_partition_conditions(tmp_path):
    ov = {"dataset.num_episodes": 1}
    seen = []
    for i in (1, 2, 3):
        seen += run_experiment(str(CFG), ov, shard=(i, 3), results_root=tmp_path)
    assert len(seen) == len(set(seen)) == 12
