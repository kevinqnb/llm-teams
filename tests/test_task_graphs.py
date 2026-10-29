import math

import numpy as np
import pytest

from llm_teams.comms.generators import build_graph, round_half_up
from llm_teams.comms.graph import CommGraph
from llm_teams.seeding import derive_rng
from llm_teams.tasks.sum_to_target import SumToTarget, optimal_turn_bound


def make_task(n, target, feedback="direction"):
    t = SumToTarget()
    t.reset({"n": n, "target": target, "feedback": feedback}, np.random.default_rng(0))
    return t


def test_feedback_and_success():
    t = make_task(4, 2)
    r = t.step({0: 1, 1: 1, 2: 1, 3: 0})
    assert not r.success and r.feedback[0]["result"] == "too_high"
    r = t.step({0: 1, 1: 0, 2: 0, 3: 0})
    assert r.feedback[3]["result"] == "too_low"
    assert t.step({0: 1, 1: 1, 2: 0, 3: 0}).success


@pytest.mark.parametrize("target", [0, 4])
def test_boundary_targets(target):
    t = make_task(4, target)
    r = t.step({i: int(target == 4) for i in range(4)})
    assert r.success


def test_feedback_variants_and_no_target_leak_in_describe():
    t = make_task(4, 3, "exact_sum")
    assert t.step({0: 1, 1: 0, 2: 0, 3: 0}).feedback[0]["sum"] == 1
    assert "target" not in t.describe(0)
    d = make_task(4, 3, "direction_and_distance").step({i: 0 for i in range(4)})
    assert d.feedback[0]["distance"] == 3


def test_instances_deterministic_and_in_range():
    ds = {"n": 8, "num_episodes": 30, "seed": 3}
    a, b = SumToTarget.make_instances(ds), SumToTarget.make_instances(ds)
    assert a == b and all(0 <= x["target"] <= 8 for x in a)
    assert a != SumToTarget.make_instances({**ds, "seed": 4})


def test_optimal_bound_is_tight_for_small_n():
    assert optimal_turn_bound(1) == 2 and optimal_turn_bound(3) == 3
    assert optimal_turn_bound(7) == math.ceil(math.log2(9))


# ------------------------------------------------------------------ graphs


@pytest.mark.parametrize("n,p", [(4, 0.1), (4, 0.5), (8, 0.5), (10, 0.75), (6, 1.0)])
def test_clique(n, p):
    g = build_graph("clique", n, {"p": p}, derive_rng(1))
    m = round_half_up(p * n)
    members = g.meta["members"]
    if m <= 1:
        assert g.edges() == set() and members == []
    else:
        assert len(members) == m
        assert all((a, b) in g.edges() for a in members for b in members if a != b)
        assert all(g.is_isolated(i) for i in range(n) if i not in members)
    assert g.meta["m"] == m


def test_clique_p1_complete():
    assert build_graph("clique", 5, {"p": 1.0}, derive_rng(0)) == build_graph(
        "complete", 5, {}, derive_rng(0)
    )


def test_random_k():
    g = build_graph("random_k", 8, {"p": 0.5}, derive_rng(2))
    k = round_half_up(0.5 * 7)
    assert all(len(g.neighbors_in(i)) == k and i not in g.neighbors_in(i) for i in range(8))
    assert g.meta["k"] == k
    assert build_graph("random_k", 8, {"p": 0.0}, derive_rng(2)).edges() == set()


def test_determinism_and_roundtrip():
    a = build_graph("random_k", 8, {"p": 0.5}, derive_rng(5))
    assert a == build_graph("random_k", 8, {"p": 0.5}, derive_rng(5))
    assert a != build_graph("random_k", 8, {"p": 0.5}, derive_rng(6))
    assert CommGraph.from_dict(a.to_dict()) == a
    assert CommGraph.from_dict(a.to_dict()).meta == a.meta


def test_ring_and_components():
    g = build_graph("ring", 5, {}, derive_rng(0))
    assert all(len(g.neighbors_in(i)) == 2 for i in range(5))
    assert g.components() == [[0, 1, 2, 3, 4]]
    c = build_graph("clique", 6, {"m": 3}, derive_rng(0))
    assert sorted(len(x) for x in c.components()) == [1, 1, 1, 3]
