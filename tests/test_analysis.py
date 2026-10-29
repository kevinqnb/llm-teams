import math

import numpy as np
import pytest
from analysis import metrics as M
from analysis.features import episode_features
from analysis.judge_strategies import build_prompt, judge, parse_label
from analysis.load import load_experiment, load_notes

from llm_teams.backends.mock import MockBackend
from llm_teams.config import repo_root
from llm_teams.runner.run import run_experiment

# n = 15, target 11: perfect bisection sums are 7 -> 11
BISECT = ([7, 11], ["too_low", "correct"])
# linear scan up to target 6 from 0
LINEAR = ([0, 1, 2, 3, 4, 5, 6], ["too_low"] * 6 + ["correct"])
OSC = ([12, 3, 12, 3, 12, 3], ["too_high", "too_low", "too_high", "too_low", "too_high", "too_low"])
STUCK = ([4, 4, 4, 4], ["too_low"] * 4)


def test_intervals_and_consistency():
    sums, res = [7, 11, 9], ["too_low", "too_high", "correct"]
    iv = M.intervals_before(15, sums, res)
    assert iv == [(0, 15), (8, 15), (8, 10)]
    assert M.consistency_rate(sums, iv) == 1.0
    assert M.consistency_rate([7, 3], M.intervals_before(15, [7, 3], ["too_low", "too_low"])) == 0.5


def test_bisection_score_and_info_gain():
    assert M.bisection_scores([7, 8], [(0, 15), (0, 15)]) == [1.0, 1.0]  # both middles of 16 values
    assert M.bisection_scores([0], [(0, 14)])[0] == pytest.approx(0.0)    # an endpoint scores 0
    assert M.bisection_scores([3], [(8, 15)])[0] < 0                      # ruled-out value
    (gain, mx), = M.info_gains(15, [7, 11], ["too_low", "correct"])[:1]
    assert gain == pytest.approx(1.0) and mx == pytest.approx(math.log2(16 / 8))


def test_step_profile_oscillation_overshoot():
    assert M.step_profile([0, 1, 3, 3]) == [1, 2, 0]
    assert M.oscillations(OSC[0]) == 4 and M.oscillations(LINEAR[0]) == 0
    assert M.overshoots(OSC[1]) == 5 and M.overshoots(LINEAR[1]) == 0


@pytest.mark.parametrize("traj,n,label", [
    (BISECT, 15, "binary_search_like"), (LINEAR, 15, "linear_scan"), (OSC, 15, "random_walk"),
    (STUCK, 15, "stuck"),
])
def test_classifier(traj, n, label):
    assert M.classify_episode(n, *traj) == label


def test_classifier_oracle_always_binary_search():
    from analysis.plot_strategy import oracle_trajectory

    for n in (4, 8, 16, 31):
        for t in range(n + 1):
            sums = oracle_trajectory(n, t)
            res = ["too_high" if s > t else "too_low" for s in sums[:-1]] + ["correct"]
            assert M.classify_episode(n, sums, res) == "binary_search_like", (n, t)


def test_success_at_k_and_kaplan_meier():
    turns, ok = [1, 2, 2, 5, 5], [True, True, True, False, False]
    assert M.success_at_k(turns, ok, 5) == [0.2, 0.6, 0.6, 0.6, 0.6]
    km = dict(M.kaplan_meier(turns, ok))
    assert km[1] == pytest.approx(0.8) and km[2] == pytest.approx(0.4)  # = 1 - success@k here
    assert M.normalized_turns(6, 3) == 2


def test_flips_and_direction_consistency():
    actions = [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
    res = ["too_high", "too_low", "correct"]
    assert M.flip_rates(actions) == [0.0, 0.0, 1.0, 0.5]
    c = M.direction_flips(actions, res)
    assert (c.total, c.consistent, c.inconsistent) == (3, 3, 0)
    bad = M.direction_flips([[1, 0], [0, 1]], ["too_low", "correct"])
    assert (bad.consistent, bad.inconsistent) == (1, 1)


def test_overcorrection_herding():
    # n=4: all play 1 (sum 4, too high), then all flip to 0 although ~2 flips were needed
    oc = M.overcorrection(4, [[1] * 4, [0] * 4], ["too_high", "too_low"])
    assert oc == [{"turn": 1, "flipped": 4, "needed": 2, "excess": 2}]


def test_role_stability_and_id_correlation():
    a = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
    assert all(r["fixed_role"] for r in M.role_stability(a))
    assert M.id_correlation(a) < -0.8
    assert np.isnan(M.id_correlation([[1, 1], [1, 1]]))


def test_clique_metrics():
    a = [[1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]]  # clique {0,1}; outsider 2 flips every turn, 3 steady
    c, o = M.partial_sums(a, [0, 1])
    assert c == [2, 1, 2] and o == [1, 2, 1]
    assert M.compensation(a, [0, 1]) == pytest.approx(-1.0)  # clique absorbs outsider changes
    gb = M.group_behavior(a, ["too_high", "too_low", "correct"], [0, 1])
    assert gb["clique"]["flip_rate"] == 0.5 and gb["outsiders"]["flip_rate"] == 0.5


def test_plan_parsing_and_following():
    assert M.parse_plan("Agents 0-1 choose 1 next turn") == M.Plan(ids=frozenset({0, 1}))
    assert M.parse_plan("agents 0, 2 and 3 will play 1") == M.Plan(ids=frozenset({0, 2, 3}))
    assert M.parse_plan("Two of us will switch to 1") == M.Plan(count=2)
    assert M.parse_plan("let's think") is None
    assert M.plan_followed(M.Plan(ids=frozenset({0, 1})), [1, 1, 0, 0], [0, 1, 2])
    assert not M.plan_followed(M.Plan(count=2), [1, 0, 0, 1], [0, 1])


def test_strategy_tags_and_agreement():
    assert M.tag_strategies("let's do a binary search, halve the range") == {"binary_search"}
    assert "linear_scan" in M.tag_strategies("add one more each turn")
    assert M.stated_vs_enacted({"binary_search"}, "binary_search_like") is True
    assert M.stated_vs_enacted({"binary_search"}, "random_walk") is False
    assert M.stated_vs_enacted({"id_based"}, "other") is None


# ---------------------------------------------------- acceptance: scripted baselines


@pytest.fixture(scope="module")
def baselines(tmp_path_factory):
    root = tmp_path_factory.mktemp("res")
    cfg = repo_root() / "experiments/experiment-configs/sum_scripted_baselines.yaml"
    run_experiment(str(cfg), {"dataset.num_episodes": 60}, results_root=root)
    return load_experiment(root / "sum_scripted_baselines")


def test_oracle_sits_at_log2(baselines):
    eps = episode_features(baselines)
    for n, g in eps[eps["agents_type"] == "oracle"].groupby("n"):
        assert g["turns"].max() <= math.ceil(math.log2(n + 2))
        assert g["success"].all() and (g["label"] == "binary_search_like").all()
        assert g["turns"].mean() == pytest.approx(g["oracle_turns"].mean())
        assert g["turns"].mean() <= math.log2(n + 2)
    idb = eps[eps["agents_type"] == "id_bisection"].groupby("n")["turns"].mean()
    orc = eps[eps["agents_type"] == "oracle"].groupby("n")["turns"].mean()
    assert np.allclose(idb, orc)
    rnd = eps[eps["agents_type"] == "random"].groupby("n")["turns"].mean()
    assert (rnd > 1.5 * orc).all() and rnd.is_monotonic_increasing


def test_report_and_plots_run(baselines, tmp_path):
    from analysis import plot_clique, plot_performance, plot_strategy
    from analysis.report import performance_table

    eps = episode_features(baselines)
    assert len(performance_table(eps)) == 12
    assert plot_performance.plot_success_at_k(eps, outdir=tmp_path)[0].exists()
    assert plot_performance.plot_turns_vs_x(eps, eps, outdir=tmp_path)[1].exists()
    assert plot_strategy.plot_labels(eps, outdir=tmp_path)[0].exists()
    assert plot_strategy.plot_trajectories(baselines, eps, outdir=tmp_path)[0].exists()
    assert plot_clique.plot_clique_vs_outsiders(eps, outdir=tmp_path)[0].exists()


def test_judge_with_mock_backend(tmp_path):
    cfg = repo_root() / "experiments/experiment-configs/sum_clique_smoke.yaml"
    (d, *_) = run_experiment(str(cfg), {"dataset.seed": 5, "sweep": {}}, dry_run=True, results_root=tmp_path)
    notes = load_notes(d)
    assert not notes.empty
    out = judge(notes, MockBackend(lambda r: "LABEL: binary_search"))
    assert set(out["stated_strategy"]) == {"binary_search"}
    assert parse_label("blah LABEL: nonsense") is None
    assert "binary_search" in build_prompt(["hi"])
