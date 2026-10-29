"""Pure metric functions for sum_to_target trajectories (no pandas, no I/O).

Conventions: a trajectory is the list of team sums `sums[k]` (k = 0 is turn 1) with the matching
feedback `results[k]` in {"too_high", "too_low", "correct"}. Action matrices are `actions[k][i]`
(turn k, agent i). Interval logic assumes direction feedback.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llm_teams.tasks.sum_to_target import update_interval

Interval = tuple[int, int]

# ------------------------------------------------------------------ performance


def success_at_k(turns: Sequence[int], success: Sequence[bool], k_max: int) -> list[float]:
    """Fraction of episodes solved within k turns, k = 1..k_max (failures never count)."""
    n = len(turns)
    return [sum(1 for t, s in zip(turns, success, strict=True) if s and t <= k) / n
            for k in range(1, k_max + 1)]


def kaplan_meier(durations: Sequence[int], events: Sequence[bool]) -> list[tuple[int, float]]:
    """Survival curve S(t) = P(not yet solved after t turns); events=False means censored."""
    surv, out = 1.0, []
    for t in sorted({d for d, e in zip(durations, events, strict=True) if e}):
        at_risk = sum(1 for d in durations if d >= t)
        solved = sum(1 for d, e in zip(durations, events, strict=True) if e and d == t)
        surv *= 1 - solved / at_risk
        out.append((t, surv))
    return out


def normalized_turns(turns: int, optimal_turns: int) -> float:
    return turns / optimal_turns


# ------------------------------------------------------------------ group-level strategy


def intervals_before(n: int, sums: Sequence[int], results: Sequence[str]) -> list[Interval]:
    """Consistent target interval [lo, hi] before each turn."""
    lo, hi, out = 0, n, []
    for s, r in zip(sums, results, strict=True):
        out.append((lo, hi))
        lo, hi = update_interval(lo, hi, s, r)
    return out


def consistency_rate(sums: Sequence[int], ivals: Sequence[Interval]) -> float:
    """Fraction of turns whose sum lay inside the still-possible interval."""
    return float(np.mean([lo <= s <= hi for s, (lo, hi) in zip(sums, ivals, strict=True)]))


def bisection_scores(sums: Sequence[int], ivals: Sequence[Interval]) -> list[float]:
    """1 - d / ((hi - lo) / 2) per turn, clipped to [-1, 1]; 1 = the best integer midpoint.

    d = |s - mid| minus the unavoidable 0.5 when the interval has an even number of values (so
    perfect bisection scores exactly 1; the plan's raw formula would give e.g. 0 on [3, 4]).
    """
    out = []
    for s, (lo, hi) in zip(sums, ivals, strict=True):
        if hi == lo:
            out.append(1.0 if s == lo else -1.0)
        else:
            d = max(0.0, abs(s - (lo + hi) / 2) - 0.5 * ((hi - lo) % 2))
            out.append(float(np.clip(1 - d / ((hi - lo) / 2), -1, 1)))
    return out


def max_info_gain(size: int) -> float:
    """Worst-case-outcome gain (bits) of the best possible guess in an interval of `size` values."""
    return math.log2(size / max(1, math.ceil((size - 1) / 2))) if size > 1 else 0.0


def info_gains(n: int, sums: Sequence[int], results: Sequence[str]) -> list[tuple[float, float]]:
    """(realized, max) information gain in bits per non-final-success turn.

    Realized = log2(|interval before| / |interval after|); a lucky success is excluded.
    """
    ivals = intervals_before(n, sums, results)
    out = []
    for (lo, hi), s, r in zip(ivals, sums, results, strict=True):
        if r == "correct":
            continue
        lo2, hi2 = update_interval(lo, hi, s, r)
        size_after = max(1, hi2 - lo2 + 1)
        out.append((math.log2((hi - lo + 1) / size_after), max_info_gain(hi - lo + 1)))
    return out


def step_profile(sums: Sequence[int]) -> list[int]:
    return [abs(b - a) for a, b in zip(sums, sums[1:], strict=False)]


def oscillations(sums: Sequence[int]) -> int:
    """Reversals in the direction of the sum's movement (zero steps ignored)."""
    signs = [int(np.sign(b - a)) for a, b in zip(sums, sums[1:], strict=False) if b != a]
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)


def overshoots(results: Sequence[str]) -> int:
    """Number of times the sum crossed the target (too_high <-> too_low on consecutive turns)."""
    dirs = [r for r in results if r != "correct"]
    return sum(1 for a, b in zip(dirs, dirs[1:], strict=False) if a != b)


def max_run_same(sums: Sequence[int]) -> int:
    best = run = 0
    prev = None
    for s in sums:
        run = run + 1 if s == prev else 1
        best, prev = max(best, run), s
    return best


@dataclass(frozen=True)
class Thresholds:
    """Documented cut-offs for `classify_episode`. Checked in the order listed."""

    stuck_run: int = 3                 # same sum on >= this many consecutive turns -> stuck
    bisect_consistency: float = 0.8    # binary_search_like: consistency_rate >= this ...
    bisect_score: float = 0.6          # ... and mean bisection score >= this
    linear_step1: float = 0.7          # linear_scan: >= this fraction of steps are +-1 ...
    linear_monotone: float = 0.7       # ... and >= this fraction of steps point the same way
    linear_min_steps: int = 2
    random_consistency: float = 0.5    # random_walk: consistency below this, or
    random_oscillation: float = 0.6    # reversal rate (of consecutive step pairs) at least this


DEFAULT_THRESHOLDS = Thresholds()
LABELS = ("binary_search_like", "linear_scan", "random_walk", "stuck", "other")


def classify_episode(n: int, sums: Sequence[int], results: Sequence[str],
                     th: Thresholds = DEFAULT_THRESHOLDS) -> str:
    """Label an episode's group-level strategy from its sum trajectory and feedback."""
    ivals = intervals_before(n, sums, results)
    if max_run_same(sums) >= th.stuck_run:
        return "stuck"
    cons = consistency_rate(sums, ivals)
    score = float(np.mean(bisection_scores(sums, ivals)))
    if cons >= th.bisect_consistency and score >= th.bisect_score:
        return "binary_search_like"
    steps = [b - a for a, b in zip(sums, sums[1:], strict=False)]
    if len(steps) >= th.linear_min_steps:
        small = np.mean([abs(d) == 1 for d in steps])
        signs = [int(np.sign(d)) for d in steps if d]
        monotone = max(signs.count(1), signs.count(-1)) / len(signs) if signs else 0.0
        if small >= th.linear_step1 and monotone >= th.linear_monotone:
            return "linear_scan"
    osc_rate = oscillations(sums) / max(1, len(steps) - 1)
    if cons < th.random_consistency or (len(steps) >= 3 and osc_rate >= th.random_oscillation):
        return "random_walk"
    return "other"


# ------------------------------------------------------------------ agent-level behavior


def flip_rates(actions: Sequence[Sequence[int]]) -> list[float]:
    """Per agent: fraction of turn transitions on which the agent changed its choice."""
    a = np.asarray(actions)
    if len(a) < 2:
        return [float("nan")] * a.shape[1]
    return list((a[1:] != a[:-1]).mean(axis=0))


@dataclass
class FlipCounts:
    consistent: int = 0     # flipped in the direction the feedback asked for
    inconsistent: int = 0   # flipped the wrong way
    total: int = 0

    @property
    def consistent_fraction(self) -> float:
        return self.consistent / self.total if self.total else float("nan")


def direction_flips(actions: Sequence[Sequence[int]], results: Sequence[str],
                    agents: Sequence[int] | None = None) -> FlipCounts:
    """After "too_high" only 1->0 flips help; after "too_low" only 0->1 flips help."""
    a = np.asarray(actions)
    idx = list(range(a.shape[1])) if agents is None else list(agents)
    c = FlipCounts()
    for k in range(len(a) - 1):
        for i in idx:
            if a[k + 1, i] == a[k, i]:
                continue
            c.total += 1
            up = a[k + 1, i] > a[k, i]
            good = (results[k] == "too_low" and up) or (results[k] == "too_high" and not up)
            c.consistent += int(good)
            c.inconsistent += int(not good)
    return c


def overcorrection(n: int, actions: Sequence[Sequence[int]], results: Sequence[str]) -> list[dict]:
    """Per transition: agents that flipped vs. flips needed to land on the new interval midpoint.

    `excess` = flipped - needed; herding shows up as large excess (everyone flips at once).
    """
    a = np.asarray(actions)
    sums = a.sum(axis=1)
    ivals = intervals_before(n, list(sums), results)
    out = []
    for k in range(len(a) - 1):
        if results[k] == "correct":
            break
        lo, hi = update_interval(*ivals[k], int(sums[k]), results[k])
        needed = abs(int(round((lo + hi) / 2)) - int(sums[k]))
        flipped = int((a[k + 1] != a[k]).sum())
        out.append({"turn": k + 1, "flipped": flipped, "needed": needed, "excess": flipped - needed})
    return out


def role_stability(actions: Sequence[Sequence[int]]) -> list[dict]:
    """Per agent: share of turns on the agent's majority action, always-0/1 flag, mean action."""
    a = np.asarray(actions)
    out = []
    for i in range(a.shape[1]):
        mean = float(a[:, i].mean())
        out.append({"agent": i, "mean_action": mean, "stability": max(mean, 1 - mean),
                    "fixed_role": bool(len(a) > 1 and mean in (0.0, 1.0))})
    return out


def id_correlation(actions: Sequence[Sequence[int]]) -> float:
    """Pearson correlation between agent id and its mean action (nan if either is constant).

    Strongly negative = low ids play 1 more (the signature of ID-based bisection).
    """
    a = np.asarray(actions)
    mean = a.mean(axis=0)
    if len(mean) < 2 or np.ptp(mean) == 0:
        return float("nan")
    return float(np.corrcoef(np.arange(a.shape[1]), mean)[0, 1])


# ------------------------------------------------------------------ clique vs outsiders


def partial_sums(actions: Sequence[Sequence[int]], members: Sequence[int]) -> tuple[list[int], list[int]]:
    """(clique sum, outsider sum) per turn."""
    a = np.asarray(actions)
    mask = np.zeros(a.shape[1], dtype=bool)
    mask[list(members)] = True
    return list(a[:, mask].sum(axis=1)), list(a[:, ~mask].sum(axis=1))


def compensation(actions: Sequence[Sequence[int]], members: Sequence[int]) -> float:
    """Correlation of per-turn changes in clique sum and outsider sum.

    Strongly negative means the clique absorbs outsider noise (acts like a controller).
    """
    c, o = partial_sums(actions, members)
    dc, do = np.diff(c), np.diff(o)
    if len(dc) < 2 or np.ptp(dc) == 0 or np.ptp(do) == 0:
        return float("nan")
    return float(np.corrcoef(dc, do)[0, 1])


def group_behavior(actions: Sequence[Sequence[int]], results: Sequence[str],
                   members: Sequence[int]) -> dict[str, dict[str, float]]:
    """Flip rate and direction-consistency for clique members vs. outsiders."""
    a = np.asarray(actions)
    others = [i for i in range(a.shape[1]) if i not in set(members)]
    out = {}
    for name, ids in (("clique", list(members)), ("outsiders", others)):
        if not ids:
            continue
        fr = [flip_rates(a)[i] for i in ids]
        out[name] = {"flip_rate": float(np.nanmean(fr)) if len(a) > 1 else float("nan"),
                     "direction_consistent": direction_flips(a, results, ids).consistent_fraction}
    return out


# ------------------------------------------------------------------ stated strategy (notes / reasoning)

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_NUM_RE = "|".join(_NUM) + r"|\d+"
_ID_RANGE = re.compile(r"agents?\s+(\d+)\s*(?:-|to|through)\s*(\d+)\s+(?:will\s+|should\s+)?"
                       r"(?:choose|pick|play|submit|select|answer)\s+1", re.I)
_ID_LIST = re.compile(r"agents?\s+((?:\d+\s*(?:,|and|&)?\s*)+?)\s*(?:will\s+|should\s+)?"
                      r"(?:choose|pick|play|submit|select|answer)\s+1", re.I)
_COUNT = re.compile(rf"\b({_NUM_RE})\s+of\s+us\s+(?:will\s+|should\s+)?"
                    r"(?:switch|choose|pick|play|submit|select|go)\w*\s*(?:to\s+)?1?", re.I)


@dataclass(frozen=True)
class Plan:
    ids: frozenset[int] | None = None   # agents said to play 1
    count: int | None = None            # number of clique members said to play 1


def parse_plan(text: str) -> Plan | None:
    if m := _ID_RANGE.search(text):
        return Plan(ids=frozenset(range(int(m.group(1)), int(m.group(2)) + 1)))
    if m := _ID_LIST.search(text):
        ids = frozenset(int(x) for x in re.findall(r"\d+", m.group(1)))
        if ids:
            return Plan(ids=ids)
    if m := _COUNT.search(text):
        w = m.group(1).lower()
        return Plan(count=_NUM.get(w) or int(w))
    return None


def plan_followed(plan: Plan, next_actions: Sequence[int], members: Sequence[int]) -> bool:
    """Did the clique's next-turn actions match the stated plan?"""
    if plan.ids is not None:
        ones = {i for i in members if next_actions[i] == 1}
        return plan.ids & set(members) == ones
    assert plan.count is not None
    return sum(next_actions[i] for i in members) == plan.count


STRATEGY_PATTERNS = {
    "binary_search": r"binary|bisect|halv|half\b|midpoint|middle|mid-?point",
    "linear_scan": r"one by one|one at a time|add one|one more|increase by (1|one)|increment|step by step",
    "id_based": r"\bid\b.*(<|below|less|lower)|lower ids?|higher ids?|ids? (0|zero)",
    "random": r"random|coin|guess",
}
STATED_TO_ENACTED = {"binary_search": "binary_search_like", "linear_scan": "linear_scan",
                     "random": "random_walk"}


def tag_strategies(text: str) -> set[str]:
    return {k for k, pat in STRATEGY_PATTERNS.items() if re.search(pat, text, re.I)}


def stated_vs_enacted(stated: set[str], enacted: str) -> bool | None:
    """True/False if a stated strategy with a comparable enacted label exists, else None."""
    comparable = {STATED_TO_ENACTED[s] for s in stated if s in STATED_TO_ENACTED}
    if not comparable:
        return None
    return enacted in comparable
