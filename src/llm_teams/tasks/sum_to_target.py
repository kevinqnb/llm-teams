from __future__ import annotations

import math
from typing import Any

import numpy as np

from llm_teams.seeding import derive_rng
from llm_teams.tasks.base import TASKS, Task
from llm_teams.types import Action, ActionSpec, StepResult

FEEDBACKS = ("direction", "direction_and_distance", "exact_sum")


def bisect_guess(lo: int, hi: int) -> int:
    return lo + (hi - lo) // 2


def update_interval(lo: int, hi: int, guess: int, result: str) -> tuple[int, int]:
    if result == "too_high":
        return lo, min(hi, guess - 1)
    if result == "too_low":
        return max(lo, guess + 1), hi
    return guess, guess


def optimal_turn_bound(n: int) -> int:
    """Worst-case turns of optimal bisection over targets 0..n with direction feedback.

    Equals ceil(log2(n + 2)); the plan's ceil(log2(n + 1)) is one too small at e.g. n=1.
    """
    return math.ceil(math.log2(n + 2))


def oracle_turns(n: int, target: int) -> int:
    lo, hi, turns = 0, n, 0
    while True:
        turns += 1
        guess = bisect_guess(lo, hi)
        if guess == target:
            return turns
        lo, hi = update_interval(lo, hi, guess, "too_high" if guess > target else "too_low")


@TASKS.register("sum_to_target")
class SumToTarget(Task):
    name = "sum_to_target"

    @classmethod
    def make_instances(cls, dataset: dict[str, Any]) -> list[dict[str, Any]]:
        n = int(dataset["n"])
        seed = int(dataset.get("seed", 0))
        dist = dataset.get("target_dist", "uniform")
        feedback = dataset.get("feedback", "direction")
        if feedback not in FEEDBACKS:
            raise ValueError(f"unknown feedback '{feedback}'")
        out = []
        for i in range(int(dataset["num_episodes"])):
            if dist == "uniform":
                target = int(derive_rng(seed, "instance", i).integers(0, n + 1))
            elif dist == "fixed":
                target = int(dataset["target"])
            else:
                raise ValueError(f"unknown target_dist '{dist}'")
            if not 0 <= target <= n:
                raise ValueError(f"target {target} outside [0, {n}]")
            out.append({"n": n, "target": target, "feedback": feedback})
        return out

    @staticmethod
    def n_agents(instance: dict[str, Any]) -> int:
        return int(instance["n"])

    def reset(self, instance: dict[str, Any], rng: np.random.Generator) -> None:
        self.n = int(instance["n"])
        self.target = int(instance["target"])
        self.feedback_kind = instance.get("feedback", "direction")
        self.lo, self.hi = 0, self.n
        self.last_sum: int | None = None

    def describe(self, agent_id: int) -> dict[str, Any]:
        return {"task": self.name, "n": self.n, "feedback": self.feedback_kind}

    def action_space(self) -> ActionSpec:
        return ActionSpec(values=(0, 1))

    def step(self, actions: dict[int, Action]) -> StepResult:
        s = sum(actions.values())
        before = (self.lo, self.hi)
        if s == self.target:
            result = "correct"
        else:
            result = "too_high" if s > self.target else "too_low"
        if self.feedback_kind in ("exact_sum", "direction_and_distance"):
            # sum or distance + direction pins the target down exactly
            self.lo = self.hi = self.target
        else:
            self.lo, self.hi = update_interval(self.lo, self.hi, s, result)
        fb: dict[str, Any] = {"result": result}
        if self.feedback_kind == "direction_and_distance":
            fb["distance"] = abs(s - self.target)
        elif self.feedback_kind == "exact_sum":
            fb["sum"] = s
        self.last_sum = s
        info = {"sum": s, "interval_before": list(before), "interval_after": [self.lo, self.hi]}
        return StepResult({i: dict(fb) for i in actions}, result == "correct", info)

    def state_dict(self) -> dict[str, Any]:
        return {"target": self.target, "sum": self.last_sum, "interval": [self.lo, self.hi]}

    def reference_solution(self) -> dict[str, Any] | None:
        return {
            "optimal_turns": optimal_turn_bound(self.n),
            "oracle_turns": oracle_turns(self.n, self.target),
        }
