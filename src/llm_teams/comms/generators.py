"""Graph generators: (n, params, rng) -> CommGraph, registered by name."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from llm_teams.comms.graph import CommGraph
from llm_teams.registry import Registry

GraphGenerator = Callable[[int, dict[str, Any], np.random.Generator], CommGraph]
GENERATORS: Registry[GraphGenerator] = Registry("graph generator")


def round_half_up(x: float) -> int:
    """Rounding used for m = round(p*n) and k = round(p*(n-1)) (avoids banker's rounding)."""
    return int(math.floor(x + 0.5 + 1e-9))


@GENERATORS.register("empty")
def empty(n: int, params: dict[str, Any], rng: np.random.Generator) -> CommGraph:
    return CommGraph(n)


@GENERATORS.register("complete")
def complete(n: int, params: dict[str, Any], rng: np.random.Generator) -> CommGraph:
    g = CommGraph(n, meta={"k": n - 1})
    for i in range(n):
        for j in range(i + 1, n):
            g.add_undirected(i, j)
    return g


@GENERATORS.register("ring")
def ring(n: int, params: dict[str, Any], rng: np.random.Generator) -> CommGraph:
    g = CommGraph(n)
    if n > 2:
        for i in range(n):
            g.add_undirected(i, (i + 1) % n)
    elif n == 2:
        g.add_undirected(0, 1)
    return g


@GENERATORS.register("clique")
def clique(n: int, params: dict[str, Any], rng: np.random.Generator) -> CommGraph:
    m = int(params["m"]) if "m" in params else round_half_up(float(params["p"]) * n)
    m = max(0, min(n, m))
    members = sorted(int(x) for x in rng.choice(n, size=m, replace=False)) if m > 0 else []
    g = CommGraph(n, meta={"m": m, "members": members if m > 1 else []})
    if m > 1:
        for a in members:
            for b in members:
                if a < b:
                    g.add_undirected(a, b)
    return g


@GENERATORS.register("random_k")
def random_k(n: int, params: dict[str, Any], rng: np.random.Generator) -> CommGraph:
    k = int(params["k"]) if "k" in params else round_half_up(float(params["p"]) * (n - 1))
    k = max(0, min(n - 1, k))
    g = CommGraph(n, meta={"k": k})
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for j in rng.choice(others, size=k, replace=False) if k > 0 else []:
            g.add_edge(int(j), i)
    return g


def build_graph(generator: str, n: int, params: dict[str, Any], rng: np.random.Generator) -> CommGraph:
    return GENERATORS.get(generator)(n, params, rng)
