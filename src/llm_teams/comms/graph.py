"""Directed communication graph over agent ids. Edge j -> i means agent i can see j."""

from __future__ import annotations

from typing import Any


class CommGraph:
    def __init__(self, n: int, edges: set[tuple[int, int]] | None = None,
                 meta: dict[str, Any] | None = None) -> None:
        self.n = n
        self._in: dict[int, set[int]] = {i: set() for i in range(n)}
        self.meta: dict[str, Any] = dict(meta or {})
        for src, dst in edges or ():
            self.add_edge(src, dst)

    def add_edge(self, src: int, dst: int) -> None:
        if src == dst:
            raise ValueError("self-loops are not allowed")
        self._in[dst].add(src)

    def add_undirected(self, a: int, b: int) -> None:
        self.add_edge(a, b)
        self.add_edge(b, a)

    def neighbors_in(self, i: int) -> set[int]:
        return set(self._in[i])

    def neighbors_out(self, i: int) -> set[int]:
        return {dst for dst, srcs in self._in.items() if i in srcs}

    def is_isolated(self, i: int) -> bool:
        return not self._in[i] and not self.neighbors_out(i)

    def edges(self) -> set[tuple[int, int]]:
        return {(src, dst) for dst, srcs in self._in.items() for src in srcs}

    def components(self) -> list[list[int]]:
        """Weakly connected components, singletons included."""
        adj = {i: self._in[i] | self.neighbors_out(i) for i in range(self.n)}
        seen: set[int] = set()
        comps: list[list[int]] = []
        for start in range(self.n):
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u] - seen:
                    seen.add(v)
                    stack.append(v)
            comps.append(sorted(comp))
        return comps

    def group_of(self, i: int) -> list[int]:
        return next(c for c in self.components() if i in c)

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "edges": sorted([s, d] for s, d in self.edges()), "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CommGraph:
        return cls(d["n"], {(s, t) for s, t in d["edges"]}, d.get("meta"))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CommGraph) and self.n == other.n and self.edges() == other.edges()
