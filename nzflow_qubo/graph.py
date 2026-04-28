"""
Loopless multigraphs, orientations, and signed incidence.

Implements the structures from Section 2 of the paper:
- Loopless multigraphs as edge-indexed objects (parallel edges allowed)
- Orientations as a choice of (tail, head) per edge
- Signed incidence sigma_{v,e} in {-1, 0, +1}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Multigraph:
    """A loopless multigraph G = (V, E).

    Vertices are integers 0, 1, ..., n-1. Edges are stored as a list of
    (u, v) pairs with u != v. Parallel edges are permitted: two distinct
    indices may carry the same endpoint pair.
    """

    n: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"vertex count must be >= 1, got {self.n}")
        for i, (u, v) in enumerate(self.edges):
            if not (0 <= u < self.n and 0 <= v < self.n):
                raise ValueError(f"edge {i}=({u},{v}) out of range [0,{self.n})")
            if u == v:
                raise ValueError(f"edge {i}=({u},{v}) is a loop; loops are not allowed")

    @property
    def m(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def degree(self, v: int) -> int:
        """Degree d(v): number of edges incident to v.

        Parallel edges contribute multiplicity. Self-loops would contribute
        two but loops are forbidden.
        """
        return sum(1 for (a, b) in self.edges if a == v or b == v)

    def is_connected(self) -> bool:
        if self.n == 0:
            return True
        adj: list[set[int]] = [set() for _ in range(self.n)]
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        seen = {0}
        stack = [0]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return len(seen) == self.n

    def has_bridge(self) -> bool:
        """Return True if G has a bridge (Definition 2.10)."""
        if not self.is_connected():
            # A disconnected graph counts as having "bridges" in the usual
            # sense for flow purposes: any edge in a tree component is a
            # bridge. We use a stricter check: edge e is a bridge iff
            # G - e has more components than G.
            base_components = _count_components(self.n, self.edges)
            for i in range(self.m):
                rest = self.edges[:i] + self.edges[i+1:]
                if _count_components(self.n, rest) > base_components:
                    return True
            return False
        for i in range(self.m):
            rest = self.edges[:i] + self.edges[i+1:]
            if _count_components(self.n, rest) > 1:
                return True
        return False


def _count_components(n: int, edges: Iterable[tuple[int, int]]) -> int:
    """Helper: number of connected components of (V={0..n-1}, edges)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for u, v in edges:
        union(u, v)
    roots = {find(x) for x in range(n)}
    return len(roots)


@dataclass(frozen=True)
class Orientation:
    """A choice of head and tail per edge.

    `tails[i]` is the tail of edge i; `heads[i]` is its head. Together
    these define D : E -> oriented edges (Definition 2.6).
    """

    tails: tuple[int, ...]
    heads: tuple[int, ...]

    @classmethod
    def default(cls, G: Multigraph) -> "Orientation":
        """Use the (u, v) pair as written: tail = u, head = v."""
        tails = tuple(u for (u, _) in G.edges)
        heads = tuple(v for (_, v) in G.edges)
        return cls(tails=tails, heads=heads)

    @classmethod
    def from_bitmask(cls, G: Multigraph, mask: int) -> "Orientation":
        """Build an orientation by flipping each edge whose bit in `mask` is 1.

        Used for exhaustive orientation sweeps in robustness tests.
        """
        tails: list[int] = []
        heads: list[int] = []
        for i, (u, v) in enumerate(G.edges):
            if (mask >> i) & 1:
                tails.append(v)
                heads.append(u)
            else:
                tails.append(u)
                heads.append(v)
        return cls(tails=tuple(tails), heads=tuple(heads))


def signed_incidence(D: Orientation, v: int, e: int) -> int:
    """Return sigma_{v,e} in {-1, 0, +1} (Definition 2.7)."""
    if D.tails[e] == v:
        return +1
    if D.heads[e] == v:
        return -1
    return 0
