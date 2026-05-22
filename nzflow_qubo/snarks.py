"""
Verified snark constructions for the benchmark suite in Section 4.4 of the paper.

A *snark* is a cubic, bridgeless graph with chromatic index 4 (i.e.,
NOT 3-edge-colorable) and girth >= 5. By the classical equivalence of
4-flows with 3-edge-colorings on cubic graphs, every snark satisfies
phi(G) >= 5; consequently snarks are the canonical near-threshold
instances of the nowhere-zero 4-flow problem.

This module uses CONSTRUCTIVE definitions (not pasted edge lists). Each
constructor is followed by an `is_snark` check during table generation
so any future modification that breaks the construction fails loudly.

Currently provided:
    petersen_snark()  : Petersen graph (10v, 15e; smallest snark).
    flower_snark(n)   : Isaacs flower snark J_n for odd n >= 5 (4n vertices,
                        6n edges). Verified snarks for all odd n >= 5.

Not included:
    Blanusa snarks B_1, B_2 -- We do not include these because we have
    been unable to verify an edge list against a primary source within
    the revision timeline (an attempted LCF reconstruction failed the
    cubic-graph check). If a verified construction is obtained later it
    can be added here behind the same `is_snark` check.

References:
    R. Isaacs, "Infinite families of nontrivial trivalent graphs which
    are not Tait colorable", Amer. Math. Monthly, 82(3):221-239, 1975.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

from .graph import Multigraph


def petersen_snark() -> Multigraph:
    """The Petersen graph. 10 vertices, 15 edges, girth 5."""
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),     # outer 5-cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),     # inner 5-cycle (pentagram)
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),     # spokes
    )
    return Multigraph(n=10, edges=edges)


def flower_snark(n: int) -> Multigraph:
    """Isaacs flower snark J_n for odd n >= 3.

    For odd n >= 5 this is a snark. For n = 3 the construction yields
    a cubic bridgeless graph that is 3-edge-colorable (not a snark); we
    still allow n = 3 so the constructor's behavior is uniform, and we
    leave the snark-or-not decision to `is_snark`.

    Construction:
        Vertices: a_i, b_i, c_i, d_i for i = 0, ..., n-1  (4n vertices).
        Edges:
          - Star edges at each a_i: (a_i, b_i), (a_i, c_i), (a_i, d_i)
          - b-cycle: (b_i, b_{i+1}) for all i (mod n)
          - twisted c-d cycle: (c_i, d_{i+1}) and (d_i, c_{i+1})
            for all i (mod n)

    For odd n the c and d vertices together form a single 2n-cycle with
    one half-twist; this twist is what destroys 3-edge-colorability
    (Isaacs 1975, Theorem 1.1).
    """
    if not isinstance(n, int) or n < 3:
        raise ValueError(f"flower_snark requires integer n >= 3, got {n!r}")
    if n % 2 == 0:
        raise ValueError(
            f"flower_snark is defined for odd n; got n={n}. "
            f"For even n the construction yields a non-snark."
        )

    def a(i: int) -> int: return 4 * i
    def b(i: int) -> int: return 4 * i + 1
    def c(i: int) -> int: return 4 * i + 2
    def d(i: int) -> int: return 4 * i + 3

    edges: list[tuple[int, int]] = []
    for i in range(n):
        edges.append((a(i), b(i)))
        edges.append((a(i), c(i)))
        edges.append((a(i), d(i)))
    for i in range(n):
        j = (i + 1) % n
        edges.append((b(i), b(j)))
        edges.append((c(i), d(j)))
        edges.append((d(i), c(j)))
    return Multigraph(n=4 * n, edges=tuple(edges))


def girth(G: Multigraph) -> float:
    """Length of the shortest cycle of G, or float('inf') if G is acyclic.

    BFS from every vertex; a non-tree edge of length d(x)+d(y)+1 closes
    a cycle of that length.
    """
    n = G.n
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(G.edges):
        adj[u].append((v, e))
        adj[v].append((u, e))
    best: float = float("inf")
    for start in range(n):
        dist = [-1] * n
        parent_edge = [-1] * n
        dist[start] = 0
        Q: deque[int] = deque([start])
        while Q:
            x = Q.popleft()
            for (y, e) in adj[x]:
                if e == parent_edge[x]:
                    continue
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    parent_edge[y] = e
                    Q.append(y)
                else:
                    cyc = dist[x] + dist[y] + 1
                    if cyc < best:
                        best = cyc
    return best


def is_3_edge_colorable(G: Multigraph) -> bool:
    """Brute-force 3-edge-coloring search via backtracking.

    Tractable for m up to roughly 40-45 edges in seconds; worst case is
    3^m but heavy pruning helps. Used to certify non-3-edge-colorability
    (i.e., snark status) of cubic bridgeless graphs.
    """
    m = G.m
    if m == 0:
        return True
    incident: list[list[int]] = [[] for _ in range(G.n)]
    for e, (u, v) in enumerate(G.edges):
        incident[u].append(e)
        incident[v].append(e)
    color = [-1] * m

    def backtrack(e: int) -> bool:
        if e == m:
            return True
        u, v = G.edges[e]
        used: set[int] = set()
        for f in incident[u]:
            if color[f] >= 0:
                used.add(color[f])
        for f in incident[v]:
            if color[f] >= 0:
                used.add(color[f])
        for c in (0, 1, 2):
            if c not in used:
                color[e] = c
                if backtrack(e + 1):
                    return True
                color[e] = -1
        return False

    return backtrack(0)


def is_snark(G: Multigraph, *, check_colorable: bool = True) -> bool:
    """Return True iff G satisfies the snark conditions:

      - cubic (every vertex has degree 3),
      - bridgeless,
      - girth >= 5,
      - 3-edge-chromatic index 4 (i.e., NOT 3-edge-colorable).

    Set check_colorable=False to skip the (potentially slow) backtracking
    check when verifying very large snarks whose colorability is known
    from the literature.
    """
    if G.n == 0:
        return False
    for v in range(G.n):
        if G.degree(v) != 3:
            return False
    if G.has_bridge():
        return False
    if girth(G) < 5:
        return False
    if check_colorable and is_3_edge_colorable(G):
        return False
    return True


# Curated snark suite for table generation.
#
# Format: (display_name, builder, expected_n, expected_m, do_colorability_check)
#
# SNARK_SUITE contains the three snarks reported in Table 7 of the paper:
# Petersen, the Isaacs flower snark J_5, and the Isaacs flower snark J_7.
# For all three the 3-edge-colorability check is performed at table-generation
# time as an audit (do_color_check=True) — none of the three are colorable, so
# each row's snark status is re-verified before the table is emitted.
SNARK_SUITE: list[tuple[str, Callable[[], Multigraph], int, int, bool]] = [
    ("Petersen",          petersen_snark,             10, 15, True),
    ("Flower snark $J_5$", lambda: flower_snark(5),    20, 30, True),
    ("Flower snark $J_7$", lambda: flower_snark(7),    28, 42, True),
]


__all__ = [
    "petersen_snark",
    "flower_snark",
    "girth",
    "is_3_edge_colorable",
    "is_snark",
    "SNARK_SUITE",
]
