"""
Standard benchmark graphs used in Section 4 of the paper.
"""

from __future__ import annotations

from .graph import Multigraph


def K(n: int) -> Multigraph:
    """Complete graph K_n."""
    edges = tuple((i, j) for i in range(n) for j in range(i + 1, n))
    return Multigraph(n=n, edges=edges)


def K_bipartite(m: int, n: int) -> Multigraph:
    """Complete bipartite graph K_{m,n}."""
    edges = tuple((i, m + j) for i in range(m) for j in range(n))
    return Multigraph(n=m + n, edges=edges)


def cycle(n: int) -> Multigraph:
    """Cycle graph C_n."""
    if n < 3:
        raise ValueError(f"cycle length must be >= 3, got {n}")
    edges = tuple((i, (i + 1) % n) for i in range(n))
    return Multigraph(n=n, edges=edges)


def cube() -> Multigraph:
    """The 3-cube Q_3: 8 vertices, 12 edges."""
    edges = tuple(
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if bin(i ^ j).count("1") == 1
    )
    return Multigraph(n=8, edges=edges)


def petersen() -> Multigraph:
    """The Petersen graph: 10 vertices, 15 edges, flow number 5."""
    # Outer 5-cycle: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    edges = tuple(outer + inner + spokes)
    return Multigraph(n=10, edges=edges)


def theta_3() -> Multigraph:
    """Theta_3: two vertices joined by three parallel edges. Flow number 3."""
    return Multigraph(n=2, edges=((0, 1), (0, 1), (0, 1)))


def K4_doubled() -> Multigraph:
    """K_4 with every edge doubled. 4 vertices, 12 edges. Flow number 3."""
    base = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    edges = tuple(base + base)
    return Multigraph(n=4, edges=edges)
