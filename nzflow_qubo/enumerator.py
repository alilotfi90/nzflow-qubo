"""
Reference enumerator for nowhere-zero Z_k flows.

Used as ground truth in tests (the (C3) check of Section 4.1). Iterates
over every labelling f : E -> {1, ..., k-1} and checks Kirchhoff modulo k
at every non-root vertex. Exponential in |E|.
"""

from __future__ import annotations
from itertools import product

from .graph import Multigraph, Orientation, signed_incidence


def enumerate_flows(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
) -> list[dict[int, int]]:
    """Return all nowhere-zero Z_k flows on G, as a list of edge -> residue dicts.

    For every labelling f : E -> {1, ..., k-1}, check that
    sum_e sigma_{v,e} * f(e) = 0 (mod k) for every v in V \\ {root}.
    Conservation at the root then follows from the global identity.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if D is None:
        D = Orientation.default(G)

    flows: list[dict[int, int]] = []
    # Precompute incidence patterns for non-root vertices
    non_root_vertices = [v for v in range(G.n) if v != root]
    # incidence[v_idx][e] = sigma_{v,e}
    incidence = [
        [signed_incidence(D, v, e) for e in range(G.m)]
        for v in non_root_vertices
    ]

    for labelling in product(range(1, k), repeat=G.m):
        ok = True
        for v_idx in range(len(non_root_vertices)):
            row = incidence[v_idx]
            s = 0
            for e in range(G.m):
                s += row[e] * labelling[e]
            if s % k != 0:
                ok = False
                break
        if ok:
            flows.append({e: labelling[e] for e in range(G.m)})
    return flows


def count_flows(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
) -> int:
    """Return the number of nowhere-zero Z_k flows on G.

    Equivalent to len(enumerate_flows(...)) but does not store the flows.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if D is None:
        D = Orientation.default(G)

    non_root_vertices = [v for v in range(G.n) if v != root]
    incidence = [
        [signed_incidence(D, v, e) for e in range(G.m)]
        for v in non_root_vertices
    ]

    count = 0
    for labelling in product(range(1, k), repeat=G.m):
        ok = True
        for row in incidence:
            s = sum(row[e] * labelling[e] for e in range(G.m))
            if s % k != 0:
                ok = False
                break
        if ok:
            count += 1
    return count
