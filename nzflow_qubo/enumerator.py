"""
Reference and fast enumerators for nowhere-zero Z_k flows.

The default path uses a spanning-tree extension method: choose values on the
non-tree edges (cycle-space coordinates), then uniquely solve the tree-edge
values by leaf elimination. For a connected component with cycle rank
beta = m - n + 1, this reduces the search space from (k-1)^m to (k-1)^beta.

A brute-force enumerator is kept for validation and for the original (C3)
interpretation in the paper.
"""

from __future__ import annotations
from collections import deque
from itertools import product

from .graph import Multigraph, Orientation, signed_incidence


def _components(G: Multigraph) -> list[tuple[list[int], list[int]]]:
    """Return connected components as (vertices, edge_indices) pairs."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(G.n)]
    for e, (u, v) in enumerate(G.edges):
        adj[u].append((v, e))
        adj[v].append((u, e))

    seen = [False] * G.n
    comps: list[tuple[list[int], list[int]]] = []
    for start in range(G.n):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        verts: list[int] = []
        edge_set: set[int] = set()
        while q:
            x = q.popleft()
            verts.append(x)
            for y, e in adj[x]:
                edge_set.add(e)
                if not seen[y]:
                    seen[y] = True
                    q.append(y)
        comps.append((verts, sorted(edge_set)))
    return comps


def _spanning_tree_edges(
    G: Multigraph, vertices: list[int], edge_indices: list[int]
) -> tuple[list[int], list[int]]:
    """Return (tree_edges, chord_edges) for one connected component."""
    vertex_set = set(vertices)
    start = vertices[0]
    adj: dict[int, list[tuple[int, int]]] = {v: [] for v in vertices}
    for e in edge_indices:
        u, v = G.edges[e]
        if u in vertex_set and v in vertex_set:
            adj[u].append((v, e))
            adj[v].append((u, e))

    seen = {start}
    q = deque([start])
    tree_edges: list[int] = []
    tree_set: set[int] = set()
    while q:
        x = q.popleft()
        for y, e in adj[x]:
            if y not in seen:
                seen.add(y)
                q.append(y)
                tree_edges.append(e)
                tree_set.add(e)
    chord_edges = [e for e in edge_indices if e not in tree_set]
    return tree_edges, chord_edges


def _extend_component_from_chords(
    G: Multigraph,
    k: int,
    D: Orientation,
    vertices: list[int],
    edge_indices: list[int],
    chord_assignment: dict[int, int],
) -> dict[int, int] | None:
    """Given nonzero values on the chord edges, solve the tree edges uniquely.

    Returns a full component flow dict e -> residue in {1,...,k-1}, or None if
    the unique extension uses 0 on some tree edge.
    """
    if not edge_indices:
        return {}

    tree_edges, chord_edges = _spanning_tree_edges(G, vertices, edge_indices)
    vertex_set = set(vertices)

    residual: dict[int, int] = {v: 0 for v in vertices}
    flow: dict[int, int] = {}
    for e in chord_edges:
        val = chord_assignment[e] % k
        if val == 0:
            return None
        flow[e] = val
        u, v = G.edges[e]
        if u in vertex_set:
            residual[u] = (residual[u] + signed_incidence(D, u, e) * val) % k
        if v in vertex_set:
            residual[v] = (residual[v] + signed_incidence(D, v, e) * val) % k

    tree_adj: dict[int, list[tuple[int, int]]] = {v: [] for v in vertices}
    tree_deg: dict[int, int] = {v: 0 for v in vertices}
    for e in tree_edges:
        u, v = G.edges[e]
        tree_adj[u].append((v, e))
        tree_adj[v].append((u, e))
        tree_deg[u] += 1
        tree_deg[v] += 1

    unresolved = set(tree_edges)
    q = deque(v for v in vertices if tree_deg[v] == 1)

    while unresolved and q:
        leaf = q.popleft()
        if tree_deg[leaf] != 1:
            continue
        other = None
        e_leaf = None
        for nbr, e in tree_adj[leaf]:
            if e in unresolved:
                other = nbr
                e_leaf = e
                break
        if other is None or e_leaf is None:
            continue

        sigma_leaf = signed_incidence(D, leaf, e_leaf)
        x = (-sigma_leaf * residual[leaf]) % k
        if x == 0:
            return None
        flow[e_leaf] = x

        unresolved.remove(e_leaf)
        tree_deg[leaf] -= 1
        tree_deg[other] -= 1

        sigma_other = signed_incidence(D, other, e_leaf)
        residual[other] = (residual[other] + sigma_other * x) % k
        residual[leaf] = 0

        if tree_deg[other] == 1:
            q.append(other)

    if unresolved:
        return None

    for v in vertices:
        total = sum(signed_incidence(D, v, e) * flow[e] for e in edge_indices) % k
        if total != 0:
            return None
    return flow


def _enumerate_component_flows(
    G: Multigraph, k: int, D: Orientation, vertices: list[int], edge_indices: list[int]
) -> list[dict[int, int]]:
    if not edge_indices:
        return [{}]
    tree_edges, chord_edges = _spanning_tree_edges(G, vertices, edge_indices)
    if not chord_edges:
        return []
    flows: list[dict[int, int]] = []
    for vals in product(range(1, k), repeat=len(chord_edges)):
        assignment = {e: vals[i] for i, e in enumerate(chord_edges)}
        ext = _extend_component_from_chords(G, k, D, vertices, edge_indices, assignment)
        if ext is not None:
            flows.append(ext)
    return flows


def enumerate_flows_fast(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
) -> list[dict[int, int]]:
    """Enumerate all nowhere-zero Z_k flows using cycle-space coordinates.

    The `root` parameter is accepted for API compatibility; the result is
    independent of root.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if D is None:
        D = Orientation.default(G)

    comp_flows: list[list[dict[int, int]]] = []
    for verts, edge_indices in _components(G):
        flows = _enumerate_component_flows(G, k, D, verts, edge_indices)
        if not flows:
            if edge_indices:
                return []
            flows = [{}]
        comp_flows.append(flows)

    out: list[dict[int, int]] = [{}]
    for flows in comp_flows:
        new_out: list[dict[int, int]] = []
        for base in out:
            for f in flows:
                merged = dict(base)
                merged.update(f)
                new_out.append(merged)
        out = new_out
    return out


def count_flows_fast(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
) -> int:
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if D is None:
        D = Orientation.default(G)

    total = 1
    for verts, edge_indices in _components(G):
        if not edge_indices:
            continue
        flows = _enumerate_component_flows(G, k, D, verts, edge_indices)
        if not flows:
            return 0
        total *= len(flows)
    return total


def enumerate_flows_bruteforce(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
) -> list[dict[int, int]]:
    """Brute-force enumeration over all labelings f:E->{1,...,k-1}."""
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if D is None:
        D = Orientation.default(G)

    flows: list[dict[int, int]] = []
    non_root_vertices = [v for v in range(G.n) if v != root]
    incidence = [[signed_incidence(D, v, e) for e in range(G.m)] for v in non_root_vertices]

    for labelling in product(range(1, k), repeat=G.m):
        ok = True
        for row in incidence:
            s = 0
            for e in range(G.m):
                s += row[e] * labelling[e]
            if s % k != 0:
                ok = False
                break
        if ok:
            flows.append({e: labelling[e] for e in range(G.m)})
    return flows


def count_flows_bruteforce(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
) -> int:
    """Brute-force count of nowhere-zero Z_k flows."""
    return len(enumerate_flows_bruteforce(G, k, D=D, root=root))


def enumerate_flows(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
    method: str = "fast",
) -> list[dict[int, int]]:
    if method == "fast":
        return enumerate_flows_fast(G, k, D=D, root=root)
    if method == "bruteforce":
        return enumerate_flows_bruteforce(G, k, D=D, root=root)
    raise ValueError(f"unknown method {method!r}")


def count_flows(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
    method: str = "fast",
) -> int:
    if method == "fast":
        return count_flows_fast(G, k, D=D, root=root)
    if method == "bruteforce":
        return count_flows_bruteforce(G, k, D=D, root=root)
    raise ValueError(f"unknown method {method!r}")
