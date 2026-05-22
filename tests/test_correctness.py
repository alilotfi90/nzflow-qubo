"""
Verification tests, mirroring Section 4 of the paper.

Run from repo root with: python -m pytest tests/ -v

For each (G, k) in a representative subset:
  C1: every nowhere-zero Z_k flow encodes to H_{mod,k} = 0
  C2: random non-flow labellings have positive minimum energy
  C3: when feasible, exact solver's count of zero-energy states equals
      the number of nowhere-zero Z_k flows from the brute-force enumerator
"""

from __future__ import annotations
import random

import pytest

from nzflow_qubo import (
    Multigraph,
    Orientation,
    build_hamiltonian,
    encode_flow,
    decode_flow,
    enumerate_flows,
    count_flows,
    solve_exact,
    solve_neal,
    quotient_range,
    qubo_stats,
    count_flows_bruteforce,
)
from nzflow_qubo.benchmarks import K, K_bipartite, cycle, cube, theta_3, K4_doubled, triangular_prism, petersen


# ----- C1: encoding theorem ----------------------------------------------

@pytest.mark.parametrize("graph_fn,k,name", [
    (lambda: K(3), 2, "K3 k=2"),
    (lambda: K(4), 3, "K4 k=3"),
    (lambda: K(4), 4, "K4 k=4"),
    (lambda: K_bipartite(3, 3), 3, "K3,3 k=3"),
    (lambda: cycle(5), 2, "C5 k=2"),
    (lambda: cycle(6), 2, "C6 k=2"),
    (lambda: theta_3(), 3, "Theta3 k=3"),
    (lambda: K4_doubled(), 3, "K4_doubled k=3"),
])
def test_C1_every_flow_encodes_to_zero(graph_fn, k, name):
    """C1: encoding a valid flow gives BQM energy 0."""
    G = graph_fn()
    spec = build_hamiltonian(G, k)
    flows = enumerate_flows(G, k)
    if not flows:
        pytest.skip(f"{name}: no nowhere-zero Z_{k} flows exist (no-instance)")
    for flow in flows:
        sample = encode_flow(spec, flow)
        energy = spec.bqm.energy(sample)
        assert abs(energy) < 1e-9, f"{name}: flow encoded to nonzero energy {energy}"


# ----- C2: non-flows give positive energy --------------------------------

@pytest.mark.parametrize("graph_fn,k,name", [
    (lambda: K(4), 3, "K4 k=3"),  # no flows
    (lambda: K_bipartite(3, 3), 3, "K3,3 k=3"),
])
def test_C2_random_non_flow_gives_positive_energy(graph_fn, k, name):
    """C2: a labelling that violates Kirchhoff has minimum energy > 0
    (after optimizing over quotient bits)."""
    G = graph_fn()
    spec = build_hamiltonian(G, k)
    rng = random.Random(42)

    valid_flows = {tuple(f[e] for e in range(G.m)) for f in enumerate_flows(G, k)}

    tried = 0
    while tried < 5:
        # Pick a random labelling
        labelling = tuple(rng.randrange(1, k) for _ in range(G.m))
        if labelling in valid_flows:
            continue
        tried += 1
        # For each fixed labelling, the conservation block is a quadratic in
        # the quotient bits. Min over quotient bits: search exhaustively.
        # (Small enough for K_4, K_{3,3}.)
        # Build a partial sample with x set, then iterate over all p assignments.
        partial = {}
        from nzflow_qubo.hamiltonian import edge_var, quot_var
        for e in range(G.m):
            for a in range(1, k):
                partial[edge_var(e, a)] = 1 if a == labelling[e] else 0
        # Iterate over all 2^(sum B_v) assignments to quotient bits
        quot_labels = []
        for v, Bv in spec.quotient_bits.items():
            for b in range(Bv):
                quot_labels.append(quot_var(v, b))
        n_quot = len(quot_labels)
        if n_quot > 16:
            pytest.skip("too many quotient bits to enumerate")
        min_e = float("inf")
        for mask in range(1 << n_quot):
            s = dict(partial)
            for i, lab in enumerate(quot_labels):
                s[lab] = (mask >> i) & 1
            e = spec.bqm.energy(s)
            if e < min_e:
                min_e = e
        assert min_e > 0, f"{name}: non-flow {labelling} achieved energy {min_e}"


# ----- C3: exact solver matches enumerator -------------------------------

@pytest.mark.parametrize("graph_fn,k,name", [
    (lambda: K(3), 2, "K3 k=2"),
    (lambda: K(4), 3, "K4 k=3"),
    (lambda: K(4), 4, "K4 k=4"),
    (lambda: cycle(4), 2, "C4 k=2"),
    (lambda: theta_3(), 3, "Theta3 k=3"),
    (lambda: theta_3(), 4, "Theta3 k=4"),
])
def test_C3_exact_solver_matches_enumerator(graph_fn, k, name):
    """C3: ExactSolver finds exactly as many zero-energy states as there
    are nowhere-zero Z_k flows (one quotient assignment per flow)."""
    G = graph_fn()
    spec = build_hamiltonian(G, k)

    if spec.n_vars > 18:
        pytest.skip(f"{name}: {spec.n_vars} vars too many for ExactSolver")

    expected = count_flows(G, k)

    result = solve_exact(spec)
    zero_flows = result.zero_energy_flows()
    # Each flow has exactly one valid quotient assignment per non-root vertex,
    # but the quotient bits may have "overshoot" extras that also yield
    # M_v(p) = q_v if 2^{B_v} > U_v - L_v + 1. We count distinct flows.
    distinct_flows = {tuple(f[e] for e in range(G.m)) for f in zero_flows}
    assert len(distinct_flows) == expected, (
        f"{name}: exact solver found {len(distinct_flows)} distinct flows, "
        f"enumerator found {expected}"
    )


# ----- robustness: orientation independence ------------------------------

def test_orientation_sweep_K4_k4():
    """All 2^6 = 64 orientations of K_4 give the same flow count at k=4."""
    G = K(4)
    expected = count_flows(G, 4)
    for mask in range(1 << G.m):
        D = Orientation.from_bitmask(G, mask)
        c = count_flows(G, 4, D=D)
        assert c == expected, f"mask {mask}: count {c} != {expected}"


def test_orientation_sweep_theta3_k3():
    """All 2^3 = 8 orientations of Theta_3 give the same flow count at k=3."""
    G = theta_3()
    expected = count_flows(G, 3)
    for mask in range(1 << G.m):
        D = Orientation.from_bitmask(G, mask)
        c = count_flows(G, 3, D=D)
        assert c == expected, f"mask {mask}: count {c} != {expected}"


# ----- robustness: penalty-weight independence ---------------------------

@pytest.mark.parametrize("A,B", [
    (1.0, 1.0), (10.0, 1.0), (1.0, 10.0),
    (0.1, 1.0), (1.0, 0.1), (5.0, 0.5),
])
def test_penalty_weight_sweep_theta3_k4(A, B):
    """Six (A, B) pairs all produce the right ground-state count for Theta_3 at k=4.

    Theta_3 at k=4 has only 12 logical variables, so ExactSolver remains practical.
    Using K_4 at k=4 would require an exact solve over 27 variables and is too slow for CI.
    """
    G = theta_3()
    spec = build_hamiltonian(G, 4, A=A, B=B)
    expected = count_flows(G, 4)
    result = solve_exact(spec)
    distinct_flows = {tuple(f[e] for e in range(G.m)) for f in result.zero_energy_flows()}
    assert len(distinct_flows) == expected, (
        f"A={A}, B={B}: {len(distinct_flows)} flows vs expected {expected}"
    )


# ----- robustness: root choice independence ------------------------------

def test_root_sweep_K33_k3():
    """All 6 root choices on K_{3,3} at k=3 give the same flow count."""
    G = K_bipartite(3, 3)
    expected = count_flows(G, 3, root=0)
    for r in range(G.n):
        c = count_flows(G, 3, root=r)
        assert c == expected, f"root {r}: count {c} != {expected}"



def test_orientation_sweep_K33_k3():
    """All 2^9 = 512 orientations of K_{3,3} give the same flow count at k=3."""
    G = K_bipartite(3, 3)
    expected = count_flows(G, 3)
    for mask in range(1 << G.m):
        D = Orientation.from_bitmask(G, mask)
        c = count_flows(G, 3, D=D)
        assert c == expected, f"mask {mask}: count {c} != {expected}"


def test_exact_root_sweep_theta3_k3():
    """Every root choice on Theta_3 gives the same exact zero-energy flow count."""
    G = theta_3()
    expected = count_flows(G, 3, root=0)
    for r in range(G.n):
        spec = build_hamiltonian(G, 3, root=r)
        result = solve_exact(spec)
        distinct_flows = {tuple(f[e] for e in range(G.m)) for f in result.zero_energy_flows()}
        assert len(distinct_flows) == expected


def test_quotient_range_width_one_uses_zero_bits():
    """If the quotient range has size 1, no auxiliary bit is needed."""
    assert quotient_range(0, 2) == (0, 0, 0)
    assert quotient_range(1, 2) == (0, 0, 0)
    assert quotient_range(1, 3) == (0, 0, 0)


def test_edgeless_graph_has_zero_energy_empty_flow():
    """An edgeless graph has one empty flow and needs no quotient bits."""
    G = Multigraph(n=3, edges=())
    spec = build_hamiltonian(G, 2)
    assert spec.n_vars == 0
    result = solve_exact(spec)
    assert abs(result.best_energy) < 1e-9
    assert result.zero_energy_flows() == [{}]


def test_single_edge_graph_has_positive_ground_energy():
    """A single-edge graph is a bridge graph and has no zero-energy state."""
    G = Multigraph(n=2, edges=((0, 1),))
    spec = build_hamiltonian(G, 3)
    result = solve_exact(spec)
    assert result.best_energy > 0
    assert result.zero_energy_flows() == []


def test_best_flow_requires_conservation():
    """best_flow() must reject one-hot samples that violate conservation."""
    G = K(4)
    spec = build_hamiltonian(G, 4)
    result = solve_neal(spec, num_reads=50, seed=42, num_sweeps=500)
    if result.best_energy > 1e-9:
        assert result.best_flow() is None


def test_qubo_stats_exposes_structure():
    """QUBO structural statistics are exposed for use in paper tables."""
    spec = build_hamiltonian(K(4), 4)
    s = qubo_stats(spec)
    assert s["n_vars"] == spec.n_vars
    assert s["n_quadratic"] > 0
    assert s["quadratic_max"] >= s["quadratic_min"]


# ----- end-to-end: simulated annealing finds a flow ----------------------

def test_neal_finds_flow_K4_k4():
    """Simulated annealing locates a zero-energy state on K_4 at k=4."""
    G = K(4)
    spec = build_hamiltonian(G, 4)
    result = solve_neal(spec, num_reads=200, seed=42, num_sweeps=2000)
    # K_4 has flow number 4, so a Z_4-flow exists; SA should find one.
    assert result.best_energy < 0.5, (
        f"neal best energy {result.best_energy} should be near 0 for K_4 at k=4"
    )
    flow = result.best_flow()
    assert flow is not None, "best sample was not a valid zero-energy flow"


def test_fast_enumerator_matches_bruteforce_on_small_samples():
    """Cycle-space enumeration matches brute force on a deterministic sample of small multigraphs."""
    samples = [
        Multigraph(n=3, edges=((0, 1), (1, 2), (2, 0))),
        Multigraph(n=4, edges=((0, 1), (1, 2), (2, 3), (3, 0))),
        Multigraph(n=4, edges=((0, 1), (0, 1), (1, 2), (2, 3), (3, 0))),
        Multigraph(n=5, edges=((0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2))),
    ]
    for G in samples:
        for k in (2, 3, 4):
            if (k - 1) ** G.m > 5000:
                continue
            assert count_flows(G, k) == count_flows_bruteforce(G, k)


def test_known_threshold_counts():
    """Harder benchmark counts used in the paper tables remain stable."""
    assert count_flows(triangular_prism(), 3) == 0
    assert count_flows(triangular_prism(), 4) == 6
    assert count_flows(petersen(), 4) == 0
    assert count_flows(petersen(), 5) == 240


# ----- Snark suite (Section 4.4, Table 7) -------------------------------

def test_petersen_snark_is_a_snark():
    """Smallest snark, sanity baseline."""
    from nzflow_qubo import petersen_snark, is_snark, girth
    G = petersen_snark()
    assert G.n == 10 and G.m == 15
    assert all(G.degree(v) == 3 for v in range(G.n))
    assert not G.has_bridge()
    assert girth(G) == 5
    assert is_snark(G)


def test_flower_snark_J5_is_a_snark():
    """J_5 (Isaacs 1975): 20 vertices, 30 edges, smallest flower snark."""
    from nzflow_qubo import flower_snark, is_snark, girth
    G = flower_snark(5)
    assert G.n == 20 and G.m == 30
    assert all(G.degree(v) == 3 for v in range(G.n))
    assert not G.has_bridge()
    assert girth(G) >= 5
    assert is_snark(G)


def test_flower_snark_J7_is_a_snark():
    """J_7: 28 vertices, 42 edges. Colorability check takes ~2.5s on a laptop."""
    from nzflow_qubo import flower_snark, is_snark
    G = flower_snark(7)
    assert G.n == 28 and G.m == 42
    assert is_snark(G)


def test_flower_snark_rejects_invalid_n():
    """flower_snark requires odd n >= 3."""
    from nzflow_qubo import flower_snark
    with pytest.raises(ValueError):
        flower_snark(4)        # even
    with pytest.raises(ValueError):
        flower_snark(2)        # too small
    with pytest.raises(ValueError):
        flower_snark(-1)       # negative


def test_snark_suite_integrity():
    """Every entry in SNARK_SUITE constructs to a verified snark."""
    from nzflow_qubo import SNARK_SUITE, is_snark
    for name, builder, n_exp, m_exp, check_col in SNARK_SUITE:
        G = builder()
        assert G.n == n_exp, f"{name}: vertex count tripwire"
        assert G.m == m_exp, f"{name}: edge count tripwire"
        # cubic + bridgeless + girth >= 5 are cheap on all suite entries;
        # 3-edge-colorability is skipped for J_9 and J_{11} per the per-entry flag.
        assert is_snark(G, check_colorable=check_col), f"{name}: is_snark failed"


def test_petersen_snark_at_k4_has_zero_flows():
    """The snark property forces phi(Petersen) = 5. So 0 nowhere-zero Z_4-flows."""
    from nzflow_qubo import petersen_snark
    assert count_flows(petersen_snark(), 4) == 0


def test_flower_snark_J5_at_k3_and_k4_has_zero_flows():
    """J_5 is a snark, so no Z_3- or Z_4-flow. k=4 count uses ~177K
    spanning-tree iterations (~6s on a laptop)."""
    from nzflow_qubo import flower_snark
    G = flower_snark(5)
    assert count_flows(G, 3) == 0
    assert count_flows(G, 4) == 0


def test_snark_builds_clean_hamiltonian():
    """Hamiltonian builder must produce a sane BQM on every snark in the suite."""
    from nzflow_qubo import SNARK_SUITE
    for name, builder, _, _, _ in SNARK_SUITE:
        G = builder()
        for k in (4, 5):
            spec = build_hamiltonian(G, k)
            assert spec.n_vars > 0
            assert spec.bqm.offset > 0
            assert spec.G is G
            assert spec.k == k
            assert len(spec.bqm.quadratic) > 0


def test_girth_basic():
    """Sanity checks on the girth helper."""
    from nzflow_qubo import girth, petersen_snark
    G_k3 = Multigraph(n=3, edges=((0, 1), (1, 2), (0, 2)))
    assert girth(G_k3) == 3
    G_k4 = Multigraph(n=4, edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    assert girth(G_k4) == 3
    assert girth(petersen_snark()) == 5
    G_tree = Multigraph(n=4, edges=((0, 1), (1, 2), (2, 3)))
    assert girth(G_tree) == float("inf")


def test_is_3_edge_colorable_trivial():
    """K_4 is 3-edge-colorable; Petersen is not."""
    from nzflow_qubo import is_3_edge_colorable, petersen_snark
    G_k4 = Multigraph(n=4, edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    assert is_3_edge_colorable(G_k4)
    assert not is_3_edge_colorable(petersen_snark())
