"""
Reproduce the representative results from Table 1 of the paper.

For each (G, k) configuration:
  - count nowhere-zero Z_k flows by direct enumeration when feasible
  - encode every flow via the proof of Theorem 3.5 and verify
    H_{mod,k} = 0 (the C1 check)

Run with:
    python examples/reproduce_table1.py
"""

from __future__ import annotations

from nzflow_qubo import build_hamiltonian, count_flows, encode_flow, enumerate_flows
from nzflow_qubo.benchmarks import K, K_bipartite, cube, petersen, theta_3, K4_doubled


CONFIGS = [
    ("K_3",          K(3),              2),
    ("K_4",          K(4),              3),
    ("K_4",          K(4),              4),
    ("K_{3,3}",      K_bipartite(3, 3), 3),
    ("Petersen",     petersen(),        4),
    ("Petersen",     petersen(),        5),
    ("Theta_3",      theta_3(),         3),
    ("K_4 doubled",  K4_doubled(),      3),
    ("Q_3",          cube(),            5),
]

# (k-1)^|E| budget for direct enumeration. Set lower than the paper's 2e7
# for a quick interactive demo; raise to 2e7 to match the paper's (C3) bound.
BUDGET = 1 * 10 ** 6


def main() -> None:
    print(f"{'Graph':<14}{'k':>4}{'|E|':>6}{'flows':>9}{'C1':>6}{'enum_cost':>12}")
    print("-" * 51)
    for name, G, k in CONFIGS:
        cost = (k - 1) ** G.m
        if cost > BUDGET:
            n_flows = "skip"
            c1 = "n/a"
            enum_str = "skip"
        else:
            n_flows = count_flows(G, k)
            spec = build_hamiltonian(G, k)
            flows = enumerate_flows(G, k)
            c1_pass = all(
                abs(spec.bqm.energy(encode_flow(spec, f))) < 1e-9
                for f in flows
            )
            c1 = "yes" if c1_pass else "FAIL"
            enum_str = f"{cost:.0e}"
        print(f"{name:<14}{k:>4}{G.m:>6}{str(n_flows):>9}{c1:>6}{enum_str:>12}")


if __name__ == "__main__":
    main()
