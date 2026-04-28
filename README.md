# nzflow-qubo

A QUBO formulation for nowhere-zero $k$-flows on graphs, ready to run on
D-Wave quantum annealers, hybrid solvers, or local simulated annealing.

This is the reference implementation accompanying the paper
**"A QUBO Formulation for Nowhere-Zero $k$-Flows"** by Ali Lotfi
(University of Saskatchewan).

## What this does

For any loopless multigraph $G$ and integer $k > 1$, build a
quadratic Hamiltonian $H_{\mathrm{mod},k}$ over binary variables whose
ground-state energy is zero if and only if $G$ admits a nowhere-zero
$\mathbb{Z}_k$-flow. By Tutte's equivalence theorem this is equivalent
to the existence of a $k$-flow, the original object of Tutte's $5$-flow
conjecture (1954).

The Hamiltonian is built as a `dimod.BinaryQuadraticModel`, so it works
with:

- `dimod.ExactSolver` for exact ground-state enumeration on small graphs
- `neal.SimulatedAnnealingSampler` for local simulated annealing (no API
  token needed)
- `dwave.system.EmbeddingComposite` + `DWaveSampler` for D-Wave QPU
  hardware (Pegasus or Zephyr topology)
- D-Wave hybrid BQM solver for larger graphs

## Installation

```bash
git clone <repo url>
cd nzflow-qubo
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
from nzflow_qubo import build_hamiltonian, solve_neal
from nzflow_qubo.benchmarks import petersen

# Build H_{mod, 5} for the Petersen graph
spec = build_hamiltonian(petersen(), k=5)
print(f"Variables: {spec.n_vars}")  # 87

# Run local simulated annealing
result = solve_neal(spec, num_reads=1000, seed=42, num_sweeps=5000)
print(f"Best energy: {result.best_energy}")  # ~0 (Petersen has phi = 5)
flow = result.best_flow()
print(f"Recovered flow: {flow}")
```

## Running on D-Wave hardware

Once you have a D-Wave Leap API token configured, swap the solver:

```python
from nzflow_qubo import build_hamiltonian, solve_qpu
from nzflow_qubo.benchmarks import K

spec = build_hamiltonian(K(4), k=4)
result = solve_qpu(spec, num_reads=1000, label="K4 k=4 test")
print(f"Best energy: {result.best_energy}")
```

## Reproducing paper results

To reproduce the representative cases from Table 1:

```bash
python examples/reproduce_table1.py
```

Output:

```
Graph            k   |E|    flows    C1   enum_cost
---------------------------------------------------
K_3              2     3        1   yes       1e+00
K_4              3     6        0   yes       6e+01
K_4              4     6        6   yes       7e+02
K_{3,3}          3     9        2   yes       5e+02
Theta_3          3     3        2   yes       8e+00
K_4 doubled      3    12      176   yes       4e+03
```

For full Table 1 including Petersen and $Q_3$, raise `BUDGET` in the
example to $2 \times 10^7$ (matches the paper's (C3) bound; takes
roughly 20 seconds on a laptop).

## Running tests

```bash
pytest tests/ -v
```

The test suite checks all three verification levels from the paper:

- **C1** — every nowhere-zero $\mathbb{Z}_k$-flow encodes to
  $H_{\mathrm{mod},k} = 0$
- **C2** — random non-flow labellings have positive minimum energy after
  optimizing over quotient bits
- **C3** — exact ground-state enumeration finds exactly as many
  zero-energy states as the brute-force flow enumerator

It also runs robustness sweeps over orientation, root choice, and
penalty weights $(A, B)$, confirming the parameter-independence
guaranteed by Theorem 3.5.

## Structure

```
nzflow_qubo/
  graph.py        # Multigraph, Orientation, signed_incidence
  hamiltonian.py  # build_hamiltonian, encode_flow, decode_flow
  enumerator.py   # ground-truth enumeration of nowhere-zero Z_k flows
  solvers.py      # exact, neal, qpu wrappers
  benchmarks.py   # K_n, K_{m,n}, C_n, Q_3, Petersen, Theta_3, K_4-doubled
tests/
  test_correctness.py    # C1, C2, C3 + robustness suite
examples/
  reproduce_table1.py    # paper's representative table
```

## Mapping to the paper

| Code | Paper |
| --- | --- |
| `Multigraph`, `Orientation`, `signed_incidence` | Section 2.1, 2.2; Definitions 2.1, 2.6, 2.7 |
| `Multigraph.has_bridge()` | Definition 2.10 |
| `enumerate_flows`, `count_flows` | (C3) of Section 4.1 |
| `build_hamiltonian` | Equation (3.1) and Definition 3.4 |
| `encode_flow` | Reverse direction of the proof of Theorem 3.5 |
| `decode_flow` | Forward direction of the proof of Theorem 3.5 |
| `solve_exact` | Ground-state enumeration in (C3) |
| `solve_neal` | Simulated-annealing baseline (Section 4.5, future work) |
| `solve_qpu` | D-Wave hardware deployment (Section 5, future work) |

## License

MIT.

## Citation

If this code is useful for your research, please cite:

```bibtex
@misc{lotfi2026nzflow,
  author = {Lotfi, Ali},
  title  = {A QUBO Formulation for Nowhere-Zero $k$-Flows},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```
