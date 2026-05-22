# nzflow-qubo

A QUBO formulation for nowhere-zero $k$-flows on graphs, ready to run on
D-Wave quantum annealers, hybrid solvers, or local simulated annealing.

This is the reference implementation accompanying the paper
**"A QUBO Formulation for Nowhere-Zero $k$-Flows"** by Ali Lotfi,
Adam Carter, Mohammad Meysami, Thuan Ha, Kwabena Abrefa Nketia,
Steven J. Shirtliffe, and Steven Rayan.

## What this does

For any nonempty loopless multigraph $G$ and integer $k > 1$, build a
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
git clone https://github.com/alilotfi90/nzflow-qubo.git
cd nzflow-qubo
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
from nzflow_qubo import build_hamiltonian, solve_neal
from nzflow_qubo.benchmarks import K

# Build H_{mod,4} for K_4
spec = build_hamiltonian(K(4), k=4)
print(f"Variables: {spec.n_vars}")  # 27

# Run local simulated annealing
result = solve_neal(spec, num_reads=200, seed=42, num_sweeps=2000)
print(f"Best energy: {result.best_energy}")
flow = result.best_flow()
if flow is not None:
    print(f"Recovered zero-energy flow: {flow}")
else:
    print("Best sample was not a valid flow; increase reads/sweeps or use an exact solver.")
```

For harder instances such as Petersen at $k=5$, simulated annealing may still return
strictly positive energies even though a zero-energy state exists.

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

To generate the paper-ready LaTeX/CSV/JSON tables:

```bash
python scripts/generate_paper_tables.py
```

This writes the following files into `generated/`:

- `theorem_representative.tex` / `.csv` / `.json`
- `qubo_structure.tex` / `.csv` / `.json`
- `robustness_summary.tex` / `.csv` / `.json`
- `snark_benchmarks.tex` / `.csv` / `.json`

The snark table reproduces Table 7 of the paper (Petersen graph and the
Isaacs flower snarks $J_5$ and $J_7$) at $k\in\{3,4,5\}$.

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

The test suite checks all three verification levels from the paper together with edge-case, robustness, and snark tests:

- **C1** — every nowhere-zero $\mathbb{Z}_k$-flow encodes to $H_{\mathrm{mod},k}=0$
- **C2** — random non-flow labellings have positive minimum energy after
  optimizing over quotient bits
- **C3** — exact ground-state enumeration finds exactly as many
  zero-energy states as the brute-force flow enumerator

It also runs robustness sweeps over orientation, root choice, and
penalty weights $(A,B)$, confirming the parameter independence
guaranteed by Theorem 3.6, and verifies the snark suite (Petersen, $J_5$, $J_7$).

## Structure

```
nzflow_qubo/
  graph.py        # Multigraph, Orientation, signed_incidence
  hamiltonian.py  # build_hamiltonian, encode_flow, decode_flow
  enumerator.py   # ground-truth enumeration of nowhere-zero Z_k flows
  solvers.py      # exact, neal, qpu wrappers
  benchmarks.py   # K_n, K_{m,n}, C_n, Q_3, Petersen, Theta_3, K_4-doubled
  snarks.py       # Petersen and Isaacs flower snarks J_n
tests/
  test_correctness.py    # C1, C2, C3 + robustness + snark suite
examples/
  reproduce_table1.py    # representative table reproduction
scripts/
  generate_paper_tables.py  # generates the paper's LaTeX/CSV/JSON tables
generated/
  *.tex, *.csv, *.json      # generated on demand
```

## Mapping to the paper

| Code | Paper |
| --- | --- |
| `Multigraph`, `Orientation`, `signed_incidence` | Section 2.1, 2.2; Definitions 2.1, 2.6, 2.7 |
| `Multigraph.has_bridge()` | Definition 2.10 |
| `enumerate_flows`, `count_flows` | (C3) of Section 4.1 |
| `build_hamiltonian` | Equation (3.1) and Definition 3.5 |
| `encode_flow` | Forward direction of the proof of Theorem 3.6 |
| `decode_flow` | Converse direction of the proof of Theorem 3.6 |
| `solve_exact` | Ground-state enumeration in (C3) |
| `solve_neal` | Simulated-annealing baseline (Section 4.6) |
| `solve_qpu` | D-Wave hardware deployment (Section 5, future work) |
| `petersen_snark`, `flower_snark`, `SNARK_SUITE` | Section 4.4, Table 7 |

## License

MIT.
