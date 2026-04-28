"""nzflow_qubo: A QUBO formulation for nowhere-zero k-flows on graphs.

Implements the construction and verification protocol from the paper
"A QUBO Formulation for Nowhere-Zero k-Flows" by A. Lotfi.
"""

from .graph import Multigraph, Orientation, signed_incidence
from .hamiltonian import (
    HamiltonianSpec,
    build_hamiltonian,
    encode_flow,
    decode_flow,
    quotient_range,
)
from .enumerator import enumerate_flows, count_flows
from .solvers import SolveResult, solve_exact, solve_neal, solve_qpu

__version__ = "0.1.0"

__all__ = [
    "Multigraph",
    "Orientation",
    "signed_incidence",
    "HamiltonianSpec",
    "build_hamiltonian",
    "encode_flow",
    "decode_flow",
    "quotient_range",
    "enumerate_flows",
    "count_flows",
    "SolveResult",
    "solve_exact",
    "solve_neal",
    "solve_qpu",
]
