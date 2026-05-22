"""nzflow_qubo: A QUBO formulation for nowhere-zero k-flows on graphs.

Implements the construction and verification protocol from the paper
"A QUBO Formulation for Nowhere-Zero k-Flows" by A. Lotfi, A. Carter,
M. Meysami, T. Ha, K. A. Nketia, S. J. Shirtliffe, and S. Rayan.
"""

from .graph import Multigraph, Orientation, signed_incidence
from .hamiltonian import (
    HamiltonianSpec,
    build_hamiltonian,
    encode_flow,
    decode_flow,
    quotient_range,
    flow_residuals,
    is_valid_flow,
    qubo_stats,
)
from .enumerator import (
    enumerate_flows,
    count_flows,
    enumerate_flows_fast,
    count_flows_fast,
    enumerate_flows_bruteforce,
    count_flows_bruteforce,
)
from .solvers import SolveResult, solve_exact, solve_neal, solve_qpu
from .snarks import (
    petersen_snark,
    flower_snark,
    girth,
    is_3_edge_colorable,
    is_snark,
    SNARK_SUITE,
)

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
    "flow_residuals",
    "is_valid_flow",
    "qubo_stats",
    "enumerate_flows",
    "count_flows",
    "enumerate_flows_fast",
    "count_flows_fast",
    "enumerate_flows_bruteforce",
    "count_flows_bruteforce",
    "SolveResult",
    "solve_exact",
    "solve_neal",
    "solve_qpu",
    "petersen_snark",
    "flower_snark",
    "girth",
    "is_3_edge_colorable",
    "is_snark",
    "SNARK_SUITE",
]
