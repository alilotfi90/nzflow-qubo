"""
Construction of the modular QUBO Hamiltonian H_{mod,k}.

Implements equation (3.1) from Section 3.3 of the paper. The Hamiltonian
is returned as a dimod.BinaryQuadraticModel for use with any
QUBO-compatible solver (D-Wave QPU, neal simulated annealer, hybrid
solvers, exact solvers).
"""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil, floor, log2

import dimod

from .graph import Multigraph, Orientation, signed_incidence


@dataclass(frozen=True)
class HamiltonianSpec:
    """Container for the BQM together with metadata needed for decoding."""

    bqm: dimod.BinaryQuadraticModel
    G: Multigraph
    D: Orientation
    k: int
    root: int
    A: float
    B: float
    n_one_hot_vars: int
    n_quotient_vars: int
    quotient_bits: dict[int, int]  # vertex v in V* -> B_v

    @property
    def n_vars(self) -> int:
        return self.n_one_hot_vars + self.n_quotient_vars


def edge_var(e: int, a: int) -> str:
    """Variable label for x_{e,a}: edge e gets residue a."""
    return f"x_e{e}_a{a}"


def quot_var(v: int, b: int) -> str:
    """Variable label for p_{v,b}: bit b of the quotient at vertex v."""
    return f"p_v{v}_b{b}"


def quotient_range(d: int, k: int) -> tuple[int, int, int]:
    """Compute (L_v, U_v, B_v) for a vertex of degree d at level k.

    L_v = ceil(-(k-1)*d / k), U_v = floor((k-1)*d / k),
    B_v = ceil(log2(U_v - L_v + 1)).
    """
    L = ceil(-((k - 1) * d) / k)
    U = floor(((k - 1) * d) / k)
    width = U - L + 1
    if width <= 1:
        B = 1  # always need at least one bit even if range is a single value
    else:
        B = max(1, ceil(log2(width)))
    return L, U, B


def build_hamiltonian(
    G: Multigraph,
    k: int,
    *,
    D: Orientation | None = None,
    root: int = 0,
    A: float = 1.0,
    B: float = 1.0,
) -> HamiltonianSpec:
    """Build H_{mod,k} as a dimod BQM.

    Parameters
    ----------
    G : Multigraph
    k : int, > 1
    D : Orientation, optional
        If None, uses Orientation.default(G).
    root : int
        Vertex r in V; conservation is imposed only on V \\ {r}.
    A, B : float
        Penalty weights, both must be > 0.

    Returns
    -------
    HamiltonianSpec
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if A <= 0 or B <= 0:
        raise ValueError(f"penalty weights must be positive, got A={A}, B={B}")
    if not (0 <= root < G.n):
        raise ValueError(f"root {root} not in [0, {G.n})")
    if D is None:
        D = Orientation.default(G)

    bqm = dimod.BinaryQuadraticModel("BINARY")

    # ---- one-hot edge variables ----
    # x_{e,a} for each edge e in E, residue a in {1, ..., k-1}
    for e in range(G.m):
        for a in range(1, k):
            bqm.add_variable(edge_var(e, a))

    n_one_hot = G.m * (k - 1)

    # ---- quotient variables for non-root vertices ----
    # p_{v, b} for v in V*, b = 0, ..., B_v - 1
    quotient_bits: dict[int, int] = {}
    quotient_L: dict[int, int] = {}
    n_quot = 0
    for v in range(G.n):
        if v == root:
            continue
        d_v = G.degree(v)
        L, U, Bv = quotient_range(d_v, k)
        quotient_bits[v] = Bv
        quotient_L[v] = L
        for b in range(Bv):
            bqm.add_variable(quot_var(v, b))
        n_quot += Bv

    # ---- one-hot penalty: A * sum_e (sum_a x_{e,a} - 1)^2 ----
    # Expand (sum_a x_{e,a} - 1)^2
    #   = sum_a x_{e,a}^2 + 2*sum_{a<b} x_{e,a} x_{e,b}
    #     - 2*sum_a x_{e,a} + 1
    # Since x is binary, x^2 = x. Drop the constant +1 (it just shifts energy).
    for e in range(G.m):
        labels = [edge_var(e, a) for a in range(1, k)]
        # Linear: -2 * A * x_{e,a} + 1 * A * x_{e,a} (from x^2 = x term)
        # Combined: -A on each linear term
        for lab in labels:
            bqm.add_linear(lab, -A)
        # Quadratic: +2 * A * x_{e,a} x_{e,b} for a < b
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                bqm.add_quadratic(labels[i], labels[j], 2 * A)
        # Constant: +A
        bqm.offset += A

    # ---- conservation penalty: B * sum_{v in V*} (R_v - k * M_v)^2 ----
    # R_v(x)   = sum_{e in E} sigma_{v,e} * sum_{a=1}^{k-1} a * x_{e,a}
    # M_v(p)   = L_v + sum_{b=0}^{B_v - 1} 2^b * p_{v,b}
    # Expand C_v := R_v - k*M_v as a linear combination of binary variables
    # plus a constant offset, then square it.
    for v in range(G.n):
        if v == root:
            continue
        # Build the linear form C_v = sum_label (coeff * variable) + const
        # as a dict label -> coeff
        coeffs: dict[str, float] = {}
        const = 0.0

        # R_v contribution: + sigma_{v,e} * a * x_{e,a}
        for e in range(G.m):
            s = signed_incidence(D, v, e)
            if s == 0:
                continue
            for a in range(1, k):
                coeffs[edge_var(e, a)] = coeffs.get(edge_var(e, a), 0.0) + s * a

        # -k*M_v contribution: -k * L_v (constant) - k * 2^b * p_{v,b}
        Bv = quotient_bits[v]
        L = quotient_L[v]
        const += -k * L
        for b in range(Bv):
            coeffs[quot_var(v, b)] = coeffs.get(quot_var(v, b), 0.0) - k * (2 ** b)

        # Square it: (sum_i c_i x_i + const)^2
        #   = sum_i c_i^2 x_i^2 + 2 * sum_{i<j} c_i c_j x_i x_j
        #     + 2 * const * sum_i c_i x_i + const^2
        # Use x^2 = x.
        labels = list(coeffs.keys())
        for lab in labels:
            ci = coeffs[lab]
            bqm.add_linear(lab, B * (ci * ci + 2 * const * ci))
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                ci = coeffs[labels[i]]
                cj = coeffs[labels[j]]
                if ci != 0 and cj != 0:
                    bqm.add_quadratic(labels[i], labels[j], B * 2 * ci * cj)
        bqm.offset += B * const * const

    return HamiltonianSpec(
        bqm=bqm,
        G=G,
        D=D,
        k=k,
        root=root,
        A=A,
        B=B,
        n_one_hot_vars=n_one_hot,
        n_quotient_vars=n_quot,
        quotient_bits=quotient_bits,
    )


def encode_flow(spec: HamiltonianSpec, flow: dict[int, int]) -> dict[str, int]:
    """Encode a nowhere-zero Z_k flow as a sample dict.

    Implements the assignment from the reverse direction of the proof of
    Theorem 3.5: x_{e,a} = 1 iff a = flow[e], plus the quotient bits
    chosen so that M_v(p) = q_v where R_v = k * q_v.

    Parameters
    ----------
    spec : HamiltonianSpec
    flow : dict mapping edge index e -> residue in {1, ..., k-1}

    Returns
    -------
    sample : dict mapping variable label -> 0/1
    """
    G = spec.G
    D = spec.D
    k = spec.k

    if set(flow.keys()) != set(range(G.m)):
        raise ValueError("flow must specify a value for every edge")
    for e, val in flow.items():
        if not (1 <= val <= k - 1):
            raise ValueError(f"flow[{e}]={val} not in {{1,...,{k-1}}}")

    sample: dict[str, int] = {}

    # one-hot edge variables
    for e in range(G.m):
        for a in range(1, k):
            sample[edge_var(e, a)] = 1 if a == flow[e] else 0

    # quotient bits
    for v, Bv in spec.quotient_bits.items():
        # R_v = sum_e sigma_{v,e} * flow[e]
        Rv = sum(signed_incidence(D, v, e) * flow[e] for e in range(G.m))
        if Rv % k != 0:
            raise ValueError(
                f"flow is not a valid Z_{k}-flow: R_{v} = {Rv} is not divisible by {k}"
            )
        q = Rv // k
        # find L_v
        d_v = G.degree(v)
        L, _, _ = quotient_range(d_v, k)
        offset = q - L
        if offset < 0 or offset >= 2 ** Bv:
            raise ValueError(
                f"quotient q_{v}={q} (offset {offset}) outside representable range [0, 2^{Bv})"
            )
        for b in range(Bv):
            sample[quot_var(v, b)] = (offset >> b) & 1

    return sample


def decode_flow(spec: HamiltonianSpec, sample: dict[str, int]) -> dict[int, int] | None:
    """Decode a sample to a flow assignment, if it satisfies one-hot.

    Returns None if the one-hot constraint is violated on any edge.
    Does NOT check the conservation constraint; use the BQM energy for that.
    """
    flow: dict[int, int] = {}
    for e in range(spec.G.m):
        on = [a for a in range(1, spec.k) if sample.get(edge_var(e, a), 0) == 1]
        if len(on) != 1:
            return None
        flow[e] = on[0]
    return flow
