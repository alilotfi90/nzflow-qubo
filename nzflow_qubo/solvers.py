"""
Solver wrappers for H_{mod,k}.

Provides three backends, all consuming a HamiltonianSpec:
- exact: brute-force minimization (small instances only)
- simulated annealing: dimod's neal, runs locally, no API token
- D-Wave QPU: requires API token; commented for now

The exact solver is the right "ground truth" check on small graphs and is
what populates the (C3) verification. Neal is the right hardware-free
default for larger benchmarks. The QPU path is left as a stub so that
swapping it in once you have an API token is a one-line change.
"""

from __future__ import annotations
from dataclasses import dataclass

import dimod
from dimod import ExactSolver

from .hamiltonian import HamiltonianSpec, decode_flow, is_valid_flow


@dataclass
class SolveResult:
    """Wrapper around a sample set that adds flow-level information."""
    sampleset: dimod.SampleSet
    spec: HamiltonianSpec

    @property
    def best_energy(self) -> float:
        """Energy of the lowest-energy sample."""
        return float(self.sampleset.first.energy)

    @property
    def best_sample(self) -> dict[str, int]:
        return dict(self.sampleset.first.sample)

    def best_flow(self) -> dict[int, int] | None:
        """Return the best sample decoded as a valid flow, if one is present.

        This checks both one-hot decoding and modular conservation. A low-energy
        sample need not be a valid flow unless its energy is zero.
        """
        f = decode_flow(self.spec, self.best_sample)
        return f if (f is not None and is_valid_flow(self.spec, f)) else None

    def zero_energy_flows(self) -> list[dict[int, int]]:
        """Decode every zero-energy sample to a flow.

        Useful when paired with an exact solver, where the full ground-state
        manifold is reported.
        """
        flows: list[dict[int, int]] = []
        for sample, energy, _ in self.sampleset.data(
            ['sample', 'energy', 'num_occurrences']
        ):
            if abs(energy) < 1e-9:
                f = decode_flow(self.spec, dict(sample))
                if f is not None and is_valid_flow(self.spec, f):
                    flows.append(f)
        return flows


def solve_exact(spec: HamiltonianSpec) -> SolveResult:
    """Brute-force ground-state enumeration. O(2^N) — small graphs only.

    For a zero-variable BQM, return the single empty sample at the model offset.
    """
    if len(spec.bqm.variables) == 0:
        ss = dimod.SampleSet.from_samples([{}], vartype="BINARY", energy=[float(spec.bqm.offset)])
        return SolveResult(sampleset=ss, spec=spec)
    sampler = ExactSolver()
    ss = sampler.sample(spec.bqm)
    return SolveResult(sampleset=ss, spec=spec)


def solve_neal(
    spec: HamiltonianSpec,
    *,
    num_reads: int = 1000,
    seed: int | None = None,
    num_sweeps: int = 1000,
) -> SolveResult:
    """Run simulated annealing using dimod's neal sampler.

    Local, no API token, no quota. Good for development and larger
    benchmarks where ExactSolver is infeasible.
    """
    try:
        import neal
    except ImportError as exc:
        raise ImportError(
            "neal is required for solve_neal. Install with: pip install dwave-neal"
        ) from exc

    sampler = neal.SimulatedAnnealingSampler()
    kwargs: dict = {"num_reads": num_reads, "num_sweeps": num_sweeps}
    if seed is not None:
        kwargs["seed"] = seed
    ss = sampler.sample(spec.bqm, **kwargs)
    return SolveResult(sampleset=ss, spec=spec)


def solve_qpu(
    spec: HamiltonianSpec,
    *,
    num_reads: int = 1000,
    chain_strength: float | None = None,
    label: str | None = None,
) -> SolveResult:
    """Run on a D-Wave QPU via EmbeddingComposite.

    Requires DWAVE_API_TOKEN configured. Embeds H_{mod,k} onto the QPU's
    Pegasus or Zephyr topology automatically.
    """
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
    except ImportError as exc:
        raise ImportError(
            "dwave-system is required for solve_qpu. "
            "Install with: pip install dwave-system"
        ) from exc

    sampler = EmbeddingComposite(DWaveSampler())
    kwargs: dict = {"num_reads": num_reads}
    if chain_strength is not None:
        kwargs["chain_strength"] = chain_strength
    if label is not None:
        kwargs["label"] = label
    ss = sampler.sample(spec.bqm, **kwargs)
    return SolveResult(sampleset=ss, spec=spec)
