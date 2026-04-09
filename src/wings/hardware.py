"""
Hardware-native transpilation and execution for WINGS circuits.

Provides tools for running optimized circuits on real quantum hardware
via Qiskit Runtime, with measurement verification against simulation.

Reference: Courtney Dissertation, Ch. 3 (ibm_torino results)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile

__all__ = [
    "transpile_for_hardware",
    "classical_state_fidelity",
    "counts_to_probabilities",
    "HardwareResult",
]


@dataclass
class HardwareResult:
    """Result from hardware execution."""

    counts: dict[str, int]
    classical_fidelity: float
    n_shots: int
    device_name: str
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)


def transpile_for_hardware(
    circuit: QuantumCircuit,
    basis_gates: Optional[list[str]] = None,
    coupling_map: Optional[list[list[int]]] = None,
    optimization_level: int = 3,
) -> QuantumCircuit:
    """
    Transpile a WINGS circuit for hardware execution.

    Args:
        circuit: The optimized quantum circuit
        basis_gates: Target gate set (default: ['cx', 'rz', 'sx', 'x'])
        coupling_map: Device connectivity (None = all-to-all)
        optimization_level: Qiskit transpiler optimization (0-3)

    Returns:
        Transpiled circuit in the target gate set
    """
    if basis_gates is None:
        basis_gates = ["cx", "rz", "sx", "x"]

    return transpile(
        circuit,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        optimization_level=optimization_level,
    )


def counts_to_probabilities(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """
    Convert measurement counts to probability array.

    Args:
        counts: Dictionary mapping bitstrings to counts
        n_qubits: Number of qubits

    Returns:
        Probability array of length 2^n_qubits, indexed by integer basis state
    """
    n_states = 2**n_qubits
    total_shots = sum(counts.values())
    probs = np.zeros(n_states)

    for bitstring, count in counts.items():
        # Convert bitstring to integer index
        idx = int(bitstring, 2)
        probs[idx] = count / total_shots

    return probs


def classical_state_fidelity(
    target_probs: np.ndarray,
    measured_probs: np.ndarray,
) -> float:
    """
    Compute classical state fidelity (Bhattacharyya coefficient).

    F_st = (sum_i sqrt(p_i * q_i))^2

    where p_i are measured probabilities and q_i are target probabilities.
    From Courtney Dissertation, Eq. 2.1.2.

    Args:
        target_probs: Ideal probability distribution |psi_target|^2
        measured_probs: Measured probability distribution from hardware

    Returns:
        Classical state fidelity in [0, 1]
    """
    # Bhattacharyya coefficient: BC = sum(sqrt(p*q))
    bc: float = np.sum(np.sqrt(target_probs * measured_probs))
    return float(bc**2)


def execute_on_hardware(
    circuit: QuantumCircuit,
    backend_name: str = "ibm_brisbane",
    shots: int = 8192,
    service=None,
) -> HardwareResult:
    """
    Execute circuit on IBM Quantum hardware.

    Requires qiskit-ibm-runtime.

    Args:
        circuit: Transpiled circuit with measurements
        backend_name: IBM Quantum device name
        shots: Number of measurement shots
        service: QiskitRuntimeService instance (None = create new)

    Returns:
        HardwareResult with counts and metadata
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as err:
        raise ImportError(
            "qiskit-ibm-runtime is required for hardware execution. "
            "Install with: pip install qiskit-ibm-runtime"
        ) from err

    import time

    if service is None:
        service = QiskitRuntimeService()

    backend = service.backend(backend_name)

    # Add measurements if not present
    if circuit.num_clbits == 0:
        circuit.measure_all()

    sampler = SamplerV2(backend)
    start = time.time()
    job = sampler.run([circuit], shots=shots)
    result = job.result()
    elapsed = time.time() - start

    counts = result[0].data.meas.get_counts()

    return HardwareResult(
        counts=counts,
        classical_fidelity=0.0,  # Computed separately with target
        n_shots=shots,
        device_name=backend_name,
        execution_time=elapsed,
    )
