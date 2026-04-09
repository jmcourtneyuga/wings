"""Unit tests for hardware-native execution (v0.4.0 WI-5)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit


@pytest.mark.unit
class TestHardwareTranspilation:
    def test_transpile_circuit_valid(self):
        from wings.hardware import transpile_for_hardware

        qc = QuantumCircuit(4)
        qc.ry(0.5, 0)
        qc.cx(0, 1)
        qc.ry(0.3, 1)
        result = transpile_for_hardware(qc, basis_gates=["cx", "rz", "sx", "x"])
        assert result.num_qubits >= 4

    def test_transpile_preserves_unitary(self):
        from qiskit.quantum_info import Operator

        from wings.hardware import transpile_for_hardware

        qc = QuantumCircuit(2)
        qc.ry(0.5, 0)
        qc.cx(0, 1)
        original = Operator(qc)
        transpiled = transpile_for_hardware(qc, basis_gates=["cx", "rz", "sx", "x"])
        trans_op = Operator(transpiled)
        # Should be equivalent up to global phase
        fid = np.abs(np.trace(original.adjoint().compose(trans_op).data)) / 4
        assert fid > 0.99


@pytest.mark.unit
class TestVerification:
    def test_classical_state_fidelity_perfect(self):
        """Perfect counts should give fidelity 1."""
        from wings.hardware import classical_state_fidelity

        # Target: all probability on |0>
        target_probs = np.zeros(4)
        target_probs[0] = 1.0
        measured_probs = np.zeros(4)
        measured_probs[0] = 1.0
        fid = classical_state_fidelity(target_probs, measured_probs)
        assert abs(fid - 1.0) < 1e-10

    def test_classical_state_fidelity_uniform(self):
        """Uniform measured vs peaked target should give low fidelity."""
        from wings.hardware import classical_state_fidelity

        target_probs = np.zeros(8)
        target_probs[0] = 1.0
        measured_probs = np.ones(8) / 8
        fid = classical_state_fidelity(target_probs, measured_probs)
        assert fid < 0.5

    def test_counts_to_probabilities(self):
        from wings.hardware import counts_to_probabilities

        counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        probs = counts_to_probabilities(counts, n_qubits=2)
        assert probs.shape == (4,)
        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert abs(probs[0] - 0.5) < 0.01

    def test_counts_to_probabilities_missing_keys(self):
        from wings.hardware import counts_to_probabilities

        counts = {"000": 1000}
        probs = counts_to_probabilities(counts, n_qubits=3)
        assert probs.shape == (8,)
        assert abs(probs[0] - 1.0) < 1e-10

    def test_hardware_result_dataclass(self):
        from wings.hardware import HardwareResult

        result = HardwareResult(
            counts={"00": 500, "11": 500},
            classical_fidelity=0.7,
            n_shots=1000,
            device_name="simulator",
        )
        assert result.n_shots == 1000
        assert result.classical_fidelity == 0.7
