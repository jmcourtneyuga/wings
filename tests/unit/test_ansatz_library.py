"""Unit tests for ansatz_library module."""

import numpy as np
import pytest
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


@pytest.mark.unit
class TestEfficientSU2Ansatz:
    """Tests for EfficientSU2Ansatz class."""

    def test_initialization(self):
        """Test EfficientSU2Ansatz initialization with n_qubits=4, layers=3."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=3)

        assert ansatz.n_params == 24  # 4 * 2 * 3
        assert ansatz.depth == 3

    def test_param_count_scales(self):
        """Test n_params == n_qubits * 2 * layers for various combos."""
        from wings.ansatz_library import EfficientSU2Ansatz

        for n, d in [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5)]:
            ansatz = EfficientSU2Ansatz(n_qubits=n, layers=d)
            assert ansatz.n_params == n * 2 * d, (
                f"Failed for n_qubits={n}, layers={d}: expected {n * 2 * d}, got {ansatz.n_params}"
            )

    def test_circuit_construction(self):
        """Test circuit has correct num_qubits and parameter count."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=3)
        params = ParameterVector("theta", ansatz.n_params)

        circuit = ansatz(params, n_qubits=4)

        assert circuit.num_qubits == 4
        assert len(circuit.parameters) == ansatz.n_params

    def test_produces_valid_statevector(self):
        """Test random params produce a valid Statevector."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=3)
        np.random.seed(42)
        params = np.random.randn(ansatz.n_params)

        circuit = ansatz(params, n_qubits=4)
        sv = Statevector(circuit)

        assert sv.is_valid()
        assert sv.num_qubits == 4

    def test_produces_complex_amplitudes(self):
        """Test nonzero params give states with imaginary parts (RZ gates)."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=3)
        # Use params that are not multiples of pi to ensure complex amplitudes
        params = np.ones(ansatz.n_params) * 0.7

        circuit = ansatz(params, n_qubits=4)
        sv = Statevector(circuit)
        amplitudes = sv.data

        # RZ gates introduce complex phases, so imaginary parts should be nonzero
        assert np.any(np.abs(amplitudes.imag) > 1e-10), "Expected complex amplitudes from RZ gates"

    def test_satisfies_ansatz_protocol(self):
        """Test isinstance(ansatz, AnsatzProtocol) is True."""
        from wings.ansatz import AnsatzProtocol
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=3)

        assert isinstance(ansatz, AnsatzProtocol)

    def test_linear_entanglement(self):
        """Test linear entanglement builds without error."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=2, entanglement="linear")
        params = np.random.randn(ansatz.n_params)
        circuit = ansatz(params, n_qubits=4)

        assert circuit.num_qubits == 4

    def test_circular_entanglement(self):
        """Test circular entanglement builds without error."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=2, entanglement="circular")
        params = np.random.randn(ansatz.n_params)
        circuit = ansatz(params, n_qubits=4)

        assert circuit.num_qubits == 4

    def test_reverse_linear_entanglement(self):
        """Test reverse_linear entanglement builds without error."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=2, entanglement="reverse_linear")
        params = np.random.randn(ansatz.n_params)
        circuit = ansatz(params, n_qubits=4)

        assert circuit.num_qubits == 4

    def test_parity_entanglement(self):
        """Test parity entanglement builds without error (6 qubits)."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=6, layers=2, entanglement="parity")
        params = np.random.randn(ansatz.n_params)
        circuit = ansatz(params, n_qubits=6)

        assert circuit.num_qubits == 6

    def test_works_with_optimizer(self):
        """Test drop-in works in GaussianOptimizer."""
        from wings import GaussianOptimizer, OptimizerConfig
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=6, layers=4)

        config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            ansatz=ansatz,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )

        opt = GaussianOptimizer(config)

        assert opt.ansatz is ansatz
        assert opt.n_params == ansatz.n_params

    def test_zero_params_produces_valid_state(self):
        """Test zero params produce valid Statevector."""
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=4, layers=3)
        params = np.zeros(ansatz.n_params)

        circuit = ansatz(params, n_qubits=4)
        sv = Statevector(circuit)

        assert sv.is_valid()


@pytest.mark.unit
class TestDefaultAnsatzEntanglement:
    """Tests for DefaultAnsatz entanglement topology options."""

    def test_default_is_linear(self):
        """Test DefaultAnsatz defaults to linear entanglement."""
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(4)
        assert ansatz.entanglement == "linear"

    def test_circular_entanglement(self):
        """Test DefaultAnsatz with circular entanglement builds circuit."""
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(4, entanglement="circular")
        params = np.random.randn(ansatz.n_params)
        circuit = ansatz(params)
        assert circuit.num_qubits == 4

    def test_log_distance_entanglement(self):
        """Test DefaultAnsatz with log_distance entanglement builds circuit."""
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(8, entanglement="log_distance")
        params = np.random.randn(ansatz.n_params)
        circuit = ansatz(params)
        assert circuit.num_qubits == 8

    def test_parity_entanglement(self):
        """Test DefaultAnsatz with parity entanglement builds circuit."""
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(6, entanglement="parity")
        params = np.random.randn(ansatz.n_params)
        circuit = ansatz(params)
        assert circuit.num_qubits == 6

    def test_backward_compat_no_entanglement_arg(self):
        """Test DefaultAnsatz(6) has 36 params and produces valid Statevector."""
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(6)
        assert ansatz.n_params == 36
        params = np.random.randn(ansatz.n_params)
        circuit = ansatz(params)
        sv = Statevector(circuit)
        assert sv.is_valid()

    def test_different_topologies_different_states(self):
        """Test linear vs circular with same params gives different statevectors."""
        from wings.ansatz import DefaultAnsatz

        np.random.seed(123)
        linear = DefaultAnsatz(4, entanglement="linear")
        circular = DefaultAnsatz(4, entanglement="circular")
        params = np.random.randn(linear.n_params)

        sv_lin = Statevector(linear(params))
        sv_circ = Statevector(circular(params))

        assert not np.allclose(sv_lin.data, sv_circ.data)


@pytest.mark.unit
class TestGenerateEntanglementMap:
    """Tests for generate_entanglement_map function."""

    def test_linear(self):
        """Test linear returns [(0,1),(1,2),(2,3)] for 4 qubits."""
        from wings.ansatz_library import generate_entanglement_map

        result = generate_entanglement_map(4, "linear")
        assert result == [(0, 1), (1, 2), (2, 3)]

    def test_circular(self):
        """Test circular returns linear + (3,0) for 4 qubits."""
        from wings.ansatz_library import generate_entanglement_map

        result = generate_entanglement_map(4, "circular")
        assert result == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_parity(self):
        """Test parity contains (0,1), (2,3), (1,2) for 6 qubits."""
        from wings.ansatz_library import generate_entanglement_map

        result = generate_entanglement_map(6, "parity")
        assert (0, 1) in result
        assert (2, 3) in result
        assert (1, 2) in result

    def test_log_distance(self):
        """Test log_distance contains (0,1), (0,2), (0,4) for 8 qubits."""
        from wings.ansatz_library import generate_entanglement_map

        result = generate_entanglement_map(8, "log_distance")
        assert (0, 1) in result
        assert (0, 2) in result
        assert (0, 4) in result

    def test_full(self):
        """Test full has len == 6 for 4 qubits (C(4,2))."""
        from wings.ansatz_library import generate_entanglement_map

        result = generate_entanglement_map(4, "full")
        assert len(result) == 6

    def test_invalid_pattern_raises(self):
        """Test ValueError for invalid pattern."""
        from wings.ansatz_library import generate_entanglement_map

        with pytest.raises(ValueError):
            generate_entanglement_map(4, "invalid")
