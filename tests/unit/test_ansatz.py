"""Unit tests for ansatz module."""


class TestDefaultAnsatz:
    """Tests for DefaultAnsatz class."""

    def test_initialization(self):
        """Test default ansatz initialization."""
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(n_qubits=6)

        assert ansatz._n_qubits == 6
        assert ansatz.n_params == 36  # 6 * 6

    def test_n_params_property(self):
        """Test n_params property."""
        from wings.ansatz import DefaultAnsatz

        for n in [4, 6, 8, 10]:
            ansatz = DefaultAnsatz(n_qubits=n)
            assert ansatz.n_params == n * n

    def test_callable(self):
        """Test ansatz is callable."""
        from qiskit.circuit import ParameterVector

        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(n_qubits=4)
        params = ParameterVector("theta", 16)

        circuit = ansatz(params, n_qubits=4)

        assert circuit is not None
        assert circuit.num_qubits == 4

    def test_circuit_depth(self):
        """Test circuit has expected structure."""
        from qiskit.circuit import ParameterVector

        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(n_qubits=4)
        params = ParameterVector("theta", 16)

        circuit = ansatz(params, n_qubits=4)

        # Should have gates
        assert circuit.depth() > 0
        assert circuit.size() > 0

    def test_parameterized_circuit(self):
        """Test circuit has correct number of parameters."""
        from qiskit.circuit import ParameterVector

        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(n_qubits=6)
        params = ParameterVector("theta", 36)

        circuit = ansatz(params, n_qubits=6)

        assert len(circuit.parameters) == 36


class TestCustomHardwareEfficientAnsatz:
    """Tests for CustomHardwareEfficientAnsatz class."""

    def test_initialization(self):
        """Test custom ansatz initialization."""
        from wings.ansatz import CustomHardwareEfficientAnsatz

        ansatz = CustomHardwareEfficientAnsatz(
            n_qubits=6,
            layers=4,
            entanglement="linear",
        )

        assert ansatz._n_qubits == 6
        assert ansatz._layers == 4

    def test_linear_entanglement(self):
        """Test linear entanglement pattern."""
        from qiskit.circuit import ParameterVector

        from wings.ansatz import CustomHardwareEfficientAnsatz

        ansatz = CustomHardwareEfficientAnsatz(
            n_qubits=4,
            layers=2,
            entanglement="linear",
        )

        params = ParameterVector("theta", ansatz.n_params)
        circuit = ansatz(params, n_qubits=4)

        assert circuit.num_qubits == 4

    def test_circular_entanglement(self):
        """Test circular entanglement pattern."""
        from qiskit.circuit import ParameterVector

        from wings.ansatz import CustomHardwareEfficientAnsatz

        ansatz = CustomHardwareEfficientAnsatz(
            n_qubits=4,
            layers=2,
            entanglement="circular",
        )

        params = ParameterVector("theta", ansatz.n_params)
        circuit = ansatz(params, n_qubits=4)

        assert circuit.num_qubits == 4

    def test_full_entanglement(self):
        """Test full entanglement pattern."""
        from qiskit.circuit import ParameterVector

        from wings.ansatz import CustomHardwareEfficientAnsatz

        ansatz = CustomHardwareEfficientAnsatz(
            n_qubits=4,
            layers=2,
            entanglement="full",
        )

        params = ParameterVector("theta", ansatz.n_params)
        circuit = ansatz(params, n_qubits=4)

        assert circuit.num_qubits == 4

    def test_custom_rotation_gates(self):
        """Test custom rotation gates."""
        from wings.ansatz import CustomHardwareEfficientAnsatz

        ansatz = CustomHardwareEfficientAnsatz(
            n_qubits=4,
            layers=2,
            rotation_gates=["ry", "rz"],
        )

        # Should have more parameters with two rotation gates
        expected_params_per_layer = 4 * 2  # 4 qubits * 2 gates
        assert ansatz.n_params >= expected_params_per_layer * 2


class TestAnsatzProtocol:
    """Tests for AnsatzProtocol compliance."""

    def test_default_ansatz_protocol(self):
        """Test DefaultAnsatz implements protocol."""
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(n_qubits=4)

        # Should have required attributes/methods
        assert hasattr(ansatz, "n_params")
        assert callable(ansatz)

    def test_custom_ansatz_protocol(self):
        """Test CustomHardwareEfficientAnsatz implements protocol."""
        from wings.ansatz import CustomHardwareEfficientAnsatz

        ansatz = CustomHardwareEfficientAnsatz(n_qubits=4, layers=2)

        # Should have required attributes/methods
        assert hasattr(ansatz, "n_params")
        assert callable(ansatz)


class TestAnsatzWithOptimizer:
    """Tests for ansatz integration with optimizer."""

    def test_default_ansatz_in_optimizer(self, small_config):
        """Test optimizer with default ansatz."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(small_config)

        # Should use default ansatz
        assert opt.ansatz is not None
        assert opt.n_params == 36

    def test_custom_ansatz_in_config(self):
        """Test custom ansatz in config."""
        from wings import GaussianOptimizer, OptimizerConfig
        from wings.ansatz import CustomHardwareEfficientAnsatz

        ansatz = CustomHardwareEfficientAnsatz(n_qubits=6, layers=4)

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

    def test_ansatz_produces_valid_statevector(
        self, small_config, random_params_6q, assert_valid_statevector
    ):
        """Test ansatz produces valid statevector."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(small_config)
        sv = opt.get_statevector(random_params_6q)

        assert_valid_statevector(sv, 6)
