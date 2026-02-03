"""Unit tests for evaluators module."""

import numpy as np
import pytest


class TestThreadSafeCircuitEvaluator:
    """Tests for CPU evaluator."""

    def test_initialization(self, small_config, make_gaussian):
        """Test evaluator initialization."""
        from wings.evaluators.cpu import ThreadSafeCircuitEvaluator

        target = make_gaussian(64, sigma=0.5)
        evaluator = ThreadSafeCircuitEvaluator(small_config, target)

        assert evaluator.config.n_qubits == 6
        assert evaluator.n_params == 36

    def test_compute_fidelity(
        self, small_config, make_gaussian, random_params_6q, assert_valid_fidelity
    ):
        """Test fidelity computation."""
        from wings.evaluators.cpu import ThreadSafeCircuitEvaluator

        target = make_gaussian(64, sigma=0.5)
        evaluator = ThreadSafeCircuitEvaluator(small_config, target)

        fidelity = evaluator.compute_fidelity(random_params_6q)
        assert_valid_fidelity(fidelity)

    def test_deterministic(self, small_config, make_gaussian, random_params_6q):
        """Test evaluator is deterministic."""
        from wings.evaluators.cpu import ThreadSafeCircuitEvaluator

        target = make_gaussian(64, sigma=0.5)
        evaluator = ThreadSafeCircuitEvaluator(small_config, target)

        fid1 = evaluator.compute_fidelity(random_params_6q)
        fid2 = evaluator.compute_fidelity(random_params_6q)

        assert np.isclose(fid1, fid2)

    def test_get_statevector(
        self, small_config, make_gaussian, random_params_6q, assert_valid_statevector
    ):
        """Test statevector retrieval."""
        from wings.evaluators.cpu import ThreadSafeCircuitEvaluator

        target = make_gaussian(64, sigma=0.5)
        evaluator = ThreadSafeCircuitEvaluator(small_config, target)

        sv = evaluator.get_statevector(random_params_6q)
        assert_valid_statevector(sv, 6)


@pytest.mark.gpu
class TestGPUCircuitEvaluator:
    """Tests for GPU (Aer) evaluator."""

    def test_initialization(self, gpu_config, make_gaussian):
        """Test GPU evaluator initialization."""
        from wings.evaluators.gpu import GPUCircuitEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = GPUCircuitEvaluator(gpu_config, target)

        assert evaluator.config.n_qubits == 8

    def test_compute_fidelity(
        self, gpu_config, make_gaussian, random_params_8q, assert_valid_fidelity
    ):
        """Test GPU fidelity computation."""
        from wings.evaluators.gpu import GPUCircuitEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = GPUCircuitEvaluator(gpu_config, target)

        if evaluator.gpu_available:
            fidelity = evaluator.compute_fidelity(random_params_8q)
            assert_valid_fidelity(fidelity)
        else:
            pytest.skip("GPU not available")

    def test_matches_cpu(self, make_gaussian, random_params_6q):
        """Test GPU matches CPU results."""
        from wings import OptimizerConfig
        from wings.evaluators.cpu import ThreadSafeCircuitEvaluator
        from wings.evaluators.gpu import GPUCircuitEvaluator

        target = make_gaussian(64, sigma=0.5)

        # CPU config
        cpu_config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            box_size=4.0,
            verbose=False,
            use_gpu=False,
        )
        cpu_eval = ThreadSafeCircuitEvaluator(cpu_config, target)

        # GPU config
        gpu_config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            box_size=4.0,
            verbose=False,
            use_gpu=True,
        )
        gpu_eval = GPUCircuitEvaluator(gpu_config, target)

        if gpu_eval.gpu_available:
            fid_cpu = cpu_eval.compute_fidelity(random_params_6q)
            fid_gpu = gpu_eval.compute_fidelity(random_params_6q)

            assert np.isclose(fid_cpu, fid_gpu, rtol=1e-6)
        else:
            pytest.skip("GPU not available")


@pytest.mark.custatevec
class TestCuStateVecSimulator:
    """Tests for cuStateVec simulator."""

    def test_initialization(self, cleanup_gpu):
        """Test simulator initialization."""
        from wings.evaluators.custatevec import CuStateVecSimulator

        sim = CuStateVecSimulator(n_qubits=6, precision="double")

        assert sim.n_qubits == 6
        assert sim.n_states == 64
        assert sim.device_id == 0

        sim.destroy()

    def test_reset_state(self, cleanup_gpu):
        """Test state reset to |0...0>."""
        from wings.evaluators.custatevec import CuStateVecSimulator

        sim = CuStateVecSimulator(n_qubits=4)
        sim.reset_state()

        sv = sim.get_statevector_cpu()

        # Should be |0000> state
        assert np.isclose(np.abs(sv[0]), 1.0)
        assert np.allclose(np.abs(sv[1:]), 0.0)

        sim.destroy()

    def test_apply_x(self, cleanup_gpu):
        """Test X gate application."""
        from wings.evaluators.custatevec import CuStateVecSimulator

        sim = CuStateVecSimulator(n_qubits=2)
        sim.reset_state()
        sim.apply_x(0)  # X on qubit 0

        sv = sim.get_statevector_cpu()

        # State should be normalized
        assert np.isclose(np.sum(np.abs(sv) ** 2), 1.0)

        sim.destroy()

    def test_apply_ry(self, cleanup_gpu):
        """Test RY gate application."""
        from wings.evaluators.custatevec import CuStateVecSimulator

        sim = CuStateVecSimulator(n_qubits=1)
        sim.reset_state()

        # RY(pi) should flip |0> to |1>
        sim.apply_ry(np.pi, 0)

        sv = sim.get_statevector_cpu()

        # Should be close to |1>
        assert np.abs(sv[1]) > 0.99

        sim.destroy()

    def test_apply_cnot(self, cleanup_gpu):
        """Test CNOT gate application."""
        from wings.evaluators.custatevec import CuStateVecSimulator

        sim = CuStateVecSimulator(n_qubits=2)
        sim.reset_state()

        # Prepare |10>
        sim.apply_x(1)

        # Apply CNOT with qubit 1 as control, qubit 0 as target
        sim.apply_cnot(1, 0)

        sv = sim.get_statevector_cpu()

        # Should be normalized
        assert np.isclose(np.sum(np.abs(sv) ** 2), 1.0)

        sim.destroy()

    def test_statevector_normalized(self, cleanup_gpu):
        """Test statevector remains normalized after gates."""
        from wings.evaluators.custatevec import CuStateVecSimulator

        sim = CuStateVecSimulator(n_qubits=4)
        sim.reset_state()

        # Apply some random gates
        np.random.seed(42)
        for _ in range(10):
            sim.apply_ry(np.random.randn(), np.random.randint(4))

        for _ in range(5):
            ctrl = np.random.randint(3)
            tgt = ctrl + 1
            sim.apply_cnot(ctrl, tgt)

        sv = sim.get_statevector_cpu()
        norm = np.linalg.norm(sv)

        assert np.isclose(norm, 1.0, atol=1e-10)

        sim.destroy()


@pytest.mark.custatevec
class TestCuStateVecEvaluator:
    """Tests for cuStateVec evaluator."""

    def test_initialization(self, custatevec_config, make_gaussian, cleanup_gpu):
        """Test evaluator initialization."""
        from wings.evaluators.custatevec import CuStateVecEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = CuStateVecEvaluator(custatevec_config, target)

        assert evaluator.n_qubits == 8
        assert evaluator.n_params == 64

        evaluator.cleanup()

    def test_compute_fidelity(
        self, custatevec_config, make_gaussian, random_params_8q, assert_valid_fidelity, cleanup_gpu
    ):
        """Test fidelity computation."""
        from wings.evaluators.custatevec import CuStateVecEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = CuStateVecEvaluator(custatevec_config, target)

        fidelity = evaluator.compute_fidelity(random_params_8q)
        assert_valid_fidelity(fidelity)

        evaluator.cleanup()

    def test_get_statevector(
        self,
        custatevec_config,
        make_gaussian,
        random_params_8q,
        assert_valid_statevector,
        cleanup_gpu,
    ):
        """Test statevector retrieval."""
        from wings.evaluators.custatevec import CuStateVecEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = CuStateVecEvaluator(custatevec_config, target)

        sv = evaluator.get_statevector(random_params_8q)
        assert_valid_statevector(sv, 8)

        evaluator.cleanup()

    def test_get_statevector_qiskit_order(
        self,
        custatevec_config,
        make_gaussian,
        random_params_8q,
        assert_valid_statevector,
        cleanup_gpu,
    ):
        """Test statevector in Qiskit ordering for plotting."""
        from wings.evaluators.custatevec import CuStateVecEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = CuStateVecEvaluator(custatevec_config, target)

        sv = evaluator.get_statevector_qiskit_order(random_params_8q)
        assert_valid_statevector(sv, 8)

        evaluator.cleanup()

    def test_fidelity_matches_cpu(self, make_gaussian, cleanup_gpu):
        """Test cuStateVec fidelity matches CPU computation."""
        from wings import GaussianOptimizer, OptimizerConfig

        np.random.seed(42)
        params = np.random.randn(36) * 0.1

        # CPU optimizer
        cpu_config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        cpu_opt = GaussianOptimizer(cpu_config)

        # cuStateVec optimizer
        cusv_config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            verbose=False,
            use_gpu=False,
            use_custatevec=True,
        )
        cusv_opt = GaussianOptimizer(cusv_config)

        if cusv_opt._custatevec_evaluator is not None:
            fid_cpu = cpu_opt.compute_fidelity(params=params)
            fid_cusv = cusv_opt.compute_fidelity(params=params)

            # Fidelities should match closely
            assert np.isclose(fid_cpu, fid_cusv, rtol=1e-5), (
                f"CPU fidelity {fid_cpu} != cuStateVec fidelity {fid_cusv}"
            )
        else:
            pytest.skip("cuStateVec not available")


@pytest.mark.custatevec
class TestBatchedCuStateVecEvaluator:
    """Tests for batched cuStateVec evaluator."""

    def test_initialization(self, custatevec_config, make_gaussian, cleanup_gpu):
        """Test batched evaluator initialization."""
        from wings.evaluators.custatevec import BatchedCuStateVecEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = BatchedCuStateVecEvaluator(custatevec_config, target, n_simulators=4)

        assert evaluator.n_simulators == 4
        assert len(evaluator.simulators) == 4

        evaluator.cleanup()

    def test_evaluate_batch(
        self,
        custatevec_config,
        make_gaussian,
        population_medium,
        assert_valid_fidelity,
        cleanup_gpu,
    ):
        """Test batch evaluation."""
        from wings.evaluators.custatevec import BatchedCuStateVecEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = BatchedCuStateVecEvaluator(custatevec_config, target, n_simulators=4)

        fidelities = evaluator.evaluate_batch(population_medium)

        assert fidelities.shape == (32,)
        for fid in fidelities:
            assert_valid_fidelity(fid)

        evaluator.cleanup()

    def test_gradient_batched(
        self, custatevec_config, make_gaussian, random_params_8q, cleanup_gpu
    ):
        """Test batched gradient computation."""
        from wings.evaluators.custatevec import BatchedCuStateVecEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = BatchedCuStateVecEvaluator(custatevec_config, target, n_simulators=4)

        gradient = evaluator.compute_gradient_batched(random_params_8q)

        assert gradient.shape == (64,)
        assert np.all(np.isfinite(gradient))

        evaluator.cleanup()

    def test_get_statevector_qiskit_order(
        self,
        custatevec_config,
        make_gaussian,
        random_params_8q,
        assert_valid_statevector,
        cleanup_gpu,
    ):
        """Test statevector in Qiskit order for plotting."""
        from wings.evaluators.custatevec import BatchedCuStateVecEvaluator

        target = make_gaussian(256, sigma=0.5)
        evaluator = BatchedCuStateVecEvaluator(custatevec_config, target, n_simulators=2)

        sv = evaluator.get_statevector_qiskit_order(random_params_8q)
        assert_valid_statevector(sv, 8)

        evaluator.cleanup()


@pytest.mark.custatevec
class TestCuStateVecBitReversal:
    """Tests specifically for bit reversal handling in cuStateVec."""

    def test_bit_reversal_function(self, cleanup_gpu):
        """Test the bit reversal helper function."""
        from wings import OptimizerConfig
        from wings.evaluators.custatevec import CuStateVecEvaluator

        config = OptimizerConfig(n_qubits=4, sigma=0.5, verbose=False, use_custatevec=True)
        target = np.zeros(16, dtype=np.complex128)
        target[0] = 1.0  # |0000>

        evaluator = CuStateVecEvaluator(config, target)

        # Test bit reversal
        test_sv = np.arange(16, dtype=np.complex128)
        reversed_sv = evaluator._bit_reverse_statevector(test_sv, 4)

        # Index 0 (0000) should stay at 0
        assert reversed_sv[0] == test_sv[0]

        # Index 1 (0001) should go to index 8 (1000)
        assert reversed_sv[8] == test_sv[1]

        # Index 8 (1000) should go to index 1 (0001)
        assert reversed_sv[1] == test_sv[8]

        evaluator.cleanup()

    def test_double_reversal_identity(self, cleanup_gpu):
        """Test that double bit reversal returns original."""
        from wings import OptimizerConfig
        from wings.evaluators.custatevec import CuStateVecEvaluator

        config = OptimizerConfig(n_qubits=6, sigma=0.5, verbose=False, use_custatevec=True)
        target = np.zeros(64, dtype=np.complex128)
        target[0] = 1.0

        evaluator = CuStateVecEvaluator(config, target)

        # Random statevector
        np.random.seed(42)
        original = np.random.randn(64) + 1j * np.random.randn(64)

        # Double reversal should give back original
        reversed_once = evaluator._bit_reverse_statevector(original, 6)
        reversed_twice = evaluator._bit_reverse_statevector(reversed_once, 6)

        np.testing.assert_array_almost_equal(original, reversed_twice)

        evaluator.cleanup()

    def test_qiskit_order_for_plotting(self, cleanup_gpu):
        """Test that get_statevector_qiskit_order gives correct results for plotting."""
        from wings import GaussianOptimizer, OptimizerConfig

        # Create optimizers
        cpu_config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        cusv_config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            verbose=False,
            use_gpu=False,
            use_custatevec=True,
        )

        cpu_opt = GaussianOptimizer(cpu_config)
        cusv_opt = GaussianOptimizer(cusv_config)

        if cusv_opt._custatevec_evaluator is None:
            pytest.skip("cuStateVec not available")

        # Use same params
        np.random.seed(123)
        params = np.random.randn(36) * 0.1

        # Get statevectors
        sv_cpu = cpu_opt.get_statevector(params)
        sv_cusv_qiskit = cusv_opt._custatevec_evaluator.get_statevector_qiskit_order(params)

        # They should match (both in Qiskit ordering)
        np.testing.assert_allclose(sv_cpu, sv_cusv_qiskit, rtol=1e-5, atol=1e-10)


@pytest.mark.multi_gpu
class TestMultiGPUBatchEvaluator:
    """Tests for multi-GPU evaluator."""

    def test_initialization(self, multi_gpu_config, make_gaussian, cleanup_gpu):
        """Test multi-GPU evaluator initialization."""
        from wings.evaluators.custatevec import MultiGPUBatchEvaluator

        target = make_gaussian(1024, sigma=0.5)

        try:
            evaluator = MultiGPUBatchEvaluator(multi_gpu_config, target)

            assert evaluator.n_gpus >= 1
            assert len(evaluator.simulators) == evaluator.n_gpus

            evaluator.cleanup()
        except RuntimeError as e:
            if "No GPUs" in str(e) or "single GPU" in str(e).lower():
                pytest.skip("Multiple GPUs not available")
            raise

    def test_evaluate_batch_parallel(self, multi_gpu_config, make_gaussian, cleanup_gpu):
        """Test parallel batch evaluation."""
        from wings.evaluators.custatevec import MultiGPUBatchEvaluator

        target = make_gaussian(1024, sigma=0.5)

        try:
            evaluator = MultiGPUBatchEvaluator(multi_gpu_config, target)

            # Create batch
            np.random.seed(42)
            batch = np.random.randn(64, 100) * 0.1

            fidelities = evaluator.evaluate_batch_parallel(batch)

            assert fidelities.shape == (64,)
            assert np.all(fidelities >= 0)
            assert np.all(fidelities <= 1)

            evaluator.cleanup()
        except RuntimeError as e:
            if "No GPUs" in str(e) or "single GPU" in str(e).lower():
                pytest.skip("Multiple GPUs not available")
            raise
