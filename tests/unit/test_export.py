"""Unit tests for circuit export module."""

import tempfile
from pathlib import Path

import pytest


class TestBuildOptimizedCircuit:
    """Tests for build_optimized_circuit function."""

    def test_builds_circuit_with_params(self, small_optimizer, random_params_6q):
        """Test building circuit with explicit parameters."""
        from wings.export import build_optimized_circuit

        circuit = build_optimized_circuit(small_optimizer, random_params_6q)

        assert circuit is not None
        assert circuit.num_qubits == 6
        # Should have no unbound parameters
        assert len(circuit.parameters) == 0

    def test_builds_circuit_with_best_params(self, small_optimizer, random_params_6q):
        """Test building circuit using optimizer.best_params."""
        from wings.export import build_optimized_circuit

        # Set best_params
        small_optimizer.best_params = random_params_6q

        circuit = build_optimized_circuit(small_optimizer)

        assert circuit is not None
        assert circuit.num_qubits == 6
        assert len(circuit.parameters) == 0

    def test_raises_without_params(self, small_optimizer):
        """Test error when no parameters available."""
        from wings.export import build_optimized_circuit

        # Ensure no best_params
        small_optimizer.best_params = None

        with pytest.raises(ValueError, match="No parameters provided"):
            build_optimized_circuit(small_optimizer)

    def test_includes_measurements(self, small_optimizer, random_params_6q):
        """Test adding measurement gates."""
        from wings.export import build_optimized_circuit

        circuit = build_optimized_circuit(
            small_optimizer, random_params_6q, include_measurements=True
        )

        # Should have classical registers for measurements
        assert circuit.num_clbits == 6

    def test_no_measurements_by_default(self, small_optimizer, random_params_6q):
        """Test no measurements by default."""
        from wings.export import build_optimized_circuit

        circuit = build_optimized_circuit(small_optimizer, random_params_6q)

        assert circuit.num_clbits == 0


class TestExportToQasm:
    """Tests for OpenQASM export functions."""

    def test_export_qasm2(self, small_optimizer, random_params_6q):
        """Test OpenQASM 2.0 export."""
        from wings.export import export_to_qasm

        small_optimizer.best_params = random_params_6q
        qasm_str = export_to_qasm(small_optimizer)

        assert isinstance(qasm_str, str)
        assert "OPENQASM 2.0" in qasm_str
        assert "qreg" in qasm_str
        assert "ry(" in qasm_str or "RY(" in qasm_str.upper()

    def test_export_qasm3(self, small_optimizer, random_params_6q):
        """Test OpenQASM 3.0 export."""
        from wings.export import export_to_qasm3

        small_optimizer.best_params = random_params_6q

        try:
            qasm3_str = export_to_qasm3(small_optimizer)
            assert isinstance(qasm3_str, str)
            # QASM3 uses 'qubit' instead of 'qreg'
            assert "qubit" in qasm3_str.lower() or "qreg" in qasm3_str.lower()
        except ImportError:
            pytest.skip("OpenQASM 3.0 export requires qiskit >= 1.0")

    def test_qasm_with_measurements(self, small_optimizer, random_params_6q):
        """Test QASM export includes measurements when requested."""
        from wings.export import export_to_qasm

        small_optimizer.best_params = random_params_6q
        qasm_str = export_to_qasm(small_optimizer, include_measurements=True)

        assert "creg" in qasm_str or "measure" in qasm_str


class TestSaveCircuit:
    """Tests for save_circuit function."""

    def test_save_qasm(self, small_optimizer, random_params_6q):
        """Test saving to QASM file."""
        from wings.export import save_circuit

        small_optimizer.best_params = random_params_6q

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_circuit.qasm"
            save_circuit(small_optimizer, filepath, format="qasm")

            assert filepath.exists()
            content = filepath.read_text()
            assert "OPENQASM 2.0" in content

    def test_save_qasm3(self, small_optimizer, random_params_6q):
        """Test saving to QASM3 file."""
        from wings.export import save_circuit

        small_optimizer.best_params = random_params_6q

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_circuit.qasm3"
            try:
                save_circuit(small_optimizer, filepath, format="qasm3")
                assert filepath.exists()
            except ImportError:
                pytest.skip("OpenQASM 3.0 requires qiskit >= 1.0")

    def test_save_qpy(self, small_optimizer, random_params_6q):
        """Test saving to QPY binary format."""
        from wings.export import save_circuit

        small_optimizer.best_params = random_params_6q

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_circuit.qpy"
            save_circuit(small_optimizer, filepath, format="qpy")

            assert filepath.exists()
            # QPY is binary, just check it's not empty
            assert filepath.stat().st_size > 0

    @pytest.mark.slow
    def test_save_png(self, small_optimizer, random_params_6q):
        """Test saving circuit diagram as PNG."""
        from wings.export import save_circuit

        small_optimizer.best_params = random_params_6q

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_circuit.png"
            try:
                save_circuit(small_optimizer, filepath, format="png")
                assert filepath.exists()
                assert filepath.stat().st_size > 0
            except Exception as e:
                # May fail in headless environments
                if "display" in str(e).lower() or "backend" in str(e).lower():
                    pytest.skip("Matplotlib display not available")
                raise

    def test_auto_format_from_extension(self, small_optimizer, random_params_6q):
        """Test automatic format detection from file extension."""
        from wings.export import save_circuit

        small_optimizer.best_params = random_params_6q

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_circuit.qasm"
            save_circuit(small_optimizer, filepath, format="auto")

            assert filepath.exists()
            content = filepath.read_text()
            assert "OPENQASM" in content

    def test_invalid_format_raises(self, small_optimizer, random_params_6q):
        """Test invalid format raises error."""
        from wings.export import save_circuit

        small_optimizer.best_params = random_params_6q

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_circuit.xyz"
            with pytest.raises(ValueError, match="Unknown format"):
                save_circuit(small_optimizer, filepath, format="xyz")


class TestOptimizerExportMethods:
    """Tests for export methods on GaussianOptimizer."""

    def test_get_optimized_circuit(self, small_optimizer, random_params_6q):
        """Test optimizer.get_optimized_circuit method."""
        small_optimizer.best_params = random_params_6q

        # This tests the method if it's been added to the optimizer
        if hasattr(small_optimizer, "get_optimized_circuit"):
            circuit = small_optimizer.get_optimized_circuit()
            assert circuit is not None
            assert circuit.num_qubits == 6
        else:
            pytest.skip("get_optimized_circuit not yet added to optimizer")

    def test_export_qasm_method(self, small_optimizer, random_params_6q):
        """Test optimizer.export_qasm method."""
        small_optimizer.best_params = random_params_6q

        if hasattr(small_optimizer, "export_qasm"):
            qasm_str = small_optimizer.export_qasm()
            assert "OPENQASM" in qasm_str
        else:
            pytest.skip("export_qasm not yet added to optimizer")

    def test_save_circuit_method(self, small_optimizer, random_params_6q):
        """Test optimizer.save_circuit method."""
        small_optimizer.best_params = random_params_6q

        if hasattr(small_optimizer, "save_circuit"):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = Path(tmpdir) / "test.qasm"
                small_optimizer.save_circuit(str(filepath))
                assert filepath.exists()
        else:
            pytest.skip("save_circuit not yet added to optimizer")


class TestQasmRoundTrip:
    """Tests verifying QASM can be loaded back."""

    def test_qasm_loadable(self, small_optimizer, random_params_6q):
        """Test exported QASM can be loaded by Qiskit."""
        from qiskit import QuantumCircuit

        from wings.export import export_to_qasm

        small_optimizer.best_params = random_params_6q
        qasm_str = export_to_qasm(small_optimizer)

        # Load it back
        loaded_circuit = QuantumCircuit.from_qasm_str(qasm_str)

        assert loaded_circuit.num_qubits == 6

    def test_qpy_round_trip(self, small_optimizer, random_params_6q):
        """Test QPY save and load round-trip."""
        from qiskit.qpy import load

        from wings.export import build_optimized_circuit, save_circuit

        small_optimizer.best_params = random_params_6q
        original = build_optimized_circuit(small_optimizer, random_params_6q)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.qpy"
            save_circuit(small_optimizer, filepath, format="qpy")

            with open(filepath, "rb") as f:
                loaded_circuits = load(f)

            loaded = loaded_circuits[0]
            assert loaded.num_qubits == original.num_qubits
            assert loaded.depth() == original.depth()
