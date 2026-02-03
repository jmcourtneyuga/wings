"""Circuit export utilities for WINGS.

This module provides functions to export optimized circuits to various formats,
including OpenQASM 2.0, OpenQASM 3.0, and Qiskit QuantumCircuit objects.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from .optimizer import GaussianOptimizer

__all__ = [
    "build_optimized_circuit",
    "export_to_qasm",
    "export_to_qasm3",
    "save_circuit",
]


def build_optimized_circuit(
    optimizer: "GaussianOptimizer",
    params: Optional[NDArray[np.float64]] = None,
    include_measurements: bool = False,
) -> QuantumCircuit:
    """
    Build a concrete QuantumCircuit with optimized parameters bound.

    Parameters
    ----------
    optimizer : GaussianOptimizer
        The optimizer instance containing the ansatz
    params : np.ndarray, optional
        Parameter values to bind. If None, uses optimizer.best_params
    include_measurements : bool, default False
        Whether to add measurement gates to all qubits

    Returns
    -------
    QuantumCircuit
        Qiskit circuit with parameters bound to concrete values

    Raises
    ------
    ValueError
        If no parameters provided and optimizer has no best_params

    Examples
    --------
    >>> results = optimizer.optimize_ultra_precision(target_infidelity=1e-10)
    >>> circuit = build_optimized_circuit(optimizer)
    >>> print(circuit.draw())
    """
    # Get parameters
    if params is None:
        if optimizer.best_params is None:
            raise ValueError(
                "No parameters provided and optimizer has no best_params. "
                "Run optimization first or provide params explicitly."
            )
        params = optimizer.best_params

    if optimizer.ansatz is None:
        raise ValueError("Optimizer has no ansatz defined")

    # Build circuit using ansatz
    circuit = optimizer.ansatz(
        params, optimizer.config.n_qubits, **(optimizer.config.ansatz_kwargs or {})
    )

    # If circuit still has unbound parameters, bind them
    if circuit.parameters:
        param_dict = dict(zip(circuit.parameters, params))
        circuit = circuit.assign_parameters(param_dict)

    # Optionally add measurements
    if include_measurements:
        circuit.measure_all()

    return circuit


def export_to_qasm(
    optimizer: "GaussianOptimizer",
    params: Optional[NDArray[np.float64]] = None,
    include_measurements: bool = False,
) -> str:
    """
    Export optimized circuit to OpenQASM 2.0 string.

    Parameters
    ----------
    optimizer : GaussianOptimizer
        The optimizer instance
    params : np.ndarray, optional
        Parameter values. If None, uses optimizer.best_params
    include_measurements : bool, default False
        Whether to include measurement gates

    Returns
    -------
    str
        OpenQASM 2.0 format string

    Examples
    --------
    >>> qasm_str = export_to_qasm(optimizer)
    >>> print(qasm_str)
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[8];
    ry(0.123456) q[0];
    ...
    """
    from qiskit.qasm2 import dumps

    circuit = build_optimized_circuit(optimizer, params, include_measurements)
    return dumps(circuit)


def export_to_qasm3(
    optimizer: "GaussianOptimizer",
    params: Optional[NDArray[np.float64]] = None,
    include_measurements: bool = False,
) -> str:
    """
    Export optimized circuit to OpenQASM 3.0 string.

    Parameters
    ----------
    optimizer : GaussianOptimizer
        The optimizer instance
    params : np.ndarray, optional
        Parameter values. If None, uses optimizer.best_params
    include_measurements : bool, default False
        Whether to include measurement gates

    Returns
    -------
    str
        OpenQASM 3.0 format string

    Notes
    -----
    Requires qiskit >= 1.0 for OpenQASM 3.0 support.
    """
    from qiskit.qasm3 import dumps

    circuit = build_optimized_circuit(optimizer, params, include_measurements)
    return dumps(circuit)


def save_circuit(
    optimizer: "GaussianOptimizer",
    filepath: Union[str, Path],
    params: Optional[NDArray[np.float64]] = None,
    format: str = "qasm",
    include_measurements: bool = False,
    **kwargs,
) -> Path:
    """
    Save optimized circuit to file.

    Parameters
    ----------
    optimizer : GaussianOptimizer
        The optimizer instance
    filepath : str or Path
        Output file path. Extension determines format if format='auto'
    params : np.ndarray, optional
        Parameter values. If None, uses optimizer.best_params
    format : str, default 'qasm'
        Output format: 'qasm' (OpenQASM 2.0), 'qasm3' (OpenQASM 3.0),
        'qpy' (Qiskit QPY binary), 'png' (circuit diagram), 'svg', 'pdf'
    include_measurements : bool, default False
        Whether to include measurement gates
    **kwargs
        Additional arguments passed to the export function

    Returns
    -------
    Path
        Path to the saved file

    Examples
    --------
    >>> save_circuit(optimizer, "my_circuit.qasm")
    >>> save_circuit(optimizer, "my_circuit.png", format='png')
    >>> save_circuit(optimizer, "my_circuit.qpy", format='qpy')
    """
    filepath = Path(filepath)

    # Auto-detect format from extension
    if format == "auto":
        ext = filepath.suffix.lower()
        format_map = {
            ".qasm": "qasm",
            ".qasm3": "qasm3",
            ".qpy": "qpy",
            ".png": "png",
            ".svg": "svg",
            ".pdf": "pdf",
        }
        format = format_map.get(ext, "qasm")

    circuit = build_optimized_circuit(optimizer, params, include_measurements)

    if format == "qasm":
        from qiskit.qasm2 import dumps

        qasm_str = dumps(circuit)
        filepath.write_text(qasm_str)

    elif format == "qasm3":
        from qiskit.qasm3 import dumps

        qasm3_str = dumps(circuit)
        filepath.write_text(qasm3_str)

    elif format == "qpy":
        from qiskit.qpy import dump

        with open(filepath, "wb") as f:
            dump(circuit, f)

    elif format in ("png", "svg", "pdf"):
        # Circuit diagram
        fig = circuit.draw(output="mpl", **kwargs)
        fig.savefig(filepath, format=format, bbox_inches="tight", dpi=150)
        import matplotlib.pyplot as plt

        plt.close(fig)

    else:
        raise ValueError(
            f"Unknown format: {format}. Use 'qasm', 'qasm3', 'qpy', 'png', 'svg', or 'pdf'"
        )

    return filepath
