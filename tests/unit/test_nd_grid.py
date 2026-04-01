"""Unit tests for multi-dimensional wavefunctions (v0.4.0 WI-3)."""

import numpy as np
import pytest


@pytest.mark.unit
class TestNDGrid:

    def test_1d_grid_shape(self):
        from wings.nd_grid import NDGrid
        grid = NDGrid(n_qubits_per_dim=[8], box_sizes=[4.0])
        assert grid.n_dimensions == 1
        assert grid.total_qubits == 8
        assert grid.total_states == 256

    def test_2d_grid_shape(self):
        from wings.nd_grid import NDGrid
        grid = NDGrid(n_qubits_per_dim=[4, 4], box_sizes=[4.0, 4.0])
        assert grid.n_dimensions == 2
        assert grid.total_qubits == 8
        assert grid.total_states == 256

    def test_3d_grid_shape(self):
        from wings.nd_grid import NDGrid
        grid = NDGrid(n_qubits_per_dim=[3, 3, 3], box_sizes=[3.0, 3.0, 3.0])
        assert grid.n_dimensions == 3
        assert grid.total_qubits == 9
        assert grid.total_states == 512

    def test_positions_1d(self):
        from wings.nd_grid import NDGrid
        grid = NDGrid(n_qubits_per_dim=[4], box_sizes=[2.0])
        pos = grid.positions()
        assert len(pos) == 1
        assert len(pos[0]) == 16

    def test_positions_2d(self):
        from wings.nd_grid import NDGrid
        grid = NDGrid(n_qubits_per_dim=[3, 3], box_sizes=[2.0, 3.0])
        pos = grid.positions()
        assert len(pos) == 2
        assert len(pos[0]) == 8
        assert len(pos[1]) == 8


@pytest.mark.unit
class TestNDGaussian:

    def test_2d_gaussian_shape(self):
        from wings.nd_grid import NDGrid, gaussian_nd
        grid = NDGrid(n_qubits_per_dim=[4, 4], box_sizes=[4.0, 4.0])
        psi = gaussian_nd(grid, sigmas=[0.5, 0.5])
        assert psi.shape == (256,)
        assert psi.dtype == np.complex128

    def test_2d_gaussian_normalized(self):
        from wings.nd_grid import NDGrid, gaussian_nd
        grid = NDGrid(n_qubits_per_dim=[4, 4], box_sizes=[4.0, 4.0])
        psi = gaussian_nd(grid, sigmas=[0.5, 0.5])
        norm = np.linalg.norm(psi)
        assert abs(norm - 1.0) < 0.01

    def test_2d_product_matches_tensor(self):
        """2D Gaussian should equal tensor product of two 1D Gaussians."""
        from wings.nd_grid import NDGrid, gaussian_nd
        grid = NDGrid(n_qubits_per_dim=[4, 4], box_sizes=[4.0, 4.0])
        psi_2d = gaussian_nd(grid, sigmas=[0.5, 0.8])

        # Build product state manually
        x = np.linspace(-4, 4, 16)
        psi_x = np.exp(-x**2 / (2*0.5**2)); psi_x /= np.linalg.norm(psi_x)
        psi_y = np.exp(-x**2 / (2*0.8**2)); psi_y /= np.linalg.norm(psi_y)
        psi_product = np.outer(psi_x, psi_y).reshape(-1).astype(np.complex128)
        psi_product /= np.linalg.norm(psi_product)

        fid = np.abs(np.vdot(psi_2d, psi_product))**2
        assert fid > 0.99

    def test_1d_backward_compat(self):
        """1D grid should produce same result as direct 1D computation."""
        from wings.nd_grid import NDGrid, gaussian_nd
        grid = NDGrid(n_qubits_per_dim=[6], box_sizes=[4.0])
        psi = gaussian_nd(grid, sigmas=[0.5], centers=[0.0])

        x = np.linspace(-4, 4, 64)
        expected = np.exp(-x**2 / (2*0.5**2)).astype(np.complex128)
        expected /= np.linalg.norm(expected)

        fid = np.abs(np.vdot(psi, expected))**2
        assert fid > 0.999

    def test_shifted_2d_gaussian(self):
        from wings.nd_grid import NDGrid, gaussian_nd
        grid = NDGrid(n_qubits_per_dim=[4, 4], box_sizes=[4.0, 4.0])
        psi = gaussian_nd(grid, sigmas=[0.5, 0.5], centers=[1.0, -1.0])
        assert psi.shape == (256,)
        assert np.all(np.isfinite(psi))

    def test_2d_gaussian_optimizable(self):
        """2D Gaussian should work as target in GaussianOptimizer."""
        from wings import GaussianOptimizer, OptimizerConfig, TargetFunction
        from wings.nd_grid import NDGrid, gaussian_nd

        grid = NDGrid(n_qubits_per_dim=[4, 4], box_sizes=[4.0, 4.0])
        target_fn = lambda x: gaussian_nd(grid, sigmas=[0.5, 0.5])

        # Use as custom target (the optimizer calls custom_target_fn(positions))
        # For multi-dim, we provide the pre-computed state directly
        psi_2d = gaussian_nd(grid, sigmas=[0.5, 0.5])

        config = OptimizerConfig(
            n_qubits=8, sigma=0.5,
            target_function=TargetFunction.CUSTOM,
            custom_target_fn=lambda x: psi_2d,  # Return pre-computed
            box_size=4.0,
            verbose=False, use_gpu=False, use_custatevec=False,
        )
        opt = GaussianOptimizer(config)
        params = opt.get_initial_params("random")
        fid = opt.compute_fidelity(params=params)
        assert 0 <= fid <= 1
