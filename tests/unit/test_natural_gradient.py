"""Tests for natural gradient descent with diagonal QFIM."""

import numpy as np
import pytest


@pytest.mark.unit
class TestQuantumFisherInformation:
    """Tests for QFIM diagonal and natural gradient computation."""

    def test_qfim_diagonal_shape(self, small_optimizer, random_params_6q):
        """QFIM diagonal should have shape (n_params,)."""
        from wings.natural_gradient import compute_qfim_diagonal

        n_params = len(random_params_6q)
        qfim_diag = compute_qfim_diagonal(small_optimizer, random_params_6q)
        assert qfim_diag.shape == (n_params,)

    def test_qfim_diagonal_nonnegative(self, small_optimizer, random_params_6q):
        """QFIM diagonal entries should be non-negative (up to numerical noise)."""
        from wings.natural_gradient import compute_qfim_diagonal

        qfim_diag = compute_qfim_diagonal(small_optimizer, random_params_6q)
        assert np.all(qfim_diag >= -1e-10)

    def test_qfim_diagonal_finite(self, small_optimizer, random_params_6q):
        """QFIM diagonal entries should all be finite."""
        from wings.natural_gradient import compute_qfim_diagonal

        qfim_diag = compute_qfim_diagonal(small_optimizer, random_params_6q)
        assert np.all(np.isfinite(qfim_diag))

    def test_natural_gradient_shape(self, small_optimizer, random_params_6q):
        """Natural gradient should have shape (n_params,)."""
        from wings.natural_gradient import compute_natural_gradient

        n_params = len(random_params_6q)
        nat_grad = compute_natural_gradient(small_optimizer, random_params_6q)
        assert nat_grad.shape == (n_params,)

    def test_natural_gradient_finite(self, small_optimizer, random_params_6q):
        """Natural gradient entries should all be finite."""
        from wings.natural_gradient import compute_natural_gradient

        nat_grad = compute_natural_gradient(small_optimizer, random_params_6q)
        assert np.all(np.isfinite(nat_grad))

    def test_natural_gradient_not_zero(self, small_optimizer, random_params_6q):
        """Natural gradient should have at least one nonzero entry."""
        from wings.natural_gradient import compute_natural_gradient

        nat_grad = compute_natural_gradient(small_optimizer, random_params_6q)
        assert np.any(nat_grad != 0.0)

    def test_regularization_prevents_explosion(self, small_optimizer):
        """Zero params with regularization should give finite, bounded natural gradient."""
        from wings.natural_gradient import compute_natural_gradient

        zero_params = np.zeros(36)
        nat_grad = compute_natural_gradient(small_optimizer, zero_params, regularization=0.01)
        assert np.all(np.isfinite(nat_grad))
        assert np.linalg.norm(nat_grad) < 1e6
