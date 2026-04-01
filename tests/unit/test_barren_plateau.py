"""Tests for barren plateau detection."""

import numpy as np
import pytest

from wings.barren_plateau import BarrenPlateauDetector


@pytest.mark.unit
class TestBarrenPlateauDetector:
    """Tests for BarrenPlateauDetector."""

    def test_initialization(self):
        """Fresh detector should not flag barren plateau."""
        detector = BarrenPlateauDetector(36, window=50)
        assert not detector.is_barren_plateau()

    def test_no_detection_with_large_gradients(self):
        """Large gradients should not trigger detection."""
        detector = BarrenPlateauDetector(36, window=10)
        for _ in range(20):
            grad = np.random.randn(36) * 0.5
            detector.update(grad, 0.5)
        assert not detector.is_barren_plateau()

    def test_detects_vanishing_gradients(self):
        """Tiny gradients at low fidelity should trigger detection."""
        detector = BarrenPlateauDetector(36, window=10)
        for _ in range(20):
            grad = np.random.randn(36) * 1e-8
            detector.update(grad, 0.3)
        assert detector.is_barren_plateau()

    def test_no_detection_near_convergence(self):
        """Tiny gradients near convergence (high fidelity) should NOT trigger."""
        detector = BarrenPlateauDetector(36, window=10)
        for _ in range(20):
            grad = np.random.randn(36) * 1e-8
            detector.update(grad, 0.999)
        assert not detector.is_barren_plateau()

    def test_reset(self):
        """After reset, detection state should be cleared."""
        detector = BarrenPlateauDetector(36, window=10)
        for _ in range(20):
            grad = np.random.randn(36) * 1e-8
            detector.update(grad, 0.3)
        assert detector.is_barren_plateau()
        detector.reset()
        assert not detector.is_barren_plateau()
        assert len(detector.gradient_norm_history) == 0

    def test_suggest_mitigation(self):
        """Should return a non-empty mitigation string."""
        detector = BarrenPlateauDetector(36)
        result = detector.suggest_mitigation()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_gradient_norm_history(self):
        """History should track all updates."""
        detector = BarrenPlateauDetector(36, window=50)
        for _ in range(10):
            grad = np.random.randn(36)
            detector.update(grad, 0.5)
        assert len(detector.gradient_norm_history) == 10
