"""
Tests for the visual_biometrics module.

Camera-dependent functionality is tested using mocks so that the tests can
run in a headless CI environment without a physical webcam.
"""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from visual_biometrics import (
    _euclidean,
    _mean_displacement,
    eye_aspect_ratio,
    EAR_THRESHOLD,
)


# ---------------------------------------------------------------------------
# _euclidean
# ---------------------------------------------------------------------------


class TestEuclidean:
    def test_same_point(self):
        assert _euclidean((0, 0), (0, 0)) == pytest.approx(0.0)

    def test_horizontal(self):
        assert _euclidean((0, 0), (3, 0)) == pytest.approx(3.0)

    def test_vertical(self):
        assert _euclidean((0, 0), (0, 4)) == pytest.approx(4.0)

    def test_diagonal(self):
        assert _euclidean((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_float_coordinates(self):
        assert _euclidean((1.5, 2.5), (4.5, 6.5)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _mean_displacement
# ---------------------------------------------------------------------------


class TestMeanDisplacement:
    def test_empty_list(self):
        assert _mean_displacement([]) == pytest.approx(0.0)

    def test_single_point(self):
        assert _mean_displacement([(10, 20)]) == pytest.approx(0.0)

    def test_two_identical_points(self):
        assert _mean_displacement([(5, 5), (5, 5)]) == pytest.approx(0.0)

    def test_two_points_known_distance(self):
        # distance = 5 (3-4-5 triangle)
        assert _mean_displacement([(0, 0), (3, 4)]) == pytest.approx(5.0)

    def test_multiple_points_mean(self):
        # displacements: 1, 1, 1 → mean = 1
        positions = [(0, 0), (1, 0), (2, 0), (3, 0)]
        assert _mean_displacement(positions) == pytest.approx(1.0)

    def test_returns_float(self):
        result = _mean_displacement([(0, 0), (1, 0)])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# eye_aspect_ratio
# ---------------------------------------------------------------------------


def _make_landmark(x_norm, y_norm):
    lm = MagicMock()
    lm.x = x_norm
    lm.y = y_norm
    return lm


def _open_eye_landmarks(img_w=100, img_h=100):
    """
    Construct 6 landmarks representing an open eye.

    Layout (pixel coords):
      p1=(10,50), p2=(30,30), p3=(70,30), p4=(90,50), p5=(70,70), p6=(30,70)

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        = (||p2-p6|| + ||p3-p5||) / (2 * 80)

    p2=(30,30), p6=(30,70) → distance = 40
    p3=(70,30), p5=(70,70) → distance = 40
    EAR = (40 + 40) / (2 * 80) = 80 / 160 = 0.5
    """
    coords = [(10, 50), (30, 30), (70, 30), (90, 50), (70, 70), (30, 70)]
    return [_make_landmark(x / img_w, y / img_h) for x, y in coords]


def _closed_eye_landmarks(img_w=100, img_h=100):
    """
    Landmarks for a nearly closed eye (very small vertical distance).

    p1=(10,50), p2=(30,49), p3=(70,49), p4=(90,50), p5=(70,51), p6=(30,51)

    EAR ≈ (2 + 2) / (2 * 80) = 4 / 160 = 0.025
    """
    coords = [(10, 50), (30, 49), (70, 49), (90, 50), (70, 51), (30, 51)]
    return [_make_landmark(x / img_w, y / img_h) for x, y in coords]


class TestEyeAspectRatio:
    def test_open_eye_above_threshold(self):
        landmarks = _open_eye_landmarks()
        # eye_aspect_ratio takes the full landmark list; indices point into it
        indices = [0, 1, 2, 3, 4, 5]
        ear = eye_aspect_ratio(landmarks, indices, img_w=100, img_h=100)
        assert ear > EAR_THRESHOLD

    def test_closed_eye_below_threshold(self):
        landmarks = _closed_eye_landmarks()
        indices = [0, 1, 2, 3, 4, 5]
        ear = eye_aspect_ratio(landmarks, indices, img_w=100, img_h=100)
        assert ear < EAR_THRESHOLD

    def test_open_eye_ear_value(self):
        landmarks = _open_eye_landmarks()
        indices = [0, 1, 2, 3, 4, 5]
        ear = eye_aspect_ratio(landmarks, indices, img_w=100, img_h=100)
        assert ear == pytest.approx(0.5, abs=1e-3)

    def test_ear_non_negative(self):
        landmarks = _open_eye_landmarks()
        indices = [0, 1, 2, 3, 4, 5]
        ear = eye_aspect_ratio(landmarks, indices, img_w=100, img_h=100)
        assert ear >= 0.0


# ---------------------------------------------------------------------------
# VisualBiometrics (dependency-injected / mocked)
# ---------------------------------------------------------------------------


class TestVisualBiometricsImportGuard:
    """Verify that VisualBiometrics raises ImportError when deps are missing."""

    def test_import_error_without_cv2(self):
        import visual_biometrics as vb_mod

        original = vb_mod._DEPS_AVAILABLE
        try:
            vb_mod._DEPS_AVAILABLE = False
            with pytest.raises(ImportError):
                vb_mod.VisualBiometrics()
        finally:
            vb_mod._DEPS_AVAILABLE = original
