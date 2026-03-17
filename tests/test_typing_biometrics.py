"""
Tests for the typing_biometrics module.
"""

import numpy as np
import pytest

from typing_biometrics import (
    extract_features,
    features_to_vector,
    generate_impostor_samples,
    TypingAuthenticator,
    FEATURE_NAMES,
    CONFIDENCE_THRESHOLD,
)

TARGET_TEXT = "The quick brown fox"


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_returns_all_feature_keys(self):
        features = extract_features(TARGET_TEXT, TARGET_TEXT, elapsed_time=3.0)
        assert set(features.keys()) == set(FEATURE_NAMES)

    def test_perfect_accuracy(self):
        features = extract_features(TARGET_TEXT, TARGET_TEXT, elapsed_time=3.0)
        assert features["char_accuracy"] == pytest.approx(1.0)
        assert features["word_accuracy"] == pytest.approx(1.0)

    def test_partial_accuracy(self):
        typed = "The quick brown dog"  # last word differs
        features = extract_features(TARGET_TEXT, typed, elapsed_time=3.0)
        assert 0.0 < features["char_accuracy"] < 1.0
        assert 0.0 < features["word_accuracy"] < 1.0

    def test_wpm_positive(self):
        features = extract_features(TARGET_TEXT, TARGET_TEXT, elapsed_time=5.0)
        assert features["wpm"] > 0

    def test_typing_speed_proportional_to_length(self):
        short_text = "hi"
        long_text = "hello world how are you"
        f_short = extract_features(short_text, short_text, elapsed_time=1.0)
        f_long = extract_features(long_text, long_text, elapsed_time=1.0)
        assert f_long["typing_speed"] > f_short["typing_speed"]

    def test_zero_elapsed_time_does_not_raise(self):
        # elapsed_time=0 should be handled without ZeroDivisionError
        features = extract_features(TARGET_TEXT, TARGET_TEXT, elapsed_time=0.0)
        for key in FEATURE_NAMES:
            assert np.isfinite(features[key])

    def test_empty_typed_text(self):
        features = extract_features(TARGET_TEXT, "", elapsed_time=1.0)
        assert features["char_accuracy"] == pytest.approx(0.0)
        assert features["word_accuracy"] == pytest.approx(0.0)

    def test_total_time_matches_elapsed(self):
        elapsed = 7.5
        features = extract_features(TARGET_TEXT, TARGET_TEXT, elapsed_time=elapsed)
        assert features["total_time"] == pytest.approx(elapsed)


# ---------------------------------------------------------------------------
# features_to_vector
# ---------------------------------------------------------------------------


class TestFeaturesToVector:
    def test_vector_length(self):
        features = extract_features(TARGET_TEXT, TARGET_TEXT, elapsed_time=3.0)
        vec = features_to_vector(features)
        assert len(vec) == len(FEATURE_NAMES)

    def test_vector_is_numpy_array(self):
        features = extract_features(TARGET_TEXT, TARGET_TEXT, elapsed_time=3.0)
        vec = features_to_vector(features)
        assert isinstance(vec, np.ndarray)

    def test_vector_order_matches_feature_names(self):
        features = extract_features(TARGET_TEXT, TARGET_TEXT, elapsed_time=3.0)
        vec = features_to_vector(features)
        for i, name in enumerate(FEATURE_NAMES):
            assert vec[i] == pytest.approx(features[name])


# ---------------------------------------------------------------------------
# generate_impostor_samples
# ---------------------------------------------------------------------------


class TestGenerateImpostorSamples:
    def _make_genuine_vectors(self, n=10):
        rng = np.random.default_rng(seed=0)
        return rng.uniform(0.5, 2.0, size=(n, len(FEATURE_NAMES)))

    def test_output_shape(self):
        genuine = self._make_genuine_vectors()
        impostors = generate_impostor_samples(genuine, n_samples=50)
        assert impostors.shape == (50, len(FEATURE_NAMES))

    def test_impostors_non_negative(self):
        genuine = self._make_genuine_vectors()
        impostors = generate_impostor_samples(genuine, n_samples=100)
        assert (impostors >= 0).all()

    def test_impostors_differ_from_genuine_mean(self):
        genuine = self._make_genuine_vectors()
        impostors = generate_impostor_samples(genuine, n_samples=100)
        genuine_mean = genuine.mean(axis=0)
        impostor_mean = impostors.mean(axis=0)
        # The means should not be identical
        assert not np.allclose(genuine_mean, impostor_mean)


# ---------------------------------------------------------------------------
# TypingAuthenticator
# ---------------------------------------------------------------------------


def _make_consistent_genuine_vectors(n=20) -> np.ndarray:
    """Return a set of vectors that simulate a genuine user with low variance."""
    rng = np.random.default_rng(seed=7)
    base = np.array([5.0, 8.0, 48.0, 0.92, 0.90, 0.6, 0.12])
    noise = rng.normal(0, 0.05, size=(n, len(FEATURE_NAMES)))
    return base + noise


class TestTypingAuthenticator:
    def test_train_returns_accuracy_metrics(self):
        genuine = _make_consistent_genuine_vectors()
        auth = TypingAuthenticator()
        metrics = auth.train(genuine)
        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        assert 0.0 <= metrics["test_accuracy"] <= 1.0

    def test_authenticate_returns_bool_and_float(self):
        genuine = _make_consistent_genuine_vectors()
        auth = TypingAuthenticator()
        auth.train(genuine)
        result, confidence = auth.authenticate(genuine[0])
        assert isinstance(result, bool)
        assert isinstance(confidence, float)

    def test_genuine_sample_authenticates(self):
        genuine = _make_consistent_genuine_vectors(n=50)
        auth = TypingAuthenticator()
        auth.train(genuine)
        # The mean of genuine vectors is the most representative genuine sample
        mean_vec = genuine.mean(axis=0)
        result, confidence = auth.authenticate(mean_vec)
        assert confidence >= CONFIDENCE_THRESHOLD
        assert result is True

    def test_impostor_sample_rejected(self):
        # Genuine user: base typing profile
        genuine = _make_consistent_genuine_vectors(n=50)
        base = np.array([5.0, 8.0, 48.0, 0.92, 0.90, 0.6, 0.12])
        # A much slower typist (0.4× genuine speed) — clearly a different user
        impostor_vec = base * 0.4
        auth = TypingAuthenticator()
        auth.train(genuine)
        result, confidence = auth.authenticate(impostor_vec)
        assert confidence < CONFIDENCE_THRESHOLD
        assert result is False

    def test_authenticate_before_training_raises(self):
        auth = TypingAuthenticator()
        with pytest.raises(RuntimeError, match="not trained"):
            auth.authenticate(np.zeros(len(FEATURE_NAMES)))

    def test_save_and_load(self, tmp_path):
        genuine = _make_consistent_genuine_vectors()
        auth = TypingAuthenticator()
        auth.train(genuine)

        model_file = str(tmp_path / "model.pkl")
        auth.save(model_file)

        auth2 = TypingAuthenticator()
        auth2.load(model_file)

        result1, conf1 = auth.authenticate(genuine[0])
        result2, conf2 = auth2.authenticate(genuine[0])

        assert result1 == result2
        assert conf1 == pytest.approx(conf2)
