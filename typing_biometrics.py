"""
Typing Biometrics module for behavioral authentication.

Captures typing dynamics, extracts features such as WPM, typing speed,
character/word accuracy, and trains a Random Forest classifier to
distinguish genuine users from impostors.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os


FEATURE_NAMES = [
    "total_time",
    "typing_speed",
    "wpm",
    "char_accuracy",
    "word_accuracy",
    "avg_time_per_word",
    "avg_time_per_char",
]

CONFIDENCE_THRESHOLD = 0.7


def extract_features(target_text: str, typed_text: str, elapsed_time: float) -> dict:
    """
    Extract typing biometric features from a single typing session.

    Parameters
    ----------
    target_text : str
        The text the user was asked to type.
    typed_text : str
        The actual text typed by the user.
    elapsed_time : float
        Total time (seconds) taken to complete typing.

    Returns
    -------
    dict
        A dictionary with feature names as keys and numeric values.
    """
    if elapsed_time <= 0:
        elapsed_time = 1e-6

    total_chars = len(typed_text)
    total_words = len(typed_text.split())

    target_chars = list(target_text)
    typed_chars = list(typed_text)
    min_len = min(len(target_chars), len(typed_chars))
    correct_chars = sum(1 for i in range(min_len) if target_chars[i] == typed_chars[i])

    char_accuracy = correct_chars / max(len(target_chars), 1)

    target_words = target_text.split()
    typed_words = typed_text.split()
    min_words = min(len(target_words), len(typed_words))
    correct_words = sum(
        1 for i in range(min_words) if target_words[i] == typed_words[i]
    )
    word_accuracy = correct_words / max(len(target_words), 1)

    typing_speed = total_chars / elapsed_time
    wpm = (total_words / elapsed_time) * 60.0
    avg_time_per_word = elapsed_time / max(total_words, 1)
    avg_time_per_char = elapsed_time / max(total_chars, 1)

    return {
        "total_time": elapsed_time,
        "typing_speed": typing_speed,
        "wpm": wpm,
        "char_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "avg_time_per_word": avg_time_per_word,
        "avg_time_per_char": avg_time_per_char,
    }


def features_to_vector(features: dict) -> np.ndarray:
    """Convert a feature dictionary to a NumPy feature vector."""
    return np.array([features[name] for name in FEATURE_NAMES])


def generate_impostor_samples(
    genuine_vectors: np.ndarray, n_samples: int = 200
) -> np.ndarray:
    """
    Generate synthetic impostor samples that represent different user typing profiles.

    Different users exhibit clearly different typing behaviours: some type much
    slower, some much faster, and accuracy patterns vary widely.  We simulate
    this by randomly scaling the genuine user's mean feature vector by factors
    that place the synthetic samples clearly outside the genuine distribution.

    Parameters
    ----------
    genuine_vectors : np.ndarray, shape (n, n_features)
        Feature vectors from the genuine user.
    n_samples : int
        Number of synthetic impostor samples to generate.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
    """
    rng = np.random.default_rng(seed=42)
    mean = genuine_vectors.mean(axis=0)
    noise_std = np.abs(mean) * 0.05 + 1e-6

    impostor_vectors = []
    for _ in range(n_samples):
        # Alternate between slower and faster typists so both tails are covered
        if rng.random() < 0.5:
            scale = rng.uniform(0.25, 0.65)   # much slower than genuine user
        else:
            scale = rng.uniform(1.45, 3.0)    # much faster than genuine user

        sample = mean * scale + rng.normal(0, noise_std)

        # Accuracy features must stay in [0, 1]
        sample[FEATURE_NAMES.index("char_accuracy")] = np.clip(
            sample[FEATURE_NAMES.index("char_accuracy")], 0.0, 1.0
        )
        sample[FEATURE_NAMES.index("word_accuracy")] = np.clip(
            sample[FEATURE_NAMES.index("word_accuracy")], 0.0, 1.0
        )
        sample = np.clip(sample, 0.0, None)
        impostor_vectors.append(sample)

    return np.array(impostor_vectors)


class TypingAuthenticator:
    """
    Trains and uses a Random Forest model to authenticate users based on
    typing biometric features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._is_trained = False

    def train(
        self,
        genuine_vectors: np.ndarray,
        n_impostor_samples: int = 200,
    ) -> dict:
        """
        Train the classifier using genuine samples and synthetic impostor data.

        Parameters
        ----------
        genuine_vectors : np.ndarray, shape (n, n_features)
        n_impostor_samples : int

        Returns
        -------
        dict with 'train_accuracy' and 'test_accuracy'.
        """
        impostor_vectors = generate_impostor_samples(
            genuine_vectors, n_samples=n_impostor_samples
        )

        X = np.vstack([genuine_vectors, impostor_vectors])
        y = np.array([1] * len(genuine_vectors) + [0] * len(impostor_vectors))

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self._is_trained = True

        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)

        return {"train_accuracy": train_acc, "test_accuracy": test_acc}

    def authenticate(self, feature_vector: np.ndarray) -> tuple:
        """
        Predict whether a feature vector belongs to the genuine user.

        Parameters
        ----------
        feature_vector : np.ndarray, shape (n_features,)

        Returns
        -------
        (authenticated: bool, confidence: float)
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        vector = feature_vector.reshape(1, -1)
        vector_scaled = self.scaler.transform(vector)
        proba = self.model.predict_proba(vector_scaled)[0]
        genuine_class_idx = list(self.model.classes_).index(1)
        confidence = proba[genuine_class_idx]
        authenticated = bool(confidence >= CONFIDENCE_THRESHOLD)

        return authenticated, confidence

    def save(self, path: str) -> None:
        """Persist the trained scaler and model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model}, f)

    def load(self, path: str) -> None:
        """Load a previously saved scaler and model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self._is_trained = True


def collect_typing_sample(target_text: str) -> tuple:
    """
    Interactively collect a single typing sample from the user.

    Parameters
    ----------
    target_text : str
        The passage the user should type.

    Returns
    -------
    (typed_text: str, elapsed_time: float)
    """
    print(f"\nPlease type the following text:\n  \"{target_text}\"")
    print("Press ENTER when ready, then start typing. Press ENTER again when done.\n")
    input()
    start = time.time()
    typed = input(">>> ")
    elapsed = time.time() - start
    return typed, elapsed
