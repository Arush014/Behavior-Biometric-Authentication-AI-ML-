"""
Behavioral Biometric Authentication System

Entry point that orchestrates:
  1. Registration  – collects typing samples, trains and saves the RF model.
  2. Authentication – collects a new typing sample and predicts identity.

Optionally combines typing biometrics with a visual biometrics capture session
when a webcam is available.

Run directly:
    python auth_system.py register
    python auth_system.py authenticate
    python auth_system.py register --no-visual
    python auth_system.py authenticate --no-visual
"""

import argparse
import os
import sys
import numpy as np

from typing_biometrics import (
    TypingAuthenticator,
    collect_typing_sample,
    extract_features,
    features_to_vector,
    FEATURE_NAMES,
)

MODEL_PATH = os.path.join("models", "typing_auth.pkl")

TARGET_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Behavioral biometrics capture unique typing patterns."
)

MIN_REGISTRATION_SAMPLES = 5


def register(use_visual: bool = True) -> None:
    """
    Registration phase.

    Collects multiple typing samples, trains a Random Forest classifier, and
    saves the model to disk.  Optionally runs a visual biometrics capture at
    the end of registration to display the user's behavioral baseline.
    """
    print("=" * 60)
    print("  REGISTRATION PHASE")
    print("=" * 60)
    print(
        f"\nYou will be asked to type the same passage {MIN_REGISTRATION_SAMPLES} times."
    )
    print("This builds your unique typing profile.\n")

    genuine_vectors = []

    for i in range(1, MIN_REGISTRATION_SAMPLES + 1):
        print(f"\n--- Sample {i} of {MIN_REGISTRATION_SAMPLES} ---")
        typed, elapsed = collect_typing_sample(TARGET_TEXT)
        features = extract_features(TARGET_TEXT, typed, elapsed)
        vector = features_to_vector(features)
        genuine_vectors.append(vector)

        print("\nExtracted features:")
        for name, val in zip(FEATURE_NAMES, vector):
            print(f"  {name:25s}: {val:.4f}")

    genuine_matrix = np.array(genuine_vectors)

    authenticator = TypingAuthenticator()
    print("\nTraining model …")
    metrics = authenticator.train(genuine_matrix)
    print(f"  Train accuracy : {metrics['train_accuracy']:.2%}")
    print(f"  Test  accuracy : {metrics['test_accuracy']:.2%}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    authenticator.save(MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'.")

    if use_visual:
        _run_visual_capture(phase="registration")

    print("\nRegistration complete.\n")


def authenticate(use_visual: bool = True) -> bool:
    """
    Authentication phase.

    Loads the trained model and evaluates a single typing sample.
    Optionally captures visual biometrics for a secondary verification signal.

    Returns
    -------
    bool : True if authentication is successful, False otherwise.
    """
    print("=" * 60)
    print("  AUTHENTICATION PHASE")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(
            f"\nError: No model found at '{MODEL_PATH}'. "
            "Please run registration first."
        )
        return False

    authenticator = TypingAuthenticator()
    authenticator.load(MODEL_PATH)

    typed, elapsed = collect_typing_sample(TARGET_TEXT)
    features = extract_features(TARGET_TEXT, typed, elapsed)
    vector = features_to_vector(features)

    print("\nExtracted features:")
    for name, val in zip(FEATURE_NAMES, vector):
        print(f"  {name:25s}: {val:.4f}")

    authenticated, confidence = authenticator.authenticate(vector)

    print(f"\nTyping confidence : {confidence:.2%}")

    if use_visual:
        _run_visual_capture(phase="authentication")

    if authenticated:
        print("\n[✓] Authentication SUCCESSFUL")
    else:
        print("\n[✗] Authentication FAILED")

    return authenticated


def _run_visual_capture(phase: str) -> None:
    """Attempt a visual biometrics capture; gracefully skip if unavailable."""
    try:
        from visual_biometrics import VisualBiometrics

        print(
            f"\nStarting visual biometrics capture ({phase}) for 15 seconds …"
        )
        print("Look at the camera naturally. Press 'q' to stop early.\n")

        vb = VisualBiometrics()
        features = vb.capture(duration=15)

        print("\nVisual biometric features:")
        for name, val in features.items():
            print(f"  {name:25s}: {val:.4f}")

    except ImportError as exc:
        print(f"\n[!] Visual biometrics skipped: {exc}")
    except RuntimeError as exc:
        print(f"\n[!] Visual biometrics skipped: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral Biometric Authentication System"
    )
    parser.add_argument(
        "mode",
        choices=["register", "authenticate"],
        help="'register' to enroll a new user, 'authenticate' to verify identity.",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable the visual biometrics capture step.",
    )
    args = parser.parse_args()

    use_visual = not args.no_visual

    if args.mode == "register":
        register(use_visual=use_visual)
    else:
        success = authenticate(use_visual=use_visual)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
