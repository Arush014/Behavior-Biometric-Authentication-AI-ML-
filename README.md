# Behavioral Biometric Authentication (AI/ML)

A multi-modal behavioral biometric authentication system developed during an internship.  
It combines **typing dynamics** with **computer-vision-based behavioral analysis** (eye blink patterns and head/body movement) to verify user identity without relying on passwords.

---

## Overview

Traditional authentication methods (passwords, PINs) are vulnerable to theft and guessing.  
Behavioral biometrics capture *how* a person interacts with a device — patterns that are unique and difficult to replicate.

This system fuses two complementary modalities:

| Modality | Signal | Features |
|---|---|---|
| **Typing biometrics** | Keyboard dynamics | WPM, typing speed, character/word accuracy, avg time per word/char |
| **Visual biometrics** | Webcam feed | Blink count, avg blink duration, head movement, shoulder movement, body movement |

---

## Key Features

### Typing Biometrics (`typing_biometrics.py`)
- Captures typing speed, rhythm, and accuracy
- Extracts features: WPM, typing speed (chars/sec), character accuracy, word accuracy, average time per word/character
- Trains a **Random Forest** classifier with **StandardScaler** normalisation
- Generates synthetic impostor data (slow and fast typists) for robust training
- Decision threshold: **0.70** confidence

### Visual Behavioral Biometrics (`visual_biometrics.py`)
- Real-time face tracking using **MediaPipe FaceMesh**
- Eye blink detection using **Eye Aspect Ratio (EAR)**
- Blink count and average blink duration analysis
- Head movement tracking via nose landmarks
- Shoulder and body movement tracking via **MediaPipe Pose**
- Webcam-based real-time processing using **OpenCV**

---

## Technologies

- Python 3.8+
- NumPy
- scikit-learn (Random Forest, StandardScaler)
- OpenCV (`opencv-python`)
- MediaPipe

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Registration (enrol a new user)

```bash
python auth_system.py register
```

You will be prompted to type a passage **5 times** to build your typing profile.
The trained model is saved to `models/typing_auth.pkl`.

To skip the webcam step:

```bash
python auth_system.py register --no-visual
```

### Authentication (verify identity)

```bash
python auth_system.py authenticate
```

Type the same passage once.  The system reports whether authentication succeeded and the confidence score.

```bash
python auth_system.py authenticate --no-visual
```

---

## System Architecture

### 1 — Registration Phase
1. User provides 5 typing samples of a fixed passage.
2. Behavioral features are extracted from each sample.
3. Synthetic impostor data is generated (slower and faster typists).
4. A Random Forest classifier is trained (80/20 train/test split).
5. The model is serialised to disk.

### 2 — Authentication Phase
1. User types the passage once.
2. Features are extracted in real time.
3. The model predicts genuine vs impostor with a probability score.
4. Access is granted when confidence ≥ 0.70.

### 3 — Visual Behavioral Pattern Detection
1. Webcam captures a 15-second real-time stream.
2. MediaPipe FaceMesh extracts 478 facial landmarks per frame.
3. EAR is computed each frame; blinks are counted when EAR < 0.20 for ≥ 2 consecutive frames.
4. Nose landmark displacement measures head movement.
5. MediaPipe Pose extracts shoulder and hip landmarks for body movement.

---

## Eye Aspect Ratio (EAR)

```
       p2  p3
p1  ●──●──●──●  p4
       p6  p5

EAR = (||p2–p6|| + ||p3–p5||) / (2 × ||p1–p4||)
```

- EAR decreases when the eye closes (blink).
- EAR rises above threshold when the eye reopens.
- Threshold: **0.20**; minimum consecutive frames below threshold: **2**.

---

## Machine Learning Details

| Parameter | Value |
|---|---|
| Model | Random Forest (100 estimators) |
| Feature scaling | StandardScaler |
| Train / test split | 80 % / 20 % |
| Synthetic impostors | 200 samples (slow and fast typists) |
| Decision threshold | 0.70 confidence |

---

## Features Extracted

### Typing Features
| Feature | Description |
|---|---|
| `total_time` | Total elapsed typing time (seconds) |
| `typing_speed` | Characters typed per second |
| `wpm` | Words per minute |
| `char_accuracy` | Fraction of correctly typed characters |
| `word_accuracy` | Fraction of correctly typed words |
| `avg_time_per_word` | Average seconds per word |
| `avg_time_per_char` | Average seconds per character |

### Visual Features
| Feature | Description |
|---|---|
| `blink_count` | Total blinks detected in the session |
| `avg_blink_duration` | Mean duration of each blink (seconds) |
| `head_movement` | Mean frame-to-frame nose displacement (pixels) |
| `shoulder_movement` | Mean mid-shoulder point displacement (pixels) |
| `body_movement` | Mean mid-hip point displacement (pixels) |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests cover feature extraction, impostor generation, model training, model persistence, EAR computation, and displacement calculation.  
Camera-dependent code is fully mocked so tests run in headless / CI environments.
