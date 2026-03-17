# Behavioral Biometric Authentication System 

## Overview
This project was developed during my internship and implements a multi-modal behavioral biometric authentication system. It combines typing dynamics with computer vision-based behavioral analysis (eye blink patterns and head movement) to verify user identity.

---

## Key Features

### Typing Biometrics
- Typing speed, WPM, accuracy
- Time per word/character
- Machine learning authentication (Random Forest)

### Visual Biometrics
- Face tracking using MediaPipe
- Eye blink detection using Eye Aspect Ratio (EAR)
- Head and body movement tracking
- Real-time webcam processing using OpenCV

---

## Technologies Used
- Python
- NumPy
- Scikit-learn
- OpenCV
- MediaPipe

---

## Machine Learning
- Random Forest Classifier
- StandardScaler for feature scaling
- Synthetic impostor data generation

---

## How It Works
1. User registers with typing samples  
2. Features are extracted  
3. ML model is trained  
4. During login, behavior is compared  
5. System authenticates based on confidence  

---

## Applications
- Secure login systems  
- Fraud detection  
- Continuous authentication  

---

## Author
Arush Gupta
