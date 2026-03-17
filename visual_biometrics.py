"""
Visual Biometrics module for behavioral authentication.

Uses MediaPipe FaceMesh and Pose estimation together with OpenCV to capture:
  - Eye blink count and average blink duration (via Eye Aspect Ratio)
  - Head movement (nose landmark displacement)
  - Shoulder and body movement (pose landmarks)

The module provides a VisualBiometrics class that processes webcam frames in
real time and accumulates behavioral features over a configurable duration.
"""

import math
import numpy as np

try:
    import cv2
    import mediapipe as mp
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Eye landmark indices for MediaPipe FaceMesh (478-landmark model)
# Each list: [outer, top1, top2, inner, bot2, bot1]
# ---------------------------------------------------------------------------
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Nose tip landmark index (used for head movement tracking)
NOSE_TIP_LANDMARK = 1

# EAR threshold below which a blink is detected
EAR_THRESHOLD = 0.20

# Minimum consecutive frames below threshold to count as a blink
EAR_CONSEC_FRAMES = 2


def _euclidean(p1, p2) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(landmarks, eye_indices, img_w: int, img_h: int) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a set of 6 eye landmarks.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A lower EAR indicates the eye is closing (blink).

    Parameters
    ----------
    landmarks : list of MediaPipe NormalizedLandmark
    eye_indices : list of int, length 6
        Landmark indices [p1, p2, p3, p4, p5, p6].
    img_w, img_h : int
        Image dimensions used to convert normalized coordinates to pixels.

    Returns
    -------
    float
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    p1, p2, p3, p4, p5, p6 = pts
    ear = (_euclidean(p2, p6) + _euclidean(p3, p5)) / (2.0 * _euclidean(p1, p4) + 1e-6)
    return ear


class VisualBiometrics:
    """
    Real-time visual biometric feature extractor.

    Processes webcam frames using MediaPipe FaceMesh and Pose to track:
      - Eye blinks (count + average duration)
      - Head movement (nose displacement)
      - Shoulder / body movement (pose landmarks)

    Usage
    -----
    ::

        vb = VisualBiometrics()
        features = vb.capture(duration=15)
        print(features)
    """

    def __init__(self, ear_threshold: float = EAR_THRESHOLD,
                 ear_consec_frames: int = EAR_CONSEC_FRAMES):
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "opencv-python and mediapipe are required for VisualBiometrics. "
                "Install them with: pip install opencv-python mediapipe"
            )
        self.ear_threshold = ear_threshold
        self.ear_consec_frames = ear_consec_frames

        self._mp_face = mp.solutions.face_mesh
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils

    def capture(self, duration: float = 15.0, camera_index: int = 0) -> dict:
        """
        Capture visual behavioral features from the webcam for `duration` seconds.

        Parameters
        ----------
        duration : float
            Number of seconds to record.
        camera_index : int
            OpenCV camera device index.

        Returns
        -------
        dict with keys:
          - blink_count (int)
          - avg_blink_duration (float, seconds)
          - head_movement (float, mean pixel displacement of nose)
          - shoulder_movement (float, mean pixel displacement of mid-shoulder)
          - body_movement (float, mean pixel displacement of hip mid-point)
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera (index={camera_index}).")

        import time

        face_mesh = self._mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose = self._mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        blink_count = 0
        blink_frames = 0
        blink_durations = []
        blink_start = None

        nose_positions = []
        shoulder_positions = []
        hip_positions = []

        start_time = time.time()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # -- FaceMesh -----------------------------------------------
                face_results = face_mesh.process(rgb)
                if face_results.multi_face_landmarks:
                    lm = face_results.multi_face_landmarks[0].landmark

                    left_ear = eye_aspect_ratio(lm, LEFT_EYE_LANDMARKS, w, h)
                    right_ear = eye_aspect_ratio(lm, RIGHT_EYE_LANDMARKS, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < self.ear_threshold:
                        blink_frames += 1
                        if blink_start is None:
                            blink_start = time.time()
                    else:
                        if blink_frames >= self.ear_consec_frames:
                            blink_count += 1
                            if blink_start is not None:
                                blink_durations.append(time.time() - blink_start)
                        blink_frames = 0
                        blink_start = None

                    nose = lm[NOSE_TIP_LANDMARK]
                    nose_positions.append((nose.x * w, nose.y * h))

                # -- Pose estimation ----------------------------------------
                pose_results = pose.process(rgb)
                if pose_results.pose_landmarks:
                    plm = pose_results.pose_landmarks.landmark
                    PoseLandmark = self._mp_pose.PoseLandmark

                    ls = plm[PoseLandmark.LEFT_SHOULDER]
                    rs = plm[PoseLandmark.RIGHT_SHOULDER]
                    mid_shoulder = (
                        (ls.x + rs.x) / 2.0 * w,
                        (ls.y + rs.y) / 2.0 * h,
                    )
                    shoulder_positions.append(mid_shoulder)

                    lh = plm[PoseLandmark.LEFT_HIP]
                    rh = plm[PoseLandmark.RIGHT_HIP]
                    mid_hip = (
                        (lh.x + rh.x) / 2.0 * w,
                        (lh.y + rh.y) / 2.0 * h,
                    )
                    hip_positions.append(mid_hip)

                # -- Display overlay ----------------------------------------
                elapsed = time.time() - start_time
                remaining = max(0, duration - elapsed)
                cv2.putText(
                    frame,
                    f"Recording... {remaining:.1f}s remaining",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Blinks: {blink_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("Visual Biometrics", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            face_mesh.close()
            pose.close()

        head_movement = _mean_displacement(nose_positions)
        shoulder_movement = _mean_displacement(shoulder_positions)
        body_movement = _mean_displacement(hip_positions)
        avg_blink_duration = (
            float(np.mean(blink_durations)) if blink_durations else 0.0
        )

        return {
            "blink_count": blink_count,
            "avg_blink_duration": avg_blink_duration,
            "head_movement": head_movement,
            "shoulder_movement": shoulder_movement,
            "body_movement": body_movement,
        }


def _mean_displacement(positions: list) -> float:
    """
    Compute the mean frame-to-frame Euclidean displacement of a tracked point.

    Parameters
    ----------
    positions : list of (x, y) tuples

    Returns
    -------
    float : mean displacement in pixels (0.0 if fewer than 2 positions).
    """
    if len(positions) < 2:
        return 0.0
    displacements = [
        _euclidean(positions[i], positions[i + 1])
        for i in range(len(positions) - 1)
    ]
    return float(np.mean(displacements))
