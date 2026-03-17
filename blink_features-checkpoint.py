import numpy as np
import cv2
import mediapipe as mp
import time

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_EAR(eye):
    # FIXED: Use correct landmark indices
    vertical_1 = euclidean_distance(eye[1], eye[5])
    vertical_2 = euclidean_distance(eye[2], eye[3])
    horizontal = euclidean_distance(eye, eye[4])
    if horizontal == 0:
        return 0.0
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def init_blink_state():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True
    )
    state = {
        'face_mesh': face_mesh,
        'blink_count': 0,
        'blink_durations': [],
        'was_closed': False,
        'blink_start_time': None,
        'LEFT_EYE': [33, 160, 158, 133, 153, 144],
        'RIGHT_EYE': [362, 385, 387, 263, 373, 380],
        'EAR_THRESHOLD': 0.3,
        'ear_history': [],
    }
    return state

def process_blink_frame(frame, state, visualize=True):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = state['face_mesh'].process(rgb_frame)
    LEFT_EYE = state['LEFT_EYE']
    RIGHT_EYE = state['RIGHT_EYE']
    
    if results_face.multi_face_landmarks:
        mesh_points = np.array(
            [[p.x * w, p.y * h] for p in results_face.multi_face_landmarks[0].landmark]
        )
        left_eye_pts = mesh_points[LEFT_EYE]
        right_eye_pts = mesh_points[RIGHT_EYE]
        left_ear = calculate_EAR(left_eye_pts)
        right_ear = calculate_EAR(right_eye_pts)
        ear = (left_ear + right_ear) / 2.0
        
        # Store EAR history for analysis
        state['ear_history'].append(ear)
        if len(state['ear_history']) > 30:
            state['ear_history'].pop(0)
        
        # DIAGNOSTIC PRINTS
        print(f"EAR: {ear:.4f} | Threshold: {state['EAR_THRESHOLD']:.4f} | Below: {ear < state['EAR_THRESHOLD']}")
        
        eyes_closed = ear < state['EAR_THRESHOLD']
        
        if eyes_closed and not state['was_closed']:
            state['was_closed'] = True
            state['blink_start_time'] = time.time()
            print("EYES CLOSING...")
            
        elif not eyes_closed and state['was_closed']:
            state['blink_count'] += 1
            if state['blink_start_time']:
                duration = time.time() - state['blink_start_time']
                state['blink_durations'].append(duration)
            
            print(f"BLINK COUNTED! Total: {state['blink_count']}")
            state['was_closed'] = False
            state['blink_start_time'] = None
        
        if visualize:
            # FIXED: Use correct coordinate indexing
            for idx in LEFT_EYE + RIGHT_EYE:
                x, y = int(mesh_points[idx][0]), int(mesh_points[idx][1])  # Fixed!
                cv2.circle(frame, (x, y), 2, (0,255,0), -1)
            
            status = "CLOSED" if eyes_closed else "OPEN"
            color = (0,0,255) if eyes_closed else (0,255,0)
            cv2.putText(frame, status, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame, f"Blinks: {state['blink_count']}", (30, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
            cv2.putText(frame, f"EAR: {ear:.3f}", (30, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"Threshold: {state['EAR_THRESHOLD']:.3f}", (30, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    return state

def get_blink_stats(state):
    avg_blink_duration = np.mean(state['blink_durations']) if state['blink_durations'] else 0.0
    if state['ear_history']:
        print(f"\nEAR Analysis:")
        print(f"Min EAR: {min(state['ear_history']):.4f}")
        print(f"Max EAR: {max(state['ear_history']):.4f}")
        print(f"Avg EAR: {np.mean(state['ear_history']):.4f}")
        print(f"Current Threshold: {state['EAR_THRESHOLD']:.4f}")
    
    return {
        "blink_count": state['blink_count'],
        "avg_blink_duration": avg_blink_duration
    }
