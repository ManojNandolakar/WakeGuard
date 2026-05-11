import cv2
import mediapipe as mp
import numpy as np
import time
import math
import subprocess

# ================== CONFIG ==================
EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.60
EYE_CLOSED_TIME = 1.2
YAWN_TIME = 1.2
HEAD_TILT_THRESHOLD = 12
NO_FACE_TIME = 1.2
RECOVERY_SPEED = 0.5
# ============================================

closed_start = None
no_face_start = None
yawn_start = None
attention_score = 100
alarm_process = None
alarm_active = False

# ------------------ SOUND CONTROL ------------------

def start_alarm():
    global alarm_process, alarm_active
    if not alarm_active:
        alarm_process = subprocess.Popen(
            ["afplay", "-v", "2", "alarm.wav"]
        )
        alarm_active = True

def stop_alarm():
    global alarm_process, alarm_active
    if alarm_active and alarm_process:
        alarm_process.terminate()
        alarm_process = None
        alarm_active = False

# ------------------ MediaPipe Setup ------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ------------------ Math ------------------

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_EAR(landmarks, eye, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2 * C)

def compute_MAR(landmarks, w, h):
    top = (int(landmarks[13].x*w), int(landmarks[13].y*h))
    bottom = (int(landmarks[14].x*w), int(landmarks[14].y*h))
    left = (int(landmarks[78].x*w), int(landmarks[78].y*h))
    right = (int(landmarks[308].x*w), int(landmarks[308].y*h))
    return euclidean(top, bottom) / euclidean(left, right)

def compute_head_tilt(landmarks, w, h):
    left_eye = (int(landmarks[33].x*w), int(landmarks[33].y*h))
    right_eye = (int(landmarks[263].x*w), int(landmarks[263].y*h))
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return math.degrees(math.atan2(dy, dx))

# ------------------ Camera ------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)

    current_time = time.time()
    critical = False
    status = "ALERT"
    color = (0, 255, 0)

    # ================= FACE DETECTED =================
    if result.face_landmarks:

        no_face_start = None
        landmarks = result.face_landmarks[0]

        ear = (compute_EAR(landmarks, LEFT_EYE, w, h) +
               compute_EAR(landmarks, RIGHT_EYE, w, h)) / 2
        mar = compute_MAR(landmarks, w, h)
        tilt = compute_head_tilt(landmarks, w, h)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Tilt: {tilt:.2f}", (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # -------- DROWSY --------
        if ear < EAR_THRESHOLD:
            if closed_start is None:
                closed_start = current_time
            elif current_time - closed_start > EYE_CLOSED_TIME:
                status = "DROWSY"
                color = (0, 0, 255)
                attention_score -= 3
                critical = True
        else:
            closed_start = None

        # -------- YAWN --------
        if mar > MAR_THRESHOLD:
            if yawn_start is None:
                yawn_start = current_time
            elif current_time - yawn_start > YAWN_TIME:
                status = "YAWNING"
                color = (0, 165, 255)
                attention_score -= 2
        else:
            yawn_start = None

        # -------- HEAD TILT --------
        if abs(tilt) > HEAD_TILT_THRESHOLD:
            status = "HEAD TILT"
            color = (255, 0, 0)
            attention_score -= 1

        # Recovery
        if not critical:
            attention_score += RECOVERY_SPEED

    # ================= NO FACE =================
    else:
        if no_face_start is None:
            no_face_start = current_time
        elif current_time - no_face_start > NO_FACE_TIME:
            status = "NO DRIVER"
            color = (0, 0, 255)
            attention_score -= 4
            critical = True

    # Clamp score
    attention_score = max(0, min(100, attention_score))

    if attention_score <= 20:
        critical = True

    # ================= CRITICAL RESPONSE =================
    if critical:
        red_overlay = np.zeros_like(frame)
        red_overlay[:] = (0, 0, 255)
        frame = cv2.addWeighted(red_overlay, 0.7, frame, 0.3, 0)
        start_alarm()
    else:
        stop_alarm()

    # ================= UI =================
    cv2.putText(frame, status, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    bar_width = int((attention_score / 100) * 300)

    cv2.rectangle(frame, (30, 110), (330, 140), (50,50,50), -1)
    cv2.rectangle(frame, (30, 110), (30 + bar_width, 140), (0,255,0), -1)

    cv2.putText(frame, f"Attention: {int(attention_score)}%",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2)

    cv2.imshow("AI Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stop_alarm()
cap.release()
cv2.destroyAllWindows()
