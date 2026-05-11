print("FILE LOADED")
# --- ai/driver_monitor.py ---
import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist

print("IMPORTS DONE")

class DriverMonitor:
    def __init__(self):
        # Setup MediaPipe Tasks API
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

        # Eye landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def compute_EAR(self, landmarks, eye, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]

        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        C = dist.euclidean(pts[0], pts[3])

        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = int(time.time() * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = self.compute_EAR(landmarks, self.LEFT_EYE, w, h)
            right_ear = self.compute_EAR(landmarks, self.RIGHT_EYE, w, h)

            ear = (left_ear + right_ear) / 2.0

            if ear < 0.23:
                cv2.putText(frame, "DROWSY ALERT",
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

        return frame