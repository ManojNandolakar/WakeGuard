# AI Driver Monitoring System 🚗💤

An intelligent real-time AI-based Driver Monitoring System built using **Python, Flask, OpenCV, and MediaPipe** to detect unsafe driving behaviors such as drowsiness, yawning, distraction, phone usage, and driver absence.

The system uses Computer Vision and facial landmark detection to monitor the driver's attention level and provide real-time alerts to help prevent road accidents.


Main Features
1. Real-Time Face Detection

The system continuously detects the driver’s face using MediaPipe Face Landmarker.

Purpose
-- Ensures the driver is visible
-- Tracks facial landmarks in real time
-- Acts as the base for all detection modules
2. Drowsiness Detection

The system calculates the Eye Aspect Ratio (EAR) to determine whether the driver’s eyes are closed for a dangerous duration.

Working
-- Detects eye landmarks
-- Measures eye openness
    -- If eyes remain closed beyond threshold time:
    -- Driver marked as drowsy
Alert triggered
-- Technologies Used
-- MediaPipe Face Mesh
-- OpenCV
-- NumPy


3. Yawning Detection
The system calculates the Mouth Aspect Ratio (MAR) to detect yawning behavior.

Working
Tracks mouth landmarks
Measures mouth opening distance
If mouth remains open for a specific duration:
Yawning alert generated
Importance
Yawning is a major indicator of fatigue and low alertness.

4. Head Tilt Detection
The application detects abnormal head movement and tilt angles.

Working
Tracks eye positions
Calculates head angle using trigonometry
Detects:
Sleeping posture
Distraction
Unusual head orientation


5. Phone Usage Detection
The system uses hand landmark detection to identify possible phone usage while driving.

Working
Detects hand presence
Tracks continuous hand visibility
If detected for long duration:
Marks driver as distracted


6. No Driver Detection
If the driver’s face disappears from the camera for a specific time:

Warning generated
Attention score reduced

This helps detect:

Driver leaving the seat
Camera obstruction
Unsafe conditions


7. Attention Score System
The project maintains a dynamic Attention Score between 0–100.

Behavior
Safe driving increases score
Unsafe behavior decreases score
Factors Affecting Score
Drowsiness
Yawning
Phone usage
Head tilt
No face detected
Purpose
Provides a simple real-time representation of driver focus level.

8. Alarm System
When dangerous conditions are detected:

Audio alarm activates
Visual warnings appear
Red danger overlay displayed
Purpose
Immediately alert the driver before accidents occur.

9. Live Video Streaming
The system streams processed webcam footage directly to the browser using Flask.

Features
Real-time monitoring
Browser-based interface
No external software required


10. User Authentication System
The project includes secure login and signup functionality.

Features
User registration
Password authentication
Session handling
Logout functionality
Security

Passwords are stored using hashing.

11. Web Dashboard
The system includes a modern dashboard interface.

Dashboard Sections
Home
Live Monitoring
Model Code Viewer
About Page
Team Section

# Technologies Used

- Python
- Flask
- OpenCV
- MediaPipe
- NumPy
- HTML/CSS/JavaScript

---

# Machine Learning & AI Concepts

- Facial Landmark Detection
- Real-Time Computer Vision
- Human Attention Monitoring
- Behavioral Analysis
- Feature Extraction

---

# System Workflow

1. Capture live video from webcam
2. Detect face and facial landmarks using MediaPipe
3. Calculate EAR and MAR values
4. Analyze driver behavior:
   - Drowsiness
   - Yawning
   - Head Tilt
   - Phone Usage
5. Update attention score
6. Trigger alerts if unsafe behavior is detected
7. Stream live monitoring feed to browser

---

- Sends WhatsApp alert messages automatically to the stakeholder’s phone when driver drowsiness is detected.

- Activates vehicle indicator lights when the driver’s eyes remain closed to warn nearby drivers and improve road safety.

# Project Structure

```bash
AI-Driver-Monitoring-System/
│
├── app.py
├── model.py
├── face_landmarker.task
├── hand_landmarker.task
├── alarm.wav
│
├── templates/
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── video_page.html
│   ├── about.html
│   ├── team.html
│   └── model_code.html
│
├── static/
│
└── README.md


this project is created in macbook if it is not working in windows then code or change  according to the windows
