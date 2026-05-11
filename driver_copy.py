from flask import Flask, render_template, request, redirect, url_for, Response, session
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import subprocess
import time
import platform
import os
from datetime import datetime
import sqlite3
import pywhatkit
import threading

import serial

try:
    arduino = serial.Serial('/dev/cu.usbserial-1120', 9600)
    time.sleep(2)
    print("✅ Arduino connected")
except:
    arduino = None
    print("❌ Arduino not connected")

DATABASE = "driver_monitoring.db"

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

app = Flask(__name__)
app.secret_key = "secretkey"

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        login_date TEXT,
        login_time TEXT,
        logout_date TEXT,
        logout_time TEXT,
        drive_time TEXT,
        blink_count INTEGER,
        eye_closed_count INTEGER,
        yawn_count INTEGER,
        no_face_count INTEGER
    )
    """)

    conn.commit()
    conn.close()

# ================= USER STORAGE =================
users = {}

# ================= MODEL CONFIG =================
EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.55
EYE_CLOSED_TIME = 2
YAWN_TIME = 2
HEAD_TILT_THRESHOLD = 10
NO_FACE_TIME = 2
PHONE_TIME = 2
MIN_HAND_BOX = 25

# ================= GLOBAL VARIABLES =================
# ================= GLOBAL VARIABLES =================
blink_count = 0
closed_start = None
no_face_start = None
yawn_start = None
phone_start = None
attention_score = 100
alarm_active = False
eye_closed_count = 0
drive_time = 0
stakeholder_number = "+918660675730"  # change this
last_sent_time = 0 
motor_triggered = False
motor_start_time = None


# ================= EVENT COUNTERS =================
blink_count = 0
yawn_count = 0
no_face_count = 0
hand_distraction_count = 0
eye_closed_event = False
yawn_state = False



#=======Status=====
hand_distraction_event = False
blink_count_status = False
face_missing_event = False
hand_state = False
alert_sent = False

#============sms===========
def send_alert_message():

    global last_sent_time

    if time.time() - last_sent_time < 30:
        return

    try:
        message = "⚠ ALERT: Driver is drowsy. Eye closed events exceeded safe limit."

        # IMPORTANT FIXES
        pywhatkit.sendwhatmsg_instantly(
            stakeholder_number,
            message,
            wait_time=20,   # increase time
            tab_close=True,
            close_time=3    # allow sending time
        )

        print("✅ Message sent")

        last_sent_time = time.time()

    except Exception as e:
        print("❌ Message failed:", e)


# ✅ NEW FUNCTION
def send_async():
    threading.Thread(target=send_alert_message).start()


#==========motor ========
def turn_motor_on():
    global motor_triggered, motor_start_time

    if not motor_triggered:
        if arduino:
            print("MOTOR ON SENT")
            arduino.write(b'M')

        motor_triggered = True
        motor_start_time = time.time()


def turn_motor_off():
    global motor_triggered, motor_start_time

    if motor_triggered:
        if arduino:
            print("MOTOR OFF SENT")
            arduino.write(b'S')

        motor_triggered = False
        motor_start_time = None

# ================= SOUND CONTROL =================
def start_alarm():
    global alarm_process, alarm_active
    global alarm_active
    if not alarm_active:
        alarm_process = subprocess.Popen(["afplay", "alarm.wav"])
        alarm_active = True

def stop_alarm():
    global alarm_process, alarm_active
    global alarm_active
    if alarm_active and alarm_process:
        alarm_process.terminate()
        alarm_process = None
        alarm_active = False


# ================= MEDIAPIPE SETUP =================
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(face_options)

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

# ================= MATH FUNCTIONS =================
def euclidean(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def compute_EAR(landmarks,eye,w,h):
    pts=[(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in eye]
    A=euclidean(pts[1],pts[5])
    B=euclidean(pts[2],pts[4])
    C=euclidean(pts[0],pts[3])
    return (A+B)/(2*C)

def compute_MAR(landmarks,w,h):
    top=(int(landmarks[13].x*w),int(landmarks[13].y*h))
    bottom=(int(landmarks[14].x*w),int(landmarks[14].y*h))
    left=(int(landmarks[78].x*w),int(landmarks[78].y*h))
    right=(int(landmarks[308].x*w),int(landmarks[308].y*h))
    return euclidean(top,bottom)/euclidean(left,right)

def compute_head_tilt(landmarks,w,h):
    left_eye=(int(landmarks[33].x*w),int(landmarks[33].y*h))
    right_eye=(int(landmarks[263].x*w),int(landmarks[263].y*h))
    dx=right_eye[0]-left_eye[0]
    dy=right_eye[1]-left_eye[1]
    return math.degrees(math.atan2(dy,dx))



# ================= VIDEO STREAM =================
def generate_frames():


    global closed_start, no_face_start, yawn_start, phone_start, attention_score
    global blink_count, eye_closed_count, yawn_count, no_face_count, hand_distraction_count
    global blink_count_status, eye_closed_event, face_missing_event, hand_state
    global eye_closed_state, yawn_state, hand_state, face_missing_state
    global drive_time
    global motor_triggered, motor_start_time

    

    cap = cv2.VideoCapture(0)

    start_time = time.time()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        timestamp = int(time.time()*1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

        face_result = landmarker.detect_for_video(mp_image,timestamp)
        hand_result = hand_landmarker.detect_for_video(mp_image,timestamp)

        current_time = time.time()
        drive_time = int(current_time - start_time)

        critical=False
        status="ALERT"
        color=(0,255,0)
        phone_detected=False

        if face_result.face_landmarks:

            no_face_start=None
            face_missing_event = False
            landmarks=face_result.face_landmarks[0]
            

            ear=(compute_EAR(landmarks,LEFT_EYE,w,h)+
                 compute_EAR(landmarks,RIGHT_EYE,w,h))/2

            mar=compute_MAR(landmarks,w,h)
            tilt=compute_head_tilt(landmarks,w,h)

#======Left side metrics========
            cv2.putText(frame,f"EAR:{ear:.2f}",(30,170),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(frame,f"MAR:{mar:.2f}",(30,200),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(frame,f"Tilt:{tilt:.2f}",(30,230),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(frame,f"Driving Time:{drive_time}s",(30,260),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
#======Right side metrics=======
            cv2.putText(frame,f"Blink Count:{blink_count}",(w-300,170),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(frame,f"Eye Closed Count:{eye_closed_count}",(w-300,200),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(frame,f"Yawn Count:{yawn_count}",(w-300,230),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(frame,f"Hand Distraction Count:{hand_distraction_count}",(w-300,260),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)


            if hand_result.hand_landmarks:

                if phone_start is None:
                    phone_start = current_time

                elif current_time - phone_start > PHONE_TIME:

                    if not hand_state:
                        hand_distraction_count += 1
                        hand_state = True

                    status = "HAND DISTRACTION"
                    color = (255,0,255)
                    attention_score -= 3

            else:
                phone_start = None
                hand_state = False

            # Yawn
            # Yawn detection
            if mar > MAR_THRESHOLD:
                if not yawn_state:
                    yawn_count += 1
                    yawn_state = True

                if yawn_start is None:
                    yawn_start = current_time
                elif current_time - yawn_start > YAWN_TIME:
                    status = "YAWNING"
                    color = (0,165,255)
                    attention_score -= 2
            else:
                yawn_state = False
                yawn_start = None

            # Head tilt
            if abs(tilt)>HEAD_TILT_THRESHOLD:
                status="HEAD TILT"
                color=(255,0,0)
                attention_score-=1

            # Eye closed
            # Eye closed detection
            if ear < EAR_THRESHOLD:

                if not blink_count_status:
                    blink_count += 1
                    blink_count_status = True

                if closed_start is None:
                    closed_start = current_time

                elif current_time - closed_start > EYE_CLOSED_TIME:

                    if not eye_closed_event:
                        eye_closed_count += 1
                        eye_closed_event = True

                        if eye_closed_count > 5:
                            send_alert_message()

                        if eye_closed_count >= 7:
                            turn_motor_on()
                        

                    status = "EYE CLOSED"
                    color = (0,0,255)
                    attention_score -= 4
                    critical = True

            else:
                closed_start = None
                blink_count_status = False
                eye_closed_event = False

        else:
            if no_face_start is None:
                no_face_start = current_time

            elif current_time - no_face_start > NO_FACE_TIME:

                if not face_missing_event:
                    no_face_count += 1
                    face_missing_event = True

                status = "NO DRIVER DETECTED"
                color = (0,0,255)
                attention_score -= 6
                critical = True

        attention_score=max(0,min(100,attention_score))

        if not critical and attention_score < 100:
            attention_score += 0.5

        if critical:
            red_overlay = np.zeros_like(frame)
            red_overlay[:] = (0,0,255)
            frame = cv2.addWeighted(red_overlay,0.7,frame,0.3,0)
            start_alarm()

            # 🔴 LED ON
            if arduino:
                arduino.write(b'1')

        else:
            stop_alarm()

            # ⚫ LED OFF
            if arduino:
                arduino.write(b'0')

        cv2.putText(frame,status,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)

        bar_width=int((attention_score/100)*300)

        cv2.rectangle(frame,(30,110),(330,140),(40,40,40),-1)
        cv2.rectangle(frame,(30,110),(30+bar_width,140),(0,255,0),-1)

        cv2.putText(frame,f"Attention:{int(attention_score)}%",
                    (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        
        # AUTO TURN OFF MOTOR AFTER 2 SECONDS
        if motor_triggered and motor_start_time:
            if time.time() - motor_start_time > 2:
                turn_motor_off()

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

# ================= ROUTES =================

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET","POST"])
def signup():

    message = ""

    if request.method == "POST":

        name = request.form["name"]
        password = request.form["password"]
        repassword = request.form["repassword"]

        if name in users:
            message = "Username already exists!"

        elif password != repassword:
            message = "Passwords do not match!"

        else:
            users[name] = password
            return redirect(url_for("login"))

    return render_template("signup.html", message=message)

@app.route("/login", methods=["GET","POST"])
def login():

    global login_time, login_date

    message = ""

    if request.method == "POST":

        name = request.form["name"]
        password = request.form["password"]

        if name in users and users[name] == password:

            session["user"] = name

            now = datetime.now()
            login_time = now.strftime("%H:%M:%S")
            login_date = now.strftime("%Y-%m-%d")

            return redirect(url_for("dashboard"))

        else:
            message = "Invalid username or password"

    return render_template("login.html", message=message)


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route('/model_code')
def model_code():
    # Read your model code from a file
    with open('model.py', 'r') as f:
        code = f.read()
    return render_template('model_code.html', model_code=code)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home_back():
    return render_template('dashboard.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route("/video")
def video():
    return render_template("video_page.html")

@app.route("/live_feed")
def live_feed():
    return render_template("live_feed.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route("/stop")
def stop():
    global running
    running = False
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():

    global blink_count, yawn_count, no_face_count, hand_distraction_count,eye_closed_count,drive_time
    global login_time, login_date

    if "user" in session:

        username = session["user"]

        now = datetime.now()
        logout_time = now.strftime("%H:%M:%S")
        logout_date = now.strftime("%Y-%m-%d")

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO logs (
            username,
            login_date,
            login_time,
            logout_date,
            logout_time,
            drive_time,
            blink_count,
            eye_closed_count,
            yawn_count,
            no_face_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            login_date,
            login_time,
            logout_date,
            logout_time,
            drive_time,
            blink_count,
            eye_closed_count,
            yawn_count,
            no_face_count,
        ))

        conn.commit()
        conn.close()
    # reset counters
    blink_count = 0
    eye_closed_count=0
    yawn_count = 0
    no_face_count = 0
    drive_time=0


    return redirect("/")

@app.route("/logs")
def logs():

    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM logs
        WHERE username = ?
        ORDER BY logout_date DESC, logout_time DESC
    """, (username,))

    data = cursor.fetchall()

    conn.close()

    return render_template("logs.html", data=data)

if __name__ == "__main__":
    init_db()
    app.run(debug=True, use_reloader=False)