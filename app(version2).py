import cv2
import face_recognition
import numpy as np
import pickle
from ultralytics import YOLO
from flask import Flask, request, Response, jsonify
import datetime
import pandas as pd
import os
import requests
import threading
import time
from functools import lru_cache
from flask import render_template_string
from threading import Lock

app = Flask(__name__)

# Cấu hình
ESP32_CAM_URL = "http://192.168.1.105/stream"
INFO_URL = "http://192.168.1.105/info"
ATTENDANCE_FILE = "attendance.csv"
UPLOAD_DIR = "uploads"
API_TOKEN = "123456"  # Token cho xác thực API

# Load encodings
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
except FileNotFoundError:
    print("Error: encodings.pickle not found")
    known_encodings = []
    known_names = []

# Load thông tin nhân viên
employee_info_dict = {}
if os.path.exists("employee_info.csv"):
    employee_df = pd.read_csv("employee_info.csv")
    employee_info_dict = employee_df.set_index("Name").to_dict(orient="index")

# Load YOLO
yolo_model = YOLO("yolov8n.pt")

# Bộ nhớ đệm để tránh nhận diện lặp lại
last_detected = {}
CACHE_TIMEOUT = 10  # Giây, thời gian lưu trữ nhận diện

# Tạo thư mục uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Biến lưu khung hình cho stream
output_frame = None
frame_lock = Lock()
attendance_lock = Lock()

# Hàm gửi thông tin về ESP32-CAM
def send_to_esp32(name, info):
    try:
        payload = {"name": name, "info": info}
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(INFO_URL, data=payload, headers=headers, timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"Failed to send info to ESP32-CAM: {e}")
        return False

# Hàm nhận diện khuôn mặt
@lru_cache(maxsize=100)
def recognize_face(encoding_tuple):
    if not known_encodings:
        return "Unknown"
    encoding = np.array(encoding_tuple)
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
    name = "Unknown"
    if True in matches:
        matched_idxs = [i for i, b in enumerate(matches) if b]
        counts = {known_names[i]: face_recognition.face_distance(known_encodings, encoding)[i] for i in matched_idxs}
        name = min(counts, key=counts.get)
    return name

# Endpoint nhận ảnh từ ESP32-CAM
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if request.headers.get('Authorization') != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401

        file_bytes = request.data
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        results = yolo_model(frame, conf=0.5)[0]
        face_locations = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        names = []
        confidences = []

        for encoding in encodings:
            current_time = time.time()
            encoding_tuple = tuple(encoding)
            name = recognize_face(encoding_tuple)

            if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                name = last_detected[name]["name"]
                confidence = last_detected[name]["confidence"]
            else:
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    confidence = 1 - min(distances)
                else:
                    confidence = 0
                last_detected[name] = {"name": name, "time": current_time, "confidence": confidence}

            names.append(name)
            confidences.append(confidence)

            if name != "Unknown":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                emp = employee_info_dict.get(name, {})
                info = f"ID: {emp.get('ID', 'N/A')}, DOB: {emp.get('DOB', '')}, Position: {emp.get('Position', '')}, Confidence: {confidence:.2f}"
                data = {"Name": name, "Time": timestamp, "Confidence": confidence, "ID": emp.get('ID', ''), "DOB": emp.get('DOB', ''), "Position": emp.get('Position', '')}
                with attendance_lock:
                    df = pd.DataFrame([data])
                    df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE))

                img_path = os.path.join(UPLOAD_DIR, f"{name}_{timestamp.replace(':', '-')}.jpg")
                cv2.imwrite(img_path, frame)

                send_to_esp32(name, info)

        for (top, right, bottom, left), name in zip(face_locations, names):
            emp = employee_info_dict.get(name, {})
            label = f"{name} | ID: {emp.get('ID', '')} | DOB: {emp.get('DOB', '')} | Position: {emp.get('Position', '')}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return jsonify({"names": names, "confidences": confidences})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server error"}), 500

@app.route('/report', methods=['GET'])
def get_report():
    try:
        if not os.path.exists(ATTENDANCE_FILE):
            return jsonify({"error": "No attendance data found"}), 404
        df = pd.read_csv(ATTENDANCE_FILE)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server error"}), 500

# Luồng xử lý video cập nhật frame toàn cục
def process_stream():
    global output_frame, frame_lock
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    if not cap.isOpened():
        print("Error: Cannot open ESP32-CAM stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = yolo_model(frame, conf=0.5)[0]
        face_locations = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        names = []
        confidences = []

        for encoding in encodings:
            current_time = time.time()
            encoding_tuple = tuple(encoding)
            name = recognize_face(encoding_tuple)

            if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                name = last_detected[name]["name"]
                confidence = last_detected[name]["confidence"]
            else:
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    confidence = 1 - min(distances)
                else:
                    confidence = 0
                last_detected[name] = {"name": name, "time": current_time, "confidence": confidence}

            names.append(name)
            confidences.append(confidence)

            if name != "Unknown":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                emp = employee_info_dict.get(name, {})
                info = f"ID: {emp.get('ID', 'N/A')}, DOB: {emp.get('DOB', '')}, Position: {emp.get('Position', '')}, Confidence: {confidence:.2f}"
                data = {"Name": name, "Time": timestamp, "Confidence": confidence, "ID": emp.get('ID', ''), "DOB": emp.get('DOB', ''), "Position": emp.get('Position', '')}
                with attendance_lock:
                    df = pd.DataFrame([data])
                    df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE))
                img_path = os.path.join(UPLOAD_DIR, f"{name}_{timestamp.replace(':', '-')}.jpg")
                cv2.imwrite(img_path, frame)
                send_to_esp32(name, info)

        for (top, right, bottom, left), name in zip(face_locations, names):
            emp = employee_info_dict.get(name, {})
            label = f"{name} | ID: {emp.get('ID', '')} | DOB: {emp.get('DOB', '')} | Position: {emp.get('Position', '')}"
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        with frame_lock:
            output_frame = frame.copy()
        time.sleep(0.05)

# Endpoint stream video MJPEG
@app.route('/video_feed')
def video_feed():
    def generate():
        global output_frame, frame_lock
        try:
            while True:
                with frame_lock:
                    if output_frame is None:
                        continue
                    ret, jpeg = cv2.imencode('.jpg', output_frame)
                    if not ret:
                        continue
                    frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.05)
        except GeneratorExit:
            print("Client disconnected from video feed.")
        except Exception as e:
            print(f"Stream error: {e}")
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESP32-CAM Stream</title>
    </head>
    <body>
        <h1>Live Stream from ESP32-CAM</h1>
        <h2 id="recognized-name">Đang nhận diện...</h2>
        <img src="{{ url_for('video_feed') }}" width="640" height="480" />
        <script>
        setInterval(async () => {
            const res = await fetch('/report');
            const data = await res.json();
            if (Array.isArray(data) && data.length > 0) {
                document.getElementById('recognized-name').innerText = "Người nhận diện gần nhất: " + data[data.length - 1].Name;
            }
        }, 2000);
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

if __name__ == '__main__':
    threading.Thread(target=process_stream, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
