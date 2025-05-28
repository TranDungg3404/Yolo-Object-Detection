import cv2
import face_recognition
import numpy as np
import pickle
from ultralytics import YOLO
from flask import Flask, request, Response, jsonify, render_template_string, send_file
import datetime
import os
import requests
import threading
import time
import sqlite3
from functools import lru_cache
from threading import Lock
import pandas as pd
import io

app = Flask(__name__)

# Cấu hình
ESP32_CAM_URL = "http://192.168.1.105/stream"
INFO_URL = "http://192.168.1.105/info"
UPLOAD_DIR = "uploads"
API_TOKEN = "123456"
DB_FILE = "attendance.db"

# Tạo thư mục uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Kết nối SQLite
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}. Attempting to recreate database...")
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            time TEXT,
            confidence REAL,
            employee_id TEXT,
            dob TEXT,
            position TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT,
            dob TEXT,
            position TEXT
        )''')
        return conn

# Khởi tạo bảng nếu chưa có
with get_db_connection() as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        time TEXT,
        confidence REAL,
        employee_id TEXT,
        dob TEXT,
        position TEXT
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS employees (
        employee_id TEXT PRIMARY KEY,
        name TEXT,
        dob TEXT,
        position TEXT
    )''')

# Load thông tin nhân viên từ CSV vào SQLite (chạy một lần hoặc nếu chưa có)
if os.path.exists("employee_info.csv"):
    import pandas as pd
    df = pd.read_csv("employee_info.csv")
    with get_db_connection() as conn:
        for _, row in df.iterrows():
            conn.execute("INSERT OR REPLACE INTO employees (employee_id, name, dob, position) VALUES (?, ?, ?, ?)",
                         (row['ID'], row['Name'], row['DOB'], row['Position']))

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

# Load YOLO
yolo_model = YOLO("yolov8n.pt")

# Bộ nhớ đệm tránh nhận diện lặp lại
last_detected = {}
CACHE_TIMEOUT = 10
output_frame = None
frame_lock = Lock()
attendance_lock = Lock()
last_detected_lock = Lock()

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

def send_to_esp32(name, info):
    try:
        payload = {"name": name, "info": info}
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(INFO_URL, data=payload, headers=headers, timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"Failed to send info to ESP32-CAM: {e}")
        return False

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
        face_locations = [(int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), int(box.xyxy[0][0]))
                          for box in results.boxes if int(box.cls[0]) == 0]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        names = []

        for encoding in encodings:
            encoding_tuple = tuple(encoding)
            name = recognize_face(encoding_tuple)
            current_time = time.time()

            if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                continue

            confidence = 0
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, encoding)
                confidence = 1 - min(distances)
            with last_detected_lock:
                last_detected[name] = {"time": current_time}

            names.append(name)

            if name != "Unknown":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with get_db_connection() as conn:
                    emp = conn.execute("SELECT * FROM employees WHERE LOWER(name) = LOWER(?)", (name,)).fetchone()
                    if emp:
                        conn.execute("INSERT INTO attendance (name, time, confidence, employee_id, dob, position) VALUES (?, ?, ?, ?, ?, ?)",
                                     (name, timestamp, confidence, emp['employee_id'], emp['dob'], emp['position']))
                        conn.commit()
                        info = f"ID: {emp['employee_id']}, DOB: {emp['dob']}, Position: {emp['position']}, Confidence: {confidence:.2f}"
                        send_to_esp32(name, info)
                img_path = os.path.join(UPLOAD_DIR, f"{name}_{timestamp.replace(':', '-')}.jpg")
                cv2.imwrite(img_path, frame)

        return jsonify({"names": names})

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": "Server error"}), 500

@app.route('/report')
def get_report():
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM attendance ORDER BY time DESC LIMIT 100").fetchall()
            return jsonify([dict(row) for row in rows])
    except Exception as e:
        print(f"Report error: {e}")
        return jsonify({"error": "Server error"}), 500

@app.route('/export', methods=['GET'])
def export_to_excel():
    try:
        # Lấy dữ liệu từ bảng attendance
        with get_db_connection() as conn:
            cursor = conn.execute("SELECT * FROM attendance ORDER BY time DESC")
            rows = cursor.fetchall()
            # Chuyển dữ liệu thành danh sách dictionary
            data = [dict(row) for row in rows]

        if not data:
            return jsonify({"error": "No attendance data to export"}), 404

        # Tạo DataFrame từ dữ liệu
        df = pd.DataFrame(data)

        # Tạo buffer để lưu file Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
        output.seek(0)

        # Tạo tên file với timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_export_{timestamp}.xlsx"

        # Trả về file Excel để tải về
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({"error": "Server error while exporting"}), 500

def process_stream():
    global output_frame, frame_lock

    cap = cv2.VideoCapture(ESP32_CAM_URL)
    if not cap.isOpened():
        print("Error: Cannot open ESP32-CAM stream. Retrying in 5 seconds...")
        time.sleep(5)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Stream disconnected. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            encoding_tuple = tuple(encoding)
            name = recognize_face(encoding_tuple)

            # Lấy thông tin nhân viên từ database
            with get_db_connection() as conn:
                emp = conn.execute("SELECT * FROM employees WHERE name = ?", (name,)).fetchone()

            # Ghi thông tin điểm danh vào database nếu nhận diện thành công
            if name != "Unknown" and emp:
                current_time = time.time()
                with last_detected_lock:
                    if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                        continue

                    confidence = 0
                    if known_encodings:
                        distances = face_recognition.face_distance(known_encodings, encoding)
                        confidence = 1 - min(distances)

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute(
                        "INSERT INTO attendance (name, time, confidence, employee_id, dob, position) VALUES (?, ?, ?, ?, ?, ?)",
                        (name, timestamp, confidence, emp['employee_id'], emp['dob'], emp['position'])
                    )
                    conn.commit()
                    print(f"Đã ghi vào database: {name}, {timestamp}, {confidence}")  # Log debug
                    with last_detected_lock:
                        last_detected[name] = {"time": current_time}

                    # Lưu ảnh (tùy chọn)
                    img_path = os.path.join(UPLOAD_DIR, f"{name}_{timestamp.replace(':', '-')}.jpg")
                    cv2.imwrite(img_path, frame)

                    info = f"ID: {emp['employee_id']}, DOB: {emp['dob']}, Position: {emp['position']}, Confidence: {confidence:.2f}"
                    send_to_esp32(name, info)

            # Vẽ khung và nhãn
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = name
            if emp:
                label += f" ({emp['employee_id']})"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with frame_lock:
            output_frame = frame.copy()

        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    def generate():
        global output_frame, frame_lock
        while True:
            with frame_lock:
                if output_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', output_frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head><title>ESP32-CAM Stream</title></head>
    <body>
    <h1>Live Stream from ESP32-CAM</h1>
    <h2 id="recognized-name">Đang nhận diện...</h2>
    <p id="employee-info"></p>
    <img src="{{ url_for('video_feed') }}" width="640" height="480" />
    <br><br>
    <a href="{{ url_for('export_to_excel') }}"><button>Export to Excel</button></a>
    <script>
    setInterval(async () => {
        try {
            const res = await fetch('/report');
            if (!res.ok) {
                console.error("Error fetching report:", res.status, res.statusText);
                return;
            }
            const data = await res.json();
            console.log("Report data:", data);
            if (Array.isArray(data) && data.length > 0) {
                const latest = data[0];
                document.getElementById('recognized-name').innerText = "Người nhận diện gần nhất: " + latest.name;
                document.getElementById('employee-info').innerText = 
                    `ID: ${latest.employee_id || 'N/A'}, DOB: ${latest.dob || 'N/A'}, Position: ${latest.position || 'N/A'}, Confidence: ${latest.confidence?.toFixed(2) || 'N/A'}`;
            }
        } catch (error) {
            console.error("Fetch error:", error);
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
