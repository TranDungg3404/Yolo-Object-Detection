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

# Load YOLO
yolo_model = YOLO("yolov8n.pt")

# Bộ nhớ đệm để tránh nhận diện lặp lại
last_detected = {}
CACHE_TIMEOUT = 10  # Giây, thời gian lưu trữ nhận diện

# Tạo thư mục uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    encoding = np.array(encoding_tuple)
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
    name = "Unknown"
    if True in matches:
        matched_idxs = [i for i, b in enumerate(matches) if b]
        counts = {known_names[i]: face_recognition.face_distance(known_encodings, encoding)[i] for i in matched_idxs}
        name = min(counts, key=counts.get)  # Chọn tên có khoảng cách nhỏ nhất
    return name

# Endpoint nhận ảnh từ ESP32-CAM
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Kiểm tra token xác thực
        if request.headers.get('Authorization') != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401

        # Nhận dữ liệu ảnh
        file_bytes = request.data
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        # Dò khuôn mặt bằng YOLO
        results = yolo_model(frame, conf=0.5)[0]
        face_locations = []
        for box in results.boxes:
            if box.cls == 0:  # Class 0 là "person" trong YOLOv8
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1))  # top, right, bottom, left

        # Nhận diện khuôn mặt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        names = []
        confidences = []

        for encoding in encodings:
            current_time = time.time()
            encoding_tuple = tuple(encoding)  # Chuyển encoding thành tuple để cache
            name = recognize_face(encoding_tuple)

            # Kiểm tra bộ đệm
            if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                name = last_detected[name]["name"]
                confidence = last_detected[name]["confidence"]
            else:
                distances = face_recognition.face_distance(known_encodings, encoding)
                confidence = 1 - min(distances) if distances.size > 0 else 0
                last_detected[name] = {"name": name, "time": current_time, "confidence": confidence}

            names.append(name)
            confidences.append(confidence)

            # Lưu thông tin điểm danh và ảnh
            if name != "Unknown":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                info = f"ID: {name}_001, Confidence: {confidence:.2f}"
                data = {"Name": name, "Time": timestamp, "Confidence": confidence}
                df = pd.DataFrame([data])
                df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE))
                
                # Lưu ảnh
                img_path = os.path.join(UPLOAD_DIR, f"{name}_{timestamp.replace(':', '-')}.jpg")
                cv2.imwrite(img_path, frame)

                # Gửi thông tin về ESP32-CAM
                send_to_esp32(name, info)

        return jsonify({"names": names, "confidences": confidences})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server error"}), 500

# Endpoint lấy báo cáo điểm danh
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

# Xử lý luồng video từ ESP32-CAM
def process_stream():
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    if not cap.isOpened():
        print("Error: Cannot open ESP32-CAM stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Dò khuôn mặt bằng YOLO
        results = yolo_model(frame, conf=0.5)[0]
        face_locations = []
        for box in results.boxes:
            if box.cls == 0:  # Class 0 là "person" trong YOLOv8
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1))  # top, right, bottom, left

        # Nhận diện khuôn mặt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        names = []
        confidences = []

        for encoding in encodings:
            current_time = time.time()
            encoding_tuple = tuple(encoding)  # Chuyển encoding thành tuple để cache
            name = recognize_face(encoding_tuple)

            # Kiểm tra bộ đệm
            if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                name = last_detected[name]["name"]
                confidence = last_detected[name]["confidence"]
            else:
                distances = face_recognition.face_distance(known_encodings, encoding)
                confidence = 1 - min(distances) if distances.size > 0 else 0
                last_detected[name] = {"name": name, "time": current_time, "confidence": confidence}

            names.append(name)
            confidences.append(confidence)

            # Lưu thông tin điểm danh và ảnh
            if name != "Unknown":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                info = f"ID: {name}_001, Confidence: {confidence:.2f}"
                data = {"Name": name, "Time": timestamp, "Confidence": confidence}
                df = pd.DataFrame([data])
                df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE))
                
                # Lưu ảnh
                img_path = os.path.join(UPLOAD_DIR, f"{name}_{timestamp.replace(':', '-')}.jpg")
                cv2.imwrite(img_path, frame)

                # Gửi thông tin về ESP32-CAM
                send_to_esp32(name, info)

        # Hiển thị video (tùy chọn, để debug)
        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('ESP32-CAM Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Định nghĩa hàm generator stream video
def generate_stream():
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    if not cap.isOpened():
        print("Cannot open ESP32-CAM stream")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Mã hóa frame sang JPEG
        ret2, jpeg = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = jpeg.tobytes()
        # Yield theo định dạng multipart để trình duyệt nhận dạng stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# Route để stream video MJPEG
@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route trang chủ với HTML nhúng stream video
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
        <img src="{{ url_for('video_feed') }}" width="640" height="480" />
    </body>
    </html>
    '''
    return render_template_string(html)


if __name__ == '__main__':
    # Khởi động luồng xử lý video
    threading.Thread(target=process_stream, daemon=True).start()
    # Khởi động Flask server
    app.run(host='0.0.0.0', port=5000)