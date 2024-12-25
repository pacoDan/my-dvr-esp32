import cv2
import numpy as np
import datetime
import os
from flask import Flask, Response

app = Flask(__name__)

# Configuración
ESP32_CAM_URL = "http://192.168.0.164:81/stream"
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Función para agregar marca de tiempo
def add_timestamp(frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# Detección de movimiento y grabación
def detect_motion_and_record():
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    last_frame = None
    recording = False
    video_writer = None
    start_time = datetime.datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Agregar marca de tiempo
        frame = add_timestamp(frame)

        # Detección de movimiento
        if last_frame is not None:
            diff = cv2.absdiff(last_frame, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                if not recording:
                    # Iniciar grabación
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = os.path.join(VIDEO_DIR, f"recording_{timestamp}.avi")
                    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))
                    recording = True
                    print(f"Recording started: {video_path}")

        if recording:
            video_writer.write(frame)

        last_frame = frame

        # Detener grabación después de 10 horas
        if recording and (datetime.datetime.now() - start_time).seconds > 36000:  # 10 horas
            recording = False
            video_writer.release()
            print("Recording stopped after 10 hours.")

    cap.release()
    if recording:
        video_writer.release()

@app.route('/start', methods=['GET'])
def start_recording():
    detect_motion_and_record()
    return "Recording started."

@app.route('/stop', methods=['GET'])
def stop_recording():
    # Implementar lógica para detener la grabación
    return "Recording stopped."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)