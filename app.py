import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from scipy.signal import butter, filtfilt

# ---------------- CONFIG ----------------
FPS = 30
BUFFER_SIZE = 300  # 10 sec buffer
LOW_CUT = 0.7
HIGH_CUT = 3.5

# ---------------- FILTER ----------------
def bandpass_filter(signal, fs):
    nyquist = 0.5 * fs
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(layout="wide")
st.title("Real-Time rPPG Heart Rate Monitor")

frame_placeholder = st.empty()
bpm_placeholder = st.empty()
chart_placeholder = st.empty()

# ---------------- MEDIAPIPE ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)
green_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        points = [10, 338, 297, 332]
        coords = []

        for p in points:
            x = int(landmarks.landmark[p].x * w)
            y = int(landmarks.landmark[p].y * h)
            coords.append((x, y))

        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size > 0:
            green_mean = np.mean(roi[:, :, 1])
            green_buffer.append(green_mean)

            if len(green_buffer) > BUFFER_SIZE:
                green_buffer.pop(0)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if len(green_buffer) == BUFFER_SIZE:
                signal = np.array(green_buffer)
                signal = signal - np.mean(signal)

                filtered = bandpass_filter(signal, FPS)

                fft = np.fft.rfft(filtered)
                freqs = np.fft.rfftfreq(len(filtered), 1 / FPS)

                mask = (freqs >= LOW_CUT) & (freqs <= HIGH_CUT)
                freqs = freqs[mask]
                fft = np.abs(fft[mask])

                peak_freq = freqs[np.argmax(fft)]
                bpm = peak_freq * 60

                bpm_placeholder.metric("Heart Rate (BPM)", f"{bpm:.1f}")
                chart_placeholder.line_chart(filtered[-150:])

    frame_placeholder.image(frame, channels="BGR")

cap.release()