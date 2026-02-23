import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt

# ---------------- CONFIG ----------------
FPS = 30
BUFFER_SIZE = 300
LOW_CUT = 0.7
HIGH_CUT = 3.5

# ---------------- FILTER ----------------
def bandpass_filter(signal, fs):
    nyquist = 0.5 * fs
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ---------------- MEDIAPIPE ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)
green_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    bpm = 0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        points = [10, 338, 297, 332]
        coords = [(int(landmarks.landmark[p].x * w),
                   int(landmarks.landmark[p].y * h)) for p in points]

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

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            if len(green_buffer) == BUFFER_SIZE:
                signal = np.array(green_buffer)
                signal -= np.mean(signal)

                filtered = bandpass_filter(signal, FPS)

                fft = np.fft.rfft(filtered)
                freqs = np.fft.rfftfreq(len(filtered), 1/FPS)

                mask = (freqs >= LOW_CUT) & (freqs <= HIGH_CUT)
                freqs = freqs[mask]
                fft = np.abs(fft[mask])

                peak_freq = freqs[np.argmax(fft)]
                bpm = peak_freq * 60

    # Display BPM
    cv2.putText(frame,
                f"BPM: {int(bpm)}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3)

    cv2.imshow("rPPG Live Heart Rate Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        breaki

cap.release()
cv2.destroyAllWindows()