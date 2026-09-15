import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd


MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "face_landmarker.task"


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def play_alarm():
    frequency = 1000
    duration = 0.5
    sample_rate = 44100

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.3 * np.sin(2 * np.pi * frequency * t)

    sd.play(tone, sample_rate)
    sd.wait()


def get_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face landmark model not found: {MODEL_PATH}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
    )

    return FaceLandmarker.create_from_options(options)


def main():
    ear_threshold = 0.25
    drowsiness_threshold = 2.0
    eyes_closed_start = None
    drowsiness_detected = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Make sure a camera is connected.")

    left_eye_indices = set()
    for connection in mp.solutions.face_mesh.FACEMESH_LEFT_EYE:
        left_eye_indices.update(connection)

    right_eye_indices = set()
    for connection in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE:
        right_eye_indices.update(connection)

    with get_landmarker() as landmarker:
        print("Face Landmarker ready!")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                h, w, _ = frame.shape

                p1 = landmarks[33]
                p2 = landmarks[160]
                p3 = landmarks[158]
                p4 = landmarks[133]
                p5 = landmarks[153]
                p6 = landmarks[144]

                vertical_1 = distance(p2, p6)
                vertical_2 = distance(p3, p5)
                horizontal = distance(p1, p4)
                ear = (vertical_1 + vertical_2) / (2 * horizontal)

                if ear < ear_threshold:
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()

                    closed_duration = time.time() - eyes_closed_start

                    if closed_duration >= drowsiness_threshold and not drowsiness_detected:
                        drowsiness_detected = True
                        print("DROWSINESS DETECTED!")
                        play_alarm()
                else:
                    eyes_closed_start = None
                    drowsiness_detected = False

                if drowsiness_detected:
                    status = "DROWSINESS DETECTED!"
                elif ear < ear_threshold:
                    status = "EYES CLOSED"
                else:
                    status = "EYES OPEN"

                for i, landmark in enumerate(landmarks):
                    if i in left_eye_indices or i in right_eye_indices:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                if drowsiness_detected:
                    triangle = np.array([[30, 45], [18, 68], [42, 68]])
                    cv2.drawContours(frame, [triangle], -1, (0, 0, 255), 3)

                status_x = 55 if drowsiness_detected else 30
                status_color = (0, 0, 255) if drowsiness_detected else (0, 255, 0)

                cv2.putText(frame, status, (status_x, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO FACE DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("VirgilDrive", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
