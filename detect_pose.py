"""
detect_pose.py

Real-time POSTURE/ADAVU detection: classifies the full webcam frame directly
(no hand-cropping, since these are full-body poses), with an optional
MediaPipe Pose skeleton overlay for visualization. Use this if your model
was trained on full-body posture images (e.g. new_Dataset).

For hand mudra classification instead, use detect_mudra.py.

Usage:
    python detect_pose.py

Press 'q' to quit.
"""

import json

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

MODEL_PATH = "mudra_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 160          # must match --img-size used in train_model.py
CONFIDENCE_THRESHOLD = 0.5

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            batch = np.expand_dims(frame_rgb.astype(np.float32), axis=0)

            preds = model.predict(batch, verbose=0)[0]
            best_idx = int(np.argmax(preds))
            confidence = float(preds[best_idx])
            label = class_names[best_idx]

            if confidence < CONFIDENCE_THRESHOLD:
                text = f"Uncertain ({confidence*100:.0f}%)"
                color = (0, 0, 255)
            else:
                text = f"{label} ({confidence*100:.0f}%)"
                color = (0, 255, 0)

            cv2.putText(
                frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
            )
            cv2.putText(
                frame, "Press 'q' to quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

            cv2.imshow("Bharatanatyam Posture Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
