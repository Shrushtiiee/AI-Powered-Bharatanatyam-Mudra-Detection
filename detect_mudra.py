
import os
import json
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# -----------------------------
# CONFIGURATION
# -----------------------------

MODEL_PATH = "mudra_model.keras"
CLASS_NAMES_PATH = "class_names.json"

CONFIDENCE_THRESHOLD = 0.70

SMOOTHING_FRAMES = 10

BOX_MARGIN = 0.35


# -----------------------------
# LOAD MODEL
# -----------------------------

def load_model_and_classes():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train_model.py first."
        )

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class names not found: "
            f"{CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    # Automatically use the trained model's input size.
    img_size = model.input_shape[1]

    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            "Class names do not match the model output.\n"
            f"Model classes: {model.output_shape[-1]}\n"
            f"JSON classes: {len(class_names)}\n"
            "Retrain the model and use its generated "
            "class_names.json."
        )

    print("\nModel loaded successfully.")

    print("Input image size:", img_size)

    print("Mudra classes:")

    for index, name in enumerate(class_names):
        print(index, ":", name)

    return model, class_names, img_size


# -----------------------------
# HAND BOUNDING BOX
# -----------------------------

def get_hand_bbox(
    hand_landmarks,
    frame_width,
    frame_height
):

    xs = [
        landmark.x
        for landmark in hand_landmarks.landmark
    ]

    ys = [
        landmark.y
        for landmark in hand_landmarks.landmark
    ]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    # Add padding around the hand
    x_min -= width * BOX_MARGIN
    x_max += width * BOX_MARGIN

    y_min -= height * BOX_MARGIN
    y_max += height * BOX_MARGIN

    # Convert normalized coordinates to pixels
    x_min = int(x_min * frame_width)
    x_max = int(x_max * frame_width)

    y_min = int(y_min * frame_height)
    y_max = int(y_max * frame_height)

    # Keep coordinates inside the frame
    x_min = max(0, x_min)
    y_min = max(0, y_min)

    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)

    return x_min, y_min, x_max, y_max


# -----------------------------
# PREPROCESS HAND IMAGE
# -----------------------------

def preprocess_hand(crop, img_size):

    if crop is None or crop.size == 0:
        return None

    # Keep the original RGB color order expected
    # by the trained model.
    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    crop_resized = cv2.resize(
        crop_rgb,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    # The model includes MobileNetV2 preprocessing.
    # Therefore, supply RGB pixels in the 0-255 range.
    batch = np.expand_dims(
        crop_resized.astype(np.float32),
        axis=0
    )

    return batch


# -----------------------------
# MAIN WEBCAM FUNCTION
# -----------------------------

def main():

    model, class_names, img_size = (
        load_model_and_classes()
    )

    # MediaPipe Hands
    mp_hands = mp.solutions.hands

    mp_drawing = mp.solutions.drawing_utils

    # Try webcam index 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        cap.release()

        # Try webcam index 1
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():

        print("ERROR: Webcam could not be opened.")

        return

    # Set webcam resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # Store recent predictions
    prediction_history = deque(
        maxlen=SMOOTHING_FRAMES
    )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror webcam view
            frame = cv2.flip(frame, 1)

            frame_height, frame_width = (
                frame.shape[:2]
            )

            # MediaPipe requires RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb_frame.flags.writeable = False

            results = hands.process(rgb_frame)

            rgb_frame.flags.writeable = True

            if results.multi_hand_landmarks:

                hand_landmarks = (
                    results.multi_hand_landmarks[0]
                )

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Find hand bounding box
                x_min, y_min, x_max, y_max = (
                    get_hand_bbox(
                        hand_landmarks,
                        frame_width,
                        frame_height
                    )
                )

                crop = frame[
                    y_min:y_max,
                    x_min:x_max
                ]

                batch = preprocess_hand(
                    crop,
                    img_size
                )

                if batch is not None:

                    predictions = model.predict(
                        batch,
                        verbose=0
                    )[0]

                    # Smooth prediction probabilities
                    prediction_history.append(
                        predictions
                    )

                    avg_predictions = np.mean(
                        prediction_history,
                        axis=0
                    )

                    best_index = int(
                        np.argmax(avg_predictions)
                    )

                    confidence = float(
                        avg_predictions[best_index]
                    )

                    label = class_names[best_index]

                    # Determine displayed prediction
                    if confidence >= CONFIDENCE_THRESHOLD:

                        display_text = (
                            f"{label}: "
                            f"{confidence * 100:.1f}%"
                        )

                        color = (0, 255, 0)

                    else:

                        display_text = (
                            f"Uncertain: "
                            f"{confidence * 100:.1f}%"
                        )

                        color = (0, 165, 255)

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        color,
                        2
                    )

                    # Draw prediction background
                    text_y = max(
                        30,
                        y_min - 12
                    )

                    cv2.putText(
                        frame,
                        display_text,
                        (x_min, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA
                    )

            else:

                # Clear old predictions when no hand is visible
                prediction_history.clear()

                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # Instructions
            cv2.putText(
                frame,
                "Press Q to quit",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Display webcam
            cv2.imshow(
                "Bharatanatyam Mudra Detection",
                frame
            )

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()