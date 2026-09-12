"""
predict_image.py

Predicts the class for a single image file, using the trained model.
Useful for quick testing/demos without a webcam.

Usage:
    python predict_image.py path/to/image.jpg
"""

import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "mudra_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 128  # must match --img-size used in train_model.py


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    batch = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    preds = model.predict(batch, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]

    print(f"\nPredictions for '{image_path}':")
    for idx in top5_idx:
        print(f"  {class_names[idx]:<20s} {preds[idx]*100:5.1f}%")


if __name__ == "__main__":
    main()
