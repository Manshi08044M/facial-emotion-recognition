"""Webcam facial emotion detection using the trained Keras model."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "emotion_model.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "emotion_labels.json"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live webcam emotion detection.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence required before showing an emotion label.",
    )
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=7,
        help="Number of recent predictions used to smooth the displayed result.",
    )
    return parser.parse_args()


def load_labels(labels_path: Path, dataset_dir: Path) -> list[str]:
    if labels_path.is_file():
        return json.loads(labels_path.read_text(encoding="utf-8"))

    train_dir = dataset_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Labels file not found and dataset folder is missing: {train_dir}"
        )
    return sorted(path.name for path in train_dir.iterdir() if path.is_dir())


def model_input_config(model: tf.keras.Model) -> tuple[tuple[int, int], int]:
    _, height, width, channels = model.input_shape
    if height is None or width is None or channels is None:
        raise ValueError(f"Model has unsupported input shape: {model.input_shape}")
    return (int(width), int(height)), int(channels)


def preprocess_face(
    frame: np.ndarray,
    gray_frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    channels: int,
) -> np.ndarray:
    x, y, w, h = face_box
    if channels == 1:
        face = gray_frame[y : y + h, x : x + w]
        face = cv2.resize(face, image_size, interpolation=cv2.INTER_AREA)
        face = np.expand_dims(face, axis=-1)
    else:
        face = frame[y : y + h, x : x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, image_size, interpolation=cv2.INTER_AREA)
    face = face.astype("float32")
    return np.expand_dims(face, axis=0)


def average_prediction(prediction_window: deque[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(list(prediction_window), axis=0), axis=0)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    labels = load_labels(args.labels_path, args.dataset_dir)
    model = tf.keras.models.load_model(args.model_path)
    image_size, channels = model_input_config(model)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    prediction_window: deque[np.ndarray] = deque(maxlen=max(1, args.smooth_frames))
    print("Camera started. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for face_box in faces:
            x, y, w, h = face_box
            face_input = preprocess_face(frame, gray, face_box, image_size, channels)
            prediction = model.predict(face_input, verbose=0)[0]
            prediction_window.append(prediction)
            smooth_prediction = average_prediction(prediction_window)

            class_index = int(np.argmax(smooth_prediction))
            confidence = float(smooth_prediction[class_index])
            label = labels[class_index] if confidence >= args.min_confidence else "uncertain"

            color = (0, 220, 0) if label != "uncertain" else (0, 180, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.0%}",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Facial Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
