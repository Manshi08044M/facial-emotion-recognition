"""Facial emotion recognition training and inference script.

Expected dataset layout:

dataset/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  validation/
    angry/
    ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models


DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "emotion_model.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "emotion_labels.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
CNN_IMG_SIZE = (48, 48)
TRANSFER_IMG_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or run prediction for a facial emotion model."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Action to run.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset folder containing train and validation directories.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path used for saving/loading the Keras model.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path used for saving/loading class labels.",
    )
    parser.add_argument("--image", type=Path, help="Image path for prediction mode.")
    parser.add_argument(
        "--architecture",
        choices=["cnn", "mobilenetv2"],
        default="cnn",
        help="Model architecture for fresh training.",
    )
    parser.add_argument(
        "--imagenet-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet pretrained weights for transfer learning models.",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=40,
        help="Number of final base-model layers to unfreeze for transfer learning.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from --model-path if the model file exists.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam learning rate."
    )
    parser.add_argument(
        "--patience", type=int, default=8, help="Early stopping patience."
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting during training.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Optional limit for quick training smoke tests.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Optional limit for quick validation smoke tests.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Folder for confusion matrix image.",
    )
    return parser.parse_args()


def save_labels(class_names: list[str], labels_path: Path) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def load_labels(labels_path: Path, dataset_dir: Path) -> list[str]:
    if labels_path.is_file():
        return json.loads(labels_path.read_text(encoding="utf-8"))

    train_dir, _ = validate_dataset(dataset_dir)
    return sorted(path.name for path in train_dir.iterdir() if path.is_dir())


def validate_dataset(dataset_dir: Path) -> tuple[Path, Path]:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"

    missing = [path for path in (train_dir, val_dir) if not path.is_dir()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing dataset folder(s): {missing_text}")

    train_classes = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    val_classes = sorted(path.name for path in val_dir.iterdir() if path.is_dir())
    if not train_classes:
        raise ValueError(f"No class folders found in {train_dir}")
    if train_classes != val_classes:
        raise ValueError(
            "Train and validation classes do not match.\n"
            f"Train: {train_classes}\nValidation: {val_classes}"
        )

    return train_dir, val_dir


def load_datasets(
    dataset_dir: Path,
    batch_size: int,
    image_size: tuple[int, int],
    color_mode: str,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dir, val_dir = validate_dataset(dataset_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode=color_mode,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode=color_mode,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    if max_train_batches:
        train_ds = train_ds.take(max_train_batches)
    if max_val_batches:
        val_ds = val_ds.take(max_val_batches)

    if image_size[0] <= 64 and image_size[1] <= 64:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def get_class_weights(dataset_dir: Path, class_names: list[str]) -> dict[int, float]:
    train_dir, _ = validate_dataset(dataset_dir)
    labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        class_dir = train_dir / class_name
        image_count = sum(1 for path in class_dir.iterdir() if path.is_file())
        labels.extend([class_index] * image_count)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.asarray(labels),
    )
    return {index: float(weight) for index, weight in enumerate(weights)}


def model_input_config(model: tf.keras.Model) -> tuple[tuple[int, int], str]:
    _, height, width, channels = model.input_shape
    if height is None or width is None or channels is None:
        raise ValueError(f"Model has unsupported input shape: {model.input_shape}")

    color_mode = "grayscale" if channels == 1 else "rgb"
    return (int(height), int(width)), color_mode


def build_cnn_model(num_classes: int, learning_rate: float) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(48, 48, 1)),
            layers.Rescaling(1.0 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.30),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.35),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.50),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenetv2_model(
    num_classes: int,
    learning_rate: float,
    imagenet_weights: bool,
    fine_tune_layers: int,
) -> tf.keras.Model:
    weights = "imagenet" if imagenet_weights else None
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - fine_tune_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.08)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(args: argparse.Namespace, num_classes: int) -> tf.keras.Model:
    if args.architecture == "mobilenetv2":
        return build_mobilenetv2_model(
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            imagenet_weights=args.imagenet_weights,
            fine_tune_layers=args.fine_tune_layers,
        )

    return build_cnn_model(num_classes, args.learning_rate)


def fresh_model_config(args: argparse.Namespace) -> tuple[tuple[int, int], str]:
    if args.architecture == "mobilenetv2":
        return TRANSFER_IMG_SIZE, "rgb"
    return CNN_IMG_SIZE, "grayscale"


def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list[str],
    report_dir: Path,
) -> None:
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(tf.argmax(labels, axis=1).numpy().tolist())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy().tolist())

    labels = list(range(len(class_names)))

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = report_dir / "confusion_matrix.png"
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {matrix_path}")


def train(args: argparse.Namespace) -> None:
    resume_from_existing = args.resume and args.model_path.is_file()
    initial_val_accuracy = None

    if resume_from_existing:
        print(f"Resuming training from: {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        image_size, color_mode = model_input_config(model)
    else:
        image_size, color_mode = fresh_model_config(args)

    train_ds, val_ds, class_names = load_datasets(
        args.dataset_dir,
        args.batch_size,
        image_size,
        color_mode,
        args.max_train_batches,
        args.max_val_batches,
    )
    save_labels(class_names, args.labels_path)

    if resume_from_existing:
        print("Checking current saved model validation accuracy before resume...")
        initial_val_accuracy = float(model.evaluate(val_ds, verbose=0)[1])
        print(f"Current saved model val_accuracy: {initial_val_accuracy:.4f}")
    else:
        if args.resume:
            print(f"No existing model found at {args.model_path}; starting fresh.")
        model = build_model(args, len(class_names))

    model.summary()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    class_weight = None if args.no_class_weights else get_class_weights(args.dataset_dir, class_names)
    if class_weight:
        print(f"Using class weights: {class_weight}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
            initial_value_threshold=initial_val_accuracy,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(args.report_dir / "training_history.csv", append=args.resume),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    if args.model_path.is_file():
        model = tf.keras.models.load_model(args.model_path)
        print(f"Best model available at: {args.model_path}")
    else:
        model.save(args.model_path)
        print(f"Model saved to: {args.model_path}")

    evaluate_model(model, val_ds, class_names, args.report_dir)


def evaluate(args: argparse.Namespace) -> None:
    model = tf.keras.models.load_model(args.model_path)
    image_size, color_mode = model_input_config(model)
    _, val_ds, class_names = load_datasets(
        args.dataset_dir,
        args.batch_size,
        image_size,
        color_mode,
        max_train_batches=1,
        max_val_batches=args.max_val_batches,
    )
    save_labels(class_names, args.labels_path)
    evaluate_model(model, val_ds, class_names, args.report_dir)


def preprocess_image(image_path: Path, model: tf.keras.Model) -> np.ndarray:
    image_size, color_mode = model_input_config(model)
    image = Image.open(image_path)
    image = image.convert("L" if color_mode == "grayscale" else "RGB").resize(image_size)
    image_array = np.asarray(image, dtype=np.float32)
    if color_mode == "grayscale":
        image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict(args: argparse.Namespace) -> None:
    if not args.image:
        raise ValueError("--image is required in predict mode.")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    class_names = load_labels(args.labels_path, args.dataset_dir)
    model = tf.keras.models.load_model(args.model_path)
    image_array = preprocess_image(args.image, model)
    prediction = model.predict(image_array, verbose=0)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])

    print(f"Predicted emotion: {class_names[class_index]} ({confidence:.2%})")


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "predict":
        predict(args)


if __name__ == "__main__":
    main()
