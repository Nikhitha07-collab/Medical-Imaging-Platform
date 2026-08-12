from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "tn5000"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ultrasound"
    / "tn5000_classifier_v2_best.keras"
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 16

# Fixed threshold selected only from validation data
THRESHOLD = 0.58


def decode_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ],
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    label = tf.cast(
        label,
        tf.float32,
    )

    return image, label


print()
print("=" * 60)
print("FINAL V2 TEST EVALUATION")
print("=" * 60)
print()

test_df = pd.read_csv(
    TEST_CSV
)

test_df["image_path"] = (
    test_df["image_path"].apply(
        lambda path: str(
            PROJECT_ROOT
            / Path(path)
        )
    )
)

print(
    f"Test samples: {len(test_df)}"
)

true_labels = (
    test_df["label"]
    .values.astype(
        np.int32
    )
)

paths = (
    test_df["image_path"]
    .values
)

dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (
            paths,
            true_labels,
        )
    )
)

dataset = dataset.map(
    decode_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

dataset = dataset.batch(
    BATCH_SIZE
)

dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)


print(
    f"Loading model:\n{MODEL_PATH}"
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print(
    "Running final test predictions..."
)

probabilities = model.predict(
    dataset,
    verbose=1,
).reshape(-1)

predicted_labels = (
    probabilities
    >= THRESHOLD
).astype(
    np.int32
)


accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

auc = roc_auc_score(
    true_labels,
    probabilities,
)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1],
)


print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Threshold : {THRESHOLD:.2f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"AUC       : {auc:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print()

print(
    "Classification Report:"
)

print()

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=[
            "Benign",
            "Malignant",
        ],
        zero_division=0,
    )
)

print(
    "Confusion Matrix:"
)

print(
    cm
)

print()

tn, fp, fn, tp = cm.ravel()

benign_recall = (
    tn / (tn + fp)
)

malignant_recall = (
    tp / (tp + fn)
)

balanced_accuracy = (
    benign_recall
    + malignant_recall
) / 2.0

print(
    f"Benign Recall     : {benign_recall:.4f}"
)

print(
    f"Malignant Recall  : {malignant_recall:.4f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy:.4f}"
)