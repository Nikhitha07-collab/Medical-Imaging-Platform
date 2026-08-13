from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "brain_tumor"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mri"
    / "brain_mri_classifier_best.keras"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

CLASS_NAMES = [
    "meningioma",
    "glioma",
    "pituitary",
]


def decode_image(path, label):

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_png(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    label = tf.cast(
        label,
        tf.int32,
    )

    return image, label


print()
print("=" * 60)
print("BRAIN MRI TEST EVALUATION")
print("=" * 60)


test_df = pd.read_csv(TEST_CSV)

test_df["image_path"] = (
    test_df["image_path"].apply(
        lambda path: str(
            PROJECT_ROOT / Path(path)
        )
    )
)

print(f"Test samples: {len(test_df)}")


missing = [
    path
    for path in test_df["image_path"]
    if not Path(path).exists()
]

if missing:
    raise FileNotFoundError(
        f"{len(missing)} test images are missing.\n"
        f"First missing image:\n{missing[0]}"
    )

print("TEST: all image files verified")


true_labels = (
    test_df["label"]
    .values.astype(np.int32)
)

paths = test_df["image_path"].values


dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (paths, true_labels)
    )
)

dataset = dataset.map(
    decode_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

dataset = dataset.batch(BATCH_SIZE)

dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)


print()
print(f"Loading model:\n{MODEL_PATH}")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print("Running MRI predictions...")


probabilities = model.predict(
    dataset,
    verbose=1,
)

predicted_labels = np.argmax(
    probabilities,
    axis=1,
)


accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1, 2],
)


print()
print("=" * 60)
print("MRI TEST RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")

print()
print("Classification Report:")
print()

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(cm)