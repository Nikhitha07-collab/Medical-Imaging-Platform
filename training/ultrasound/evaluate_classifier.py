from pathlib import Path

import matplotlib.pyplot as plt
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


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

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
    / "tn5000_classifier_best.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "training"
    / "ultrasound"
    / "results"
    / "test_evaluation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

BATCH_SIZE = 16

THRESHOLD = 0.5


# ---------------------------------------------------------
# LOAD TEST MANIFEST
# ---------------------------------------------------------

print()
print("=" * 60)
print("TN5000 TEST EVALUATION")
print("=" * 60)
print()

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"test.csv not found:\n{TEST_CSV}"
    )

test_df = pd.read_csv(
    TEST_CSV
)

required_columns = {
    "image_id",
    "image_path",
    "label",
}

missing_columns = (
    required_columns
    - set(test_df.columns)
)

if missing_columns:
    raise ValueError(
        "test.csv is missing columns: "
        f"{missing_columns}"
    )

if test_df.empty:
    raise ValueError(
        "test.csv is empty."
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

print()
print("Test class distribution:")

print(
    test_df["label"]
    .value_counts()
    .sort_index()
)

print()


# ---------------------------------------------------------
# VERIFY TEST IMAGES
# ---------------------------------------------------------

missing_files = []

for image_path in test_df[
    "image_path"
]:
    if not Path(
        image_path
    ).exists():
        missing_files.append(
            image_path
        )

if missing_files:
    raise FileNotFoundError(
        f"{len(missing_files)} test images "
        "could not be found.\n"
        f"First missing file:\n"
        f"{missing_files[0]}"
    )

print(
    "TEST: all image files verified"
)

print()


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# Must match training preprocessing exactly.
# ---------------------------------------------------------

def decode_image(
    image_path,
    label,
):
    image_bytes = tf.io.read_file(
        image_path
    )

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

    image = image / 255.0

    label = tf.cast(
        label,
        tf.float32,
    )

    return image, label


# ---------------------------------------------------------
# CREATE TEST DATASET
# ---------------------------------------------------------

test_paths = (
    test_df[
        "image_path"
    ].values
)

true_labels = (
    test_df[
        "label"
    ].values.astype(
        np.int32
    )
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (
            test_paths,
            true_labels,
        )
    )
)

test_dataset = test_dataset.map(
    decode_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Best trained model was not found:\n"
        f"{MODEL_PATH}"
    )

print(
    f"Loading model:\n{MODEL_PATH}"
)

print()

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)

print()


# ---------------------------------------------------------
# MODEL PREDICTIONS
# ---------------------------------------------------------

print(
    "Running predictions on test set..."
)

probabilities = model.predict(
    test_dataset,
    verbose=1,
)

probabilities = (
    probabilities
    .reshape(-1)
)

predicted_labels = (
    probabilities
    >= THRESHOLD
).astype(
    np.int32
)


# ---------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------

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

try:
    auc = roc_auc_score(
        true_labels,
        probabilities,
    )

except ValueError:
    auc = float("nan")


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"AUC      : {auc:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print()


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

report = classification_report(
    true_labels,
    predicted_labels,
    labels=[0, 1],
    target_names=[
        "Benign",
        "Malignant",
    ],
    zero_division=0,
)

print(
    "Classification Report:"
)

print()
print(report)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1],
)

print(
    "Confusion Matrix:"
)

print(cm)

print()


# ---------------------------------------------------------
# SAVE PREDICTIONS
# ---------------------------------------------------------

predictions_df = pd.DataFrame(
    {
        "image_id": (
            test_df[
                "image_id"
            ].values
        ),
        "image_path": (
            test_df[
                "image_path"
            ].values
        ),
        "true_label": true_labels,
        "predicted_probability": (
            probabilities
        ),
        "predicted_label": (
            predicted_labels
        ),
        "correct": (
            true_labels
            == predicted_labels
        ),
    }
)

prediction_csv = (
    RESULTS_DIR
    / "test_predictions.csv"
)

predictions_df.to_csv(
    prediction_csv,
    index=False,
)


# ---------------------------------------------------------
# SAVE METRICS
# ---------------------------------------------------------

metrics_df = pd.DataFrame(
    [
        {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "threshold": THRESHOLD,
            "test_samples": len(
                test_df
            ),
        }
    ]
)

metrics_csv = (
    RESULTS_DIR
    / "test_metrics.csv"
)

metrics_df.to_csv(
    metrics_csv,
    index=False,
)


# ---------------------------------------------------------
# SAVE CLASSIFICATION REPORT
# ---------------------------------------------------------

report_file = (
    RESULTS_DIR
    / "classification_report.txt"
)

with report_file.open(
    "w",
    encoding="utf-8",
) as file:
    file.write(
        report
    )


# ---------------------------------------------------------
# SAVE CONFUSION MATRIX
# ---------------------------------------------------------

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm
)

plt.title(
    "TN5000 Test Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    [
        "Benign",
        "Malignant",
    ],
)

plt.yticks(
    [0, 1],
    [
        "Benign",
        "Malignant",
    ],
)

for i in range(
    cm.shape[0]
):
    for j in range(
        cm.shape[1]
    ):
        plt.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center",
        )

plt.tight_layout()

confusion_matrix_path = (
    RESULTS_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path
)

plt.close()


# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------

print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print()

print(
    "Results saved to:"
)

print(
    RESULTS_DIR
)

print()

print(
    f"Predictions:\n{prediction_csv}"
)

print()

print(
    f"Metrics:\n{metrics_csv}"
)

print()

print(
    "Confusion matrix:\n"
    f"{confusion_matrix_path}"
)