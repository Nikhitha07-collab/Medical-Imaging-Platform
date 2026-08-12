from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "tn5000"
    / "splits"
    / "val.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ultrasound"
    / "tn5000_classifier_v2_best.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "training"
    / "ultrasound"
    / "results_v2"
    / "validation_threshold"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

THRESHOLD_CONFIG = (
    RESULTS_DIR
    / "selected_threshold.json"
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 16


# ---------------------------------------------------------
# LOAD VALIDATION MANIFEST
# ---------------------------------------------------------

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )

val_df = pd.read_csv(
    VAL_CSV
)

required_columns = {
    "image_id",
    "image_path",
    "label",
}

missing_columns = (
    required_columns
    - set(val_df.columns)
)

if missing_columns:
    raise ValueError(
        "Validation CSV is missing columns: "
        f"{missing_columns}"
    )

val_df["image_path"] = (
    val_df["image_path"].apply(
        lambda path: str(
            PROJECT_ROOT
            / Path(path)
        )
    )
)

print()
print("=" * 60)
print("V2 VALIDATION THRESHOLD SELECTION")
print("=" * 60)
print()

print(
    f"Validation samples: {len(val_df)}"
)

print()

print(
    "Validation class distribution:"
)

print(
    val_df["label"]
    .value_counts()
    .sort_index()
)

print()


# ---------------------------------------------------------
# VERIFY FILES
# ---------------------------------------------------------

missing_files = [
    image_path
    for image_path
    in val_df["image_path"]
    if not Path(image_path).exists()
]

if missing_files:
    raise FileNotFoundError(
        f"{len(missing_files)} validation images "
        "are missing.\n"
        f"First missing file:\n"
        f"{missing_files[0]}"
    )

print(
    "VALIDATION: all image files verified"
)

print()


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# Must match V2 training.
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

    label = tf.cast(
        label,
        tf.float32,
    )

    return image, label


# ---------------------------------------------------------
# CREATE VALIDATION DATASET
# ---------------------------------------------------------

paths = (
    val_df["image_path"]
    .values
)

true_labels = (
    val_df["label"]
    .values.astype(
        np.int32
    )
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


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V2 model not found:\n{MODEL_PATH}"
    )

print(
    "Loading model:"
)

print(
    MODEL_PATH
)

print()

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded."
)

print()


# ---------------------------------------------------------
# VALIDATION PREDICTIONS
# ---------------------------------------------------------

print(
    "Running validation predictions..."
)

probabilities = model.predict(
    dataset,
    verbose=1,
).reshape(-1)


# ---------------------------------------------------------
# VALIDATION AUC
# ---------------------------------------------------------

auc = roc_auc_score(
    true_labels,
    probabilities,
)

print()
print(
    f"Validation AUC: {auc:.4f}"
)
print()


# ---------------------------------------------------------
# SEARCH THRESHOLDS
# ---------------------------------------------------------

results = []

for threshold in np.arange(
    0.10,
    0.91,
    0.01,
):

    predicted = (
        probabilities
        >= threshold
    ).astype(
        np.int32
    )

    cm = confusion_matrix(
        true_labels,
        predicted,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    benign_recall = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    malignant_recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    balanced_accuracy = (
        benign_recall
        + malignant_recall
    ) / 2.0

    results.append(
        {
            "threshold": float(
                threshold
            ),
            "balanced_accuracy": float(
                balanced_accuracy
            ),
            "benign_recall": float(
                benign_recall
            ),
            "malignant_recall": float(
                malignant_recall
            ),
            "accuracy": float(
                accuracy_score(
                    true_labels,
                    predicted,
                )
            ),
            "precision": float(
                precision_score(
                    true_labels,
                    predicted,
                    zero_division=0,
                )
            ),
            "f1_score": float(
                f1_score(
                    true_labels,
                    predicted,
                    zero_division=0,
                )
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )


results_df = pd.DataFrame(
    results
)

best = (
    results_df
    .sort_values(
        [
            "balanced_accuracy",
            "f1_score",
        ],
        ascending=False,
    )
    .iloc[0]
)


# ---------------------------------------------------------
# SAVE FULL TABLE
# ---------------------------------------------------------

results_csv = (
    RESULTS_DIR
    / "validation_threshold_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False,
)


# ---------------------------------------------------------
# SAVE CHOSEN THRESHOLD
# ---------------------------------------------------------

selected_threshold = float(
    best["threshold"]
)

config = {
    "threshold": selected_threshold,
    "selection_dataset": "validation",
    "selection_metric": "balanced_accuracy",
    "validation_auc": float(auc),
    "balanced_accuracy": float(
        best["balanced_accuracy"]
    ),
    "benign_recall": float(
        best["benign_recall"]
    ),
    "malignant_recall": float(
        best["malignant_recall"]
    ),
}

with THRESHOLD_CONFIG.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("=" * 60)
print("SELECTED VALIDATION THRESHOLD")
print("=" * 60)

print(
    f"Threshold           : "
    f"{selected_threshold:.2f}"
)

print(
    f"Balanced Accuracy   : "
    f"{best['balanced_accuracy']:.4f}"
)

print(
    f"Benign Recall       : "
    f"{best['benign_recall']:.4f}"
)

print(
    f"Malignant Recall    : "
    f"{best['malignant_recall']:.4f}"
)

print(
    f"Accuracy            : "
    f"{best['accuracy']:.4f}"
)

print(
    f"Precision           : "
    f"{best['precision']:.4f}"
)

print(
    f"F1 Score            : "
    f"{best['f1_score']:.4f}"
)

print()

print(
    "Confusion Matrix:"
)

print(
    [
        [
            int(best["tn"]),
            int(best["fp"]),
        ],
        [
            int(best["fn"]),
            int(best["tp"]),
        ],
    ]
)

print()

print(
    "Threshold config saved to:"
)

print(
    THRESHOLD_CONFIG
)