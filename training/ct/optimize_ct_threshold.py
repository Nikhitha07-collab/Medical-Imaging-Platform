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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "covid_ct"
    / "splits"
    / "val.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ct"
    / "covid_ct_classifier_best.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "training"
    / "ct"
    / "threshold_analysis"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 16


def decode_image(image_path, label):

    image_bytes = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape(
        [None, None, 3]
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
print("CT VALIDATION THRESHOLD SELECTION")
print("=" * 60)

val_df = pd.read_csv(
    VAL_CSV
)

val_df["image_path"] = (
    val_df["image_path"]
    .apply(
        lambda path: str(
            PROJECT_ROOT
            / Path(path)
        )
    )
)

true_labels = (
    val_df["label"]
    .values.astype(
        np.int32
    )
)

paths = (
    val_df["image_path"]
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


print()
print(
    f"Validation samples: {len(val_df)}"
)

print()

print(
    f"Loading model:\n{MODEL_PATH}"
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print(
    "Running validation predictions..."
)

probabilities = model.predict(
    dataset,
    verbose=1,
).reshape(-1)


auc = roc_auc_score(
    true_labels,
    probabilities,
)

print()
print(
    f"Validation AUC: {auc:.4f}"
)


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

    tn, fp, fn, tp = (
        cm.ravel()
    )

    noncovid_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    covid_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    balanced_accuracy = (
        noncovid_recall
        + covid_recall
    ) / 2.0

    results.append(
        {
            "threshold": float(
                threshold
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
            "noncovid_recall": float(
                noncovid_recall
            ),
            "covid_recall": float(
                covid_recall
            ),
            "balanced_accuracy": float(
                balanced_accuracy
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


results_csv = (
    RESULTS_DIR
    / "ct_threshold_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False,
)


selected_threshold = float(
    best["threshold"]
)


config = {
    "threshold": selected_threshold,
    "validation_auc": float(
        auc
    ),
    "balanced_accuracy": float(
        best["balanced_accuracy"]
    ),
    "noncovid_recall": float(
        best["noncovid_recall"]
    ),
    "covid_recall": float(
        best["covid_recall"]
    ),
}


config_file = (
    RESULTS_DIR
    / "selected_ct_threshold.json"
)

with config_file.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        config,
        file,
        indent=4,
    )


print()
print("=" * 60)
print("SELECTED CT THRESHOLD")
print("=" * 60)

print(
    f"Threshold          : "
    f"{selected_threshold:.2f}"
)

print(
    f"Balanced Accuracy  : "
    f"{best['balanced_accuracy']:.4f}"
)

print(
    f"NonCOVID Recall    : "
    f"{best['noncovid_recall']:.4f}"
)

print(
    f"COVID Recall       : "
    f"{best['covid_recall']:.4f}"
)

print(
    f"Accuracy           : "
    f"{best['accuracy']:.4f}"
)

print(
    f"Precision          : "
    f"{best['precision']:.4f}"
)

print(
    f"F1 Score           : "
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