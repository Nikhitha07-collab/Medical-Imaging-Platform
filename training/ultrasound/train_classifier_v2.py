from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "tn5000"
    / "splits"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "tn5000"
    / "splits"
    / "val.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "ultrasound"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "training"
    / "ultrasound"
    / "results_v2"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

EPOCHS_HEAD = 12
EPOCHS_FINE_TUNE = 12

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------
# LOAD MANIFESTS
# ---------------------------------------------------------

def load_manifest(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        csv_path
    )

    if dataframe.empty:
        raise ValueError(
            f"Manifest is empty: {csv_path}"
        )

    required_columns = {
        "image_id",
        "image_path",
        "label",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            f"{missing_columns}"
        )

    dataframe["image_path"] = (
        dataframe[
            "image_path"
        ].apply(
            lambda path: str(
                PROJECT_ROOT
                / Path(path)
            )
        )
    )

    return dataframe


train_df = load_manifest(
    TRAIN_CSV
)

val_df = load_manifest(
    VAL_CSV
)


print()
print("=" * 60)
print("TN5000 ULTRASOUND CLASSIFIER V2")
print("=" * 60)

print(
    f"Training samples: {len(train_df)}"
)

print(
    f"Validation samples: {len(val_df)}"
)

print()

print(
    "Training class distribution:"
)

print(
    train_df[
        "label"
    ]
    .value_counts()
    .sort_index()
)

print()


# ---------------------------------------------------------
# VERIFY FILES
# ---------------------------------------------------------

def verify_files(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    missing_files = []

    for image_path in dataframe[
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
            f"{split_name}: "
            f"{len(missing_files)} "
            "image files are missing. "
            f"First missing file: "
            f"{missing_files[0]}"
        )

    print(
        f"{split_name}: all image files verified"
    )


verify_files(
    train_df,
    "TRAIN",
)

verify_files(
    val_df,
    "VALIDATION",
)


# ---------------------------------------------------------
# IMAGE DECODING
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
# DATA AUGMENTATION
# ---------------------------------------------------------

data_augmentation = (
    tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal"
            ),
            tf.keras.layers.RandomRotation(
                0.04
            ),
            tf.keras.layers.RandomZoom(
                0.08
            ),
            tf.keras.layers.RandomContrast(
                0.10
            ),
        ],
        name="augmentation",
    )
)


def augment_image(
    image,
    label,
):
    image = data_augmentation(
        image,
        training=True,
    )

    return image, label


# ---------------------------------------------------------
# DATASETS
# ---------------------------------------------------------

def create_dataset(
    dataframe: pd.DataFrame,
    training: bool,
):
    paths = dataframe[
        "image_path"
    ].values

    labels = dataframe[
        "label"
    ].values.astype(
        np.float32
    )

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                paths,
                labels,
            )
        )
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(
                dataframe
            ),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_image,
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


train_dataset = create_dataset(
    train_df,
    training=True,
)

val_dataset = create_dataset(
    val_df,
    training=False,
)


# ---------------------------------------------------------
# CLASS WEIGHTS
# ---------------------------------------------------------

classes = np.array(
    sorted(
        train_df[
            "label"
        ].unique()
    )
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df[
        "label"
    ].values,
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight
    in zip(
        classes,
        weights,
    )
}

print()
print(
    "Class weights:"
)
print(
    class_weights
)
print()


# ---------------------------------------------------------
# BUILD TRANSFER-LEARNING MODEL
# ---------------------------------------------------------

def build_model():
    inputs = tf.keras.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        )
    )

    x = tf.keras.applications.efficientnet.preprocess_input(
        inputs
    )

    base_model = (
        tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                3,
            ),
        )
    )

    base_model.trainable = False

    x = base_model(
        x,
        training=False,
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(
        x
    )

    x = tf.keras.layers.Dropout(
        0.30
    )(
        x
    )

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(
        x
    )

    x = tf.keras.layers.Dropout(
        0.30
    )(
        x
    )

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(
        x
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="tn5000_ultrasound_classifier_v2",
    )

    return (
        model,
        base_model,
    )


model, base_model = build_model()


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

METRICS = [
    tf.keras.metrics.BinaryAccuracy(
        name="accuracy"
    ),
    tf.keras.metrics.AUC(
        name="auc"
    ),
    tf.keras.metrics.Precision(
        name="precision"
    ),
    tf.keras.metrics.Recall(
        name="recall"
    ),
]


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BEST_MODEL_PATH = (
    MODEL_DIR
    / "tn5000_classifier_v2_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "tn5000_classifier_v2_final.keras"
)


# ---------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(
            BEST_MODEL_PATH
        ),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ---------------------------------------------------------
# PHASE 1: TRAIN CLASSIFIER HEAD
# ---------------------------------------------------------

print()
print("=" * 60)
print("PHASE 1: TRAINING CLASSIFIER HEAD")
print("=" * 60)
print()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=METRICS,
)

model.summary()

history_head = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks,
)


# ---------------------------------------------------------
# PHASE 2: FINE-TUNE TOP EFFICIENTNET LAYERS
# ---------------------------------------------------------

print()
print("=" * 60)
print("PHASE 2: FINE-TUNING")
print("=" * 60)
print()

base_model.trainable = True

fine_tune_at = max(
    len(
        base_model.layers
    )
    - 30,
    0,
)

for layer in base_model.layers[
    :fine_tune_at
]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=METRICS,
)


history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=callbacks,
)


# ---------------------------------------------------------
# SAVE FINAL MODEL
# ---------------------------------------------------------

model.save(
    FINAL_MODEL_PATH
)

print()
print(
    f"Final V2 model saved to:"
)

print(
    FINAL_MODEL_PATH
)


# ---------------------------------------------------------
# MERGE TRAINING HISTORY
# ---------------------------------------------------------

combined_history = {}

for history in [
    history_head.history,
    history_fine.history,
]:
    for key, values in history.items():
        combined_history.setdefault(
            key,
            []
        )

        combined_history[
            key
        ].extend(
            [
                float(value)
                for value in values
            ]
        )


history_file = (
    RESULTS_DIR
    / "training_history_v2.json"
)

with history_file.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        combined_history,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# CURVES
# ---------------------------------------------------------

plt.figure()

plt.plot(
    combined_history[
        "accuracy"
    ],
    label="Training Accuracy",
)

plt.plot(
    combined_history[
        "val_accuracy"
    ],
    label="Validation Accuracy",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "TN5000 V2 Classification Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "accuracy_curve_v2.png"
)

plt.close()


plt.figure()

plt.plot(
    combined_history[
        "auc"
    ],
    label="Training AUC",
)

plt.plot(
    combined_history[
        "val_auc"
    ],
    label="Validation AUC",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "AUC"
)

plt.title(
    "TN5000 V2 Classification AUC"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "auc_curve_v2.png"
)

plt.close()


print()
print("=" * 60)
print("V2 TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best model: {BEST_MODEL_PATH}"
)

print(
    f"Final model: {FINAL_MODEL_PATH}"
)

print(
    f"Results: {RESULTS_DIR}"
)