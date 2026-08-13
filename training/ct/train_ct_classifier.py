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
    / "ct"
    / "covid_ct"
    / "splits"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "covid_ct"
    / "splits"
    / "val.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "ct"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "training"
    / "ct"
    / "results"
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

EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 10

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------
# LOAD MANIFESTS
# ---------------------------------------------------------

def load_manifest(
    csv_path: Path,
) -> pd.DataFrame:

    dataframe = pd.read_csv(
        csv_path
    )

    if dataframe.empty:
        raise ValueError(
            f"Manifest is empty: {csv_path}"
        )

    required_columns = {
        "image_path",
        "label",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest missing columns: "
            f"{missing_columns}"
        )

    dataframe["image_path"] = (
        dataframe["image_path"]
        .apply(
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
print("COVID-CT CLASSIFIER")
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
# VERIFY IMAGE FILES
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
            "image files missing.\n"
            f"First missing:\n"
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


# ---------------------------------------------------------
# AUGMENTATION
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
        name="ct_augmentation",
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
# BUILD MODEL
# ---------------------------------------------------------

def build_model():

    inputs = tf.keras.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        )
    )

    x = (
        tf.keras.applications
        .efficientnet
        .preprocess_input(
            inputs
        )
    )

    base_model = (
        tf.keras.applications
        .EfficientNetB0(
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

    x = (
        tf.keras.layers
        .GlobalAveragePooling2D()(
            x
        )
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
        name="covid_ct_classifier",
    )

    return model, base_model


model, base_model = build_model()


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def build_metrics():

    return [
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
# MODEL PATHS
# ---------------------------------------------------------

BEST_MODEL_PATH = (
    MODEL_DIR
    / "covid_ct_classifier_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "covid_ct_classifier_final.keras"
)


# ---------------------------------------------------------
# CALLBACK FACTORY
# ---------------------------------------------------------

def build_callbacks():

    return [
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
# PHASE 1
# ---------------------------------------------------------

print()
print("=" * 60)
print("PHASE 1: TRAIN CLASSIFIER HEAD")
print("=" * 60)
print()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=build_metrics(),
)

model.summary()

history_head = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=build_callbacks(),
)


# ---------------------------------------------------------
# PHASE 2
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
    metrics=build_metrics(),
)


history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=build_callbacks(),
)


# ---------------------------------------------------------
# SAVE FINAL MODEL
# ---------------------------------------------------------

model.save(
    FINAL_MODEL_PATH
)


# ---------------------------------------------------------
# SAVE HISTORY
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
                for value
                in values
            ]
        )


history_file = (
    RESULTS_DIR
    / "training_history.json"
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
# TRAINING CURVES
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
    "COVID-CT Classification Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "accuracy_curve.png"
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
    "COVID-CT Classification AUC"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "auc_curve.png"
)

plt.close()


# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print()
print("=" * 60)
print("CT TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best model:\n{BEST_MODEL_PATH}"
)

print()

print(
    f"Final model:\n{FINAL_MODEL_PATH}"
)

print()

print(
    f"Results:\n{RESULTS_DIR}"
)