from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "brain_tumor"
    / "splits"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "brain_tumor"
    / "splits"
    / "val.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "mri"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "training"
    / "mri"
    / "results"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_SIZE = 224
BATCH_SIZE = 16

HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 8

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

NUM_CLASSES = 3

CLASS_NAMES = [
    "meningioma",
    "glioma",
    "pituitary",
]


def load_manifest(path):

    df = pd.read_csv(path)

    df["image_path"] = df["image_path"].apply(
        lambda p: str(
            PROJECT_ROOT / Path(p)
        )
    )

    return df


train_df = load_manifest(TRAIN_CSV)
val_df = load_manifest(VAL_CSV)


print()
print("=" * 60)
print("BRAIN MRI CLASSIFIER")
print("=" * 60)

print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")

print()
print("Training distribution:")
print(train_df["class_name"].value_counts())


def verify_files(df, name):

    missing = [
        p
        for p in df["image_path"]
        if not Path(p).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} files missing.\n"
            f"First missing file:\n{missing[0]}"
        )

    print(
        f"{name}: all image files verified"
    )


verify_files(train_df, "TRAIN")
verify_files(val_df, "VALIDATION")


def decode_image(path, label):

    data = tf.io.read_file(path)

    image = tf.io.decode_png(
        data,
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


augmentation = tf.keras.Sequential(
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
    name="mri_augmentation",
)


def augment(image, label):

    image = augmentation(
        image,
        training=True,
    )

    return image, label


def create_dataset(df, training=False):

    paths = df["image_path"].values

    labels = (
        df["label"]
        .values
        .astype(np.int32)
    )

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        ds = ds.shuffle(
            len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        decode_image,
        num_parallel_calls=AUTOTUNE,
    )

    if training:

        ds = ds.map(
            augment,
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTOTUNE)

    return ds


train_ds = create_dataset(
    train_df,
    training=True,
)

val_ds = create_dataset(
    val_df,
    training=False,
)


classes = np.array(
    sorted(
        train_df["label"].unique()
    )
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].values,
)

class_weights = {
    int(c): float(w)
    for c, w in zip(
        classes,
        weights,
    )
}

print()
print("Class weights:")
print(class_weights)


def build_model():

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        )
    )

    x = (
        tf.keras.applications
        .efficientnet
        .preprocess_input(inputs)
    )

    base_model = (
        tf.keras.applications
        .EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
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
        .GlobalAveragePooling2D()(x)
    )

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="brain_mri_classifier",
    )

    return model, base_model


model, base_model = build_model()


BEST_MODEL = (
    MODEL_DIR
    / "brain_mri_classifier_best.keras"
)

FINAL_MODEL = (
    MODEL_DIR
    / "brain_mri_classifier_final.keras"
)


def callbacks():

    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_MODEL),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
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


print()
print("=" * 60)
print("PHASE 1: TRAIN CLASSIFIER HEAD")
print("=" * 60)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=(
        tf.keras.losses
        .SparseCategoricalCrossentropy()
    ),
    metrics=["accuracy"],
)


history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks(),
)


print()
print("=" * 60)
print("PHASE 2: FINE-TUNING")
print("=" * 60)


base_model.trainable = True

fine_tune_at = max(
    len(base_model.layers) - 30,
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
    loss=(
        tf.keras.losses
        .SparseCategoricalCrossentropy()
    ),
    metrics=["accuracy"],
)


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks(),
)


model.save(FINAL_MODEL)


history = {}

for h in [
    history1.history,
    history2.history,
]:

    for key, values in h.items():

        history.setdefault(
            key,
            []
        )

        history[key].extend(
            [
                float(v)
                for v in values
            ]
        )


with open(
    RESULTS_DIR / "training_history.json",
    "w",
) as f:

    json.dump(
        history,
        f,
        indent=4,
    )


plt.figure()

plt.plot(
    history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Brain MRI Classification Accuracy"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "accuracy_curve.png"
)

plt.close()


print()
print("=" * 60)
print("MRI TRAINING COMPLETE")
print("=" * 60)

print(f"Best model:\n{BEST_MODEL}")

print()

print(f"Final model:\n{FINAL_MODEL}")

print()

print(f"Results:\n{RESULTS_DIR}")