from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "segmentation"
    / "processed"
)

MANIFEST_PATH = (
    DATA_ROOT
    / "manifest.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "ct"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "training"
    / "ct_segmentation"
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


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(
    MANIFEST_PATH
)

print()
print("=" * 60)
print("CT INFECTION SEGMENTATION TRAINING")
print("=" * 60)

print(
    f"Total infection slices: {len(df)}"
)

cases = sorted(
    df["case_id"].unique()
)

print(
    f"Total CT cases: {len(cases)}"
)


# ============================================================
# CASE-LEVEL SPLIT
# ============================================================

train_cases, temp_cases = train_test_split(
    cases,
    test_size=0.30,
    random_state=SEED,
)

val_cases, test_cases = train_test_split(
    temp_cases,
    test_size=0.50,
    random_state=SEED,
)

train_df = df[
    df["case_id"].isin(
        train_cases
    )
].copy()

val_df = df[
    df["case_id"].isin(
        val_cases
    )
].copy()

test_df = df[
    df["case_id"].isin(
        test_cases
    )
].copy()


print()
print("CASE SPLIT")
print("-" * 60)

print(
    f"Training cases   : {len(train_cases)}"
)

print(
    f"Validation cases : {len(val_cases)}"
)

print(
    f"Test cases       : {len(test_cases)}"
)


print()
print("SLICE SPLIT")
print("-" * 60)

print(
    f"Training slices   : {len(train_df)}"
)

print(
    f"Validation slices : {len(val_df)}"
)

print(
    f"Test slices       : {len(test_df)}"
)


# ============================================================
# SAVE SPLITS
# ============================================================

SPLIT_DIR = (
    DATA_ROOT
    / "splits"
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

train_df.to_csv(
    SPLIT_DIR / "train.csv",
    index=False,
)

val_df.to_csv(
    SPLIT_DIR / "val.csv",
    index=False,
)

test_df.to_csv(
    SPLIT_DIR / "test.csv",
    index=False,
)


# ============================================================
# ABSOLUTE PATHS
# ============================================================

def absolute_path(relative_path):

    return str(
        PROJECT_ROOT
        / Path(relative_path)
    )


train_images = [
    absolute_path(path)
    for path in train_df[
        "image_path"
    ]
]

train_masks = [
    absolute_path(path)
    for path in train_df[
        "mask_path"
    ]
]

val_images = [
    absolute_path(path)
    for path in val_df[
        "image_path"
    ]
]

val_masks = [
    absolute_path(path)
    for path in val_df[
        "mask_path"
    ]
]


# ============================================================
# LOAD IMAGE + MASK
# ============================================================

def load_pair(
    image_path,
    mask_path,
):

    image = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_png(
        image,
        channels=1,
    )

    image = tf.image.resize(
        image,
        [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
    )

    image = tf.cast(
        image,
        tf.float32,
    ) / 255.0


    mask = tf.io.read_file(
        mask_path
    )

    mask = tf.io.decode_png(
        mask,
        channels=1,
    )

    mask = tf.image.resize(
        mask,
        [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
        method="nearest",
    )

    mask = tf.cast(
        mask,
        tf.float32,
    ) / 255.0

    mask = tf.where(
        mask > 0.5,
        1.0,
        0.0,
    )

    return image, mask


# ============================================================
# AUGMENTATION
# ============================================================

def augment_pair(
    image,
    mask,
):

    flip = (
        tf.random.uniform([])
        > 0.5
    )

    image = tf.cond(
        flip,
        lambda:
            tf.image.flip_left_right(
                image
            ),
        lambda:
            image,
    )

    mask = tf.cond(
        flip,
        lambda:
            tf.image.flip_left_right(
                mask
            ),
        lambda:
            mask,
    )

    return image, mask


# ============================================================
# DATASET
# ============================================================

def create_dataset(
    image_paths,
    mask_paths,
    training=False,
):

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                image_paths,
                mask_paths,
            )
        )
    )

    if training:

        dataset = dataset.shuffle(
            min(
                len(image_paths),
                1500,
            ),
            seed=SEED,
        )

    dataset = dataset.map(
        load_pair,
        num_parallel_calls=AUTOTUNE,
    )

    if training:

        dataset = dataset.map(
            augment_pair,
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
    train_images,
    train_masks,
    training=True,
)

val_dataset = create_dataset(
    val_images,
    val_masks,
)


# ============================================================
# DICE
# ============================================================

def dice_coefficient(
    y_true,
    y_pred,
):

    smooth = 1e-6

    y_true = tf.reshape(
        y_true,
        [-1],
    )

    y_pred = tf.reshape(
        y_pred,
        [-1],
    )

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + smooth
    )


# ============================================================
# IoU
# ============================================================

def iou_metric(
    y_true,
    y_pred,
):

    smooth = 1e-6

    y_true = tf.reshape(
        y_true,
        [-1],
    )

    y_pred = tf.reshape(
        y_pred,
        [-1],
    )

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    union = (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        - intersection
    )

    return (
        intersection + smooth
    ) / (
        union + smooth
    )


# ============================================================
# LOSS
# ============================================================

bce = (
    tf.keras.losses
    .BinaryCrossentropy()
)


def combined_loss(
    y_true,
    y_pred,
):

    dice_loss = (
        1.0
        - dice_coefficient(
            y_true,
            y_pred,
        )
    )

    return (
        bce(
            y_true,
            y_pred,
        )
        + dice_loss
    )


# ============================================================
# U-NET
# ============================================================

def conv_block(
    inputs,
    filters,
):

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
    )(x)

    return x


def build_unet():

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            1,
        )
    )


    c1 = conv_block(
        inputs,
        16,
    )

    p1 = (
        tf.keras.layers
        .MaxPooling2D()
        (c1)
    )


    c2 = conv_block(
        p1,
        32,
    )

    p2 = (
        tf.keras.layers
        .MaxPooling2D()
        (c2)
    )


    c3 = conv_block(
        p2,
        64,
    )

    p3 = (
        tf.keras.layers
        .MaxPooling2D()
        (c3)
    )


    bottleneck = conv_block(
        p3,
        128,
    )


    u3 = (
        tf.keras.layers
        .Conv2DTranspose(
            64,
            2,
            strides=2,
            padding="same",
        )
        (bottleneck)
    )

    u3 = (
        tf.keras.layers
        .Concatenate()
        (
            [
                u3,
                c3,
            ]
        )
    )

    c4 = conv_block(
        u3,
        64,
    )


    u2 = (
        tf.keras.layers
        .Conv2DTranspose(
            32,
            2,
            strides=2,
            padding="same",
        )
        (c4)
    )

    u2 = (
        tf.keras.layers
        .Concatenate()
        (
            [
                u2,
                c2,
            ]
        )
    )

    c5 = conv_block(
        u2,
        32,
    )


    u1 = (
        tf.keras.layers
        .Conv2DTranspose(
            16,
            2,
            strides=2,
            padding="same",
        )
        (c5)
    )

    u1 = (
        tf.keras.layers
        .Concatenate()
        (
            [
                u1,
                c1,
            ]
        )
    )

    c6 = conv_block(
        u1,
        16,
    )


    outputs = (
        tf.keras.layers
        .Conv2D(
            1,
            1,
            activation="sigmoid",
        )
        (c6)
    )


    return (
        tf.keras.Model(
            inputs,
            outputs,
            name="CT_Infection_UNet",
        )
    )


# ============================================================
# BUILD + COMPILE
# ============================================================

model = build_unet()

model.compile(
    optimizer=(
        tf.keras.optimizers.Adam(
            learning_rate=1e-3,
        )
    ),
    loss=combined_loss,
    metrics=[
        dice_coefficient,
        iou_metric,
    ],
)

model.summary()


# ============================================================
# MODEL PATHS
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR
    / "ct_infection_segmenter_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "ct_infection_segmenter_final.keras"
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(
            BEST_MODEL_PATH
        ),
        monitor=(
            "val_dice_coefficient"
        ),
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor=(
            "val_dice_coefficient"
        ),
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING CT U-NET TRAINING")
print("=" * 60)
print()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_data = {

    key: [
        float(value)
        for value in values
    ]

    for key, values
    in history.history.items()
}

with (
    RESULTS_DIR
    / "training_history.json"
).open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        history_data,
        file,
        indent=4,
    )


# ============================================================
# DICE GRAPH
# ============================================================

plt.figure()

plt.plot(
    history.history[
        "dice_coefficient"
    ],
    label="Training Dice",
)

plt.plot(
    history.history[
        "val_dice_coefficient"
    ],
    label="Validation Dice",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Dice"
)

plt.title(
    "CT Infection Segmentation Dice"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "dice_curve.png"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("CT U-NET TRAINING COMPLETE")
print("=" * 60)

print()
print("Best model:")
print(
    BEST_MODEL_PATH
)

print()
print("Final model:")
print(
    FINAL_MODEL_PATH
)

print()
print("Results:")
print(
    RESULTS_DIR
)