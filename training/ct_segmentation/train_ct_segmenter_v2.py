from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "segmentation"
    / "processed"
)

SPLIT_DIR = (
    DATA_ROOT
    / "splits"
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


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

tf.random.set_seed(
    SEED
)


# ============================================================
# LOAD SAME SPLITS AS V1
# ============================================================

train_df = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

val_df = pd.read_csv(
    SPLIT_DIR / "val.csv"
)

test_df = pd.read_csv(
    SPLIT_DIR / "test.csv"
)


print()
print("=" * 60)
print("CT INFECTION SEGMENTATION V2")
print("=" * 60)

print()
print("USING EXISTING CASE-LEVEL SPLITS")
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
# ABSOLUTE PATH
# ============================================================

def absolute_path(
    relative_path,
):

    return str(
        PROJECT_ROOT
        / Path(relative_path)
    )


train_images = [
    absolute_path(path)
    for path
    in train_df["image_path"]
]

train_masks = [
    absolute_path(path)
    for path
    in train_df["mask_path"]
]

val_images = [
    absolute_path(path)
    for path
    in val_df["image_path"]
]

val_masks = [
    absolute_path(path)
    for path
    in val_df["mask_path"]
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
        method="bilinear",
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
# STRONGER AUGMENTATION
# ============================================================

def augment_pair(
    image,
    mask,
):

    # Random left/right flip
    do_lr_flip = (
        tf.random.uniform([])
        > 0.5
    )

    image = tf.cond(
        do_lr_flip,
        lambda:
            tf.image.flip_left_right(
                image
            ),
        lambda:
            image,
    )

    mask = tf.cond(
        do_lr_flip,
        lambda:
            tf.image.flip_left_right(
                mask
            ),
        lambda:
            mask,
    )


    # Random up/down flip
    do_ud_flip = (
        tf.random.uniform([])
        > 0.5
    )

    image = tf.cond(
        do_ud_flip,
        lambda:
            tf.image.flip_up_down(
                image
            ),
        lambda:
            image,
    )

    mask = tf.cond(
        do_ud_flip,
        lambda:
            tf.image.flip_up_down(
                mask
            ),
        lambda:
            mask,
    )


    # Rotate image and mask together
    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )

    image = tf.image.rot90(
        image,
        k=k,
    )

    mask = tf.image.rot90(
        mask,
        k=k,
    )


    # Small CT intensity variation
    image = tf.image.random_brightness(
        image,
        max_delta=0.08,
    )

    image = tf.image.random_contrast(
        image,
        lower=0.90,
        upper=1.10,
    )

    image = tf.clip_by_value(
        image,
        0.0,
        1.0,
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
            reshuffle_each_iteration=True,
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
    training=False,
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
        y_true
        * y_pred
    )

    return (
        2.0
        * intersection
        + smooth
    ) / (
        tf.reduce_sum(
            y_true
        )
        +
        tf.reduce_sum(
            y_pred
        )
        +
        smooth
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
        y_true
        * y_pred
    )

    union = (
        tf.reduce_sum(
            y_true
        )
        +
        tf.reduce_sum(
            y_pred
        )
        -
        intersection
    )

    return (
        intersection
        + smooth
    ) / (
        union
        + smooth
    )


# ============================================================
# TVERSKY
# ============================================================

def tversky_index(
    y_true,
    y_pred,
):

    smooth = 1e-6

    # Higher beta penalizes missed lesion pixels
    alpha = 0.30
    beta = 0.70

    y_true = tf.reshape(
        y_true,
        [-1],
    )

    y_pred = tf.reshape(
        y_pred,
        [-1],
    )

    true_positive = tf.reduce_sum(
        y_true
        * y_pred
    )

    false_positive = tf.reduce_sum(
        (1.0 - y_true)
        * y_pred
    )

    false_negative = tf.reduce_sum(
        y_true
        * (1.0 - y_pred)
    )

    return (
        true_positive
        + smooth
    ) / (
        true_positive
        +
        alpha
        * false_positive
        +
        beta
        * false_negative
        +
        smooth
    )


# ============================================================
# FOCAL TVERSKY LOSS
# ============================================================

def focal_tversky_loss(
    y_true,
    y_pred,
):

    tversky = tversky_index(
        y_true,
        y_pred,
    )

    gamma = 0.75

    return tf.pow(
        1.0 - tversky,
        gamma,
    )


# ============================================================
# BCE + FOCAL TVERSKY
# ============================================================

binary_crossentropy = (
    tf.keras.losses
    .BinaryCrossentropy()
)


def combined_focal_tversky_loss(
    y_true,
    y_pred,
):

    bce_loss = binary_crossentropy(
        y_true,
        y_pred,
    )

    ft_loss = focal_tversky_loss(
        y_true,
        y_pred,
    )

    return (
        0.25
        * bce_loss
        +
        ft_loss
    )


# ============================================================
# CONVOLUTION BLOCK
# ============================================================

def conv_block(
    inputs,
    filters,
):

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
    )(inputs)

    x = (
        tf.keras.layers
        .BatchNormalization()
        (x)
    )

    x = (
        tf.keras.layers
        .Activation("relu")
        (x)
    )


    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
    )(x)

    x = (
        tf.keras.layers
        .BatchNormalization()
        (x)
    )

    x = (
        tf.keras.layers
        .Activation("relu")
        (x)
    )

    return x


# ============================================================
# U-NET V2
# ============================================================

def build_unet():

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            1,
        ),
        name="ct_input",
    )


    # Encoder 1
    c1 = conv_block(
        inputs,
        16,
    )

    p1 = (
        tf.keras.layers
        .MaxPooling2D()
        (c1)
    )


    # Encoder 2
    c2 = conv_block(
        p1,
        32,
    )

    p2 = (
        tf.keras.layers
        .MaxPooling2D()
        (c2)
    )


    # Encoder 3
    c3 = conv_block(
        p2,
        64,
    )

    p3 = (
        tf.keras.layers
        .MaxPooling2D()
        (c3)
    )


    # Bottleneck
    bottleneck = conv_block(
        p3,
        128,
    )


    # Decoder 3
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


    # Decoder 2
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


    # Decoder 1
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
            name="infection_mask",
        )
        (c6)
    )


    return tf.keras.Model(
        inputs,
        outputs,
        name="CT_Infection_UNet_V2",
    )


# ============================================================
# BUILD MODEL
# ============================================================

model = build_unet()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=(
        tf.keras.optimizers.Adam(
            learning_rate=5e-4,
        )
    ),
    loss=combined_focal_tversky_loss,
    metrics=[
        dice_coefficient,
        iou_metric,
        tversky_index,
    ],
)

model.summary()


# ============================================================
# MODEL PATHS
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR
    / "ct_infection_segmenter_v2_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "ct_infection_segmenter_v2_final.keras"
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
        patience=4,
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
print("STARTING CT U-NET V2 TRAINING")
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
# SAVE TRAINING HISTORY
# ============================================================

history_data = {

    key: [
        float(value)
        for value
        in values
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
# SAVE DICE GRAPH
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
    "CT Infection Segmentation V2"
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
print("CT U-NET V2 TRAINING COMPLETE")
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