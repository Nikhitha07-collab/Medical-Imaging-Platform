from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "segmentation"
    / "processed"
    / "localization_manifest.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "ultrasound"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "training"
    / "ultrasound_segmentation"
    / "results"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "segmentation"
    / "processed"
    / "splits"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
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
print("TN5000 ULTRASOUND LESION LOCALIZATION TRAINING")
print("=" * 60)

print(
    f"Total samples: {len(df)}"
)


# ============================================================
# SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["class_label"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["class_label"],
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

print()
print("SPLIT")
print("-" * 60)

print(
    f"Training samples   : {len(train_df)}"
)

print(
    f"Validation samples : {len(val_df)}"
)

print(
    f"Test samples       : {len(test_df)}"
)


# ============================================================
# ABSOLUTE PATHS
# ============================================================

def absolute_path(
    relative_path,
):

    return str(
        PROJECT_ROOT
        / Path(relative_path)
    )


# ============================================================
# DATA
# ============================================================

def dataframe_to_arrays(
    dataframe,
):

    paths = [
        absolute_path(path)
        for path
        in dataframe["image_path"]
    ]

    boxes = dataframe[
        [
            "xmin_norm",
            "ymin_norm",
            "xmax_norm",
            "ymax_norm",
        ]
    ].values.astype(
        np.float32
    )

    return paths, boxes


train_paths, train_boxes = (
    dataframe_to_arrays(
        train_df
    )
)

val_paths, val_boxes = (
    dataframe_to_arrays(
        val_df
    )
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_sample(
    image_path,
    box,
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
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
    )

    image = tf.cast(
        image,
        tf.float32,
    ) / 255.0

    box = tf.cast(
        box,
        tf.float32,
    )

    return image, box


# ============================================================
# HORIZONTAL FLIP WITH BOX UPDATE
# ============================================================

def augment_sample(
    image,
    box,
):

    flip = (
        tf.random.uniform([])
        > 0.5
    )

    def do_flip():

        flipped_image = (
            tf.image.flip_left_right(
                image
            )
        )

        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]

        new_xmin = (
            1.0 - xmax
        )

        new_xmax = (
            1.0 - xmin
        )

        new_box = tf.stack(
            [
                new_xmin,
                ymin,
                new_xmax,
                ymax,
            ]
        )

        return (
            flipped_image,
            new_box,
        )

    return tf.cond(
        flip,
        do_flip,
        lambda: (
            image,
            box,
        ),
    )


# ============================================================
# DATASET
# ============================================================

def create_dataset(
    paths,
    boxes,
    training=False,
):

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                paths,
                boxes,
            )
        )
    )

    if training:

        dataset = dataset.shuffle(
            min(
                len(paths),
                3000,
            ),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_sample,
        num_parallel_calls=AUTOTUNE,
    )

    if training:

        dataset = dataset.map(
            augment_sample,
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
    train_paths,
    train_boxes,
    training=True,
)

val_dataset = create_dataset(
    val_paths,
    val_boxes,
    training=False,
)


# ============================================================
# IoU METRIC
# ============================================================

def bbox_iou(
    y_true,
    y_pred,
):

    y_pred = tf.clip_by_value(
        y_pred,
        0.0,
        1.0,
    )

    true_xmin = y_true[:, 0]
    true_ymin = y_true[:, 1]
    true_xmax = y_true[:, 2]
    true_ymax = y_true[:, 3]

    pred_xmin = y_pred[:, 0]
    pred_ymin = y_pred[:, 1]
    pred_xmax = y_pred[:, 2]
    pred_ymax = y_pred[:, 3]

    inter_xmin = tf.maximum(
        true_xmin,
        pred_xmin,
    )

    inter_ymin = tf.maximum(
        true_ymin,
        pred_ymin,
    )

    inter_xmax = tf.minimum(
        true_xmax,
        pred_xmax,
    )

    inter_ymax = tf.minimum(
        true_ymax,
        pred_ymax,
    )

    inter_width = tf.maximum(
        0.0,
        inter_xmax - inter_xmin,
    )

    inter_height = tf.maximum(
        0.0,
        inter_ymax - inter_ymin,
    )

    intersection = (
        inter_width
        * inter_height
    )

    true_area = tf.maximum(
        0.0,
        true_xmax - true_xmin,
    ) * tf.maximum(
        0.0,
        true_ymax - true_ymin,
    )

    pred_area = tf.maximum(
        0.0,
        pred_xmax - pred_xmin,
    ) * tf.maximum(
        0.0,
        pred_ymax - pred_ymin,
    )

    union = (
        true_area
        + pred_area
        - intersection
    )

    iou = (
        intersection
        / (
            union
            + 1e-6
        )
    )

    return tf.reduce_mean(
        iou
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

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


    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        )
    )

    x = base_model(
        inputs,
        training=False,
    )

    x = (
        tf.keras.layers
        .GlobalAveragePooling2D()
        (x)
    )

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    outputs = (
        tf.keras.layers.Dense(
            4,
            activation="sigmoid",
            name="bounding_box",
        )
        (x)
    )

    return tf.keras.Model(
        inputs,
        outputs,
        name="TN5000_Lesion_Localizer",
    )


model = build_model()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=(
        tf.keras.optimizers.Adam(
            learning_rate=1e-3,
        )
    ),
    loss=(
        tf.keras.losses.Huber()
    ),
    metrics=[
        bbox_iou,
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        ),
    ],
)

model.summary()


# ============================================================
# PATHS
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR
    / "tn5000_localizer_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "tn5000_localizer_final.keras"
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(
            BEST_MODEL_PATH
        ),
        monitor="val_bbox_iou",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_bbox_iou",
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
print("STARTING ULTRASOUND LOCALIZATION TRAINING")
print("=" * 60)
print()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# SAVE
# ============================================================

model.save(
    FINAL_MODEL_PATH
)


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


plt.figure()

plt.plot(
    history.history[
        "bbox_iou"
    ],
    label="Training IoU",
)

plt.plot(
    history.history[
        "val_bbox_iou"
    ],
    label="Validation IoU",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Bounding-box IoU"
)

plt.title(
    "TN5000 Ultrasound Lesion Localization"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "bbox_iou_curve.png"
)

plt.close()


print()
print("=" * 60)
print("ULTRASOUND LOCALIZATION TRAINING COMPLETE")
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