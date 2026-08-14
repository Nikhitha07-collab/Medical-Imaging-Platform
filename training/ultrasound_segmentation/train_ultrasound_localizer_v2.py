from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "segmentation"
    / "processed"
)

SPLIT_DIR = (
    PROCESSED_ROOT
    / "splits"
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


IMAGE_SIZE = 224
BATCH_SIZE = 16
SEED = 42

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 10

AUTOTUNE = tf.data.AUTOTUNE

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD EXISTING TRAIN / VALIDATION / TEST SPLITS
# ============================================================

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"


for required_file in [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Required split file not found:\n{required_file}"
        )


train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)


print()
print("=" * 60)
print("TN5000 ULTRASOUND LOCALIZATION V2")
print("=" * 60)

print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Test samples       : {len(test_df)}")


# ============================================================
# CONVERT RELATIVE PATH TO ABSOLUTE PATH
# ============================================================

def absolute_path(relative_path):

    return str(
        PROJECT_ROOT
        / Path(relative_path)
    )


# ============================================================
# DATAFRAME -> IMAGE PATHS + BOXES
# ============================================================

def dataframe_to_arrays(dataframe):

    paths = [
        absolute_path(path)
        for path in dataframe["image_path"]
    ]

    boxes = dataframe[
        [
            "xmin_norm",
            "ymin_norm",
            "xmax_norm",
            "ymax_norm",
        ]
    ].values.astype(np.float32)

    return paths, boxes


train_paths, train_boxes = dataframe_to_arrays(
    train_df
)

val_paths, val_boxes = dataframe_to_arrays(
    val_df
)

test_paths, test_boxes = dataframe_to_arrays(
    test_df
)


# ============================================================
# LOAD ULTRASOUND IMAGE
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

    # IMPORTANT:
    # EfficientNetB0 already contains its own input rescaling.
    # Keep the image in the 0-255 range.
    image = tf.cast(
        image,
        tf.float32,
    )

    box = tf.cast(
        box,
        tf.float32,
    )

    return image, box


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_sample(
    image,
    box,
):

    do_flip = (
        tf.random.uniform([])
        > 0.5
    )

    def horizontal_flip():

        flipped_image = (
            tf.image.flip_left_right(
                image
            )
        )

        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]

        flipped_box = tf.stack(
            [
                1.0 - xmax,
                ymin,
                1.0 - xmin,
                ymax,
            ]
        )

        return (
            flipped_image,
            flipped_box,
        )

    image, box = tf.cond(
        do_flip,
        horizontal_flip,
        lambda: (
            image,
            box,
        ),
    )

    image = tf.image.random_brightness(
        image,
        max_delta=12.0,
    )

    image = tf.image.random_contrast(
        image,
        lower=0.90,
        upper=1.10,
    )

    image = tf.clip_by_value(
        image,
        0.0,
        255.0,
    )

    return image, box


# ============================================================
# CREATE TF.DATA DATASET
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
            buffer_size=min(
                len(paths),
                3500,
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

test_dataset = create_dataset(
    test_paths,
    test_boxes,
    training=False,
)


# ============================================================
# BOUNDING BOX IoU
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


    true_width = tf.maximum(
        0.0,
        true_xmax - true_xmin,
    )

    true_height = tf.maximum(
        0.0,
        true_ymax - true_ymin,
    )

    pred_width = tf.maximum(
        0.0,
        pred_xmax - pred_xmin,
    )

    pred_height = tf.maximum(
        0.0,
        pred_ymax - pred_ymin,
    )


    true_area = (
        true_width
        * true_height
    )

    pred_area = (
        pred_width
        * pred_height
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
# IoU LOSS
# ============================================================

def iou_loss(
    y_true,
    y_pred,
):

    return (
        1.0
        - bbox_iou(
            y_true,
            y_pred,
        )
    )


# ============================================================
# COMBINED LOCALIZATION LOSS
# ============================================================

huber_loss = (
    tf.keras.losses.Huber(
        delta=0.10
    )
)


def localization_loss(
    y_true,
    y_pred,
):

    coordinate_loss = (
        huber_loss(
            y_true,
            y_pred,
        )
    )

    overlap_loss = (
        iou_loss(
            y_true,
            y_pred,
        )
    )

    return (
        coordinate_loss
        + (
            0.50
            * overlap_loss
        )
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    backbone = (
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

    backbone.trainable = False


    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        name="ultrasound_input",
    )


    x = backbone(
        inputs,
        training=False,
    )


    x = (
        tf.keras.layers
        .GlobalAveragePooling2D()
        (x)
    )


    x = tf.keras.layers.Dense(
        512,
        activation="relu",
    )(x)


    x = tf.keras.layers.Dropout(
        0.30
    )(x)


    x = tf.keras.layers.Dense(
        256,
        activation="relu",
    )(x)


    x = tf.keras.layers.Dropout(
        0.20
    )(x)


    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)


    outputs = tf.keras.layers.Dense(
        4,
        activation="sigmoid",
        name="bounding_box",
    )(x)


    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="TN5000_Lesion_Localizer_V2",
    )

    return model, backbone


model, backbone = build_model()


# ============================================================
# MODEL PATHS
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR
    / "tn5000_localizer_v2_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "tn5000_localizer_v2_final.keras"
)


# ============================================================
# CALLBACK FACTORY
# ============================================================

def make_callbacks():

    return [

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
            patience=5,
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
# STAGE 1
# TRAIN LOCALIZATION HEAD
# ============================================================

model.compile(
    optimizer=(
        tf.keras.optimizers.Adam(
            learning_rate=5e-4,
        )
    ),
    loss=localization_loss,
    metrics=[
        bbox_iou,
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        ),
    ],
)


model.summary()


print()
print("=" * 60)
print("STAGE 1 - TRAINING LOCALIZATION HEAD")
print("=" * 60)
print()


history_stage1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=STAGE1_EPOCHS,
    callbacks=make_callbacks(),
)


# ============================================================
# STAGE 2
# FINE-TUNE TOP EFFICIENTNET LAYERS
# ============================================================

print()
print("=" * 60)
print("STAGE 2 - FINE-TUNING EFFICIENTNET")
print("=" * 60)
print()


backbone.trainable = True


# Freeze most of EfficientNet.
for layer in backbone.layers[:-30]:

    layer.trainable = False


# Keep BatchNormalization frozen for stable fine-tuning.
for layer in backbone.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization,
    ):

        layer.trainable = False


model.compile(
    optimizer=(
        tf.keras.optimizers.Adam(
            learning_rate=1e-5,
        )
    ),
    loss=localization_loss,
    metrics=[
        bbox_iou,
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        ),
    ],
)


history_stage2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=STAGE2_EPOCHS,
    callbacks=make_callbacks(),
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 60)
print("EVALUATING FINAL MODEL ON TEST SET")
print("=" * 60)
print()


test_results = model.evaluate(
    test_dataset,
    verbose=1,
    return_dict=True,
)


print()
print("TEST RESULTS")
print("-" * 60)

for metric_name, metric_value in test_results.items():

    print(
        f"{metric_name}: "
        f"{float(metric_value):.4f}"
    )


# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

all_keys = set(
    history_stage1.history.keys()
) | set(
    history_stage2.history.keys()
)


combined_history = {}


for key in all_keys:

    stage1_values = (
        history_stage1.history.get(
            key,
            [],
        )
    )

    stage2_values = (
        history_stage2.history.get(
            key,
            [],
        )
    )

    combined_history[key] = (
        stage1_values
        + stage2_values
    )


history_data = {

    key: [
        float(value)
        for value in values
    ]

    for key, values
    in combined_history.items()
}


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

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
# SAVE TEST RESULTS
# ============================================================

test_results_json = {

    key: float(value)

    for key, value
    in test_results.items()
}


with (
    RESULTS_DIR
    / "test_results.json"
).open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        test_results_json,
        file,
        indent=4,
    )


# ============================================================
# SAVE IoU GRAPH
# ============================================================

if (
    "bbox_iou"
    in combined_history
    and
    "val_bbox_iou"
    in combined_history
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        combined_history[
            "bbox_iou"
        ],
        label="Training IoU",
    )

    plt.plot(
        combined_history[
            "val_bbox_iou"
        ],
        label="Validation IoU",
    )

    stage1_length = len(
        history_stage1.history.get(
            "bbox_iou",
            [],
        )
    )

    if stage1_length > 0:

        plt.axvline(
            x=stage1_length - 0.5,
            linestyle="--",
            label="Fine-tuning begins",
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Bounding-box IoU"
    )

    plt.title(
        "TN5000 Ultrasound Localization V2"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "bbox_iou_curve.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("ULTRASOUND LOCALIZATION V2 COMPLETE")
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

print()