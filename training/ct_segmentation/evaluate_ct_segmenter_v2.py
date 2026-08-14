from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


TEST_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "segmentation"
    / "processed"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ct"
    / "ct_infection_segmenter_v2_best.keras"
)

IMAGE_SIZE = 128
THRESHOLD = 0.50


def dice_coefficient(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

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


def iou_metric(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

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


def tversky_index(y_true, y_pred):
    smooth = 1e-6
    alpha = 0.30
    beta = 0.70

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    true_positive = tf.reduce_sum(
        y_true * y_pred
    )

    false_positive = tf.reduce_sum(
        (1.0 - y_true) * y_pred
    )

    false_negative = tf.reduce_sum(
        y_true * (1.0 - y_pred)
    )

    return (
        true_positive + smooth
    ) / (
        true_positive
        + alpha * false_positive
        + beta * false_negative
        + smooth
    )


def focal_tversky_loss(y_true, y_pred):
    gamma = 0.75

    return tf.pow(
        1.0 - tversky_index(
            y_true,
            y_pred,
        ),
        gamma,
    )


binary_crossentropy = (
    tf.keras.losses.BinaryCrossentropy()
)


def combined_focal_tversky_loss(
    y_true,
    y_pred,
):
    return (
        0.25
        * binary_crossentropy(
            y_true,
            y_pred,
        )
        + focal_tversky_loss(
            y_true,
            y_pred,
        )
    )


print()
print("=" * 60)
print("CT SEGMENTATION V2 TEST EVALUATION")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coefficient": dice_coefficient,
        "iou_metric": iou_metric,
        "tversky_index": tversky_index,
        "focal_tversky_loss": focal_tversky_loss,
        "combined_focal_tversky_loss":
            combined_focal_tversky_loss,
    },
    compile=False,
)

print("Model loaded successfully.")

test_df = pd.read_csv(
    TEST_CSV
)

print(
    f"Test slices: {len(test_df)}"
)


dice_scores = []
iou_scores = []


for index, row in test_df.iterrows():

    image_path = (
        PROJECT_ROOT
        / Path(
            row["image_path"]
        )
    )

    mask_path = (
        PROJECT_ROOT
        / Path(
            row["mask_path"]
        )
    )


    image = Image.open(
        image_path
    ).convert("L")

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        resample=Image.Resampling.BILINEAR,
    )

    image_array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    input_tensor = (
        image_array[
            None,
            ...,
            None,
        ]
    )


    true_mask = Image.open(
        mask_path
    ).convert("L")

    true_mask = true_mask.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        resample=Image.Resampling.NEAREST,
    )

    true_mask = (
        np.asarray(
            true_mask
        ) > 127
    ).astype(np.uint8)


    prediction = model.predict(
        input_tensor,
        verbose=0,
    )[0, :, :, 0]

    predicted_mask = (
        prediction >= THRESHOLD
    ).astype(np.uint8)


    intersection = np.sum(
        true_mask * predicted_mask
    )

    dice = (
        2.0 * intersection
        + 1e-6
    ) / (
        np.sum(true_mask)
        + np.sum(predicted_mask)
        + 1e-6
    )

    union = (
        np.sum(true_mask)
        + np.sum(predicted_mask)
        - intersection
    )

    iou = (
        intersection
        + 1e-6
    ) / (
        union
        + 1e-6
    )


    dice_scores.append(
        dice
    )

    iou_scores.append(
        iou
    )


    if (index + 1) % 100 == 0:
        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(test_df)}"
        )


print()
print("=" * 60)
print("CT SEGMENTATION V2 TEST RESULTS")
print("=" * 60)

print(
    f"Mean Dice   : "
    f"{np.mean(dice_scores):.4f}"
)

print(
    f"Median Dice : "
    f"{np.median(dice_scores):.4f}"
)

print(
    f"Mean IoU    : "
    f"{np.mean(iou_scores):.4f}"
)

print(
    f"Median IoU  : "
    f"{np.median(iou_scores):.4f}"
)