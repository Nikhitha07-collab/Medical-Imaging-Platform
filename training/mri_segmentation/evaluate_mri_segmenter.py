from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "segmentation"
    / "processed"
)

TEST_CSV = DATA_ROOT / "splits" / "test.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mri"
    / "brain_tumor_segmenter_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "mri_segmentation"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_SIZE = 128
THRESHOLD = 0.5


def dice_coefficient(y_true, y_pred):
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


def iou_metric(y_true, y_pred):
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


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred,
    )

    return (
        tf.reduce_mean(bce)
        + 1.0
        - dice_coefficient(
            y_true,
            y_pred,
        )
    )


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coefficient": dice_coefficient,
        "iou_metric": iou_metric,
        "combined_loss": combined_loss,
    },
)

test_df = pd.read_csv(
    TEST_CSV
)

print()
print("=" * 60)
print("MRI SEGMENTATION TEST EVALUATION")
print("=" * 60)
print()

print(
    f"Test slices: {len(test_df)}"
)

dice_scores = []
iou_scores = []

preview_saved = 0


for index, row in test_df.iterrows():

    image_path = (
        PROJECT_ROOT
        / Path(row["image_path"])
    )

    mask_path = (
        PROJECT_ROOT
        / Path(row["mask_path"])
    )

    image = Image.open(
        image_path
    ).convert("L")

    original_size = image.size

    image_resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = np.array(
        image_resized,
        dtype=np.float32,
    ) / 255.0

    image_input = image_array[
        None,
        ...,
        None,
    ]


    true_mask = Image.open(
        mask_path
    ).convert("L")

    true_mask = true_mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.NEAREST,
    )

    true_mask = (
        np.array(
            true_mask,
            dtype=np.float32,
        )
        > 127
    ).astype(np.float32)


    prediction = model.predict(
        image_input,
        verbose=0,
    )[0, :, :, 0]

    predicted_mask = (
        prediction >= THRESHOLD
    ).astype(np.float32)


    intersection = np.sum(
        true_mask * predicted_mask
    )

    dice = (
        2.0 * intersection + 1e-6
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
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


    dice_scores.append(
        dice
    )

    iou_scores.append(
        iou
    )


    # Save a few visual predictions
    if preview_saved < 10:

        predicted_mask_image = Image.fromarray(
            (
                predicted_mask * 255
            ).astype(np.uint8)
        )

        predicted_mask_image = predicted_mask_image.resize(
            original_size,
            resample=Image.Resampling.NEAREST,
        )

        output_path = (
            OUTPUT_DIR
            / f"prediction_{preview_saved + 1:02d}.png"
        )

        predicted_mask_image.save(
            output_path
        )

        preview_saved += 1


    if (
        index + 1
    ) % 250 == 0:

        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(test_df)}"
        )


mean_dice = float(
    np.mean(
        dice_scores
    )
)

mean_iou = float(
    np.mean(
        iou_scores
    )
)

median_dice = float(
    np.median(
        dice_scores
    )
)

median_iou = float(
    np.median(
        iou_scores
    )
)


print()
print("=" * 60)
print("MRI SEGMENTATION TEST RESULTS")
print("=" * 60)

print()
print(
    f"Mean Dice   : "
    f"{mean_dice:.4f}"
)

print(
    f"Median Dice : "
    f"{median_dice:.4f}"
)

print(
    f"Mean IoU    : "
    f"{mean_iou:.4f}"
)

print(
    f"Median IoU  : "
    f"{median_iou:.4f}"
)

print()
print(
    "Saved prediction masks:"
)

print(
    OUTPUT_DIR
)