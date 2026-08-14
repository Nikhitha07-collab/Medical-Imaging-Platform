from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ultrasound"
    / "tn5000_localizer_v2_best.keras"
)

IMAGE_SIZE = 224


def bbox_iou(y_true, y_pred):

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


def iou_loss(y_true, y_pred):

    return (
        1.0
        - bbox_iou(
            y_true,
            y_pred,
        )
    )


huber_loss = tf.keras.losses.Huber(
    delta=0.10
)


def localization_loss(
    y_true,
    y_pred,
):

    coordinate_loss = huber_loss(
        y_true,
        y_pred,
    )

    overlap_loss = iou_loss(
        y_true,
        y_pred,
    )

    return (
        coordinate_loss
        + 0.50
        * overlap_loss
    )


class UltrasoundLocalizer:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Ultrasound localization model not found:\n"
                f"{MODEL_PATH}"
            )

        print(
            "Loading ultrasound lesion localization model:"
        )

        print(
            MODEL_PATH
        )

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "bbox_iou":
                    bbox_iou,
                "iou_loss":
                    iou_loss,
                "localization_loss":
                    localization_loss,
            },
            compile=False,
        )

        print(
            "Ultrasound localization model loaded successfully."
        )


    def localize(
        self,
        image_path,
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Ultrasound image not found:\n{image_path}"
            )


        original_image = Image.open(
            image_path
        ).convert("RGB")

        original_array = np.asarray(
            original_image
        )

        original_height = int(
            original_array.shape[0]
        )

        original_width = int(
            original_array.shape[1]
        )


        resized_image = original_image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            resample=Image.Resampling.BILINEAR,
        )


        # EfficientNetB0 contains its own input rescaling,
        # so keep values in 0-255 range.
        input_array = np.asarray(
            resized_image,
            dtype=np.float32,
        )


        input_tensor = np.expand_dims(
            input_array,
            axis=0,
        )


        prediction = self.model.predict(
            input_tensor,
            verbose=0,
        )[0]


        prediction = np.clip(
            prediction,
            0.0,
            1.0,
        )


        xmin_norm = float(
            prediction[0]
        )

        ymin_norm = float(
            prediction[1]
        )

        xmax_norm = float(
            prediction[2]
        )

        ymax_norm = float(
            prediction[3]
        )


        # Ensure correct coordinate order.
        xmin_norm, xmax_norm = sorted(
            [
                xmin_norm,
                xmax_norm,
            ]
        )

        ymin_norm, ymax_norm = sorted(
            [
                ymin_norm,
                ymax_norm,
            ]
        )


        xmin = int(
            round(
                xmin_norm
                * original_width
            )
        )

        ymin = int(
            round(
                ymin_norm
                * original_height
            )
        )

        xmax = int(
            round(
                xmax_norm
                * original_width
            )
        )

        ymax = int(
            round(
                ymax_norm
                * original_height
            )
        )


        xmin = max(
            0,
            min(
                xmin,
                original_width - 1,
            ),
        )

        ymin = max(
            0,
            min(
                ymin,
                original_height - 1,
            ),
        )

        xmax = max(
            xmin + 1,
            min(
                xmax,
                original_width,
            ),
        )

        ymax = max(
            ymin + 1,
            min(
                ymax,
                original_height,
            ),
        )


        box_width = (
            xmax - xmin
        )

        box_height = (
            ymax - ymin
        )


        overlay = original_array.copy()


        # Add transparent red fill inside predicted lesion box.
        overlay_float = overlay.astype(
            np.float32
        )

        region = overlay_float[
            ymin:ymax,
            xmin:xmax,
        ]

        if region.size > 0:

            highlight_color = np.array(
                [
                    255,
                    70,
                    40,
                ],
                dtype=np.float32,
            )

            overlay_float[
                ymin:ymax,
                xmin:xmax,
            ] = (
                region * 0.70
                + highlight_color * 0.30
            )


        overlay = np.clip(
            overlay_float,
            0,
            255,
        ).astype(
            np.uint8
        )


        # Draw bounding box.
        cv2.rectangle(
            overlay,
            (
                xmin,
                ymin,
            ),
            (
                xmax,
                ymax,
            ),
            (
                255,
                255,
                255,
            ),
            3,
        )


        # Calculate predicted box coverage.
        box_area = (
            box_width
            * box_height
        )

        image_area = (
            original_width
            * original_height
        )

        coverage = (
            box_area / image_area
            if image_area > 0
            else 0.0
        )


        return {

            "overlay":
                overlay,

            "bounding_box":
                {
                    "x":
                        xmin,

                    "y":
                        ymin,

                    "width":
                        box_width,

                    "height":
                        box_height,

                    "x2":
                        xmax,

                    "y2":
                        ymax,
                },

            "normalized_box":
                {
                    "xmin":
                        xmin_norm,

                    "ymin":
                        ymin_norm,

                    "xmax":
                        xmax_norm,

                    "ymax":
                        ymax_norm,
                },

            "box_coverage":
                float(
                    coverage
                ),
        }


_localizer_instance = None


def get_ultrasound_localizer():

    global _localizer_instance

    if _localizer_instance is None:

        _localizer_instance = (
            UltrasoundLocalizer()
        )

    return _localizer_instance


def localize_ultrasound(
    image_path,
):

    return (
        get_ultrasound_localizer()
        .localize(
            image_path
        )
    )