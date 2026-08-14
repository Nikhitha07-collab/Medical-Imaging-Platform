from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ct"
    / "ct_infection_segmenter_v2_best.keras"
)

IMAGE_SIZE = 128
MASK_THRESHOLD = 0.50


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


class CTSegmenter:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "CT segmentation model not found:\n"
                f"{MODEL_PATH}"
            )

        print(
            "Loading CT infection segmentation model:"
        )

        print(
            MODEL_PATH
        )

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "dice_coefficient":
                    dice_coefficient,
                "iou_metric":
                    iou_metric,
                "tversky_index":
                    tversky_index,
                "focal_tversky_loss":
                    focal_tversky_loss,
                "combined_focal_tversky_loss":
                    combined_focal_tversky_loss,
            },
            compile=False,
        )

        print(
            "CT segmentation model loaded successfully."
        )


    def _prepare_image(
        self,
        image_path,
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"CT image not found:\n"
                f"{image_path}"
            )

        original = Image.open(
            image_path
        ).convert("RGB")

        original_array = np.asarray(
            original
        )

        grayscale = original.convert(
            "L"
        )

        resized = grayscale.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        input_array = np.asarray(
            resized,
            dtype=np.float32,
        ) / 255.0

        input_tensor = (
            input_array[
                None,
                ...,
                None,
            ]
        )

        return (
            original_array,
            input_tensor,
        )


    def _clean_mask(
        self,
        mask,
    ):

        mask = mask.astype(
            np.uint8
        )

        (
            number_labels,
            labels,
            stats,
            _
        ) = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        cleaned = np.zeros_like(
            mask
        )

        image_area = (
            mask.shape[0]
            * mask.shape[1]
        )

        minimum_area = max(
            30,
            int(
                image_area
                * 0.0008
            ),
        )

        for label_index in range(
            1,
            number_labels,
        ):

            component_area = stats[
                label_index,
                cv2.CC_STAT_AREA,
            ]

            if component_area >= minimum_area:
                cleaned[
                    labels == label_index
                ] = 1

        return cleaned


    def segment(
        self,
        image_path,
    ):

        (
            original_array,
            input_tensor,
        ) = self._prepare_image(
            image_path
        )

        original_height = (
            original_array.shape[0]
        )

        original_width = (
            original_array.shape[1]
        )


        prediction = self.model.predict(
            input_tensor,
            verbose=0,
        )[0, :, :, 0]


        binary_mask = (
            prediction
            >= MASK_THRESHOLD
        ).astype(
            np.uint8
        )


        resized_mask = cv2.resize(
            binary_mask,
            (
                original_width,
                original_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )


        cleaned_mask = self._clean_mask(
            resized_mask
        )


        contours, _ = cv2.findContours(
            cleaned_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )


        mask_image = (
            cleaned_mask * 255
        ).astype(
            np.uint8
        )


        overlay = (
            original_array.copy()
        )

        mask_pixels = (
            cleaned_mask > 0
        )


        if np.any(
            mask_pixels
        ):

            overlay_float = (
                overlay.astype(
                    np.float32
                )
            )

            highlight_color = np.array(
                [
                    255,
                    70,
                    40,
                ],
                dtype=np.float32,
            )

            overlay_float[
                mask_pixels
            ] = (
                overlay_float[
                    mask_pixels
                ]
                * 0.45
                +
                highlight_color
                * 0.55
            )

            overlay = np.clip(
                overlay_float,
                0,
                255,
            ).astype(
                np.uint8
            )


        if contours:

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )


        bounding_box = None

        if contours:

            all_points = np.vstack(
                contours
            )

            (
                x,
                y,
                width,
                height,
            ) = cv2.boundingRect(
                all_points
            )

            bounding_box = {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "x2": int(
                    x + width
                ),
                "y2": int(
                    y + height
                ),
            }

            cv2.rectangle(
                overlay,
                (
                    int(x),
                    int(y),
                ),
                (
                    int(
                        x + width
                    ),
                    int(
                        y + height
                    ),
                ),
                (
                    255,
                    255,
                    255,
                ),
                3,
            )


        lesion_pixels = int(
            np.count_nonzero(
                cleaned_mask
            )
        )

        total_pixels = int(
            cleaned_mask.size
        )

        lesion_coverage = (
            lesion_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )


        probability_resized = cv2.resize(
            prediction,
            (
                original_width,
                original_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )


        if lesion_pixels > 0:

            region_values = (
                probability_resized[
                    cleaned_mask > 0
                ]
            )

            mean_probability = float(
                np.mean(
                    region_values
                )
            )

            maximum_probability = float(
                np.max(
                    region_values
                )
            )

        else:

            mean_probability = 0.0

            maximum_probability = float(
                np.max(
                    prediction
                )
            )


        return {

            "mask":
                mask_image,

            "overlay":
                overlay,

            "bounding_box":
                bounding_box,

            "has_detected_region":
                (
                    bounding_box
                    is not None
                ),

            "lesion_pixels":
                lesion_pixels,

            "lesion_coverage":
                float(
                    lesion_coverage
                ),

            "mean_region_probability":
                mean_probability,

            "maximum_probability":
                maximum_probability,

            "threshold":
                MASK_THRESHOLD,
        }


_segmenter_instance = None


def get_ct_segmenter():

    global _segmenter_instance

    if _segmenter_instance is None:
        _segmenter_instance = CTSegmenter()

    return _segmenter_instance


def segment_ct(
    image_path,
):

    return get_ct_segmenter().segment(
        image_path
    )