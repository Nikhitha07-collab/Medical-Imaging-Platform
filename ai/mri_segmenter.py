from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mri"
    / "figshare_mri_segmenter_best.keras"
)

IMAGE_SIZE = 128
MASK_THRESHOLD = 0.50


# ============================================================
# CUSTOM METRICS / LOSS
# ============================================================

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


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred,
    )

    dice_loss = (
        1.0
        - dice_coefficient(
            y_true,
            y_pred,
        )
    )

    return (
        tf.reduce_mean(bce)
        + dice_loss
    )


# ============================================================
# MRI SEGMENTER
# ============================================================

class MRISegmenter:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "MRI segmentation model was not found:\n"
                f"{MODEL_PATH}"
            )

        print(
            "Loading MRI segmentation model:"
        )

        print(
            MODEL_PATH
        )

        self.model = (
            tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    "dice_coefficient":
                        dice_coefficient,
                    "iou_metric":
                        iou_metric,
                    "combined_loss":
                        combined_loss,
                },
                compile=False,
            )
        )

        print(
            "MRI segmentation model loaded successfully."
        )


    # ========================================================
    # PREPROCESS MRI
    # ========================================================

    def _prepare_image(
        self,
        image_path,
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"MRI image not found:\n"
                f"{image_path}"
            )

        original_image = Image.open(
            image_path
        ).convert("RGB")

        original_array = np.array(
            original_image
        )

        grayscale = (
            original_image.convert(
                "L"
            )
        )

        resized = grayscale.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            resample=(
                Image.Resampling.BILINEAR
            ),
        )

        input_array = np.array(
            resized,
            dtype=np.float32,
        )

        input_array = (
            input_array / 255.0
        )

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


    # ========================================================
    # CLEAN SMALL REGIONS
    # ========================================================

    def _clean_mask(
        self,
        mask,
    ):

        mask = (
            mask.astype(
                np.uint8
            )
        )

        (
            number_labels,
            labels,
            stats,
            _
        ) = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        cleaned_mask = np.zeros_like(
            mask
        )

        image_area = (
            mask.shape[0]
            * mask.shape[1]
        )

        minimum_area = max(
            20,
            int(
                image_area
                * 0.0005
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

            if (
                component_area
                >= minimum_area
            ):

                cleaned_mask[
                    labels
                    == label_index
                ] = 1

        return cleaned_mask


    # ========================================================
    # SEGMENT MRI
    # ========================================================

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


        # ----------------------------------------------------
        # RUN U-NET
        # ----------------------------------------------------

        prediction = (
            self.model.predict(
                input_tensor,
                verbose=0,
            )[0, :, :, 0]
        )


        binary_mask = (
            prediction
            >= MASK_THRESHOLD
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # RESIZE MASK TO ORIGINAL IMAGE SIZE
        # ----------------------------------------------------

        resized_mask = cv2.resize(
            binary_mask,
            (
                original_width,
                original_height,
            ),
            interpolation=(
                cv2.INTER_NEAREST
            ),
        )


        cleaned_mask = (
            self._clean_mask(
                resized_mask
            )
        )


        # ----------------------------------------------------
        # FIND REGIONS
        # ----------------------------------------------------

        contours, _ = (
            cv2.findContours(
                cleaned_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )


        # ----------------------------------------------------
        # CREATE MASK IMAGE
        # ----------------------------------------------------

        mask_image = (
            cleaned_mask
            * 255
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # CREATE HIGHLIGHT OVERLAY
        # ----------------------------------------------------

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

            highlight_color = (
                np.array(
                    [
                        255,
                        60,
                        60,
                    ],
                    dtype=np.float32,
                )
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


        # ----------------------------------------------------
        # DRAW REGION CONTOURS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        bounding_box = None

        if contours:

            all_points = (
                np.vstack(
                    contours
                )
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


        # ----------------------------------------------------
        # REGION COVERAGE
        # ----------------------------------------------------

        lesion_pixels = int(
            np.count_nonzero(
                cleaned_mask
            )
        )

        total_pixels = int(
            cleaned_mask.size
        )

        if total_pixels > 0:

            lesion_coverage = (
                lesion_pixels
                / total_pixels
            )

        else:

            lesion_coverage = 0.0


        # ----------------------------------------------------
        # SEGMENTATION PROBABILITY
        # ----------------------------------------------------

        probability_resized = (
            cv2.resize(
                prediction,
                (
                    original_width,
                    original_height,
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                ),
            )
        )


        if lesion_pixels > 0:

            region_values = (
                probability_resized[
                    cleaned_mask > 0
                ]
            )

            mean_region_probability = float(
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

            mean_region_probability = 0.0

            maximum_probability = float(
                np.max(
                    prediction
                )
            )


        # ----------------------------------------------------
        # RETURN RESULTS
        # ----------------------------------------------------

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
                float(
                    mean_region_probability
                ),

            "maximum_probability":
                float(
                    maximum_probability
                ),

            "threshold":
                MASK_THRESHOLD,
        }


# ============================================================
# LAZY GLOBAL INSTANCE
# ============================================================

_segmenter_instance = None


def get_mri_segmenter():

    global _segmenter_instance

    if (
        _segmenter_instance
        is None
    ):

        _segmenter_instance = (
            MRISegmenter()
        )

    return (
        _segmenter_instance
    )


def segment_mri(
    image_path,
):

    segmenter = (
        get_mri_segmenter()
    )

    return (
        segmenter.segment(
            image_path
        )
    )