from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom


def check_ct_lung_content(
    dicom_path: str | Path,
) -> dict[str, Any]:
    """
    Research/demo CT slice suitability check.

    The goal is to decide whether enough lung-like anatomy is
    visible before allowing the chest/COVID CT model to run.

    This is not a clinical anatomy classifier.
    """

    dicom_path = Path(
        dicom_path
    )

    if not dicom_path.exists():
        return {
            "supported": False,
            "reason": "DICOM file does not exist.",
        }

    try:
        dataset = pydicom.dcmread(
            str(dicom_path)
        )

        pixels = np.asarray(
            dataset.pixel_array,
            dtype=np.float32,
        )

    except Exception as error:
        return {
            "supported": False,
            "reason": (
                f"Unable to read CT pixels: {error}"
            ),
        }

    pixels = np.squeeze(
        pixels
    )

    if pixels.ndim != 2:
        return {
            "supported": False,
            "reason": (
                "Current CT frame is not a single "
                "2D axial image."
            ),
        }

    # ========================================================
    # CONVERT STORED PIXELS TO HOUNSFIELD UNITS
    # ========================================================

    slope = float(
        getattr(
            dataset,
            "RescaleSlope",
            1.0,
        )
        or 1.0
    )

    intercept = float(
        getattr(
            dataset,
            "RescaleIntercept",
            0.0,
        )
        or 0.0
    )

    hu = (
        pixels * slope
        + intercept
    )

    image_height, image_width = (
        hu.shape
    )


    # ========================================================
    # CENTRAL SEARCH REGION
    #
    # Avoid image borders and most of the CT table.
    # ========================================================

    x1 = int(
        image_width * 0.12
    )

    x2 = int(
        image_width * 0.88
    )

    y1 = int(
        image_height * 0.12
    )

    y2 = int(
        image_height * 0.82
    )

    roi_hu = hu[
        y1:y2,
        x1:x2,
    ]


    # ========================================================
    # LUNG-LIKE HU RANGE
    #
    # We are looking for substantial low-density regions
    # consistent with aerated lung.
    # ========================================================

    lung_candidate = (
        (roi_hu < -450)
        & (roi_hu > -1000)
    ).astype(
        np.uint8
    )


    # ========================================================
    # REMOVE SMALL NOISE
    # ========================================================

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    lung_candidate = cv2.morphologyEx(
        lung_candidate,
        cv2.MORPH_OPEN,
        kernel,
    )

    lung_candidate = cv2.morphologyEx(
        lung_candidate,
        cv2.MORPH_CLOSE,
        kernel,
    )


    # ========================================================
    # CONNECTED COMPONENT ANALYSIS
    # ========================================================

    (
        number_labels,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(
        lung_candidate,
        connectivity=8,
    )

    component_areas = []

    for index in range(
        1,
        number_labels,
    ):
        area = int(
            stats[
                index,
                cv2.CC_STAT_AREA,
            ]
        )

        if area > 0:
            component_areas.append(
                area
            )

    component_areas.sort(
        reverse=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    roi_area = float(
        lung_candidate.size
    )

    lung_pixels = float(
        np.count_nonzero(
            lung_candidate
        )
    )

    lung_fraction = (
        lung_pixels / roi_area
        if roi_area > 0
        else 0.0
    )

    largest_area = (
        component_areas[0]
        if len(component_areas) >= 1
        else 0
    )

    second_largest_area = (
        component_areas[1]
        if len(component_areas) >= 2
        else 0
    )

    largest_fraction = (
        largest_area / roi_area
        if roi_area > 0
        else 0.0
    )

    second_fraction = (
        second_largest_area / roi_area
        if roi_area > 0
        else 0.0
    )


    # ========================================================
    # STRICTER LUNG SUITABILITY DECISION
    #
    # The previous version was too permissive and accepted
    # abdominal bowel gas as lung-like anatomy.
    #
    # Now we require:
    #
    # 1. at least 10% lung-like pixels
    # 2. one substantial low-density region
    # 3. a second meaningful low-density region
    #
    # Requiring the second region helps reject scattered
    # abdominal gas pockets.
    # ========================================================

    enough_lung_pixels = (
        lung_fraction >= 0.10
    )

    first_lung_region = (
        largest_fraction >= 0.035
    )

    second_lung_region = (
        second_fraction >= 0.020
    )

    supported = bool(
        enough_lung_pixels
        and first_lung_region
        and second_lung_region
    )


    # ========================================================
    # RESULT MESSAGE
    # ========================================================

    if supported:

        reason = (
            "Suitable lung-like CT anatomy detected. "
            f"Lung-like fraction="
            f"{lung_fraction * 100:.1f}%. "
            f"Largest region="
            f"{largest_fraction * 100:.1f}%. "
            f"Second region="
            f"{second_fraction * 100:.1f}%."
        )

    else:

        reason = (
            "Suitable lung fields were not detected "
            "in this CT slice. "
            f"Lung-like fraction="
            f"{lung_fraction * 100:.1f}%. "
            f"Largest region="
            f"{largest_fraction * 100:.1f}%. "
            f"Second region="
            f"{second_fraction * 100:.1f}%. "
            "Chest/COVID AI should not run on this slice."
        )


    return {
        "supported": supported,
        "reason": reason,
        "lung_fraction": lung_fraction,
        "largest_component_fraction": (
            largest_fraction
        ),
        "second_component_fraction": (
            second_fraction
        ),
        "image_width": int(
            image_width
        ),
        "image_height": int(
            image_height
        ),
        "component_count": int(
            len(component_areas)
        ),
    }