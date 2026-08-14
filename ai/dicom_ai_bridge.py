from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pydicom
from PIL import Image


# ============================================================
# DICOM AI BRIDGE
# ============================================================


def _first_number(value: Any) -> float | None:
    """
    Convert a DICOM value such as WindowCenter or WindowWidth
    into one float.

    Some DICOM files store multiple values.
    """

    if value is None:
        return None

    try:
        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            return float(value[0])

        # pydicom MultiValue-like objects
        if hasattr(value, "__len__") and not isinstance(
            value,
            (
                str,
                bytes,
            ),
        ):
            try:
                return float(value[0])
            except Exception:
                pass

        return float(value)

    except Exception:
        return None


# ============================================================
# NORMALIZATION
# ============================================================


def _percentile_normalize(
    array: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:

    image = np.asarray(
        array,
        dtype=np.float32,
    )

    finite = image[
        np.isfinite(image)
    ]

    if finite.size == 0:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    low = float(
        np.percentile(
            finite,
            lower_percentile,
        )
    )

    high = float(
        np.percentile(
            finite,
            upper_percentile,
        )
    )

    if high <= low:
        low = float(
            np.min(finite)
        )

        high = float(
            np.max(finite)
        )

    if high <= low:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    image = np.clip(
        image,
        low,
        high,
    )

    image = (
        image - low
    ) / (
        high - low
    )

    image = np.clip(
        image * 255.0,
        0.0,
        255.0,
    )

    return image.astype(
        np.uint8
    )


# ============================================================
# CT WINDOWING
# ============================================================


def _ct_to_uint8(
    dataset,
    array: np.ndarray,
) -> np.ndarray:

    image = np.asarray(
        array,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Convert stored pixel values to HU
    # --------------------------------------------------------

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

    image = (
        image * slope
        + intercept
    )

    # --------------------------------------------------------
    # Use DICOM window when available
    # --------------------------------------------------------

    center = _first_number(
        getattr(
            dataset,
            "WindowCenter",
            None,
        )
    )

    width = _first_number(
        getattr(
            dataset,
            "WindowWidth",
            None,
        )
    )

    if (
        center is not None
        and width is not None
        and width > 1.0
    ):

        low = (
            center
            - width / 2.0
        )

        high = (
            center
            + width / 2.0
        )

        image = np.clip(
            image,
            low,
            high,
        )

        image = (
            image - low
        ) / (
            high - low
        )

        image = np.clip(
            image * 255.0,
            0.0,
            255.0,
        ).astype(
            np.uint8
        )

    else:

        # Fallback if DICOM has no window values.
        image = _percentile_normalize(
            image,
            1.0,
            99.0,
        )

    return image


# ============================================================
# FRAME EXTRACTION
# ============================================================


def _extract_frame(
    dataset,
    frame_index: int = 0,
) -> np.ndarray:

    pixels = np.asarray(
        dataset.pixel_array
    )

    samples_per_pixel = int(
        getattr(
            dataset,
            "SamplesPerPixel",
            1,
        )
        or 1
    )

    number_of_frames = int(
        getattr(
            dataset,
            "NumberOfFrames",
            1,
        )
        or 1
    )

    # --------------------------------------------------------
    # Single grayscale image
    # H x W
    # --------------------------------------------------------

    if pixels.ndim == 2:
        return pixels


    # --------------------------------------------------------
    # Color image
    # H x W x 3 or H x W x 4
    # --------------------------------------------------------

    if (
        pixels.ndim == 3
        and samples_per_pixel > 1
        and pixels.shape[-1] in (
            3,
            4,
        )
    ):
        return pixels


    # --------------------------------------------------------
    # Multi-frame grayscale
    # Frames x H x W
    # --------------------------------------------------------

    if (
        pixels.ndim == 3
        and number_of_frames > 1
    ):

        index = max(
            0,
            min(
                int(frame_index),
                pixels.shape[0] - 1,
            ),
        )

        return pixels[index]


    # --------------------------------------------------------
    # Multi-frame color
    # Frames x H x W x Channels
    # --------------------------------------------------------

    if pixels.ndim == 4:

        index = max(
            0,
            min(
                int(frame_index),
                pixels.shape[0] - 1,
            ),
        )

        return pixels[index]


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    squeezed = np.squeeze(
        pixels
    )

    if squeezed.ndim in (
        2,
        3,
    ):
        return squeezed

    raise ValueError(
        "Unsupported DICOM pixel-array shape: "
        f"{pixels.shape}"
    )


# ============================================================
# COLOR NORMALIZATION
# ============================================================


def _color_to_uint8(
    array: np.ndarray,
) -> np.ndarray:

    image = np.asarray(
        array
    )

    if image.dtype == np.uint8:
        result = image.copy()

    else:

        result = _percentile_normalize(
            image,
            1.0,
            99.0,
        )

    if (
        result.ndim == 3
        and result.shape[-1] == 4
    ):
        result = result[
            ...,
            :3,
        ]

    return result


# ============================================================
# DICOM → DISPLAY / MODEL IMAGE
# ============================================================


def dicom_to_ai_array(
    dicom_path: str | Path,
    frame_index: int = 0,
    modality_override: str | None = None,
) -> dict[str, Any]:

    dicom_path = Path(
        dicom_path
    )

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"DICOM file not found:\n{dicom_path}"
        )

    dataset = pydicom.dcmread(
        str(dicom_path)
    )

    modality = (
        modality_override
        or getattr(
            dataset,
            "Modality",
            "",
        )
        or ""
    )

    modality = str(
        modality
    ).upper()

    frame = _extract_frame(
        dataset,
        frame_index=frame_index,
    )

    photometric = str(
        getattr(
            dataset,
            "PhotometricInterpretation",
            "",
        )
        or ""
    ).upper()

    # ========================================================
    # CT
    # ========================================================

    if modality == "CT":

        image = _ct_to_uint8(
            dataset,
            frame,
        )

        if photometric == "MONOCHROME1":
            image = (
                255
                - image
            )

        rgb = np.stack(
            [
                image,
                image,
                image,
            ],
            axis=-1,
        )

    # ========================================================
    # MRI
    # ========================================================

    elif modality in (
        "MR",
        "MRI",
    ):

        image = _percentile_normalize(
            frame,
            1.0,
            99.0,
        )

        if photometric == "MONOCHROME1":
            image = (
                255
                - image
            )

        if image.ndim == 2:

            rgb = np.stack(
                [
                    image,
                    image,
                    image,
                ],
                axis=-1,
            )

        else:

            rgb = _color_to_uint8(
                image
            )

    # ========================================================
    # ULTRASOUND
    # ========================================================

    elif modality in (
        "US",
        "ULTRASOUND",
    ):

        if frame.ndim == 2:

            image = _percentile_normalize(
                frame,
                1.0,
                99.0,
            )

            if photometric == "MONOCHROME1":
                image = (
                    255
                    - image
                )

            rgb = np.stack(
                [
                    image,
                    image,
                    image,
                ],
                axis=-1,
            )

        else:

            rgb = _color_to_uint8(
                frame
            )

    # ========================================================
    # UNKNOWN MODALITY
    # ========================================================

    else:

        if frame.ndim == 2:

            image = _percentile_normalize(
                frame,
                1.0,
                99.0,
            )

            if photometric == "MONOCHROME1":
                image = (
                    255
                    - image
                )

            rgb = np.stack(
                [
                    image,
                    image,
                    image,
                ],
                axis=-1,
            )

        else:

            rgb = _color_to_uint8(
                frame
            )

    rgb = np.clip(
        rgb,
        0,
        255,
    ).astype(
        np.uint8
    )

    return {
        "image": rgb,
        "dataset": dataset,
        "modality": modality,
        "frame_index": int(
            frame_index
        ),
        "width": int(
            rgb.shape[1]
        ),
        "height": int(
            rgb.shape[0]
        ),
        "photometric_interpretation": photometric,
    }


# ============================================================
# TEMPORARY MODEL PNG
# ============================================================


def dicom_to_temporary_png(
    dicom_path: str | Path,
    frame_index: int = 0,
    modality_override: str | None = None,
) -> dict[str, Any]:

    result = dicom_to_ai_array(
        dicom_path=dicom_path,
        frame_index=frame_index,
        modality_override=modality_override,
    )

    image = Image.fromarray(
        result["image"]
    )

    temp_file = NamedTemporaryFile(
        suffix=".png",
        delete=False,
    )

    temp_path = Path(
        temp_file.name
    )

    temp_file.close()

    image.save(
        temp_path,
        format="PNG",
    )

    result["temporary_png"] = (
        temp_path
    )

    return result


# ============================================================
# DELETE TEMP FILE
# ============================================================


def cleanup_temporary_ai_image(
    path: str | Path | None,
) -> None:

    if path is None:
        return

    try:

        temp_path = Path(
            path
        )

        if temp_path.exists():
            temp_path.unlink()

    except Exception:
        pass