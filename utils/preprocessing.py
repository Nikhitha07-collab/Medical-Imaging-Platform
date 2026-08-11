import numpy as np
from pydicom.dataset import FileDataset


def get_pixel_array(dataset: FileDataset) -> np.ndarray:
    """Decode DICOM PixelData into a NumPy array."""

    try:
        image = dataset.pixel_array
    except Exception as error:
        raise ValueError(
            f"Unable to decode DICOM pixel data: {error}"
        ) from error

    image = np.asarray(image)

    number_of_frames = int(
        getattr(dataset, "NumberOfFrames", 1)
    )

    samples_per_pixel = int(
        getattr(dataset, "SamplesPerPixel", 1)
    )

    # Multi-frame grayscale DICOM:
    # (frames, rows, columns)
    if (
        number_of_frames > 1
        and samples_per_pixel == 1
        and image.ndim >= 3
    ):
        image = image[0]

    # Multi-frame color DICOM:
    # (frames, rows, columns, channels)
    if (
        number_of_frames > 1
        and samples_per_pixel > 1
        and image.ndim == 4
    ):
        image = image[0]

    return image


def apply_modality_rescale(
    image: np.ndarray,
    dataset: FileDataset,
) -> np.ndarray:
    """Apply DICOM rescale slope and intercept."""

    slope = float(
        getattr(dataset, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(dataset, "RescaleIntercept", 0.0)
    )

    return (
        image.astype(np.float32) * slope
        + intercept
    )


def normalize_image(
    image: np.ndarray,
) -> np.ndarray:
    """Normalize image intensity into the 0-255 range."""

    image = image.astype(np.float32)

    minimum = float(np.min(image))
    maximum = float(np.max(image))

    if maximum == minimum:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    normalized = (
        image - minimum
    ) / (
        maximum - minimum
    )

    normalized *= 255.0

    return normalized.astype(np.uint8)


def apply_window(
    image: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Apply Window Center and Window Width."""

    if window_width <= 0:
        raise ValueError(
            "Window Width must be greater than zero."
        )

    lower = (
        window_center
        - window_width / 2.0
    )

    upper = (
        window_center
        + window_width / 2.0
    )

    windowed = np.clip(
        image,
        lower,
        upper,
    )

    windowed = (
        windowed - lower
    ) / (
        upper - lower
    )

    windowed *= 255.0

    return windowed.astype(np.uint8)


def dicom_to_image(
    dataset: FileDataset,
    window_center: float | None = None,
    window_width: float | None = None,
) -> np.ndarray:
    """Convert a DICOM dataset into a display-ready image."""

    image = get_pixel_array(dataset)

    photometric = str(
        getattr(
            dataset,
            "PhotometricInterpretation",
            "",
        )
    ).upper()

    samples_per_pixel = int(
        getattr(
            dataset,
            "SamplesPerPixel",
            1,
        )
    )

    is_color = (
        samples_per_pixel > 1
        or photometric.startswith("RGB")
        or photometric.startswith("YBR")
    )

    # Color Ultrasound or other RGB DICOM
    if is_color:
        if image.dtype != np.uint8:
            image = normalize_image(image)

        return image.astype(np.uint8)

    # Grayscale CT/MRI/Ultrasound
    image = apply_modality_rescale(
        image,
        dataset,
    )

    if (
        window_center is not None
        and window_width is not None
    ):
        display_image = apply_window(
            image,
            window_center,
            window_width,
        )

    else:
        display_image = normalize_image(
            image
        )

    # MONOCHROME1 means low values should appear white.
    if photometric == "MONOCHROME1":
        display_image = (
            255 - display_image
        )

    return display_image.astype(np.uint8)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Apply display brightness and contrast."""

    adjusted = image.astype(
        np.float32
    )

    adjusted = (
        adjusted * contrast
        + brightness
    )

    adjusted = np.clip(
        adjusted,
        0,
        255,
    )

    return adjusted.astype(np.uint8)