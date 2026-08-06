import numpy as np
from pydicom.dataset import FileDataset


def dicom_to_image(dataset: FileDataset) -> np.ndarray:
    """Convert DICOM pixel data into an 8-bit display image."""

    try:
        image = dataset.pixel_array.astype(np.float32)
    except Exception as error:
        raise ValueError(
            f"Unable to decode DICOM pixels: {error}"
        ) from error

    if image.ndim > 2:
        image = image[0]

    minimum = float(image.min())
    maximum = float(image.max())

    if maximum == minimum:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - minimum) / (maximum - minimum)
    image = image * 255.0

    photometric = getattr(
        dataset,
        "PhotometricInterpretation",
        "",
    )

    if photometric == "MONOCHROME1":
        image = 255.0 - image

    return image.astype(np.uint8)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Apply brightness and contrast adjustments."""

    adjusted = image.astype(np.float32)

    adjusted = adjusted * contrast
    adjusted = adjusted + brightness
    adjusted = np.clip(adjusted, 0, 255)

    return adjusted.astype(np.uint8)