from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


def is_standard_image(
    file_path: Path,
) -> bool:
    """Return True for supported PNG/JPG/JPEG files."""

    return (
        file_path.suffix.lower()
        in SUPPORTED_IMAGE_EXTENSIONS
    )


def load_standard_image(
    file_path: Path,
) -> np.ndarray:
    """Load a PNG/JPG/JPEG file as a NumPy image."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {file_path}"
        )

    if not is_standard_image(file_path):
        raise ValueError(
            "Unsupported standard image format."
        )

    try:
        image = Image.open(file_path)

        if image.mode not in {
            "L",
            "RGB",
            "RGBA",
        }:
            image = image.convert("RGB")

        if image.mode == "RGBA":
            image = image.convert("RGB")

        return np.asarray(image)

    except Exception as error:
        raise ValueError(
            f"Unable to load image: {error}"
        ) from error