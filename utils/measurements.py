from math import sqrt
from typing import Sequence


def pixel_distance(
    point_a: Sequence[float],
    point_b: Sequence[float],
) -> float:
    """Return straight-line distance in pixels."""

    x1, y1 = point_a
    x2, y2 = point_b

    return sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def physical_distance_mm(
    point_a: Sequence[float],
    point_b: Sequence[float],
    pixel_spacing: Sequence[float] | None,
) -> float | None:
    """
    Convert a two-point image distance to millimeters
    when DICOM PixelSpacing is available.
    """

    if not pixel_spacing:
        return None

    if len(pixel_spacing) < 2:
        return None

    row_spacing = float(
        pixel_spacing[0]
    )

    column_spacing = float(
        pixel_spacing[1]
    )

    x1, y1 = point_a
    x2, y2 = point_b

    dx_mm = (
        x2 - x1
    ) * column_spacing

    dy_mm = (
        y2 - y1
    ) * row_spacing

    return sqrt(
        dx_mm ** 2
        + dy_mm ** 2
    )