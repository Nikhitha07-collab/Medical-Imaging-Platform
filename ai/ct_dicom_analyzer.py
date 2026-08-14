from pathlib import Path
from typing import Any

from ai.dicom_ai_bridge import (
    cleanup_temporary_ai_image,
    dicom_to_temporary_png,
)

from ai.ct_segmenter import segment_ct


def analyze_ct_dicom(
    dicom_path: str | Path,
    frame_index: int = 0,
) -> dict[str, Any]:

    bridge_result = None

    try:
        bridge_result = dicom_to_temporary_png(
            dicom_path=dicom_path,
            frame_index=frame_index,
            modality_override="CT",
        )

        temporary_png = bridge_result["temporary_png"]

        localization = segment_ct(
            temporary_png
        )

        return {
            "bridge": bridge_result,
            "localization": localization,
        }

    finally:
        if bridge_result is not None:
            cleanup_temporary_ai_image(
                bridge_result.get("temporary_png")
            )