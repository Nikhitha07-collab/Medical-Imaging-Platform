from pathlib import Path
from typing import Any

from ai.dicom_ai_bridge import (
    cleanup_temporary_ai_image,
    dicom_to_temporary_png,
)

from ai.mri_classifier import MRIClassifier
from ai.mri_segmenter import segment_mri


mri_classifier = MRIClassifier()


def analyze_mri_dicom(
    dicom_path: str | Path,
    frame_index: int = 0,
) -> dict[str, Any]:
    """
    Convert one MRI DICOM slice/frame to a temporary PNG,
    then run the existing MRI classifier and tumor localizer.
    """

    bridge_result = None

    try:
        bridge_result = dicom_to_temporary_png(
            dicom_path=dicom_path,
            frame_index=frame_index,
            modality_override="MR",
        )

        temporary_png = bridge_result[
            "temporary_png"
        ]

        classification = mri_classifier.predict(
            temporary_png
        )

        localization = segment_mri(
            temporary_png
        )

        return {
            "bridge": bridge_result,
            "classification": classification,
            "localization": localization,
        }

    finally:
        if bridge_result is not None:
            cleanup_temporary_ai_image(
                bridge_result.get(
                    "temporary_png"
                )
            )