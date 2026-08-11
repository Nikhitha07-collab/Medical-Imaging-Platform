from pathlib import Path
import shutil

import pydicom


UPLOAD_FOLDER = Path(
    "uploaded_data"
)


def prepare_upload_folder() -> Path:
    """Create a clean temporary DICOM upload folder."""

    if UPLOAD_FOLDER.exists():
        shutil.rmtree(
            UPLOAD_FOLDER
        )

    UPLOAD_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    return UPLOAD_FOLDER


def _dicom_sort_key(
    file_path: Path,
) -> tuple:
    """Create a useful ordering key for DICOM slices."""

    try:
        dataset = pydicom.dcmread(
            file_path,
            stop_before_pixels=True,
        )

        series_uid = str(
            getattr(
                dataset,
                "SeriesInstanceUID",
                "",
            )
        )

        instance_number = int(
            getattr(
                dataset,
                "InstanceNumber",
                0,
            )
            or 0
        )

        image_position = getattr(
            dataset,
            "ImagePositionPatient",
            None,
        )

        if (
            image_position is not None
            and len(image_position) >= 3
        ):
            slice_position = float(
                image_position[2]
            )
        else:
            slice_position = 0.0

        return (
            series_uid,
            instance_number,
            slice_position,
            file_path.name,
        )

    except Exception:
        return (
            "",
            0,
            0.0,
            file_path.name,
        )


def find_dicom_files(
    folder: Path,
) -> list[Path]:
    """Find valid DICOM files recursively."""

    dicom_files: list[Path] = []

    if not folder.exists():
        return dicom_files

    for file_path in folder.rglob("*"):

        if not file_path.is_file():
            continue

        try:
            pydicom.dcmread(
                file_path,
                stop_before_pixels=True,
            )

            dicom_files.append(
                file_path
            )

        except Exception:
            continue

    return sorted(
        dicom_files,
        key=_dicom_sort_key,
    )


def detect_modality(
    file_path: Path,
) -> str:
    """Detect the modality from a DICOM header."""

    dataset = pydicom.dcmread(
        file_path,
        stop_before_pixels=True,
    )

    modality = str(
        getattr(
            dataset,
            "Modality",
            "Unknown",
        )
    ).upper()

    modality_names = {
        "CT": "CT",
        "MR": "MRI",
        "US": "Ultrasound",
    }

    return modality_names.get(
        modality,
        modality,
    )