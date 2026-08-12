from collections import defaultdict
from pathlib import Path
import shutil

import pydicom

from utils.image_loader import is_standard_image


UPLOAD_FOLDER = Path("uploaded_data")


def prepare_upload_folder() -> Path:
    """Create a clean temporary upload folder."""

    if UPLOAD_FOLDER.exists():
        shutil.rmtree(UPLOAD_FOLDER)

    UPLOAD_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    return UPLOAD_FOLDER


def read_header(file_path: Path):
    """Read DICOM metadata without loading pixel data."""

    return pydicom.dcmread(
        file_path,
        stop_before_pixels=True,
    )


def is_valid_dicom(file_path: Path) -> bool:
    """Check whether a file is a valid DICOM file."""

    try:
        read_header(file_path)
        return True

    except Exception:
        return False


def find_dicom_files(folder: Path) -> list[Path]:
    """Find valid DICOM files recursively."""

    if not folder.exists():
        return []

    dicom_files: list[Path] = []

    for file_path in folder.rglob("*"):

        if not file_path.is_file():
            continue

        if is_valid_dicom(file_path):
            dicom_files.append(file_path)

    return dicom_files


def find_standard_images(folder: Path) -> list[Path]:
    """Find PNG, JPG, and JPEG files."""

    if not folder.exists():
        return []

    image_files: list[Path] = []

    for file_path in folder.rglob("*"):

        if not file_path.is_file():
            continue

        if is_standard_image(file_path):
            image_files.append(file_path)

    return sorted(image_files)


def normalize_modality(value: str) -> str:
    """Convert DICOM modality codes into readable names."""

    modality = str(value).upper()

    mapping = {
        "CT": "CT",
        "MR": "MRI",
        "US": "Ultrasound",
    }

    return mapping.get(
        modality,
        modality or "Unknown",
    )


def detect_modality(file_path: Path) -> str:
    """Detect modality from a DICOM file."""

    dataset = read_header(file_path)

    return normalize_modality(
        getattr(
            dataset,
            "Modality",
            "Unknown",
        )
    )


def get_instance_number(file_path: Path) -> int:
    """Get InstanceNumber for correct slice ordering."""

    try:
        dataset = read_header(file_path)

        return int(
            getattr(
                dataset,
                "InstanceNumber",
                0,
            )
            or 0
        )

    except Exception:
        return 0


def group_dicom_series(
    files: list[Path],
) -> dict[str, dict]:
    """Group DICOM files by StudyInstanceUID and SeriesInstanceUID."""

    grouped_files = defaultdict(list)

    for file_path in files:

        try:
            dataset = read_header(file_path)

            study_uid = str(
                getattr(
                    dataset,
                    "StudyInstanceUID",
                    "UNKNOWN_STUDY",
                )
            )

            series_uid = str(
                getattr(
                    dataset,
                    "SeriesInstanceUID",
                    "UNKNOWN_SERIES",
                )
            )

            grouped_files[
                (
                    study_uid,
                    series_uid,
                )
            ].append(file_path)

        except Exception:
            continue

    series_data: dict[str, dict] = {}

    for (
        study_uid,
        series_uid,
    ), series_files in grouped_files.items():

        series_files = sorted(
            series_files,
            key=get_instance_number,
        )

        first_dataset = read_header(
            series_files[0]
        )

        modality = normalize_modality(
            getattr(
                first_dataset,
                "Modality",
                "Unknown",
            )
        )

        description = str(
            getattr(
                first_dataset,
                "SeriesDescription",
                "No description",
            )
        )

        body_part = str(
            getattr(
                first_dataset,
                "BodyPartExamined",
                "Not available",
            )
        )

        display_name = (
            f"{modality} | "
            f"{description} | "
            f"{len(series_files)} image(s)"
        )

        series_data[series_uid] = {
            "study_uid": study_uid,
            "series_uid": series_uid,
            "modality": modality,
            "description": description,
            "body_part": body_part,
            "files": series_files,
            "display_name": display_name,
        }

    return series_data


def classify_uploaded_files(
    folder: Path,
) -> dict[str, list[Path]]:
    """Separate uploaded files into DICOM and PNG/JPG groups."""

    return {
        "dicom": find_dicom_files(folder),
        "standard": find_standard_images(folder),
    }