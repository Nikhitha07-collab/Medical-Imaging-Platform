from pathlib import Path
import shutil

import pydicom


UPLOAD_FOLDER = Path("uploaded_data")


def prepare_upload_folder() -> Path:
    """Create a clean temporary folder for uploaded DICOM files."""

    if UPLOAD_FOLDER.exists():
        shutil.rmtree(UPLOAD_FOLDER)

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    return UPLOAD_FOLDER


def find_dicom_files(folder: Path) -> list[Path]:
    """Find valid DICOM files recursively inside a folder."""

    dicom_files: list[Path] = []

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            pydicom.dcmread(
                file_path,
                stop_before_pixels=True,
            )
            dicom_files.append(file_path)
        except Exception:
            continue

    return sorted(dicom_files)


def detect_modality(file_path: Path) -> str:
    """Detect CT, MRI, or Ultrasound from a DICOM file."""

    dataset = pydicom.dcmread(
        file_path,
        stop_before_pixels=True,
    )

    modality = str(
        getattr(dataset, "Modality", "Unknown")
    ).upper()

    modality_names = {
        "CT": "CT",
        "MR": "MRI",
        "US": "Ultrasound",
    }

    return modality_names.get(modality, modality)