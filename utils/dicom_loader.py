from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset


def load_dicom(file_path: str | Path) -> FileDataset:
    """Load one DICOM file and return its dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        dataset = pydicom.dcmread(path)
    except Exception as error:
        raise ValueError(f"Unable to read DICOM file: {error}") from error

    if "PixelData" not in dataset:
        raise ValueError("This DICOM file does not contain image pixel data.")

    return dataset


def print_basic_information(dataset: FileDataset) -> None:
    """Print safe, basic DICOM information."""

    print("DICOM file loaded successfully")
    print(f"Modality: {getattr(dataset, 'Modality', 'Not available')}")
    print(f"Manufacturer: {getattr(dataset, 'Manufacturer', 'Not available')}")
    print(
        f"Series Description: "
        f"{getattr(dataset, 'SeriesDescription', 'Not available')}"
    )
    print(f"Rows: {getattr(dataset, 'Rows', 'Not available')}")
    print(f"Columns: {getattr(dataset, 'Columns', 'Not available')}")
    print("PixelData: Available")