from pathlib import Path

from utils.dicom_loader import load_dicom, print_basic_information


ct_folder = Path("test_data/CT/chest_ct/27548")
dicom_files = list(ct_folder.glob("*.dcm"))

if not dicom_files:
    raise FileNotFoundError(f"No DICOM files found inside: {ct_folder}")

first_file = dicom_files[0]

print(f"Testing file: {first_file.name}")

dataset = load_dicom(first_file)
print_basic_information(dataset)