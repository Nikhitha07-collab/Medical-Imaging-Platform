from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.dicom_ai_bridge import dicom_to_temporary_png


MRI_DICOM = (
    PROJECT_ROOT
    / "uploaded_data"
    / "2be6e33b-42b9-4928-8368-2398c4ef5a1f.dcm"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "mri_dicom_ai_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 60)
print("MRI DICOM BRIDGE TEST")
print("=" * 60)

print()
print("Input MRI:")
print(MRI_DICOM)


result = dicom_to_temporary_png(
    dicom_path=MRI_DICOM,
    frame_index=0,
    modality_override="MR",
)


print()
print("Detected modality:")
print(result["modality"])

print()
print("Dimensions:")
print(
    result["width"],
    "x",
    result["height"],
)

print()
print("Photometric interpretation:")
print(
    result["photometric_interpretation"]
)


temporary_png = Path(
    result["temporary_png"]
)

output_path = (
    OUTPUT_DIR
    / "mri_dicom_render.png"
)

output_path.write_bytes(
    temporary_png.read_bytes()
)


print()
print("=" * 60)
print("MRI DICOM BRIDGE TEST COMPLETE")
print("=" * 60)

print()
print("Saved MRI preview:")
print(output_path)