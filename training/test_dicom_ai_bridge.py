from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.dicom_ai_bridge import dicom_to_temporary_png


CT_DICOM = (
    PROJECT_ROOT
    / "uploaded_data"
    / "b4f0d4c4-fdf1-41fa-801c-6ea56e10c516.dcm"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "dicom_ai_bridge_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 60)
print("DICOM AI BRIDGE TEST")
print("=" * 60)

print()
print("Input DICOM:")
print(CT_DICOM)


result = dicom_to_temporary_png(
    dicom_path=CT_DICOM,
    frame_index=0,
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

print()
print("Temporary PNG:")
print(
    result["temporary_png"]
)


output_path = (
    OUTPUT_DIR
    / "ct_dicom_ai_render.png"
)

output_path.write_bytes(
    Path(
        result["temporary_png"]
    ).read_bytes()
)


print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)

print()
print("Saved preview:")
print(output_path)