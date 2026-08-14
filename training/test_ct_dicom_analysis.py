from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from PIL import Image

from ai.ct_dicom_analyzer import analyze_ct_dicom


CT_DICOM = (
    PROJECT_ROOT
    / "uploaded_data"
    / "b4f0d4c4-fdf1-41fa-801c-6ea56e10c516.dcm"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "ct_dicom_ai_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 60)
print("CT DICOM AI ANALYSIS TEST")
print("=" * 60)

print()
print("Input DICOM:")
print(CT_DICOM)

print()
print("Running CT DICOM localization...")


result = analyze_ct_dicom(
    CT_DICOM
)

localization = result[
    "localization"
]


print()
print("-" * 60)
print("CT LOCALIZATION RESULT")
print("-" * 60)

print(
    "Detected region:",
    localization.get(
        "detected_region"
    ),
)

print(
    "Bounding box:",
    localization.get(
        "bounding_box"
    ),
)

print(
    "Coverage:",
    localization.get(
        "coverage"
    ),
)

print(
    "Mean probability:",
    localization.get(
        "mean_probability"
    ),
)


overlay = localization.get(
    "overlay"
)

if overlay is not None:

    output_path = (
        OUTPUT_DIR
        / "ct_dicom_localization.png"
    )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )

    print()
    print("Saved overlay:")
    print(output_path)

else:

    print()
    print(
        "No overlay returned."
    )


print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)