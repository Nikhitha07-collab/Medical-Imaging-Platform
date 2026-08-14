from pathlib import Path
import sys

import cv2


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MRI DICOM ANALYZER
# ============================================================

from ai.mri_dicom_analyzer import analyze_mri_dicom


# ============================================================
# FIND MRI DICOM FILE
# ============================================================

uploaded_data_dir = PROJECT_ROOT / "uploaded_data"

dicom_files = sorted(
    uploaded_data_dir.glob("*.dcm")
)

if not dicom_files:
    raise FileNotFoundError(
        f"No DICOM files found in:\n{uploaded_data_dir}"
    )


# Find an MR/MRI DICOM automatically.
import pydicom

mri_file = None

for dicom_file in dicom_files:
    try:
        ds = pydicom.dcmread(
            str(dicom_file),
            stop_before_pixels=True,
            force=True,
        )

        modality = str(
            getattr(ds, "Modality", "")
        ).upper()

        if modality in {"MR", "MRI"}:
            mri_file = dicom_file
            break

    except Exception:
        continue


if mri_file is None:
    raise FileNotFoundError(
        "No MRI DICOM file was found inside uploaded_data."
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = (
    PROJECT_ROOT
    / "training"
    / "mri_dicom_ai_test"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    output_dir
    / "mri_dicom_localization.png"
)


# ============================================================
# RUN MRI DICOM ANALYSIS
# ============================================================

print()
print("=" * 60)
print("MRI DICOM AI LOCALIZATION TEST")
print("=" * 60)

print()
print("Input MRI DICOM:")
print(mri_file)

print()
print("Running MRI DICOM analysis...")


result = analyze_mri_dicom(
    mri_file
)


# ============================================================
# READ RESULTS
# ============================================================

if not isinstance(result, dict):
    raise TypeError(
        "analyze_mri_dicom() did not return a dictionary."
    )


bridge = result.get(
    "bridge",
    {},
)

localization = result.get(
    "localization",
    {},
)


print()
print("-" * 60)
print("MRI DICOM INFORMATION")
print("-" * 60)

print(
    "Detected modality:",
    bridge.get(
        "modality",
        bridge.get(
            "detected_modality",
            "Unknown",
        ),
    ),
)


width = bridge.get(
    "width",
    bridge.get(
        "image_width",
        "Unknown",
    ),
)

height = bridge.get(
    "height",
    bridge.get(
        "image_height",
        "Unknown",
    ),
)

print(
    "Dimensions:",
    f"{width} x {height}",
)


# ============================================================
# LOCALIZATION INFORMATION
# ============================================================

print()
print("-" * 60)
print("MRI LOCALIZATION RESULT")
print("-" * 60)


has_region = localization.get(
    "has_detected_region",
    localization.get(
        "detected",
        False,
    ),
)

print(
    "Detected region:",
    has_region,
)


bounding_box = localization.get(
    "bounding_box"
)

print(
    "Bounding box:",
    bounding_box,
)


coverage = localization.get(
    "lesion_coverage",
    localization.get(
        "coverage"
    ),
)

if coverage is not None:
    print(
        "Lesion coverage:",
        f"{float(coverage) * 100:.2f}%",
    )
else:
    print(
        "Lesion coverage:",
        "Not available",
    )


mean_probability = localization.get(
    "mean_region_probability",
    localization.get(
        "mean_probability"
    ),
)

if mean_probability is not None:
    print(
        "Mean probability:",
        f"{float(mean_probability) * 100:.2f}%",
    )
else:
    print(
        "Mean probability:",
        "Not available",
    )


maximum_probability = localization.get(
    "maximum_probability"
)

if maximum_probability is not None:
    print(
        "Maximum probability:",
        f"{float(maximum_probability) * 100:.2f}%",
    )


# ============================================================
# SAVE MRI LOCALIZATION OVERLAY
# ============================================================

overlay = localization.get(
    "overlay"
)

if overlay is None:
    raise RuntimeError(
        "MRI analyzer returned no localization overlay."
    )


success = cv2.imwrite(
    str(output_path),
    overlay,
)

if not success:
    raise RuntimeError(
        f"Unable to save localization image:\n{output_path}"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("MRI DICOM AI TEST COMPLETE")
print("=" * 60)

print()
print("Saved MRI localization:")
print(output_path)

print()
print(
    "Open this image to inspect the MRI AI localization."
)