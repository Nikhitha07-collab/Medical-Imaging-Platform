from pathlib import Path
import sys

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow this script to import modules from the project root,
# including ai.mri_segmenter
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
from PIL import Image

from ai.mri_segmenter import MRISegmenter


# ============================================================
# PATHS
# ============================================================

TEST_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "segmentation"
    / "processed"
    / "splits"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "mri_segmentation"
    / "localization_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VERIFY TEST CSV
# ============================================================

if not TEST_CSV.exists():
    raise FileNotFoundError(
        "MRI segmentation test CSV was not found:\n"
        f"{TEST_CSV}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(
    TEST_CSV
)

if len(test_df) == 0:
    raise RuntimeError(
        "MRI segmentation test split is empty."
    )

print()
print("=" * 60)
print("MRI LOCALIZATION TEST")
print("=" * 60)

print()
print(
    f"Test slices available: {len(test_df)}"
)


# ============================================================
# SELECT TEST IMAGE
# ============================================================

# Use an image around the middle of the held-out test set.
sample_index = len(test_df) // 2

row = test_df.iloc[
    sample_index
]

image_path = (
    PROJECT_ROOT
    / Path(
        row["image_path"]
    )
)

true_mask_path = (
    PROJECT_ROOT
    / Path(
        row["mask_path"]
    )
)


# ============================================================
# VERIFY IMAGE AND MASK
# ============================================================

if not image_path.exists():
    raise FileNotFoundError(
        "MRI test image was not found:\n"
        f"{image_path}"
    )

if not true_mask_path.exists():
    raise FileNotFoundError(
        "MRI ground-truth mask was not found:\n"
        f"{true_mask_path}"
    )


print()
print("Input image:")
print(image_path)

print()
print("Ground-truth mask:")
print(true_mask_path)


# ============================================================
# LOAD MRI SEGMENTATION MODEL
# ============================================================

print()
print("Loading MRI tumor segmentation model...")

segmenter = MRISegmenter()


# ============================================================
# RUN TUMOR LOCALIZATION
# ============================================================

print()
print("Running tumor localization...")

result = segmenter.segment(
    image_path
)


# ============================================================
# SAVE ORIGINAL MRI
# ============================================================

original_image = Image.open(
    image_path
).convert("RGB")

original_output = (
    OUTPUT_DIR
    / "original_mri.png"
)

original_image.save(
    original_output
)


# ============================================================
# SAVE GROUND-TRUTH MASK
# ============================================================

true_mask = Image.open(
    true_mask_path
).convert("L")

true_mask_output = (
    OUTPUT_DIR
    / "ground_truth_mask.png"
)

true_mask.save(
    true_mask_output
)


# ============================================================
# SAVE PREDICTED MASK
# ============================================================

predicted_mask_output = (
    OUTPUT_DIR
    / "predicted_mask.png"
)

predicted_mask_image = Image.fromarray(
    result["mask"]
)

predicted_mask_image.save(
    predicted_mask_output
)


# ============================================================
# SAVE HIGHLIGHTED MRI
# ============================================================

highlight_output = (
    OUTPUT_DIR
    / "highlighted_mri.png"
)

highlight_image = Image.fromarray(
    result["overlay"]
)

highlight_image.save(
    highlight_output
)


# ============================================================
# PRINT LOCALIZATION RESULTS
# ============================================================

print()
print("-" * 60)
print("LOCALIZATION RESULTS")
print("-" * 60)

print(
    "Detected region:",
    result["has_detected_region"],
)

print(
    "Bounding box:",
    result["bounding_box"],
)

print(
    "Lesion pixels:",
    result["lesion_pixels"],
)

print(
    "Lesion coverage:",
    f"{result['lesion_coverage'] * 100:.2f}%",
)

print(
    "Mean region probability:",
    f"{result['mean_region_probability'] * 100:.2f}%",
)

print(
    "Maximum probability:",
    f"{result['maximum_probability'] * 100:.2f}%",
)

print(
    "Mask threshold:",
    result["threshold"],
)


# ============================================================
# PRINT OUTPUT FILES
# ============================================================

print()
print("=" * 60)
print("MRI LOCALIZATION TEST COMPLETE")
print("=" * 60)

print()
print("Original MRI:")
print(original_output)

print()
print("Ground-truth mask:")
print(true_mask_output)

print()
print("Predicted mask:")
print(predicted_mask_output)

print()
print("Highlighted MRI:")
print(highlight_output)

print()
print(
    "Open highlighted_mri.png to inspect "
    "the predicted tumor region and bounding box."
)