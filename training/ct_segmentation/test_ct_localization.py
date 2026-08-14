from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image

from ai.ct_segmenter import CTSegmenter


TEST_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "segmentation"
    / "processed"
    / "splits"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "ct_segmentation"
    / "localization_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 60)
print("CT LOCALIZATION TEST")
print("=" * 60)


test_df = pd.read_csv(
    TEST_CSV
)

print(
    f"Test slices available: {len(test_df)}"
)


# Pick a sample from the middle of the held-out test set
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

mask_path = (
    PROJECT_ROOT
    / Path(
        row["mask_path"]
    )
)


print()
print("Input CT:")
print(
    image_path
)

print()
print("Ground-truth mask:")
print(
    mask_path
)


print()
print(
    "Loading CT infection segmentation model..."
)

segmenter = CTSegmenter()


print()
print(
    "Running CT infection localization..."
)

result = segmenter.segment(
    image_path
)


# ============================================================
# SAVE ORIGINAL
# ============================================================

original = Image.open(
    image_path
).convert("RGB")

original_path = (
    OUTPUT_DIR
    / "original_ct.png"
)

original.save(
    original_path
)


# ============================================================
# SAVE GROUND TRUTH
# ============================================================

ground_truth = Image.open(
    mask_path
).convert("L")

ground_truth_path = (
    OUTPUT_DIR
    / "ground_truth_mask.png"
)

ground_truth.save(
    ground_truth_path
)


# ============================================================
# SAVE PREDICTED MASK
# ============================================================

predicted_mask_path = (
    OUTPUT_DIR
    / "predicted_mask.png"
)

Image.fromarray(
    result["mask"]
).save(
    predicted_mask_path
)


# ============================================================
# SAVE HIGHLIGHTED CT
# ============================================================

highlighted_path = (
    OUTPUT_DIR
    / "highlighted_ct.png"
)

Image.fromarray(
    result["overlay"]
).save(
    highlighted_path
)


# ============================================================
# RESULTS
# ============================================================

print()
print("-" * 60)
print("CT LOCALIZATION RESULTS")
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
    "Predicted infection coverage:",
    f"{result['lesion_coverage'] * 100:.2f}%",
)

print(
    "Mean segmentation probability:",
    f"{result['mean_region_probability'] * 100:.2f}%",
)

print(
    "Maximum segmentation probability:",
    f"{result['maximum_probability'] * 100:.2f}%",
)

print(
    "Mask threshold:",
    result["threshold"],
)


print()
print("=" * 60)
print("CT LOCALIZATION TEST COMPLETE")
print("=" * 60)

print()
print("Original CT:")
print(
    original_path
)

print()
print("Ground-truth mask:")
print(
    ground_truth_path
)

print()
print("Predicted mask:")
print(
    predicted_mask_path
)

print()
print("Highlighted CT:")
print(
    highlighted_path
)