from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image, ImageDraw

from ai.ultrasound_localizer import UltrasoundLocalizer


TEST_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "segmentation"
    / "processed"
    / "splits"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "ultrasound_segmentation"
    / "localization_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 60)
print("ULTRASOUND LOCALIZATION TEST")
print("=" * 60)


test_df = pd.read_csv(
    TEST_CSV
)

print(
    f"Test samples available: {len(test_df)}"
)


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


print()
print("Input ultrasound:")
print(
    image_path
)

print()
print("Ground-truth bounding box:")
print(
    {
        "xmin": int(row["xmin"]),
        "ymin": int(row["ymin"]),
        "xmax": int(row["xmax"]),
        "ymax": int(row["ymax"]),
    }
)


print()
print(
    "Loading ultrasound localization model..."
)

localizer = UltrasoundLocalizer()


print()
print(
    "Running lesion localization..."
)

result = localizer.localize(
    image_path
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

original = Image.open(
    image_path
).convert("RGB")

original_path = (
    OUTPUT_DIR
    / "original_ultrasound.png"
)

original.save(
    original_path
)


# ============================================================
# GROUND-TRUTH BOX IMAGE
# ============================================================

ground_truth = original.copy()

draw_gt = ImageDraw.Draw(
    ground_truth
)

gt_xmin = int(
    row["xmin"]
)

gt_ymin = int(
    row["ymin"]
)

gt_xmax = int(
    row["xmax"]
)

gt_ymax = int(
    row["ymax"]
)


draw_gt.rectangle(
    [
        gt_xmin,
        gt_ymin,
        gt_xmax,
        gt_ymax,
    ],
    outline="white",
    width=4,
)


ground_truth_path = (
    OUTPUT_DIR
    / "ground_truth_box.png"
)

ground_truth.save(
    ground_truth_path
)


# ============================================================
# PREDICTED BOX IMAGE
# ============================================================

predicted_path = (
    OUTPUT_DIR
    / "predicted_localization.png"
)

Image.fromarray(
    result["overlay"]
).save(
    predicted_path
)


# ============================================================
# CALCULATE IoU
# ============================================================

pred_box = result[
    "bounding_box"
]

pred_xmin = pred_box["x"]
pred_ymin = pred_box["y"]
pred_xmax = pred_box["x2"]
pred_ymax = pred_box["y2"]


intersection_xmin = max(
    gt_xmin,
    pred_xmin,
)

intersection_ymin = max(
    gt_ymin,
    pred_ymin,
)

intersection_xmax = min(
    gt_xmax,
    pred_xmax,
)

intersection_ymax = min(
    gt_ymax,
    pred_ymax,
)


intersection_width = max(
    0,
    intersection_xmax
    - intersection_xmin,
)

intersection_height = max(
    0,
    intersection_ymax
    - intersection_ymin,
)


intersection_area = (
    intersection_width
    * intersection_height
)


ground_truth_area = (
    max(
        0,
        gt_xmax - gt_xmin,
    )
    *
    max(
        0,
        gt_ymax - gt_ymin,
    )
)


predicted_area = (
    max(
        0,
        pred_xmax - pred_xmin,
    )
    *
    max(
        0,
        pred_ymax - pred_ymin,
    )
)


union_area = (
    ground_truth_area
    + predicted_area
    - intersection_area
)


iou = (
    intersection_area
    / union_area
    if union_area > 0
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("-" * 60)
print("ULTRASOUND LOCALIZATION RESULTS")
print("-" * 60)

print(
    "Predicted bounding box:",
    result["bounding_box"],
)

print(
    "Predicted box coverage:",
    f"{result['box_coverage'] * 100:.2f}%",
)

print(
    "Ground-truth IoU:",
    f"{iou:.4f}",
)


print()
print("=" * 60)
print("ULTRASOUND LOCALIZATION TEST COMPLETE")
print("=" * 60)

print()
print("Original:")
print(
    original_path
)

print()
print("Ground-truth box:")
print(
    ground_truth_path
)

print()
print("Predicted localization:")
print(
    predicted_path
)