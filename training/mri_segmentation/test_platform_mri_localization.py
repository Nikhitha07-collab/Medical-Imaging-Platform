from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image

from ai.mri_segmenter import MRISegmenter


# ============================================================
# CURRENT MRI CLASSIFICATION TEST DATA
# ============================================================

TEST_CSV = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "brain_tumor"
    / "splits"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "mri_segmentation"
    / "platform_compatibility_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"MRI classifier test CSV not found:\n{TEST_CSV}"
    )


df = pd.read_csv(TEST_CSV)

if len(df) == 0:
    raise RuntimeError(
        "MRI classifier test CSV is empty."
    )


print()
print("=" * 60)
print("PLATFORM MRI LOCALIZATION COMPATIBILITY TEST")
print("=" * 60)

print(f"Classifier test images: {len(df)}")


# ============================================================
# SHOW COLUMN NAMES
# ============================================================

print()
print("CSV columns:")
print(list(df.columns))


# ============================================================
# FIND IMAGE PATH COLUMN
# ============================================================

possible_columns = [
    "image_path",
    "filepath",
    "file_path",
    "path",
]

image_column = None

for column in possible_columns:
    if column in df.columns:
        image_column = column
        break

if image_column is None:
    raise RuntimeError(
        "Could not determine image-path column. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# SELECT ONE TEST IMAGE
# ============================================================

sample_index = len(df) // 2

row = df.iloc[sample_index]

image_path = Path(
    str(row[image_column])
)

if not image_path.is_absolute():
    image_path = (
        PROJECT_ROOT
        / image_path
    )


if not image_path.exists():
    raise FileNotFoundError(
        f"Test MRI image not found:\n{image_path}"
    )


print()
print("Testing image:")
print(image_path)


# ============================================================
# CLASS INFORMATION IF AVAILABLE
# ============================================================

for class_column in [
    "class_name",
    "label",
    "class",
]:
    if class_column in df.columns:
        print()
        print(
            "Known class:",
            row[class_column],
        )
        break


# ============================================================
# RUN SEGMENTATION
# ============================================================

segmenter = MRISegmenter()

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
    / "platform_mri_original.png"
)

original.save(
    original_path
)


# ============================================================
# SAVE MASK
# ============================================================

mask_path = (
    OUTPUT_DIR
    / "platform_mri_predicted_mask.png"
)

Image.fromarray(
    result["mask"]
).save(
    mask_path
)


# ============================================================
# SAVE HIGHLIGHTED IMAGE
# ============================================================

highlight_path = (
    OUTPUT_DIR
    / "platform_mri_highlighted.png"
)

Image.fromarray(
    result["overlay"]
).save(
    highlight_path
)


# ============================================================
# RESULTS
# ============================================================

print()
print("-" * 60)
print("LOCALIZATION OUTPUT")
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
    "Predicted region coverage:",
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

print()
print("=" * 60)
print("COMPATIBILITY TEST COMPLETE")
print("=" * 60)

print()
print("Original:")
print(original_path)

print()
print("Predicted mask:")
print(mask_path)

print()
print("Highlighted:")
print(highlight_path)