from pathlib import Path
import random
import sys

import cv2


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ai.ultrasound_yolo_detector import UltrasoundYOLODetector


# ============================================================
# PATHS
# ============================================================

TEST_IMAGES = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "yolo"
    / "images"
    / "test"
)

TEST_LABELS = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "yolo"
    / "labels"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "ultrasound_segmentation"
    / "yolo_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GET TEST IMAGES
# ============================================================

image_files = sorted(
    list(TEST_IMAGES.glob("*.jpg"))
    + list(TEST_IMAGES.glob("*.jpeg"))
    + list(TEST_IMAGES.glob("*.png"))
)

# Remove duplicates safely.
image_files = list(
    dict.fromkeys(image_files)
)

if not image_files:
    raise FileNotFoundError(
        f"No test images found in:\n{TEST_IMAGES}"
    )


print()
print("=" * 60)
print("TN5000 YOLO LOCALIZATION EVALUATION")
print("=" * 60)

print()
print(
    "Unique test images:",
    len(image_files),
)


# ============================================================
# SELECT TEST IMAGE
# ============================================================

random.seed(42)

image_path = random.choice(
    image_files
)

label_path = (
    TEST_LABELS
    / f"{image_path.stem}.txt"
)


print()
print("Selected test image:")
print(image_path)

print()
print("Ground-truth label:")
print(label_path)


if not label_path.exists():
    raise FileNotFoundError(
        f"Ground-truth label missing:\n{label_path}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise RuntimeError(
        f"Unable to read image:\n{image_path}"
    )


image_height = image.shape[0]
image_width = image.shape[1]


# ============================================================
# READ YOLO GROUND-TRUTH BOX
# ============================================================

label_line = (
    label_path
    .read_text(
        encoding="utf-8"
    )
    .strip()
    .splitlines()[0]
)

parts = label_line.split()

if len(parts) != 5:
    raise ValueError(
        "Unexpected YOLO ground-truth label format."
    )


class_id = int(
    float(parts[0])
)

center_x = float(
    parts[1]
)

center_y = float(
    parts[2]
)

box_width_norm = float(
    parts[3]
)

box_height_norm = float(
    parts[4]
)


gt_width = (
    box_width_norm
    * image_width
)

gt_height = (
    box_height_norm
    * image_height
)


gt_center_x = (
    center_x
    * image_width
)

gt_center_y = (
    center_y
    * image_height
)


gt_x1 = int(
    round(
        gt_center_x
        - gt_width / 2
    )
)

gt_y1 = int(
    round(
        gt_center_y
        - gt_height / 2
    )
)

gt_x2 = int(
    round(
        gt_center_x
        + gt_width / 2
    )
)

gt_y2 = int(
    round(
        gt_center_y
        + gt_height / 2
    )
)


gt_x1 = max(
    0,
    min(
        gt_x1,
        image_width - 1,
    ),
)

gt_y1 = max(
    0,
    min(
        gt_y1,
        image_height - 1,
    ),
)

gt_x2 = max(
    gt_x1 + 1,
    min(
        gt_x2,
        image_width,
    ),
)

gt_y2 = max(
    gt_y1 + 1,
    min(
        gt_y2,
        image_height,
    ),
)


# ============================================================
# RUN YOLO
# ============================================================

detector = UltrasoundYOLODetector()


print()
print(
    "Running trained YOLO detector..."
)


result = detector.predict(
    image_path=image_path,
    confidence_threshold=0.25,
)


# ============================================================
# PREDICTED BOX
# ============================================================

predicted_box = result[
    "bbox"
]


if result["detected"]:

    pred_x1 = int(
        predicted_box["x"]
    )

    pred_y1 = int(
        predicted_box["y"]
    )

    pred_x2 = int(
        predicted_box["x2"]
    )

    pred_y2 = int(
        predicted_box["y2"]
    )

else:

    pred_x1 = 0
    pred_y1 = 0
    pred_x2 = 0
    pred_y2 = 0


# ============================================================
# CALCULATE IoU
# ============================================================

if result["detected"]:

    intersection_x1 = max(
        gt_x1,
        pred_x1,
    )

    intersection_y1 = max(
        gt_y1,
        pred_y1,
    )

    intersection_x2 = min(
        gt_x2,
        pred_x2,
    )

    intersection_y2 = min(
        gt_y2,
        pred_y2,
    )


    intersection_width = max(
        0,
        intersection_x2
        - intersection_x1,
    )

    intersection_height = max(
        0,
        intersection_y2
        - intersection_y1,
    )


    intersection_area = (
        intersection_width
        * intersection_height
    )


    ground_truth_area = (
        (gt_x2 - gt_x1)
        * (gt_y2 - gt_y1)
    )


    predicted_area = (
        (pred_x2 - pred_x1)
        * (pred_y2 - pred_y1)
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

else:

    iou = 0.0


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("-" * 60)
print("YOLO LOCALIZATION RESULT")
print("-" * 60)

print(
    "Detected:",
    result["detected"],
)

print(
    "Number of detections:",
    result["number_of_detections"],
)

print(
    "Confidence:",
    f'{result["confidence_percent"]:.2f}%',
)

print()

print(
    "Ground-truth box:"
)

print(
    {
        "x": gt_x1,
        "y": gt_y1,
        "x2": gt_x2,
        "y2": gt_y2,
    }
)

print()

print(
    "Predicted box:"
)

print(
    predicted_box
)

print()

print(
    "IoU:",
    f"{iou:.4f}",
)


# ============================================================
# DRAW COMPARISON IMAGE
# ============================================================

comparison_image = image.copy()


# Ground truth = BLUE
cv2.rectangle(
    comparison_image,
    (
        gt_x1,
        gt_y1,
    ),
    (
        gt_x2,
        gt_y2,
    ),
    (
        255,
        0,
        0,
    ),
    3,
)


cv2.putText(
    comparison_image,
    "Ground truth",
    (
        gt_x1,
        max(
            25,
            gt_y1 - 10,
        ),
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (
        255,
        0,
        0,
    ),
    2,
    cv2.LINE_AA,
)


# Prediction = GREEN
if result["detected"]:

    cv2.rectangle(
        comparison_image,
        (
            pred_x1,
            pred_y1,
        ),
        (
            pred_x2,
            pred_y2,
        ),
        (
            0,
            255,
            0,
        ),
        3,
    )


    prediction_label = (
        "YOLO "
        f"{result['confidence_percent']:.1f}%"
    )


    cv2.putText(
        comparison_image,
        prediction_label,
        (
            pred_x1,
            min(
                image_height - 10,
                pred_y2 + 25,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            0,
            255,
            0,
        ),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# SAVE
# ============================================================

comparison_path = (
    OUTPUT_DIR
    / "yolo_groundtruth_vs_prediction.jpg"
)


cv2.imwrite(
    str(comparison_path),
    comparison_image,
)


print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print()
print(
    "Comparison image:"
)

print(
    comparison_path
)

print()
print(
    "BLUE  = ground-truth lesion box"
)

print(
    "GREEN = YOLO predicted lesion box"
)