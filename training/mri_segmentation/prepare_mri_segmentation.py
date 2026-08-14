from pathlib import Path
import csv

import nibabel as nib
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "segmentation"
    / "raw"
    / "Task01_BrainTumour"
)

IMAGES_DIR = DATA_ROOT / "imagesTr"
LABELS_DIR = DATA_ROOT / "labelsTr"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "segmentation"
    / "processed"
)

IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "images"
MASK_OUTPUT_DIR = OUTPUT_ROOT / "masks"

MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"

IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_slice(image_slice: np.ndarray) -> np.ndarray:
    image_slice = image_slice.astype(np.float32)

    lower = np.percentile(image_slice, 1)
    upper = np.percentile(image_slice, 99)

    image_slice = np.clip(
        image_slice,
        lower,
        upper,
    )

    minimum = image_slice.min()
    maximum = image_slice.max()

    if maximum > minimum:
        image_slice = (
            image_slice - minimum
        ) / (
            maximum - minimum
        )
    else:
        image_slice = np.zeros_like(image_slice)

    image_slice = (
        image_slice * 255.0
    ).astype(np.uint8)

    return image_slice


def binary_mask(mask_slice: np.ndarray) -> np.ndarray:
    mask = (
        mask_slice > 0
    ).astype(np.uint8) * 255

    return mask


def find_training_cases():
    return sorted(
        IMAGES_DIR.glob("BRATS_*.nii.gz")
    )


def main():
    print("=" * 60)
    print("MRI SEGMENTATION DATASET PREPARATION")
    print("=" * 60)

    image_files = find_training_cases()

    print(f"MRI volumes found: {len(image_files)}")

    rows = []

    total_slices = 0
    tumor_slices = 0

    for case_index, image_path in enumerate(
        image_files,
        start=1,
    ):
        case_name = image_path.name.replace(
            ".nii.gz",
            "",
        )

        label_path = (
            LABELS_DIR
            / f"{case_name}.nii.gz"
        )

        if not label_path.exists():
            print(
                f"Skipping {case_name}: "
                "matching label not found"
            )
            continue

        image_volume = nib.load(
            str(image_path)
        ).get_fdata()

        mask_volume = nib.load(
            str(label_path)
        ).get_fdata()

        if image_volume.ndim == 4:
            image_volume = image_volume[..., 0]

        if mask_volume.ndim == 4:
            mask_volume = mask_volume[..., 0]

        if image_volume.shape[:3] != mask_volume.shape[:3]:
            print(
                f"Skipping {case_name}: "
                "image/mask shape mismatch"
            )
            continue

        depth = image_volume.shape[2]

        for slice_index in range(depth):
            image_slice = image_volume[
                :,
                :,
                slice_index,
            ]

            mask_slice = mask_volume[
                :,
                :,
                slice_index,
            ]

            if np.max(image_slice) == 0:
                continue

            has_tumor = (
                np.max(mask_slice) > 0
            )

            if not has_tumor:
                continue

            image_slice = normalize_slice(
                image_slice
            )

            mask_slice = binary_mask(
                mask_slice
            )

            file_stem = (
                f"{case_name}_slice_"
                f"{slice_index:03d}"
            )

            image_output_path = (
                IMAGE_OUTPUT_DIR
                / f"{file_stem}.png"
            )

            mask_output_path = (
                MASK_OUTPUT_DIR
                / f"{file_stem}.png"
            )

            Image.fromarray(
                image_slice
            ).save(
                image_output_path
            )

            Image.fromarray(
                mask_slice
            ).save(
                mask_output_path
            )

            rows.append(
                {
                    "case_id": case_name,
                    "slice_index": slice_index,
                    "image_path": str(
                        image_output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "mask_path": str(
                        mask_output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "has_tumor": 1,
                }
            )

            total_slices += 1
            tumor_slices += 1

        if case_index % 25 == 0:
            print(
                f"Processed "
                f"{case_index}/"
                f"{len(image_files)} volumes"
            )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "case_id",
                "slice_index",
                "image_path",
                "mask_path",
                "has_tumor",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("MRI SEGMENTATION PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Saved tumor slices: {tumor_slices}")
    print(f"Total saved slices: {total_slices}")

    print()
    print("Images:")
    print(IMAGE_OUTPUT_DIR)

    print()
    print("Masks:")
    print(MASK_OUTPUT_DIR)

    print()
    print("Manifest:")
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()