from pathlib import Path
import csv

import nibabel as nib
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CT_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "segmentation"
    / "raw"
    / "ct_scans"
)

MASK_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "segmentation"
    / "raw"
    / "infection_masks"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "segmentation"
    / "processed"
)

IMAGE_DIR = OUTPUT_ROOT / "images"
MASK_OUTPUT_DIR = OUTPUT_ROOT / "masks"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"


def normalize_ct_slice(slice_array):
    slice_array = np.asarray(
        slice_array,
        dtype=np.float32,
    )

    # Lung-oriented CT window
    low = -1000.0
    high = 400.0

    slice_array = np.clip(
        slice_array,
        low,
        high,
    )

    slice_array = (
        slice_array - low
    ) / (
        high - low
    )

    slice_array = (
        slice_array * 255.0
    ).astype(np.uint8)

    return slice_array


def normalize_mask(mask_slice):
    mask_slice = np.asarray(mask_slice)

    return (
        mask_slice > 0
    ).astype(np.uint8) * 255


def main():

    print()
    print("=" * 60)
    print("CT INFECTION SEGMENTATION PREPARATION")
    print("=" * 60)

    ct_files = sorted(
        CT_DIR.glob("*.nii.gz")
    )

    print(
        f"CT volumes found: {len(ct_files)}"
    )

    rows = []
    total_saved = 0
    skipped_empty = 0
    errors = 0

    for volume_index, ct_path in enumerate(
        ct_files,
        start=1,
    ):

        mask_path = (
            MASK_DIR
            / ct_path.name
        )

        if not mask_path.exists():
            print(
                f"Missing mask: {ct_path.name}"
            )
            errors += 1
            continue

        try:

            ct_volume = nib.load(
                str(ct_path)
            ).get_fdata()

            mask_volume = nib.load(
                str(mask_path)
            ).get_fdata()

            if ct_volume.shape != mask_volume.shape:
                print(
                    f"Shape mismatch for {ct_path.name}: "
                    f"{ct_volume.shape} vs {mask_volume.shape}"
                )
                errors += 1
                continue

            case_id = (
                ct_path.name
                .replace(".nii.gz", "")
            )

            # Assume axial slices are along axis 2
            num_slices = ct_volume.shape[2]

            saved_for_case = 0

            for slice_index in range(
                num_slices
            ):

                ct_slice = (
                    ct_volume[:, :, slice_index]
                )

                mask_slice = (
                    mask_volume[:, :, slice_index]
                )

                binary_mask = (
                    mask_slice > 0
                )

                # Keep only slices containing infection.
                if not np.any(binary_mask):
                    skipped_empty += 1
                    continue

                ct_image = normalize_ct_slice(
                    ct_slice
                )

                mask_image = normalize_mask(
                    mask_slice
                )

                image_name = (
                    f"{case_id}_slice_"
                    f"{slice_index:04d}.png"
                )

                mask_name = image_name

                image_output = (
                    IMAGE_DIR
                    / image_name
                )

                mask_output = (
                    MASK_OUTPUT_DIR
                    / mask_name
                )

                Image.fromarray(
                    ct_image
                ).save(
                    image_output
                )

                Image.fromarray(
                    mask_image
                ).save(
                    mask_output
                )

                infection_pixels = int(
                    np.count_nonzero(
                        mask_image
                    )
                )

                coverage = (
                    infection_pixels
                    / mask_image.size
                )

                rows.append(
                    {
                        "case_id": case_id,
                        "slice_index": slice_index,
                        "image_path": str(
                            image_output.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                        "mask_path": str(
                            mask_output.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                        "infection_pixels": infection_pixels,
                        "infection_coverage": coverage,
                    }
                )

                total_saved += 1
                saved_for_case += 1

            print(
                f"{volume_index:02d}/"
                f"{len(ct_files)} "
                f"{case_id}: "
                f"{saved_for_case} infection slices"
            )

        except Exception as error:
            errors += 1

            print(
                f"ERROR {ct_path.name}: "
                f"{error}"
            )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "slice_index",
                "image_path",
                "mask_path",
                "infection_pixels",
                "infection_coverage",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 60)
    print("CT PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Saved infection slices: {total_saved}"
    )

    print(
        f"Skipped empty slices: {skipped_empty}"
    )

    print(
        f"Errors: {errors}"
    )

    print()
    print(
        f"Images: {IMAGE_DIR}"
    )

    print(
        f"Masks: {MASK_OUTPUT_DIR}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()