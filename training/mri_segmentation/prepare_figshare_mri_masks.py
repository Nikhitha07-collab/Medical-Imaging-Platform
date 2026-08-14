from pathlib import Path
import csv

import h5py
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAT_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "brain_tumor"
    / "raw"
    / "mat_files"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "segmentation"
    / "figshare_processed"
)

IMAGE_DIR = OUTPUT_ROOT / "images"
MASK_DIR = OUTPUT_ROOT / "masks"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"


def normalize_image(image):
    image = image.astype(np.float32)

    minimum = float(np.min(image))
    maximum = float(np.max(image))

    if maximum > minimum:
        image = (
            image - minimum
        ) / (
            maximum - minimum
        )
    else:
        image = np.zeros_like(
            image,
            dtype=np.float32,
        )

    image = (
        image * 255.0
    ).astype(np.uint8)

    return image


def normalize_mask(mask):
    mask = np.asarray(mask)

    mask = (
        mask > 0
    ).astype(np.uint8) * 255

    return mask


def read_scalar(dataset):
    value = np.array(dataset)

    value = np.squeeze(value)

    if value.size == 1:
        return int(value)

    return None


def main():

    print()
    print("=" * 60)
    print("FIGSHARE MRI SEGMENTATION PREPARATION")
    print("=" * 60)

    mat_files = sorted(
        MAT_DIR.glob("*.mat"),
        key=lambda path: int(path.stem),
    )

    print(
        f"MAT files found: {len(mat_files)}"
    )

    rows = []
    processed = 0
    errors = 0

    class_map = {
        1: "meningioma",
        2: "glioma",
        3: "pituitary",
    }

    for index, mat_path in enumerate(
        mat_files,
        start=1,
    ):

        try:

            with h5py.File(
                mat_path,
                "r",
            ) as file:

                cjdata = file[
                    "cjdata"
                ]

                image = np.array(
                    cjdata["image"]
                )

                mask = np.array(
                    cjdata["tumorMask"]
                )

                label = read_scalar(
                    cjdata["label"]
                )


            image = np.squeeze(
                image
            )

            mask = np.squeeze(
                mask
            )


            # h5py-loaded MATLAB arrays may need
            # transposition to match expected orientation.
            if (
                image.ndim == 2
                and mask.ndim == 2
            ):

                if image.shape != mask.shape:
                    raise ValueError(
                        f"Image/mask shape mismatch: "
                        f"{image.shape} vs {mask.shape}"
                    )

            else:
                raise ValueError(
                    f"Unexpected image/mask dimensions: "
                    f"{image.shape}, {mask.shape}"
                )


            image = normalize_image(
                image
            )

            mask = normalize_mask(
                mask
            )


            image_id = mat_path.stem

            class_name = class_map.get(
                label,
                "unknown",
            )


            image_output = (
                IMAGE_DIR
                / f"{image_id}.png"
            )

            mask_output = (
                MASK_DIR
                / f"{image_id}.png"
            )


            Image.fromarray(
                image
            ).save(
                image_output
            )

            Image.fromarray(
                mask
            ).save(
                mask_output
            )


            tumor_pixels = int(
                np.count_nonzero(
                    mask
                )
            )

            total_pixels = int(
                mask.size
            )

            tumor_coverage = (
                tumor_pixels
                / total_pixels
                if total_pixels
                else 0.0
            )


            rows.append(
                {
                    "image_id": image_id,
                    "class_label": label,
                    "class_name": class_name,
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
                    "tumor_pixels": tumor_pixels,
                    "tumor_coverage": tumor_coverage,
                }
            )

            processed += 1


        except Exception as error:

            errors += 1

            print(
                f"ERROR {mat_path.name}: "
                f"{error}"
            )


        if index % 250 == 0:

            print(
                f"Processed "
                f"{index}/"
                f"{len(mat_files)}"
            )


    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "class_label",
                "class_name",
                "image_path",
                "mask_path",
                "tumor_pixels",
                "tumor_coverage",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


    print()
    print("=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Processed: {processed}"
    )

    print(
        f"Errors: {errors}"
    )

    print()
    print(
        f"Images: {IMAGE_DIR}"
    )

    print(
        f"Masks: {MASK_DIR}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()