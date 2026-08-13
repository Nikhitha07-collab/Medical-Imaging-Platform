from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAT_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "brain_tumor"
    / "raw"
    / "mat_files"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "brain_tumor"
    / "processed"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "mri"
    / "brain_tumor"
    / "splits"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# LABEL MAPPING
# Original dataset:
# 1 = meningioma
# 2 = glioma
# 3 = pituitary
# ---------------------------------------------------------

LABEL_MAP = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}

for class_name in LABEL_MAP.values():
    (
        PROCESSED_DIR
        / class_name
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------
# READ ONE MATLAB FILE
# ---------------------------------------------------------

def read_mat_file(mat_path):

    with h5py.File(
        mat_path,
        "r",
    ) as file:

        cjdata = file["cjdata"]

        label = int(
            np.array(
                cjdata["label"]
            ).squeeze()
        )

        image = np.array(
            cjdata["image"]
        )

    return image, label


# ---------------------------------------------------------
# NORMALIZE MRI IMAGE
# ---------------------------------------------------------

def normalize_image(image):

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = float(
        image.min()
    )

    maximum = float(
        image.max()
    )

    if maximum > minimum:

        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    else:

        image = np.zeros_like(
            image
        )

    image = (
        image * 255.0
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    return image


# ---------------------------------------------------------
# CONVERT ALL FILES
# ---------------------------------------------------------

def convert_dataset():

    mat_files = sorted(
        MAT_DIR.glob("*.mat"),
        key=lambda path: int(
            path.stem
        ),
    )

    print()
    print("=" * 60)
    print("MRI DATASET PREPARATION")
    print("=" * 60)

    print(
        f"MAT files found: {len(mat_files)}"
    )

    if len(mat_files) != 3064:
        print(
            "WARNING: Expected 3064 MRI files."
        )

    rows = []
    errors = []

    for index, mat_path in enumerate(
        mat_files,
        start=1,
    ):

        try:

            image, label = read_mat_file(
                mat_path
            )

            if label not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label: {label}"
                )

            class_name = LABEL_MAP[
                label
            ]

            image = normalize_image(
                image
            )

            output_path = (
                PROCESSED_DIR
                / class_name
                / f"{mat_path.stem}.png"
            )

            Image.fromarray(
                image
            ).save(
                output_path
            )

            rows.append(
                {
                    "image_id": mat_path.stem,
                    "image_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "label": label - 1,
                    "class_name": class_name,
                }
            )

        except Exception as error:

            errors.append(
                (
                    mat_path.name,
                    str(error),
                )
            )

        if (
            index % 250 == 0
            or index == len(mat_files)
        ):

            print(
                f"Processed "
                f"{index}/{len(mat_files)}"
            )

    return (
        pd.DataFrame(rows),
        errors,
    )


# ---------------------------------------------------------
# CREATE STRATIFIED SPLITS
# ---------------------------------------------------------

def create_splits(dataframe):

    # 70% train
    # 15% validation
    # 15% test

    train_df, temp_df = (
        train_test_split(
            dataframe,
            test_size=0.30,
            random_state=42,
            stratify=dataframe[
                "label"
            ],
        )
    )

    val_df, test_df = (
        train_test_split(
            temp_df,
            test_size=0.50,
            random_state=42,
            stratify=temp_df[
                "label"
            ],
        )
    )

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    train_df.to_csv(
        SPLIT_DIR / "train.csv",
        index=False,
    )

    val_df.to_csv(
        SPLIT_DIR / "val.csv",
        index=False,
    )

    test_df.to_csv(
        SPLIT_DIR / "test.csv",
        index=False,
    )

    return (
        train_df,
        val_df,
        test_df,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    dataframe, errors = (
        convert_dataset()
    )

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(
        f"Converted: {len(dataframe)}"
    )

    print(
        f"Errors: {len(errors)}"
    )

    print()
    print("Class distribution:")

    print(
        dataframe[
            "class_name"
        ].value_counts()
    )

    if errors:

        print()
        print(
            "First conversion errors:"
        )

        for filename, error in errors[:10]:

            print(
                f"{filename}: {error}"
            )

    if dataframe.empty:

        raise RuntimeError(
            "No MRI images were converted."
        )

    train_df, val_df, test_df = (
        create_splits(
            dataframe
        )
    )

    print()
    print("=" * 60)
    print("MRI SPLITS COMPLETE")
    print("=" * 60)

    print(
        f"Train      : {len(train_df)}"
    )

    print(
        f"Validation : {len(val_df)}"
    )

    print(
        f"Test       : {len(test_df)}"
    )

    print()
    print(
        "Training classes:"
    )

    print(
        train_df[
            "class_name"
        ].value_counts()
    )

    print()
    print(
        f"PNG folder:\n{PROCESSED_DIR}"
    )

    print()
    print(
        f"Split folder:\n{SPLIT_DIR}"
    )


if __name__ == "__main__":
    main()