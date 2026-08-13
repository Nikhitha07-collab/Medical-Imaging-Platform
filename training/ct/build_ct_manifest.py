from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "covid_ct"
    / "raw"
)

COVID_DIR = (
    RAW_DIR
    / "Images-processed"
    / "CT_COVID"
)

NONCOVID_DIR = (
    RAW_DIR
    / "Images-processed"
    / "CT_NonCOVID"
)

SPLIT_DIR = (
    RAW_DIR
    / "Data-split"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "ct"
    / "covid_ct"
    / "splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# OFFICIAL SPLIT FILES
# ---------------------------------------------------------

SPLIT_FILES = {
    "train": {
        "COVID": (
            SPLIT_DIR
            / "COVID"
            / "trainCT_COVID.txt"
        ),
        "NonCOVID": (
            SPLIT_DIR
            / "NonCOVID"
            / "trainCT_NonCOVID.txt"
        ),
    },
    "val": {
        "COVID": (
            SPLIT_DIR
            / "COVID"
            / "valCT_COVID.txt"
        ),
        "NonCOVID": (
            SPLIT_DIR
            / "NonCOVID"
            / "valCT_NonCOVID.txt"
        ),
    },
    "test": {
        "COVID": (
            SPLIT_DIR
            / "COVID"
            / "testCT_COVID.txt"
        ),
        "NonCOVID": (
            SPLIT_DIR
            / "NonCOVID"
            / "testCT_NonCOVID.txt"
        ),
    },
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def read_split_file(
    path: Path,
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def normalize_name(
    value: str,
) -> str:
    return (
        value
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\r", "")
        .replace("\n", "")
        .strip()
        .casefold()
    )


def find_image(
    folder: Path,
    filename: str,
) -> Path | None:

    filename = (
        filename
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .strip()
    )

    exact_path = (
        folder
        / filename
    )

    if exact_path.exists():
        return exact_path

    target_name = normalize_name(
        filename
    )

    for file_path in folder.iterdir():

        if not file_path.is_file():
            continue

        if normalize_name(
            file_path.name
        ) == target_name:
            return file_path

    target_stem = normalize_name(
        Path(filename).stem
    )

    for file_path in folder.iterdir():

        if not file_path.is_file():
            continue

        if normalize_name(
            file_path.stem
        ) == target_stem:
            return file_path

    return None


def verify_dataset_folders() -> None:

    required_folders = [
        COVID_DIR,
        NONCOVID_DIR,
        SPLIT_DIR,
    ]

    for folder in required_folders:

        if not folder.exists():
            raise FileNotFoundError(
                f"Required folder not found:\n{folder}"
            )

    print()
    print(
        "Dataset folders verified."
    )

    print(
        f"COVID image files: "
        f"{len(list(COVID_DIR.iterdir()))}"
    )

    print(
        f"NonCOVID image files: "
        f"{len(list(NONCOVID_DIR.iterdir()))}"
    )


# ---------------------------------------------------------
# BUILD ONE MANIFEST
# ---------------------------------------------------------

def build_manifest(
    split_name: str,
) -> pd.DataFrame:

    rows = []
    missing = []

    covid_names = read_split_file(
        SPLIT_FILES[
            split_name
        ]["COVID"]
    )

    noncovid_names = read_split_file(
        SPLIT_FILES[
            split_name
        ]["NonCOVID"]
    )

    # -----------------------------------------------------
    # COVID
    # Label 1
    # -----------------------------------------------------

    for filename in covid_names:

        image_path = find_image(
            COVID_DIR,
            filename,
        )

        if image_path is None:

            missing.append(
                f"COVID: {filename}"
            )

            continue

        relative_path = (
            image_path
            .relative_to(
                PROJECT_ROOT
            )
        )

        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(
                    relative_path
                ),
                "label": 1,
                "class_name": "COVID",
            }
        )

    # -----------------------------------------------------
    # NonCOVID
    # Label 0
    # -----------------------------------------------------

    for filename in noncovid_names:

        image_path = find_image(
            NONCOVID_DIR,
            filename,
        )

        if image_path is None:

            missing.append(
                f"NonCOVID: {filename}"
            )

            continue

        relative_path = (
            image_path
            .relative_to(
                PROJECT_ROOT
            )
        )

        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(
                    relative_path
                ),
                "label": 0,
                "class_name": "NonCOVID",
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    output_file = (
        OUTPUT_DIR
        / f"{split_name}.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 60)
    print(
        f"{split_name.upper()} SPLIT"
    )
    print("=" * 60)

    print(
        f"Requested COVID     : "
        f"{len(covid_names)}"
    )

    print(
        f"Requested NonCOVID  : "
        f"{len(noncovid_names)}"
    )

    print(
        f"Total requested     : "
        f"{len(covid_names) + len(noncovid_names)}"
    )

    print(
        f"Total samples found : "
        f"{len(dataframe)}"
    )

    print()

    if not dataframe.empty:

        print(
            "Class counts:"
        )

        print(
            dataframe[
                "class_name"
            ].value_counts()
        )

    print()

    print(
        f"Missing files: "
        f"{len(missing)}"
    )

    if missing:

        print()
        print(
            "Missing file names:"
        )

        for item in missing:
            print(
                f"  {item}"
            )

    print()

    print(
        f"Saved manifest:\n"
        f"{output_file}"
    )

    return dataframe


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:

    print()
    print("=" * 60)
    print("COVID-CT MANIFEST BUILDER")
    print("=" * 60)

    verify_dataset_folders()

    train_df = build_manifest(
        "train"
    )

    val_df = build_manifest(
        "val"
    )

    test_df = build_manifest(
        "test"
    )

    print()
    print("=" * 60)
    print("CT MANIFEST BUILD COMPLETE")
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
        "Manifest folder:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()