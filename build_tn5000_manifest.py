from pathlib import Path
import csv
import xml.etree.ElementTree as ET


BASE = Path(
    "training_data/ultrasound/tn5000/raw/"
    "dataset/TN5000_forReview"
)

IMAGE_FOLDER = BASE / "JPEGImages"
ANNOTATION_FOLDER = BASE / "Annotations"
SPLIT_FOLDER = BASE / "ImageSets" / "Main"

OUTPUT_FOLDER = Path(
    "training_data/ultrasound/tn5000/splits"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


def read_split(split_name: str) -> list[str]:
    split_file = (
        SPLIT_FOLDER
        / f"{split_name}.txt"
    )

    return [
        line.strip()
        for line in split_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def read_label(image_id: str) -> int:
    xml_file = (
        ANNOTATION_FOLDER
        / f"{image_id}.xml"
    )

    root = ET.parse(
        xml_file
    ).getroot()

    label_element = root.find(
        ".//object/name"
    )

    if label_element is None:
        raise ValueError(
            f"No label found for {image_id}"
        )

    label = int(
        label_element.text
    )

    if label not in {0, 1}:
        raise ValueError(
            f"Unexpected label {label} "
            f"for {image_id}"
        )

    return label


def build_manifest(
    split_name: str,
) -> None:

    image_ids = read_split(
        split_name
    )

    output_file = (
        OUTPUT_FOLDER
        / f"{split_name}.csv"
    )

    rows = []

    class_counts = {
        0: 0,
        1: 0,
    }

    for image_id in image_ids:

        image_path = (
            IMAGE_FOLDER
            / f"{image_id}.jpg"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        label = read_label(
            image_id
        )

        class_counts[
            label
        ] += 1

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(
                    image_path
                ),
                "label": label,
            }
        )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_id",
                "image_path",
                "label",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        f"{split_name.upper()}: "
        f"{len(rows)} samples"
    )

    print(
        f"  Label 0: "
        f"{class_counts[0]}"
    )

    print(
        f"  Label 1: "
        f"{class_counts[1]}"
    )

    print(
        f"  Saved: "
        f"{output_file}"
    )

    print()


for split in [
    "train",
    "val",
    "test",
]:
    build_manifest(
        split
    )