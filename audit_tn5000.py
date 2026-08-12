from pathlib import Path
import xml.etree.ElementTree as ET


BASE = Path(
    "training_data/ultrasound/tn5000/raw/"
    "dataset/TN5000_forReview"
)

IMAGE_FOLDER = BASE / "JPEGImages"
ANNOTATION_FOLDER = BASE / "Annotations"
SPLIT_FOLDER = BASE / "ImageSets" / "Main"


def read_split(name: str) -> list[str]:
    path = SPLIT_FOLDER / f"{name}.txt"

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def annotation_label(image_id: str):
    xml_path = (
        ANNOTATION_FOLDER
        / f"{image_id}.xml"
    )

    if not xml_path.exists():
        return None

    root = ET.parse(xml_path).getroot()

    names = [
        item.text
        for item in root.findall(".//object/name")
        if item.text is not None
    ]

    return names


print("\nTN5000 DATASET AUDIT")
print("=" * 50)

jpg_files = list(
    IMAGE_FOLDER.glob("*.jpg")
)

xml_files = list(
    ANNOTATION_FOLDER.glob("*.xml")
)

print(
    f"Extracted JPG files: {len(jpg_files)}"
)

print(
    f"Annotation XML files: {len(xml_files)}"
)

print()

for split_name in [
    "train",
    "val",
    "test",
]:

    ids = read_split(split_name)

    missing_images = []

    missing_annotations = []

    for image_id in ids:

        image_path = (
            IMAGE_FOLDER
            / f"{image_id}.jpg"
        )

        annotation_path = (
            ANNOTATION_FOLDER
            / f"{image_id}.xml"
        )

        if not image_path.exists():
            missing_images.append(
                image_id
            )

        if not annotation_path.exists():
            missing_annotations.append(
                image_id
            )

    print(
        f"{split_name.upper()}: "
        f"{len(ids)} IDs"
    )

    print(
        "  Missing images: "
        f"{len(missing_images)}"
    )

    print(
        "  Missing annotations: "
        f"{len(missing_annotations)}"
    )

    if missing_images:
        print(
            "  First missing image IDs:",
            missing_images[:10],
        )

    print()


print("SAMPLE LABELS")
print("=" * 50)

for image_id in read_split(
    "train"
)[:10]:

    print(
        image_id,
        "->",
        annotation_label(image_id),
    )