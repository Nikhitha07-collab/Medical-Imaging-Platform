from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TN5000_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "tn5000"
    / "raw"
    / "dataset"
    / "TN5000_forReview"
)

JPEG_DIR = (
    TN5000_ROOT
    / "JPEGImages"
)

ANNOTATION_DIR = (
    TN5000_ROOT
    / "Annotations"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "segmentation"
    / "processed"
    / "splits"
)

YOLO_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "yolo"
)

IMAGES_ROOT = (
    YOLO_ROOT
    / "images"
)

LABELS_ROOT = (
    YOLO_ROOT
    / "labels"
)

TRAIN_IMAGES = (
    IMAGES_ROOT
    / "train"
)

VAL_IMAGES = (
    IMAGES_ROOT
    / "val"
)

TEST_IMAGES = (
    IMAGES_ROOT
    / "test"
)

TRAIN_LABELS = (
    LABELS_ROOT
    / "train"
)

VAL_LABELS = (
    LABELS_ROOT
    / "val"
)

TEST_LABELS = (
    LABELS_ROOT
    / "test"
)

DATA_YAML = (
    YOLO_ROOT
    / "data.yaml"
)


# ============================================================
# CREATE FOLDERS
# ============================================================

for folder in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TEST_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
    TEST_LABELS,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# HELPERS
# ============================================================

def load_box_from_xml(
    xml_path,
):

    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    size = root.find(
        "size"
    )

    width = int(
        float(
            size.findtext(
                "width"
            )
        )
    )

    height = int(
        float(
            size.findtext(
                "height"
            )
        )
    )


    object_node = root.find(
        "object"
    )

    if object_node is None:
        raise ValueError(
            f"No object found in {xml_path.name}"
        )


    bbox = object_node.find(
        "bndbox"
    )

    if bbox is None:
        raise ValueError(
            f"No bounding box found in {xml_path.name}"
        )


    xmin = float(
        bbox.findtext(
            "xmin"
        )
    )

    ymin = float(
        bbox.findtext(
            "ymin"
        )
    )

    xmax = float(
        bbox.findtext(
            "xmax"
        )
    )

    ymax = float(
        bbox.findtext(
            "ymax"
        )
    )


    xmin = max(
        0.0,
        min(
            xmin,
            width,
        ),
    )

    ymin = max(
        0.0,
        min(
            ymin,
            height,
        ),
    )

    xmax = max(
        0.0,
        min(
            xmax,
            width,
        ),
    )

    ymax = max(
        0.0,
        min(
            ymax,
            height,
        ),
    )


    box_width = (
        xmax - xmin
    )

    box_height = (
        ymax - ymin
    )


    if (
        box_width <= 0
        or box_height <= 0
    ):
        raise ValueError(
            f"Invalid box in {xml_path.name}"
        )


    center_x = (
        xmin + xmax
    ) / 2.0

    center_y = (
        ymin + ymax
    ) / 2.0


    center_x_norm = (
        center_x / width
    )

    center_y_norm = (
        center_y / height
    )

    width_norm = (
        box_width / width
    )

    height_norm = (
        box_height / height
    )


    return (
        center_x_norm,
        center_y_norm,
        width_norm,
        height_norm,
    )


def prepare_split(
    csv_path,
    image_output_dir,
    label_output_dir,
    split_name,
):

    dataframe = pd.read_csv(
        csv_path
    )

    print()
    print(
        f"Preparing {split_name}: "
        f"{len(dataframe)} samples"
    )

    copied = 0
    errors = 0


    for index, row in dataframe.iterrows():

        image_path = (
            PROJECT_ROOT
            / Path(
                row["image_path"]
            )
        )

        image_id = (
            image_path.stem
        )

        xml_path = (
            ANNOTATION_DIR
            / f"{image_id}.xml"
        )


        if not image_path.exists():

            print(
                f"Missing image: "
                f"{image_path}"
            )

            errors += 1
            continue


        if not xml_path.exists():

            print(
                f"Missing annotation: "
                f"{xml_path}"
            )

            errors += 1
            continue


        try:

            (
                center_x,
                center_y,
                box_width,
                box_height,
            ) = load_box_from_xml(
                xml_path
            )


            destination_image = (
                image_output_dir
                / image_path.name
            )

            destination_label = (
                label_output_dir
                / f"{image_id}.txt"
            )


            shutil.copy2(
                image_path,
                destination_image,
            )


            with destination_label.open(
                "w",
                encoding="utf-8",
            ) as label_file:

                # One lesion class only.
                class_id = 0

                label_file.write(
                    f"{class_id} "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}\n"
                )


            copied += 1


        except Exception as error:

            print(
                f"ERROR "
                f"{image_id}: "
                f"{error}"
            )

            errors += 1


        if (
            index + 1
        ) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(dataframe)}"
            )


    print(
        f"{split_name} complete: "
        f"{copied} prepared, "
        f"{errors} errors"
    )

    return copied, errors


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TN5000 YOLO DATASET PREPARATION")
    print("=" * 60)


    train_csv = (
        SPLIT_DIR
        / "train.csv"
    )

    val_csv = (
        SPLIT_DIR
        / "val.csv"
    )

    test_csv = (
        SPLIT_DIR
        / "test.csv"
    )


    for required_file in [
        train_csv,
        val_csv,
        test_csv,
    ]:

        if not required_file.exists():

            raise FileNotFoundError(
                f"Split file not found:\n"
                f"{required_file}"
            )


    train_count, train_errors = prepare_split(
        train_csv,
        TRAIN_IMAGES,
        TRAIN_LABELS,
        "TRAIN",
    )

    val_count, val_errors = prepare_split(
        val_csv,
        VAL_IMAGES,
        VAL_LABELS,
        "VALIDATION",
    )

    test_count, test_errors = prepare_split(
        test_csv,
        TEST_IMAGES,
        TEST_LABELS,
        "TEST",
    )


    # ========================================================
    # WRITE YOLO data.yaml
    # ========================================================

    yaml_text = (
        f"path: {YOLO_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: thyroid_lesion\n"
    )


    DATA_YAML.write_text(
        yaml_text,
        encoding="utf-8",
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("YOLO DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Train images : {train_count}"
    )

    print(
        f"Val images   : {val_count}"
    )

    print(
        f"Test images  : {test_count}"
    )

    print(
        f"Total errors : "
        f"{train_errors + val_errors + test_errors}"
    )

    print()
    print(
        "Dataset YAML:"
    )

    print(
        DATA_YAML
    )


if __name__ == "__main__":
    main()