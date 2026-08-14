from pathlib import Path
import csv
import xml.etree.ElementTree as ET

from PIL import Image


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

IMAGE_DIR = (
    TN5000_ROOT
    / "JPEGImages"
)

ANNOTATION_DIR = (
    TN5000_ROOT
    / "Annotations"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound"
    / "segmentation"
    / "processed"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_PATH = (
    OUTPUT_ROOT
    / "localization_manifest.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TN5000 ULTRASOUND LOCALIZATION PREPARATION")
    print("=" * 60)

    xml_files = sorted(
        ANNOTATION_DIR.glob("*.xml")
    )

    print(
        f"Annotation files found: {len(xml_files)}"
    )

    rows = []

    processed = 0
    missing_images = 0
    invalid_boxes = 0
    errors = 0


    for index, xml_path in enumerate(
        xml_files,
        start=1,
    ):

        try:

            tree = ET.parse(
                xml_path
            )

            root = tree.getroot()


            # ------------------------------------------------
            # IMAGE FILE
            # ------------------------------------------------

            filename = root.findtext(
                "filename"
            )

            if not filename:
                filename = (
                    xml_path.stem
                    + ".jpg"
                )

            image_path = (
                IMAGE_DIR
                / filename
            )

            if not image_path.exists():

                print(
                    f"Missing image: "
                    f"{image_path.name}"
                )

                missing_images += 1
                continue


            # ------------------------------------------------
            # IMAGE SIZE
            # ------------------------------------------------

            with Image.open(
                image_path
            ) as image:

                actual_width = int(
                    image.width
                )

                actual_height = int(
                    image.height
                )


            width_text = root.findtext(
                "size/width"
            )

            height_text = root.findtext(
                "size/height"
            )

            width = (
                int(width_text)
                if width_text
                else actual_width
            )

            height = (
                int(height_text)
                if height_text
                else actual_height
            )


            # Use actual image dimensions if XML differs
            width = actual_width
            height = actual_height


            # ------------------------------------------------
            # OBJECT
            # ------------------------------------------------

            object_node = root.find(
                "object"
            )

            if object_node is None:

                print(
                    f"No lesion object: "
                    f"{xml_path.name}"
                )

                errors += 1
                continue


            class_label = (
                object_node.findtext(
                    "name"
                )
                or "unknown"
            )


            bbox = object_node.find(
                "bndbox"
            )

            if bbox is None:

                print(
                    f"No bounding box: "
                    f"{xml_path.name}"
                )

                errors += 1
                continue


            xmin = int(
                float(
                    bbox.findtext(
                        "xmin"
                    )
                )
            )

            ymin = int(
                float(
                    bbox.findtext(
                        "ymin"
                    )
                )
            )

            xmax = int(
                float(
                    bbox.findtext(
                        "xmax"
                    )
                )
            )

            ymax = int(
                float(
                    bbox.findtext(
                        "ymax"
                    )
                )
            )


            # ------------------------------------------------
            # CLAMP BOX TO IMAGE
            # ------------------------------------------------

            xmin = max(
                0,
                min(
                    xmin,
                    width - 1,
                ),
            )

            ymin = max(
                0,
                min(
                    ymin,
                    height - 1,
                ),
            )

            xmax = max(
                0,
                min(
                    xmax,
                    width,
                ),
            )

            ymax = max(
                0,
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

                print(
                    f"Invalid bounding box: "
                    f"{xml_path.name}"
                )

                invalid_boxes += 1
                continue


            # ------------------------------------------------
            # NORMALIZED BOX
            # ------------------------------------------------

            xmin_norm = (
                xmin / width
            )

            ymin_norm = (
                ymin / height
            )

            xmax_norm = (
                xmax / width
            )

            ymax_norm = (
                ymax / height
            )


            center_x_norm = (
                (
                    xmin + xmax
                )
                / 2.0
                / width
            )

            center_y_norm = (
                (
                    ymin + ymax
                )
                / 2.0
                / height
            )

            width_norm = (
                box_width
                / width
            )

            height_norm = (
                box_height
                / height
            )


            # ------------------------------------------------
            # SAVE MANIFEST ROW
            # ------------------------------------------------

            rows.append(
                {
                    "image_id":
                        xml_path.stem,

                    "image_path":
                        str(
                            image_path.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "annotation_path":
                        str(
                            xml_path.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "class_label":
                        class_label,

                    "image_width":
                        width,

                    "image_height":
                        height,

                    "xmin":
                        xmin,

                    "ymin":
                        ymin,

                    "xmax":
                        xmax,

                    "ymax":
                        ymax,

                    "box_width":
                        box_width,

                    "box_height":
                        box_height,

                    "xmin_norm":
                        xmin_norm,

                    "ymin_norm":
                        ymin_norm,

                    "xmax_norm":
                        xmax_norm,

                    "ymax_norm":
                        ymax_norm,

                    "center_x_norm":
                        center_x_norm,

                    "center_y_norm":
                        center_y_norm,

                    "width_norm":
                        width_norm,

                    "height_norm":
                        height_norm,
                }
            )

            processed += 1


        except Exception as error:

            errors += 1

            print(
                f"ERROR "
                f"{xml_path.name}: "
                f"{error}"
            )


        if index % 500 == 0:

            print(
                f"Processed "
                f"{index}/"
                f"{len(xml_files)}"
            )


    # ========================================================
    # WRITE CSV
    # ========================================================

    fieldnames = [
        "image_id",
        "image_path",
        "annotation_path",
        "class_label",
        "image_width",
        "image_height",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "box_width",
        "box_height",
        "xmin_norm",
        "ymin_norm",
        "xmax_norm",
        "ymax_norm",
        "center_x_norm",
        "center_y_norm",
        "width_norm",
        "height_norm",
    ]


    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("ULTRASOUND LOCALIZATION PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Processed annotations : {processed}"
    )

    print(
        f"Missing images        : {missing_images}"
    )

    print(
        f"Invalid boxes         : {invalid_boxes}"
    )

    print(
        f"Other errors          : {errors}"
    )

    print()
    print(
        f"Manifest:"
    )

    print(
        MANIFEST_PATH
    )


if __name__ == "__main__":
    main()