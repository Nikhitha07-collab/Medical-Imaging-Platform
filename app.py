from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Any

from nicegui import events, ui
from PIL import Image
import numpy as np

from ai.ct_classifier import CTClassifier
from ai.mri_classifier import MRIClassifier
from ai.ultrasound_classifier import UltrasoundClassifier
from utils.dicom_loader import load_dicom
from utils.image_loader import load_standard_image
from utils.measurements import (
    physical_distance_mm,
    pixel_distance,
)
from utils.preprocessing import (
    adjust_brightness_contrast,
    dicom_to_image,
)
from utils.upload_handler import (
    UPLOAD_FOLDER,
    classify_uploaded_files,
    find_dicom_files,
    group_dicom_series,
    prepare_upload_folder,
)


PROJECT_TITLE = "Medical Imaging Platform"


MODALITY_FOLDERS = {
    "CT": Path("test_data/CT/chest_ct/27548"),
    "MRI": Path("test_data/MRI/abdomen_mri/80231"),
    "Ultrasound": Path(
        "test_data/Ultrasound/thyroid_us/46711"
    ),
}


CT_WINDOW_PRESETS = {
    "Lung": (-600.0, 1500.0),
    "Soft Tissue": (40.0, 400.0),
    "Bone": (300.0, 1500.0),
    "Brain": (40.0, 80.0),
}


current_files: list[Path] = []
current_modality = "CT"
current_file_type = "DICOM"
current_slice_index = 0
current_source = "Sample CT"

current_dataset = None

dicom_uploaded_count = 0
uploaded_series: dict[str, dict] = {}
series_display_map: dict[str, str] = {}
uploaded_standard_images: list[Path] = []

current_zoom = 1.0
current_rotation = 0
flip_horizontal = False
flip_vertical = False

measurement_mode = False
measurement_points: list[tuple[float, float]] = []
hu_probe_mode = False
roi_mode = False
roi_points: list[tuple[float, float]] = []
saved_annotations: list[str] = []

ct_classifier = CTClassifier()
mri_classifier = MRIClassifier()
ultrasound_classifier = UltrasoundClassifier()


def image_to_data_url(image_array) -> str:
    image = Image.fromarray(image_array)

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    encoded = b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def safe_value(
    dataset: Any,
    attribute: str,
) -> str:
    value = getattr(
        dataset,
        attribute,
        "Not available",
    )

    if value in (None, ""):
        return "Not available"

    return str(value)


def get_transfer_syntax(
    dataset: Any,
) -> str:
    try:
        return str(
            dataset.file_meta.TransferSyntaxUID
        )

    except Exception:
        return "Not available"


def clear_metadata() -> None:
    modality_label.set_text(
        f"Modality: {current_modality}"
    )

    manufacturer_label.set_text(
        "Manufacturer: Not available"
    )

    model_label.set_text(
        "Model: Not available"
    )

    study_date_label.set_text(
        "Study Date: Not available"
    )

    series_label.set_text(
        "Series Description: Not available"
    )

    body_part_label.set_text(
        "Body Part: Not available"
    )

    dimensions_label.set_text(
        "Dimensions: Not available"
    )

    frames_label.set_text(
        "Number of Frames: Not available"
    )

    photometric_label.set_text(
        "Photometric Interpretation: Not available"
    )

    pixel_spacing_label.set_text(
        "Pixel Spacing: Not available"
    )

    slice_thickness_label.set_text(
        "Slice Thickness: Not available"
    )

    window_center_label.set_text(
        "Window Center: Not available"
    )

    window_width_label.set_text(
        "Window Width: Not available"
    )

    sop_class_label.set_text(
        "SOP Class UID: Not available"
    )

    study_uid_label.set_text(
        "Study Instance UID: Not available"
    )

    series_uid_label.set_text(
        "Series Instance UID: Not available"
    )

    transfer_syntax_label.set_text(
        "Transfer Syntax UID: Not available"
    )
    if "sequence_name_label" in globals():
        sequence_name_label.set_text("Sequence Name: Not available")
        repetition_time_label.set_text("TR: Not available")
        echo_time_label.set_text("TE: Not available")
        flip_angle_label.set_text("Flip Angle: Not available")
        field_strength_label.set_text("Magnetic Field Strength: Not available")


def update_metadata(
    dataset: Any,
) -> None:
    modality_label.set_text(
        "Modality: "
        f"{safe_value(dataset, 'Modality')}"
    )

    manufacturer_label.set_text(
        "Manufacturer: "
        f"{safe_value(dataset, 'Manufacturer')}"
    )

    model_label.set_text(
        "Model: "
        f"{safe_value(dataset, 'ManufacturerModelName')}"
    )

    study_date_label.set_text(
        "Study Date: "
        f"{safe_value(dataset, 'StudyDate')}"
    )

    series_label.set_text(
        "Series Description: "
        f"{safe_value(dataset, 'SeriesDescription')}"
    )

    body_part = safe_value(
        dataset,
        "BodyPartExamined",
    )

    body_part_source = None

    if (
        body_part == "Not available"
        or not str(body_part).strip()
    ):
        descriptions = [
            safe_value(dataset, "SeriesDescription"),
            safe_value(dataset, "StudyDescription"),
            safe_value(dataset, "ProtocolName"),
        ]

        combined_description = " ".join(
            str(value).upper()
            for value in descriptions
            if value != "Not available"
        )

        body_part_keywords = {
            "CHEST": "Chest",
            "THORAX": "Chest",
            "LUNG": "Chest",
            "BRAIN": "Brain",
            "HEAD": "Head",
            "ABDOMEN": "Abdomen",
            "ABDOMINAL": "Abdomen",
            "PELVIS": "Pelvis",
            "SPINE": "Spine",
            "CERVICAL": "Cervical Spine",
            "THORACIC": "Thoracic Spine",
            "LUMBAR": "Lumbar Spine",
            "KNEE": "Knee",
            "SHOULDER": "Shoulder",
            "THYROID": "Thyroid",
            "BREAST": "Breast",
            "CARDIAC": "Heart",
            "HEART": "Heart",
        }

        for keyword, inferred_part in body_part_keywords.items():
            if keyword in combined_description:
                body_part = inferred_part
                body_part_source = "inferred"
                break

    if (
        body_part == "Not available"
        or not str(body_part).strip()
    ):
        body_part_label.set_text(
            "Body Part: Not available"
        )
    elif body_part_source == "inferred":
        body_part_label.set_text(
            f"Body Part: {body_part} "
            "(inferred from study/series metadata)"
        )
    else:
        body_part_label.set_text(
            f"Body Part: {body_part}"
        )

    dimensions_label.set_text(
        "Dimensions: "
        f"{safe_value(dataset, 'Rows')} × "
        f"{safe_value(dataset, 'Columns')}"
    )

    frames_label.set_text(
        "Number of Frames: "
        f"{safe_value(dataset, 'NumberOfFrames')}"
    )

    photometric_label.set_text(
        "Photometric Interpretation: "
        f"{safe_value(dataset, 'PhotometricInterpretation')}"
    )

    pixel_spacing_label.set_text(
        "Pixel Spacing: "
        f"{safe_value(dataset, 'PixelSpacing')}"
    )

    slice_thickness_label.set_text(
        "Slice Thickness: "
        f"{safe_value(dataset, 'SliceThickness')}"
    )

    window_center_label.set_text(
        "Window Center: "
        f"{safe_value(dataset, 'WindowCenter')}"
    )

    window_width_label.set_text(
        "Window Width: "
        f"{safe_value(dataset, 'WindowWidth')}"
    )

    sop_class_label.set_text(
        "SOP Class UID: "
        f"{safe_value(dataset, 'SOPClassUID')}"
    )

    study_uid_label.set_text(
        "Study Instance UID: "
        f"{safe_value(dataset, 'StudyInstanceUID')}"
    )

    series_uid_label.set_text(
        "Series Instance UID: "
        f"{safe_value(dataset, 'SeriesInstanceUID')}"
    )

    transfer_syntax_label.set_text(
        "Transfer Syntax UID: "
        f"{get_transfer_syntax(dataset)}"
    )
    if "sequence_name_label" in globals():
        sequence_name_label.set_text(f"Sequence Name: {safe_value(dataset, 'SequenceName')}")
        repetition_time_label.set_text(f"TR: {safe_value(dataset, 'RepetitionTime')}")
        echo_time_label.set_text(f"TE: {safe_value(dataset, 'EchoTime')}")
        flip_angle_label.set_text(f"Flip Angle: {safe_value(dataset, 'FlipAngle')}")
        field_strength_label.set_text(f"Magnetic Field Strength: {safe_value(dataset, 'MagneticFieldStrength')}")


def get_current_window():
    if (
        current_modality != "CT"
        or current_file_type != "DICOM"
    ):
        return None, None

    try:
        center = float(
            window_center_input.value or 40
        )

        width = float(
            window_width_input.value or 400
        )

    except (TypeError, ValueError):
        return 40.0, 400.0

    if width <= 0:
        width = 1.0

    return center, width


# ---------------------------------------------------------
# VIEW TRANSFORMS
# ---------------------------------------------------------

def apply_view_transform() -> None:
    scale_x = -1 if flip_horizontal else 1
    scale_y = -1 if flip_vertical else 1

    transform = (
        f"scale({current_zoom}) "
        f"rotate({current_rotation}deg) "
        f"scaleX({scale_x}) "
        f"scaleY({scale_y})"
    )

    viewer_image.style(
        f"transform: {transform}; "
        "transform-origin: center center;"
    )

    zoom_label.set_text(
        f"Zoom: {int(current_zoom * 100)}%"
    )

    rotation_label.set_text(
        f"Rotation: {current_rotation}°"
    )


def zoom_in() -> None:
    global current_zoom

    current_zoom = min(
        current_zoom + 0.1,
        4.0,
    )

    apply_view_transform()


def zoom_out() -> None:
    global current_zoom

    current_zoom = max(
        current_zoom - 0.1,
        0.25,
    )

    apply_view_transform()


def fit_to_screen() -> None:
    global current_zoom

    current_zoom = 1.0

    apply_view_transform()


def rotate_left() -> None:
    global current_rotation

    current_rotation = (
        current_rotation - 90
    ) % 360

    apply_view_transform()


def rotate_right() -> None:
    global current_rotation

    current_rotation = (
        current_rotation + 90
    ) % 360

    apply_view_transform()


def toggle_flip_horizontal() -> None:
    global flip_horizontal

    flip_horizontal = not flip_horizontal

    apply_view_transform()


def toggle_flip_vertical() -> None:
    global flip_vertical

    flip_vertical = not flip_vertical

    apply_view_transform()


def reset_view() -> None:
    global current_zoom
    global current_rotation
    global flip_horizontal
    global flip_vertical

    current_zoom = 1.0
    current_rotation = 0

    flip_horizontal = False
    flip_vertical = False

    apply_view_transform()


async def enter_fullscreen() -> None:
    await ui.run_javascript(
        """
        const element =
            document.querySelector('.viewer-frame');

        if (
            element &&
            element.requestFullscreen
        ) {
            await element.requestFullscreen();
        }
        """
    )


async def save_screenshot() -> None:
    await ui.run_javascript(
        """
        const viewer =
            document.querySelector('.viewer-frame');

        const image =
            viewer?.querySelector('img');

        if (!image) {
            alert('No image is displayed.');
            return;
        }

        const link =
            document.createElement('a');

        link.href = image.src;

        link.download =
            'medical_imaging_export.png';

        document.body.appendChild(link);

        link.click();

        document.body.removeChild(link);
        """
    )



# ---------------------------------------------------------
# DICOM ANALYSIS TOOLS
# ---------------------------------------------------------

def _dicom_numeric_array(dataset: Any) -> np.ndarray:
    array = np.asarray(dataset.pixel_array, dtype=np.float32)
    if array.ndim > 2:
        array = np.asarray(array[0], dtype=np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    return array * slope + intercept


def _point_to_pixel(point, shape):
    x, y = point
    h, w = shape
    return (
        int(round(max(0, min(float(x), w - 1)))),
        int(round(max(0, min(float(y), h - 1)))),
    )


def clear_analysis_overlay() -> None:
    roi_points.clear()
    viewer_image.set_content("")
    hu_probe_result_label.set_text("Click a CT pixel to read its HU value.")
    roi_result_label.set_text("No ROI selected")


def toggle_hu_probe_mode() -> None:
    global hu_probe_mode, roi_mode, measurement_mode
    if current_modality != "CT" or current_file_type != "DICOM":
        ui.notify("HU Probe is available for CT DICOM images only.", type="warning")
        return
    hu_probe_mode = not hu_probe_mode
    if hu_probe_mode:
        roi_mode = False
        measurement_mode = False
        roi_points.clear()
        measurement_points.clear()
        viewer_image.set_content("")
        measurement_mode_label.set_text("Measurement Mode: OFF")
        roi_mode_label.set_text("ROI Mode: OFF")
        hu_probe_mode_label.set_text("HU Probe: ON")
        ui.notify("HU Probe enabled. Click a point on the CT image.", type="positive")
    else:
        hu_probe_mode_label.set_text("HU Probe: OFF")


def toggle_roi_mode() -> None:
    global roi_mode, hu_probe_mode, measurement_mode
    if current_file_type != "DICOM":
        ui.notify("ROI statistics require a DICOM image.", type="warning")
        return
    roi_mode = not roi_mode
    if roi_mode:
        hu_probe_mode = False
        measurement_mode = False
        roi_points.clear()
        measurement_points.clear()
        viewer_image.set_content("")
        measurement_mode_label.set_text("Measurement Mode: OFF")
        hu_probe_mode_label.set_text("HU Probe: OFF")
        roi_mode_label.set_text("ROI Mode: ON")
        roi_result_label.set_text("Select two opposite corners of a rectangular ROI.")
    else:
        roi_mode_label.set_text("ROI Mode: OFF")


def _update_roi_overlay() -> None:
    if not roi_points:
        viewer_image.set_content("")
        return
    parts = []
    for x, y in roi_points:
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#22c55e" stroke="white" stroke-width="2" />')
    if len(roi_points) == 2:
        x1, y1 = roi_points[0]
        x2, y2 = roi_points[1]
        parts.append(
            f'<rect x="{min(x1,x2)}" y="{min(y1,y2)}" '
            f'width="{abs(x2-x1)}" height="{abs(y2-y1)}" '
            f'fill="none" stroke="#22c55e" stroke-width="3" />'
        )
    viewer_image.set_content("".join(parts))


def _handle_hu_probe(point) -> None:
    if current_dataset is None:
        return
    try:
        array = _dicom_numeric_array(current_dataset)
        x, y = _point_to_pixel(point, array.shape)
        value = float(array[y, x])
        hu_probe_result_label.set_text(f"Pixel ({x}, {y}) | CT value: {value:.1f} HU")
        viewer_image.set_content(
            f'<circle cx="{point[0]}" cy="{point[1]}" r="7" fill="red" stroke="white" stroke-width="2" />'
        )
    except Exception as error:
        ui.notify(f"Unable to read CT value: {error}", type="negative")


def _handle_roi_click(point) -> None:
    if current_dataset is None:
        return
    if len(roi_points) >= 2:
        roi_points.clear()
    roi_points.append(point)
    _update_roi_overlay()
    if len(roi_points) == 1:
        roi_result_label.set_text("First corner selected. Select the opposite corner.")
        return
    try:
        array = _dicom_numeric_array(current_dataset)
        x1, y1 = _point_to_pixel(roi_points[0], array.shape)
        x2, y2 = _point_to_pixel(roi_points[1], array.shape)
        x0, x1 = sorted((x1, x2))
        y0, y1 = sorted((y1, y2))
        region = array[y0:y1+1, x0:x1+1]
        if region.size == 0:
            roi_result_label.set_text("ROI is empty")
            return
        text = (
            f"Mean: {np.mean(region):.1f} | Min: {np.min(region):.1f} | "
            f"Max: {np.max(region):.1f} | SD: {np.std(region):.1f}"
        )
        if current_modality == "CT":
            text = "CT ROI (HU) | " + text
        spacing = getattr(current_dataset, "PixelSpacing", None)
        if spacing is not None and len(spacing) >= 2:
            area = region.shape[0] * float(spacing[0]) * region.shape[1] * float(spacing[1])
            text += f" | Area: {area:.1f} mm²"
        roi_result_label.set_text(text)
    except Exception as error:
        ui.notify(f"Unable to calculate ROI statistics: {error}", type="negative")


def previous_slice() -> None:
    if not current_files:
        return
    new_index = max(current_slice_index - 1, 0)
    slice_slider.value = new_index
    slice_slider.update()
    load_current_image(new_index)


def next_slice() -> None:
    if not current_files:
        return
    new_index = min(current_slice_index + 1, len(current_files) - 1)
    slice_slider.value = new_index
    slice_slider.update()
    load_current_image(new_index)


def show_mpr_preview() -> None:
    if (
        current_file_type != "DICOM"
        or current_modality not in {"CT", "MRI"}
        or len(current_files) < 3
    ):
        ui.notify(
            "Load a multi-slice CT or MRI DICOM series first.",
            type="warning",
        )
        return

    try:
        records = []

        for path in current_files:
            dataset = load_dicom(path)

            array = np.asarray(
                dataset.pixel_array,
                dtype=np.float32,
            )

            if array.ndim > 2:
                array = array[0]

            if current_modality == "CT":
                slope = float(
                    getattr(dataset, "RescaleSlope", 1.0) or 1.0
                )
                intercept = float(
                    getattr(dataset, "RescaleIntercept", 0.0) or 0.0
                )
                array = array * slope + intercept

            position = getattr(
                dataset,
                "ImagePositionPatient",
                None,
            )

            instance_number = getattr(
                dataset,
                "InstanceNumber",
                None,
            )

            pixel_spacing = getattr(
                dataset,
                "PixelSpacing",
                None,
            )

            records.append(
                {
                    "dataset": dataset,
                    "array": array,
                    "position": position,
                    "instance": instance_number,
                    "pixel_spacing": pixel_spacing,
                }
            )

        def sort_key(record):
            position = record["position"]

            if position is not None and len(position) >= 3:
                try:
                    return (0, float(position[2]))
                except Exception:
                    pass

            instance = record["instance"]

            try:
                return (1, float(instance))
            except Exception:
                return (2, 0.0)

        records.sort(key=sort_key)

        slices = [record["array"] for record in records]

        h = min(a.shape[0] for a in slices)
        w = min(a.shape[1] for a in slices)

        volume = np.stack(
            [a[:h, :w] for a in slices],
            axis=0,
        )

        # In-plane spacing: DICOM PixelSpacing = [row_spacing, column_spacing].
        row_spacing = 1.0
        col_spacing = 1.0

        for record in records:
            spacing = record["pixel_spacing"]

            if spacing is not None and len(spacing) >= 2:
                try:
                    row_spacing = float(spacing[0])
                    col_spacing = float(spacing[1])
                    break
                except Exception:
                    pass

        # Prefer geometric slice spacing derived from ImagePositionPatient.
        z_positions = []

        for record in records:
            position = record["position"]

            if position is not None and len(position) >= 3:
                try:
                    z_positions.append(float(position[2]))
                except Exception:
                    pass

        slice_spacing = None

        if len(z_positions) >= 2:
            differences = np.abs(np.diff(np.asarray(z_positions, dtype=np.float32)))
            differences = differences[differences > 1e-6]

            if differences.size:
                slice_spacing = float(np.median(differences))

        if slice_spacing is None:
            first_dataset = records[0]["dataset"]

            for attribute in ("SpacingBetweenSlices", "SliceThickness"):
                value = getattr(first_dataset, attribute, None)

                try:
                    if value is not None and float(value) > 0:
                        slice_spacing = float(value)
                        break
                except Exception:
                    pass

        if slice_spacing is None:
            slice_spacing = row_spacing

        axial = volume[volume.shape[0] // 2]
        coronal = volume[:, volume.shape[1] // 2, :]
        sagittal = volume[:, :, volume.shape[2] // 2]

        def normalize_for_display(array):
            array = np.asarray(array, dtype=np.float32)

            if current_modality == "CT":
                center, width = get_current_window()

                if center is None or width is None:
                    center, width = 40.0, 400.0

                low = float(center) - float(width) / 2.0
                high = float(center) + float(width) / 2.0
            else:
                low, high = np.percentile(array, [1, 99])

            if high <= low:
                low = float(array.min())
                high = float(array.max())

            if high <= low:
                return np.zeros(array.shape, dtype=np.uint8)

            normalized = np.clip(
                (array - low) / (high - low),
                0.0,
                1.0,
            )

            return (normalized * 255.0).astype(np.uint8)

        def resize_for_physical_spacing(
            image_array,
            vertical_spacing,
            horizontal_spacing,
        ):
            image_array = normalize_for_display(image_array)

            rows, cols = image_array.shape[:2]

            physical_height = max(rows * float(vertical_spacing), 1e-6)
            physical_width = max(cols * float(horizontal_spacing), 1e-6)

            target_width = min(max(cols, 256), 900)
            target_height = int(
                round(
                    target_width
                    * physical_height
                    / physical_width
                )
            )

            target_height = min(max(target_height, 96), 900)

            pil_image = Image.fromarray(image_array)
            pil_image = pil_image.resize(
                (target_width, target_height),
                Image.Resampling.BILINEAR,
            )

            return np.asarray(pil_image)

        axial_display = resize_for_physical_spacing(
            axial,
            row_spacing,
            col_spacing,
        )

        coronal_display = resize_for_physical_spacing(
            coronal,
            slice_spacing,
            col_spacing,
        )

        sagittal_display = resize_for_physical_spacing(
            sagittal,
            slice_spacing,
            row_spacing,
        )

        views = {
            "Axial": axial_display,
            "Coronal": coronal_display,
            "Sagittal": sagittal_display,
        }

        spacing_summary = (
            f"Voxel spacing: "
            f"{row_spacing:.3f} × "
            f"{col_spacing:.3f} × "
            f"{slice_spacing:.3f} mm"
        )

        with ui.dialog() as dialog, ui.card().classes(
            "w-[95vw] max-w-6xl"
        ):
            ui.label(
                f"{current_modality} MPR Preview"
            ).classes(
                "text-xl font-semibold"
            )

            ui.label(
                "Research/demo reconstruction using DICOM spatial spacing."
            ).classes(
                "text-sm text-gray-600"
            )

            ui.label(
                spacing_summary
            ).classes(
                "text-xs text-gray-500"
            )

            with ui.row().classes(
                "w-full gap-4 no-wrap items-start"
            ):
                for title, image_array in views.items():
                    with ui.column().classes(
                        "flex-1 items-center min-w-0"
                    ):
                        ui.label(
                            title
                        ).classes(
                            "font-semibold"
                        )

                        ui.image(
                            image_to_data_url(image_array)
                        ).classes(
                            "w-full max-h-[60vh] object-contain bg-black"
                        )

            ui.button(
                "Close",
                on_click=dialog.close,
            ).props(
                "outline"
            )

        dialog.open()

    except Exception as error:
        ui.notify(
            f"Unable to build MPR preview: {error}",
            type="negative",
            position="top",
        )


# ---------------------------------------------------------
# MEASUREMENT
# ---------------------------------------------------------

def update_measurement_overlay() -> None:
    if not measurement_points:
        viewer_image.set_content("")
        return

    svg_parts = []

    for x, y in measurement_points:
        svg_parts.append(
            f'<circle '
            f'cx="{x}" '
            f'cy="{y}" '
            f'r="6" '
            f'fill="red" '
            f'stroke="white" '
            f'stroke-width="2" />'
        )

    if len(measurement_points) == 2:
        x1, y1 = measurement_points[0]
        x2, y2 = measurement_points[1]

        svg_parts.append(
            f'<line '
            f'x1="{x1}" '
            f'y1="{y1}" '
            f'x2="{x2}" '
            f'y2="{y2}" '
            f'stroke="yellow" '
            f'stroke-width="3" />'
        )

    viewer_image.set_content(
        "".join(svg_parts)
    )


def clear_measurement(
    silent: bool = False,
) -> None:
    measurement_points.clear()

    viewer_image.set_content("")

    measurement_result_label.set_text(
        "No measurement"
    )

    if not silent:
        ui.notify(
            "Measurement cleared",
            type="info",
        )


def toggle_measurement_mode() -> None:
    global measurement_mode

    measurement_mode = (
        not measurement_mode
    )

    if measurement_mode:
        global hu_probe_mode, roi_mode
        hu_probe_mode = False
        roi_mode = False
        hu_probe_mode_label.set_text("HU Probe: OFF")
        roi_mode_label.set_text("ROI Mode: OFF")
        reset_view()

        clear_measurement(
            silent=True
        )

        measurement_mode_label.set_text(
            "Measurement Mode: ON"
        )

        ui.notify(
            "Measurement mode enabled. "
            "Click two points on the image.",
            type="positive",
        )

    else:
        measurement_mode_label.set_text(
            "Measurement Mode: OFF"
        )


def handle_image_click(
    event: events.MouseEventArguments,
) -> None:
    point = (float(event.image_x), float(event.image_y))

    if hu_probe_mode:
        _handle_hu_probe(point)
        return

    if roi_mode:
        _handle_roi_click(point)
        return

    if not measurement_mode:
        return

    if len(measurement_points) >= 2:
        clear_measurement(
            silent=True
        )

    measurement_points.append(
        point
    )

    update_measurement_overlay()

    if len(measurement_points) == 1:
        measurement_result_label.set_text(
            "First point selected. "
            "Select the second point."
        )

        return

    point_a = measurement_points[0]
    point_b = measurement_points[1]

    distance_pixels = pixel_distance(
        point_a,
        point_b,
    )

    distance_mm = None

    if (
        current_file_type == "DICOM"
        and current_dataset is not None
    ):
        pixel_spacing = getattr(
            current_dataset,
            "PixelSpacing",
            None,
        )

        try:
            distance_mm = (
                physical_distance_mm(
                    point_a,
                    point_b,
                    pixel_spacing,
                )
            )

        except Exception:
            distance_mm = None

    if distance_mm is not None:
        measurement_result_label.set_text(
            f"Distance: "
            f"{distance_pixels:.2f} px "
            f"| {distance_mm:.2f} mm"
        )

    else:
        measurement_result_label.set_text(
            f"Distance: "
            f"{distance_pixels:.2f} px"
        )


# ---------------------------------------------------------
# ANNOTATIONS
# ---------------------------------------------------------

def save_annotation() -> None:
    text = str(
        annotation_input.value or ""
    ).strip()

    if not text:
        ui.notify(
            "Enter annotation text first.",
            type="warning",
        )
        return

    file_name = (
        current_files[
            current_slice_index
        ].name
        if current_files
        else "No image"
    )

    annotation = (
        f"{file_name}: {text}"
    )

    saved_annotations.append(
        annotation
    )

    annotation_list.options = (
        saved_annotations.copy()
    )

    annotation_list.update()

    annotation_input.value = ""
    annotation_input.update()

    ui.notify(
        "Annotation saved",
        type="positive",
    )


def clear_annotations() -> None:
    saved_annotations.clear()

    annotation_list.options = []
    annotation_list.update()

    ui.notify(
        "Annotations cleared",
        type="info",
    )


# ---------------------------------------------------------
# CT / MRI STANDARD-IMAGE ANALYSIS
# ---------------------------------------------------------

def _confidence_level(confidence: float) -> str:
    if confidence < 0.60:
        return "Low"
    if confidence < 0.80:
        return "Moderate"
    return "High"


def reset_ct_result() -> None:
    ct_prediction_label.set_text("Prediction: Not run")
    ct_noncovid_label.set_text("NonCOVID probability: --")
    ct_covid_label.set_text("COVID probability: --")
    ct_confidence_label.set_text("Model confidence: --")
    ct_confidence_level_label.set_text("Confidence level: --")
    ct_noncovid_progress.value = 0.0
    ct_covid_progress.value = 0.0
    ct_noncovid_progress.update()
    ct_covid_progress.update()


def reset_mri_result() -> None:
    mri_prediction_label.set_text("Prediction: Not run")
    mri_meningioma_label.set_text("Meningioma probability: --")
    mri_glioma_label.set_text("Glioma probability: --")
    mri_pituitary_label.set_text("Pituitary probability: --")
    mri_confidence_label.set_text("Model confidence: --")
    mri_confidence_level_label.set_text("Confidence level: --")
    mri_meningioma_progress.value = 0.0
    mri_glioma_progress.value = 0.0
    mri_pituitary_progress.value = 0.0
    mri_meningioma_progress.update()
    mri_glioma_progress.update()
    mri_pituitary_progress.update()


def update_ct_mri_analysis_availability() -> None:
    reset_ct_result()
    reset_mri_result()
    ct_supported = current_modality == "CT" and current_file_type == "PNG/JPG"
    mri_supported = current_modality == "MRI" and current_file_type == "PNG/JPG"
    ct_analysis_panel.set_visibility(ct_supported)
    mri_analysis_panel.set_visibility(mri_supported)

    if ct_supported:
        ct_status_label.set_text("Status: Ready for CT classification.")
        ct_analysis_button.enable()
    else:
        ct_analysis_button.disable()

    if mri_supported:
        mri_status_label.set_text("Status: Ready for brain MRI classification.")
        mri_analysis_button.enable()
    else:
        mri_analysis_button.disable()


def run_ct_classification() -> None:
    if current_modality != "CT" or current_file_type != "PNG/JPG" or not current_files:
        ui.notify("Load a CT PNG/JPG image first.", type="warning")
        return
    selected_file = current_files[current_slice_index]
    try:
        ct_status_label.set_text("Status: Running classification...")
        ct_analysis_button.disable()
        result = ct_classifier.predict(selected_file)
        noncovid = float(result["noncovid_probability"])
        covid = float(result["covid_probability"])
        confidence = float(result["confidence"])
        ct_prediction_label.set_text(f"Prediction: {result['prediction']}")
        ct_noncovid_label.set_text(f"NonCOVID probability: {noncovid * 100:.1f}%")
        ct_covid_label.set_text(f"COVID probability: {covid * 100:.1f}%")
        ct_noncovid_progress.value = noncovid
        ct_covid_progress.value = covid
        ct_noncovid_progress.update()
        ct_covid_progress.update()
        ct_confidence_label.set_text(f"Model confidence: {confidence * 100:.1f}%")
        ct_confidence_level_label.set_text(f"Confidence level: {_confidence_level(confidence)}")
        ct_threshold_label.set_text(f"Decision threshold: {result['threshold']:.2f}")
        ct_status_label.set_text("Status: Classification complete.")
    except Exception as error:
        ct_status_label.set_text("Status: Classification failed.")
        ui.notify(f"Unable to run CT classification: {error}", type="negative", position="top")
    finally:
        ct_analysis_button.enable()


def run_mri_classification() -> None:
    if current_modality != "MRI" or current_file_type != "PNG/JPG" or not current_files:
        ui.notify("Load a brain MRI PNG/JPG image first.", type="warning")
        return
    selected_file = current_files[current_slice_index]
    try:
        mri_status_label.set_text("Status: Running classification...")
        mri_analysis_button.disable()
        result = mri_classifier.predict(selected_file)
        meningioma = float(result["meningioma_probability"])
        glioma = float(result["glioma_probability"])
        pituitary = float(result["pituitary_probability"])
        confidence = float(result["confidence"])
        mri_prediction_label.set_text(f"Prediction: {result['prediction']}")
        mri_meningioma_label.set_text(f"Meningioma probability: {meningioma * 100:.1f}%")
        mri_glioma_label.set_text(f"Glioma probability: {glioma * 100:.1f}%")
        mri_pituitary_label.set_text(f"Pituitary probability: {pituitary * 100:.1f}%")
        mri_meningioma_progress.value = meningioma
        mri_glioma_progress.value = glioma
        mri_pituitary_progress.value = pituitary
        mri_meningioma_progress.update()
        mri_glioma_progress.update()
        mri_pituitary_progress.update()
        mri_confidence_label.set_text(f"Model confidence: {confidence * 100:.1f}%")
        mri_confidence_level_label.set_text(f"Confidence level: {_confidence_level(confidence)}")
        mri_status_label.set_text("Status: Classification complete.")
    except Exception as error:
        mri_status_label.set_text("Status: Classification failed.")
        ui.notify(f"Unable to run MRI classification: {error}", type="negative", position="top")
    finally:
        mri_analysis_button.enable()


# ---------------------------------------------------------
# THYROID ULTRASOUND CLASSIFICATION
# ---------------------------------------------------------

def reset_classification_result() -> None:
    classification_prediction_label.set_text(
        "Prediction: Not run"
    )
    classification_benign_label.set_text(
        "Benign probability: --"
    )
    classification_malignant_label.set_text(
        "Malignant probability: --"
    )
    classification_confidence_label.set_text(
        "Model confidence: --"
    )
    classification_confidence_level_label.set_text(
        "Confidence level: --"
    )
    classification_warning_label.set_text(
        ""
    )
    classification_threshold_label.set_text(
        "Decision threshold: 0.58"
    )
    benign_progress.value = 0.0
    malignant_progress.value = 0.0
    benign_progress.update()
    malignant_progress.update()


def update_classification_availability() -> None:
    """Update modality-specific analysis panels for standard images."""

    update_ct_mri_analysis_availability()
    reset_classification_result()

    is_supported = (
        current_modality == "Ultrasound"
        and current_file_type == "PNG/JPG"
    )
    classification_panel.set_visibility(is_supported)

    if current_modality != "Ultrasound":
        classification_status_label.set_text(
            "Status: Load an ultrasound image to use this classifier."
        )
        confirm_thyroid_checkbox.value = False
        confirm_thyroid_checkbox.update()
        classification_button.disable()
        return

    if current_file_type != "PNG/JPG":
        classification_status_label.set_text(
            "Status: Classifier currently supports thyroid ultrasound PNG/JPG images only."
        )
        confirm_thyroid_checkbox.value = False
        confirm_thyroid_checkbox.update()
        classification_button.disable()
        return

    classification_status_label.set_text(
        "Status: Confirm the image is a thyroid ultrasound image."
    )

    if bool(confirm_thyroid_checkbox.value):
        classification_button.enable()
    else:
        classification_button.disable()


def handle_thyroid_confirmation(event) -> None:
    if current_modality != "Ultrasound" or current_file_type != "PNG/JPG":
        classification_button.disable()
        return

    if bool(event.value):
        classification_status_label.set_text(
            "Status: Ready for classification."
        )
        classification_button.enable()
    else:
        classification_status_label.set_text(
            "Status: Confirm the image is a thyroid ultrasound image."
        )
        classification_button.disable()


def run_ultrasound_classification() -> None:
    """Run the trained TN5000 V2 classifier on the current standard image."""

    if current_modality != "Ultrasound":
        ui.notify(
            "Classification is available only for ultrasound images.",
            type="warning",
        )
        return

    if current_file_type != "PNG/JPG":
        ui.notify(
            "Load a thyroid ultrasound PNG/JPG image before classification.",
            type="warning",
        )
        return

    if not bool(confirm_thyroid_checkbox.value):
        ui.notify(
            "Confirm that the current image is a thyroid ultrasound image first.",
            type="warning",
        )
        return

    if not current_files:
        ui.notify(
            "No image is currently loaded.",
            type="warning",
        )
        return

    selected_file = current_files[current_slice_index]

    try:
        classification_status_label.set_text(
            "Status: Running classification..."
        )
        classification_button.disable()

        result = ultrasound_classifier.predict(
            selected_file
        )

        classification_prediction_label.set_text(
            f"Prediction: {result['prediction']}"
        )

        benign_probability = float(result['benign_probability'])
        malignant_probability = float(result['malignant_probability'])
        confidence = float(result['confidence'])

        classification_benign_label.set_text(
            "Benign probability: "
            f"{benign_probability * 100:.1f}%"
        )

        classification_malignant_label.set_text(
            "Malignant probability: "
            f"{malignant_probability * 100:.1f}%"
        )

        benign_progress.value = benign_probability
        malignant_progress.value = malignant_probability
        benign_progress.update()
        malignant_progress.update()

        classification_confidence_label.set_text(
            "Model confidence: "
            f"{confidence * 100:.1f}%"
        )

        if confidence < 0.60:
            confidence_level = "Low"
        elif confidence < 0.80:
            confidence_level = "Moderate"
        else:
            confidence_level = "High"

        classification_confidence_level_label.set_text(
            f"Confidence level: {confidence_level}"
        )

        probability_gap = abs(
            benign_probability - malignant_probability
        )

        if probability_gap < 0.20:
            classification_warning_label.set_text(
                "Close probabilities: interpret this result cautiously."
            )
        else:
            classification_warning_label.set_text(
                ""
            )

        classification_threshold_label.set_text(
            "Decision threshold: "
            f"{result['threshold']:.2f}"
        )

        classification_status_label.set_text(
            "Status: Classification complete."
        )

    except Exception as error:
        classification_status_label.set_text(
            "Status: Classification failed."
        )

        ui.notify(
            f"Unable to run thyroid ultrasound classification: {error}",
            type="negative",
            position="top",
        )

    finally:
        if bool(confirm_thyroid_checkbox.value):
            classification_button.enable()


# ---------------------------------------------------------
# IMAGE LOADING
# ---------------------------------------------------------

def load_current_image(
    index: int | float,
) -> None:
    global current_slice_index
    global current_dataset

    if not current_files:
        return

    selected_index = int(
        float(index)
    )

    selected_index = max(
        0,
        min(
            selected_index,
            len(current_files) - 1,
        ),
    )

    current_slice_index = selected_index

    selected_file = (
        current_files[
            selected_index
        ]
    )

    clear_measurement(
        silent=True
    )
    roi_points.clear()
    if "hu_probe_result_label" in globals():
        hu_probe_result_label.set_text("Click a CT pixel to read its HU value.")
    if "roi_result_label" in globals():
        roi_result_label.set_text("No ROI selected")

    update_classification_availability()

    try:
        if current_file_type == "DICOM":
            dataset = load_dicom(
                selected_file
            )

            current_dataset = dataset

            (
                window_center,
                window_width,
            ) = get_current_window()

            base_image = dicom_to_image(
                dataset,
                window_center=window_center,
                window_width=window_width,
            )

            update_metadata(
                dataset
            )

        else:
            current_dataset = None

            base_image = (
                load_standard_image(
                    selected_file
                )
            )

            clear_metadata()

            height = base_image.shape[0]
            width = base_image.shape[1]

            dimensions_label.set_text(
                f"Dimensions: "
                f"{height} × {width}"
            )

        brightness = int(
            float(
                brightness_slider.value
                or 0
            )
        )

        contrast = float(
            contrast_slider.value
            or 1.0
        )

        processed_image = (
            adjust_brightness_contrast(
                base_image,
                brightness=brightness,
                contrast=contrast,
            )
        )

        viewer_image.set_source(
            image_to_data_url(
                processed_image
            )
        )

        image_title.set_text(
            f"{current_modality} Image"
        )

        slice_label.set_text(
            f"Image "
            f"{selected_index + 1} "
            f"of {len(current_files)}"
        )

        file_name_label.set_text(
            f"File: {selected_file.name}"
        )

        current_study_label.set_text(
            f"Current Study: "
            f"{current_source}"
        )

        file_type_label.set_text(
            f"File Type: "
            f"{current_file_type}"
        )

        viewer_status_label.set_text(
            f"Loaded "
            f"{selected_file.name}"
        )

    except Exception as error:
        ui.notify(
            f"Unable to load image: {error}",
            type="negative",
            position="top",
        )


def refresh_image() -> None:
    load_current_image(
        current_slice_index
    )


def configure_files(
    files: list[Path],
    modality: str,
    source: str,
    file_type: str,
) -> None:
    global current_files
    global current_modality
    global current_slice_index
    global current_source
    global current_file_type

    current_files = files
    current_modality = modality
    current_slice_index = 0
    current_source = source
    current_file_type = file_type

    slice_slider.min = 0

    slice_slider.max = max(
        len(files) - 1,
        0,
    )

    slice_slider.value = 0
    slice_slider.update()

    brightness_slider.value = 0
    contrast_slider.value = 1.0

    brightness_slider.update()
    contrast_slider.update()

    window_panel.set_visibility(
        (
            modality == "CT"
            and file_type == "DICOM"
        )
    )

    reset_view()

    load_current_image(0)


# ---------------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------------

def change_sample_modality(
    event,
) -> None:
    modality = str(
        event.value
    )

    if modality not in MODALITY_FOLDERS:
        return

    files = find_dicom_files(
        MODALITY_FOLDERS[
            modality
        ]
    )

    if not files:
        ui.notify(
            "No sample DICOM files found.",
            type="negative",
        )
        return

    configure_files(
        files=files,
        modality=modality,
        source=f"Sample {modality}",
        file_type="DICOM",
    )


def change_slice(
    event,
) -> None:
    load_current_image(
        event.value
    )


# ---------------------------------------------------------
# IMAGE PROCESSING CONTROLS
# ---------------------------------------------------------

def change_brightness(
    event,
) -> None:
    brightness_label.set_text(
        "Brightness: "
        f"{int(float(event.value))}"
    )

    refresh_image()


def change_contrast(
    event,
) -> None:
    contrast_label.set_text(
        "Contrast: "
        f"{float(event.value):.2f}"
    )

    refresh_image()


def apply_window_preset(
    preset_name: str,
) -> None:
    if (
        current_modality != "CT"
        or current_file_type != "DICOM"
    ):
        return

    center, width = (
        CT_WINDOW_PRESETS[
            preset_name
        ]
    )

    window_center_input.value = center
    window_width_input.value = width

    window_center_input.update()
    window_width_input.update()

    refresh_image()


def apply_custom_window() -> None:
    refresh_image()


def reset_controls() -> None:
    brightness_slider.value = 0
    contrast_slider.value = 1.0

    brightness_slider.update()
    contrast_slider.update()

    if (
        current_modality == "CT"
        and current_file_type == "DICOM"
    ):
        window_center_input.value = 40
        window_width_input.value = 400

        window_center_input.update()
        window_width_input.update()

    reset_view()

    refresh_image()


# ---------------------------------------------------------
# UPLOADS
# ---------------------------------------------------------

async def handle_dicom_upload(
    event: events.UploadEventArguments,
) -> None:
    global dicom_uploaded_count

    safe_name = Path(event.file.name).name
    destination = UPLOAD_FOLDER / safe_name

    try:
        await event.file.save(str(destination))
        dicom_uploaded_count += 1
        dicom_upload_status_label.set_text(
            f"DICOM files uploaded: {dicom_uploaded_count}"
        )
    except Exception as error:
        ui.notify(
            f"DICOM upload failed: {error}",
            type="negative",
        )


async def handle_standard_upload(
    event: events.UploadEventArguments,
) -> None:
    global uploaded_standard_images

    # Keep only one standard image at a time. DICOM files remain untouched.
    for existing in list(UPLOAD_FOLDER.glob("*")):
        if (
            existing.is_file()
            and existing.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ):
            try:
                existing.unlink()
            except OSError:
                pass

    safe_name = Path(event.file.name).name
    destination = UPLOAD_FOLDER / safe_name

    try:
        await event.file.save(str(destination))
        uploaded_standard_images = [destination]
        standard_upload_status_label.set_text(
            f"Selected image: {safe_name}"
        )
        ui.notify(
            "Standard image uploaded. Choose its modality and load it.",
            type="positive",
        )
    except Exception as error:
        uploaded_standard_images = []
        ui.notify(
            f"Image upload failed: {error}",
            type="negative",
        )


def process_dicom_uploads() -> None:
    global uploaded_series
    global series_display_map

    classified = classify_uploaded_files(UPLOAD_FOLDER)
    dicom_files = classified["dicom"]

    if not dicom_files:
        uploaded_series = {}
        series_display_map = {}
        uploaded_series_selector.options = []
        uploaded_series_selector.value = None
        uploaded_series_selector.update()
        series_status_label.set_text("DICOM series detected: 0")
        ui.notify(
            "No DICOM files are available to process.",
            type="warning",
        )
        return

    uploaded_series = group_dicom_series(dicom_files)

    series_display_map = {
        data["display_name"]: uid
        for uid, data in uploaded_series.items()
    }

    uploaded_series_selector.options = list(series_display_map.keys())
    uploaded_series_selector.value = None
    uploaded_series_selector.update()

    series_status_label.set_text(
        f"DICOM series detected: {len(uploaded_series)}"
    )

    ui.notify(
        "DICOM study processed. Select a series to view it.",
        type="positive",
    )


def load_selected_dicom_series() -> None:
    selection = (
        uploaded_series_selector.value
    )

    if not selection:
        ui.notify(
            "Select a DICOM series first.",
            type="warning",
        )
        return

    uid = (
        series_display_map[
            selection
        ]
    )

    series = (
        uploaded_series[
            uid
        ]
    )

    configure_files(
        files=series["files"],
        modality=series["modality"],
        source=series["display_name"],
        file_type="DICOM",
    )


def load_standard_image_for_viewing() -> None:
    if not uploaded_standard_images:
        ui.notify(
            "Upload one PNG/JPG image first.",
            type="warning",
        )
        return

    selected_modality = (
        standard_modality_selector.value
    )

    if not selected_modality:
        ui.notify(
            "Select the modality first.",
            type="warning",
        )
        return

    configure_files(
        files=uploaded_standard_images,
        modality=str(
            selected_modality
        ),
        source="Uploaded Standard Image",
        file_type="PNG/JPG",
    )


def clear_uploads() -> None:
    global dicom_uploaded_count
    global uploaded_series
    global series_display_map
    global uploaded_standard_images

    prepare_upload_folder()

    dicom_uploaded_count = 0
    uploaded_series = {}
    series_display_map = {}
    uploaded_standard_images = []

    dicom_upload_status_label.set_text(
        "DICOM files uploaded: 0"
    )
    standard_upload_status_label.set_text(
        "No standard image selected"
    )
    series_status_label.set_text(
        "DICOM series detected: 0"
    )

    uploaded_series_selector.options = []
    uploaded_series_selector.value = None
    uploaded_series_selector.update()

    dicom_upload_control.reset()
    standard_upload_control.reset()

    ui.notify(
        "Uploaded files cleared.",
        type="info",
    )


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------

prepare_upload_folder()

ui.page_title(
    PROJECT_TITLE
)


ui.add_css(
    """
    body {
        background: #f8fafc;
    }

    .viewer-frame {
        overflow: hidden;
        min-height: 520px;
        max-height: 780px;
        width: 100%;
        background: black;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
    }

    .viewer-frame:fullscreen {
        width: 100vw;
        height: 100vh;
        max-height: none;
        border-radius: 0;
    }

    .interactive-viewer {
        max-height: 720px;
        width: 100%;
    }

    .control-panel {
        min-width: 340px;
    }

    .metadata-panel {
        min-width: 360px;
    }
    """
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

with ui.header().classes(
    "items-center justify-between "
    "bg-slate-900 text-white px-6"
):
    ui.label(
        PROJECT_TITLE
    ).classes(
        "text-2xl font-bold"
    )

    ui.label(
        "CT • MRI • Ultrasound"
    )


# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------

with ui.row().classes(
    "w-full no-wrap gap-6 p-6 items-start"
):

    # LEFT PANEL
    with ui.column().classes(
        "control-panel w-80 "
        "bg-slate-100 rounded-lg "
        "p-4 shadow"
    ):
        ui.label(
            "Viewer Controls"
        ).classes(
            "text-xl font-semibold"
        )

        current_study_label = ui.label(
            "Current Study: Sample CT"
        )

        file_type_label = ui.label(
            "File Type: DICOM"
        )

        sample_selector = ui.select(
            options=[
                "CT",
                "MRI",
                "Ultrasound",
            ],
            value="CT",
            label="Load Sample Study",
            on_change=change_sample_modality,
        ).classes(
            "w-full"
        )

        ui.separator()

        ui.label(
            "Upload Study / Image"
        ).classes(
            "text-lg font-semibold"
        )

        ui.label(
            "DICOM Study"
        ).classes(
            "font-semibold text-slate-700"
        )

        ui.label(
            "Choose multiple DICOM slices when they belong to one CT/MRI/US study."
        ).classes(
            "text-xs text-gray-600"
        )

        dicom_upload_status_label = ui.label(
            "DICOM files uploaded: 0"
        )

        dicom_upload_control = ui.upload(
            label="Select DICOM Files",
            on_upload=handle_dicom_upload,
            multiple=True,
            auto_upload=True,
            max_files=2000,
        ).props(
            "accept=.dcm,application/dicom"
        ).classes(
            "w-full"
        )

        ui.button(
            "Process DICOM Study",
            icon="folder_open",
            on_click=process_dicom_uploads,
        ).classes(
            "w-full"
        )

        series_status_label = ui.label(
            "DICOM series detected: 0"
        )

        uploaded_series_selector = ui.select(
            options=[],
            label="Select DICOM Series",
        ).classes(
            "w-full"
        )

        ui.button(
            "Load DICOM Series",
            on_click=load_selected_dicom_series,
        ).classes(
            "w-full"
        )

        ui.separator()

        ui.label(
            "Single PNG/JPG Image"
        ).classes(
            "font-semibold text-slate-700"
        )

        ui.label(
            "Use one standard image at a time. Thyroid classification appears only for Ultrasound."
        ).classes(
            "text-xs text-gray-600"
        )

        standard_upload_status_label = ui.label(
            "No standard image selected"
        )

        standard_upload_control = ui.upload(
            label="Select One PNG/JPG",
            on_upload=handle_standard_upload,
            multiple=False,
            auto_upload=True,
            max_files=1,
        ).props(
            "accept=.png,.jpg,.jpeg,image/png,image/jpeg"
        ).classes(
            "w-full"
        )

        standard_modality_selector = ui.select(
            options=[
                "CT",
                "MRI",
                "Ultrasound",
            ],
            label="Image Modality",
        ).classes(
            "w-full"
        )

        ui.button(
            "Load Standard Image",
            icon="image",
            on_click=load_standard_image_for_viewing,
        ).classes(
            "w-full"
        )

        ui.button(
            "Clear Uploaded Files",
            icon="delete",
            on_click=clear_uploads,
        ).props(
            "outline"
        ).classes(
            "w-full"
        )

        ui.separator()

        slice_label = ui.label(
            "Image 1 of 1"
        )

        slice_slider = ui.slider(
            min=0,
            max=0,
            value=0,
            step=1,
            on_change=change_slice,
        ).classes(
            "w-full"
        )

        with ui.row().classes("w-full gap-2"):
            ui.button("Previous", icon="chevron_left", on_click=previous_slice).classes("flex-1")
            ui.button("Next", icon="chevron_right", on_click=next_slice).classes("flex-1")

        ui.button("MPR Preview", icon="view_in_ar", on_click=show_mpr_preview).props("outline").classes("w-full")

        # CT Windowing
        with ui.column().classes(
            "w-full gap-2"
        ) as window_panel:
            ui.label(
                "CT Windowing"
            ).classes(
                "text-lg font-semibold"
            )

            with ui.row().classes(
                "w-full gap-2"
            ):
                ui.button(
                    "Lung",
                    on_click=lambda:
                    apply_window_preset(
                        "Lung"
                    ),
                )

                ui.button(
                    "Soft Tissue",
                    on_click=lambda:
                    apply_window_preset(
                        "Soft Tissue"
                    ),
                )

                ui.button(
                    "Bone",
                    on_click=lambda:
                    apply_window_preset(
                        "Bone"
                    ),
                )

                ui.button(
                    "Brain",
                    on_click=lambda:
                    apply_window_preset(
                        "Brain"
                    ),
                )

            window_center_input = ui.number(
                "Window Center",
                value=40,
            ).classes(
                "w-full"
            )

            window_width_input = ui.number(
                "Window Width",
                value=400,
                min=1,
            ).classes(
                "w-full"
            )

            ui.button(
                "Apply Custom Window",
                on_click=apply_custom_window,
            ).classes(
                "w-full"
            )

        ui.separator()

        # View tools
        ui.label(
            "View Tools"
        ).classes(
            "text-lg font-semibold"
        )

        zoom_label = ui.label(
            "Zoom: 100%"
        )

        rotation_label = ui.label(
            "Rotation: 0°"
        )

        with ui.row().classes(
            "w-full gap-2"
        ):
            ui.button(
                "Zoom In",
                on_click=zoom_in,
            )

            ui.button(
                "Zoom Out",
                on_click=zoom_out,
            )

        with ui.row().classes(
            "w-full gap-2"
        ):
            ui.button(
                "Rotate Left",
                on_click=rotate_left,
            )

            ui.button(
                "Rotate Right",
                on_click=rotate_right,
            )

        with ui.row().classes(
            "w-full gap-2"
        ):
            ui.button(
                "Flip H",
                on_click=toggle_flip_horizontal,
            )

            ui.button(
                "Flip V",
                on_click=toggle_flip_vertical,
            )

        ui.button(
            "Fit to Screen",
            on_click=fit_to_screen,
        ).classes(
            "w-full"
        )

        ui.button(
            "Full Screen",
            on_click=enter_fullscreen,
        ).classes(
            "w-full"
        )

        ui.button(
            "Save Image",
            on_click=save_screenshot,
        ).classes(
            "w-full"
        )

        ui.separator()

        brightness_label = ui.label(
            "Brightness: 0"
        )

        brightness_slider = ui.slider(
            min=-100,
            max=100,
            value=0,
            step=1,
            on_change=change_brightness,
        ).classes(
            "w-full"
        )

        contrast_label = ui.label(
            "Contrast: 1.00"
        )

        contrast_slider = ui.slider(
            min=0.25,
            max=3.0,
            value=1.0,
            step=0.05,
            on_change=change_contrast,
        ).classes(
            "w-full"
        )

        ui.button(
            "Reset Controls",
            on_click=reset_controls,
        ).classes(
            "w-full"
        )

        ui.separator()

        # Measurements
        ui.label(
            "Measurement"
        ).classes(
            "text-lg font-semibold"
        )

        measurement_mode_label = ui.label(
            "Measurement Mode: OFF"
        )

        ui.button(
            "Toggle Measurement Mode",
            icon="straighten",
            on_click=toggle_measurement_mode,
        ).classes(
            "w-full"
        )

        measurement_result_label = ui.label(
            "No measurement"
        )

        ui.button(
            "Clear Measurement",
            on_click=clear_measurement,
        ).classes(
            "w-full"
        )

        ui.separator()
        ui.label("DICOM Analysis Tools").classes("text-lg font-semibold")

        hu_probe_mode_label = ui.label("HU Probe: OFF")
        ui.button(
            "Toggle HU Probe",
            icon="my_location",
            on_click=toggle_hu_probe_mode,
        ).classes("w-full")
        hu_probe_result_label = ui.label(
            "Click a CT pixel to read its HU value."
        ).classes("text-sm")

        roi_mode_label = ui.label("ROI Mode: OFF")
        ui.button(
            "Toggle ROI Mode",
            icon="crop_square",
            on_click=toggle_roi_mode,
        ).classes("w-full")
        roi_result_label = ui.label("No ROI selected").classes("text-sm break-words")

        ui.button(
            "Clear Analysis Overlay",
            icon="layers_clear",
            on_click=clear_analysis_overlay,
        ).props("outline").classes("w-full")


    # CENTER
    with ui.column().classes(
        "flex-grow min-w-0"
    ):
        image_title = ui.label(
            "CT Image"
        ).classes(
            "text-2xl font-semibold"
        )

        file_name_label = ui.label(
            "File:"
        )

        with ui.element(
            "div"
        ).classes(
            "viewer-frame"
        ):
            viewer_image = ui.interactive_image(
                source="",
                on_mouse=handle_image_click,
                events=["mousedown"],
                cross=False,
            ).classes(
                "interactive-viewer"
            )

        viewer_status_label = ui.label(
            "Viewer ready"
        )


    # RIGHT
    with ui.column().classes(
        "metadata-panel w-80 "
        "bg-slate-100 rounded-lg "
        "p-5 shadow"
    ):
        ui.label(
            "Safe Metadata"
        ).classes(
            "text-2xl font-semibold"
        )

        modality_label = ui.label()
        manufacturer_label = ui.label()
        model_label = ui.label()
        study_date_label = ui.label()
        series_label = ui.label()
        body_part_label = ui.label()
        dimensions_label = ui.label()
        frames_label = ui.label()
        photometric_label = ui.label()
        pixel_spacing_label = ui.label()
        slice_thickness_label = ui.label()
        window_center_label = ui.label()
        window_width_label = ui.label()
        sop_class_label = ui.label()

        study_uid_label = ui.label().classes(
            "break-all"
        )

        series_uid_label = ui.label().classes(
            "break-all"
        )

        transfer_syntax_label = ui.label().classes(
            "break-all"
        )

        with ui.expansion(
            "MRI Acquisition Details",
            icon="settings_input_component",
        ).classes("w-full"):
            sequence_name_label = ui.label("Sequence Name: Not available")
            repetition_time_label = ui.label("TR: Not available")
            echo_time_label = ui.label("TE: Not available")
            flip_angle_label = ui.label("Flip Angle: Not available")
            field_strength_label = ui.label("Magnetic Field Strength: Not available")

        ui.separator()

        ui.label(
            "Annotations"
        ).classes(
            "text-xl font-semibold"
        )

        annotation_input = ui.input(
            label="Annotation text"
        ).classes(
            "w-full"
        )

        ui.button(
            "Save Annotation",
            on_click=save_annotation,
        ).classes(
            "w-full"
        )

        annotation_list = ui.select(
            options=[],
            label="Saved Annotations",
        ).classes(
            "w-full"
        )

        ui.button(
            "Clear Annotations",
            on_click=clear_annotations,
        ).classes(
            "w-full"
        )

        ui.separator()


        with ui.column().classes(
            "w-full gap-2 rounded-lg border border-slate-300 bg-white p-3"
        ) as ct_analysis_panel:
            ui.label("CT Image Analysis").classes("text-xl font-semibold")
            ui.label("Experimental COVID-CT EfficientNetB0 classifier").classes("text-sm text-gray-600")
            ct_status_label = ui.label("Status: Load a CT PNG/JPG image.").classes("text-sm")
            ct_analysis_button = ui.button("Run CT Classification", icon="analytics", on_click=run_ct_classification).classes("w-full")
            ct_analysis_button.disable()
            ct_prediction_label = ui.label("Prediction: Not run").classes("font-semibold")
            ct_noncovid_label = ui.label("NonCOVID probability: --")
            ct_noncovid_progress = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
            ct_covid_label = ui.label("COVID probability: --")
            ct_covid_progress = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
            ct_confidence_label = ui.label("Model confidence: --")
            ct_confidence_level_label = ui.label("Confidence level: --").classes("font-medium")
            ct_threshold_label = ui.label("Decision threshold: 0.46").classes("text-sm text-gray-600")
            ui.button("Reset CT Analysis", icon="restart_alt", on_click=reset_ct_result).props("outline").classes("w-full")
            ui.label("Research/demo output only. COVID-CT image classification is not for clinical diagnosis.").classes("text-xs text-gray-500")

        ct_analysis_panel.set_visibility(False)

        with ui.column().classes(
            "w-full gap-2 rounded-lg border border-slate-300 bg-white p-3"
        ) as mri_analysis_panel:
            ui.label("Brain MRI Analysis").classes("text-xl font-semibold")
            ui.label("Experimental EfficientNetB0 3-class classifier").classes("text-sm text-gray-600")
            mri_status_label = ui.label("Status: Load a brain MRI PNG/JPG image.").classes("text-sm")
            mri_analysis_button = ui.button("Run MRI Classification", icon="analytics", on_click=run_mri_classification).classes("w-full")
            mri_analysis_button.disable()
            mri_prediction_label = ui.label("Prediction: Not run").classes("font-semibold")
            mri_meningioma_label = ui.label("Meningioma probability: --")
            mri_meningioma_progress = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
            mri_glioma_label = ui.label("Glioma probability: --")
            mri_glioma_progress = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
            mri_pituitary_label = ui.label("Pituitary probability: --")
            mri_pituitary_progress = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
            mri_confidence_label = ui.label("Model confidence: --")
            mri_confidence_level_label = ui.label("Confidence level: --").classes("font-medium")
            ui.button("Reset MRI Analysis", icon="restart_alt", on_click=reset_mri_result).props("outline").classes("w-full")
            ui.label("Research/demo output only. This brain MRI classifier is not for clinical diagnosis.").classes("text-xs text-gray-500")

        mri_analysis_panel.set_visibility(False)

        with ui.column().classes(
            "w-full gap-2 rounded-lg border border-slate-300 bg-white p-3"
        ) as classification_panel:
            ui.label(
                "Thyroid Ultrasound Analysis"
            ).classes(
                "text-xl font-semibold"
            )

            ui.label(
                "Experimental TN5000 V2 classifier"
            ).classes(
                "text-sm text-gray-600"
            )

            classification_status_label = ui.label(
                "Status: Load an ultrasound PNG/JPG image."
            ).classes(
                "text-sm"
            )

            confirm_thyroid_checkbox = ui.checkbox(
                "I confirm this is a thyroid ultrasound image",
                value=False,
                on_change=handle_thyroid_confirmation,
            ).classes(
                "w-full"
            )

            classification_button = ui.button(
                "Run Classification",
                icon="analytics",
                on_click=run_ultrasound_classification,
            ).classes(
                "w-full"
            )

            classification_button.disable()

            classification_prediction_label = ui.label(
                "Prediction: Not run"
            ).classes(
                "font-semibold"
            )

            classification_benign_label = ui.label(
                "Benign probability: --"
            )

            benign_progress = ui.linear_progress(
                value=0.0,
                show_value=False,
            ).classes(
                "w-full"
            )

            classification_malignant_label = ui.label(
                "Malignant probability: --"
            )

            malignant_progress = ui.linear_progress(
                value=0.0,
                show_value=False,
            ).classes(
                "w-full"
            )

            classification_confidence_label = ui.label(
                "Model confidence: --"
            )

            classification_confidence_level_label = ui.label(
                "Confidence level: --"
            ).classes(
                "font-medium"
            )

            classification_warning_label = ui.label(
                ""
            ).classes(
                "text-sm text-amber-700"
            )

            classification_threshold_label = ui.label(
                "Decision threshold: 0.58"
            ).classes(
                "text-sm text-gray-600"
            )

            ui.button(
                "Reset Analysis",
                icon="restart_alt",
                on_click=lambda: (
                    reset_classification_result(),
                    classification_status_label.set_text(
                        "Status: Ready for classification."
                    )
                    if bool(confirm_thyroid_checkbox.value)
                    else classification_status_label.set_text(
                        "Status: Confirm the image is a thyroid ultrasound image."
                    )
                ),
            ).props(
                "outline"
            ).classes(
                "w-full"
            )

            ui.label(
                "Research/demo output only. This classifier was trained on the "
                "TN5000 thyroid ultrasound dataset and is not for clinical diagnosis."
            ).classes(
                "text-xs text-gray-500"
            )

classification_panel.set_visibility(False)


def handle_keyboard(event) -> None:
    try:
        if not event.action.keydown:
            return
        if event.key in {"ArrowLeft", "ArrowUp"}:
            previous_slice()
        elif event.key in {"ArrowRight", "ArrowDown"}:
            next_slice()
    except Exception:
        return


ui.keyboard(on_key=handle_keyboard)


# ---------------------------------------------------------
# LOAD INITIAL CT
# ---------------------------------------------------------

sample_files = find_dicom_files(
    MODALITY_FOLDERS["CT"]
)

if sample_files:
    configure_files(
        files=sample_files,
        modality="CT",
        source="Sample CT",
        file_type="DICOM",
    )


ui.run(
    title=PROJECT_TITLE,
    port=8090,
    reload=True,
)