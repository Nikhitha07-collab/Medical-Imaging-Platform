from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Any

from nicegui import ui
from PIL import Image

from utils.dicom_loader import load_dicom
from utils.preprocessing import (
    adjust_brightness_contrast,
    dicom_to_image,
)


PROJECT_TITLE = "Medical Imaging Platform"

MODALITY_FOLDERS: dict[str, Path] = {
    "CT": Path("test_data/CT/chest_ct/27548"),
    "MRI": Path("test_data/MRI/abdomen_mri/80231"),
    "Ultrasound": Path(
        "test_data/Ultrasound/thyroid_us/46711"
    ),
}


current_files: list[Path] = []
current_modality = "CT"
current_slice_index = 0


def image_to_data_url(image_array) -> str:
    """Convert a NumPy image into a PNG data URL."""

    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    encoded = b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def find_dicom_files(folder: Path) -> list[Path]:
    """Find DICOM files recursively inside a folder."""

    files = sorted(folder.rglob("*.dcm"))

    if files:
        return files

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
    )


def safe_value(
    dataset: Any,
    attribute: str,
) -> str:
    """Return a safe display value from a DICOM dataset."""

    value = getattr(
        dataset,
        attribute,
        "Not available",
    )

    if value in (None, ""):
        return "Not available"

    return str(value)


def update_metadata(dataset: Any) -> None:
    """Update the metadata panel."""

    modality_label.set_text(
        f"Modality: {safe_value(dataset, 'Modality')}"
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

    body_part_label.set_text(
        "Body Part: "
        f"{safe_value(dataset, 'BodyPartExamined')}"
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


def refresh_image() -> None:
    """Reload the current image using current controls."""

    load_slice(current_slice_index)


def load_slice(index: int | float) -> None:
    """Load and display one DICOM image."""

    global current_slice_index

    if not current_files:
        return

    selected_index = int(float(index))
    selected_index = max(
        0,
        min(
            selected_index,
            len(current_files) - 1,
        ),
    )

    current_slice_index = selected_index
    selected_file = current_files[selected_index]

    try:
        dataset = load_dicom(selected_file)
        base_image = dicom_to_image(dataset)

        brightness = int(
            float(brightness_slider.value or 0)
        )

        contrast = float(
            contrast_slider.value or 1.0
        )

        processed_image = adjust_brightness_contrast(
            base_image,
            brightness=brightness,
            contrast=contrast,
        )

        dicom_image.set_source(
            image_to_data_url(processed_image)
        )

        image_title.set_text(
            f"{current_modality} Image"
        )

        slice_label.set_text(
            f"Image {selected_index + 1} "
            f"of {len(current_files)}"
        )

        brightness_label.set_text(
            f"Brightness: {brightness}"
        )

        contrast_label.set_text(
            f"Contrast: {contrast:.2f}"
        )

        file_name_label.set_text(
            f"File: {selected_file.name}"
        )

        update_metadata(dataset)

    except Exception as error:
        ui.notify(
            f"Unable to load DICOM image: {error}",
            type="negative",
            position="top",
        )


def change_modality(event) -> None:
    """Load the selected modality."""

    global current_files
    global current_modality
    global current_slice_index

    selected_modality = str(event.value)

    if selected_modality not in MODALITY_FOLDERS:
        return

    current_modality = selected_modality
    current_slice_index = 0

    selected_folder = MODALITY_FOLDERS[
        selected_modality
    ]

    current_files = find_dicom_files(
        selected_folder
    )

    if not current_files:
        ui.notify(
            "No DICOM files found inside "
            f"{selected_folder}",
            type="negative",
            position="top",
        )
        return

    slice_slider.min = 0
    slice_slider.max = len(current_files) - 1
    slice_slider.value = 0
    slice_slider.update()

    reset_controls(
        show_notification=False
    )

    load_slice(0)

    ui.notify(
        f"{selected_modality} study loaded",
        type="positive",
        position="top",
    )


def change_slice(event) -> None:
    """Handle slice-slider changes."""

    load_slice(event.value)


def change_brightness(event) -> None:
    """Handle brightness changes."""

    brightness_label.set_text(
        f"Brightness: {int(float(event.value))}"
    )

    refresh_image()


def change_contrast(event) -> None:
    """Handle contrast changes."""

    contrast_label.set_text(
        f"Contrast: {float(event.value):.2f}"
    )

    refresh_image()


def reset_controls(
    show_notification: bool = True,
) -> None:
    """Reset image controls."""

    brightness_slider.value = 0
    brightness_slider.update()

    contrast_slider.value = 1.0
    contrast_slider.update()

    if current_files:
        refresh_image()

    if show_notification:
        ui.notify(
            "Viewer controls reset",
            type="info",
            position="top",
        )


ui.page_title(PROJECT_TITLE)

ui.add_css(
    """
    body {
        background: #f8fafc;
    }

    .viewer-image img {
        object-fit: contain;
        max-height: 760px;
        background: black;
    }

    .control-panel {
        min-width: 270px;
    }

    .metadata-panel {
        min-width: 330px;
    }
    """
)


with ui.header().classes(
    "items-center justify-between "
    "bg-slate-900 text-white px-6"
):
    ui.label(PROJECT_TITLE).classes(
        "text-2xl font-bold"
    )

    ui.label(
        "CT • MRI • Ultrasound"
    ).classes(
        "text-sm"
    )


with ui.row().classes(
    "w-full no-wrap gap-6 p-6 items-start"
):

    with ui.column().classes(
        "control-panel w-72 "
        "bg-slate-100 rounded-lg p-4 shadow"
    ):
        ui.label(
            "Viewer Controls"
        ).classes(
            "text-xl font-semibold"
        )

        modality_selector = ui.select(
            options=[
                "CT",
                "MRI",
                "Ultrasound",
            ],
            value="CT",
            label="Select modality",
            on_change=change_modality,
        ).classes(
            "w-full"
        ).props(
            "outlined"
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
            icon="restart_alt",
            on_click=reset_controls,
        ).classes(
            "w-full"
        )

        ui.separator()

        ui.label(
            "Viewer mode only. "
            "AI modules will be added later."
        ).classes(
            "text-sm text-gray-600"
        )


    with ui.column().classes(
        "flex-grow items-center min-w-0"
    ):
        image_title = ui.label(
            "CT Image"
        ).classes(
            "text-2xl font-semibold self-start"
        )

        file_name_label = ui.label(
            "File:"
        ).classes(
            "text-xs text-gray-500 self-start"
        )

        dicom_image = ui.image().classes(
            "viewer-image w-full max-w-4xl "
            "rounded-lg shadow-lg bg-black"
        )


    with ui.column().classes(
        "metadata-panel w-80 "
        "bg-slate-100 rounded-lg p-5 shadow"
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


current_files = find_dicom_files(
    MODALITY_FOLDERS["CT"]
)

if current_files:
    slice_slider.max = len(current_files) - 1
    slice_slider.update()
    load_slice(0)
else:
    ui.notify(
        "No CT DICOM files found.",
        type="negative",
        position="top",
    )


ui.run(
    title=PROJECT_TITLE,
    port=8090,
    reload=True,
)