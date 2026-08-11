from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Any

from nicegui import events, ui
from PIL import Image

from utils.dicom_loader import load_dicom
from utils.preprocessing import (
    adjust_brightness_contrast,
    dicom_to_image,
)
from utils.upload_handler import (
    UPLOAD_FOLDER,
    detect_modality,
    find_dicom_files,
    prepare_upload_folder,
)


PROJECT_TITLE = "Medical Imaging Platform"


MODALITY_FOLDERS: dict[str, Path] = {
    "CT": Path(
        "test_data/CT/chest_ct/27548"
    ),
    "MRI": Path(
        "test_data/MRI/abdomen_mri/80231"
    ),
    "Ultrasound": Path(
        "test_data/Ultrasound/"
        "thyroid_us/46711"
    ),
}


CT_WINDOW_PRESETS: dict[
    str,
    tuple[float, float],
] = {
    "Lung": (
        -600.0,
        1500.0,
    ),
    "Soft Tissue": (
        40.0,
        400.0,
    ),
    "Bone": (
        300.0,
        1500.0,
    ),
    "Brain": (
        40.0,
        80.0,
    ),
}


current_files: list[Path] = []

current_modality = "CT"

current_slice_index = 0

uploaded_file_count = 0

current_zoom = 1.0

current_source = "Sample CT"


def image_to_data_url(
    image_array,
) -> str:
    """Convert a NumPy image into a PNG data URL."""

    image = Image.fromarray(
        image_array
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    encoded = b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return (
        "data:image/png;base64,"
        f"{encoded}"
    )


def safe_value(
    dataset: Any,
    attribute: str,
) -> str:
    """Read a safe DICOM metadata value."""

    value = getattr(
        dataset,
        attribute,
        "Not available",
    )

    if value in (
        None,
        "",
    ):
        return "Not available"

    return str(value)


def get_transfer_syntax(
    dataset: Any,
) -> str:
    """Read Transfer Syntax UID safely."""

    try:
        value = (
            dataset.file_meta
            .TransferSyntaxUID
        )

        return str(value)

    except Exception:
        return "Not available"


def update_metadata(
    dataset: Any,
) -> None:
    """Update safe DICOM metadata."""

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

    body_part_label.set_text(
        "Body Part: "
        f"{safe_value(dataset, 'BodyPartExamined')}"
    )

    dimensions_label.set_text(
        "Dimensions: "
        f"{safe_value(dataset, 'Rows')}"
        " × "
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


def get_current_window(
) -> tuple[
    float | None,
    float | None,
]:
    """Get current CT window values."""

    if current_modality != "CT":
        return (
            None,
            None,
        )

    try:
        center = float(
            window_center_input.value
            or 40
        )

        width = float(
            window_width_input.value
            or 400
        )

    except (
        TypeError,
        ValueError,
    ):
        return (
            40.0,
            400.0,
        )

    if width <= 0:
        width = 1.0

    return (
        center,
        width,
    )


def apply_zoom() -> None:
    """Apply image zoom."""

    zoom_percent = int(
        current_zoom * 100
    )

    zoom_label.set_text(
        f"Zoom: {zoom_percent}%"
    )

    dicom_image.style(
        f"transform: scale({current_zoom}); "
        "transform-origin: center center;"
    )


def zoom_in() -> None:
    """Zoom into the image."""

    global current_zoom

    current_zoom = min(
        current_zoom + 0.1,
        3.0,
    )

    apply_zoom()


def zoom_out() -> None:
    """Zoom out of the image."""

    global current_zoom

    current_zoom = max(
        current_zoom - 0.1,
        0.5,
    )

    apply_zoom()


def fit_to_screen() -> None:
    """Reset zoom to 100 percent."""

    global current_zoom

    current_zoom = 1.0

    apply_zoom()


def load_slice(
    index: int | float,
) -> None:
    """Load and display one DICOM image."""

    global current_slice_index

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

    current_slice_index = (
        selected_index
    )

    selected_file = (
        current_files[
            selected_index
        ]
    )

    try:
        dataset = load_dicom(
            selected_file
        )

        (
            window_center,
            window_width,
        ) = get_current_window()

        base_image = dicom_to_image(
            dataset,
            window_center=window_center,
            window_width=window_width,
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

        dicom_image.set_source(
            image_to_data_url(
                processed_image
            )
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

        viewer_status_label.set_text(
            f"Loaded {selected_file.name}"
        )

        current_study_label.set_text(
            f"Current Study: {current_source}"
        )

        update_metadata(
            dataset
        )

    except Exception as error:

        viewer_status_label.set_text(
            "Unable to display image"
        )

        ui.notify(
            "Unable to load DICOM image: "
            f"{error}",
            type="negative",
            position="top",
        )


def refresh_image() -> None:
    """Refresh the active image."""

    load_slice(
        current_slice_index
    )


def set_window_visibility() -> None:
    """Show window controls for CT only."""

    window_panel.set_visibility(
        current_modality == "CT"
    )


def configure_study(
    files: list[Path],
    modality: str,
    source: str,
) -> None:
    """Configure viewer for a study."""

    global current_files
    global current_modality
    global current_slice_index
    global current_zoom
    global current_source

    current_files = files

    current_modality = modality

    current_slice_index = 0

    current_zoom = 1.0

    current_source = source

    slice_slider.min = 0

    slice_slider.max = max(
        len(current_files) - 1,
        0,
    )

    slice_slider.value = 0

    slice_slider.update()

    brightness_slider.value = 0

    brightness_slider.update()

    contrast_slider.value = 1.0

    contrast_slider.update()

    if modality == "CT":

        window_center_input.value = 40

        window_center_input.update()

        window_width_input.value = 400

        window_width_input.update()

    set_window_visibility()

    apply_zoom()

    load_slice(0)


def change_modality(
    event,
) -> None:
    """Load included sample data."""

    selected_modality = str(
        event.value
    )

    if (
        selected_modality
        not in MODALITY_FOLDERS
    ):
        return

    selected_folder = (
        MODALITY_FOLDERS[
            selected_modality
        ]
    )

    files = find_dicom_files(
        selected_folder
    )

    if not files:

        ui.notify(
            "No valid DICOM files found in "
            f"{selected_folder}",
            type="negative",
            position="top",
        )

        return

    configure_study(
        files=files,
        modality=selected_modality,
        source=(
            f"Sample {selected_modality}"
        ),
    )

    ui.notify(
        f"{selected_modality} "
        "sample study loaded",
        type="positive",
        position="top",
    )


def change_slice(
    event,
) -> None:
    """Handle slice changes."""

    load_slice(
        event.value
    )


def change_brightness(
    event,
) -> None:
    """Handle brightness changes."""

    brightness_label.set_text(
        "Brightness: "
        f"{int(float(event.value))}"
    )

    refresh_image()


def change_contrast(
    event,
) -> None:
    """Handle contrast changes."""

    contrast_label.set_text(
        "Contrast: "
        f"{float(event.value):.2f}"
    )

    refresh_image()


def apply_window_preset(
    preset_name: str,
) -> None:
    """Apply a CT window preset."""

    if current_modality != "CT":

        ui.notify(
            "Window presets are available "
            "only for CT studies.",
            type="warning",
            position="top",
        )

        return

    center, width = (
        CT_WINDOW_PRESETS[
            preset_name
        ]
    )

    window_center_input.value = (
        center
    )

    window_center_input.update()

    window_width_input.value = (
        width
    )

    window_width_input.update()

    refresh_image()

    ui.notify(
        f"{preset_name} window applied",
        type="info",
        position="top",
    )


def apply_custom_window() -> None:
    """Apply custom CT window values."""

    if current_modality != "CT":

        ui.notify(
            "Custom windowing is available "
            "only for CT studies.",
            type="warning",
            position="top",
        )

        return

    try:
        width = float(
            window_width_input.value
        )

        float(
            window_center_input.value
        )

    except (
        TypeError,
        ValueError,
    ):

        ui.notify(
            "Enter valid window values.",
            type="negative",
            position="top",
        )

        return

    if width <= 0:

        ui.notify(
            "Window Width must be "
            "greater than zero.",
            type="negative",
            position="top",
        )

        return

    refresh_image()

    ui.notify(
        "Custom CT window applied",
        type="info",
        position="top",
    )


def reset_controls(
    show_notification: bool = True,
) -> None:
    """Reset all viewer controls."""

    global current_zoom

    brightness_slider.value = 0

    brightness_slider.update()

    contrast_slider.value = 1.0

    contrast_slider.update()

    current_zoom = 1.0

    apply_zoom()

    if current_modality == "CT":

        window_center_input.value = 40

        window_center_input.update()

        window_width_input.value = 400

        window_width_input.update()

    if current_files:
        refresh_image()

    if show_notification:

        ui.notify(
            "Viewer controls reset",
            type="info",
            position="top",
        )


async def handle_dicom_upload(
    event: events.UploadEventArguments,
) -> None:
    """Save one uploaded DICOM file."""

    global uploaded_file_count

    UPLOAD_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_file_name = Path(
        event.file.name
    ).name

    if not safe_file_name:

        ui.notify(
            "Invalid uploaded filename.",
            type="negative",
            position="top",
        )

        return

    destination = (
        UPLOAD_FOLDER
        / safe_file_name
    )

    try:
        await event.file.save(
            str(destination)
        )

        uploaded_file_count += 1

        upload_status_label.set_text(
            "Uploaded files: "
            f"{uploaded_file_count}"
        )

    except Exception as error:

        ui.notify(
            "Unable to save "
            f"{safe_file_name}: {error}",
            type="negative",
            position="top",
        )


def load_uploaded_study() -> None:
    """Load uploaded DICOM study."""

    uploaded_files = (
        find_dicom_files(
            UPLOAD_FOLDER
        )
    )

    if not uploaded_files:

        ui.notify(
            "No valid uploaded DICOM "
            "files were found.",
            type="negative",
            position="top",
        )

        return

    try:
        detected_modality = (
            detect_modality(
                uploaded_files[0]
            )
        )

        configure_study(
            files=uploaded_files,
            modality=detected_modality,
            source=(
                "Uploaded "
                f"{detected_modality}"
            ),
        )

        ui.notify(
            "Uploaded "
            f"{detected_modality} "
            "study loaded",
            type="positive",
            position="top",
        )

    except Exception as error:

        ui.notify(
            "Unable to load uploaded "
            f"study: {error}",
            type="negative",
            position="top",
        )


def clear_uploaded_study() -> None:
    """Clear temporary uploads."""

    global uploaded_file_count

    prepare_upload_folder()

    uploaded_file_count = 0

    upload_status_label.set_text(
        "Uploaded files: 0"
    )

    upload_control.reset()

    ui.notify(
        "Uploaded files cleared",
        type="info",
        position="top",
    )


prepare_upload_folder()


ui.page_title(
    PROJECT_TITLE
)


ui.add_css(
    """
    body {
        background: #f8fafc;
    }

    .viewer-image img {
        object-fit: contain;
        max-height: 720px;
        background: black;
    }

    .viewer-frame {
        overflow: auto;
        min-height: 500px;
        max-height: 780px;
        width: 100%;
        background: #000000;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
    }

    .control-panel {
        min-width: 310px;
    }

    .metadata-panel {
        min-width: 350px;
    }
    """
)


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
    ).classes(
        "text-sm"
    )


with ui.row().classes(
    "w-full no-wrap gap-6 "
    "p-6 items-start"
):

    # LEFT CONTROL PANEL
    with ui.column().classes(
        "control-panel w-72 "
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
        ).classes(
            "text-sm text-gray-600"
        )

        modality_selector = ui.select(
            options=[
                "CT",
                "MRI",
                "Ultrasound",
            ],
            value="CT",
            label="Load Sample Study",
            on_change=change_modality,
        ).classes(
            "w-full"
        ).props(
            "outlined"
        )

        ui.separator()

        ui.label(
            "Upload DICOM Study"
        ).classes(
            "text-lg font-semibold"
        )

        upload_status_label = ui.label(
            "Uploaded files: 0"
        )

        upload_control = ui.upload(
            label="Choose DICOM files",
            on_upload=handle_dicom_upload,
            multiple=True,
            auto_upload=True,
            max_files=1000,
        ).props(
            "accept=.dcm,application/dicom"
        ).classes(
            "w-full"
        )

        ui.button(
            "Load Uploaded Study",
            icon="folder_open",
            on_click=load_uploaded_study,
        ).classes(
            "w-full"
        )

        ui.button(
            "Clear Uploads",
            icon="delete",
            on_click=clear_uploaded_study,
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

        ui.separator()

        # CT WINDOW CONTROLS
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
                ).props(
                    "outline dense"
                ).classes(
                    "flex-1"
                )

                ui.button(
                    "Soft Tissue",
                    on_click=lambda:
                    apply_window_preset(
                        "Soft Tissue"
                    ),
                ).props(
                    "outline dense"
                ).classes(
                    "flex-1"
                )

            with ui.row().classes(
                "w-full gap-2"
            ):

                ui.button(
                    "Bone",
                    on_click=lambda:
                    apply_window_preset(
                        "Bone"
                    ),
                ).props(
                    "outline dense"
                ).classes(
                    "flex-1"
                )

                ui.button(
                    "Brain",
                    on_click=lambda:
                    apply_window_preset(
                        "Brain"
                    ),
                ).props(
                    "outline dense"
                ).classes(
                    "flex-1"
                )

            window_center_input = ui.number(
                label="Window Center",
                value=40,
                step=1,
            ).classes(
                "w-full"
            )

            window_width_input = ui.number(
                label="Window Width",
                value=400,
                min=1,
                step=1,
            ).classes(
                "w-full"
            )

            ui.button(
                "Apply Custom Window",
                icon="tune",
                on_click=apply_custom_window,
            ).classes(
                "w-full"
            )

        ui.separator()

        # ZOOM CONTROLS
        ui.label(
            "Zoom"
        ).classes(
            "text-lg font-semibold"
        )

        zoom_label = ui.label(
            "Zoom: 100%"
        )

        with ui.row().classes(
            "w-full gap-2"
        ):

            ui.button(
                "Zoom In",
                icon="zoom_in",
                on_click=zoom_in,
            ).props(
                "outline dense"
            ).classes(
                "flex-1"
            )

            ui.button(
                "Zoom Out",
                icon="zoom_out",
                on_click=zoom_out,
            ).props(
                "outline dense"
            ).classes(
                "flex-1"
            )

        ui.button(
            "Fit to Screen",
            icon="fit_screen",
            on_click=fit_to_screen,
        ).props(
            "outline"
        ).classes(
            "w-full"
        )

        ui.separator()

        # BRIGHTNESS / CONTRAST
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
            "Educational medical imaging "
            "viewer. AI modules will be "
            "integrated later."
        ).classes(
            "text-sm text-gray-600"
        )

    # CENTER IMAGE VIEWER
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

        with ui.element(
            "div"
        ).classes(
            "viewer-frame"
        ):

            dicom_image = ui.image().classes(
                "viewer-image "
                "w-full max-w-4xl"
            )

        viewer_status_label = ui.label(
            "Viewer ready"
        ).classes(
            "text-sm text-gray-600 self-start"
        )

    # RIGHT METADATA PANEL
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


sample_ct_files = find_dicom_files(
    MODALITY_FOLDERS["CT"]
)


if sample_ct_files:

    configure_study(
        files=sample_ct_files,
        modality="CT",
        source="Sample CT",
    )

else:

    ui.notify(
        "No sample CT DICOM "
        "files were found.",
        type="negative",
        position="top",
    )


ui.run(
    title=PROJECT_TITLE,
    port=8090,
    reload=True,
)