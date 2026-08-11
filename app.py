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
    find_dicom_files,
    group_dicom_series,
    prepare_upload_folder,
)


PROJECT_TITLE = "Medical Imaging Platform"

MODALITY_FOLDERS = {
    "CT": Path("test_data/CT/chest_ct/27548"),
    "MRI": Path("test_data/MRI/abdomen_mri/80231"),
    "Ultrasound": Path("test_data/Ultrasound/thyroid_us/46711"),
}

CT_WINDOW_PRESETS = {
    "Lung": (-600.0, 1500.0),
    "Soft Tissue": (40.0, 400.0),
    "Bone": (300.0, 1500.0),
    "Brain": (40.0, 80.0),
}


current_files: list[Path] = []
current_modality = "CT"
current_slice_index = 0
current_zoom = 1.0
current_rotation = 0
flip_horizontal = False
flip_vertical = False
current_source = "Sample CT"

uploaded_file_count = 0
uploaded_series: dict[str, dict] = {}
series_display_map: dict[str, str] = {}


def image_to_data_url(image_array) -> str:
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    encoded = b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def safe_value(dataset: Any, attribute: str) -> str:
    value = getattr(
        dataset,
        attribute,
        "Not available",
    )

    if value in (None, ""):
        return "Not available"

    return str(value)


def get_transfer_syntax(dataset: Any) -> str:
    try:
        return str(
            dataset.file_meta.TransferSyntaxUID
        )
    except Exception:
        return "Not available"


def update_metadata(dataset: Any) -> None:
    modality_label.set_text(
        f"Modality: {safe_value(dataset, 'Modality')}"
    )

    manufacturer_label.set_text(
        f"Manufacturer: {safe_value(dataset, 'Manufacturer')}"
    )

    model_label.set_text(
        f"Model: {safe_value(dataset, 'ManufacturerModelName')}"
    )

    study_date_label.set_text(
        f"Study Date: {safe_value(dataset, 'StudyDate')}"
    )

    series_label.set_text(
        "Series Description: "
        f"{safe_value(dataset, 'SeriesDescription')}"
    )

    body_part_label.set_text(
        f"Body Part: {safe_value(dataset, 'BodyPartExamined')}"
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
        f"Pixel Spacing: {safe_value(dataset, 'PixelSpacing')}"
    )

    slice_thickness_label.set_text(
        f"Slice Thickness: {safe_value(dataset, 'SliceThickness')}"
    )

    window_center_label.set_text(
        f"Window Center: {safe_value(dataset, 'WindowCenter')}"
    )

    window_width_label.set_text(
        f"Window Width: {safe_value(dataset, 'WindowWidth')}"
    )

    sop_class_label.set_text(
        f"SOP Class UID: {safe_value(dataset, 'SOPClassUID')}"
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
        f"Transfer Syntax UID: {get_transfer_syntax(dataset)}"
    )


def get_current_window():
    if current_modality != "CT":
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


def apply_view_transform() -> None:
    scale_x = -1 if flip_horizontal else 1
    scale_y = -1 if flip_vertical else 1

    transform = (
        f"scale({current_zoom}) "
        f"rotate({current_rotation}deg) "
        f"scaleX({scale_x}) "
        f"scaleY({scale_y})"
    )

    dicom_image.style(
        f"transform: {transform}; "
        "transform-origin: center center; "
        "transition: transform 0.15s ease;"
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
        3.0,
    )

    apply_view_transform()


def zoom_out() -> None:
    global current_zoom

    current_zoom = max(
        current_zoom - 0.1,
        0.5,
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


async def enter_fullscreen() -> None:
    await ui.run_javascript(
        """
        const element = document.querySelector('.viewer-frame');
        if (element && element.requestFullscreen) {
            element.requestFullscreen();
        }
        """
    )


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


def load_slice(index: int | float) -> None:
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

    current_slice_index = selected_index
    selected_file = current_files[selected_index]

    try:
        dataset = load_dicom(
            selected_file
        )

        window_center, window_width = (
            get_current_window()
        )

        base_image = dicom_to_image(
            dataset,
            window_center=window_center,
            window_width=window_width,
        )

        brightness = int(
            float(
                brightness_slider.value or 0
            )
        )

        contrast = float(
            contrast_slider.value or 1.0
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

        file_name_label.set_text(
            f"File: {selected_file.name}"
        )

        current_study_label.set_text(
            f"Current Study: {current_source}"
        )

        brightness_label.set_text(
            f"Brightness: {brightness}"
        )

        contrast_label.set_text(
            f"Contrast: {contrast:.2f}"
        )

        viewer_status_label.set_text(
            f"Loaded {selected_file.name}"
        )

        update_metadata(dataset)

    except Exception as error:
        ui.notify(
            f"Unable to load DICOM image: {error}",
            type="negative",
            position="top",
        )


def refresh_image() -> None:
    load_slice(
        current_slice_index
    )


def set_window_visibility() -> None:
    window_panel.set_visibility(
        current_modality == "CT"
    )


def configure_study(
    files: list[Path],
    modality: str,
    source: str,
) -> None:
    global current_files
    global current_modality
    global current_slice_index
    global current_source

    current_files = files
    current_modality = modality
    current_slice_index = 0
    current_source = source

    slice_slider.min = 0
    slice_slider.max = max(
        len(files) - 1,
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
        window_width_input.value = 400

        window_center_input.update()
        window_width_input.update()

    set_window_visibility()
    reset_view()
    load_slice(0)


def change_modality(event) -> None:
    selected_modality = str(
        event.value
    )

    if selected_modality not in MODALITY_FOLDERS:
        return

    files = find_dicom_files(
        MODALITY_FOLDERS[
            selected_modality
        ]
    )

    if not files:
        ui.notify(
            "No valid DICOM files found.",
            type="negative",
        )
        return

    configure_study(
        files,
        selected_modality,
        f"Sample {selected_modality}",
    )


def change_slice(event) -> None:
    load_slice(
        event.value
    )


def change_brightness(event) -> None:
    brightness_label.set_text(
        f"Brightness: {int(float(event.value))}"
    )

    refresh_image()


def change_contrast(event) -> None:
    contrast_label.set_text(
        f"Contrast: {float(event.value):.2f}"
    )

    refresh_image()


def apply_window_preset(
    preset_name: str,
) -> None:
    if current_modality != "CT":
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
    if current_modality != "CT":
        return

    refresh_image()


def reset_controls() -> None:
    brightness_slider.value = 0
    contrast_slider.value = 1.0

    brightness_slider.update()
    contrast_slider.update()

    if current_modality == "CT":
        window_center_input.value = 40
        window_width_input.value = 400

        window_center_input.update()
        window_width_input.update()

    reset_view()
    refresh_image()


async def handle_dicom_upload(
    event: events.UploadEventArguments,
) -> None:
    global uploaded_file_count

    safe_name = Path(
        event.file.name
    ).name

    destination = (
        UPLOAD_FOLDER / safe_name
    )

    try:
        await event.file.save(
            str(destination)
        )

        uploaded_file_count += 1

        upload_status_label.set_text(
            f"Uploaded files: {uploaded_file_count}"
        )

    except Exception as error:
        ui.notify(
            f"Upload failed: {error}",
            type="negative",
        )


def analyze_uploaded_files() -> None:
    global uploaded_series
    global series_display_map

    files = find_dicom_files(
        UPLOAD_FOLDER
    )

    if not files:
        ui.notify(
            "No valid uploaded DICOM files found.",
            type="negative",
        )
        return

    uploaded_series = (
        group_dicom_series(files)
    )

    series_display_map = {
        data["display_name"]: uid
        for uid, data
        in uploaded_series.items()
    }

    uploaded_series_selector.options = list(
        series_display_map.keys()
    )

    uploaded_series_selector.value = None
    uploaded_series_selector.update()

    series_status_label.set_text(
        f"Detected series: {len(uploaded_series)}"
    )


def load_selected_uploaded_series() -> None:
    selected_display = (
        uploaded_series_selector.value
    )

    if not selected_display:
        ui.notify(
            "Select an uploaded series first.",
            type="warning",
        )
        return

    series_uid = (
        series_display_map[
            selected_display
        ]
    )

    series = (
        uploaded_series[
            series_uid
        ]
    )

    configure_study(
        files=series["files"],
        modality=series["modality"],
        source=series["display_name"],
    )


def clear_uploaded_study() -> None:
    global uploaded_file_count
    global uploaded_series
    global series_display_map

    prepare_upload_folder()

    uploaded_file_count = 0
    uploaded_series = {}
    series_display_map = {}

    upload_status_label.set_text(
        "Uploaded files: 0"
    )

    series_status_label.set_text(
        "Detected series: 0"
    )

    uploaded_series_selector.options = []
    uploaded_series_selector.value = None
    uploaded_series_selector.update()

    upload_control.reset()


prepare_upload_folder()

ui.page_title(PROJECT_TITLE)

ui.add_css(
    """
    body {
        background: #f8fafc;
    }

    .viewer-frame {
        overflow: auto;
        min-height: 500px;
        max-height: 780px;
        width: 100%;
        background: black;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 10px;
        position: relative;
    }

    .viewer-frame:fullscreen {
        width: 100vw;
        height: 100vh;
        max-height: none;
        background: black;
    }

    .viewer-image img {
        object-fit: contain;
        max-height: 720px;
        user-select: none;
    }

    .viewer-frame:fullscreen .viewer-image img {
        max-height: 95vh;
    }

    .control-panel {
        min-width: 320px;
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
    )


with ui.row().classes(
    "w-full no-wrap gap-6 p-6 items-start"
):

    with ui.column().classes(
        "control-panel w-80 "
        "bg-slate-100 rounded-lg p-4 shadow"
    ):

        ui.label(
            "Viewer Controls"
        ).classes(
            "text-xl font-semibold"
        )

        current_study_label = ui.label(
            "Current Study: Sample CT"
        )

        modality_selector = ui.select(
            ["CT", "MRI", "Ultrasound"],
            value="CT",
            label="Load Sample Study",
            on_change=change_modality,
        ).classes(
            "w-full"
        )

        ui.separator()

        ui.label(
            "Upload DICOM Files"
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
            max_files=2000,
        ).props(
            "accept=.dcm,application/dicom"
        ).classes(
            "w-full"
        )

        ui.button(
            "Analyze Uploaded Files",
            icon="search",
            on_click=analyze_uploaded_files,
        ).classes(
            "w-full"
        )

        series_status_label = ui.label(
            "Detected series: 0"
        )

        uploaded_series_selector = ui.select(
            options=[],
            label="Select Uploaded Series",
        ).classes(
            "w-full"
        )

        ui.button(
            "Load Selected Series",
            icon="folder_open",
            on_click=load_selected_uploaded_series,
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

        with ui.column().classes(
            "w-full gap-2"
        ) as window_panel:

            ui.label(
                "CT Windowing"
            ).classes(
                "text-lg font-semibold"
            )

            with ui.row().classes(
                "w-full"
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

            with ui.row().classes(
                "w-full"
            ):
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
                icon="zoom_in",
                on_click=zoom_in,
            ).classes(
                "flex-1"
            )

            ui.button(
                "Zoom Out",
                icon="zoom_out",
                on_click=zoom_out,
            ).classes(
                "flex-1"
            )

        with ui.row().classes(
            "w-full gap-2"
        ):
            ui.button(
                "Rotate Left",
                icon="rotate_left",
                on_click=rotate_left,
            ).classes(
                "flex-1"
            )

            ui.button(
                "Rotate Right",
                icon="rotate_right",
                on_click=rotate_right,
            ).classes(
                "flex-1"
            )

        with ui.row().classes(
            "w-full gap-2"
        ):
            ui.button(
                "Flip H",
                on_click=toggle_flip_horizontal,
            ).classes(
                "flex-1"
            )

            ui.button(
                "Flip V",
                on_click=toggle_flip_vertical,
            ).classes(
                "flex-1"
            )

        ui.button(
            "Fit to Screen",
            icon="fit_screen",
            on_click=fit_to_screen,
        ).classes(
            "w-full"
        )

        ui.button(
            "Full Screen",
            icon="fullscreen",
            on_click=enter_fullscreen,
        ).classes(
            "w-full"
        )

        ui.button(
            "Reset View",
            icon="refresh",
            on_click=reset_view,
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

            dicom_image = ui.image().classes(
                "viewer-image w-full max-w-4xl"
            )

        viewer_status_label = ui.label(
            "Viewer ready"
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


sample_files = find_dicom_files(
    MODALITY_FOLDERS["CT"]
)

if sample_files:
    configure_study(
        sample_files,
        "CT",
        "Sample CT",
    )


ui.run(
    title=PROJECT_TITLE,
    port=8090,
    reload=True,
)