from pathlib import Path

import streamlit as st

from utils.dicom_loader import load_dicom
from utils.preprocessing import dicom_to_image


st.set_page_config(
    page_title="Medical Imaging AI Platform",
    page_icon="🩻",
    layout="wide",
)

st.title("Medical Imaging AI Platform")
st.caption(
    "Multi-modality DICOM viewer for CT, MRI, and Ultrasound "
    "with future AI integration."
)

ct_folder = Path("test_data/CT/chest_ct/27548")
dicom_files = sorted(ct_folder.glob("*.dcm"))

if not dicom_files:
    st.error(f"No DICOM files were found inside: {ct_folder}")
    st.stop()

selected_index = st.sidebar.slider(
    "CT slice",
    min_value=0,
    max_value=len(dicom_files) - 1,
    value=0,
)

selected_file = dicom_files[selected_index]

try:
    dataset = load_dicom(selected_file)
    image = dicom_to_image(dataset)

    st.sidebar.success(f"Loaded slice {selected_index + 1} of {len(dicom_files)}")

    image_column, metadata_column = st.columns([2, 1])

    with image_column:
        st.subheader("CT image")
        st.image(image, clamp=True, width="stretch")

    with metadata_column:
        st.subheader("Safe metadata")
        st.write("**Modality:**", getattr(dataset, "Modality", "Not available"))
        st.write(
            "**Manufacturer:**",
            getattr(dataset, "Manufacturer", "Not available"),
        )
        st.write(
            "**Series Description:**",
            getattr(dataset, "SeriesDescription", "Not available"),
        )
        st.write("**Rows:**", getattr(dataset, "Rows", "Not available"))
        st.write("**Columns:**", getattr(dataset, "Columns", "Not available"))
        st.write(
            "**Photometric Interpretation:**",
            getattr(dataset, "PhotometricInterpretation", "Not available"),
        )

except Exception as error:
    st.error(str(error))