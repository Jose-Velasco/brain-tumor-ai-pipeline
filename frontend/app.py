from enum import StrEnum, auto
from typing import Any
import streamlit as st
from mri_app.visualization import make_overlay_figure
from mri_app.backend_client import run_segmentation, generate_llm_report
from mri_app.io import load_nifti_uploaded
from mri_app.pdf_report import build_case_report, render_slice_overlays, create_pdf_report

# add models here based on the ones setup in backend/models/models.py
class ModelName(StrEnum):
    DEV_MODEL = auto()

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("Brain Tumor Segmentation Demo")

uploaded = st.file_uploader("Upload NIfTI (.nii or .nii.gz)", type=["nii", "nii.gz"])

if uploaded is not None:
    image, spacing, case_id = load_nifti_uploaded(uploaded)  # [C, H, W, D]
    st.success(f"Loaded case {case_id} with shape {image.shape}")

    # model_name = st.selectbox("Model", ["dev_model"], index=0)
    model_name = st.selectbox("Model", [model_names.value for model_names in ModelName], index=0)

    if st.button("Run Segmentation"):
        with st.spinner("Running model..."):
            mask = run_segmentation(image, model_name)  # [H, W, D]
            print(f"{mask.shape = }")
            st.session_state["image"] = image
            st.session_state["mask"] = mask
            st.session_state["spacing"] = spacing
            st.session_state["case_id"] = case_id
        st.success("Segmentation complete!")

if "image" in st.session_state and "mask" in st.session_state:
    image = st.session_state["image"]
    mask = st.session_state["mask"]
    spacing = st.session_state["spacing"]
    case_id = st.session_state["case_id"]

    modality_idx = 3
    D = image.shape[modality_idx]
    slice_idx = st.slider("Axial slice index", 0, D - 1, D // 2)

    fig = make_overlay_figure(image, mask, slice_idx, modality_idx=modality_idx, figsize=(4,4))

    st.pyplot(fig, width="content")

    if st.button("Generate LLM Report"):
        with st.spinner("Asking Gemma (via Ollama)..."):
            case_report: dict[str, Any] = build_case_report(case_id, image, mask, spacing)
            llm_text = generate_llm_report(case_report)
            st.session_state["llm_text"] = llm_text
        st.success("Report generated.")
    
    if "llm_text" in st.session_state:
        st.subheader("LLM Narrative Report")
        st.write(st.session_state["llm_text"])

        # # PDF generation
        # slice_images = render_slice_overlays(image, mask)  # dict[name] -> png bytes
        # pdf_bytes = create_pdf_report(case_report, slice_images, st.session_state["llm_text"])

        # st.download_button(
        #     "Download PDF report",
        #     data=pdf_bytes,
        #     file_name=f"{case_id}_report.pdf",
        #     mime="application/pdf",
        # )

# streamlit run app.py --logger.level=debug