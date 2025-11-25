# HTTP calls to /model and /api/report
import json
import requests
import numpy as np
from typing import Dict, Any
from mri_app.config import settings
from mri_app.pdf_report import png_bytes_to_base64

def run_segmentation(image: np.ndarray, model_name: str) -> np.ndarray:
    """Call /model/api/predict and return mask [H,W,D]."""
    # image: [C,H,W,D]
    payload = {
        "volume": image.astype("float32").ravel().tolist(),
        "shape": list(image.shape),
        "model_name": model_name,
    }
    resp = requests.post(settings.MODEL_PREDICT_URL, json=payload, timeout=60)
    # print(f"{data['segmentation']}")
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()
    print(f"{data['shape'] = }")
    seg = np.array(data["segmentation"], dtype=np.int32).reshape(data["shape"])
    return seg  # [H,W,D]

def generate_llm_report(case_report: Dict[str, Any], images_bytes: list[bytes]) -> str:
    """
    Calls Ollama api and return report text
    images_bytes = [png_bytes_axial, png_bytes_coronal, png_bytes_sagittal] that will be sent to llm
    """
    # convert each PNG to base64 text
    images_b64 = [png_bytes_to_base64(p) for p in images_bytes]
    prompt = (
        "You are a radiology assistant.\n"
        "Here is the case information in JSON format:\n"
        f"{json.dumps(case_report, indent=2)}\n\n"
        "Write a medical-style findings summary based on this and the provided images. Include a disclaimer at the end of the report"
    )

    payload = {
        "model": settings.model_name,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
    }

    resp = requests.post(settings.ollama_url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["response"]