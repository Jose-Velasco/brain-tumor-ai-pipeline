# HTTP calls to /model and /api/report
import requests
import numpy as np
from typing import Dict, Any
from mri_app. config import settings

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

def generate_llm_report(case_report: Dict[str, Any]) -> str:
    """Call /api/health/report (LLM via Ollama) and return report text."""
    payload = {"case_report": case_report, "include_images": False}
    resp = requests.post(settings.REPORT_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["text"]