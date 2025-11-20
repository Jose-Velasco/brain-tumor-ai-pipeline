# Core inference logic in one place
import numpy as np
import torch

from ..schemas.predict import PredictRequest, PredictResponse

def run_inference(model: torch.nn.Module, device: torch.device, payload: PredictRequest) -> PredictResponse:
    """Shared inference logic: used by both uvicorn and Ray Serve."""
    vol_np = np.array(payload.volume, dtype=np.float32).reshape(payload.shape)
    image = torch.from_numpy(vol_np).unsqueeze(0).to(device)  # [1, C, D, H, W]

    with torch.no_grad():
        logits = model(image)              # [1, C, D, H, W]
        pred = torch.argmax(logits, dim=1) # [1, D, H, W]

    seg = pred.squeeze(0).cpu().numpy()

    return PredictResponse(
        shape=list(seg.shape),
        segmentation=seg.astype(np.int16).ravel().tolist(),
    )