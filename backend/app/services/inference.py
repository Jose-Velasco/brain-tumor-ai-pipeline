# Core inference logic in one place
import numpy as np
import torch

from ..schemas.predict import PredictRequest, PredictResponse

@torch.inference_mode()
def run_inference(model: torch.nn.Module, device: torch.device, payload: PredictRequest) -> PredictResponse:
    """Shared inference logic: used by both uvicorn and Ray Serve."""
    vol_np = np.array(payload.volume, dtype=np.float32).reshape(payload.shape) # [C, H, W, D]
    image = torch.from_numpy(vol_np).unsqueeze(0).to(device)  # [1, C, H, W, D]

    print(f"{vol_np.shape = }")
    print(f"{image.shape = }")
    # with torch.no_grad():
    logits = model(image)              # [1, C, H, W, D]
    pred = torch.argmax(logits, dim=1) # [1, H, W, D]
    print(f"{pred.shape = }")
    # Crop depth to match input
    if pred.shape[-1] != image.shape[-1]:
        _, _, H, W, D_in = image.shape
        pred = pred[..., :D_in]          # now aligns: (1,240,240,155)
        print(f"pred was reshaped at D dim to align with original img{pred.shape = }")

    seg = pred.squeeze(0).cpu().numpy()
    print(f"{seg.shape = }")

    return PredictResponse(
        shape=list(seg.shape),
        segmentation=seg.astype(np.int16).ravel().tolist(),
    )