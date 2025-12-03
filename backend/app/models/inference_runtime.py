from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from typing import Tuple
import torch.nn as nn
from monai.inferers.utils import sliding_window_inference

class OnnxSegmentationModel:
    """
    Thin wrapper around an ONNX Runtime session so it looks like a nn.Module-ish object.
    Expects input as numpy or torch [1, C, H, W, D].
    """
    def __init__(self, model_path: Path, use_cuda: bool = True):
        self.model_path = Path(model_path)
        providers: list[str] = []

        if use_cuda:
            # will fall back to CPU if CUDA EP not available
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            self.model_path.as_posix(),
            providers=providers,
        )

        # Cache I/O names
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        assert len(inputs) == 1, "Expected single input for ONNX model"
        assert len(outputs) == 1, "Expected single output for ONNX model"
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        x: [1, C, H, W, D] (torch or numpy).
        Returns torch.Tensor [1, num_classes, H, W, D] (or similar), depending on export.
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        ort_out = self.session.run(
            [self.output_name],
            {self.input_name: x_np},
        )[0]  # numpy array

        return torch.from_numpy(ort_out)  # caller decides device later if needed

class TorchSegmentationModel:
    """
    Thin wrapper around a MONAI/PyTorch segmentation model so it behaves similar
    to the OnnxSegmentationModel.

    Expects input as numpy or torch [1, C, H, W, D].
    Uses MONAI's sliding_window_inference under the hood.

    Returns torch.Tensor [1, num_classes, H, W, D].
    """

    def __init__(
        self,
        model: nn.Module,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 1,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.model = model

        self.roi_size = roi_size        # (H, W, D)
        self.sw_batch_size = sw_batch_size

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        x: [1, C, H, W, D] (torch or numpy).
        Returns: torch.Tensor [1, num_classes, H, W, D] on CPU.
        """
        # convert to torch, float32
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x)
        else:
            x_t = x

        # ensure 5D
        if x_t.ndim != 5:
            raise ValueError(f"Expected input with 5 dims [B,C,H,W,D], got {x_t.shape}")

        x_t = x_t.to(self.device, dtype=torch.float32)

        #  sliding window inference like was done when training our model 
        # outputs: [B, num_classes, D, H, W]
        logits = sliding_window_inference(
            inputs=x_t,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
        )

        # move to CPU
        logits = logits.cpu() # pyright: ignore[reportAttributeAccessIssue]

        return logits
