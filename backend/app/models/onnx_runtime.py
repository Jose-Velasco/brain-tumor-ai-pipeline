from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

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