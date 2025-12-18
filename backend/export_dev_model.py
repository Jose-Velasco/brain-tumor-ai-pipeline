from monai.networks.nets.unet import UNet
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

model = UNet(
    spatial_dims=3,
    in_channels=4,      # 4 MRI modalities
    out_channels=4,     # 4 tumor/background classes
    channels=(16, 32, 64),  # small, for speed
    strides=(2, 2),
    num_res_units=1,
).to("cpu")
model.eval()

onnx_path: str = "unetr_dev.onnx"
H: int = 240
W: int = 240
D: int = 155

# Dummy input: [B, C, H, W, D]
dummy = torch.randn(1, 4, H, W, D, dtype=torch.float32, device="cpu")

# Optional: dynamic axes for batch + spatial dims
dynamic_axes = {
    "input":  {0: "batch", 2: "height", 3: "width", 4: "depth"},
    "output": {0: "batch", 2: "height", 3: "width", 4: "depth"},
}

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
    opset_version=17,
    do_constant_folding=True,
)

print(f"Exported UNet to {onnx_path}")

sess = ort.InferenceSession("/app/backend/app/models/artifacts/dev_model/unetr_dev.onnx")
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

dummy_np = np.random.randn(1, 4, 240, 240, 155).astype("float32")
out = sess.run([out_name], {inp_name: dummy_np})[0]
print("ONNX output shape:", out.shape)  # should be [1, 4, 240, 240, 155] (or clo