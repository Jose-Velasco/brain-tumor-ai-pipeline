from typing import Callable, Protocol
import torch
# from monai.networks.nets.unet import UNet
from enum import StrEnum, auto
from app.models.onnx_runtime import OnnxSegmentationModel
from ..core.config import settings

class ModelName(StrEnum):
    DEV_MODEL = auto()

class InferenceFn(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor: 
        ...

# Registry maps model_name -> builder function
# Builder signature: (device: torch.device) -> InferenceFn
MODEL_REGISTRY: dict[ModelName, Callable[[torch.device], InferenceFn]] = {}

def build_model_from_name(model_name: ModelName, device: torch.device) -> InferenceFn:
    """
    Factory to build a model by name using the registry.

    Raises:
        KeyError if model_name is not registered.
        RuntimeError (from builder) if model is disabled/misconfigured.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](device)

def register_model(name: ModelName):
    """
    Decorator to register a model builder function.

    Usage:
        @register_model(ModelName.DEV_MODEL)
        def build_densenet121(device: torch.device) -> nn.Module: ...

    """
    def decorator(fn: Callable[[torch.device], InferenceFn]):
        model_name = name
        if model_name in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' already registered.")
        MODEL_REGISTRY[model_name] = fn
        return fn
    return decorator

# @register_model(ModelName.DEV_MODEL)
# def build_small_unet(device: torch.device) -> nn.Module:
#     """
#     Small 3D UNet for dev/testing segmentation pipeline.
#     Input:  [B, 4, H, W, D]
#     Output: [B, 4, H, W, D] logits
#     """
#     model = UNet(
#         spatial_dims=3,
#         in_channels=4,      # 4 MRI modalities
#         out_channels=4,     # 4 tumor/background classes
#         channels=(16, 32, 64),  # small, for speed
#         strides=(2, 2),
#         num_res_units=1,
#     ).to(device)
#     model.eval()
#     return model

@register_model(ModelName.DEV_MODEL)
def build_small_unet(device: torch.device) -> InferenceFn:
    """
    Small 3D UNet for dev/testing segmentation pipeline.
    Input:  [B, 4, H, W, D]
    Output: [B, 4, H, W, D] logits
    """
    model_dir = settings.model_root_dir / "dev_model"
    model_path = model_dir / "unetr_dev.onnx"
    return OnnxSegmentationModel(model_path, use_cuda=device.type == "cuda")

# add more models here
