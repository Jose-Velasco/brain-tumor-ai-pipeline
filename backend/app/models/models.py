from typing import Callable
import torch
import torch.nn as nn
from monai.networks.nets.densenet import DenseNet121
from monai.networks.nets.unet import UNet
from enum import StrEnum, auto

class ModelName(StrEnum):
    DEV_MODEL = auto()

# Registry maps model_name -> builder function
# Builder signature: (device: torch.device) -> nn.Module
MODEL_REGISTRY: dict[ModelName, Callable[[torch.device], nn.Module]] = {}

def build_model_from_name(model_name: ModelName, device: torch.device) -> nn.Module:
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
    def decorator(fn: Callable[[torch.device], nn.Module]):
        # def decorator(fn: Callable[[torch.device], nn.Module]):
        model_name = name
        if model_name in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' already registered.")
        MODEL_REGISTRY[model_name] = fn
        return fn
    return decorator

@register_model(ModelName.DEV_MODEL)
def build_small_unet(device: torch.device) -> nn.Module:
    """
    Small 3D UNet for dev/testing segmentation pipeline.
    Input:  [B, 4, H, W, D]
    Output: [B, 4, H, W, D] logits
    """
    model = UNet(
        spatial_dims=3,
        in_channels=4,      # 4 MRI modalities
        out_channels=4,     # 4 tumor/background classes
        channels=(16, 32, 64),  # small, for speed
        strides=(2, 2),
        num_res_units=1,
    ).to(device)
    model.eval()
    return model

# add more models here
