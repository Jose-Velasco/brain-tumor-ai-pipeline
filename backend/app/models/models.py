from typing import Callable
import torch
import torch.nn as nn
from monai.networks.nets.densenet import DenseNet121
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
def build_densenet121(device: torch.device) -> nn.Module:
    """
    Small 3D DenseNet with pretrained weights.
    Great for fast dev and testing.
    """
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        pretrained=False,
    ).to(device)
    model.eval()
    return model

# add more models here
