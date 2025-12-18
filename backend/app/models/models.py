from typing import Callable, Protocol
import torch
from monai.networks.nets.unet import UNet
from monai.networks.nets.segresnet import SegResNet
from monai.networks.nets.flexible_unet import FlexibleUNet

from enum import StrEnum, auto
from app.models.inference_runtime import OnnxSegmentationModel, TorchSegmentationModel
from ..core.config import settings

# TODO: make this share between frontend and back. for now add model here and in frontend too
class ModelName(StrEnum):
    DEV_MODEL = auto()
    SEGRESNET_TEACHER_TRAINED = auto()
    UNET_STUDENT_TRAINED = auto()
    
    UNET_TEACHER_PRETRAINED_SEGRESNET = auto()
    FLEXABLEUNET_TEACHER_PRETRAINED_SEGRESNET = auto()
    FLEXABLEUNET_STUDENT_PRETRAINED_SwinUNETR = auto()

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

@register_model(ModelName.SEGRESNET_TEACHER_TRAINED)
def build_segresnet_trained(device: torch.device) -> InferenceFn:
    """
    segresnet_trained for dev/testing segmentation pipeline.

    Input:  [B, 4, H, W, D]

    Output: [B, 4, H, W, D] logits
    """
    model_dir = settings.model_root_dir / "segresnet"
    model_weight_path = model_dir / "teacher17.pth"
    teacher_trained = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
    ).to(device)

    teacher_trained.load_state_dict(torch.load(model_weight_path, map_location=device))
    teacher_trained.eval()
    return TorchSegmentationModel(teacher_trained)

@register_model(ModelName.UNET_STUDENT_TRAINED)
def build_unet_trained(device: torch.device) -> InferenceFn:
    """
    segresnet_trained for dev/testing segmentation pipeline.

    Input:  [B, 4, H, W, D]

    Output: [B, 4, H, W, D] logits
    """
    model_dir = settings.model_root_dir / "unet"
    model_weight_path = model_dir / "student_trained_teacher17.pth"
    student = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2,2,2),
        num_res_units=1,
    ).to(device)

    student.load_state_dict(torch.load(model_weight_path, map_location=device))
    student.eval()
    return TorchSegmentationModel(student)

@register_model(ModelName.FLEXABLEUNET_STUDENT_PRETRAINED_SwinUNETR)
def build_FLEXABLEUNET_pretrained_SwinUNETR(device: torch.device) -> InferenceFn:
    """
    segresnet_trained for dev/testing segmentation pipeline.

    Input:  [B, 4, H, W, D]

    Output: [B, 4, H, W, D] logits
    """
    model_dir = settings.model_root_dir / "flex"
    model_weight_path = model_dir / "swin_student_trained_teacher.pth"
    student = FlexibleUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        backbone="resnet18",
        pretrained=False,
    ).to(device)

    student.load_state_dict(torch.load(model_weight_path, map_location=device))
    student.eval()
    return TorchSegmentationModel(student)

@register_model(ModelName.UNET_TEACHER_PRETRAINED_SEGRESNET)
def build_unet_pretrained_SEGRESNET(device: torch.device) -> InferenceFn:
    """
    segresnet_trained for dev/testing segmentation pipeline.

    Input:  [B, 4, H, W, D]

    Output: [B, 4, H, W, D] logits
    """
    model_dir = settings.model_root_dir / "unet"
    model_weight_path = model_dir / "s-UNet_t-SegResNet-pretrained_2025-12-17_17-10-52.pth"
    student = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2,2,2),
        num_res_units=1,
        dropout=0.1
    ).to(device)

    student.load_state_dict(torch.load(model_weight_path, map_location=device))
    student.eval()
    return TorchSegmentationModel(student)

@register_model(ModelName.FLEXABLEUNET_TEACHER_PRETRAINED_SEGRESNET)
def build_FLEXABLEUNET_pretrained_SEGRESNET(device: torch.device) -> InferenceFn:
    """
    segresnet_trained for dev/testing segmentation pipeline.

    Input:  [B, 4, H, W, D]

    Output: [B, 4, H, W, D] logits
    """
    model_dir = settings.model_root_dir / "flex"
    model_weight_path = model_dir / "s-FelxibleUNett_t-SegResNet-pretrained_2025-12-17_17-9-6.pth"
    student = FlexibleUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        backbone="resnet18",   # lightweight student
        pretrained=False,     # no ImageNet weights for 3D
        dropout=0.1
    ).to(device)

    student.load_state_dict(torch.load(model_weight_path, map_location=device))
    student.eval()
    return TorchSegmentationModel(student)