# Pydantic models for request/response
from pydantic import BaseModel
from ..models.models import ModelName

class PredictRequest(BaseModel):
    """Input volume as flattened list plus shape + model selection."""
    volume: list[float]
    shape: list[int]  # [B, C, H, W, D] or [1, 4, 128, 128, 155]
    # model_name: Literal["dev_model", "unetr"] = "dev_model"
    model_name: ModelName  = ModelName.DEV_MODEL


class PredictResponse(BaseModel):
    """Predicted segmentation as flattened label map."""
    shape: list[int]
    segmentation: list[int]
