# model loading + global access
# Serve entrypoint for inference, loading the model.
# Heavy model object is managed is initialized when the Ray Serve deployment starts.
from typing import Callable
from fastapi import FastAPI, HTTPException
from ray import serve
import torch

from backend.app.schemas.predict import PredictRequest, PredictResponse
from app.services.inference import run_inference
from app.models.models import MODEL_REGISTRY, ModelName, build_model_from_name
import torch.nn as nn

# FastAPI app for model endpoints only
model_app = FastAPI(title="Model API")

@serve.deployment(
    route_prefix="/model",        # e.g. http://host:8000/model/api/predict
    ray_actor_options={"num_gpus": 1}) # or 0 for CPU
@serve.ingress(model_app)
class ModelService:
    """
    Ray Serve deployment for HEAVY model inference endpoints.
    Loads the model once per replica on startup.
    """
    def __init__(self):
        self.device: torch.device = self._get_device()
        self.models: dict[ModelName, Callable[[torch.device], nn.Module]] = MODEL_REGISTRY

    @model_app.post("/api/predict", response_model=PredictResponse)
    async def predict(self, payload: PredictRequest):
        model_name = payload.model_name

        if model_name not in self.models:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown or disabled model: {model_name}",
            )

        model = build_model_from_name(model_name, self.device)
        # TODO: add logic to preprocess 3d image based on model chosen to align with trained preprocess
        return run_inference(model, self.device, payload)
    
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device('cuda')  # CUDA GPU
        elif torch.backends.mps.is_available():
            device = torch.device('mps') #Apple GPU
        else:
            device = torch.device("cpu")
        return device

# Object used by `serve run`
model_app_deployment = ModelService.bind()