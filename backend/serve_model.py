# model loading + global access
# Serve entrypoint for inference, loading the model.
# Heavy model object is managed is initialized when the Ray Serve deployment starts.
from typing import Callable, cast
from fastapi import FastAPI, HTTPException
from ray import serve
import torch

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.inference import prepare_model_inference_response, prepare_payload_for_inference, run_inference
from app.models.models import MODEL_REGISTRY, ModelName, build_model_from_name
import torch.nn as nn

from app.models.image_transformations import get_standard_student_teacher_transform

# FastAPI app for model endpoints only
model_app = FastAPI(title="Model API")

@serve.deployment(
    # route_prefix="/model",        # e.g. http://host:8000/model/api/predict
    ray_actor_options={"num_gpus": 1}) # or 0 for CPU
@serve.ingress(model_app)
class ModelService:
    """
    Ray Serve deployment for HEAVY model inference endpoints.
    Loads the model once per replica on startup.
    """
    def __init__(self):
        self.device: torch.device = self._get_device()
        self.models: dict[ModelName, Callable[[torch.device], Callable[[torch.Tensor], torch.Tensor]]] = MODEL_REGISTRY

    @model_app.post("/api/predict", response_model=PredictResponse)
    async def predict(self, payload: PredictRequest) -> PredictResponse:
        model_name = payload.model_name

        if model_name not in self.models:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown or disabled model: {model_name}",
            )

        model = build_model_from_name(model_name, self.device)

        vol_np, image = prepare_payload_for_inference(device=self.device, payload=payload)
        print(f"In ModelService: {type(image) = }")
        
        # TODO: add logic to preprocess 3d image based on model chosen to align with trained preprocess
        image_transformation = get_standard_student_teacher_transform()
        image = image_transformation(image)
        print(f"In ModelService: {type(image) = }")
        image = cast(torch.Tensor, image)

        pred = run_inference(model, vol_np=vol_np, image=image)
        inference_response = prepare_model_inference_response(pred=pred, image=image)
        return inference_response
    
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

# to run ray serve run below (only need to run ray start ... one)
# ray start --head --dashboard-host=127.0.0.1
# serve run serve_model:model_app_deployment --name model --route-prefix /model