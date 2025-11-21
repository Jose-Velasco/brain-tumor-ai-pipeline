# Router file is for uvicorn dev only; Ray Serve has its own class endpoint.
from fastapi import APIRouter
from ...schemas.predict import PredictRequest, PredictResponse
# from ...services.model_runtime import get_model, get_device

router = APIRouter()

# @router.post("/", response_model=PredictResponse, summary="Run segmentation")
# async def predict(payload: PredictRequest):
#     """Dev / uvicorn endpoint (non-Ray)"""
#     model, device = get_dev_model_and_device()
#     return run_inference(model, device, payload)
