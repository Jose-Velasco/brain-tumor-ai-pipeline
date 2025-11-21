from fastapi import APIRouter
from .endpoints import predict, health

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
# Router file is for uvicorn dev only; Ray Serve has its own class endpoint.
api_router.include_router(predict.router, prefix="/predict", tags=["predict"])