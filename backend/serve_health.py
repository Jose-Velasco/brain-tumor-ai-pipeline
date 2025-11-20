# Ray Serve deployment: light app (no model)
# Serve entrypoint for health status, etc.
from fastapi import FastAPI
from ray import serve

from app.api.endpoints.health import router as health_router

# FastAPI app for health endpoints only
health_app = FastAPI(title="Health API")
health_app.include_router(health_router, prefix="/api/health")

@serve.deployment(
    route_prefix="/",             # e.g. http://host:8000/api/health
    ray_actor_options={"num_gpus": 0}
)
@serve.ingress(health_app)
class HealthService:
    """Ray Serve deployment for LIGHTWEIGHT health endpoints."""
    def __init__(self):
        # No model load here; startup is cheap
        pass

# Object used by `serve run`
health_app_deployment = HealthService.bind()