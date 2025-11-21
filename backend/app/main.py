from fastapi import FastAPI
from .api.router import api_router

app = FastAPI(title="Brain Tumor Segmentation API")

# endpoints
app.include_router(api_router, prefix="/api")