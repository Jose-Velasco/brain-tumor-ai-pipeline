import os

# BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
# MODEL_PREDICT_URL = f"{BACKEND_BASE_URL}/model/api/predict"
# GNEREAL_API_URL = f"{BACKEND_BASE_URL}/api/health"
# REPORT_URL = f"{BACKEND_BASE_URL}/api/report"

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Path to BasicUNet student model checkpoint (update to your real path)
    # UNETR_CHECKPOINT: str = "/models/unetr.ckpt"
    BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    MODEL_PREDICT_URL: str = f"{BACKEND_BASE_URL}/model/api/predict"
    GNEREAL_API_URL: str = f"{BACKEND_BASE_URL}/api/health"
    REPORT_URL: str = f"{BACKEND_BASE_URL}/api/health/report"

    class Config:
        env_file = ".env"

settings = Settings()