import os

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    MODEL_PREDICT_URL: str = f"{BACKEND_BASE_URL}/model/api/predict"
    GNEREAL_API_URL: str = f"{BACKEND_BASE_URL}/api/health"
    REPORT_URL: str = f"{BACKEND_BASE_URL}/api/health/report"
    
    ollama_base_url: str ="http://ollama:11434"
    ollama_url: str = f"{ollama_base_url}/api/generate"
    model_name: str = "gemma3:4b-it-qat"

    class Config:
        env_file = ".env"

settings = Settings()