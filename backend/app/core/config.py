from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    model_root_dir: Path = Path("app/models/artifacts")

    class Config:
        env_file = ".env"

settings = Settings()