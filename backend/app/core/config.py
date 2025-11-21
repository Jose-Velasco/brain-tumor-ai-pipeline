# settings, env varsfrom pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Path to BasicUNet student model checkpoint (update to your real path)
    # UNETR_CHECKPOINT: str = "/models/unetr.ckpt"

    class Config:
        env_file = ".env"

settings = Settings()