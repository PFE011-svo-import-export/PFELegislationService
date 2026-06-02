from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    service_name: str = "PFE Legislation Service"
    anthropic_api_key: str
    open_api_key: str = ""

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()