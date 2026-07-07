from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    service_name: str = "PFE Legislation Service"
    anthropic_api_key: str
    openai_api_key: str = ""
    openai_api_model: str = "text-embedding-3-large"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_use_https: bool = False
    documents_path: str = "app/documents"
    cohere_api_key: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8-sig",
        "extra": "ignore"
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()