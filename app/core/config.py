from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    service_name: str = "PFE Legislation Service"
    anthropic_api_key: str
    open_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    documents_path: str = "app/documents"

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()