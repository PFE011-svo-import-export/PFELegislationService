from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    service_name: str = "PFE Legislation Service"
    anthropic_api_key: str
    open_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "qwen3-embedding:4b"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_use_https: bool = False
    documents_path: str = "app/documents"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8-sig",
        "extra": "ignore"
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()