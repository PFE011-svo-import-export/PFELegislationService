from anthropic import Anthropic
from app.core.config import settings
from app.storage.qdrant_vectordb import VectorStore
from functools import lru_cache
from openai import OpenAI
import cohere

@lru_cache()
def get_anthropic_client() -> Anthropic:
    return Anthropic(api_key=settings.anthropic_api_key)

@lru_cache()
def get_cohere_client() -> cohere.ClientV2:
    return cohere.ClientV2(api_key=settings.cohere_api_key)

@lru_cache()
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)

@lru_cache()
def get_vector_store() -> VectorStore:
    return VectorStore(
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
        qdrant_api_key=settings.qdrant_api_key,
        qdrant_use_https=settings.qdrant_use_https,
    )