from anthropic import Anthropic
from ollama import Client as OllamaClient
from app.core.config import settings
from functools import lru_cache

@lru_cache()
def get_anthropic_client() -> Anthropic:
    return Anthropic(api_key=settings.anthropic_api_key)

@lru_cache()
def get_ollama_client() -> OllamaClient:
    return OllamaClient(host=settings.ollama_base_url)