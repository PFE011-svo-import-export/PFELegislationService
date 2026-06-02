from anthropic import Anthropic
from app.core.config import settings
from functools import lru_cache

# Singleton instance of the Anthropic client
@lru_cache()
def get_anthropic_client() -> Anthropic:
    return Anthropic(api_key=settings.anthropic_api_key)