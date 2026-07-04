from anthropic import Anthropic
from ollama import Client as OllamaClient
from app.core.config import settings
from app.storage.qdrant_vectordb import VectorStore
from functools import lru_cache
from FlagEmbedding import FlagReranker
from transformers.utils import logging as hf_logging

# Le tokenizer XLM-Roberta du reranker émet un simple hint de performance (usage de __call__)
# qui n'a aucun impact fonctionnel : on réduit la verbosité de transformers pour l'éviter.
hf_logging.set_verbosity_error()

@lru_cache()
def get_anthropic_client() -> Anthropic:
    return Anthropic(api_key=settings.anthropic_api_key)

@lru_cache()
def get_ollama_client() -> OllamaClient:
    return OllamaClient(host=settings.ollama_base_url)

@lru_cache()
def get_vector_store() -> VectorStore:
    return VectorStore(
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
        qdrant_api_key=settings.qdrant_api_key,
        qdrant_use_https=settings.qdrant_use_https,
    )

@lru_cache()
def get_reranker_model() -> FlagReranker:
    return FlagReranker(
        settings.reranker_model,
        use_fp16=True,
        devices="cuda",
    )