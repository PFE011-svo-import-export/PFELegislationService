from app.core.dependencies import get_anthropic_client, get_cohere_client, get_openai_client, get_vector_store
from app.core.config import settings
from app.services.chat_service import ChatService
from app.services.rag_service import RagService
from functools import lru_cache

@lru_cache()
def get_rag_service() -> RagService:
    return RagService(
        client=get_openai_client(),
        embed_model=settings.openai_api_model,
        vector_store=get_vector_store(),
        reranker_model=get_cohere_client(),
    )

@lru_cache()
def get_chat_service() -> ChatService:
    return ChatService(
        client=get_anthropic_client(),
        rag_service=get_rag_service(),
    )
