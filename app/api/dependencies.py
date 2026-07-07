from fastapi import Depends
from anthropic import Anthropic
from app.core.dependencies import get_anthropic_client, get_cohere_client, get_openai_client, get_vector_store
from app.core.config import settings
from app.services.chat_service import ChatService
from app.services.rag_service import RagService
from app.storage.qdrant_vectordb import VectorStore
from openai import OpenAI
import cohere

def get_rag_service(
    openai_client: OpenAI = Depends(get_openai_client),
    vector_store: VectorStore = Depends(get_vector_store),
    reranker_model: cohere.ClientV2 = Depends(get_cohere_client)
) -> RagService:
    return RagService(client=openai_client, embed_model=settings.openai_api_model, vector_store=vector_store, reranker_model=reranker_model)

def get_chat_service(
    client: Anthropic = Depends(get_anthropic_client),
    rag_service: RagService = Depends(get_rag_service),
) -> ChatService:
    return ChatService(client=client, rag_service=rag_service)