from fastapi import Depends
from anthropic import Anthropic
from ollama import Client as OllamaClient
from app.core.dependencies import get_anthropic_client, get_ollama_client, get_vector_store
from app.core.config import settings
from app.services.chat_service import ChatService
from app.services.rag_service import RagService
from app.storage.qdrant_vectordb import VectorStore

def get_rag_service(
    ollama_client: OllamaClient = Depends(get_ollama_client),
    vector_store: VectorStore = Depends(get_vector_store),
) -> RagService:
    return RagService(client=ollama_client, embed_model=settings.ollama_embed_model, vector_store=vector_store)

def get_chat_service(
    client: Anthropic = Depends(get_anthropic_client),
    rag_service: RagService = Depends(get_rag_service),
) -> ChatService:
    return ChatService(client=client, rag_service=rag_service)