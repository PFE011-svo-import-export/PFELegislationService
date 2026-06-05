from fastapi import Depends
from anthropic import Anthropic
from ollama import Client as OllamaClient
from app.core.dependencies import get_anthropic_client, get_ollama_client
from app.core.config import settings
from app.services.chat_service import ChatService
from app.services.rag_service import RagService

def get_rag_service(
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> RagService:
    return RagService(client=ollama_client, embed_model=settings.ollama_embed_model)

def get_chat_service(
    client: Anthropic = Depends(get_anthropic_client),
    rag_service: RagService = Depends(get_rag_service),
) -> ChatService:
    return ChatService(client=client, rag_service=rag_service)