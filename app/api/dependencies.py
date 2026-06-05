from fastapi import Depends
from anthropic import Anthropic
from app.core.dependencies import get_anthropic_client
from app.services.chat_service import ChatService
from app.services.rag_service import RagService

def get_rag_service() -> RagService:
    return RagService()

def get_chat_service(
    client: Anthropic = Depends(get_anthropic_client),
    rag_service: RagService = Depends(get_rag_service)
) -> ChatService:
    return ChatService(client=client, rag_service=rag_service)