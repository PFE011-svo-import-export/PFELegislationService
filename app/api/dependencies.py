from fastapi import Depends
from anthropic import Anthropic
from app.core.dependencies import get_anthropic_client
from app.generation.chat_service import ChatService

def get_chat_service(client: Anthropic = Depends(get_anthropic_client)) -> ChatService:
    return ChatService(client=client)