from fastapi import APIRouter, Depends
from app.api.dependencies import get_chat_service
from app.services.chat_service import ChatService
from app.models.chat_schema import ChatRequest

chat_router = APIRouter()

@chat_router.post("/generate")
def generate(req: ChatRequest, service: ChatService = Depends(get_chat_service)):
    print(f"Received prompt: {req.prompt}")
    return {"response": service.generate(req.prompt)}

