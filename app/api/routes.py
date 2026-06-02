from fastapi import APIRouter, Depends
from app.api.dependencies import get_chat_service
from app.generation.chat_service import ChatService
from app.models.schemas import QueryRequest

router = APIRouter()

@router.post("/generate")
def generate(req: QueryRequest, service: ChatService = Depends(get_chat_service)):
    return {"response": service.generate(req.prompt)}