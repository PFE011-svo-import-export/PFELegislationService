from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.api.dependencies import get_chat_service
from app.services.chat_service import ChatService

chat_router = APIRouter()

class GenerateRequest(BaseModel):
    prompt: str

@chat_router.post("/generate")
def generate_answer(req: GenerateRequest, service: ChatService = Depends(get_chat_service)):
    return service.answer_prompt(req.prompt)

@chat_router.get("/tarifs/{pays}")
def search_traitement_tarifiaires(pays: str, service: ChatService = Depends(get_chat_service)):
    return service.search_traitement_tarifiaires(pays)

@chat_router.get("/exigences/{produit}/{pays}")
def search_traitement_tarifiaires(produit:str, pays: str, service: ChatService = Depends(get_chat_service)):
    return service.search_exigences_importation(produit, pays)

