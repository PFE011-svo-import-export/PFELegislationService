from fastapi import APIRouter, Depends
from app.api.dependencies import get_chat_service
from app.services.chat_service import ChatService

chat_router = APIRouter()

@chat_router.get("/tarifs/{pays}")
def search_traitement_tarifiaires(pays: str, service: ChatService = Depends(get_chat_service)):
    return service.search_traitement_tarifiaires(pays)

@chat_router.get("/exigences/{produit}/{pays}")
def search_traitement_tarifiaires(produit:str, pays: str, service: ChatService = Depends(get_chat_service)):
    return service.search_exigences_importation(produit, pays)

