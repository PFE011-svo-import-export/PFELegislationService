import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.api.dependencies import get_chat_service
from app.services.chat_service import ChatService

chat_router = APIRouter()

class GenerateRequest(BaseModel):
    prompt: str

@chat_router.post("/generate")
def generate_answer(req: GenerateRequest, service: ChatService = Depends(get_chat_service)):
    '''
    Generates an answer to the given prompt using the RAG service. This endpoint will be used for
    the AI chatbot assistant.
    '''
    return service.answer_prompt(req.prompt)


@chat_router.post("/generate/stream")
def stream_answer(req: GenerateRequest, service: ChatService = Depends(get_chat_service)):
    '''
    Same as /generate, but streams the answer token by token over SSE so the
    chatbot can display it as it is written.
    '''
    def event_stream():
        try:
            for chunk in service.stream_prompt(req.prompt):
                # Le texte est encodé en JSON : un chunk peut contenir des sauts
                # de ligne, qui couperaient l'événement SSE en deux.
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as exc:
            # Le handler global ne peut rien renvoyer une fois le streaming
            # commencé (les en-têtes sont déjà partis) : on passe par le flux.
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Empêche le proxy (Render, nginx) de bufferiser la réponse.
            "X-Accel-Buffering": "no",
        },
    )


@chat_router.get("/tarifs/{pays}")
def search_traitement_tarifiaires(pays: str, service: ChatService = Depends(get_chat_service)):
    '''
    Searches for tariff treatments based on the specified country.    
    '''
    return service.search_traitement_tarifiaires(pays)

@chat_router.get("/exigences/{produit}/{pays}")
def search_exigences_importation(produit:str, pays: str,
                                 service: ChatService = Depends(get_chat_service)
                                ):
    '''
    Searches for import requirements based on the specified product and country.
    '''
    return service.search_exigences_importation(produit, pays)
