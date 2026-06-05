from fastapi import APIRouter, Depends
from app.api.dependencies import get_rag_service

rag_router = APIRouter()

@rag_router.post("/ingest")
def ingest(rag_service = Depends(get_rag_service)):
    temp = rag_service.chunk_md_file("app/documents/01-99-2023-fra-4105.md")
    embedding = rag_service.embed(temp[1]["content"])
    return { "data": temp[1], "embedding": embedding }