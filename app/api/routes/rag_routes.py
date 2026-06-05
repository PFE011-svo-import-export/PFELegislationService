from fastapi import APIRouter, Depends
from app.api.dependencies import get_rag_service

rag_router = APIRouter()

@rag_router.post("/ingest")
def ingest(rag_service = Depends(get_rag_service)):
    chunks = rag_service.embed_md_file("app/documents/01-99-2023-fra-4105.md")
    return {"message": "Document ingested successfully.", "chunks_stored": len(chunks)}