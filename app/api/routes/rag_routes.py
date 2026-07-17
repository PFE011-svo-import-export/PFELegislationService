from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.api.dependencies import get_rag_service, get_vector_store
from app.core.config import settings

class CompareSearchRequest(BaseModel):
    prompt: str
    top_k: int = 10

rag_router = APIRouter()

@rag_router.post("/ingest")
def ingest(rag_service = Depends(get_rag_service)):
    result = rag_service.ingest_folder(settings.documents_path)
    return {
        "ingested": result["ingested"],
        "ingested_count": len(result["ingested"]),
        "skipped": result["skipped"],
        "skipped_count": len(result["skipped"]),
    }

@rag_router.get("/check/{filename}")
def check_ingested(filename: str, vector_store = Depends(get_vector_store)):
    ingested = vector_store.is_source_ingested(filename)
    return {"filename": filename, "ingested": ingested}

@rag_router.post("/compare")
def compare_search(req: CompareSearchRequest, rag_service = Depends(get_rag_service)):
    output_path = rag_service.compare_search(req.prompt, top_k=req.top_k)
    return {"output_path": output_path}