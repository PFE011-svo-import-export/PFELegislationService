from fastapi import APIRouter, Depends
from app.api.dependencies import get_rag_service, get_vector_store
from app.core.config import settings

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

@rag_router.delete("/collections")
def delete_coll(rag_service = Depends(get_rag_service)):
    rag_service.delete_coll()