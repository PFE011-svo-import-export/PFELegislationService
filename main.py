from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes.chat_routes import chat_router
from app.api.routes.rag_routes import rag_router
from app.core.dependencies import get_vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_vector_store().ensure_collection()
    yield

app = FastAPI(title="PFE Legislation Service", lifespan=lifespan)
app.include_router(chat_router, prefix="/api/v1/chat")
app.include_router(rag_router, prefix="/api/v1/rag")
