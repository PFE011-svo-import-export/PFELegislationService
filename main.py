from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference
from app.api.routes.chat_routes import chat_router
from app.api.routes.rag_routes import rag_router
from app.core.dependencies import get_vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_vector_store().ensure_collection()
    yield

app = FastAPI(
    title="PFE Legislation Service",
    description="RAG-powered legislation chat and retrieval API.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the browser-based RAG viewer (and other clients) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1/legislation")
app.include_router(rag_router, prefix="/api/v1/rag")


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
