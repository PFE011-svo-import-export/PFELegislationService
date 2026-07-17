import logging
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scalar_fastapi import get_scalar_api_reference
from app.api.routes.chat_routes import chat_router
from app.api.routes.rag_routes import rag_router
from app.core.dependencies import get_vector_store

logger = logging.getLogger("legislation-service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrer même si Qdrant est injoignable, pour éviter un crash-loop au déploiement.
    try:
        get_vector_store().ensure_collection()
        logger.info("Qdrant collection ready.")
    except Exception:
        logger.exception("Failed to ensure Qdrant collection on startup; starting anyway.")
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

# Handler global : renvoie l'erreur réelle (type + message + traceback) au lieu d'un
# « Internal Server Error » opaque. Il attrape aussi les exceptions levées pendant la
# construction des dépendances (ex: échec d'init du pipeline RAG), ce qu'un try/except
# dans le corps d'une route ne pourrait pas faire.
# NOTE: le traceback est exposé pour faciliter le debug de ce projet de test —
# à retirer (ou masquer derrière un flag DEBUG) avant une vraie mise en production.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()  # trace complète dans les logs Render
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "traceback": traceback.format_exc().splitlines()[-8:],
        },
    )


app.include_router(chat_router, prefix="/api/v1/legislation")
app.include_router(rag_router, prefix="/api/v1/rag")


# Liveness : health check Render. Ne dépend pas de Qdrant.
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


# Readiness : vérifie que Qdrant est joignable. Pour le diagnostic, pas Render.
@app.get("/ready", include_in_schema=False)
async def ready():
    try:
        get_vector_store().ping()
        return {"status": "ready", "qdrant": "ok"}
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "qdrant": "unreachable", "detail": str(exc)},
        )


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
