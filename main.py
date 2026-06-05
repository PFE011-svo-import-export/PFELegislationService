from fastapi import FastAPI
from app.api.routes.chat_routes import chat_router

app = FastAPI(title="PFE Legislation Service")
app.include_router(chat_router, prefix="/api/v1/chat")
