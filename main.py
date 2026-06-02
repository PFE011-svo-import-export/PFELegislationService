from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="PFE Legislation Service")
app.include_router(router, prefix="/api/v1")
