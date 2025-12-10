# app/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import chat, health


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="VietGenAI RAG API",
        version="0.1.0",
        description="API GenAI tiếng Việt (RAG trên tài liệu) – skeleton.",
    )

    # CORS
    origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health.router)
    app.include_router(chat.router, prefix="/v1")

    return app


app = create_app()
