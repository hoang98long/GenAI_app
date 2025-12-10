# app/api/deps.py

from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


@dataclass
class ChatAnswer:
    answer: str
    sources: list[dict[str, Any]]


class DummyRAGService:
    """
    Service tạm thời: trả lời cứng.
    Sau này ta sẽ thay bằng RAG pipeline thật từ module `llm/`.
    """

    def chat(self, question: str, session_id: str | None = None, top_k: int = 5) -> ChatAnswer:
        return ChatAnswer(
            answer=(
                "Xin chào! Đây là câu trả lời demo. "
                "Pipeline RAG thực tế sẽ được tích hợp ở bước sau."
            ),
            sources=[],
        )


def get_rag_service(settings: Settings = Depends(get_settings)) -> DummyRAGService:
    # Sau này có thể dùng settings để chọn provider, model, vector store…
    return DummyRAGService()


def verify_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Simple header-based API key auth."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
