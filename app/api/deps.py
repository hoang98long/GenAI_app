# app/api/deps.py

from dataclasses import dataclass
from typing import Any, List

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from llm.pipelines.rag_qa import RAGChatResult, get_rag_pipeline


@dataclass
class ChatAnswer:
    answer: str
    sources: List[dict[str, Any]]


class RAGService:
    """
    Service production: dùng RAGPipeline thực tế.
    """

    def __init__(self):
        self.pipeline = get_rag_pipeline()

    def chat(self, question: str, session_id: str | None = None, top_k: int = 5) -> ChatAnswer:
        # TODO: session_id có thể dùng để lưu context hội thoại sau này
        result: RAGChatResult = self.pipeline.chat(question=question, top_k=top_k)
        return ChatAnswer(answer=result.answer, sources=result.sources)


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    # settings hiện chưa dùng nhiều, nhưng sau này có thể dùng để chọn pipeline
    return RAGService()


def verify_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
