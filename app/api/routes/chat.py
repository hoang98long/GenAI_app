# app/api/routes/chat.py

from typing import Any

from fastapi import APIRouter, Depends

from app.api.deps import DummyRAGService, get_rag_service, verify_api_key

router = APIRouter(tags=["chat"])


class ChatRequest(BaseException):
    pass


from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của người dùng (tiếng Việt hoặc ngôn ngữ khác).")
    session_id: str | None = Field(
        default=None,
        description="ID session để lưu ngữ cảnh hội thoại (tùy chọn).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Số lượng đoạn văn được retrieve làm ngữ cảnh (khi có RAG).",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = []


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Hỏi đáp theo tài liệu (RAG)",
    dependencies=[Depends(verify_api_key)],
)
async def chat_endpoint(
    payload: ChatRequest,
    rag_service: DummyRAGService = Depends(get_rag_service),
) -> ChatResponse:
    """
    Endpoint chính để hỏi đáp.
    Hiện tại dùng DummyRAGService. Sau khi xây RAG thật, ta chỉ cập nhật `get_rag_service`.
    """
    result = rag_service.chat(
        question=payload.question,
        session_id=payload.session_id,
        top_k=payload.top_k,
    )
    return ChatResponse(answer=result.answer, sources=result.sources)
