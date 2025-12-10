# llm/pipelines/rag_qa.py

from dataclasses import dataclass
from typing import Any, Dict, List

from loguru import logger

from llm.models import get_llm_client
from llm.monitoring import log_chat
from llm.prompts import build_rag_prompt
from llm.retrieval.retrievers import Retriever


@dataclass
class RAGChatResult:
    answer: str
    sources: List[Dict[str, Any]]


class RAGPipeline:
    def __init__(self):
        self.llm = get_llm_client()
        self.retriever = Retriever()

    def chat(self, question: str, top_k: int = 5) -> RAGChatResult:
        # 1. Retrieve context
        chunks = self.retriever.retrieve(query=question, top_k=top_k)

        if not chunks:
            logger.warning("No context retrieved from vector store; answering without RAG.")
            prompt = (
                "Bạn là một trợ lý AI tiếng Việt.\n"
                "Không có tài liệu tham chiếu, hãy trả lời chung nhất, "
                "và nói rõ là bạn không có dữ liệu tài liệu cụ thể.\n\n"
                f"CÂU HỎI:\n{question}\n\nVUI LÒNG TRẢ LỜI:"
            )
            answer = self.llm.generate(prompt)
            result = RAGChatResult(answer=answer, sources=[])
            log_chat(question=question, answer=answer, sources=[])
            return result

        contexts = [c.text for c in chunks]
        prompt = build_rag_prompt(question=question, contexts=contexts)

        # 2. Gọi LLM
        answer = self.llm.generate(prompt)

        # 3. Chuẩn hoá nguồn
        sources: List[Dict[str, Any]] = [
            {
                "id": c.id,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        result = RAGChatResult(answer=answer, sources=sources)
        log_chat(question=question, answer=answer, sources=sources)
        return result


# Singleton tiện dụng
_pipeline_singleton: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = RAGPipeline()
    return _pipeline_singleton
