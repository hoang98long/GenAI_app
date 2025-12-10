# llm/retrieval/retrievers.py

from dataclasses import dataclass
from typing import Any, Dict, List

from llm.retrieval.vectorstore import get_vectorstore


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


class Retriever:
    def __init__(self, top_k_default: int = 5):
        self.vectorstore = get_vectorstore()
        self.top_k_default = top_k_default

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        k = top_k or self.top_k_default
        hits = self.vectorstore.query(query_text=query, top_k=k)
        return [
            RetrievedChunk(id=h["id"], text=h["text"], metadata=h.get("metadata", {}))
            for h in hits
        ]
