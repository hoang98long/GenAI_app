# llm/retrieval/chunkers.py

from dataclasses import dataclass
from typing import Dict, List

from loguru import logger

from llm.retrieval.loaders import RawDocument


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, object]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunk đơn giản theo độ dài ký tự.
    Có thể nâng cấp sau với tách theo câu / đoạn.
    """
    if not text:
        return []

    tokens = list(text)
    chunks: List[str] = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = tokens[start:end]
        chunks.append("".join(chunk_tokens))
        if end == n:
            break
        start = end - chunk_overlap

    return chunks


def chunk_raw_documents(
    docs: List[RawDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Nhận list RawDocument (page/paragraph) → trả về list Chunk nhỏ hơn,
    giữ lại metadata gốc + index chunk.
    """
    all_chunks: List[Chunk] = []
    for doc_idx, doc in enumerate(docs):
        pieces = _split_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = i
            metadata["doc_index"] = doc_idx
            all_chunks.append(Chunk(text=piece, metadata=metadata))

    logger.info("Created {} chunks from {} raw docs", len(all_chunks), len(docs))
    return all_chunks
