# scripts/ingest_docs.py

import uuid
from pathlib import Path
from typing import List

from loguru import logger

from app.config import get_settings
from llm.retrieval.chunkers import chunk_raw_documents
from llm.retrieval.loaders import RawDocument, load_file
from llm.retrieval.vectorstore import get_vectorstore


def discover_files(raw_dir: Path) -> List[Path]:
    exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
    files: List[Path] = []
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    return files


def ingest_all(rebuild: bool = False) -> None:
    settings = get_settings()
    raw_dir = Path(settings.data_raw_dir)
    processed_dir = Path(settings.data_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        logger.warning("Raw data dir {} does not exist. Nothing to ingest.", raw_dir)
        return

    files = discover_files(raw_dir)
    if not files:
        logger.warning("No documents found in {}.", raw_dir)
        return

    logger.info("Found {} files to ingest", len(files))

    vectorstore = get_vectorstore()

    if rebuild:
        # Chromadb không có clear collection trực tiếp bằng client cũ,
        # ở đây ta tạm xóa toàn bộ bằng cách recreate client (tuỳ version).
        logger.warning("Rebuild requested but low-level clear not implemented. "
                       "Bạn có thể xoá thư mục {} thủ công rồi chạy lại.", settings.vector_db_dir)

    all_texts: List[str] = []
    all_metadatas: List[dict] = []
    all_ids: List[str] = []

    for f in files:
        logger.info("Loading file {}", f)
        raw_docs: List[RawDocument] = load_file(f)
        if not raw_docs:
            continue

        chunks = chunk_raw_documents(raw_docs)
        for chunk in chunks:
            all_texts.append(chunk.text)
            all_metadatas.append(chunk.metadata)
            # id duy nhất: source + uuid4
            all_ids.append(str(uuid.uuid4()))

    if not all_texts:
        logger.warning("No chunks produced. Nothing to add to vector store.")
        return

    logger.info("Adding {} chunks to vector store", len(all_texts))
    vectorstore.add_texts(texts=all_texts, metadatas=all_metadatas, ids=all_ids)

    logger.info("Ingestion finished. Total chunks: {}", len(all_texts))


if __name__ == "__main__":
    ingest_all(rebuild=False)
