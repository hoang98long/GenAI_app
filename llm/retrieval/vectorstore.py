# llm/retrieval/vectorstore.py

from typing import Any, Dict, List

import chromadb
from chromadb.api.models.Collection import Collection
from loguru import logger

from app.config import get_settings
from llm.retrieval.embeddings import get_embedding_model


class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        self.settings = get_settings()
        if self.settings.vector_db != "chromadb":
            logger.warning(
                "VECTOR_DB khác 'chromadb' nhưng hiện chỉ hỗ trợ chromadb. "
                "Đang fallback về chromadb.",
            )

        self.client = chromadb.PersistentClient(path=self.settings.vector_db_dir)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
        )
        self.embedder = get_embedding_model()

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
        ids: List[str] | None = None,
    ) -> None:
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        embeddings = self.embedder.embed_texts(texts)
        logger.info("Adding {} documents to vector store", len(texts))

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed_text(query_text)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        out: List[Dict[str, Any]] = []
        for doc, meta, _id in zip(docs, metadatas, ids):
            out.append(
                {
                    "id": _id,
                    "text": doc,
                    "metadata": meta or {},
                }
            )
        return out


_vectorstore_singleton: VectorStore | None = None


def get_vectorstore() -> VectorStore:
    global _vectorstore_singleton
    if _vectorstore_singleton is None:
        _vectorstore_singleton = VectorStore(collection_name="documents")
    return _vectorstore_singleton
