# llm/retrieval/embeddings.py

from functools import lru_cache
from typing import Iterable, List

from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingModel:
    def __init__(self, model_name: str):
        logger.info("Loading embedding model: {}", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded: {}", model_name)

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        return self.model.encode(list(texts), show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    settings = get_settings()
    return EmbeddingModel(settings.embedding_model_name)
