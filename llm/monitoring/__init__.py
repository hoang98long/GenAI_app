# llm/monitoring/__init__.py

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class ChatLog:
    timestamp: datetime
    question: str
    answer: str
    metadata: Dict[str, Any]


def log_chat(question: str, answer: str, sources: Optional[List[Dict[str, Any]]] = None) -> None:
    payload = ChatLog(
        timestamp=datetime.utcnow(),
        question=question,
        answer=answer,
        metadata={"sources_count": len(sources or [])},
    )
    logger.info("CHAT_LOG | {}", payload)
