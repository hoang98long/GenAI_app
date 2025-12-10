# llm/models/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLLMClient(ABC):
    """Interface chung cho các LLM client (local HF, OpenAI, v.v.)."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Sinh text từ prompt đơn giản."""
        raise NotImplementedError

    def chat(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        """
        Hàm tiện ích cho chat-style: ghép system + user lại thành 1 prompt đơn giản.
        Với các model chat chuyên dụng sau này có thể override.
        """
        full_prompt = f"{system_prompt}\n\nNgười dùng: {user_prompt}\n\nAssistant:"
        return self.generate(full_prompt, **kwargs)
