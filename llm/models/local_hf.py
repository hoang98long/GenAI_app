# llm/models/local_hf.py

import os
from functools import lru_cache
from typing import Any

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.config import get_settings
from llm.models.base import BaseLLMClient


class LocalHFClient(BaseLLMClient):
    """
    Client dùng Hugging Face Transformers để chạy LLM local.
    Khởi tạo lazy (chỉ load model khi lần đầu gọi).
    """

    def __init__(self, model_name: str, hf_token: str | None = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._pipe = None

    def _load_pipeline(self) -> None:
        if self._pipe is not None:
            return

        logger.info("Loading local HF model: {}", self.model_name)
        auth_kwargs: dict[str, Any] = {}
        if self.hf_token:
            auth_kwargs["token"] = self.hf_token

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **auth_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            **auth_kwargs,
        )

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        logger.info("HF pipeline loaded for model: {}", self.model_name)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self._load_pipeline()
        assert self._pipe is not None

        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.3)

        outputs = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=self._pipe.tokenizer.eos_token_id,  # type: ignore[attr-defined]
        )

        # pipeline trả về list; lấy text đầu tiên
        text = outputs[0]["generated_text"]
        # Nếu model echo lại prompt, ta cắt phần đầu đi
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()


@lru_cache(maxsize=1)
def get_llm_client() -> LocalHFClient:
    settings = get_settings()
    if settings.llm_provider != "local_hf":
        # Sau này ta có thể thêm provider khác (OpenAI, Azure, v.v.)
        logger.warning(
            "LLM_PROVIDER != local_hf, nhưng hiện mới implement LocalHFClient. "
            "Đang fallback về local_hf.",
        )

    return LocalHFClient(model_name=settings.llm_model_name, hf_token=settings.hf_token)
