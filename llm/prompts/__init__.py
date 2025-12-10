# llm/prompts/__init__.py

from pathlib import Path
from typing import Iterable, List


def load_system_prompt() -> str:
    here = Path(__file__).parent
    prompt_path = here / "qa_system.md"
    return prompt_path.read_text(encoding="utf-8")


def build_rag_prompt(question: str, contexts: Iterable[str]) -> str:
    system = load_system_prompt()
    context_str = "\n\n---\n\n".join(contexts)
    return (
        f"{system}\n\n"
        f"NGỮ CẢNH:\n{context_str}\n\n"
        f"CÂU HỎI:\n{question}\n\n"
        "VUI LÒNG TRẢ LỜI:"
    )
