# app/config.py

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # runtime
    env: str = Field("dev", alias="ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # api
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_key: str = Field("change_me_in_production", alias="API_KEY")
    allowed_origins: str = Field("http://localhost:3000", alias="ALLOWED_ORIGINS")

    # hugging face
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    llm_model_name: str = Field("AITeamVN/GRPO-VI-Qwen2-7B-RAG", alias="LLM_MODEL_NAME")
    embedding_model_name: str = Field("hiieu/halong_embedding", alias="EMBEDDING_MODEL_NAME")
    llm_provider: str = Field("local_hf", alias="LLM_PROVIDER")

    # vector db
    vector_db: str = Field("chromadb", alias="VECTOR_DB")
    vector_db_dir: str = Field("./data/vectorstores/default", alias="VECTOR_DB_DIR")

    # data paths
    data_raw_dir: str = Field("./data/raw", alias="DATA_RAW_DIR")
    data_processed_dir: str = Field("./data/processed", alias="DATA_PROCESSED_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
