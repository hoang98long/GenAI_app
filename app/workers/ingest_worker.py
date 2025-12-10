# app/workers/ingest_worker.py

"""
Worker phụ trách ingest tài liệu vào vector store.

Sau này ta có thể:
- Dùng Celery / RQ / Dramatiq / Arq để chạy background.
- Hoặc đơn giản là script cron.

Hiện tại chỉ để hàm main() demo.
"""

from loguru import logger

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    logger.info("Ingest worker started")
    logger.info("Using raw dir: {}", settings.data_raw_dir)
    logger.info("Using processed dir: {}", settings.data_processed_dir)
    logger.info("TODO: implement real ingest logic using llm.retrieval.*")


if __name__ == "__main__":
    main()
