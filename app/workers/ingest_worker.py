# app/workers/ingest_worker.py

from loguru import logger

from scripts.ingest_docs import ingest_all


def main(rebuild: bool = False) -> None:
    logger.info("Starting ingest worker (rebuild={})", rebuild)
    ingest_all(rebuild=rebuild)
    logger.info("Ingest worker finished")


if __name__ == "__main__":
    main()
