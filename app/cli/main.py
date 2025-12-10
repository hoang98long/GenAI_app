# app/cli/main.py

import argparse

from loguru import logger

from app.workers.ingest_worker import main as ingest_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VietGenAI RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest tài liệu trong data/raw vào vector store.")
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Xoá index cũ và build lại từ đầu.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        rebuild = getattr(args, "rebuild", False)
        logger.info("Running ingest (rebuild={})", rebuild)
        ingest_main(rebuild=rebuild)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
