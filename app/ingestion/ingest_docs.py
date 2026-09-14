"""
Run once (and again whenever ICAR/KVK PDFs are added/updated):

    python -m app.ingestion.ingest_docs

Reads every PDF from settings.DOCS_SOURCE_DIR, chunks + embeds them, and persists
to settings.CHROMA_DB_DIR. The FastAPI app only ever *reads* this store at startup
(app/main.py) — it never builds it inline, unlike AiVideoAssistant's per-session
build_vector_store(transcript) call.
"""
import logging

from app.core.config import settings
from app.core.vector_store import build_vector_store_from_pdfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Ingesting PDFs from {settings.DOCS_SOURCE_DIR} -> {settings.CHROMA_DB_DIR}")
    build_vector_store_from_pdfs(
        docs_dir=settings.DOCS_SOURCE_DIR,
        persist_dir=settings.CHROMA_DB_DIR,
    )
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
