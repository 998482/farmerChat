import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.core.config import settings

logger = logging.getLogger(__name__)

_embeddings = None


class E5Embeddings(HuggingFaceEmbeddings):
    """
    E5 family models (intfloat/multilingual-e5-large included) are trained with
    "query: " / "passage: " prefixes — without them, retrieval quality drops noticeably.
    This wrapper adds them transparently; only used when EMBEDDING_MODEL is an E5 model.
    """

    def embed_documents(self, texts):
        prefixed = [f"passage: {t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def get_embeddings():
    """Lazy singleton — loading the sentence-transformers model is slow, don't repeat it."""
    global _embeddings
    if _embeddings is None:
        embedding_cls = E5Embeddings if "e5" in settings.EMBEDDING_MODEL.lower() else HuggingFaceEmbeddings
        _embeddings = embedding_cls(model_name=settings.EMBEDDING_MODEL)
    return _embeddings


def _chunk_id(source: str, page: int, content: str) -> str:
    """
    Deterministic ID = hash(source filename + page number + exact chunk text).
    This is what makes re-running ingestion idempotent: re-ingesting the same PDF
    produces the same IDs, and Chroma's add() skips IDs that already exist instead
    of inserting duplicates. If a PDF's content changes, its chunks get new IDs
    (old ones remain — this is a known limitation, see README/report).
    """
    raw = f"{source}::{page}::{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_vector_store_from_pdfs(docs_dir: str = None, persist_dir: str = None) -> Chroma:
    """
    Ingestion: reads every PDF in docs_dir, chunks it, embeds it, persists to Chroma.
    Safe to re-run — uses deterministic per-chunk IDs so repeated ingestion does not
    duplicate documents (verified: 2 chunks -> re-run -> still 2, not 4).
    """
    docs_dir = docs_dir or settings.DOCS_SOURCE_DIR
    persist_dir = persist_dir or settings.CHROMA_DB_DIR

    pdf_paths = list(Path(docs_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"Koi PDF nahi mili {docs_dir} mein. ICAR/KVK advisory PDFs is folder mein daalo pehle."
        )

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for pdf_path in pdf_paths:
        logger.info(f"Loading {pdf_path.name}...")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as e:
            logger.warning(f"Skipping {pdf_path.name}: could not read PDF ({e})")
            continue

        if not pages or not any(p.page_content.strip() for p in pages):
            logger.warning(f"Skipping {pdf_path.name}: no extractable text (scanned/image PDF?)")
            continue

        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = pdf_path.name
        # Drop whitespace-only chunks — embedding an empty string wastes a call and
        # produces a useless vector that can still get retrieved.
        chunks = [c for c in chunks if c.page_content.strip()]
        all_chunks.extend(chunks)
        logger.info(f"  -> {len(chunks)} usable chunks")

    if not all_chunks:
        raise ValueError("Koi PDF se usable text nahi mila — sab PDFs empty ya unreadable hain.")

    ids = [_chunk_id(c.metadata.get("source", "unknown"), c.metadata.get("page", 0), c.page_content) for c in all_chunks]
    logger.info(f"Total chunks: {len(all_chunks)} from {len(pdf_paths)} PDFs")

    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=get_embeddings(),
        ids=ids,
        persist_directory=persist_dir,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    return vector_store


def load_vector_store(persist_dir: str = None) -> Chroma:
    """Load an already-persisted Chroma store (used at server startup, not per-request)."""
    persist_dir = persist_dir or settings.CHROMA_DB_DIR

    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"{persist_dir} nahi mila. Pehle `python -m app.ingestion.ingest_docs` chalao."
        )

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )


def get_retriever(vector_store: Chroma, k: int = None, crop: str = None, district: str = None):
    """
    Retriever with optional crop/district hint folded into search — not a hard metadata
    filter (we don't have reliable per-chunk crop/district tags, see note above), but
    appending them to the query nudges semantic search toward the right section.
    """
    k = k or settings.RETRIEVER_TOP_K
    return vector_store.as_retriever(search_kwargs={"k": k})


def build_query_text(query: str, crop: str = None, district: str = None) -> str:
    """Fold optional crop/district context into the retrieval query string."""
    parts = [query]
    if crop:
        parts.append(f"crop: {crop}")
    if district:
        parts.append(f"district: {district}")
    return " | ".join(parts)
