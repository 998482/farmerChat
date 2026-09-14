import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.vector_store import load_vector_store
from app.routers import health, krishi_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Chroma once at startup, not per-request — this is what your original code
    # didn't have: build_vector_store() was called inline in Streamlit each session.
    try:
        app.state.vector_store = load_vector_store(settings.CHROMA_DB_DIR)
        logger.info("Vector store loaded successfully.")
    except FileNotFoundError as e:
        app.state.vector_store = None
        logger.warning(f"Vector store not loaded: {e}")
    yield


app = FastAPI(
    title="FasalGuru Backend",
    description="Dockerized FastAPI backend for FasalGuru's AI layer — Krishi Agent (RAG), "
                 "Reasoning Trace, Mandi Insight, Alerts.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(krishi_agent.router, tags=["krishi-agent"])
