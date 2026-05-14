"""
FastAPI application entry point.

Startup sequence:
  1. Load settings from environment
  2. Load FAISS vector store
  3. Initialise Groq LLM client
  4. Mount routers
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from groq import Groq

from app.config import settings
from app.routers import chat, health
from retrieval import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources on startup; release on shutdown."""
    logger.info("Starting SHL Assessment Recommender…")

    # Load vector store — fail fast if index is missing
    try:
        vector_store.load(
            index_path=settings.vector_store_path,
            metadata_path=settings.catalog_metadata_path,
        )
    except FileNotFoundError as exc:
        logger.critical(str(exc))
        sys.exit(1)

    # Initialise Groq client
    app.state.llm_client = Groq(api_key=settings.llm_api_key)
    app.state.settings = settings

    logger.info("Service ready. LLM: %s / %s", settings.llm_provider, settings.llm_model)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "A conversational REST API that uses a RAG pipeline to recommend "
        "SHL Individual Test Solutions based on hiring requirements."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
