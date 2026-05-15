"""
Application configuration — loaded from environment variables / .env file.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: str = "huggingface"
    llm_api_key: str = ""
    llm_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # Paths
    vector_store_path: str = "data/faiss.index"
    catalog_path: str = "data/catalog.json"
    catalog_metadata_path: str = "data/catalog_metadata.json"
    embed_model: str = "all-MiniLM-L6-v2"

    # Limits
    max_turns: int = 8
    request_timeout_s: int = 45
    top_k_retrieval: int = 20


# Singleton settings instance
settings = Settings()
