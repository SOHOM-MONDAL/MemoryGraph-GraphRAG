"""Configuration settings for MemoryGraph."""

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # LLM Settings
    LLM_PROVIDER: str = "ollama"  # "ollama" or "groq"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_TIMEOUT: int = 120

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384
    LOW_MEMORY_MODE: bool = False

    # Neo4j Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "change_me"
    NEO4J_ENABLED: bool = False

    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False
    REDIS_HISTORY_LIMIT: int = 20

    # PostgreSQL Settings (Optional)
    POSTGRES_URL: str = (
        "postgresql+psycopg://memorygraph:change_me@localhost:5432/memorygraph"
    )
    POSTGRES_ENABLED: bool = False

    # MySQL Settings
    MYSQL_URL: str = (
        "mysql+pymysql://memorygraph:memorygraph_secret@localhost:3306/memorygraph"
    )
    MYSQL_ENABLED: bool = False

    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_ENABLED: bool = False

    # Processing and GraphRAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5
    GRAPH_ENABLED: bool = True
    GRAPH_HOPS: int = 2
    GRAPH_MAX_SUBGRAPH_NODES: int = 25

    # Storage and Uploads
    MAX_UPLOAD_MB: int = 25
    UPLOAD_DIR: str = "./data/uploads"

    # API & Frontend
    API_URL: str = "http://localhost:8000"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
