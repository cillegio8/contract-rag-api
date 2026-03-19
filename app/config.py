"""
Configuration settings for Contract RAG API.
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Language Settings
    DEFAULT_LANGUAGE: str = "az"  # "az" for Azerbaijani, "ru" for Russian, "en" for English
    SUPPORTED_LANGUAGES: List[str] = ["az", "ru", "en"]
    
    # File Upload Settings
    MAX_FILES: int = 5
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB per file
    UPLOAD_DIR: str = "/tmp/contract_rag/uploads"
    CHUNKS_DIR: str = "/tmp/contract_rag/chunks"
    
    # Document Processing
    CHUNK_SIZE: int = 1000  # characters
    CHUNK_OVERLAP: int = 200  # characters
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384  # depends on model
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "faiss"  # "faiss" or "chromadb"
    SIMILARITY_THRESHOLD: float = 0.3
    
    # RAG Settings
    DEFAULT_TOP_K: int = 5
    MAX_CONTEXT_LENGTH: int = 4000
    MAX_QUESTION_LENGTH: int = 1000
    
    # LLM Settings (for answer generation)
    LLM_PROVIDER: str = "openai"  # "openai", "anthropic", "local"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = ""
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.1
    
    # Alternative: Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
