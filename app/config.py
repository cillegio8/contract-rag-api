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
    EMBEDDING_PROVIDER: str = "openrouter"  # "openai", "openrouter", or "sentence-transformers"
    EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"  # OpenRouter Qwen embedding model
    EMBEDDING_DIMENSION: int = 768  # 768 for Qwen embeddings, 1536 for text-embedding-3-small, 384 for MiniLM-L12-v2
    OPENAI_EMBEDDINGS_API_KEY: str = ""  # Set if different from LLM_API_KEY (for OpenAI provider)
    OPENROUTER_EMBEDDINGS_API_KEY: str = ""  # OpenRouter API key for embeddings (can be same as OPENROUTER_API_KEY)
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "faiss"  # "faiss" or "chromadb"
    SIMILARITY_THRESHOLD: float = 0.3
    
    # RAG Settings
    DEFAULT_TOP_K: int = 5
    MAX_CONTEXT_LENGTH: int = 4000
    MAX_QUESTION_LENGTH: int = 1000
    
    # LLM Settings (for answer generation)
    LLM_PROVIDER: str = "openrouter"  # "openai", "anthropic", "openrouter", "local"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = ""
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.1
    
    # Alternative: Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
    
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-4o-mini"  # OpenRouter model identifier
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
