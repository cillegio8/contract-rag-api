"""
Contract RAG API - Application Package

A FastAPI-based RAG system for contract document analysis.
"""

from app.config import settings
from app.models import (
    DocumentChunk,
    ContractSession,
    QuestionRequest,
    QuestionResponse,
    UploadResponse,
    SessionStatusResponse,
    SourceReference,
)
from app.document_processor import DocumentProcessor
from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine

__version__ = "1.0.0"
__all__ = [
    "settings",
    "DocumentChunk",
    "ContractSession",
    "QuestionRequest",
    "QuestionResponse",
    "UploadResponse",
    "SessionStatusResponse",
    "SourceReference",
    "DocumentProcessor",
    "EmbeddingService",
    "VectorStore",
    "RAGEngine",
]
