"""
Pydantic models for Contract RAG API.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# ============== Document Models ==============

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "abc123-0",
                "text": "This Agreement is entered into as of...",
                "source_file": "contract.pdf",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 1000,
                "metadata": {"section": "header"}
            }
        }


class ContractSession(BaseModel):
    """Represents an active document processing session."""
    session_id: str
    created_at: datetime
    files: List[str]
    status: str = "processing"  # processing, ready, error
    total_chunks: int = 0
    questions_asked: int = 0
    last_question_at: Optional[datetime] = None
    error_message: Optional[str] = None


# ============== API Request Models ==============

class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    session_id: str = Field(..., description="Session ID from document upload")
    question: str = Field(..., description="Question about the contract(s)", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, description="Number of relevant chunks to retrieve", ge=1, le=20)
    language: Optional[str] = Field(default="auto", description="Response language: 'ru', 'en', or 'auto'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "question": "Какова общая сумма контракта?",
                "top_k": 5,
                "language": "ru"
            }
        }


# ============== API Response Models ==============

class SourceReference(BaseModel):
    """Reference to a source chunk used in answering."""
    source_file: str
    chunk_text: str
    relevance_score: float
    chunk_index: int


class UploadResponse(BaseModel):
    """Response after successful document upload."""
    session_id: str
    message: str
    files_processed: int
    total_chunks: int
    status: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Documents uploaded and processed successfully",
                "files_processed": 3,
                "total_chunks": 45,
                "status": "ready"
            }
        }


class QuestionResponse(BaseModel):
    """Response to a question about the contract."""
    answer: str
    sources: List[SourceReference]
    confidence: float = Field(..., description="Confidence score 0-1")
    session_id: str
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Общая сумма контракта составляет 1,500,000 USD согласно разделу 3.1 договора.",
                "sources": [
                    {
                        "source_file": "contract.pdf",
                        "chunk_text": "The total contract value is USD 1,500,000...",
                        "relevance_score": 0.92,
                        "chunk_index": 12
                    }
                ],
                "confidence": 0.85,
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "question": "Какова общая сумма контракта?"
            }
        }


class SessionStatusResponse(BaseModel):
    """Response with session status information."""
    session_id: str
    status: str
    files: List[str]
    total_chunks: int
    created_at: datetime
    questions_asked: int
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "ready",
                "files": ["contract.pdf", "amendment1.docx"],
                "total_chunks": 45,
                "created_at": "2024-01-15T10:30:00Z",
                "questions_asked": 5,
                "error_message": None
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Session not found",
                "error_code": "SESSION_NOT_FOUND"
            }
        }
