"""
Contract RAG API - Backend for Contract Document Analysis

This API provides:
1. File upload endpoint (up to 5 files: PDF, DOCX, TXT)
2. Document chunking and embedding
3. Question answering about contracts using RAG

Categories of questions supported:
- General contract information (status, dates, parties, amounts)
- Financial terms (pricing, payment conditions, penalties)
- Deadlines and obligations (expiration, milestones, SLAs)
- Risks and penalties (termination clauses, guarantees)
- Delivery scope (goods/services, quantities, limits)
- Contract amendments (versions, changes history)
"""

import os
import uuid
import hashlib
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.models import (
    UploadResponse,
    QuestionRequest,
    QuestionResponse,
    ContractSession,
    DocumentChunk,
    SessionStatusResponse,
)
from app.document_processor import DocumentProcessor
from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine


# In-memory session storage (replace with Redis/DB in production)
sessions: dict[str, ContractSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting Contract RAG API...")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHUNKS_DIR, exist_ok=True)
    yield
    # Shutdown
    print("👋 Shutting down Contract RAG API...")


app = FastAPI(
    title="Contract RAG API",
    description="API for uploading contracts and asking questions about them using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend HTML."""
    candidates = [
        FRONTEND_DIR / "index.html",
        Path.cwd() / "index.html",
        Path("/app/index.html"),
    ]
    for index_path in candidates:
        if index_path.exists():
            return FileResponse(index_path)
    checked = [str(p) for p in candidates]
    return {"message": "index.html not found", "checked_paths": checked}


# Dependency injection

def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor()


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_vector_store() -> VectorStore:
    return VectorStore()


def get_rag_engine(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
) -> RAGEngine:
    return RAGEngine(embedding_service, vector_store)


# ============== API Endpoints ==============


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "api": "up",
            "embedding_service": "up",
            "vector_store": "up",
        },
        "active_sessions": len(sessions),
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(
    files: List[UploadFile] = File(..., description="Up to 5 contract files (PDF, DOCX, TXT)"),
    background_tasks: BackgroundTasks = None,
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Upload up to 5 contract documents for analysis.
    
    Supported formats:
    - PDF (.pdf)
    - Word Documents (.docx)
    - Text files (.txt)
    
    Returns a session_id to use for subsequent questions.
    """
    # Validate file count
    if len(files) > settings.MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_FILES} files allowed. You uploaded {len(files)}."
        )
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    
    # Validate file types and sizes
    allowed_extensions = {".pdf", ".docx", ".txt"}
    processed_files = []
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' has unsupported format. Allowed: PDF, DOCX, TXT"
            )
        
        # Read file content
        content = await file.read()
        
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' exceeds maximum size of {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        processed_files.append({
            "filename": file.filename,
            "content": content,
            "extension": ext,
            "size": len(content),
        })
    
    # Create session
    session_id = str(uuid.uuid4())
    session = ContractSession(
        session_id=session_id,
        created_at=datetime.utcnow(),
        files=[f["filename"] for f in processed_files],
        status="processing",
        total_chunks=0,
    )
    sessions[session_id] = session
    
    # Process documents
    all_chunks: List[DocumentChunk] = []
    
    for file_data in processed_files:
        try:
            # Extract text from document
            text = doc_processor.extract_text(
                file_data["content"],
                file_data["extension"],
                file_data["filename"]
            )
            
            # Chunk the text
            chunks = doc_processor.chunk_text(
                text,
                source_file=file_data["filename"],
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            session.status = "error"
            session.error_message = f"Failed to process {file_data['filename']}: {str(e)}"
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file '{file_data['filename']}': {str(e)}"
            )
    
    # Generate embeddings
    try:
        chunks_with_embeddings = embedding_service.embed_chunks(all_chunks)
    except Exception as e:
        session.status = "error"
        session.error_message = f"Embedding generation failed: {str(e)}"
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )
    
    # Store in vector store
    try:
        vector_store.store_chunks(session_id, chunks_with_embeddings)
    except Exception as e:
        session.status = "error"
        session.error_message = f"Vector storage failed: {str(e)}"
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store document vectors: {str(e)}"
        )
    
    # Update session status
    session.status = "ready"
    session.total_chunks = len(chunks_with_embeddings)
    
    return UploadResponse(
        session_id=session_id,
        message="Documents uploaded and processed successfully",
        files_processed=len(processed_files),
        total_chunks=len(chunks_with_embeddings),
        status="ready",
    )


@app.post("/ask", response_model=QuestionResponse, tags=["Questions"])
async def ask_question(
    request: QuestionRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Ask a question about the uploaded contract documents.
    
    Question categories supported:
    
    1. **General Information**
       - Contract status, signing date, expiration
       - Renewal options, supplier info, total amount
       - Currency, contract owner, related tender
    
    2. **Financial Terms**
       - Total cost, agreed prices per item
       - Price indexation, payment terms
       - Payment deadlines, late payment penalties
       - Purchase volume limits
    
    3. **Deadlines & Obligations**
       - Expiration date, milestones
       - Next delivery date, delivery terms
       - SLA conditions, delay penalties
    
    4. **Risks & Penalties**
       - Penalty clauses, guarantee obligations
       - Termination conditions, notice period
    
    5. **Delivery Scope**
       - Goods/services included
       - Agreed quantities, min/max volumes
    
    6. **Contract Amendments**
       - Amendment history, last change date
       - Changes in latest amendment
       - Existing versions
    """
    # Validate session exists
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Please upload documents first."
        )
    
    session = sessions[request.session_id]
    
    # Check session status
    if session.status == "processing":
        raise HTTPException(
            status_code=400,
            detail="Documents are still being processed. Please wait."
        )
    
    if session.status == "error":
        raise HTTPException(
            status_code=400,
            detail=f"Session has an error: {session.error_message}"
        )
    
    # Validate question
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if len(request.question) > settings.MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long. Maximum {settings.MAX_QUESTION_LENGTH} characters."
        )
    
    # Generate answer using RAG
    try:
        answer, sources, confidence = rag_engine.answer_question(
            session_id=request.session_id,
            question=request.question,
            top_k=request.top_k or settings.DEFAULT_TOP_K,
            language=request.language or "auto",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )
    
    # Update session stats
    session.questions_asked = getattr(session, 'questions_asked', 0) + 1
    session.last_question_at = datetime.utcnow()
    
    return QuestionResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        session_id=request.session_id,
        question=request.question,
    )


@app.get("/session/{session_id}", response_model=SessionStatusResponse, tags=["Sessions"])
async def get_session_status(session_id: str):
    """Get the status of a document processing session."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found."
        )
    
    session = sessions[session_id]
    
    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status,
        files=session.files,
        total_chunks=session.total_chunks,
        created_at=session.created_at,
        questions_asked=getattr(session, 'questions_asked', 0),
        error_message=session.error_message,
    )


@app.delete("/session/{session_id}", tags=["Sessions"])
async def delete_session(
    session_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
):
    """Delete a session and its associated data."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found."
        )
    
    # Clean up vector store
    try:
        vector_store.delete_session(session_id)
    except Exception as e:
        print(f"Warning: Failed to clean vector store for session {session_id}: {e}")
    
    # Remove session
    del sessions[session_id]
    
    return {"message": f"Session '{session_id}' deleted successfully."}


@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    """List all active sessions."""
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "status": s.status,
                "files": s.files,
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions.values()
        ]
    }


# ============== Example Questions Endpoint ==============

@app.get("/examples", tags=["Help"])
async def get_example_questions(language: str = "az"):
    """
    Get example questions organized by category.
    
    Args:
        language: Language code ("az" for Azerbaijani, "ru" for Russian, "en" for English)
    """
    examples = {
        "az": {
            "categories": {
                "general_info": {
                    "name": "Müqavilə haqqında ümumi məlumat",
                    "examples": [
                        "Müqavilənin statusu nədir?",
                        "Müqavilə nə vaxt imzalanıb?",
                        "Müqavilə nə vaxt bitir?",
                        "Uzadılma seçimi varmı?",
                        "Müqavilə üzrə təchizatçı kimdir?",
                        "Müqavilənin ümumi məbləği nə qədərdir?",
                        "Müqavilə hansı valyutadadır?",
                        "Müqavilənin sahibi (contract owner) kimdir?",
                        "Müqavilə hansı tenderə aiddir?",
                    ]
                },
                "financial_terms": {
                    "name": "Maliyyə şərtləri",
                    "examples": [
                        "Müqavilənin ümumi dəyəri nə qədərdir?",
                        "Maddələr üzrə hansı qiymətlər razılaşdırılıb?",
                        "Qiymət indeksasiyası varmı?",
                        "Ödəniş şərtləri hansılardır?",
                        "Ödəniş neçə gün ərzində həyata keçirilir?",
                        "Ödənişin gecikdirilməsinə görə cərimə varmı?",
                        "Alış həcmi üzrə limit varmı?",
                    ]
                },
                "deadlines_obligations": {
                    "name": "Müddətlər və öhdəliklər",
                    "examples": [
                        "Müqavilə nə vaxt bitir?",
                        "Milestone və ya çatdırılma mərhələləri varmı?",
                        "Növbəti çatdırılma nə vaxt olmalıdır?",
                        "Hansı çatdırılma müddətləri göstərilib?",
                        "SLA varmı?",
                        "Gecikmə üçün hansı cərimələr nəzərdə tutulub?",
                    ]
                },
                "risks_penalties": {
                    "name": "Risklər və cərimələr",
                    "examples": [
                        "Hansı cərimələr nəzərdə tutulub?",
                        "Penalty clauses varmı?",
                        "Zəmanət öhdəlikləri varmı?",
                        "Müqavilənin ləğvi şərtləri hansılardır?",
                        "Müqaviləni neçə gün ərzində ləğv etmək olar?",
                    ]
                },
                "delivery_scope": {
                    "name": "Təchizat həcmi",
                    "examples": [
                        "Müqaviləyə hansı mal və ya xidmətlər daxildir?",
                        "Hansı miqdar razılaşdırılıb?",
                        "Minimum həcm varmı?",
                        "Maksimum həcm varmı?",
                    ]
                },
                "amendments": {
                    "name": "Müqavilə dəyişiklikləri",
                    "examples": [
                        "Müqaviləyə əlavələr olubmu?",
                        "Son dəyişiklik nə vaxt olub?",
                        "Son əlavədə nə dəyişdirilib?",
                        "Müqavilənin hansı versiyaları mövcuddur?",
                    ]
                }
            }
        },
        "ru": {
            "categories": {
                "general_info": {
                    "name": "Общая информация о контракте",
                    "examples": [
                        "Какой статус контракта?",
                        "Когда контракт был подписан?",
                        "Когда контракт истекает?",
                        "Есть ли опция продления?",
                        "Кто является поставщиком по контракту?",
                        "Какова общая сумма контракта?",
                        "В какой валюте контракт?",
                        "Кто владелец контракта (contract owner)?",
                        "К какому тендеру относится контракт?",
                    ]
                },
                "financial_terms": {
                    "name": "Финансовые условия",
                    "examples": [
                        "Какова общая стоимость контракта?",
                        "Какие цены согласованы по позициям?",
                        "Есть ли индексация цен?",
                        "Какие условия оплаты?",
                        "Через сколько дней осуществляется платеж?",
                        "Есть ли штрафы за задержку оплаты?",
                        "Есть ли лимит по объему закупки?",
                    ]
                },
                "deadlines_obligations": {
                    "name": "Сроки и обязательства",
                    "examples": [
                        "Когда истекает контракт?",
                        "Есть ли milestone или этапы поставки?",
                        "Когда должна быть следующая поставка?",
                        "Какие сроки поставки указаны?",
                        "Есть ли SLA?",
                        "Какие штрафы предусмотрены за задержку?",
                    ]
                },
                "risks_penalties": {
                    "name": "Риски и штрафы",
                    "examples": [
                        "Какие штрафы предусмотрены?",
                        "Есть ли penalty clauses?",
                        "Есть ли обязательства по гарантии?",
                        "Какие условия расторжения контракта?",
                        "За сколько дней можно расторгнуть контракт?",
                    ]
                },
                "delivery_scope": {
                    "name": "Объем поставки",
                    "examples": [
                        "Какие товары или услуги входят в контракт?",
                        "Какое количество согласовано?",
                        "Есть ли минимальный объем?",
                        "Есть ли максимальный объем?",
                    ]
                },
                "amendments": {
                    "name": "Изменения контракта",
                    "examples": [
                        "Были ли amendments к контракту?",
                        "Когда было последнее изменение?",
                        "Что было изменено в последнем amendment?",
                        "Какие версии контракта существуют?",
                    ]
                }
            }
        },
        "en": {
            "categories": {
                "general_info": {
                    "name": "General Contract Information",
                    "examples": [
                        "What is the contract status?",
                        "When was the contract signed?",
                        "When does the contract expire?",
                        "Is there a renewal option?",
                        "Who is the supplier under the contract?",
                        "What is the total contract amount?",
                        "What currency is the contract in?",
                        "Who is the contract owner?",
                        "Which tender does the contract belong to?",
                    ]
                },
                "financial_terms": {
                    "name": "Financial Terms",
                    "examples": [
                        "What is the total contract value?",
                        "What prices are agreed per item?",
                        "Is there price indexation?",
                        "What are the payment terms?",
                        "How many days until payment is made?",
                        "Are there late payment penalties?",
                        "Is there a volume purchase limit?",
                    ]
                },
                "deadlines_obligations": {
                    "name": "Deadlines and Obligations",
                    "examples": [
                        "When does the contract expire?",
                        "Are there milestones or delivery stages?",
                        "When should the next delivery be?",
                        "What delivery deadlines are specified?",
                        "Is there an SLA?",
                        "What penalties are specified for delays?",
                    ]
                },
                "risks_penalties": {
                    "name": "Risks and Penalties",
                    "examples": [
                        "What penalties are specified?",
                        "Are there penalty clauses?",
                        "Are there warranty obligations?",
                        "What are the contract termination conditions?",
                        "How many days notice is required to terminate?",
                    ]
                },
                "delivery_scope": {
                    "name": "Delivery Scope",
                    "examples": [
                        "What goods or services are included in the contract?",
                        "What quantity is agreed?",
                        "Is there a minimum volume?",
                        "Is there a maximum volume?",
                    ]
                },
                "amendments": {
                    "name": "Contract Amendments",
                    "examples": [
                        "Have there been any amendments to the contract?",
                        "When was the last change?",
                        "What was changed in the last amendment?",
                        "What versions of the contract exist?",
                    ]
                }
            }
        }
    }
    
    return examples.get(language, examples["az"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
