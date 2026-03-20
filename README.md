[![Architecture](https://img.shields.io/badge/docs-architecture-blue)](ARCHITECTURE.md)

# Contract RAG API 📄

A backend API + web frontend for contract document analysis using RAG (Retrieval-Augmented Generation). Upload contracts, ask questions in Azerbaijani, Russian, or English, and get precise answers with source citations.

## Features

- **Web Frontend**: Built-in UI served at `/` — white & teal design, no separate deployment needed
- **Multi-file Upload**: Accept up to 5 contract files (PDF, DOCX, TXT)
- **Intelligent Chunking**: Smart text splitting with overlap for better context
- **Multilingual Support**: Azerbaijani, Russian, and English contracts and questions
- **Auto Language Detection**: Detects question language and responds accordingly
- **Semantic Search**: FAISS-powered vector similarity search
- **LLM Integration**: OpenRouter for answer generation
- **Embeddings**: OpenRouter (Qwen3-embedding-8b, default) or local sentence-transformers
- **Session Management**: Track multiple document analysis sessions

## Quick Start

### 1. Installation

```bash
git clone <repo>
cd Contracts-RAG

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```env
# Embeddings (OpenRouter Qwen, default)
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
EMBEDDING_DIMENSION=4096
OPENROUTER_EMBEDDINGS_API_KEY=your_key_here

# LLM (OpenRouter)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### 3. Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — the frontend loads automatically.

### 4. Docker

```bash
docker-compose up -d
docker-compose logs -f
```

## Deployment (Northflank)

The app is a single Docker service — the FastAPI backend serves the frontend at `/`.

1. Connect your git repo to a Northflank service
2. Set build to **Dockerfile** (repo root)
3. Set port to **8000**
4. Add environment variables (see Configuration above)
5. Push to git → Northflank auto-rebuilds

Live URL serves the UI directly. API docs at `/docs`.

## API Endpoints

### Frontend
```
GET /          → Serves index.html (web UI)
```

### Health
```
GET /health    → Service health + active session count
```

### Documents
```
POST /upload
Content-Type: multipart/form-data
files: (up to 5 files: PDF, DOCX, TXT)
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Documents uploaded and processed successfully",
  "files_processed": 3,
  "total_chunks": 45,
  "status": "ready"
}
```

### Ask Questions
```
POST /ask
Content-Type: application/json
```

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "Ödəniş şərtləri hansılardır?",
  "top_k": 5,
  "language": "auto"
}
```

Response:
```json
{
  "answer": "Müqavilə üzrə ödəniş 30 gün ərzində həyata keçirilir...",
  "sources": [
    {
      "source_file": "contract.pdf",
      "chunk_text": "Payment shall be made within 30 days...",
      "relevance_score": 0.92,
      "chunk_index": 12
    }
  ],
  "confidence": 0.85,
  "session_id": "550e8400-...",
  "question": "Ödəniş şərtləri hansılardır?"
}
```

### Sessions
```
GET    /session/{session_id}   # Status of a session
DELETE /session/{session_id}   # Delete session and its vectors
GET    /sessions               # List all active sessions
```

### Example Questions
```
GET /examples?language=az   # Azerbaijani (default)
GET /examples?language=ru   # Russian
GET /examples?language=en   # English
```

## Question Categories

The RAG engine classifies questions into categories for better retrieval:

| Category | Examples |
|----------|---------|
| **General Info** | Contract status, parties, signing date, currency |
| **Financial Terms** | Total value, prices, payment terms, late payment penalties |
| **Deadlines & Obligations** | Expiry date, milestones, delivery schedule, SLA |
| **Risks & Penalties** | Penalty clauses, warranty obligations, termination conditions |
| **Delivery Scope** | Goods/services included, quantities, min/max volumes |
| **Amendments** | Amendment history, last change date, contract versions |

## Architecture

```
Contracts-RAG/
├── app/
│   ├── main.py               # FastAPI app — serves frontend + all API endpoints
│   ├── config.py             # Settings (env vars via pydantic-settings)
│   ├── models.py             # Pydantic request/response models
│   ├── document_processor.py # PDF/DOCX/TXT extraction and chunking
│   ├── embedding_service.py  # Embeddings: OpenRouter / OpenAI / sentence-transformers
│   ├── vector_store.py       # FAISS in-memory vector store (per session)
│   ├── rag_engine.py         # Retrieval + LLM answer generation
│   └── __init__.py
├── index.html                # Frontend (served by FastAPI at /)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Key design decisions

- **Singleton services**: `EmbeddingService`, `VectorStore`, and `RAGEngine` are created once at startup (via `app.state`) and shared across all requests. This ensures uploaded vectors persist for the lifetime of the process.
- **In-memory sessions**: Session data lives in a Python dict. Restarting the container clears all sessions — users must re-upload.
- **Same-origin frontend**: `index.html` is served by FastAPI itself, so the frontend calls relative API paths and works on any deployment URL without hardcoded hostnames.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `openrouter` | `openrouter` or `sentence-transformers` |
| `EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | Model for embeddings |
| `EMBEDDING_DIMENSION` | `768` | Vector dimension (768 for Qwen3-8b via OpenRouter, 384 for MiniLM) |
| `OPENROUTER_EMBEDDINGS_API_KEY` | — | OpenRouter key for embeddings |
| `OPENROUTER_API_KEY` | — | OpenRouter key for LLM |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | OpenRouter model identifier |
| `MAX_FILES` | `5` | Max files per upload |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum score to include a chunk in results |
| `DEFAULT_TOP_K` | `5` | Chunks to retrieve per query |

## Alternative Embedding Models

| Provider | Model | Dimension | Notes |
|----------|-------|-----------|-------|
| OpenRouter | `qwen/qwen3-embedding-8b` | 768 | Default, strong multilingual |
| Local | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | No API key needed |
| Local | `paraphrase-multilingual-mpnet-base-v2` | 768 | Higher quality local option |

## Development

```bash
# API docs (Swagger UI)
open http://localhost:8000/docs

# Run tests
pytest tests/ -v
```

## License

MIT License
