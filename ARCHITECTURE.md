# Contracts-RAG Architecture

## Overview

A **Retrieval-Augmented Generation (RAG) API** for contract document analysis. Users upload contract documents (PDF, DOCX, TXT) and ask questions about them in Azerbaijani, Russian, or English. The system provides semantically-searched answers with source citations.

---

## Project Structure

```
Contracts-RAG/
├── app/
│   ├── config.py               # Configuration & environment settings
│   ├── models.py               # Pydantic data models
│   ├── main.py                 # FastAPI app & API endpoints
│   ├── document_processor.py   # Text extraction & chunking
│   ├── embedding_service.py    # Vector generation
│   ├── vector_store.py         # FAISS vector storage
│   └── rag_engine.py           # RAG pipeline & answer generation
├── index.html                  # Frontend UI (served by FastAPI)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | FastAPI + Uvicorn (ASGI) |
| Embeddings (default) | OpenRouter `qwen/qwen3-embedding-8b` (768d) |
| Embeddings (local fallback) | `sentence-transformers` (384d) |
| Vector Search | FAISS (Flat / IVF) — falls back to NumPy cosine similarity |
| LLM | OpenRouter → `openai/gpt-4o-mini` (via OpenAI SDK) |
| Document Parsing | pdfplumber, python-docx |
| Data Validation | Pydantic v2 |
| Frontend | Vanilla JS single-page HTML, co-located with the API |

---

## Component Breakdown

### `config.py` — Settings
Pydantic-based settings loaded from environment variables. Key values:

- **Chunking**: 1000-char chunks, 200-char overlap
- **Embedding provider**: `openrouter` (default) or `sentence-transformers`
- **Similarity threshold**: 0.3 · top-k: 5
- **LLM**: temperature 0.1, max 1000 tokens
- **CORS**: wildcard (allow all)

---

### `models.py` — Data Models

| Model | Purpose |
|-------|---------|
| `QuestionRequest` | session_id, question, top_k, language |
| `UploadResponse` | session_id, files_processed, total_chunks, status |
| `QuestionResponse` | answer, sources[], confidence |
| `DocumentChunk` | text segment + embedding + source metadata |
| `ContractSession` | session metadata and state |
| `SourceReference` | citation with relevance score |

---

### `document_processor.py` — Document Processing

Extracts text and splits into overlapping chunks.

**Supported formats:**
- **PDF** — pdfplumber (text + tables)
- **DOCX** — python-docx (paragraphs + tables)
- **TXT** — multi-encoding detection (utf-8, cp1251, latin-1, …)

**Chunking strategy:**
1. Try to break on paragraph boundaries
2. Fall back to sentence endings
3. Fall back to word/character boundaries
4. Maintains configurable overlap for context preservation

Also classifies contract type (supply, service, NDA, amendment) from extracted text.

---

### `embedding_service.py` — Embeddings

Generates dense vector representations. Supports two providers:

| Provider | Model | Dimensions | Notes |
|----------|-------|-----------|-------|
| OpenRouter (default) | `qwen/qwen3-embedding-8b` | 768 | API-based, multilingual |
| sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Local, offline |

Falls back to deterministic mock embeddings when neither is available (testing).

Key methods: `embed_text()`, `embed_texts()` (batched), `embed_chunks()`, `compute_similarity()`, `find_most_similar()`.

---

### `vector_store.py` — Vector Storage

Two implementations, both session-scoped:

| Class | Backend | Persistence |
|-------|---------|------------|
| `VectorStore` (default) | FAISS in-memory | None (RAM only) |
| `ChromaVectorStore` | ChromaDB | Disk-persistent |

**FAISS index selection:**
- `IndexFlatIP` for < 1000 vectors
- `IndexIVFFlat` for larger datasets

Key methods: `store_chunks()`, `search()`, `delete_session()`, `save_session()` / `load_session()`.

---

### `rag_engine.py` — RAG Pipeline

Orchestrates question answering end-to-end.

**Question classification** (6 categories × 3 languages):
`general_info` · `financial` · `deadlines` · `risks` · `scope` · `amendments`

**Full pipeline:**
```
1. Detect question language (AZ / RU / EN via character analysis)
2. Classify into category
3. Generate query embedding
4. FAISS semantic search with threshold filtering
5. Build context from top-k chunks
6. Call LLM with category-aware prompt
7. Return answer + source citations + confidence score
```

Category hints are injected into the LLM prompt to improve retrieval relevance.

---

### `main.py` — FastAPI Application

Serves the frontend HTML and exposes the REST API.

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve frontend HTML |
| GET | `/health` | Health check + stats |
| POST | `/upload` | Upload & process documents |
| POST | `/ask` | Answer question via RAG |
| GET | `/session/{id}` | Session status |
| DELETE | `/session/{id}` | Delete session & vectors |
| GET | `/sessions` | List active sessions |
| GET | `/examples` | Example questions by language |

**Key patterns:**
- **Singleton services** via `app.state` — ensures vectors persist between `/upload` and `/ask` calls
- **Lifespan context manager** — service startup/shutdown
- **Async/await** throughout — full async request handling

---

## Data Flow

```
UPLOAD FLOW
───────────
Browser  →  POST /upload  →  DocumentProcessor (extract + chunk)
                          →  EmbeddingService (embed chunks)
                          →  VectorStore (index in FAISS per session)
                          ←  { session_id, total_chunks, status }

QUESTION FLOW
─────────────
Browser  →  POST /ask  →  EmbeddingService (embed query)
                       →  VectorStore (FAISS search, top-k chunks)
                       →  RAGEngine (classify + build context)
                       →  OpenRouter LLM (generate answer)
                       ←  { answer, sources[], confidence }
```

---

## Architectural Patterns

### Singleton Services
All services (`EmbeddingService`, `VectorStore`, `RAGEngine`) are created once on startup and shared via `app.state`. This ensures the in-memory FAISS index survives between the upload and question requests.

### Dual Provider / Adapter Pattern
Both embeddings and vector storage use an adapter pattern with a primary provider and a fallback, making the system runnable without external API keys.

### In-Memory Session Storage
Sessions are stored in a plain Python dict (`dict[str, ContractSession]`). Simple and dependency-free, but does not survive restarts. Replace with Redis or PostgreSQL for production.

### Frontend Co-location
`index.html` is served by FastAPI itself, eliminating CORS issues and simplifying deployment — one container, one port.

---

## Configuration (`.env`)

```env
# LLM
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...        # set
OPENROUTER_MODEL=openai/gpt-4o-mini

# Embedding
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
EMBEDDING_DIMENSION=768
OPENROUTER_EMBEDDINGS_API_KEY=sk-or-v1-...  # set (same key as above)

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Search
SIMILARITY_THRESHOLD=0.3
DEFAULT_TOP_K=5

# Languages
DEFAULT_LANGUAGE=az
SUPPORTED_LANGUAGES=["az", "ru", "en"]
```

Both the LLM and embedding calls go through **OpenRouter** using the same API key prefix (`sk-or-v1-…`). The embedding model is `qwen/qwen3-embedding-8b` at 768 dimensions; the LLM is `openai/gpt-4o-mini` routed via OpenRouter.

---

## Deployment

Single Docker container:

```yaml
# docker-compose.yml
contract-rag-api:
  build: .
  ports: ["8000:8000"]
  volumes:
    - /tmp/contract_rag/uploads   # uploaded files
    - /root/.cache                # embedding model cache
  healthcheck: GET /health every 30s
```

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Error Handling & Fallbacks

| Failure | Fallback |
|---------|---------|
| Embedding API unreachable | Deterministic mock embeddings |
| LLM unavailable | Mock answer showing retrieved context |
| FAISS not installed | NumPy cosine similarity |
| Session not found | HTTP 404 |
| Unsupported file type / too large | HTTP 400 / 413 |

---

## Performance Notes

| Parameter | Default | Config key |
|-----------|---------|-----------|
| Chunk size | 1000 chars | `CHUNK_SIZE` |
| Chunk overlap | 200 chars | `CHUNK_OVERLAP` |
| Embedding dims | 768 | `EMBEDDING_DIMENSION` |
| Max files/upload | 5 | `MAX_FILES` |
| Max file size | 50 MB | `MAX_FILE_SIZE` |
| Top-k results | 5 | `DEFAULT_TOP_K` |
| Similarity threshold | 0.3 | `SIMILARITY_THRESHOLD` |

FAISS index type is selected automatically: `IndexFlatIP` for < 1000 vectors, `IndexIVFFlat` for larger datasets.

---

## Limitations & Production Considerations

- **No persistence** — sessions and vectors are lost on restart; add Redis + a persistent vector store (e.g., Qdrant, Weaviate) for production
- **Single-process** — FAISS lives in one process's RAM; horizontal scaling requires a shared external vector store
- **No authentication** — add API key / JWT middleware before exposing publicly
- **Wildcard CORS** — restrict `allow_origins` in production
