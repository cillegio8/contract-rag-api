# Contract RAG API 📄🤖

A powerful backend API for contract document analysis using RAG (Retrieval-Augmented Generation).

## Features

- **Multi-file Upload**: Accept up to 5 contract files (PDF, DOCX, TXT)
- **Intelligent Chunking**: Smart text splitting with overlap for better context
- **Multilingual Support**: Works with **Azerbaijani**, Russian, and English contracts
- **Auto Language Detection**: Automatically detects question language and responds accordingly
- **Semantic Search**: FAISS-powered vector similarity search
- **LLM Integration**: OpenAI and Anthropic support for answer generation
- **Session Management**: Track multiple document analysis sessions

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd contract_rag_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Required: Set your LLM API key (OpenAI or Anthropic)
```

### 3. Run the API

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Using Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

## API Endpoints

### Health Check
```
GET /
GET /health
```

### Document Upload
```
POST /upload
Content-Type: multipart/form-data

files: (up to 5 files: PDF, DOCX, TXT)
```

**Response:**
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

{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "Какова общая сумма контракта?",
  "top_k": 5,
  "language": "auto"
}
```

**Response:**
```json
{
  "answer": "Общая сумма контракта составляет 1,500,000 USD...",
  "sources": [
    {
      "source_file": "contract.pdf",
      "chunk_text": "The total contract value...",
      "relevance_score": 0.92,
      "chunk_index": 12
    }
  ],
  "confidence": 0.85,
  "session_id": "550e8400-...",
  "question": "Какова общая сумма контракта?"
}
```

### Session Management
```
GET /session/{session_id}    # Get session status
DELETE /session/{session_id} # Delete session
GET /sessions                # List all sessions
```

### Example Questions
```
GET /examples?language=az  # Get example questions in Azerbaijani (default)
GET /examples?language=ru  # Get example questions in Russian
GET /examples?language=en  # Get example questions in English
```

## Question Categories

### 1. Ümumi məlumat / Общая информация / General Information
- Müqavilənin statusu nədir? / Какой статус контракта?
- Müqavilə nə vaxt imzalanıb? / Когда контракт был подписан?
- Müqavilə nə vaxt bitir? / Когда контракт истекает?
- Müqavilə üzrə təchizatçı kimdir? / Кто является поставщиком?
- Müqavilənin ümumi məbləği nə qədərdir? / Какова общая сумма контракта?

### 2. Maliyyə şərtləri / Финансовые условия / Financial Terms
- Müqavilənin ümumi dəyəri nə qədərdir? / Какова общая стоимость контракта?
- Maddələr üzrə hansı qiymətlər razılaşdırılıb? / Какие цены согласованы по позициям?
- Qiymət indeksasiyası varmı? / Есть ли индексация цен?
- Ödəniş şərtləri hansılardır? / Какие условия оплаты?
- Ödənişin gecikdirilməsinə görə cərimə varmı? / Есть ли штрафы за задержку оплаты?

### 3. Müddətlər və öhdəliklər / Сроки и обязательства / Deadlines & Obligations
- Müqavilə nə vaxt bitir? / Когда истекает контракт?
- Milestone və ya çatdırılma mərhələləri varmı? / Есть ли milestone или этапы поставки?
- Hansı çatdırılma müddətləri göstərilib? / Какие сроки поставки указаны?
- SLA varmı? / Есть ли SLA?

### 4. Risklər və cərimələr / Риски и штрафы / Risks & Penalties
- Hansı cərimələr nəzərdə tutulub? / Какие штрафы предусмотрены?
- Penalty clauses varmı? / Есть ли penalty clauses?
- Müqavilənin ləğvi şərtləri hansılardır? / Какие условия расторжения контракта?

### 5. Təchizat həcmi / Объем поставки / Delivery Scope
- Müqaviləyə hansı mal və ya xidmətlər daxildir? / Какие товары или услуги входят в контракт?
- Hansı miqdar razılaşdırılıb? / Какое количество согласовано?
- Minimum/maksimum həcm varmı? / Есть ли минимальный/максимальный объем?

### 6. Müqavilə dəyişiklikləri / Изменения контракта / Amendments
- Müqaviləyə əlavələr olubmu? / Были ли amendments к контракту?
- Son əlavədə nə dəyişdirilib? / Что было изменено в последнем amendment?
- Müqavilənin hansı versiyaları mövcuddur? / Какие версии контракта существуют?

## Architecture

```
contract_rag_api/
├── app/
│   ├── __init__.py          # Package init
│   ├── main.py               # FastAPI application
│   ├── config.py             # Configuration settings
│   ├── models.py             # Pydantic models
│   ├── document_processor.py # Text extraction & chunking
│   ├── embedding_service.py  # Embedding generation
│   ├── vector_store.py       # FAISS vector storage
│   └── rag_engine.py         # RAG question answering
├── tests/
│   └── test_api.py           # API tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LANGUAGE` | az | Default language (az/ru/en) |
| `MAX_FILES` | 5 | Maximum files per upload |
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `EMBEDDING_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | Embedding model |
| `LLM_PROVIDER` | openai | LLM provider (openai/anthropic) |
| `LLM_MODEL` | gpt-4o-mini | LLM model name |
| `DEFAULT_TOP_K` | 5 | Default chunks to retrieve |

## Embedding Models

For multilingual (Azerbaijani + Russian + English) support:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (recommended, default)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (higher quality)

For English only:
- `sentence-transformers/all-MiniLM-L6-v2` (fast)
- `sentence-transformers/all-mpnet-base-v2` (high quality)

## Development

### Running Tests
```bash
pytest tests/ -v
```

### API Documentation
After starting the server, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT License
