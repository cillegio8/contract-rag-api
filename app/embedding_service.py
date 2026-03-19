"""
Embedding Service - Generate embeddings for document chunks.

Supports multiple embedding providers:
- Sentence Transformers (local, multilingual)
- OpenAI Embeddings
- Custom embedding endpoint
"""

from typing import List, Optional
import numpy as np

from app.config import settings
from app.models import DocumentChunk


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Default: Uses sentence-transformers for multilingual support (Russian + English).
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Override model name from settings
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            print(f"✅ Loaded embedding model: {self.model_name} (dim={self.dimension})")
        except ImportError:
            print("⚠️ sentence-transformers not available, using mock embeddings")
            self._model = "mock"
        except Exception as e:
            print(f"⚠️ Failed to load model {self.model_name}: {e}, using mock embeddings")
            self._model = "mock"
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        self._load_model()
        
        if self._model == "mock":
            return self._mock_embedding(text)
        
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        self._load_model()
        
        if self._model == "mock":
            return [self._mock_embedding(text) for text in texts]
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Add embeddings to document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Chunks with embeddings populated
        """
        if not chunks:
            return chunks
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def _mock_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic mock embedding for testing.
        
        Creates a pseudo-random but reproducible embedding based on text hash.
        """
        import hashlib
        
        # Create hash of text
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Use hash to seed random generator for reproducibility
        seed = int(text_hash[:8], 16)
        rng = np.random.RandomState(seed)
        
        # Generate and normalize embedding
        embedding = rng.randn(self.dimension)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1, higher is more similar)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query vector
            candidate_embeddings: List of candidate vectors
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity descending
        """
        if not candidate_embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)
        
        # Compute all similarities at once
        similarities = np.dot(candidates, query_vec)
        
        # Get top-k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class OpenAIEmbeddingService(EmbeddingService):
    """
    OpenAI-based embedding service.
    
    Uses text-embedding-3-small by default.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        super().__init__()
        self.api_key = api_key or settings.LLM_API_KEY
        self.model_name = model
        self.dimension = 1536  # Default for text-embedding-3-small
        self._client = None
    
    def _load_model(self):
        """Initialize OpenAI client."""
        if self._client is not None:
            return
        
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        self._load_model()
        
        response = self._client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
