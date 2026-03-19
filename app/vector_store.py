"""
Vector Store - Store and search document embeddings.

Supports:
- FAISS (default, in-memory)
- Simple NumPy-based store (fallback)
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import settings
from app.models import DocumentChunk


class VectorStore:
    """
    In-memory vector store with FAISS backend.
    
    Stores embeddings and chunks per session, enabling semantic search.
    """
    
    def __init__(self):
        """Initialize the vector store."""
        self._stores: Dict[str, dict] = {}  # session_id -> {index, chunks, embeddings}
        self._use_faiss = False
        self._init_backend()
    
    def _init_backend(self):
        """Initialize FAISS if available, otherwise use NumPy fallback."""
        try:
            import faiss
            self._use_faiss = True
            print("✅ Using FAISS for vector search")
        except ImportError:
            self._use_faiss = False
            print("⚠️ FAISS not available, using NumPy fallback")
    
    def store_chunks(self, session_id: str, chunks: List[DocumentChunk]) -> int:
        """
        Store document chunks with their embeddings.
        
        Args:
            session_id: Session identifier
            chunks: List of chunks with embeddings
            
        Returns:
            Number of chunks stored
        """
        if not chunks:
            return 0
        
        # Extract embeddings
        embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
        
        if not embeddings:
            return 0
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create store entry
        store_data = {
            "chunks": valid_chunks,
            "embeddings": embeddings_array,
            "index": None,
        }
        
        # Build FAISS index if available
        if self._use_faiss:
            import faiss
            
            dimension = embeddings_array.shape[1]
            
            # Use IVF index for larger datasets, flat for small
            if len(embeddings) > 1000:
                # IVF index with approximate search
                nlist = min(100, len(embeddings) // 10)
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(embeddings_array)
                index.add(embeddings_array)
            else:
                # Flat index for exact search (small datasets)
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings_array)
            
            store_data["index"] = index
        
        self._stores[session_id] = store_data
        return len(valid_chunks)
    
    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for most similar chunks to query.
        
        Args:
            session_id: Session identifier
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if session_id not in self._stores:
            return []
        
        store = self._stores[session_id]
        chunks = store["chunks"]
        
        if not chunks:
            return []
        
        query_vec = np.array([query_embedding], dtype=np.float32)
        
        if self._use_faiss and store["index"] is not None:
            # Use FAISS search
            index = store["index"]
            k = min(top_k, len(chunks))
            
            scores, indices = index.search(query_vec, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= threshold:
                    results.append((chunks[idx], float(score)))
            
            return results
        else:
            # NumPy fallback
            embeddings = store["embeddings"]
            
            # Cosine similarity (embeddings should be normalized)
            similarities = np.dot(embeddings, query_vec.T).flatten()
            
            # Get top-k indices
            if top_k >= len(similarities):
                top_indices = np.argsort(similarities)[::-1]
            else:
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= threshold:
                    results.append((chunks[idx], float(score)))
            
            return results
    
    def get_all_chunks(self, session_id: str) -> List[DocumentChunk]:
        """Get all chunks for a session."""
        if session_id not in self._stores:
            return []
        return self._stores[session_id]["chunks"]
    
    def get_chunk_count(self, session_id: str) -> int:
        """Get number of chunks in a session."""
        if session_id not in self._stores:
            return 0
        return len(self._stores[session_id]["chunks"])
    
    def delete_session(self, session_id: str) -> bool:
        """Delete all data for a session."""
        if session_id in self._stores:
            del self._stores[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self._stores.keys())
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        total_chunks = sum(len(s["chunks"]) for s in self._stores.values())
        return {
            "total_sessions": len(self._stores),
            "total_chunks": total_chunks,
            "backend": "faiss" if self._use_faiss else "numpy",
        }
    
    def save_session(self, session_id: str, filepath: str):
        """
        Save session data to disk.
        
        Args:
            session_id: Session to save
            filepath: Output file path
        """
        if session_id not in self._stores:
            raise ValueError(f"Session {session_id} not found")
        
        store = self._stores[session_id]
        
        # Serialize chunks
        chunks_data = []
        for chunk in store["chunks"]:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata,
            })
        
        # Save to file
        data = {
            "session_id": session_id,
            "chunks": chunks_data,
            "embeddings": store["embeddings"].tolist(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_session(self, filepath: str) -> str:
        """
        Load session data from disk.
        
        Args:
            filepath: Input file path
            
        Returns:
            Session ID
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = data["session_id"]
        
        # Reconstruct chunks
        chunks = []
        for chunk_data in data["chunks"]:
            chunk = DocumentChunk(
                chunk_id=chunk_data["chunk_id"],
                text=chunk_data["text"],
                source_file=chunk_data["source_file"],
                chunk_index=chunk_data["chunk_index"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                metadata=chunk_data.get("metadata", {}),
            )
            chunks.append(chunk)
        
        # Reconstruct embeddings
        embeddings_array = np.array(data["embeddings"], dtype=np.float32)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings_array):
            chunk.embedding = embedding.tolist()
        
        # Store
        self.store_chunks(session_id, chunks)
        
        return session_id


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-backed vector store.
    
    Provides persistent storage with automatic embedding management.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize ChromaDB store."""
        self.persist_directory = persist_directory or "/tmp/contract_rag/chroma"
        self._client = None
        self._collections: Dict[str, object] = {}
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            self._client = chromadb.Client(ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            ))
            print(f"✅ ChromaDB initialized at {self.persist_directory}")
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
    
    def store_chunks(self, session_id: str, chunks: List[DocumentChunk]) -> int:
        """Store chunks in ChromaDB collection."""
        if not chunks:
            return 0
        
        # Get or create collection
        collection = self._client.get_or_create_collection(
            name=f"session_{session_id}",
            metadata={"session_id": session_id}
        )
        
        # Prepare data
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        
        self._collections[session_id] = collection
        return len(chunks)
    
    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search ChromaDB collection."""
        if session_id not in self._collections:
            try:
                collection = self._client.get_collection(f"session_{session_id}")
                self._collections[session_id] = collection
            except Exception:
                return []
        
        collection = self._collections[session_id]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        
        output = []
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0] if "distances" in results else [0] * len(results["ids"][0]),
        )):
            # Convert distance to similarity (ChromaDB uses L2 distance)
            similarity = 1 / (1 + distance)
            
            if similarity >= threshold:
                chunk = DocumentChunk(
                    chunk_id=doc_id,
                    text=document,
                    source_file=metadata.get("source_file", "unknown"),
                    chunk_index=metadata.get("chunk_index", i),
                    start_char=0,
                    end_char=len(document),
                )
                output.append((chunk, similarity))
        
        return output
    
    def delete_session(self, session_id: str) -> bool:
        """Delete ChromaDB collection."""
        try:
            self._client.delete_collection(f"session_{session_id}")
            if session_id in self._collections:
                del self._collections[session_id]
            return True
        except Exception:
            return False
