"""
Embedding model wrapper for local memory.

Uses sentence-transformers/all-MiniLM-L6-v2 by default:
- 80MB model size
- 384-dimensional embeddings
- ~10ms inference per text (CPU)
- Good quality for semantic search
"""

import logging
from typing import Optional
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Lazy import to avoid loading model until needed
_model = None


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""
    embedding: list[float]
    dimension: int
    model: str
    inference_ms: float


def get_model():
    """Get or load the embedding model (lazy loading)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (first call, ~150MB RAM)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
    return _model


def embed(text: str) -> EmbeddingResult:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text to embed
        
    Returns:
        EmbeddingResult with embedding vector and metadata
    """
    import time
    
    model = get_model()
    start = time.time()
    embedding = model.encode(text, convert_to_numpy=True)
    elapsed = (time.time() - start) * 1000
    
    return EmbeddingResult(
        embedding=embedding.tolist(),
        dimension=len(embedding),
        model="all-MiniLM-L6-v2",
        inference_ms=round(elapsed, 2)
    )


def embed_batch(texts: list[str]) -> list[EmbeddingResult]:
    """
    Generate embeddings for multiple texts (more efficient).
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of EmbeddingResults
    """
    import time
    
    model = get_model()
    start = time.time()
    embeddings = model.encode(texts, convert_to_numpy=True)
    elapsed = (time.time() - start) * 1000
    per_text = elapsed / len(texts)
    
    return [
        EmbeddingResult(
            embedding=emb.tolist(),
            dimension=len(emb),
            model="all-MiniLM-L6-v2",
            inference_ms=round(per_text, 2)
        )
        for emb in embeddings
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


def get_model_info() -> dict:
    """Get information about the loaded model."""
    model = get_model()
    return {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "max_sequence_length": model.max_seq_length if hasattr(model, 'max_seq_length') else 256,
        "loaded": _model is not None
    }
