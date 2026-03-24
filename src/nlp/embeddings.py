"""
src/nlp/embeddings.py
=====================
Clinical text embedding utilities for semantic search and similarity.

Provides:
  - Sentence-level embeddings using sentence-transformers
  - Clinical text similarity scoring
  - Batch embedding for cohort-level analysis
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from loguru import logger


class ClinicalEmbedder:
    """
    Embeds clinical text using sentence-transformers.

    Uses MedCPT or PubMedBERT for clinical domain when available;
    falls back to all-MiniLM-L6-v2 for general-purpose embedding.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded embedding model: {self._model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using TF-IDF fallback")
            self._model = "tfidf"

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a vector."""
        self._load_model()
        if self._model == "tfidf":
            return self._tfidf_embed([text])[0]
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts, returns (N, dim) array."""
        self._load_model()
        if self._model == "tfidf":
            return self._tfidf_embed(texts)
        return self._model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=len(texts) > 100
        )

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        return float(np.dot(vec_a, vec_b))

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
    ) -> List[dict]:
        """Find most similar candidates to a query."""
        q_vec = self.embed(query)
        c_vecs = self.embed_batch(candidates)
        scores = c_vecs @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": candidates[i], "score": float(scores[i]), "rank": rank + 1}
            for rank, i in enumerate(top_indices)
        ]

    @staticmethod
    def _tfidf_embed(texts: List[str]) -> np.ndarray:
        """Fallback TF-IDF embedding when sentence-transformers unavailable."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        arr = matrix.toarray().astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms
