"""
Transformer-based Retrieval using Sentence Embeddings
"""
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from utils import top_k_indices


class TransformerRetriever:
    """Semantic retrieval using transformer-based sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 query_prompt_name: Optional[str] = None,
                 doc_prompt_name: Optional[str] = None,
                 max_seq_length: Optional[int] = 256,
                 batch_size: int = 128,
                 half_precision: bool = False,
                 **st_kwargs):
        """
        Initialize with a sentence transformer model.

        Args:
            model_name: HuggingFace model name for sentence embeddings
            query_prompt_name: Name of the model's built-in prompt applied to queries
                (e.g. "query" for Qwen3-Embedding / EmbeddingGemma)
            doc_prompt_name: Name of the model's built-in prompt applied to documents
                (e.g. "document" for EmbeddingGemma)
            max_seq_length: Truncation length for passages (None keeps model default)
            batch_size: Encoding batch size
            half_precision: Run the model in float16 on GPU (faster, less memory)
            **st_kwargs: Extra keyword arguments passed to SentenceTransformer
                (e.g. prompts={"query": "..."} to define a query instruction)
        """
        print(f"🤖 Loading transformer model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **st_kwargs)
        if half_precision and self.model.device.type == "cuda":
            self.model.half()
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.query_prompt_name = query_prompt_name
        self.doc_prompt_name = doc_prompt_name
        self.batch_size = batch_size
        self.corpus_embeddings = None
        self.corpus = None
        print(f"✅ Model loaded: {model_name}")

    def _encode(self, texts: List[str], prompt_name: Optional[str], show_progress_bar: bool) -> np.ndarray:
        """Encode texts into L2-normalized embeddings (dot product == cosine similarity)."""
        return self.model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar
        ).astype(np.float32)

    def build_index(self, corpus_texts: List[str]):
        """Build semantic index by encoding corpus texts."""
        print("🧠 Building semantic index...")
        self.corpus = corpus_texts
        self.corpus_embeddings = self._encode(corpus_texts, self.doc_prompt_name, show_progress_bar=True)

        print(f"✅ Semantic index built for {len(corpus_texts):,} documents")
        if len(corpus_texts) > 0 and self.corpus_embeddings is not None:
            print(f"   Embedding dimension: {self.corpus_embeddings.shape[1]}")

    def retrieve(self, query_texts: List[str], k: int = 20, query_batch_size: int = 256) -> Dict[int, List[int]]:
        """Retrieve top-k documents using semantic similarity."""
        if self.corpus_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"🔍 Running semantic retrieval for {len(query_texts)} queries...")
        query_embeddings = self._encode(query_texts, self.query_prompt_name,
                                        show_progress_bar=len(query_texts) > 1)

        # Score queries in batches so the full (queries x corpus) matrix never sits in memory
        k = min(k, len(self.corpus))
        results = {}
        for start in range(0, len(query_embeddings), query_batch_size):
            batch_scores = query_embeddings[start:start + query_batch_size] @ self.corpus_embeddings.T
            for offset, top in enumerate(top_k_indices(batch_scores, k)):
                results[start + offset] = top.tolist()

        print(f"✅ Retrieved top-{k} documents using semantic similarity")
        return results
