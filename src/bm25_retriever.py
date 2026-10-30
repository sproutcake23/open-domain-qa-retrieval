"""
BM25 Retrieval Implementation for Baseline Comparison
"""
from typing import List, Dict
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from utils import top_k_indices


class BM25Retriever:
    """Traditional keyword-based retrieval using BM25 algorithm."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 retriever with tunable parameters.

        Args:
            k1 (float): Controls term frequency saturation (default: 1.2)
            b (float): Controls document length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus = None

    def build_index(self, corpus_texts: List[str]):
        """Build BM25 index from corpus texts."""
        print("🔧 Building BM25 index...")
        self.corpus = corpus_texts

        # Lowercase whitespace tokenization
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        print(f"✅ BM25 index built for {len(corpus_texts):,} documents")

    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """Retrieve top-k documents for each query."""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"🔍 Running BM25 retrieval for {len(query_texts)} queries...")
        results = {}

        for q_idx, query in enumerate(tqdm(query_texts, desc="Retrieving")):
            # Score every document, then keep the k best
            doc_scores = self.bm25.get_scores(query.lower().split())
            results[q_idx] = top_k_indices(doc_scores, k)[0].tolist()

        print(f"✅ Retrieved top-{k} documents for {len(query_texts)} queries")
        return results
