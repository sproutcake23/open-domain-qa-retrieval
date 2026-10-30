"""
Hybrid retrieval: fuse keyword (BM25) and dense rankings with Reciprocal Rank Fusion.
"""
from typing import List, Dict, Any

from utils import reciprocal_rank_fusion


class HybridRetriever:
    """Combine already-built retrievers by fusing their rankings with RRF."""

    def __init__(self, retrievers: List[Any], depth: int = 20, rrf_k: int = 60):
        """
        Args:
            retrievers: Built retrievers exposing retrieve(queries, k)
            depth: How many results to take from each retriever before fusing
            rrf_k: RRF smoothing constant
        """
        self.retrievers = retrievers
        self.depth = depth
        self.rrf_k = rrf_k

    @staticmethod
    def fuse(results_list: List[Dict[int, List[int]]], k: int = 20, rrf_k: int = 60) -> Dict[int, List[int]]:
        """Fuse precomputed {query_id: [doc_ids]} results from several retrievers."""
        return {
            q_id: reciprocal_rank_fusion([results[q_id] for results in results_list], k=k, rrf_k=rrf_k)
            for q_id in results_list[0]
        }

    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """Retrieve with every component retriever, then fuse the rankings."""
        results_list = [r.retrieve(query_texts, k=max(k, self.depth)) for r in self.retrievers]
        return self.fuse(results_list, k=k, rrf_k=self.rrf_k)
