"""
Evaluation Metrics for Information Retrieval
"""
import numpy as np
from typing import Dict, List


class IRMetrics:
    """Information Retrieval evaluation metrics."""
    
    @staticmethod
    def recall_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Recall@k: fraction of relevant documents found in top-k.
        
        Args:
            results: {query_id: [doc_ids]}
            qrels: {query_id: {doc_id: relevance_score}}
            k: cutoff for top-k evaluation
        """
        recall_scores = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            relevant_docs = {
                doc_id
                for doc_id, relevance in qrels[q_id].items()
                if relevance > 0
            }

            if not relevant_docs:
                continue

            retrieved_docs = set(results[q_id][:k])

            relevant_retrieved = len(
                retrieved_docs & relevant_docs
            )

            recall = relevant_retrieved / len(relevant_docs)
            recall_scores.append(recall)
            
        return np.mean(recall_scores) if recall_scores else 0.0

    @staticmethod
    def precision_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Precision@k: fraction of top-k that are relevant.
        """
        precision_scores = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            relevant_docs = {
                doc_id
                for doc_id, relevance in qrels[q_id].items()
                if relevance > 0
            }

            retrieved_docs = results[q_id][:k]

            if not retrieved_docs:
                continue

            relevant_retrieved = sum(
                doc_id in relevant_docs
                for doc_id in retrieved_docs
            )

            precision = relevant_retrieved / len(retrieved_docs)
            precision_scores.append(precision)
            
        return np.mean(precision_scores) if precision_scores else 0.0

    @staticmethod
    def mrr(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        """
        reciprocal_ranks = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            relevant_docs = {
                doc_id
                for doc_id, relevance in qrels[q_id].items()
                if relevance > 0
            }

            reciprocal_rank = 0.0

            for rank, doc_id in enumerate(results[q_id], start=1):
                if doc_id in relevant_docs:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_ranks.append(reciprocal_rank)
            
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def evaluate_retrieval(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> Dict[str, float]:
        """
        Comprehensive evaluation with standard IR metrics.
        
        Returns:
            Dictionary with metric names and values
        """
        metrics = {
            'Recall@1': IRMetrics.recall_at_k(results, qrels, 1),
            'Recall@5': IRMetrics.recall_at_k(results, qrels, 5), 
            'Recall@10': IRMetrics.recall_at_k(results, qrels, 10),
            'Precision@5': IRMetrics.precision_at_k(results, qrels, 5),
            'MRR': IRMetrics.mrr(results, qrels)
        }
        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
        """Pretty print evaluation metrics."""
        print(f"\n📊 {title}")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric:12}: {value:.4f}")
        print("=" * 40)
