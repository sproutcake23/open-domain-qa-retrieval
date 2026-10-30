"""
Word2Vec-based Retrieval Module for Semantic Search
"""
import numpy as np
import warnings
import os
import re
import logging
from typing import List, Optional, Dict

# Suppress noisy gensim warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gensim")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils import top_k_indices

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Word2VecRetriever:
    """
    Word2Vec-based retrieval using static word embeddings.
    Represents dense retrieval from the pre-transformer era.
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1,
                 sg: int = 0, epochs: int = 10):
        """
        Initialize Word2Vec retriever.

        Args:
            vector_size (int): Dimensionality of word vectors
            window (int): Context window size
            min_count (int): Minimum word frequency threshold
            sg (int): Training algorithm (0=CBOW, 1=Skip-gram)
            epochs (int): Number of training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.model = None
        self.corpus_vectors = None
        self.corpus_texts = None

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for Word2Vec training.

        Args:
            text (str): Input text

        Returns:
            List[str]: Tokenized and cleaned words
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        words = text.split()

        # Remove stop words and short words
        words = [word for word in words
                 if word not in ENGLISH_STOP_WORDS and len(word) > 2]

        return words

    def _text_to_vector(self, text: str) -> Optional[np.ndarray]:
        """
        Convert text to vector by averaging word vectors.

        Args:
            text (str): Input text

        Returns:
            np.ndarray: Document vector (or None if no words found)
        """
        words = self._preprocess_text(text)
        word_vectors = []

        # Collect vectors for in-vocabulary words
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])

        if word_vectors:
            # Document vector = mean of its word vectors
            return np.mean(word_vectors, axis=0)
        else:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.vector_size)

    def build_index(self, corpus_texts: List[str]) -> None:
        """
        Build Word2Vec model and corpus index.

        Args:
            corpus_texts (List[str]): List of documents to index
        """
        logger.info("Building Word2Vec model...")

        processed_docs = [self._preprocess_text(doc) for doc in corpus_texts]

        self.model = Word2Vec(
            sentences=processed_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            workers=os.cpu_count() or 1
        )

        logger.info(f"Word2Vec model trained with vocabulary")

        # Embed every document and L2-normalize so dot product == cosine similarity
        self.corpus_vectors = self._normalize(
            np.array([self._text_to_vector(doc) for doc in corpus_texts], dtype=np.float32)
        )
        self.corpus_texts = corpus_texts

        logger.info(f"Indexed {len(corpus_texts)} documents")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows, leaving all-zero rows (no known words) as zeros."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def retrieve(self, queries: List[str], k: int = 10, query_batch_size: int = 256) -> Dict[int, List[int]]:
        """
        Retrieve top-k documents for each query using Word2Vec similarity.

        Args:
            queries (List[str]): List of query strings
            k (int): Number of documents to retrieve per query

        Returns:
            Dict[int, List[int]]: Dictionary mapping query indices to top-k document indices
        """
        if self.model is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_vectors = self._normalize(
            np.array([self._text_to_vector(q) for q in queries], dtype=np.float32)
        )

        # Score queries in batches to bound memory on large corpora
        results = {}
        for start in range(0, len(query_vectors), query_batch_size):
            batch_scores = query_vectors[start:start + query_batch_size] @ self.corpus_vectors.T
            for offset, top in enumerate(top_k_indices(batch_scores, k)):
                results[start + offset] = top.tolist()

        return results

    def get_vocabulary_stats(self) -> dict:
        """
        Get statistics about the Word2Vec vocabulary.

        Returns:
            dict: Vocabulary statistics
        """
        if self.model is None:
            return {"error": "Model not trained"}

        return {
            "vocabulary_size": len(self.model.wv),
            "vector_size": self.vector_size,
            "total_words_trained": self.model.corpus_count,
            "most_similar_to_search": self.model.wv.most_similar("search",
                                                                 topn=5) if "search" in self.model.wv else None
        }

    def __del__(self):
        """Clean up Word2Vec model to prevent exception warnings on deletion"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Properly clean up the model
                del self.model.wv
                del self.model
        except:
            pass  # Silently ignore any cleanup errors

    @staticmethod
    def optimize_parameters(param_combinations, corpus_texts, query_texts, qrels_dict, evaluator_class):
        """
        Optimize Word2Vec parameters using grid search.

        Args:
            param_combinations (List[dict]): List of parameter combinations to test
            corpus_texts (List[str]): Corpus documents for training
            query_texts (List[str]): Query texts for evaluation
            qrels_dict (dict): Ground truth relevance judgments
            evaluator_class: IRMetrics class for evaluation

        Returns:
            dict: Results containing best configuration, metrics, optimization log, and the
                trained best retriever with its retrieval results (so it needn't be retrained)
        """
        from contextlib import redirect_stderr
        from io import StringIO

        print(f"🔧 Starting Word2Vec Parameter Optimization")
        print(f"📋 Testing {len(param_combinations)} configurations...")

        best_config = None
        best_score = -1.0
        best_metrics = None
        best_retriever = None
        best_results = None
        results_log = []

        for i, params in enumerate(param_combinations, 1):
            print(f"\n📋 Configuration {i}/{len(param_combinations)}")
            print(f"   Parameters: {params}")

            try:
                # Suppress stderr during model creation/deletion to hide C-level exceptions
                stderr_suppressor = StringIO()

                with redirect_stderr(stderr_suppressor):
                    test_retriever = Word2VecRetriever(**params)
                    test_retriever.build_index(corpus_texts)

                    test_results = test_retriever.retrieve(query_texts, k=20)

                    test_metrics = evaluator_class.evaluate_retrieval(test_results, qrels_dict)

                    mrr_score = test_metrics["MRR"]
                    recall_5 = test_metrics["Recall@5"]

                    results_log.append({
                        "config": params,
                        "mrr": mrr_score,
                        "recall_5": recall_5,
                        "metrics": test_metrics
                    })

                print(f"   📊 MRR: {mrr_score:.4f} | Recall@5: {recall_5:.4f}")

                # Keep the configuration (and trained model) with the highest MRR
                if mrr_score > best_score:
                    best_score = mrr_score
                    best_config = params
                    best_metrics = test_metrics
                    best_retriever, best_results = test_retriever, test_results
                    print(f"   ✨ NEW BEST! (MRR improved: {best_score:.4f})")

                # Release models that are no longer needed (inside suppression to hide gensim noise)
                with redirect_stderr(StringIO()):
                    if test_retriever is not best_retriever:
                        del test_retriever

            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:100]}...")
                continue

        print(f"\n🏆 OPTIMIZATION COMPLETE")
        print(f"=" * 50)

        return {
            "best_config": best_config,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "best_retriever": best_retriever,
            "best_results": best_results,
            "results_log": results_log
        }


if __name__ == "__main__":
    print("=" * 60)
    print("🔤 Word2Vec Retriever Test")
    print("=" * 60)

    # Test with sample documents
    sample_docs = [
        "The cat sat on the mat with a hat",
        "Dogs are loyal pets and good friends",
        "Machine learning uses algorithms and data",
        "Natural language processing works with text",
        "Information retrieval finds relevant documents"
    ]

    sample_queries = [
        "pets and animals",
        "computer science and algorithms"
    ]

    try:
        # Initialize and test
        retriever = Word2VecRetriever(vector_size=50, epochs=20)
        retriever.build_index(sample_docs)

        # Show vocabulary stats
        stats = retriever.get_vocabulary_stats()
        print(f"📊 Vocabulary: {stats['vocabulary_size']} words")

        # Test retrieval
        results = retriever.retrieve(sample_queries, k=3)

        for i, query in enumerate(sample_queries):
            print(f"\n🔍 Query: '{query}'")
            print("Top results:")
            for j, doc_idx in enumerate(results[i][:3], 1):
                print(f"  {j}. {sample_docs[doc_idx]}")

        print("\n✅ Word2Vec retriever test successful!")

    except Exception as e:
        print(f"❌ Error: {e}")