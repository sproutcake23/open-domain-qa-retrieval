"""
Unit tests for TransformerRetriever
"""
import unittest
import sys
import os
import json
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformer_retriever import TransformerRetriever


class TestTransformerRetriever(unittest.TestCase):

    # Class-level variables to accumulate all test results
    _test_results = []
    _component_name = "transformer_retriever"
    _tests_run = 0
    _total_tests = 5  # Total number of test methods in this class

    def setUp(self):
        """Set up test data and results logging."""
        # Don't reset test_results here - we want to accumulate across all tests
        self.component_name = self._component_name

        self.sample_corpus = [
            "Machine learning algorithms can automatically improve through experience",
            "Deep neural networks are inspired by the structure of the human brain",
            "Natural language processing enables computers to understand human language",
            "Computer vision allows machines to interpret and analyze visual information",
            "Reinforcement learning involves training agents through rewards and penalties"
        ]

        # Use a small, fast model for testing
        self.retriever = TransformerRetriever(model_name="all-MiniLM-L6-v2")

    def log_test_result(self, test_name, passed, error_message=None):
        """Log individual test result."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        # Add to class-level results list
        TestTransformerRetriever._test_results.append(result)

    def tearDown(self):
        """Save test results to file only after the last test."""
        # Increment the test counter
        TestTransformerRetriever._tests_run += 1

        # Only save results when we've run all tests
        if (TestTransformerRetriever._tests_run >= TestTransformerRetriever._total_tests and
            TestTransformerRetriever._test_results):

            results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
            os.makedirs(results_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.component_name}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)

            summary = {
                "component": self.component_name,
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(TestTransformerRetriever._test_results),
                "passed_tests": sum(1 for r in TestTransformerRetriever._test_results if r["passed"]),
                "failed_tests": sum(1 for r in TestTransformerRetriever._test_results if not r["passed"]),
                "test_details": TestTransformerRetriever._test_results
            }

            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)

            # Reset counters for next test run
            TestTransformerRetriever._test_results = []
            TestTransformerRetriever._tests_run = 0

    def test_initialization(self):
        """Test that TransformerRetriever initializes correctly."""
        try:
            self.assertIsInstance(self.retriever, TransformerRetriever)
            self.assertIsNotNone(self.retriever.model)
            self.assertIsNone(self.retriever.corpus_embeddings)
            self.assertIsNone(self.retriever.corpus)

            self.log_test_result("test_initialization", True)
        except Exception as e:
            self.log_test_result("test_initialization", False, str(e))
            raise

    def test_build_index_implementation(self):
        """Test that build_index creates semantic embeddings."""
        try:
            # Before building index
            self.assertIsNone(self.retriever.corpus_embeddings)

            # Build index
            self.retriever.build_index(self.sample_corpus)

            # Check that embeddings were created
            self.assertIsNotNone(self.retriever.corpus_embeddings,
                               "Corpus embeddings is None - did you implement corpus encoding in build_index()?")
            self.assertEqual(len(self.retriever.corpus), len(self.sample_corpus))
            self.assertEqual(self.retriever.corpus, self.sample_corpus)

            # Check embedding format and dimensions
            self.assertIsInstance(self.retriever.corpus_embeddings, np.ndarray,
                                "Embeddings should be numpy array")
            self.assertEqual(self.retriever.corpus_embeddings.shape[0], len(self.sample_corpus),
                           "Number of embeddings should match corpus size")
            self.assertGreater(self.retriever.corpus_embeddings.shape[1], 0,
                             "Embedding dimension should be > 0")

            self.log_test_result("test_build_index_implementation", True)
        except Exception as e:
            self.log_test_result("test_build_index_implementation", False, str(e))
            raise

    def test_retrieve_implementation(self):
        """Test that retrieve returns semantic search results."""
        try:
            self.retriever.build_index(self.sample_corpus)

            # Test query retrieval
            queries = ["artificial intelligence and machine learning"]
            results = self.retriever.retrieve(queries, k=3)

            # Check that results were returned
            self.assertIsInstance(results, dict, "Results should be a dictionary")
            self.assertEqual(len(results), 1, "Should return results for one query")
            self.assertIn(0, results, "Results should contain query index 0")

            # Check result format
            retrieved_docs = results[0]
            self.assertIsInstance(retrieved_docs, list, "Retrieved docs should be a list")
            self.assertGreater(len(retrieved_docs), 0, "Should return at least one document")
            self.assertLessEqual(len(retrieved_docs), 3, "Should respect k=3 limit")

            # All indices should be valid
            for idx in retrieved_docs:
                self.assertIsInstance(idx, int, "Document indices should be integers")
                self.assertGreaterEqual(idx, 0, "Document index should be >= 0")
                self.assertLess(idx, len(self.sample_corpus), "Document index should be < corpus size")

            self.log_test_result("test_retrieve_implementation", True)
        except Exception as e:
            self.log_test_result("test_retrieve_implementation", False, str(e))
            raise

    def test_multiple_queries_and_k_values(self):
        """Test retrieval with multiple queries and different k values."""
        try:
            self.retriever.build_index(self.sample_corpus)

            # Test multiple queries
            queries = [
                "machine learning algorithms",
                "neural networks and deep learning",
                "computer vision analysis"
            ]
            results = self.retriever.retrieve(queries, k=2)

            # Verify results for all queries
            self.assertEqual(len(results), len(queries),
                           "Should return results for all queries")

            for i in range(len(queries)):
                self.assertIn(i, results, f"Missing results for query {i}")
                self.assertIsInstance(results[i], list, f"Results for query {i} should be list")
                self.assertEqual(len(results[i]), 2, f"Should return exactly k=2 results for query {i}")

            self.log_test_result("test_multiple_queries_and_k_values", True)
        except Exception as e:
            self.log_test_result("test_multiple_queries_and_k_values", False, str(e))
            raise

    def test_semantic_similarity_quality(self):
        """Test that transformer captures semantic similarity correctly."""
        try:
            # Create corpus with clear semantic relationships
            corpus = [
                "Dogs are loyal pets that bark and play fetch",
                "Cats are independent animals that meow and climb",
                "Puppies are young dogs that need training",
                "Cars are vehicles with engines and wheels"
            ]

            self.retriever.build_index(corpus)

            # Query about dogs should find dog-related documents
            queries = ["canine puppies and pet dogs"]
            results = self.retriever.retrieve(queries, k=4)

            retrieved_indices = results[0]

            # Check that semantic understanding works
            # Documents 0 and 2 are about dogs, should be ranked higher than car document (3)
            dog_docs = {0, 2}
            top_2_results = set(retrieved_indices[:2])

            # At least one dog document should be in top 2 results
            semantic_match = len(dog_docs & top_2_results) >= 1
            self.assertTrue(semantic_match,
                           f"Expected dog-related docs {dog_docs} in top 2 results {top_2_results}. "
                           f"Full ranking: {retrieved_indices}")

            self.log_test_result("test_semantic_similarity_quality", True)
        except Exception as e:
            self.log_test_result("test_semantic_similarity_quality", False, str(e))
            raise


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
