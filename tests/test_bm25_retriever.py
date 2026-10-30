"""
Unit tests for BM25Retriever
"""
import unittest
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bm25_retriever import BM25Retriever


class TestBM25Retriever(unittest.TestCase):
    
    # Class-level variables to accumulate all test results
    _test_results = []
    _component_name = "bm25_retriever"
    _tests_run = 0
    _total_tests = 5  # Total number of test methods in this class
    
    def setUp(self):
        """Set up test data and results logging."""
        # Don't reset test_results here - we want to accumulate across all tests
        self.component_name = self._component_name
        
        self.sample_corpus = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing helps computers understand text",
            "Computer vision enables machines to interpret visual information",
            "Reinforcement learning teaches agents through rewards"
        ]
        
        self.retriever = BM25Retriever()
    
    def log_test_result(self, test_name, passed, error_message=None):
        """Log individual test result."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        # Add to class-level results list
        TestBM25Retriever._test_results.append(result)
    
    def tearDown(self):
        """Save test results to file only after the last test."""
        # Increment the test counter
        TestBM25Retriever._tests_run += 1
        
        # Only save results when we've run all tests
        if (TestBM25Retriever._tests_run >= TestBM25Retriever._total_tests and 
            TestBM25Retriever._test_results):
            
            results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.component_name}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            summary = {
                "component": self.component_name,
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(TestBM25Retriever._test_results),
                "passed_tests": sum(1 for r in TestBM25Retriever._test_results if r["passed"]),
                "failed_tests": sum(1 for r in TestBM25Retriever._test_results if not r["passed"]),
                "test_details": TestBM25Retriever._test_results
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Reset counters for next test run
            TestBM25Retriever._test_results = []
            TestBM25Retriever._tests_run = 0
    
    def test_initialization(self):
        """Test that BM25Retriever initializes correctly."""
        try:
            self.assertIsInstance(self.retriever, BM25Retriever)
            self.assertIsNone(self.retriever.bm25)
            self.assertIsNone(self.retriever.corpus)
            self.log_test_result("test_initialization", True)
        except Exception as e:
            self.log_test_result("test_initialization", False, str(e))
            raise
    
    def test_build_index_implementation(self):
        """Test tokenization and BM25 index creation."""
        try:
            self.retriever.build_index(self.sample_corpus)
            
            # Check tokenization and BM25 creation
            self.assertIsNotNone(self.retriever.bm25, 
                               "BM25 index is None - did you implement the BM25Okapi creation?")
            self.assertIsNotNone(self.retriever.corpus,
                               "Corpus is None - did you store the corpus_texts?")
            
            # Verify index was built correctly
            self.assertEqual(len(self.retriever.corpus), len(self.sample_corpus))
            self.assertEqual(self.retriever.corpus, self.sample_corpus)
            self.log_test_result("test_build_index_implementation", True)
        except Exception as e:
            self.log_test_result("test_build_index_implementation", False, str(e))
            raise
    
    def test_retrieve_implementation(self):
        """Test the retrieval method correctly."""
        try:
            self.retriever.build_index(self.sample_corpus)
            
            # Test retrieval with a query
            queries = ["machine learning artificial intelligence"]
            results = self.retriever.retrieve(queries, k=3)
            
            # Check retrieval code
            self.assertIsNotNone(results, 
                               "Results is None - did you implement the retrieve method?")
            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), 1)
            self.assertIn(0, results)
            
            # Check that actual document indices are returned
            retrieved_docs = results[0]
            self.assertIsNotNone(retrieved_docs,
                               "Retrieved documents is None - did you implement top_k_indices calculation?")
            self.assertIsInstance(retrieved_docs, list)
            self.assertLessEqual(len(retrieved_docs), 3)
            
            # Verify valid document indices
            for idx in retrieved_docs:
                self.assertIsInstance(idx, int)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(self.sample_corpus))
            
            self.log_test_result("test_retrieve_implementation", True)
        except Exception as e:
            self.log_test_result("test_retrieve_implementation", False, str(e))
            raise
    
    def test_multiple_queries(self):
        """Test retrieval with multiple queries."""
        try:
            self.retriever.build_index(self.sample_corpus)
            
            queries = ["machine learning", "neural networks", "computer vision"]
            results = self.retriever.retrieve(queries, k=2)
            
            # Verify results for all queries
            self.assertEqual(len(results), len(queries))
            
            for i in range(len(queries)):
                self.assertIn(i, results)
                self.assertIsInstance(results[i], list)
                self.assertLessEqual(len(results[i]), 2)
                # Ensure actual indices returned, not None
                for idx in results[i]:
                    self.assertIsNotNone(idx, f"Query {i} returned None index - check retrieve implementation")
            
            self.log_test_result("test_multiple_queries", True)
        except Exception as e:
            self.log_test_result("test_multiple_queries", False, str(e))
            raise
    
    def test_k_parameter_functionality(self):
        """Test that k parameter controls number of results returned."""
        try:
            self.retriever.build_index(self.sample_corpus)
            
            queries = ["learning"]
            
            # Test different k values
            for k in [1, 2, 3]:
                results = self.retriever.retrieve(queries, k=k)
                retrieved = results[0]
                expected_k = min(k, len(self.sample_corpus))
                self.assertEqual(len(retrieved), expected_k,
                               f"Expected {expected_k} results for k={k}, got {len(retrieved)} - check your top-k selection")
            
            self.log_test_result("test_k_parameter_functionality", True)
        except Exception as e:
            self.log_test_result("test_k_parameter_functionality", False, str(e))
            raise


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)