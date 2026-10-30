"""
Unit tests for Word2VecRetriever
"""
import unittest
import sys
import os
import json
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from word2vec_retriever import Word2VecRetriever


class TestWord2VecRetriever(unittest.TestCase):
    
    # Class-level variables to accumulate all test results
    _test_results = []
    _component_name = "word2vec_retriever"
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
        
        # Use small parameters for faster testing
        self.retriever = Word2VecRetriever(vector_size=50, epochs=5, min_count=1)
    
    def log_test_result(self, test_name, passed, error_message=None):
        """Log individual test result."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        # Add to class-level results list
        TestWord2VecRetriever._test_results.append(result)
    
    def tearDown(self):
        """Save test results to file only after the last test."""
        # Increment the test counter
        TestWord2VecRetriever._tests_run += 1
        
        # Only save results when we've run all tests
        if (TestWord2VecRetriever._tests_run >= TestWord2VecRetriever._total_tests and 
            TestWord2VecRetriever._test_results):
            
            results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.component_name}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            summary = {
                "component": self.component_name,
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(TestWord2VecRetriever._test_results),
                "passed_tests": sum(1 for r in TestWord2VecRetriever._test_results if r["passed"]),
                "failed_tests": sum(1 for r in TestWord2VecRetriever._test_results if not r["passed"]),
                "test_details": TestWord2VecRetriever._test_results
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Reset counters for next test run
            TestWord2VecRetriever._test_results = []
            TestWord2VecRetriever._tests_run = 0
    
    def test_initialization(self):
        """Test that Word2VecRetriever initializes correctly."""
        try:
            self.assertIsInstance(self.retriever, Word2VecRetriever)
            self.assertIsNone(self.retriever.model)
            self.assertIsNone(self.retriever.corpus_texts)
            self.assertIsNone(self.retriever.corpus_vectors)
            self.log_test_result("test_initialization", True)
        except Exception as e:
            self.log_test_result("test_initialization", False, str(e))
            raise
    
    def test_build_index_implementation(self):
        """Test Word2Vec training and indexing."""
        try:
            self.retriever.build_index(self.sample_corpus)
            
            # Check Word2Vec training
            self.assertIsNotNone(self.retriever.model, 
                               "Word2Vec model is None - did you implement Word2Vec training?")
            self.assertIsNotNone(self.retriever.corpus_texts,
                               "Corpus texts is None - did you store the corpus_texts?")
            self.assertIsNotNone(self.retriever.corpus_vectors,
                               "Corpus vectors is None - did you convert documents to vectors?")
            
            # Verify shapes and content
            self.assertEqual(len(self.retriever.corpus_texts), len(self.sample_corpus))
            self.assertEqual(self.retriever.corpus_texts, self.sample_corpus)
            self.assertIsInstance(self.retriever.corpus_vectors, np.ndarray)
            self.assertEqual(self.retriever.corpus_vectors.shape[0], len(self.sample_corpus))
            self.assertEqual(self.retriever.corpus_vectors.shape[1], self.retriever.vector_size)
            
            self.log_test_result("test_build_index_implementation", True)
        except Exception as e:
            self.log_test_result("test_build_index_implementation", False, str(e))
            raise
    
    def test_text_to_vector_implementation(self):
        """Test text-to-vector conversion."""
        try:
            self.retriever.build_index(self.sample_corpus)
            
            # Check that model was built first
            self.assertIsNotNone(self.retriever.model,
                               "Cannot test text-to-vector - Word2Vec model not built. Complete build_index first.")
            
            # Test text to vector conversion
            test_text = "machine learning algorithms"
            vector = self.retriever._text_to_vector(test_text)
            
            # Check vector extraction and averaging
            self.assertIsNotNone(vector, 
                               "Text-to-vector returned None - did you implement word vector extraction and averaging?")
            self.assertIsInstance(vector, np.ndarray)
            self.assertEqual(len(vector), self.retriever.vector_size)
            
            # Vector should not be all zeros for known words
            if any(word in self.retriever.model.wv for word in test_text.split()):
                self.assertFalse(np.allclose(vector, 0), 
                               "Vector is all zeros - check your word vector averaging implementation")
            
            self.log_test_result("test_text_to_vector_implementation", True)
        except Exception as e:
            self.log_test_result("test_text_to_vector_implementation", False, str(e))
            raise
    
    def test_retrieve_implementation(self):
        """Test the retrieval method correctly."""
        try:
            self.retriever.build_index(self.sample_corpus)
            
            queries = ["machine learning algorithms"]
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
                               "Retrieved documents is None - did you implement similarity calculation and top-k selection?")
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
    
    def test_multiple_queries_and_k_parameter(self):
        """Test retrieval with multiple queries and k parameter functionality."""
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
                # Ensure actual indices returned
                for idx in results[i]:
                    self.assertIsNotNone(idx, f"Query {i} returned None index - check retrieve implementation")
                    self.assertIsInstance(idx, int)
            
            self.log_test_result("test_multiple_queries_and_k_parameter", True)
        except Exception as e:
            self.log_test_result("test_multiple_queries_and_k_parameter", False, str(e))
            raise


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
