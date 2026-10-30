"""
Unit tests for IRMetrics Evaluator
"""
import unittest
import sys
import os
import json
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluator import IRMetrics


class TestIRMetrics(unittest.TestCase):
    
    # Class-level variables to accumulate all test results
    _test_results = []
    _component_name = "evaluator"
    _tests_run = 0
    _total_tests = 5  # Total number of test methods in this class
    
    def setUp(self):
        """Set up test data and results logging."""
        # Don't reset test_results here - we want to accumulate across all tests
        self.component_name = self._component_name
        
        # Sample retrieval results: {query_id: [doc_ids]}
        self.results = {
            0: [1, 3, 5, 7, 9],   # Query 0 retrieved docs 1,3,5,7,9
            1: [2, 4, 1, 8, 6],   # Query 1 retrieved docs 2,4,1,8,6
            2: [0, 2, 4, 6, 8]    # Query 2 retrieved docs 0,2,4,6,8
        }
        
        # Sample relevance judgments: {query_id: {doc_id: relevance}}
        self.qrels = {
            0: {1: 1, 5: 1, 9: 1},      # Query 0: docs 1,5,9 are relevant
            1: {2: 1, 4: 1, 6: 1},      # Query 1: docs 2,4,6 are relevant  
            2: {0: 1, 2: 1, 8: 1}       # Query 2: docs 0,2,8 are relevant
        }
    
    def log_test_result(self, test_name, passed, error_message=None):
        """Log individual test result."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        # Add to class-level results list
        TestIRMetrics._test_results.append(result)
    
    def tearDown(self):
        """Save test results to file only after the last test."""
        # Increment the test counter
        TestIRMetrics._tests_run += 1
        
        # Only save results when we've run all tests
        if (TestIRMetrics._tests_run >= TestIRMetrics._total_tests and 
            TestIRMetrics._test_results):
            
            results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.component_name}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            summary = {
                "component": self.component_name,
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(TestIRMetrics._test_results),
                "passed_tests": sum(1 for r in TestIRMetrics._test_results if r["passed"]),
                "failed_tests": sum(1 for r in TestIRMetrics._test_results if not r["passed"]),
                "test_details": TestIRMetrics._test_results
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Reset counters for next test run
            TestIRMetrics._test_results = []
            TestIRMetrics._tests_run = 0
    
    def test_recall_at_k_implementation(self):
        """Test Recall@k correctly."""
        try:
            # Test Recall@3
            recall_3 = IRMetrics.recall_at_k(self.results, self.qrels, 3)
            
            # Query 0: found docs 1,5 out of 1,5,9 -> 2/3 = 0.6667
            # Query 1: found docs 2,4 out of 2,4,6 -> 2/3 = 0.6667  
            # Query 2: found docs 0,2 out of 0,2,8 -> 2/3 = 0.6667
            expected_recall_3 = (2/3 + 2/3 + 2/3) / 3
            
            self.assertIsNotNone(recall_3, "Recall@k returned None - did you implement the calculation?")
            self.assertIsInstance(recall_3, (int, float), "Recall@k should return a number")
            self.assertAlmostEqual(recall_3, expected_recall_3, places=3,
                                  msg=f"Expected recall@3={expected_recall_3:.4f}, got {recall_3:.4f}")
            
            self.log_test_result("test_recall_at_k_implementation", True)
        except Exception as e:
            self.log_test_result("test_recall_at_k_implementation", False, str(e))
            raise
    
    def test_precision_at_k_implementation(self):
        """Test Precision@k correctly."""
        try:
            # Test Precision@3
            precision_3 = IRMetrics.precision_at_k(self.results, self.qrels, 3)
            
            # Query 0: 2 relevant out of top 3 -> 2/3 = 0.6667
            # Query 1: 2 relevant out of top 3 -> 2/3 = 0.6667
            # Query 2: 2 relevant out of top 3 -> 2/3 = 0.6667
            expected_precision_3 = 2/3
            
            self.assertIsNotNone(precision_3, "Precision@k returned None - did you implement the calculation?")
            self.assertIsInstance(precision_3, (int, float), "Precision@k should return a number")
            self.assertAlmostEqual(precision_3, expected_precision_3, places=3,
                                  msg=f"Expected precision@3={expected_precision_3:.4f}, got {precision_3:.4f}")
            
            self.log_test_result("test_precision_at_k_implementation", True)
        except Exception as e:
            self.log_test_result("test_precision_at_k_implementation", False, str(e))
            raise
    
    def test_mrr_implementation(self):
        """Test MRR correctly."""
        try:
            mrr_score = IRMetrics.mrr(self.results, self.qrels)
            
            # Query 0: first relevant doc (1) at rank 1 -> RR = 1/1 = 1.0
            # Query 1: first relevant doc (2) at rank 1 -> RR = 1/1 = 1.0
            # Query 2: first relevant doc (0) at rank 1 -> RR = 1/1 = 1.0
            expected_mrr = (1.0 + 1.0 + 1.0) / 3
            
            self.assertIsNotNone(mrr_score, "MRR returned None - did you implement the calculation?")
            self.assertIsInstance(mrr_score, (int, float), "MRR should return a number")
            self.assertAlmostEqual(mrr_score, expected_mrr, places=3,
                                  msg=f"Expected MRR={expected_mrr:.4f}, got {mrr_score:.4f}")
            
            self.log_test_result("test_mrr_implementation", True)
        except Exception as e:
            self.log_test_result("test_mrr_implementation", False, str(e))
            raise
    
    def test_evaluation_with_different_scenarios(self):
        """Test metrics with different ranking scenarios."""
        try:
            # Test scenario where relevant docs appear at different ranks
            results_mixed = {
                0: [0, 1, 3, 5, 9],   # Relevant docs 1,5,9 at ranks 2,4,5
                1: [3, 2, 4, 6, 7],   # Relevant docs 2,4,6 at ranks 2,3,4
            }
            qrels_mixed = {
                0: {1: 1, 5: 1, 9: 1},
                1: {2: 1, 4: 1, 6: 1}
            }
            
            recall_5 = IRMetrics.recall_at_k(results_mixed, qrels_mixed, 5)
            precision_5 = IRMetrics.precision_at_k(results_mixed, qrels_mixed, 5)
            mrr_score = IRMetrics.mrr(results_mixed, qrels_mixed)
            
            # Both queries find all 3 relevant docs in top 5
            self.assertEqual(recall_5, 1.0, "Should find all relevant docs in top 5")
            
            # Each query has 3/5 = 0.6 precision
            self.assertAlmostEqual(precision_5, 0.6, places=3, msg="Expected precision@5=0.6")
            
            # Query 0: first relevant at rank 2 (1/2=0.5), Query 1: first relevant at rank 2 (1/2=0.5)
            expected_mrr = (0.5 + 0.5) / 2
            self.assertAlmostEqual(mrr_score, expected_mrr, places=3,
                                  msg=f"Expected MRR={expected_mrr:.3f}")
            
            self.log_test_result("test_evaluation_with_different_scenarios", True)
        except Exception as e:
            self.log_test_result("test_evaluation_with_different_scenarios", False, str(e))
            raise
    
    def test_complete_evaluation_pipeline(self):
        """Test the complete evaluation workflow."""
        try:
            metrics = IRMetrics.evaluate_retrieval(self.results, self.qrels)
            
            # Verify all expected metrics are computed
            expected_metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Precision@5', 'MRR']
            for metric in expected_metrics:
                self.assertIn(metric, metrics, f"Missing metric: {metric}")
                self.assertIsInstance(metrics[metric], (int, float), f"{metric} should be numeric")
                self.assertFalse(np.isnan(metrics[metric]), f"{metric} should not be NaN")
            
            # Test print function works
            try:
                IRMetrics.print_metrics(metrics, "Test Evaluation")
                print_success = True
            except Exception as e:
                print_success = False
                print(f"print_metrics failed: {e}")
            
            self.assertTrue(print_success, "print_metrics should execute without errors")
            
            self.log_test_result("test_complete_evaluation_pipeline", True)
        except Exception as e:
            self.log_test_result("test_complete_evaluation_pipeline", False, str(e))
            raise


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
