"""
Unit tests for SMAPE Calculator.

Comprehensive tests covering edge cases, boundary conditions, and validation
against known examples including zeros and extreme values.
"""

import unittest
import numpy as np
import tempfile
import os
import json
from src.evaluation.smape_calculator import SMAPECalculator


class TestSMAPECalculator(unittest.TestCase):
    """Test cases for SMAPE Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = SMAPECalculator(log_performance=False)
    
    def test_perfect_predictions(self):
        """Test SMAPE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertAlmostEqual(smape, 0.0, places=6)
    
    def test_simple_case(self):
        """Test SMAPE with simple known case."""
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([1.0, 3.0])
        
        # Manual calculation:
        # |1-2|/((1+2)/2) + |3-4|/((3+4)/2) = 1/1.5 + 1/3.5 = 2/3 + 2/7 ≈ 0.952
        # SMAPE = 100 * 0.952 / 2 ≈ 47.6%
        expected_smape = 100.0 * (1/1.5 + 1/3.5) / 2
        
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertAlmostEqual(smape, expected_smape, places=4)
    
    def test_both_zeros(self):
        """Test SMAPE when both actual and predicted are zero."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertAlmostEqual(smape, 0.0, places=6)
    
    def test_zero_actual_nonzero_predicted(self):
        """Test SMAPE with zero actual and non-zero predicted."""
        y_true = np.array([0.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        
        # |1-0|/((1+0)/2) + |2-2|/((2+2)/2) = 1/0.5 + 0/2 = 2 + 0 = 2
        # SMAPE = 100 * 2 / 2 = 100%
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertAlmostEqual(smape, 100.0, places=4)
    
    def test_zero_predicted_nonzero_actual(self):
        """Test SMAPE with zero predicted and non-zero actual."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([0.0, 2.0])
        
        # |0-1|/((0+1)/2) + |2-2|/((2+2)/2) = 1/0.5 + 0/2 = 2 + 0 = 2
        # SMAPE = 100 * 2 / 2 = 100%
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertAlmostEqual(smape, 100.0, places=4)
    
    def test_single_value(self):
        """Test SMAPE with single value."""
        y_true = np.array([5.0])
        y_pred = np.array([4.0])
        
        # |4-5|/((4+5)/2) = 1/4.5 ≈ 0.2222
        # SMAPE = 100 * 0.2222 ≈ 22.22%
        expected_smape = 100.0 * (1.0 / 4.5)
        
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertAlmostEqual(smape, expected_smape, places=4)
    
    def test_negative_values(self):
        """Test SMAPE with negative values."""
        y_true = np.array([-2.0, -4.0])
        y_pred = np.array([-1.0, -3.0])
        
        # |-1-(-2)|/((|-1|+|-2|)/2) + |-3-(-4)|/((|-3|+|-4|)/2)
        # = 1/1.5 + 1/3.5 = 2/3 + 2/7 ≈ 0.952
        # SMAPE = 100 * 0.952 / 2 ≈ 47.6%
        expected_smape = 100.0 * (1/1.5 + 1/3.5) / 2
        
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertAlmostEqual(smape, expected_smape, places=4)
    
    def test_large_values(self):
        """Test SMAPE with large values."""
        y_true = np.array([1000.0, 2000.0])
        y_pred = np.array([900.0, 1800.0])
        
        # |900-1000|/950 + |1800-2000|/1900 = 100/950 + 200/1900
        expected_smape = 100.0 * (100/950 + 200/1900) / 2
        
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertAlmostEqual(smape, expected_smape, places=4)
    
    def test_empty_arrays(self):
        """Test SMAPE with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertEqual(smape, 0.0)
    
    def test_input_validation_none(self):
        """Test input validation with None values."""
        with self.assertRaises(ValueError):
            self.calculator.calculate_smape(None, np.array([1, 2, 3]))
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_smape(np.array([1, 2, 3]), None)
    
    def test_input_validation_shape_mismatch(self):
        """Test input validation with shape mismatch."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_smape(y_true, y_pred)
    
    def test_input_validation_multidimensional(self):
        """Test input validation with multidimensional arrays."""
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 2], [3, 4]])
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_smape(y_true, y_pred)
    
    def test_input_validation_infinite_values(self):
        """Test input validation with infinite values."""
        y_true = np.array([1.0, np.inf, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_smape(y_true, y_pred)
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.inf, 3.0])
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_smape(y_true, y_pred)
    
    def test_input_validation_nan_values(self):
        """Test input validation with NaN values."""
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_smape(y_true, y_pred)
    
    def test_calculate_smape_with_details(self):
        """Test detailed SMAPE calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.8, 3.2])
        
        details = self.calculator.calculate_smape_with_details(y_true, y_pred)
        
        # Check that all expected keys are present
        expected_keys = ['smape', 'n_samples', 'mean_error', 'std_error', 
                        'median_error', 'min_error', 'max_error', 
                        'zero_denominator_count', 'per_sample_errors']
        
        for key in expected_keys:
            self.assertIn(key, details)
        
        # Check basic properties
        self.assertEqual(details['n_samples'], 3)
        self.assertEqual(len(details['per_sample_errors']), 3)
        self.assertGreaterEqual(details['min_error'], 0)
        self.assertLessEqual(details['max_error'], 200)  # SMAPE max is 200%
    
    def test_calculate_quantile_smape(self):
        """Test quantile-based SMAPE calculation."""
        # Create data with different price ranges
        y_true = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        y_pred = np.array([1.1, 1.9, 5.2, 9.8, 19.5, 52.0, 98.0])
        
        quantile_smapes = self.calculator.calculate_quantile_smape(y_true, y_pred, n_quantiles=3)
        
        # Should have 3 quantiles
        self.assertEqual(len(quantile_smapes), 3)
        
        # All SMAPE values should be non-negative
        for smape in quantile_smapes.values():
            self.assertGreaterEqual(smape, 0)
    
    def test_performance_logging(self):
        """Test performance logging functionality."""
        calculator_with_logging = SMAPECalculator(log_performance=True)
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        
        # Calculate SMAPE to trigger logging
        calculator_with_logging.calculate_smape(y_true, y_pred)
        
        # Check that performance was logged
        self.assertEqual(len(calculator_with_logging.performance_log), 1)
        
        log_entry = calculator_with_logging.performance_log[0]
        self.assertIn('timestamp', log_entry)
        self.assertIn('n_samples', log_entry)
        self.assertIn('smape', log_entry)
        self.assertIn('y_true_stats', log_entry)
        self.assertIn('y_pred_stats', log_entry)
    
    def test_save_performance_log(self):
        """Test saving performance log to file."""
        calculator_with_logging = SMAPECalculator(log_performance=True)
        
        # Generate some performance data
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        calculator_with_logging.calculate_smape(y_true, y_pred)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            calculator_with_logging.save_performance_log(temp_path)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                loaded_log = json.load(f)
            
            self.assertEqual(len(loaded_log), 1)
            self.assertIn('timestamp', loaded_log[0])
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_run_validation_tests(self):
        """Test the built-in validation test suite."""
        # This should pass all validation tests
        result = self.calculator.run_validation_tests()
        self.assertTrue(result)
    
    def test_edge_case_very_small_values(self):
        """Test SMAPE with very small values."""
        y_true = np.array([1e-10, 2e-10])
        y_pred = np.array([1.1e-10, 1.9e-10])
        
        # Should not raise errors and should return reasonable result
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertGreaterEqual(smape, 0)
        self.assertLessEqual(smape, 200)
    
    def test_edge_case_mixed_zeros(self):
        """Test SMAPE with mixed zero and non-zero values."""
        y_true = np.array([0.0, 1.0, 0.0, 2.0])
        y_pred = np.array([1.0, 0.0, 0.0, 2.0])
        
        smape = self.calculator.calculate_smape(y_true, y_pred)
        
        # Should handle mixed zeros gracefully
        self.assertGreaterEqual(smape, 0)
        self.assertLessEqual(smape, 200)
    
    def test_list_input_conversion(self):
        """Test that list inputs are properly converted to numpy arrays."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 1.9, 3.1]
        
        # Should work with list inputs
        smape = self.calculator.calculate_smape(y_true, y_pred)
        self.assertGreaterEqual(smape, 0)
        self.assertLessEqual(smape, 200)


if __name__ == '__main__':
    unittest.main()