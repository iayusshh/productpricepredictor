"""
SMAPE Calculator with validation and unit tests.

Implements Symmetric Mean Absolute Percentage Error calculation following
the exact competition formula with comprehensive validation and testing.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class SMAPECalculator:
    """
    Validated SMAPE calculator with unit tests and performance tracking.
    
    Implements SMAPE calculation following the competition formula:
    SMAPE = (100/n) * Σ(|predicted - actual| / ((|predicted| + |actual|) / 2))
    """
    
    def __init__(self, log_performance: bool = True):
        """
        Initialize SMAPE calculator.
        
        Args:
            log_performance: Whether to log performance metrics
        """
        self.log_performance = log_performance
        self.logger = logging.getLogger(__name__)
        self.performance_log = []
        
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate SMAPE following the exact competition formula.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            SMAPE value as percentage (0-200)
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        self._validate_inputs(y_true, y_pred)
        
        # Convert to numpy arrays
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Handle edge cases
        if len(y_true) == 0:
            return 0.0
            
        # Calculate SMAPE using competition formula
        # SMAPE = (100/n) * Σ(|predicted - actual| / ((|predicted| + |actual|) / 2))
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_pred) + np.abs(y_true)) / 2.0
        
        # Handle zero denominators (both predicted and actual are zero)
        # When both are zero, the error is 0, so we set the ratio to 0
        mask = denominator == 0
        ratios = np.zeros_like(numerator)
        ratios[~mask] = numerator[~mask] / denominator[~mask]
        
        # Calculate mean and convert to percentage
        smape = 100.0 * np.mean(ratios)
        
        # Log performance if enabled
        if self.log_performance:
            self._log_calculation_performance(y_true, y_pred, smape)
            
        return smape
    
    def calculate_smape_with_details(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate SMAPE with detailed breakdown and statistics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with SMAPE and detailed statistics
        """
        smape = self.calculate_smape(y_true, y_pred)
        
        # Calculate additional statistics
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Per-sample errors
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_pred) + np.abs(y_true)) / 2.0
        mask = denominator == 0
        per_sample_errors = np.zeros_like(numerator)
        per_sample_errors[~mask] = 100.0 * numerator[~mask] / denominator[~mask]
        
        # Statistics
        stats = {
            'smape': smape,
            'n_samples': len(y_true),
            'mean_error': np.mean(per_sample_errors),
            'std_error': np.std(per_sample_errors),
            'median_error': np.median(per_sample_errors),
            'min_error': np.min(per_sample_errors),
            'max_error': np.max(per_sample_errors),
            'zero_denominator_count': np.sum(mask),
            'per_sample_errors': per_sample_errors.tolist()
        }
        
        return stats
    
    def calculate_quantile_smape(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                n_quantiles: int = 5) -> Dict[str, float]:
        """
        Calculate SMAPE for different price quantiles.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            n_quantiles: Number of quantiles to analyze
            
        Returns:
            Dictionary with SMAPE for each quantile
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Calculate quantile boundaries based on true values
        quantile_boundaries = np.percentile(y_true, np.linspace(0, 100, n_quantiles + 1))
        
        quantile_smapes = {}
        for i in range(n_quantiles):
            # Find samples in this quantile
            lower_bound = quantile_boundaries[i]
            upper_bound = quantile_boundaries[i + 1]
            
            if i == n_quantiles - 1:  # Include upper boundary in last quantile
                mask = (y_true >= lower_bound) & (y_true <= upper_bound)
            else:
                mask = (y_true >= lower_bound) & (y_true < upper_bound)
            
            if np.sum(mask) > 0:
                quantile_smape = self.calculate_smape(y_true[mask], y_pred[mask])
                quantile_name = f"Q{i+1}_{lower_bound:.2f}-{upper_bound:.2f}"
                quantile_smapes[quantile_name] = quantile_smape
            
        return quantile_smapes
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Validate input arrays."""
        if y_true is None or y_pred is None:
            raise ValueError("Input arrays cannot be None")
            
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
            
        if len(y_true.shape) != 1:
            raise ValueError("Input arrays must be 1-dimensional")
            
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            raise ValueError("Input arrays must contain only finite values")
    
    def _log_calculation_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   smape: float) -> None:
        """Log performance metrics for tracking."""
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
            'smape': smape,
            'y_true_stats': {
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true)),
                'min': float(np.min(y_true)),
                'max': float(np.max(y_true))
            },
            'y_pred_stats': {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred))
            }
        }
        
        self.performance_log.append(performance_entry)
        
        # Log to file if logger is configured
        self.logger.info(f"SMAPE calculation: {smape:.4f} on {len(y_true)} samples")
    
    def save_performance_log(self, filepath: str) -> None:
        """Save performance log to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
    
    def run_validation_tests(self) -> bool:
        """
        Run comprehensive validation tests on known examples.
        
        Returns:
            True if all tests pass, False otherwise
        """
        test_cases = self._get_validation_test_cases()
        all_passed = True
        
        self.logger.info("Running SMAPE validation tests...")
        
        for i, (y_true, y_pred, expected_smape, description) in enumerate(test_cases):
            try:
                calculated_smape = self.calculate_smape(y_true, y_pred)
                
                # Allow small numerical differences
                tolerance = 1e-6
                if abs(calculated_smape - expected_smape) <= tolerance:
                    self.logger.info(f"✓ Test {i+1} passed: {description}")
                else:
                    self.logger.error(
                        f"✗ Test {i+1} failed: {description}\n"
                        f"  Expected: {expected_smape:.6f}\n"
                        f"  Got: {calculated_smape:.6f}\n"
                        f"  Difference: {abs(calculated_smape - expected_smape):.6f}"
                    )
                    all_passed = False
                    
            except Exception as e:
                self.logger.error(f"✗ Test {i+1} error: {description} - {str(e)}")
                all_passed = False
        
        if all_passed:
            self.logger.info("All SMAPE validation tests passed!")
        else:
            self.logger.error("Some SMAPE validation tests failed!")
            
        return all_passed
    
    def _get_validation_test_cases(self) -> List[Tuple[np.ndarray, np.ndarray, float, str]]:
        """Get validation test cases with known expected results."""
        return [
            # Perfect predictions
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.0, 
             "Perfect predictions should give SMAPE = 0"),
            
            # Simple case
            (np.array([2.0, 4.0]), np.array([1.0, 3.0]), 47.619048,
             "Simple case: |1-2|/1.5 + |3-4|/3.5 = 2/3 + 2/7 ≈ 0.952 → 47.6%"),
            
            # Both zeros (edge case)
            (np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0,
             "Both actual and predicted zero should give SMAPE = 0"),
            
            # One zero actual, non-zero predicted
            (np.array([0.0, 2.0]), np.array([1.0, 2.0]), 100.0,
             "Zero actual with non-zero predicted: |1-0|/0.5 + |2-2|/2 = 200% + 0% → 100%"),
            
            # One zero predicted, non-zero actual  
            (np.array([1.0, 2.0]), np.array([0.0, 2.0]), 100.0,
             "Zero predicted with non-zero actual: |0-1|/0.5 + |2-2|/2 = 200% + 0% → 100%"),
            
            # Large values
            (np.array([1000.0, 2000.0]), np.array([900.0, 1800.0]), 10.526316,
             "Large values: |900-1000|/950 + |1800-2000|/1900 ≈ 0.2105 → 10.53%"),
            
            # Single value
            (np.array([5.0]), np.array([4.0]), 22.222222,
             "Single value: |4-5|/4.5 = 1/4.5 ≈ 0.2222 → 22.22%"),
            
            # Negative values (if applicable)
            (np.array([-2.0, -4.0]), np.array([-1.0, -3.0]), 47.619048,
             "Negative values: |-1-(-2)|/1.5 + |-3-(-4)|/3.5 = 1/1.5 + 2/7 ≈ 0.952 → 47.6%"),
        ]


def create_smape_calculator(log_performance: bool = True) -> SMAPECalculator:
    """Factory function to create SMAPE calculator."""
    return SMAPECalculator(log_performance=log_performance)