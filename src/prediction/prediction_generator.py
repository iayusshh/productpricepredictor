"""
PredictionGenerator with clamping and validation for ML Product Pricing Challenge 2025

This module implements the prediction pipeline that processes test.csv end-to-end,
including prediction clamping, batch processing, and confidence estimation.
"""

import logging
import time
from typing import Any, List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from scipy import stats

from ..interfaces import PredictionGeneratorInterface
from ..models.data_models import PredictionResult, ModelConfig
from ..config import config


class PredictionGenerator(PredictionGeneratorInterface):
    """
    Enhanced prediction generator with clamping, validation, and batch processing.
    
    This class implements the complete prediction pipeline from test data loading
    through final prediction generation with comprehensive validation and error handling.
    """
    
    def __init__(self, 
                 min_threshold: float = None,
                 batch_size: int = None,
                 enable_confidence: bool = True,
                 log_predictions: bool = True):
        """
        Initialize PredictionGenerator with configuration.
        
        Args:
            min_threshold: Minimum prediction threshold (default from config)
            batch_size: Batch size for memory-efficient processing (default from config)
            enable_confidence: Whether to calculate prediction confidence
            log_predictions: Whether to log prediction statistics
        """
        self.min_threshold = min_threshold or config.prediction.min_price_threshold
        self.batch_size = batch_size or config.infrastructure.batch_size
        self.enable_confidence = enable_confidence
        self.log_predictions = log_predictions
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Prediction statistics
        self.prediction_stats = {
            'total_predictions': 0,
            'clamped_predictions': 0,
            'negative_predictions': 0,
            'zero_predictions': 0,
            'outlier_predictions': 0,
            'processing_time': 0.0
        }
        
        self.logger.info(f"PredictionGenerator initialized with min_threshold={self.min_threshold}, "
                        f"batch_size={self.batch_size}")
    
    def predict(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions for test data with comprehensive error handling.
        
        Args:
            model: Trained model (sklearn-compatible or custom)
            X_test: Test features array
            
        Returns:
            np.ndarray: Raw predictions before clamping
            
        Raises:
            ValueError: If model or test data is invalid
            RuntimeError: If prediction fails
        """
        if model is None:
            raise ValueError("Model cannot be None")
        
        if X_test is None or X_test.size == 0:
            raise ValueError("Test data cannot be None or empty")
        
        self.logger.info(f"Generating predictions for {X_test.shape[0]} samples")
        start_time = time.time()
        
        try:
            # Check if model has predict method
            if not hasattr(model, 'predict'):
                raise ValueError("Model must have a 'predict' method")
            
            # Generate predictions
            predictions = model.predict(X_test)
            
            # Validate predictions
            if predictions is None or len(predictions) == 0:
                raise RuntimeError("Model returned empty predictions")
            
            if len(predictions) != X_test.shape[0]:
                raise RuntimeError(f"Prediction count mismatch: expected {X_test.shape[0]}, "
                                 f"got {len(predictions)}")
            
            # Convert to numpy array if needed
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            # Update statistics
            self.prediction_stats['total_predictions'] = len(predictions)
            self.prediction_stats['negative_predictions'] = np.sum(predictions < 0)
            self.prediction_stats['zero_predictions'] = np.sum(predictions == 0)
            self.prediction_stats['processing_time'] = time.time() - start_time
            
            if self.log_predictions:
                self._log_prediction_statistics(predictions)
            
            self.logger.info(f"Generated {len(predictions)} predictions in "
                           f"{self.prediction_stats['processing_time']:.2f} seconds")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate predictions: {str(e)}")
    
    def predict_batch(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions using batch processing for memory efficiency.
        
        Args:
            model: Trained model
            X_test: Test features array
            
        Returns:
            np.ndarray: Predictions for all batches combined
        """
        self.logger.info(f"Starting batch prediction with batch_size={self.batch_size}")
        
        n_samples = X_test.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        all_predictions = []
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            batch_X = X_test[start_idx:end_idx]
            
            self.logger.debug(f"Processing batch {i+1}/{n_batches} "
                            f"(samples {start_idx}:{end_idx})")
            
            try:
                batch_predictions = self.predict(model, batch_X)
                all_predictions.append(batch_predictions)
                
            except Exception as e:
                self.logger.error(f"Batch {i+1} prediction failed: {str(e)}")
                # Create fallback predictions for this batch
                fallback_predictions = np.full(batch_X.shape[0], self.min_threshold)
                all_predictions.append(fallback_predictions)
                self.logger.warning(f"Using fallback predictions for batch {i+1}")
        
        # Combine all batch predictions
        final_predictions = np.concatenate(all_predictions)
        
        self.logger.info(f"Completed batch prediction: {len(final_predictions)} total predictions")
        
        return final_predictions
    
    def clamp_predictions_to_threshold(self, 
                                     predictions: np.ndarray, 
                                     min_threshold: float = None) -> np.ndarray:
        """
        Clamp predictions to minimum threshold with documented rationale.
        
        The minimum threshold is applied because:
        1. Negative prices are not economically meaningful
        2. Zero prices may indicate data quality issues
        3. Very small prices may be prediction artifacts
        4. Competition evaluation may have implicit minimum bounds
        
        Args:
            predictions: Raw predictions array
            min_threshold: Minimum threshold (default: self.min_threshold)
            
        Returns:
            np.ndarray: Clamped predictions
        """
        if predictions is None or len(predictions) == 0:
            raise ValueError("Predictions cannot be None or empty")
        
        threshold = min_threshold or self.min_threshold
        
        # Count predictions that will be clamped
        below_threshold = predictions < threshold
        clamped_count = np.sum(below_threshold)
        
        # Apply clamping
        clamped_predictions = np.maximum(predictions, threshold)
        
        # Update statistics
        self.prediction_stats['clamped_predictions'] = clamped_count
        
        if clamped_count > 0:
            self.logger.info(f"Clamped {clamped_count} predictions to minimum threshold {threshold}")
            self.logger.info(f"Clamping ratio: {clamped_count/len(predictions)*100:.2f}%")
            
            # Log details about clamped predictions
            original_clamped = predictions[below_threshold]
            self.logger.debug(f"Original values of clamped predictions - "
                            f"min: {np.min(original_clamped):.6f}, "
                            f"max: {np.max(original_clamped):.6f}, "
                            f"mean: {np.mean(original_clamped):.6f}")
        
        return clamped_predictions
    
    def estimate_prediction_confidence(self, 
                                     model: Any, 
                                     X_test: np.ndarray,
                                     predictions: np.ndarray,
                                     method: str = 'ensemble_std') -> np.ndarray:
        """
        Estimate prediction confidence/uncertainty.
        
        Args:
            model: Trained model
            X_test: Test features
            predictions: Model predictions
            method: Confidence estimation method
            
        Returns:
            np.ndarray: Confidence scores (higher = more confident)
        """
        if not self.enable_confidence:
            return np.ones(len(predictions))
        
        self.logger.info(f"Estimating prediction confidence using method: {method}")
        
        try:
            if method == 'ensemble_std':
                # If model is an ensemble, use prediction variance
                if hasattr(model, 'estimators_'):
                    # For sklearn ensemble models
                    individual_predictions = []
                    for estimator in model.estimators_:
                        pred = estimator.predict(X_test)
                        individual_predictions.append(pred)
                    
                    pred_array = np.array(individual_predictions)
                    confidence = 1.0 / (1.0 + np.std(pred_array, axis=0))
                    
                elif hasattr(model, 'predict_proba'):
                    # For models with probability estimates
                    # Use prediction entropy as uncertainty measure
                    probas = model.predict_proba(X_test)
                    entropy = -np.sum(probas * np.log(probas + 1e-8), axis=1)
                    confidence = 1.0 / (1.0 + entropy)
                    
                else:
                    # Fallback: use prediction magnitude as proxy
                    confidence = np.minimum(predictions / np.max(predictions), 1.0)
                    
            elif method == 'prediction_magnitude':
                # Higher predictions assumed more confident (domain-specific)
                confidence = np.minimum(predictions / np.percentile(predictions, 95), 1.0)
                
            elif method == 'feature_density':
                # Confidence based on feature space density (simplified)
                # Higher density = more training examples nearby = higher confidence
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=5)
                nn.fit(X_test)  # This is a simplification - should use training data
                distances, _ = nn.kneighbors(X_test)
                avg_distance = np.mean(distances, axis=1)
                confidence = 1.0 / (1.0 + avg_distance)
                
            else:
                self.logger.warning(f"Unknown confidence method: {method}, using default")
                confidence = np.ones(len(predictions))
            
            # Normalize confidence to [0, 1]
            confidence = np.clip(confidence, 0.0, 1.0)
            
            self.logger.info(f"Confidence estimation completed - "
                           f"mean: {np.mean(confidence):.3f}, "
                           f"std: {np.std(confidence):.3f}")
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Confidence estimation failed: {str(e)}")
            return np.ones(len(predictions))
    
    def ensemble_predict(self, models: List[Any], X_test: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions from multiple models.
        
        Args:
            models: List of trained models
            X_test: Test features array
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not models:
            raise ValueError("Models list cannot be empty")
        
        self.logger.info(f"Generating ensemble predictions from {len(models)} models")
        
        all_predictions = []
        model_weights = []
        
        for i, model in enumerate(models):
            try:
                predictions = self.predict(model, X_test)
                all_predictions.append(predictions)
                
                # Simple weighting - could be enhanced with validation performance
                weight = 1.0 / len(models)
                model_weights.append(weight)
                
                self.logger.debug(f"Model {i+1} predictions: "
                                f"mean={np.mean(predictions):.3f}, "
                                f"std={np.std(predictions):.3f}")
                
            except Exception as e:
                self.logger.error(f"Model {i+1} prediction failed: {str(e)}")
                continue
        
        if not all_predictions:
            raise RuntimeError("All models failed to generate predictions")
        
        # Weighted average ensemble
        predictions_array = np.array(all_predictions)
        weights_array = np.array(model_weights)
        
        # Normalize weights
        weights_array = weights_array / np.sum(weights_array)
        
        ensemble_predictions = np.average(predictions_array, axis=0, weights=weights_array)
        
        self.logger.info(f"Ensemble predictions generated - "
                        f"mean: {np.mean(ensemble_predictions):.3f}, "
                        f"std: {np.std(ensemble_predictions):.3f}")
        
        return ensemble_predictions
    
    def format_output(self, sample_ids: List[str], predictions: np.ndarray) -> pd.DataFrame:
        """
        Format predictions according to submission requirements.
        
        Args:
            sample_ids: List of sample IDs
            predictions: Prediction values
            
        Returns:
            pd.DataFrame: Formatted output with sample_id and price columns
        """
        if len(sample_ids) != len(predictions):
            raise ValueError(f"Sample ID count ({len(sample_ids)}) must match "
                           f"prediction count ({len(predictions)})")
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'sample_id': sample_ids,
            'price': predictions
        })
        
        # Ensure price precision
        output_df['price'] = output_df['price'].round(config.prediction.output_precision)
        
        self.logger.info(f"Formatted output: {len(output_df)} rows")
        
        return output_df
    
    def validate_exact_sample_id_match(self, 
                                     output_df: pd.DataFrame, 
                                     test_df: pd.DataFrame) -> bool:
        """
        Validate exact sample_id matching between output and test data.
        
        Args:
            output_df: Output DataFrame with predictions
            test_df: Original test DataFrame
            
        Returns:
            bool: True if sample_ids match exactly
        """
        output_ids = set(output_df['sample_id'].values)
        test_ids = set(test_df['sample_id'].values)
        
        if output_ids == test_ids:
            self.logger.info("Sample ID validation passed: exact match")
            return True
        
        # Detailed mismatch analysis
        missing_in_output = test_ids - output_ids
        extra_in_output = output_ids - test_ids
        
        self.logger.error(f"Sample ID validation failed:")
        if missing_in_output:
            self.logger.error(f"Missing in output: {len(missing_in_output)} IDs")
            self.logger.debug(f"First 5 missing: {list(missing_in_output)[:5]}")
        
        if extra_in_output:
            self.logger.error(f"Extra in output: {len(extra_in_output)} IDs")
            self.logger.debug(f"First 5 extra: {list(extra_in_output)[:5]}")
        
        return False
    
    def validate_row_count_match(self, 
                               output_df: pd.DataFrame, 
                               test_df: pd.DataFrame) -> bool:
        """
        Validate row count matching between output and test data.
        
        Args:
            output_df: Output DataFrame with predictions
            test_df: Original test DataFrame
            
        Returns:
            bool: True if row counts match exactly
        """
        output_count = len(output_df)
        test_count = len(test_df)
        
        if output_count == test_count:
            self.logger.info(f"Row count validation passed: {output_count} rows")
            return True
        
        self.logger.error(f"Row count validation failed: "
                         f"output has {output_count} rows, test has {test_count} rows")
        return False
    
    def validate_output(self, output_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Comprehensive output validation.
        
        Args:
            output_df: Output DataFrame with predictions
            test_df: Original test DataFrame
            
        Returns:
            bool: True if all validations pass
        """
        self.logger.info("Starting comprehensive output validation")
        
        validations = []
        
        # Check required columns
        required_columns = ['sample_id', 'price']
        missing_columns = [col for col in required_columns if col not in output_df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            validations.append(False)
        else:
            validations.append(True)
        
        # Check for null values
        null_counts = output_df.isnull().sum()
        if null_counts.any():
            self.logger.error(f"Null values found: {null_counts.to_dict()}")
            validations.append(False)
        else:
            validations.append(True)
        
        # Validate sample ID matching
        validations.append(self.validate_exact_sample_id_match(output_df, test_df))
        
        # Validate row count matching
        validations.append(self.validate_row_count_match(output_df, test_df))
        
        # Validate price values
        price_validation = self._validate_price_values(output_df['price'])
        validations.append(price_validation)
        
        # Check for duplicates
        duplicate_count = output_df['sample_id'].duplicated().sum()
        if duplicate_count > 0:
            self.logger.error(f"Found {duplicate_count} duplicate sample_ids")
            validations.append(False)
        else:
            validations.append(True)
        
        all_valid = all(validations)
        
        if all_valid:
            self.logger.info("All output validations passed")
        else:
            self.logger.error("Output validation failed")
        
        return all_valid
    
    def _validate_price_values(self, prices: pd.Series) -> bool:
        """
        Validate price values are positive floats.
        
        Args:
            prices: Series of price values
            
        Returns:
            bool: True if all prices are valid
        """
        # Check for non-numeric values
        non_numeric = prices.apply(lambda x: not isinstance(x, (int, float, np.number)))
        if non_numeric.any():
            self.logger.error(f"Found {non_numeric.sum()} non-numeric price values")
            return False
        
        # Check for negative values
        negative_count = (prices < 0).sum()
        if negative_count > 0:
            self.logger.error(f"Found {negative_count} negative price values")
            return False
        
        # Check for infinite values
        infinite_count = np.isinf(prices).sum()
        if infinite_count > 0:
            self.logger.error(f"Found {infinite_count} infinite price values")
            return False
        
        # Check for NaN values
        nan_count = np.isnan(prices).sum()
        if nan_count > 0:
            self.logger.error(f"Found {nan_count} NaN price values")
            return False
        
        self.logger.info("Price value validation passed")
        return True
    
    def _log_prediction_statistics(self, predictions: np.ndarray):
        """Log detailed prediction statistics."""
        stats_dict = {
            'count': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75)),
            'negative_count': int(np.sum(predictions < 0)),
            'zero_count': int(np.sum(predictions == 0)),
            'below_threshold_count': int(np.sum(predictions < self.min_threshold))
        }
        
        self.logger.info(f"Prediction statistics: {stats_dict}")
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive prediction statistics.
        
        Returns:
            Dict: Dictionary containing prediction statistics
        """
        return self.prediction_stats.copy()
    
    def reset_statistics(self):
        """Reset prediction statistics."""
        self.prediction_stats = {
            'total_predictions': 0,
            'clamped_predictions': 0,
            'negative_predictions': 0,
            'zero_predictions': 0,
            'outlier_predictions': 0,
            'processing_time': 0.0
        }
        self.logger.info("Prediction statistics reset")