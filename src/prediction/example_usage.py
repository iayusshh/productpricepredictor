"""
Example usage of prediction generation and output formatting components
for ML Product Pricing Challenge 2025

This module demonstrates how to use the PredictionGenerator, OutputFormatter,
and OutputValidator components together for complete prediction pipeline.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Dict

from .prediction_generator import PredictionGenerator
from .output_formatter import OutputFormatter
from .output_validator import OutputValidator
from ..config import config


def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_test_data() -> pd.DataFrame:
    """Load test data for prediction."""
    test_file = Path(config.data.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    test_df = pd.read_csv(test_file)
    logging.info(f"Loaded test data: {test_df.shape}")
    return test_df


def create_mock_model():
    """Create a mock model for demonstration purposes."""
    class MockModel:
        def predict(self, X):
            # Generate realistic-looking predictions
            np.random.seed(42)
            n_samples = X.shape[0]
            
            # Generate predictions with some realistic distribution
            base_prices = np.random.lognormal(mean=3.0, sigma=1.0, size=n_samples)
            
            # Add some variation based on "features"
            feature_effect = np.random.normal(0, 0.2, size=n_samples)
            predictions = base_prices * (1 + feature_effect)
            
            # Ensure some edge cases for testing
            predictions[0] = -5.0  # Negative value to test clamping
            predictions[1] = 0.0   # Zero value to test clamping
            predictions[2] = 1000.0  # High value
            
            return predictions
    
    return MockModel()


def create_mock_features(test_df: pd.DataFrame) -> np.ndarray:
    """Create mock features for demonstration."""
    n_samples = len(test_df)
    n_features = 100  # Mock feature dimension
    
    # Generate random features
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    
    logging.info(f"Created mock features: {features.shape}")
    return features


def example_basic_prediction_pipeline():
    """Example of basic prediction pipeline."""
    print("\n" + "="*60)
    print("BASIC PREDICTION PIPELINE EXAMPLE")
    print("="*60)
    
    # Setup
    setup_logging()
    
    # Load test data
    test_df = load_test_data()
    sample_ids = test_df['sample_id'].tolist()
    
    # Create mock model and features
    model = create_mock_model()
    X_test = create_mock_features(test_df)
    
    # Initialize components
    predictor = PredictionGenerator(
        min_threshold=0.01,
        batch_size=1000,
        enable_confidence=True
    )
    
    formatter = OutputFormatter(
        output_precision=6,
        strict_mode=True
    )
    
    validator = OutputValidator(
        strict_mode=True,
        outlier_detection=True,
        generate_report=True
    )
    
    # Step 1: Generate predictions
    print("\n1. Generating predictions...")
    raw_predictions = predictor.predict(model, X_test)
    print(f"Generated {len(raw_predictions)} predictions")
    print(f"Raw predictions - min: {np.min(raw_predictions):.6f}, max: {np.max(raw_predictions):.6f}")
    
    # Step 2: Apply clamping
    print("\n2. Applying prediction clamping...")
    clamped_predictions = predictor.clamp_predictions_to_threshold(raw_predictions)
    print(f"Clamped predictions - min: {np.min(clamped_predictions):.6f}, max: {np.max(clamped_predictions):.6f}")
    
    # Step 3: Format output
    print("\n3. Formatting output...")
    output_df = formatter.format_predictions_exact(sample_ids, clamped_predictions)
    print(f"Formatted output shape: {output_df.shape}")
    print(f"Output columns: {list(output_df.columns)}")
    
    # Step 4: Validate output
    print("\n4. Validating output...")
    validation_results = validator.validate_complete_output(output_df, test_df, raw_predictions)
    print(f"Overall validation status: {validation_results['overall_status']['status']}")
    
    # Step 5: Create submission file
    print("\n5. Creating submission file...")
    output_file = "test_out_example.csv"
    final_df, validation_passed = formatter.create_submission_file(
        sample_ids, clamped_predictions, test_df, output_file
    )
    
    if validation_passed:
        print(f"✅ Submission file created successfully: {output_file}")
    else:
        print(f"❌ Submission file created with validation issues: {output_file}")
    
    # Step 6: Final quality assurance
    print("\n6. Final quality assurance...")
    qa_results = validator.perform_final_quality_assurance(final_df, test_df, output_file)
    print(f"Final QA Status: {qa_results['overall_status']}")
    
    return final_df, validation_results, qa_results


def example_batch_prediction_pipeline():
    """Example of batch prediction pipeline for large datasets."""
    print("\n" + "="*60)
    print("BATCH PREDICTION PIPELINE EXAMPLE")
    print("="*60)
    
    # Setup
    setup_logging()
    
    # Load test data
    test_df = load_test_data()
    sample_ids = test_df['sample_id'].tolist()
    
    # Create mock model and features
    model = create_mock_model()
    X_test = create_mock_features(test_df)
    
    # Initialize predictor with smaller batch size
    predictor = PredictionGenerator(
        min_threshold=0.01,
        batch_size=500,  # Smaller batches for demonstration
        enable_confidence=True
    )
    
    # Generate predictions using batch processing
    print("\n1. Generating predictions with batch processing...")
    batch_predictions = predictor.predict_batch(model, X_test)
    print(f"Generated {len(batch_predictions)} predictions using batch processing")
    
    # Apply clamping and get statistics
    clamped_predictions = predictor.clamp_predictions_to_threshold(batch_predictions)
    stats = predictor.get_prediction_statistics()
    
    print(f"\nPrediction Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return clamped_predictions, stats


def example_ensemble_prediction_pipeline():
    """Example of ensemble prediction pipeline."""
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTION PIPELINE EXAMPLE")
    print("="*60)
    
    # Setup
    setup_logging()
    
    # Load test data
    test_df = load_test_data()
    X_test = create_mock_features(test_df)
    
    # Create multiple mock models for ensemble
    models = [create_mock_model() for _ in range(3)]
    
    # Initialize predictor
    predictor = PredictionGenerator(enable_confidence=True)
    
    # Generate ensemble predictions
    print("\n1. Generating ensemble predictions...")
    ensemble_predictions = predictor.ensemble_predict(models, X_test)
    print(f"Generated ensemble predictions from {len(models)} models")
    
    # Estimate confidence
    print("\n2. Estimating prediction confidence...")
    confidence_scores = predictor.estimate_prediction_confidence(
        models[0], X_test, ensemble_predictions, method='ensemble_std'
    )
    print(f"Confidence scores - mean: {np.mean(confidence_scores):.3f}, std: {np.std(confidence_scores):.3f}")
    
    return ensemble_predictions, confidence_scores


def example_comprehensive_validation():
    """Example of comprehensive validation workflow."""
    print("\n" + "="*60)
    print("COMPREHENSIVE VALIDATION EXAMPLE")
    print("="*60)
    
    # Setup
    setup_logging()
    
    # Load test data
    test_df = load_test_data()
    sample_ids = test_df['sample_id'].tolist()
    
    # Create predictions
    model = create_mock_model()
    X_test = create_mock_features(test_df)
    
    predictor = PredictionGenerator()
    predictions = predictor.predict(model, X_test)
    clamped_predictions = predictor.clamp_predictions_to_threshold(predictions)
    
    # Format output
    formatter = OutputFormatter()
    output_df = formatter.format_predictions_exact(sample_ids, clamped_predictions)
    
    # Initialize validator
    validator = OutputValidator(
        strict_mode=True,
        outlier_detection=True,
        generate_report=True
    )
    
    # Perform individual validations
    print("\n1. Sample ID validation...")
    sample_id_results = validator.verify_exact_sample_id_match(output_df, test_df)
    print(f"Sample ID match: {sample_id_results['exact_match']}")
    
    print("\n2. Prediction range validation...")
    range_results = validator.validate_prediction_ranges(output_df)
    print(f"Range valid: {range_results['range_valid']}")
    
    print("\n3. Submission integrity checks...")
    integrity_results = validator.create_submission_integrity_checks(output_df, test_df)
    print(f"Integrity passed: {integrity_results['overall_integrity']['all_checks_passed']}")
    
    print("\n4. Final quality assurance...")
    qa_results = validator.perform_final_quality_assurance(output_df, test_df)
    print(f"QA Status: {qa_results['overall_status']}")
    
    # Print recommendations
    if qa_results['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(qa_results['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return {
        'sample_id_results': sample_id_results,
        'range_results': range_results,
        'integrity_results': integrity_results,
        'qa_results': qa_results
    }


def example_error_handling():
    """Example of error handling in prediction pipeline."""
    print("\n" + "="*60)
    print("ERROR HANDLING EXAMPLE")
    print("="*60)
    
    setup_logging()
    
    # Test various error conditions
    predictor = PredictionGenerator()
    formatter = OutputFormatter()
    validator = OutputValidator()
    
    print("\n1. Testing invalid model...")
    try:
        predictor.predict(None, np.random.randn(10, 5))
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n2. Testing empty test data...")
    try:
        predictor.predict(create_mock_model(), np.array([]))
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n3. Testing mismatched sample IDs and predictions...")
    try:
        formatter.format_predictions_exact(['id1', 'id2'], np.array([1.0, 2.0, 3.0]))
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n4. Testing validation with mismatched data...")
    test_df = pd.DataFrame({'sample_id': ['id1', 'id2'], 'other_col': [1, 2]})
    output_df = pd.DataFrame({'sample_id': ['id1', 'id3'], 'price': [10.0, 20.0]})
    
    results = validator.verify_exact_sample_id_match(output_df, test_df)
    print(f"✓ Validation correctly identified mismatch: {not results['exact_match']}")
    
    print("\nError handling tests completed successfully!")


def run_all_examples():
    """Run all example workflows."""
    print("RUNNING ALL PREDICTION PIPELINE EXAMPLES")
    print("="*80)
    
    try:
        # Basic pipeline
        basic_results = example_basic_prediction_pipeline()
        
        # Batch processing
        batch_results = example_batch_prediction_pipeline()
        
        # Ensemble predictions
        ensemble_results = example_ensemble_prediction_pipeline()
        
        # Comprehensive validation
        validation_results = example_comprehensive_validation()
        
        # Error handling
        example_error_handling()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return {
            'basic_results': basic_results,
            'batch_results': batch_results,
            'ensemble_results': ensemble_results,
            'validation_results': validation_results
        }
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {str(e)}")
        logging.error(f"Example execution failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()