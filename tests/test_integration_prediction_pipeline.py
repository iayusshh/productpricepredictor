"""
Integration tests for Prediction Pipeline.

End-to-end integration tests for the complete prediction pipeline including
data loading, feature engineering, prediction generation, and output formatting.
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import pickle

from src.prediction_pipeline import PredictionPipeline
from src.config import config


class TestPredictionPipelineIntegration(unittest.TestCase):
    """Integration test cases for Prediction Pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        (self.temp_path / "dataset").mkdir()
        (self.temp_path / "images").mkdir()
        (self.temp_path / "cache").mkdir()
        (self.temp_path / "models").mkdir()
        (self.temp_path / "logs").mkdir()
        (self.temp_path / "embeddings").mkdir()
        (self.temp_path / "deliverables").mkdir()
        
        # Create test data files
        self.create_test_data_files()
        self.create_mock_model_files()
        
        # Mock config
        self.original_config = config
        self.mock_config()
        
        # Initialize pipeline
        self.pipeline = PredictionPipeline()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original config
        config.__dict__.update(self.original_config.__dict__)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data_files(self):
        """Create test data files."""
        # Test data for prediction
        test_data = pd.DataFrame({
            'sample_id': [f'test_{i}' for i in range(1, 51)],  # 50 samples
            'catalog_content': [
                f'Test Product {i} with pack of {i%3+1} pieces, '
                f'{"aluminum" if i%2 else "steel"} material, '
                f'{"green" if i%3 else "yellow"} finish, '
                f'weight: {20+i}g, '
                f'dimensions: {i+5}cm x {i+6}cm x {i+7}cm'
                for i in range(1, 51)
            ],
            'image_link': [f'https://example.com/test_image_{i}.jpg' for i in range(1, 51)]
        })
        test_data.to_csv(self.temp_path / "dataset" / "test.csv", index=False)
        
        # Sample output format for validation
        sample_output = pd.DataFrame({
            'sample_id': [f'test_{i}' for i in range(1, 6)],
            'price': [10.99, 25.50, 15.75, 8.25, 45.00]
        })
        sample_output.to_csv(self.temp_path / "dataset" / "sample_test_out.csv", index=False)
    
    def create_mock_model_files(self):
        """Create mock trained model files."""
        # Create a simple mock model
        class MockModel:
            def predict(self, X):
                # Simple prediction: sum of features with some noise
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                predictions = np.sum(X, axis=1) * 0.1 + np.random.uniform(10, 50, X.shape[0])
                return np.maximum(predictions, 0.01)  # Ensure positive predictions
            
            def get_params(self):
                return {'model_type': 'mock_model'}
        
        # Save mock model
        model_path = self.temp_path / "models" / "best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(MockModel(), f)
        
        # Create mock feature names
        feature_names = [f'feature_{i}' for i in range(50)]  # 50 features
        feature_names_path = self.temp_path / "models" / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f)
        
        # Create mock model metadata
        model_metadata = {
            'model_name': 'mock_model',
            'feature_count': 50,
            'training_samples': 100,
            'cv_score': 25.5,
            'timestamp': '2024-01-01T00:00:00'
        }
        metadata_path = self.temp_path / "models" / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f)
    
    def mock_config(self):
        """Mock configuration for testing."""
        # Data paths
        config.data.test_file = str(self.temp_path / "dataset" / "test.csv")
        config.data.sample_output_file = str(self.temp_path / "dataset" / "sample_test_out.csv")
        config.data.image_dir = str(self.temp_path / "images")
        config.data.cache_dir = str(self.temp_path / "cache")
        
        # Model paths
        config.model.model_dir = str(self.temp_path / "models")
        config.model.embedding_dir = str(self.temp_path / "embeddings")
        
        # Output paths
        config.output.deliverables_dir = str(self.temp_path / "deliverables")
        config.output.predictions_file = str(self.temp_path / "deliverables" / "test_out.csv")
        
        # Logging
        config.logging.log_dir = str(self.temp_path / "logs")
        
        # Prediction settings
        config.prediction.min_price_threshold = 0.01
        config.prediction.batch_size = 10
        config.prediction.confidence_threshold = 0.5
        
        # Data processing
        config.data.download_timeout = 5
        config.data.max_download_retries = 1
        config.data.batch_size = 10
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_complete_prediction_pipeline(self, mock_get):
        """Test complete prediction pipeline from data loading to output generation."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Run complete pipeline
        try:
            results = self.pipeline.run_complete_pipeline()
            
            # Verify pipeline completed successfully
            self.assertIsInstance(results, dict)
            self.assertIn('prediction_completed', results)
            self.assertTrue(results['prediction_completed'])
            
            # Verify key results are present
            expected_keys = ['data_summary', 'feature_summary', 'prediction_results', 
                           'output_validation', 'pipeline_metrics']
            for key in expected_keys:
                self.assertIn(key, results)
            
            # Verify data was processed
            self.assertGreater(results['data_summary']['total_samples'], 0)
            
            # Verify features were extracted
            self.assertGreater(results['feature_summary']['total_features'], 0)
            
            # Verify predictions were generated
            self.assertIn('total_predictions', results['prediction_results'])
            self.assertEqual(results['prediction_results']['total_predictions'], 50)
            
            # Verify output file was created
            output_file = Path(config.output.predictions_file)
            self.assertTrue(output_file.exists())
            
            # Verify output format
            output_df = pd.read_csv(output_file)
            self.assertEqual(len(output_df), 50)
            self.assertEqual(list(output_df.columns), ['sample_id', 'price'])
            
            # Verify all predictions are positive
            self.assertTrue((output_df['price'] >= config.prediction.min_price_threshold).all())
            
        except Exception as e:
            self.fail(f"Complete prediction pipeline failed: {str(e)}")
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_data_loading_and_preprocessing_integration(self, mock_get):
        """Test integration of data loading and preprocessing for prediction."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Load and preprocess test data
        test_df = self.pipeline.load_and_preprocess_test_data()
        
        # Verify data loading
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertEqual(len(test_df), 50)
        
        # Verify required columns are present
        required_cols = ['sample_id', 'catalog_content', 'image_link']
        for col in required_cols:
            self.assertIn(col, test_df.columns)
        
        # Verify no duplicate sample IDs
        self.assertEqual(len(test_df['sample_id'].unique()), len(test_df))
        
        # Verify data completeness
        self.assertFalse(test_df['sample_id'].isna().any())
        content_completeness = test_df['catalog_content'].notna().mean()
        self.assertGreater(content_completeness, 0.8)  # At least 80% complete
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_feature_extraction_integration(self, mock_get):
        """Test integration of feature extraction for prediction."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Load test data
        test_df = self.pipeline.load_and_preprocess_test_data()
        
        # Extract features
        X_test, feature_names = self.pipeline.extract_features_for_prediction(test_df)
        
        # Verify feature extraction
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(feature_names, list)
        
        # Verify feature dimensions
        self.assertEqual(X_test.shape[0], len(test_df))
        self.assertEqual(X_test.shape[1], len(feature_names))
        
        # Verify features are numeric and finite
        self.assertTrue(np.isfinite(X_test).all())
        
        # Verify feature names are meaningful
        self.assertGreater(len(feature_names), 0)
        for name in feature_names:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)
        
        # Verify feature consistency with training (same number of features)
        expected_feature_count = 50  # From mock model
        self.assertEqual(len(feature_names), expected_feature_count)
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_model_loading_and_prediction_integration(self, mock_get):
        """Test integration of model loading and prediction generation."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Prepare data and features
        test_df = self.pipeline.load_and_preprocess_test_data()
        X_test, feature_names = self.pipeline.extract_features_for_prediction(test_df)
        
        # Load model and generate predictions
        model, model_metadata = self.pipeline.load_trained_model()
        predictions = self.pipeline.generate_predictions(model, X_test, test_df['sample_id'])
        
        # Verify model loading
        self.assertIsNotNone(model)
        self.assertIsInstance(model_metadata, dict)
        self.assertIn('model_name', model_metadata)
        
        # Verify predictions
        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(predictions), len(test_df))
        
        # Verify all sample IDs are present
        for sample_id in test_df['sample_id']:
            self.assertIn(sample_id, predictions)
        
        # Verify all predictions are positive numbers
        for sample_id, price in predictions.items():
            self.assertIsInstance(price, (int, float))
            self.assertGreater(price, 0)
            self.assertGreaterEqual(price, config.prediction.min_price_threshold)
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_output_formatting_and_validation_integration(self, mock_get):
        """Test integration of output formatting and validation."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Generate predictions
        test_df = self.pipeline.load_and_preprocess_test_data()
        X_test, feature_names = self.pipeline.extract_features_for_prediction(test_df)
        model, model_metadata = self.pipeline.load_trained_model()
        predictions = self.pipeline.generate_predictions(model, X_test, test_df['sample_id'])
        
        # Format and validate output
        output_df = self.pipeline.format_predictions_for_output(predictions, test_df)
        validation_results = self.pipeline.validate_output_format(output_df, test_df)
        
        # Verify output formatting
        self.assertIsInstance(output_df, pd.DataFrame)
        self.assertEqual(list(output_df.columns), ['sample_id', 'price'])
        self.assertEqual(len(output_df), len(test_df))
        
        # Verify sample ID matching
        pd.testing.assert_series_equal(
            output_df['sample_id'].sort_values().reset_index(drop=True),
            test_df['sample_id'].sort_values().reset_index(drop=True)
        )
        
        # Verify price format
        self.assertTrue(pd.api.types.is_numeric_dtype(output_df['price']))
        self.assertTrue((output_df['price'] > 0).all())
        
        # Verify validation results
        self.assertIsInstance(validation_results, dict)
        self.assertIn('is_valid', validation_results)
        self.assertTrue(validation_results['is_valid'])
        
        if 'validation_errors' in validation_results:
            self.assertEqual(len(validation_results['validation_errors']), 0)
    
    def test_prediction_pipeline_error_handling(self):
        """Test prediction pipeline error handling with invalid data."""
        # Create invalid test data (missing required columns)
        invalid_test_data = pd.DataFrame({
            'sample_id': ['1', '2'],
            'catalog_content': ['Product 1', 'Product 2']
            # Missing image_link column
        })
        invalid_test_data.to_csv(self.temp_path / "dataset" / "test.csv", index=False)
        
        # Pipeline should handle the error gracefully
        with self.assertRaises(Exception) as context:
            self.pipeline.run_complete_pipeline()
        
        # Should be a meaningful error message
        error_message = str(context.exception)
        self.assertTrue(len(error_message) > 0)
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent across runs."""
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Set fixed random seed
            np.random.seed(42)
            
            # Run prediction pipeline twice
            results1 = self.pipeline.run_complete_pipeline()
            
            # Reset and run again
            np.random.seed(42)
            results2 = self.pipeline.run_complete_pipeline()
            
            # Compare key results
            self.assertEqual(results1['data_summary']['total_samples'], 
                           results2['data_summary']['total_samples'])
            
            self.assertEqual(results1['prediction_results']['total_predictions'], 
                           results2['prediction_results']['total_predictions'])
            
            # Load output files and compare
            output1 = pd.read_csv(config.output.predictions_file)
            
            # Run second prediction (need to change output file name)
            config.output.predictions_file = str(self.temp_path / "deliverables" / "test_out_2.csv")
            np.random.seed(42)
            self.pipeline.run_complete_pipeline()
            output2 = pd.read_csv(config.output.predictions_file)
            
            # Predictions should be very similar (allowing for small numerical differences)
            pd.testing.assert_frame_equal(output1, output2, check_exact=False, atol=0.01)
    
    def test_prediction_performance_benchmarking(self):
        """Test prediction pipeline performance and timing."""
        import time
        
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Measure pipeline execution time
            start_time = time.time()
            results = self.pipeline.run_complete_pipeline()
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Verify pipeline completed in reasonable time
            self.assertLess(execution_time, 120)  # Should complete within 2 minutes
            
            # Verify performance metrics are recorded
            if 'pipeline_metrics' in results:
                metrics = results['pipeline_metrics']
                self.assertIn('total_execution_time', metrics)
                self.assertGreater(metrics['total_execution_time'], 0)
                
                # Check prediction throughput
                if 'predictions_per_second' in metrics:
                    throughput = metrics['predictions_per_second']
                    self.assertGreater(throughput, 0.1)  # At least 0.1 predictions per second
    
    def test_batch_prediction_processing(self):
        """Test batch processing of predictions."""
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Set small batch size for testing
            original_batch_size = config.prediction.batch_size
            config.prediction.batch_size = 5
            
            try:
                # Run pipeline with batch processing
                results = self.pipeline.run_complete_pipeline()
                
                # Verify all predictions were generated despite batching
                self.assertEqual(results['prediction_results']['total_predictions'], 50)
                
                # Verify output file contains all samples
                output_df = pd.read_csv(config.output.predictions_file)
                self.assertEqual(len(output_df), 50)
                
            finally:
                # Restore original batch size
                config.prediction.batch_size = original_batch_size
    
    def test_prediction_quality_validation(self):
        """Test prediction quality and reasonableness."""
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Run prediction pipeline
            results = self.pipeline.run_complete_pipeline()
            
            # Load predictions
            output_df = pd.read_csv(config.output.predictions_file)
            
            # Validate prediction quality
            prices = output_df['price']
            
            # All prices should be positive
            self.assertTrue((prices > 0).all())
            
            # All prices should meet minimum threshold
            self.assertTrue((prices >= config.prediction.min_price_threshold).all())
            
            # Prices should be in reasonable range (not extreme outliers)
            self.assertTrue((prices < 10000).all())  # Less than $10,000
            self.assertTrue((prices > 0.001).all())  # Greater than $0.001
            
            # Price distribution should be reasonable
            price_std = prices.std()
            price_mean = prices.mean()
            cv = price_std / price_mean  # Coefficient of variation
            
            # Coefficient of variation should be reasonable (not too high)
            self.assertLess(cv, 5.0)  # CV less than 500%
            
            # Should have some price variation (not all identical)
            unique_prices = len(prices.unique())
            self.assertGreater(unique_prices, 1)
    
    def test_output_file_compliance(self):
        """Test output file compliance with submission requirements."""
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Run prediction pipeline
            results = self.pipeline.run_complete_pipeline()
            
            # Verify output file exists
            output_file = Path(config.output.predictions_file)
            self.assertTrue(output_file.exists())
            
            # Load and validate output format
            output_df = pd.read_csv(output_file)
            
            # Check exact column names and order
            self.assertEqual(list(output_df.columns), ['sample_id', 'price'])
            
            # Check data types
            self.assertTrue(pd.api.types.is_object_dtype(output_df['sample_id']))
            self.assertTrue(pd.api.types.is_numeric_dtype(output_df['price']))
            
            # Check for missing values
            self.assertFalse(output_df.isna().any().any())
            
            # Check sample ID format (should be strings)
            for sample_id in output_df['sample_id']:
                self.assertIsInstance(sample_id, str)
                self.assertTrue(len(sample_id) > 0)
            
            # Check price format (should be positive floats)
            for price in output_df['price']:
                self.assertIsInstance(price, (int, float))
                self.assertGreater(price, 0)
                self.assertTrue(np.isfinite(price))
    
    def test_missing_model_error_handling(self):
        """Test error handling when trained model is missing."""
        # Remove model files
        model_files = list((self.temp_path / "models").glob("*"))
        for file in model_files:
            file.unlink()
        
        # Pipeline should handle missing model gracefully
        with self.assertRaises(Exception) as context:
            self.pipeline.run_complete_pipeline()
        
        # Should be a meaningful error about missing model
        error_message = str(context.exception).lower()
        self.assertTrue(any(keyword in error_message for keyword in ['model', 'not found', 'missing']))
    
    def test_feature_mismatch_error_handling(self):
        """Test error handling when feature dimensions don't match."""
        # Modify feature names to have wrong count
        wrong_feature_names = [f'feature_{i}' for i in range(25)]  # Wrong count
        feature_names_path = self.temp_path / "models" / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(wrong_feature_names, f)
        
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Pipeline should handle feature mismatch
            with self.assertRaises(Exception) as context:
                self.pipeline.run_complete_pipeline()
            
            # Should be a meaningful error about feature mismatch
            error_message = str(context.exception).lower()
            self.assertTrue(any(keyword in error_message for keyword in ['feature', 'dimension', 'mismatch', 'shape']))


if __name__ == '__main__':
    unittest.main()