"""
Integration tests for Training Pipeline.

End-to-end integration tests for the complete training pipeline including
data loading, preprocessing, feature engineering, model training, and evaluation.
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from src.training_pipeline import TrainingPipeline
from src.config import config


class TestTrainingPipelineIntegration(unittest.TestCase):
    """Integration test cases for Training Pipeline."""
    
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
        
        # Create test data files
        self.create_test_data_files()
        
        # Mock config
        self.original_config = config
        self.mock_config()
        
        # Initialize pipeline
        self.pipeline = TrainingPipeline()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original config
        config.__dict__.update(self.original_config.__dict__)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data_files(self):
        """Create test data files."""
        # Training data
        train_data = pd.DataFrame({
            'sample_id': [f'train_{i}' for i in range(1, 101)],  # 100 samples
            'catalog_content': [
                f'Product {i} description with pack of {i%5+1} items, '
                f'made of {"plastic" if i%2 else "metal"}, '
                f'{"red" if i%3 else "blue"} color, '
                f'weight: {10+i}g, '
                f'dimensions: {i}cm x {i+1}cm x {i+2}cm'
                for i in range(1, 101)
            ],
            'image_link': [f'https://example.com/image_{i}.jpg' for i in range(1, 101)],
            'price': np.random.uniform(5.0, 100.0, 100)  # Random prices
        })
        train_data.to_csv(self.temp_path / "dataset" / "train.csv", index=False)
        
        # Test data
        test_data = pd.DataFrame({
            'sample_id': [f'test_{i}' for i in range(1, 51)],  # 50 samples
            'catalog_content': [
                f'Test Product {i} with pack of {i%3+1} pieces, '
                f'{"aluminum" if i%2 else "steel"} material, '
                f'{"green" if i%3 else "yellow"} finish'
                for i in range(1, 51)
            ],
            'image_link': [f'https://example.com/test_image_{i}.jpg' for i in range(1, 51)]
        })
        test_data.to_csv(self.temp_path / "dataset" / "test.csv", index=False)
    
    def mock_config(self):
        """Mock configuration for testing."""
        # Data paths
        config.data.train_file = str(self.temp_path / "dataset" / "train.csv")
        config.data.test_file = str(self.temp_path / "dataset" / "test.csv")
        config.data.image_dir = str(self.temp_path / "images")
        config.data.cache_dir = str(self.temp_path / "cache")
        
        # Model paths
        config.model.model_dir = str(self.temp_path / "models")
        config.model.embedding_dir = str(self.temp_path / "embeddings")
        
        # Logging
        config.logging.log_dir = str(self.temp_path / "logs")
        
        # Training settings (reduced for testing)
        config.training.cv_folds = 3
        config.training.max_epochs = 2
        config.training.early_stopping_patience = 1
        
        # Feature settings
        config.features.max_text_features = 100
        config.features.max_image_features = 50
        
        # Data processing
        config.data.zero_price_strategy = "epsilon"
        config.data.zero_price_epsilon = 0.01
        config.data.download_timeout = 5
        config.data.max_download_retries = 1
        config.data.batch_size = 10
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_complete_training_pipeline(self, mock_get):
        """Test complete training pipeline from data loading to model evaluation."""
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
            self.assertIn('training_completed', results)
            self.assertTrue(results['training_completed'])
            
            # Verify key results are present
            expected_keys = ['data_summary', 'feature_summary', 'model_results', 
                           'evaluation_results', 'pipeline_metrics']
            for key in expected_keys:
                self.assertIn(key, results)
            
            # Verify data was processed
            self.assertGreater(results['data_summary']['total_samples'], 0)
            
            # Verify features were extracted
            self.assertGreater(results['feature_summary']['total_features'], 0)
            
            # Verify model was trained
            self.assertIn('best_model', results['model_results'])
            self.assertIn('cv_scores', results['model_results'])
            
            # Verify evaluation was performed
            self.assertIn('smape', results['evaluation_results'])
            self.assertGreater(results['evaluation_results']['smape'], 0)
            
        except Exception as e:
            self.fail(f"Complete training pipeline failed: {str(e)}")
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_data_loading_and_preprocessing_integration(self, mock_get):
        """Test integration of data loading and preprocessing components."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Load and preprocess data
        train_df, test_df = self.pipeline.load_and_preprocess_data()
        
        # Verify data loading
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertEqual(len(train_df), 100)
        self.assertEqual(len(test_df), 50)
        
        # Verify required columns are present
        required_train_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
        required_test_cols = ['sample_id', 'catalog_content', 'image_link']
        
        for col in required_train_cols:
            self.assertIn(col, train_df.columns)
        
        for col in required_test_cols:
            self.assertIn(col, test_df.columns)
        
        # Verify data preprocessing
        self.assertTrue(pd.api.types.is_numeric_dtype(train_df['price']))
        self.assertTrue((train_df['price'] >= 0).all())  # No negative prices
        
        # Verify no duplicate sample IDs
        self.assertEqual(len(train_df['sample_id'].unique()), len(train_df))
        self.assertEqual(len(test_df['sample_id'].unique()), len(test_df))
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_feature_engineering_integration(self, mock_get):
        """Test integration of feature engineering components."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Load data
        train_df, test_df = self.pipeline.load_and_preprocess_data()
        
        # Extract features
        X_train, X_test, feature_names = self.pipeline.extract_and_fuse_features(train_df, test_df)
        
        # Verify feature extraction
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(feature_names, list)
        
        # Verify feature dimensions
        self.assertEqual(X_train.shape[0], len(train_df))
        self.assertEqual(X_test.shape[0], len(test_df))
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(X_train.shape[1], len(feature_names))
        
        # Verify features are numeric
        self.assertTrue(np.isfinite(X_train).all())
        self.assertTrue(np.isfinite(X_test).all())
        
        # Verify feature names are meaningful
        self.assertGreater(len(feature_names), 0)
        for name in feature_names:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_model_training_integration(self, mock_get):
        """Test integration of model training components."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Prepare data and features
        train_df, test_df = self.pipeline.load_and_preprocess_data()
        X_train, X_test, feature_names = self.pipeline.extract_and_fuse_features(train_df, test_df)
        y_train = train_df['price'].values
        
        # Train models
        model_results = self.pipeline.train_and_validate_models(X_train, y_train, feature_names)
        
        # Verify model training results
        self.assertIsInstance(model_results, dict)
        self.assertIn('best_model', model_results)
        self.assertIn('best_model_name', model_results)
        self.assertIn('cv_scores', model_results)
        self.assertIn('model_performance', model_results)
        
        # Verify cross-validation scores
        cv_scores = model_results['cv_scores']
        self.assertIsInstance(cv_scores, dict)
        self.assertGreater(len(cv_scores), 0)
        
        for model_name, scores in cv_scores.items():
            self.assertIsInstance(scores, list)
            self.assertEqual(len(scores), config.training.cv_folds)
            for score in scores:
                self.assertIsInstance(score, (int, float))
                self.assertGreaterEqual(score, 0)  # SMAPE should be non-negative
        
        # Verify best model selection
        self.assertIsNotNone(model_results['best_model'])
        self.assertIn(model_results['best_model_name'], cv_scores.keys())
    
    @patch('src.data_processing.image_downloader.requests.Session.get')
    def test_evaluation_integration(self, mock_get):
        """Test integration of evaluation components."""
        # Mock image downloads
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Run pipeline up to model training
        train_df, test_df = self.pipeline.load_and_preprocess_data()
        X_train, X_test, feature_names = self.pipeline.extract_and_fuse_features(train_df, test_df)
        y_train = train_df['price'].values
        model_results = self.pipeline.train_and_validate_models(X_train, y_train, feature_names)
        
        # Evaluate model
        evaluation_results = self.pipeline.evaluate_model_performance(
            model_results['best_model'], X_train, y_train, feature_names
        )
        
        # Verify evaluation results
        self.assertIsInstance(evaluation_results, dict)
        
        # Check for key evaluation metrics
        expected_metrics = ['smape', 'mae', 'rmse', 'r2']
        for metric in expected_metrics:
            self.assertIn(metric, evaluation_results)
            self.assertIsInstance(evaluation_results[metric], (int, float))
        
        # Verify SMAPE is reasonable
        smape = evaluation_results['smape']
        self.assertGreaterEqual(smape, 0)
        self.assertLessEqual(smape, 200)  # SMAPE max is 200%
        
        # Check for additional evaluation components
        if 'baseline_comparison' in evaluation_results:
            self.assertIsInstance(evaluation_results['baseline_comparison'], dict)
        
        if 'feature_importance' in evaluation_results:
            self.assertIsInstance(evaluation_results['feature_importance'], (list, dict))
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid data."""
        # Create invalid training data (missing required columns)
        invalid_train_data = pd.DataFrame({
            'sample_id': ['1', '2'],
            'catalog_content': ['Product 1', 'Product 2']
            # Missing image_link and price columns
        })
        invalid_train_data.to_csv(self.temp_path / "dataset" / "train.csv", index=False)
        
        # Pipeline should handle the error gracefully
        with self.assertRaises(Exception) as context:
            self.pipeline.run_complete_pipeline()
        
        # Should be a meaningful error message
        error_message = str(context.exception)
        self.assertTrue(len(error_message) > 0)
    
    def test_pipeline_reproducibility(self):
        """Test that pipeline produces reproducible results."""
        # Mock image downloads for both runs
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Set fixed random seed
            np.random.seed(42)
            
            # Run pipeline twice
            results1 = self.pipeline.run_complete_pipeline()
            
            # Reset and run again
            np.random.seed(42)
            results2 = self.pipeline.run_complete_pipeline()
            
            # Compare key results (allowing for small numerical differences)
            self.assertEqual(results1['data_summary']['total_samples'], 
                           results2['data_summary']['total_samples'])
            
            # Feature counts should be identical
            self.assertEqual(results1['feature_summary']['total_features'], 
                           results2['feature_summary']['total_features'])
            
            # Model performance should be similar (within tolerance)
            smape1 = results1['evaluation_results']['smape']
            smape2 = results2['evaluation_results']['smape']
            self.assertAlmostEqual(smape1, smape2, delta=1.0)  # Within 1% SMAPE
    
    def test_pipeline_performance_benchmarking(self):
        """Test pipeline performance and timing."""
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
            
            # Verify pipeline completed in reasonable time (adjust threshold as needed)
            self.assertLess(execution_time, 300)  # Should complete within 5 minutes
            
            # Verify performance metrics are recorded
            if 'pipeline_metrics' in results:
                metrics = results['pipeline_metrics']
                self.assertIn('total_execution_time', metrics)
                self.assertGreater(metrics['total_execution_time'], 0)
    
    def test_data_quality_validation(self):
        """Test data quality validation throughout pipeline."""
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Load and preprocess data
            train_df, test_df = self.pipeline.load_and_preprocess_data()
            
            # Validate data quality
            self.assertFalse(train_df.empty)
            self.assertFalse(test_df.empty)
            
            # Check for data integrity
            self.assertFalse(train_df['sample_id'].duplicated().any())
            self.assertFalse(test_df['sample_id'].duplicated().any())
            
            # Check price validity
            self.assertTrue((train_df['price'] >= 0).all())
            self.assertFalse(train_df['price'].isna().any())
            
            # Check content completeness
            content_completeness = train_df['catalog_content'].notna().mean()
            self.assertGreater(content_completeness, 0.8)  # At least 80% complete
    
    def test_feature_engineering_validation(self):
        """Test feature engineering validation and quality."""
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Extract features
            train_df, test_df = self.pipeline.load_and_preprocess_data()
            X_train, X_test, feature_names = self.pipeline.extract_and_fuse_features(train_df, test_df)
            
            # Validate feature quality
            self.assertFalse(np.isnan(X_train).any())
            self.assertFalse(np.isnan(X_test).any())
            self.assertFalse(np.isinf(X_train).any())
            self.assertFalse(np.isinf(X_test).any())
            
            # Check feature variance (features should have some variance)
            feature_variances = np.var(X_train, axis=0)
            non_zero_variance_count = np.sum(feature_variances > 1e-10)
            self.assertGreater(non_zero_variance_count, len(feature_names) * 0.5)  # At least 50% should have variance
            
            # Check feature scaling (should be reasonable range)
            feature_means = np.mean(X_train, axis=0)
            feature_stds = np.std(X_train, axis=0)
            
            # Most features should be in reasonable range after preprocessing
            reasonable_range_count = np.sum((np.abs(feature_means) < 100) & (feature_stds < 100))
            self.assertGreater(reasonable_range_count, len(feature_names) * 0.7)  # At least 70%
    
    def test_model_validation_regression_testing(self):
        """Test model validation and regression testing."""
        with patch('src.data_processing.image_downloader.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg'}
            mock_response.iter_content.return_value = [b'fake image data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Run complete pipeline
            results = self.pipeline.run_complete_pipeline()
            
            # Validate model performance is reasonable
            smape = results['evaluation_results']['smape']
            
            # SMAPE should be better than naive baseline (mean prediction)
            # For random data, mean baseline typically gives ~67% SMAPE
            self.assertLess(smape, 80)  # Should be better than 80% SMAPE
            
            # R² should be positive (better than mean prediction)
            if 'r2' in results['evaluation_results']:
                r2 = results['evaluation_results']['r2']
                self.assertGreater(r2, 0)
            
            # MAE should be reasonable relative to price range
            if 'mae' in results['evaluation_results']:
                mae = results['evaluation_results']['mae']
                price_range = results['data_summary'].get('price_range', 100)
                relative_mae = mae / price_range
                self.assertLess(relative_mae, 0.5)  # MAE should be < 50% of price range


if __name__ == '__main__':
    unittest.main()