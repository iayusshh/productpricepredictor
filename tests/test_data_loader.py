"""
Unit tests for Data Loader.

Comprehensive tests covering schema validation, data integrity checks,
error handling, and data loading functionality.
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.data_processing.data_loader import DataLoader, DataValidationError
from src.models.data_models import ProductSample


class TestDataLoader(unittest.TestCase):
    """Test cases for Data Loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.train_file = str(self.temp_path / "train.csv")
        self.mock_config.test_file = str(self.temp_path / "test.csv")
        self.mock_config.zero_price_strategy = "epsilon"
        
        self.loader = DataLoader(data_config=self.mock_config)
        
        # Create test data
        self.valid_train_data = pd.DataFrame({
            'sample_id': ['1', '2', '3', '4', '5'],
            'catalog_content': ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 
                          'http://example.com/3.jpg', 'http://example.com/4.jpg', 
                          'http://example.com/5.jpg'],
            'price': [10.99, 25.50, 0.0, 15.75, 99.99]
        })
        
        self.valid_test_data = pd.DataFrame({
            'sample_id': ['t1', 't2', 't3'],
            'catalog_content': ['Test Product 1', 'Test Product 2', 'Test Product 3'],
            'image_link': ['http://example.com/t1.jpg', 'http://example.com/t2.jpg', 
                          'http://example.com/t3.jpg']
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test data loader initialization."""
        self.assertIsNotNone(self.loader.train_schema)
        self.assertIsNotNone(self.loader.test_schema)
        self.assertEqual(self.loader.config, self.mock_config)
        
        # Check schema definitions
        expected_train_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
        expected_test_cols = ['sample_id', 'catalog_content', 'image_link']
        
        self.assertEqual(set(self.loader.train_schema.keys()), set(expected_train_cols))
        self.assertEqual(set(self.loader.test_schema.keys()), set(expected_test_cols))
    
    def test_load_training_data_success(self):
        """Test successful training data loading."""
        # Create valid training file
        self.valid_train_data.to_csv(self.mock_config.train_file, index=False)
        
        df = self.loader.load_training_data()
        
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ['sample_id', 'catalog_content', 'image_link', 'price'])
        pd.testing.assert_frame_equal(df, self.valid_train_data)
    
    def test_load_test_data_success(self):
        """Test successful test data loading."""
        # Create valid test file
        self.valid_test_data.to_csv(self.mock_config.test_file, index=False)
        
        df = self.loader.load_test_data()
        
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['sample_id', 'catalog_content', 'image_link'])
        pd.testing.assert_frame_equal(df, self.valid_test_data)
    
    def test_load_training_data_file_not_found(self):
        """Test training data loading with missing file."""
        # Don't create the file
        with self.assertRaises(DataValidationError) as context:
            self.loader.load_training_data()
        
        self.assertIn("Training file not found", str(context.exception))
        self.assertIn(self.mock_config.train_file, str(context.exception))
    
    def test_load_test_data_file_not_found(self):
        """Test test data loading with missing file."""
        # Don't create the file
        with self.assertRaises(DataValidationError) as context:
            self.loader.load_test_data()
        
        self.assertIn("Test file not found", str(context.exception))
        self.assertIn(self.mock_config.test_file, str(context.exception))
    
    def test_load_training_data_empty_file(self):
        """Test training data loading with empty file."""
        # Create empty file
        Path(self.mock_config.train_file).touch()
        
        with self.assertRaises(DataValidationError) as context:
            self.loader.load_training_data()
        
        self.assertIn("empty", str(context.exception).lower())
    
    def test_load_training_data_invalid_csv(self):
        """Test training data loading with invalid CSV format."""
        # Create invalid CSV file
        with open(self.mock_config.train_file, 'w') as f:
            f.write("invalid,csv,format\nwith,unclosed,quote\"")
        
        with self.assertRaises(DataValidationError) as context:
            self.loader.load_training_data()
        
        self.assertIn("Failed to parse", str(context.exception))
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        # Create data missing required columns
        invalid_data = pd.DataFrame({
            'sample_id': ['1', '2'],
            'catalog_content': ['Product 1', 'Product 2']
            # Missing image_link and price
        })
        
        with self.assertRaises(DataValidationError) as context:
            self.loader._validate_schema_and_types(invalid_data, self.loader.train_schema, "training")
        
        self.assertIn("Missing required columns", str(context.exception))
        self.assertIn("image_link", str(context.exception))
        self.assertIn("price", str(context.exception))
    
    def test_validate_schema_extra_columns(self):
        """Test schema validation with extra columns (should warn but not fail)."""
        # Create data with extra columns
        data_with_extra = self.valid_train_data.copy()
        data_with_extra['extra_column'] = ['extra1', 'extra2', 'extra3', 'extra4', 'extra5']
        
        # Should not raise exception
        result = self.loader._validate_schema_and_types(data_with_extra, self.loader.train_schema, "training")
        self.assertTrue(result)
    
    def test_validate_schema_empty_dataframe(self):
        """Test schema validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(DataValidationError) as context:
            self.loader._validate_schema_and_types(empty_df, self.loader.train_schema, "training")
        
        self.assertIn("empty", str(context.exception).lower())
    
    def test_validate_schema_wrong_types(self):
        """Test schema validation with wrong column types."""
        # Create data with wrong types
        wrong_type_data = pd.DataFrame({
            'sample_id': [1, 2, 3],  # Should be object/string
            'catalog_content': ['Product 1', 'Product 2', 'Product 3'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 'http://example.com/3.jpg'],
            'price': ['10.99', '25.50', '15.75']  # Should be numeric
        })
        
        # Should attempt to convert types automatically
        result = self.loader._validate_schema_and_types(wrong_type_data, self.loader.train_schema, "training")
        self.assertTrue(result)
        
        # Check that types were converted
        self.assertEqual(str(wrong_type_data['sample_id'].dtype), 'object')
        self.assertTrue(pd.api.types.is_numeric_dtype(wrong_type_data['price']))
    
    def test_validate_data_integrity_duplicate_sample_ids(self):
        """Test data integrity validation with duplicate sample IDs."""
        # Create data with duplicate sample IDs
        duplicate_data = pd.DataFrame({
            'sample_id': ['1', '2', '1', '3'],  # Duplicate '1'
            'catalog_content': ['Product 1', 'Product 2', 'Product 1 duplicate', 'Product 3'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 
                          'http://example.com/1_dup.jpg', 'http://example.com/3.jpg'],
            'price': [10.99, 25.50, 10.99, 15.75]
        })
        
        with self.assertRaises(DataValidationError) as context:
            self.loader._validate_data_integrity(duplicate_data, is_training=True)
        
        self.assertIn("duplicate sample_ids", str(context.exception))
    
    def test_validate_data_integrity_missing_sample_ids(self):
        """Test data integrity validation with missing sample IDs."""
        # Create data with missing sample IDs
        missing_id_data = pd.DataFrame({
            'sample_id': ['1', None, '3'],
            'catalog_content': ['Product 1', 'Product 2', 'Product 3'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 'http://example.com/3.jpg'],
            'price': [10.99, 25.50, 15.75]
        })
        
        with self.assertRaises(DataValidationError) as context:
            self.loader._validate_data_integrity(missing_id_data, is_training=True)
        
        self.assertIn("missing sample_ids", str(context.exception))
    
    def test_validate_data_integrity_negative_prices(self):
        """Test data integrity validation with negative prices."""
        # Create data with negative prices
        negative_price_data = pd.DataFrame({
            'sample_id': ['1', '2', '3'],
            'catalog_content': ['Product 1', 'Product 2', 'Product 3'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 'http://example.com/3.jpg'],
            'price': [10.99, -25.50, 15.75]  # Negative price
        })
        
        with self.assertRaises(DataValidationError) as context:
            self.loader._validate_data_integrity(negative_price_data, is_training=True)
        
        self.assertIn("negative prices", str(context.exception))
    
    def test_validate_data_integrity_warnings_only(self):
        """Test data integrity validation with warning-level issues."""
        # Create data with warning-level issues (missing content, zero prices)
        warning_data = pd.DataFrame({
            'sample_id': ['1', '2', '3', '4'],
            'catalog_content': ['Product 1', None, 'Product 3', 'Product 4'],  # Missing content
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 
                          None, 'http://example.com/4.jpg'],  # Missing image link
            'price': [10.99, 0.0, 15.75, 25.50]  # Zero price
        })
        
        # Should not raise exception (warnings only)
        result = self.loader._validate_data_integrity(warning_data, is_training=True)
        self.assertTrue(result)
    
    def test_validate_data_integrity_test_data(self):
        """Test data integrity validation for test data (no price checks)."""
        # Test data doesn't have price column, so price-related checks should be skipped
        test_data = pd.DataFrame({
            'sample_id': ['t1', 't2', 't3'],
            'catalog_content': ['Test Product 1', None, 'Test Product 3'],
            'image_link': ['http://example.com/t1.jpg', 'http://example.com/t2.jpg', None]
        })
        
        # Should pass (no price-related integrity issues)
        result = self.loader._validate_data_integrity(test_data, is_training=False)
        self.assertTrue(result)
    
    def test_get_data_summary_training(self):
        """Test data summary generation for training data."""
        summary = self.loader.get_data_summary(self.valid_train_data)
        
        # Check basic summary structure
        expected_keys = ['total_samples', 'columns', 'missing_values', 'data_types', 
                        'price_statistics', 'zero_prices', 'negative_prices', 'catalog_content_stats']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check specific values
        self.assertEqual(summary['total_samples'], 5)
        self.assertEqual(summary['zero_prices'], 1)  # One zero price in test data
        self.assertEqual(summary['negative_prices'], 0)
        
        # Check price statistics
        self.assertIn('mean', summary['price_statistics'])
        self.assertIn('std', summary['price_statistics'])
        self.assertIn('min', summary['price_statistics'])
        self.assertIn('max', summary['price_statistics'])
        
        # Check catalog content stats
        self.assertIn('avg_length', summary['catalog_content_stats'])
        self.assertIn('min_length', summary['catalog_content_stats'])
        self.assertIn('max_length', summary['catalog_content_stats'])
    
    def test_get_data_summary_test(self):
        """Test data summary generation for test data."""
        summary = self.loader.get_data_summary(self.valid_test_data)
        
        # Should not have price-related statistics
        self.assertNotIn('price_statistics', summary)
        self.assertNotIn('zero_prices', summary)
        self.assertNotIn('negative_prices', summary)
        
        # Should have basic statistics
        self.assertEqual(summary['total_samples'], 3)
        self.assertIn('catalog_content_stats', summary)
    
    def test_validate_sample_format_valid(self):
        """Test sample format validation with valid sample."""
        valid_sample = {
            'sample_id': '123',
            'catalog_content': 'Test product description',
            'image_link': 'http://example.com/image.jpg',
            'price': 19.99
        }
        
        result = self.loader.validate_sample_format(valid_sample)
        
        self.assertIsInstance(result, ProductSample)
        self.assertEqual(result.sample_id, '123')
        self.assertEqual(result.catalog_content, 'Test product description')
        self.assertEqual(result.image_link, 'http://example.com/image.jpg')
        self.assertEqual(result.price, 19.99)
    
    def test_validate_sample_format_missing_fields(self):
        """Test sample format validation with missing required fields."""
        invalid_sample = {
            'sample_id': '123',
            'catalog_content': 'Test product description'
            # Missing image_link
        }
        
        with self.assertRaises(DataValidationError) as context:
            self.loader.validate_sample_format(invalid_sample)
        
        self.assertIn("Missing required fields", str(context.exception))
        self.assertIn("image_link", str(context.exception))
    
    def test_validate_sample_format_wrong_types(self):
        """Test sample format validation with wrong field types."""
        invalid_samples = [
            # Invalid sample_id type
            {
                'sample_id': ['not_string_or_int'],
                'catalog_content': 'Test product',
                'image_link': 'http://example.com/image.jpg'
            },
            # Invalid catalog_content type
            {
                'sample_id': '123',
                'catalog_content': 123,  # Should be string
                'image_link': 'http://example.com/image.jpg'
            },
            # Invalid image_link type
            {
                'sample_id': '123',
                'catalog_content': 'Test product',
                'image_link': 123  # Should be string
            }
        ]
        
        for invalid_sample in invalid_samples:
            with self.subTest(sample=invalid_sample):
                with self.assertRaises(DataValidationError):
                    self.loader.validate_sample_format(invalid_sample)
    
    def test_validate_sample_format_without_price(self):
        """Test sample format validation without price (test data)."""
        test_sample = {
            'sample_id': '123',
            'catalog_content': 'Test product description',
            'image_link': 'http://example.com/image.jpg'
        }
        
        result = self.loader.validate_sample_format(test_sample)
        
        self.assertIsInstance(result, ProductSample)
        self.assertEqual(result.sample_id, '123')
        self.assertIsNone(result.price)
    
    def test_validate_schema_and_types_public_interface(self):
        """Test public interface for schema validation."""
        # Test with training data (has price column)
        result = self.loader.validate_schema_and_types(self.valid_train_data)
        self.assertTrue(result)
        
        # Test with test data (no price column)
        result = self.loader.validate_schema_and_types(self.valid_test_data)
        self.assertTrue(result)
    
    def test_edge_case_numeric_sample_ids(self):
        """Test handling of numeric sample IDs."""
        numeric_id_data = pd.DataFrame({
            'sample_id': [1, 2, 3],  # Numeric IDs
            'catalog_content': ['Product 1', 'Product 2', 'Product 3'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 'http://example.com/3.jpg'],
            'price': [10.99, 25.50, 15.75]
        })
        
        # Should convert to object type automatically
        result = self.loader._validate_schema_and_types(numeric_id_data, self.loader.train_schema, "training")
        self.assertTrue(result)
        self.assertEqual(str(numeric_id_data['sample_id'].dtype), 'object')
    
    def test_edge_case_string_prices(self):
        """Test handling of string prices that can be converted to numeric."""
        string_price_data = pd.DataFrame({
            'sample_id': ['1', '2', '3'],
            'catalog_content': ['Product 1', 'Product 2', 'Product 3'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 'http://example.com/3.jpg'],
            'price': ['10.99', '25.50', '15.75']  # String prices
        })
        
        # Should convert to numeric automatically
        result = self.loader._validate_schema_and_types(string_price_data, self.loader.train_schema, "training")
        self.assertTrue(result)
        self.assertTrue(pd.api.types.is_numeric_dtype(string_price_data['price']))
    
    def test_edge_case_extreme_price_values(self):
        """Test handling of extreme price values."""
        extreme_price_data = pd.DataFrame({
            'sample_id': ['1', '2', '3', '4'],
            'catalog_content': ['Product 1', 'Product 2', 'Product 3', 'Product 4'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 
                          'http://example.com/3.jpg', 'http://example.com/4.jpg'],
            'price': [0.01, 1000000.0, 0.0, 99999.99]  # Extreme values
        })
        
        # Should pass integrity validation (warnings only for extreme values)
        result = self.loader._validate_data_integrity(extreme_price_data, is_training=True)
        self.assertTrue(result)
    
    def test_full_loading_pipeline_training(self):
        """Test complete training data loading pipeline."""
        # Create valid training file
        self.valid_train_data.to_csv(self.mock_config.train_file, index=False)
        
        # Load data through complete pipeline
        df = self.loader.load_training_data()
        
        # Verify data was loaded correctly
        self.assertEqual(len(df), 5)
        self.assertTrue(all(col in df.columns for col in self.loader.train_schema.keys()))
        
        # Verify data types
        self.assertEqual(str(df['sample_id'].dtype), 'object')
        self.assertTrue(pd.api.types.is_numeric_dtype(df['price']))
    
    def test_full_loading_pipeline_test(self):
        """Test complete test data loading pipeline."""
        # Create valid test file
        self.valid_test_data.to_csv(self.mock_config.test_file, index=False)
        
        # Load data through complete pipeline
        df = self.loader.load_test_data()
        
        # Verify data was loaded correctly
        self.assertEqual(len(df), 3)
        self.assertTrue(all(col in df.columns for col in self.loader.test_schema.keys()))
        
        # Verify data types
        self.assertEqual(str(df['sample_id'].dtype), 'object')


if __name__ == '__main__':
    unittest.main()