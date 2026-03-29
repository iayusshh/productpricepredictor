"""
Unit tests for IPQ Extractor.

Comprehensive tests covering IPQ extraction precision validation, edge cases,
unit normalization, and batch processing functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import logging
from unittest.mock import Mock, patch

from src.features.ipq_extractor import IPQExtractor, IPQResult, ValidationCase


class TestIPQExtractor(unittest.TestCase):
    """Test cases for IPQ Extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create logger for testing
        self.logger = Mock(spec=logging.Logger)
        self.extractor = IPQExtractor(logger=self.logger)
    
    def test_initialization(self):
        """Test IPQ extractor initialization."""
        self.assertIsNotNone(self.extractor.ipq_patterns)
        self.assertIsNotNone(self.extractor.unit_mappings)
        self.assertIsNotNone(self.extractor.validation_cases)
        self.assertGreater(len(self.extractor.ipq_patterns), 0)
        self.assertGreater(len(self.extractor.unit_mappings), 0)
        self.assertGreater(len(self.extractor.validation_cases), 0)
    
    def test_explicit_pack_extraction(self):
        """Test extraction of explicit pack statements."""
        test_cases = [
            ("Pack of 12", "12", "piece"),
            ("Quantity: 24", "24", "piece"),
            ("Qty 6", "6", "piece"),
            ("Pack of 8 pieces", "8", "piece"),
        ]
        
        for text, expected_value, expected_unit in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_ipq_with_validation(text)
                self.assertEqual(result.extracted_value, expected_value)
                self.assertEqual(result.canonical_unit, expected_unit)
                self.assertGreater(result.confidence, 0.9)
    
    def test_count_with_units_extraction(self):
        """Test extraction of count with units."""
        test_cases = [
            ("12 pieces", "12", "piece"),
            ("24 pcs", "24", "piece"),
            ("6 units", "6", "piece"),
            ("10 count", "10", "piece"),
        ]
        
        for text, expected_value, expected_unit in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_ipq_with_validation(text)
                self.assertEqual(result.extracted_value, expected_value)
                self.assertEqual(result.canonical_unit, expected_unit)
                self.assertGreater(result.confidence, 0.85)
    
    def test_weight_volume_extraction(self):
        """Test extraction of weight and volume specifications."""
        test_cases = [
            ("500g", "500", "gram"),
            ("1kg", "1", "gram"),
            ("250ml", "250", "milliliter"),
            ("2 liters", "2", "milliliter"),
            ("16 oz", "16", "gram"),
        ]
        
        for text, expected_value, expected_unit in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_ipq_with_validation(text)
                self.assertEqual(result.extracted_value, expected_value)
                self.assertEqual(result.canonical_unit, expected_unit)
                self.assertGreater(result.confidence, 0.8)
    
    def test_multiplication_format_extraction(self):
        """Test extraction of multiplication format."""
        test_cases = [
            ("6 x 100g", "6", "piece"),
            ("12 x 50ml", "12", "piece"),
            ("4 x 250g", "4", "piece"),
        ]
        
        for text, expected_value, expected_unit in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_ipq_with_validation(text)
                self.assertEqual(result.extracted_value, expected_value)
                self.assertEqual(result.canonical_unit, expected_unit)
                self.assertGreater(result.confidence, 0.8)
    
    def test_unit_normalization(self):
        """Test unit normalization to canonical units."""
        test_cases = [
            # Weight units to grams
            (1.0, "kg", 1000.0, "gram"),
            (500.0, "g", 500.0, "gram"),
            (16.0, "oz", 453.592, "gram"),  # 16 oz ≈ 453.592g
            (1.0, "lb", 453.592, "gram"),
            
            # Volume units to milliliters
            (1.0, "l", 1000.0, "milliliter"),
            (250.0, "ml", 250.0, "milliliter"),
            (1.0, "fl oz", 29.5735, "milliliter"),
            
            # Count units to pieces
            (12.0, "pcs", 12.0, "piece"),
            (6.0, "units", 6.0, "piece"),
            (24.0, "count", 24.0, "piece"),
        ]
        
        for value, unit, expected_value, expected_unit in test_cases:
            with self.subTest(value=value, unit=unit):
                result = self.extractor.normalize_units_to_canonical(value, unit)
                self.assertAlmostEqual(result['value'], expected_value, places=2)
                self.assertEqual(result['unit'], expected_unit)
    
    def test_unknown_unit_handling(self):
        """Test handling of unknown units."""
        result = self.extractor.normalize_units_to_canonical(10.0, "unknown_unit")
        self.assertEqual(result['value'], 10.0)
        self.assertEqual(result['unit'], "piece")
        
        # Check that warning was logged
        self.logger.warning.assert_called()
    
    def test_empty_input_handling(self):
        """Test handling of empty or invalid inputs."""
        test_cases = [None, "", "   ", pd.NA]
        
        for test_input in test_cases:
            with self.subTest(input=test_input):
                result = self.extractor.extract_ipq_with_validation(test_input)
                self.assertEqual(result.raw_text, "" if test_input is None else "")
                self.assertIsNone(result.extracted_value)
                self.assertEqual(result.confidence, 0.0)
    
    def test_no_match_handling(self):
        """Test handling when no IPQ pattern matches."""
        test_cases = [
            "This is just regular text",
            "No quantities here",
            "Random product description",
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_ipq_with_validation(text)
                self.assertEqual(result.raw_text, text)
                self.assertIsNone(result.extracted_value)
                self.assertEqual(result.confidence, 0.0)
    
    def test_precision_validation_builtin_cases(self):
        """Test precision validation using built-in test cases."""
        precision = self.extractor.validate_ipq_extraction_precision()
        
        # Should achieve >90% precision on built-in validation cases
        self.assertGreaterEqual(precision, 0.90)
        self.assertLessEqual(precision, 1.0)
    
    def test_precision_validation_custom_cases(self):
        """Test precision validation with custom test samples."""
        # Test with None (should use built-in cases)
        precision = self.extractor.validate_ipq_extraction_precision(None)
        self.assertGreaterEqual(precision, 0.90)
        
        # Test with custom samples (should return 0.0 due to no ground truth)
        custom_samples = ["Pack of 6", "12 pieces", "500g"]
        precision = self.extractor.validate_ipq_extraction_precision(custom_samples)
        self.assertEqual(precision, 0.0)
        
        # Check that warning was logged
        self.logger.warning.assert_called()
    
    def test_extract_quantity_features(self):
        """Test extraction of numerical quantity features."""
        test_cases = [
            ("Pack of 12", {
                'has_ipq': 1.0,
                'ipq_value': 12.0,
                'is_count_unit': 1.0,
                'is_weight_unit': 0.0,
                'is_volume_unit': 0.0,
            }),
            ("500g", {
                'has_ipq': 1.0,
                'ipq_value': 500.0,
                'is_count_unit': 0.0,
                'is_weight_unit': 1.0,
                'is_volume_unit': 0.0,
            }),
            ("250ml", {
                'has_ipq': 1.0,
                'ipq_value': 250.0,
                'is_count_unit': 0.0,
                'is_weight_unit': 0.0,
                'is_volume_unit': 1.0,
            }),
            ("No quantity", {
                'has_ipq': 0.0,
                'ipq_value': 0.0,
                'is_count_unit': 0.0,
                'is_weight_unit': 0.0,
                'is_volume_unit': 0.0,
            }),
        ]
        
        for text, expected_features in test_cases:
            with self.subTest(text=text):
                features = self.extractor.extract_quantity_features(text)
                
                for key, expected_value in expected_features.items():
                    self.assertIn(key, features)
                    self.assertAlmostEqual(features[key], expected_value, places=2)
                
                # Check that all expected keys are present
                expected_keys = ['has_ipq', 'ipq_confidence', 'ipq_value', 
                               'is_weight_unit', 'is_volume_unit', 'is_count_unit', 'ipq_value_log']
                for key in expected_keys:
                    self.assertIn(key, features)
                
                # Check log transformation
                if features['ipq_value'] > 0:
                    expected_log = np.log1p(features['ipq_value'])
                    self.assertAlmostEqual(features['ipq_value_log'], expected_log, places=6)
                else:
                    self.assertEqual(features['ipq_value_log'], 0.0)
    
    def test_batch_extract_ipq(self):
        """Test batch IPQ extraction on DataFrame."""
        # Create test DataFrame
        test_data = {
            'sample_id': ['1', '2', '3', '4', '5'],
            'catalog_content': [
                'Pack of 12 items',
                '500g weight',
                '250ml volume',
                'No quantity here',
                '6 x 100g'
            ]
        }
        df = pd.DataFrame(test_data)
        
        # Extract IPQ features
        result_df = self.extractor.batch_extract_ipq(df)
        
        # Check that original columns are preserved
        for col in df.columns:
            self.assertIn(col, result_df.columns)
        
        # Check that new feature columns are added
        expected_feature_cols = ['has_ipq', 'ipq_confidence', 'ipq_value', 
                               'is_weight_unit', 'is_volume_unit', 'is_count_unit', 'ipq_value_log']
        for col in expected_feature_cols:
            self.assertIn(col, result_df.columns)
        
        # Check that detail columns are added
        expected_detail_cols = ['ipq_raw_extraction', 'ipq_unit', 
                              'ipq_canonical_unit', 'ipq_extraction_method']
        for col in expected_detail_cols:
            self.assertIn(col, result_df.columns)
        
        # Check specific extractions
        self.assertEqual(result_df.loc[0, 'has_ipq'], 1.0)  # Pack of 12
        self.assertEqual(result_df.loc[0, 'ipq_raw_extraction'], '12')
        self.assertEqual(result_df.loc[0, 'is_count_unit'], 1.0)
        
        self.assertEqual(result_df.loc[1, 'has_ipq'], 1.0)  # 500g
        self.assertEqual(result_df.loc[1, 'ipq_raw_extraction'], '500')
        self.assertEqual(result_df.loc[1, 'is_weight_unit'], 1.0)
        
        self.assertEqual(result_df.loc[3, 'has_ipq'], 0.0)  # No quantity
        self.assertIsNone(result_df.loc[3, 'ipq_raw_extraction'])
    
    def test_batch_extract_missing_column(self):
        """Test batch extraction with missing column."""
        df = pd.DataFrame({'sample_id': ['1', '2'], 'other_col': ['a', 'b']})
        
        with self.assertRaises(ValueError):
            self.extractor.batch_extract_ipq(df)
    
    def test_batch_extract_custom_column(self):
        """Test batch extraction with custom content column."""
        df = pd.DataFrame({
            'sample_id': ['1', '2'],
            'custom_content': ['Pack of 6', '250ml']
        })
        
        result_df = self.extractor.batch_extract_ipq(df, content_column='custom_content')
        
        self.assertEqual(result_df.loc[0, 'has_ipq'], 1.0)
        self.assertEqual(result_df.loc[1, 'has_ipq'], 1.0)
    
    def test_batch_extract_error_handling(self):
        """Test batch extraction error handling."""
        # Create DataFrame with problematic data
        df = pd.DataFrame({
            'sample_id': ['1', '2', '3'],
            'catalog_content': ['Pack of 6', None, 'Valid content']
        })
        
        # Mock an error in extract_ipq_with_validation for one row
        original_method = self.extractor.extract_ipq_with_validation
        def mock_extract(content):
            if content == 'Valid content':
                raise ValueError("Test error")
            return original_method(content)
        
        with patch.object(self.extractor, 'extract_ipq_with_validation', side_effect=mock_extract):
            result_df = self.extractor.batch_extract_ipq(df)
        
        # Should still return results for all rows
        self.assertEqual(len(result_df), 3)
        
        # Error row should have zero features
        self.assertEqual(result_df.loc[2, 'has_ipq'], 0.0)
        self.assertIsNone(result_df.loc[2, 'ipq_raw_extraction'])
    
    def test_edge_case_decimal_values(self):
        """Test extraction with decimal values."""
        test_cases = [
            ("1.5kg", "1.5", "gram"),
            ("2.25 liters", "2.25", "milliliter"),
            ("0.5 oz", "0.5", "gram"),
        ]
        
        for text, expected_value, expected_unit in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_ipq_with_validation(text)
                self.assertEqual(result.extracted_value, expected_value)
                self.assertEqual(result.canonical_unit, expected_unit)
    
    def test_edge_case_mixed_case(self):
        """Test extraction with mixed case text."""
        test_cases = [
            ("PACK OF 12", "12", "piece"),
            ("500G", "500", "gram"),
            ("Pack Of 6 Pieces", "6", "piece"),
        ]
        
        for text, expected_value, expected_unit in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_ipq_with_validation(text)
                self.assertEqual(result.extracted_value, expected_value)
                self.assertEqual(result.canonical_unit, expected_unit)
    
    def test_edge_case_multiple_quantities(self):
        """Test extraction when multiple quantities are present."""
        # Should extract the first/highest confidence match
        text = "Pack of 12 pieces, each 100g"
        result = self.extractor.extract_ipq_with_validation(text)
        
        # Should extract the pack quantity (higher confidence pattern)
        self.assertEqual(result.extracted_value, "12")
        self.assertEqual(result.canonical_unit, "piece")
        self.assertGreater(result.confidence, 0.9)
    
    def test_confidence_scoring(self):
        """Test that confidence scores are assigned correctly."""
        # Higher confidence patterns should have higher scores
        high_confidence_text = "Pack of 12"
        low_confidence_text = "(12g)"
        
        high_result = self.extractor.extract_ipq_with_validation(high_confidence_text)
        low_result = self.extractor.extract_ipq_with_validation(low_confidence_text)
        
        self.assertGreater(high_result.confidence, low_result.confidence)
        self.assertGreater(high_result.confidence, 0.9)
        self.assertLess(low_result.confidence, 0.8)
    
    def test_extraction_method_tracking(self):
        """Test that extraction methods are tracked correctly."""
        test_cases = [
            ("Pack of 12", "explicit_pack"),
            ("12 pieces", "count_with_units"),
            ("500g", "weight_volume"),
            ("6 x 100g", "multiplication"),
            ("Size: 250ml", "size_spec"),
            ("(100g)", "parenthetical"),
        ]
        
        for text, expected_method in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_ipq_with_validation(text)
                self.assertEqual(result.extraction_method, expected_method)
    
    def test_validation_case_structure(self):
        """Test that validation cases have correct structure."""
        for case in self.extractor.validation_cases:
            self.assertIsInstance(case, ValidationCase)
            self.assertIsInstance(case.input_text, str)
            self.assertIsInstance(case.expected_value, str)
            self.assertIsInstance(case.expected_unit, str)
            self.assertIsInstance(case.description, str)
            
            # Check that expected values are valid
            try:
                float(case.expected_value)
            except ValueError:
                self.fail(f"Invalid expected value in validation case: {case.expected_value}")
    
    def test_ipq_result_dataclass(self):
        """Test IPQResult dataclass functionality."""
        result = IPQResult(
            raw_text="test text",
            extracted_value="12",
            normalized_value=12.0,
            unit="pcs",
            canonical_unit="piece",
            confidence=0.95,
            extraction_method="test_method"
        )
        
        self.assertEqual(result.raw_text, "test text")
        self.assertEqual(result.extracted_value, "12")
        self.assertEqual(result.normalized_value, 12.0)
        self.assertEqual(result.unit, "pcs")
        self.assertEqual(result.canonical_unit, "piece")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.extraction_method, "test_method")
    
    def test_logging_integration(self):
        """Test that logging is properly integrated."""
        # Test debug logging during extraction
        self.extractor.extract_ipq_with_validation("Pack of 12")
        self.logger.debug.assert_called()
        
        # Test warning logging for unknown units
        self.extractor.normalize_units_to_canonical(10.0, "unknown")
        self.logger.warning.assert_called()
        
        # Test info logging during precision validation
        self.extractor.validate_ipq_extraction_precision()
        self.logger.info.assert_called()


if __name__ == '__main__':
    unittest.main()