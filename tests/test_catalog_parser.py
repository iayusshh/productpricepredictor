"""
Unit tests for Catalog Parser.

Comprehensive tests covering structured information extraction,
brand identification, category detection, and specification parsing.
"""

import unittest
import numpy as np
import pandas as pd
import logging
from unittest.mock import Mock

from src.features.catalog_parser import CatalogParser, ProductSpecification


class TestCatalogParser(unittest.TestCase):
    """Test cases for Catalog Parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create logger for testing
        self.logger = Mock(spec=logging.Logger)
        self.parser = CatalogParser(logger=self.logger)
    
    def test_initialization(self):
        """Test catalog parser initialization."""
        self.assertIsNotNone(self.parser.dimension_patterns)
        self.assertIsNotNone(self.parser.material_patterns)
        self.assertIsNotNone(self.parser.color_patterns)
        self.assertIsNotNone(self.parser.brand_patterns)
        self.assertIsNotNone(self.parser.category_patterns)
        self.assertIsNotNone(self.parser.specification_patterns)
        self.assertIsNotNone(self.parser.dimension_units)
        self.assertIsNotNone(self.parser.weight_units)
        
        # Check that patterns are properly initialized
        self.assertGreater(len(self.parser.dimension_patterns), 0)
        self.assertGreater(len(self.parser.material_patterns), 0)
        self.assertGreater(len(self.parser.color_patterns), 0)
        self.assertGreater(len(self.parser.brand_patterns), 0)
        self.assertGreater(len(self.parser.category_patterns), 0)
        self.assertGreater(len(self.parser.specification_patterns), 0)
    
    def test_parse_item_pack_quantity_explicit_pack(self):
        """Test IPQ parsing with explicit pack statements."""
        test_cases = [
            ("Pack of 12", 12.0, True),
            ("Quantity: 24", 24.0, True),
            ("Qty 6", 6.0, True),
            ("pack of 8 items", 8.0, True),
        ]
        
        for text, expected_value, expected_extracted in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse_item_pack_quantity(text)
                self.assertEqual(result['ipq_extracted'], expected_extracted)
                self.assertEqual(result['ipq_value'], expected_value)
                self.assertGreater(result['ipq_confidence'], 0.8)
                self.assertIsNotNone(result['ipq_method'])
    
    def test_parse_item_pack_quantity_count_units(self):
        """Test IPQ parsing with count and units."""
        test_cases = [
            ("12 pieces", 12.0, True),
            ("24 pcs", 24.0, True),
            ("6 units", 6.0, True),
            ("10 count", 10.0, True),
            ("8 items", 8.0, True),
        ]
        
        for text, expected_value, expected_extracted in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse_item_pack_quantity(text)
                self.assertEqual(result['ipq_extracted'], expected_extracted)
                self.assertEqual(result['ipq_value'], expected_value)
                self.assertGreater(result['ipq_confidence'], 0.7)
    
    def test_parse_item_pack_quantity_multiplication(self):
        """Test IPQ parsing with multiplication format."""
        test_cases = [
            ("6 x 100g", 6.0, True),
            ("12 x 50ml", 12.0, True),
            ("4 x 250g", 4.0, True),
        ]
        
        for text, expected_value, expected_extracted in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse_item_pack_quantity(text)
                self.assertEqual(result['ipq_extracted'], expected_extracted)
                self.assertEqual(result['ipq_value'], expected_value)
                self.assertGreater(result['ipq_confidence'], 0.6)
    
    def test_parse_item_pack_quantity_empty_input(self):
        """Test IPQ parsing with empty or invalid input."""
        test_cases = [None, "", "   ", pd.NA, "No quantity here"]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse_item_pack_quantity(text)
                self.assertFalse(result['ipq_extracted'])
                self.assertEqual(result['ipq_value'], 0.0)
                self.assertEqual(result['ipq_confidence'], 0.0)
                self.assertIsNone(result['ipq_method'])
    
    def test_extract_dimensions_lwh(self):
        """Test dimension extraction with length x width x height."""
        test_cases = [
            ("10cm x 5cm x 2cm", {'length': 0.1, 'width': 0.05, 'height': 0.02}),
            ("12 inches x 8 inches x 4 inches", {'length': 0.3048, 'width': 0.2032, 'height': 0.1016}),
            ("1m x 0.5m x 0.3m", {'length': 1.0, 'width': 0.5, 'height': 0.3}),
        ]
        
        for text, expected_dims in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_dimensions(text)
                self.assertTrue(result['has_dimensions'])
                for dim, expected_value in expected_dims.items():
                    self.assertAlmostEqual(result[dim], expected_value, places=3)
                
                # Check that volume is calculated
                if 'length' in result and 'width' in result and 'height' in result:
                    expected_volume = result['length'] * result['width'] * result['height']
                    self.assertAlmostEqual(result['volume'], expected_volume, places=6)
    
    def test_extract_dimensions_lw(self):
        """Test dimension extraction with length x width."""
        test_cases = [
            ("20cm x 15cm", {'length': 0.2, 'width': 0.15}),
            ("8 inches x 6 inches", {'length': 0.2032, 'width': 0.1524}),
        ]
        
        for text, expected_dims in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_dimensions(text)
                self.assertTrue(result['has_dimensions'])
                for dim, expected_value in expected_dims.items():
                    self.assertAlmostEqual(result[dim], expected_value, places=3)
                
                # Check that area is calculated
                if 'length' in result and 'width' in result:
                    expected_area = result['length'] * result['width']
                    self.assertAlmostEqual(result['area'], expected_area, places=6)
    
    def test_extract_dimensions_diameter(self):
        """Test dimension extraction with diameter."""
        test_cases = [
            ("Diameter: 10cm", {'diameter': 0.1, 'radius': 0.05}),
            ("Dia. 5 inches", {'diameter': 0.127, 'radius': 0.0635}),
        ]
        
        for text, expected_dims in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_dimensions(text)
                self.assertTrue(result['has_dimensions'])
                for dim, expected_value in expected_dims.items():
                    self.assertAlmostEqual(result[dim], expected_value, places=3)
                
                # Check that area is calculated for diameter
                if 'radius' in result:
                    expected_area = 3.14159 * result['radius'] * result['radius']
                    self.assertAlmostEqual(result['area'], expected_area, places=3)
    
    def test_extract_dimensions_no_match(self):
        """Test dimension extraction with no matching patterns."""
        test_cases = ["No dimensions here", "Just text", "", None]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_dimensions(text)
                self.assertFalse(result['has_dimensions'])
    
    def test_extract_materials(self):
        """Test material extraction."""
        test_cases = [
            ("Made of cotton and polyester", ['cotton', 'polyester']),
            ("Stainless steel construction", ['stainless steel']),
            ("Plastic and metal components", ['plastic', 'metal']),
            ("Organic cotton with bamboo fibers", ['organic', 'cotton', 'bamboo']),
            ("No materials mentioned", []),
        ]
        
        for text, expected_materials in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_materials(text)
                # Convert to sets for comparison (order doesn't matter)
                self.assertEqual(set(result), set(expected_materials))
    
    def test_extract_colors(self):
        """Test color extraction."""
        test_cases = [
            ("Available in red and blue", ['red', 'blue']),
            ("Dark green with light gray accents", ['dark green', 'light gray']),
            ("Navy blue color", ['navy']),
            ("Black, white, and silver options", ['black', 'white', 'silver']),
            ("No colors mentioned", []),
        ]
        
        for text, expected_colors in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_colors(text)
                # Check that expected colors are found (may find additional ones)
                for color in expected_colors:
                    self.assertIn(color, result)
    
    def test_identify_brand(self):
        """Test brand identification."""
        test_cases = [
            ("Apple iPhone 12", "apple", "technology", 0.9),
            ("Nike Air Max shoes", "nike", "sportswear", 0.9),
            ("Samsung Galaxy tablet", "samsung", "technology", 0.9),
            ("Gucci handbag", "gucci", "luxury_fashion", 0.9),
            ("Unknown brand product", None, None, 0.0),
        ]
        
        for text, expected_brand, expected_category, expected_confidence in test_cases:
            with self.subTest(text=text):
                result = self.parser.identify_brand(text)
                self.assertEqual(result['brand'], expected_brand)
                self.assertEqual(result['brand_category'], expected_category)
                self.assertEqual(result['brand_confidence'], expected_confidence)
    
    def test_identify_category(self):
        """Test category identification."""
        test_cases = [
            ("Smartphone with camera", "electronics", 0.9),
            ("Cotton t-shirt", "clothing", 0.85),
            ("Organic snack food", "food_beverage", 0.8),
            ("Children's toy car", "toys", 0.9),
            ("Office chair", "furniture", 0.85),
            ("Unknown product type", None, 0.0),
        ]
        
        for text, expected_category, expected_confidence in test_cases:
            with self.subTest(text=text):
                result = self.parser.identify_category(text)
                self.assertEqual(result['category'], expected_category)
                self.assertEqual(result['category_confidence'], expected_confidence)
    
    def test_extract_specifications_weight(self):
        """Test weight specification extraction."""
        test_cases = [
            ("Weight: 500g", 500.0, "gram"),
            ("Wt. 2kg", 2000.0, "gram"),  # Converted to grams
            ("Weight 16 oz", 453.592, "gram"),  # Converted to grams
        ]
        
        for text, expected_value, expected_unit in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_specifications(text)
                self.assertIn('weight', result)
                self.assertAlmostEqual(result['weight']['value'], expected_value, places=2)
                self.assertEqual(result['weight']['unit'], expected_unit)
    
    def test_extract_specifications_volume(self):
        """Test volume specification extraction."""
        test_cases = [
            ("Capacity: 250ml", 250.0),
            ("Volume 2 liters", 2.0),
            ("Size: 16 fl oz", 16.0),
        ]
        
        for text, expected_value in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_specifications(text)
                self.assertIn('volume', result)
                self.assertEqual(result['volume']['value'], expected_value)
    
    def test_extract_specifications_power(self):
        """Test power specification extraction."""
        test_cases = [
            ("Power: 100W", 100.0),
            ("Wattage 1500 watts", 1500.0),
            ("Power consumption 2kW", 2.0),
        ]
        
        for text, expected_value in test_cases:
            with self.subTest(text=text):
                result = self.parser.extract_specifications(text)
                self.assertIn('power', result)
                self.assertEqual(result['power']['value'], expected_value)
    
    def test_parse_catalog_content_comprehensive(self):
        """Test comprehensive catalog content parsing."""
        catalog_text = (
            "Apple iPhone 12 smartphone, Pack of 1, "
            "Dimensions: 14.7cm x 7.15cm x 0.74cm, "
            "Weight: 164g, Made of aluminum and glass, "
            "Available in blue and black colors, "
            "Storage: 128GB, Power: 15W wireless charging"
        )
        
        result = self.parser.parse_catalog_content(catalog_text)
        
        # Check that result is ProductSpecification
        self.assertIsInstance(result, ProductSpecification)
        
        # Check brand identification
        self.assertEqual(result.brand, "apple")
        
        # Check category identification
        self.assertEqual(result.category, "electronics")
        
        # Check materials
        self.assertIn("aluminum", result.materials)
        self.assertIn("glass", result.materials)
        
        # Check colors
        self.assertIn("blue", result.colors)
        self.assertIn("black", result.colors)
        
        # Check dimensions
        self.assertIsNotNone(result.dimensions)
        self.assertTrue(result.dimensions['has_dimensions'])
        
        # Check specifications
        self.assertIsNotNone(result.specifications)
        self.assertIn('weight', result.specifications)
        self.assertIn('power', result.specifications)
        
        # Check numerical features
        self.assertIsNotNone(result.numerical_features)
        self.assertEqual(result.numerical_features['has_brand'], 1.0)
        self.assertEqual(result.numerical_features['has_category'], 1.0)
        self.assertEqual(result.numerical_features['has_dimensions'], 1.0)
        self.assertGreater(result.numerical_features['material_count'], 0)
        self.assertGreater(result.numerical_features['color_count'], 0)
    
    def test_parse_catalog_content_empty(self):
        """Test parsing with empty catalog content."""
        test_cases = [None, "", "   ", pd.NA]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse_catalog_content(text)
                self.assertIsInstance(result, ProductSpecification)
                self.assertIsNone(result.brand)
                self.assertIsNone(result.category)
                self.assertIsNone(result.materials)
                self.assertIsNone(result.colors)
                self.assertIsNone(result.dimensions)
                self.assertIsNone(result.specifications)
                self.assertIsNone(result.numerical_features)
    
    def test_numerical_features_calculation(self):
        """Test numerical features calculation."""
        catalog_text = (
            "Pack of 6 items, 10cm x 5cm x 2cm, "
            "Made of plastic and metal, red and blue colors, "
            "Apple brand, smartphone category, Weight: 100g"
        )
        
        result = self.parser.parse_catalog_content(catalog_text)
        features = result.numerical_features
        
        # Check IPQ features
        self.assertEqual(features['has_ipq'], 1.0)
        self.assertEqual(features['ipq_value'], 6.0)
        self.assertGreater(features['ipq_confidence'], 0.0)
        
        # Check dimension features
        self.assertEqual(features['has_dimensions'], 1.0)
        self.assertGreater(features['length'], 0.0)
        self.assertGreater(features['width'], 0.0)
        self.assertGreater(features['height'], 0.0)
        self.assertGreater(features['volume'], 0.0)
        
        # Check material and color features
        self.assertEqual(features['has_materials'], 1.0)
        self.assertEqual(features['material_count'], 2.0)
        self.assertEqual(features['has_colors'], 1.0)
        self.assertEqual(features['color_count'], 2.0)
        
        # Check brand and category features
        self.assertEqual(features['has_brand'], 1.0)
        self.assertGreater(features['brand_confidence'], 0.0)
        self.assertEqual(features['has_category'], 1.0)
        self.assertGreater(features['category_confidence'], 0.0)
        
        # Check specification features
        self.assertGreater(features['spec_count'], 0.0)
        self.assertEqual(features['has_weight'], 1.0)
        self.assertGreater(features['weight_value'], 0.0)
    
    def test_batch_parse_catalog(self):
        """Test batch parsing of catalog content."""
        test_data = {
            'sample_id': ['1', '2', '3', '4'],
            'catalog_content': [
                'Apple iPhone, Pack of 1, 100g',
                'Nike shoes, red color, 12 pieces',
                'No specific information',
                'Samsung tablet, 10cm x 8cm, plastic'
            ]
        }
        df = pd.DataFrame(test_data)
        
        result_df = self.parser.batch_parse_catalog(df)
        
        # Check that original columns are preserved
        for col in df.columns:
            self.assertIn(col, result_df.columns)
        
        # Check that numerical feature columns are added
        expected_features = ['has_ipq', 'ipq_value', 'has_brand', 'has_category', 
                           'has_dimensions', 'material_count', 'color_count']
        for feature in expected_features:
            self.assertIn(feature, result_df.columns)
        
        # Check specific parsing results
        self.assertEqual(result_df.loc[0, 'has_brand'], 1.0)  # Apple
        self.assertEqual(result_df.loc[1, 'has_colors'], 1.0)  # red color
        self.assertEqual(result_df.loc[1, 'ipq_value'], 12.0)  # 12 pieces
        self.assertEqual(result_df.loc[3, 'has_dimensions'], 1.0)  # 10cm x 8cm
    
    def test_batch_parse_missing_column(self):
        """Test batch parsing with missing column."""
        df = pd.DataFrame({'sample_id': ['1', '2'], 'other_col': ['a', 'b']})
        
        with self.assertRaises(ValueError):
            self.parser.batch_parse_catalog(df)
    
    def test_batch_parse_custom_column(self):
        """Test batch parsing with custom content column."""
        df = pd.DataFrame({
            'sample_id': ['1', '2'],
            'custom_content': ['Apple iPhone', 'Nike shoes']
        })
        
        result_df = self.parser.batch_parse_catalog(df, content_column='custom_content')
        
        self.assertEqual(result_df.loc[0, 'has_brand'], 1.0)
        self.assertEqual(result_df.loc[1, 'has_brand'], 1.0)
    
    def test_unit_conversion_dimensions(self):
        """Test unit conversion for dimensions."""
        # Test various unit conversions
        test_cases = [
            ('mm', 0.001),
            ('cm', 0.01),
            ('m', 1.0),
            ('in', 0.0254),
            ('inch', 0.0254),
            ('ft', 0.3048),
        ]
        
        for unit, expected_factor in test_cases:
            with self.subTest(unit=unit):
                self.assertIn(unit, self.parser.dimension_units)
                self.assertAlmostEqual(self.parser.dimension_units[unit], expected_factor, places=4)
    
    def test_unit_conversion_weight(self):
        """Test unit conversion for weight."""
        test_cases = [
            ('g', 1.0),
            ('kg', 1000.0),
            ('oz', 28.3495),
            ('lb', 453.592),
        ]
        
        for unit, expected_factor in test_cases:
            with self.subTest(unit=unit):
                self.assertIn(unit, self.parser.weight_units)
                self.assertAlmostEqual(self.parser.weight_units[unit], expected_factor, places=3)
    
    def test_edge_case_mixed_case_text(self):
        """Test parsing with mixed case text."""
        catalog_text = "APPLE IPHONE 12, PACK OF 1, RED COLOR, 100G WEIGHT"
        
        result = self.parser.parse_catalog_content(catalog_text)
        
        # Should still identify brand, IPQ, color, and weight
        self.assertEqual(result.brand, "apple")
        self.assertIn("red", result.colors)
        self.assertEqual(result.numerical_features['has_ipq'], 1.0)
        self.assertEqual(result.numerical_features['has_weight'], 1.0)
    
    def test_edge_case_multiple_patterns(self):
        """Test parsing when multiple patterns match."""
        catalog_text = "Pack of 12, quantity: 6, 24 pieces"
        
        result = self.parser.parse_catalog_content(catalog_text)
        
        # Should extract the first/highest confidence match
        self.assertEqual(result.numerical_features['has_ipq'], 1.0)
        self.assertEqual(result.numerical_features['ipq_value'], 12.0)  # First pattern
    
    def test_product_specification_dataclass(self):
        """Test ProductSpecification dataclass functionality."""
        spec = ProductSpecification(
            dimensions={'length': 1.0, 'width': 0.5},
            materials=['plastic', 'metal'],
            colors=['red', 'blue'],
            brand='test_brand',
            category='test_category',
            specifications={'weight': {'value': 100.0, 'unit': 'gram'}},
            numerical_features={'has_brand': 1.0, 'brand_confidence': 0.9}
        )
        
        self.assertEqual(spec.dimensions['length'], 1.0)
        self.assertEqual(spec.materials, ['plastic', 'metal'])
        self.assertEqual(spec.colors, ['red', 'blue'])
        self.assertEqual(spec.brand, 'test_brand')
        self.assertEqual(spec.category, 'test_category')
        self.assertEqual(spec.specifications['weight']['value'], 100.0)
        self.assertEqual(spec.numerical_features['has_brand'], 1.0)
    
    def test_empty_numerical_features(self):
        """Test empty numerical features generation."""
        empty_features = self.parser._get_empty_numerical_features()
        
        # Check that all expected keys are present with zero values
        expected_keys = ['ipq_value', 'ipq_confidence', 'has_ipq', 'has_dimensions', 
                        'length', 'width', 'height', 'volume', 'area', 'diameter',
                        'material_count', 'color_count', 'has_materials', 'has_colors',
                        'has_brand', 'brand_confidence', 'has_category', 'category_confidence',
                        'spec_count', 'has_weight', 'has_volume_spec', 'has_power']
        
        for key in expected_keys:
            self.assertIn(key, empty_features)
            self.assertEqual(empty_features[key], 0.0)
    
    def test_logging_integration(self):
        """Test that logging is properly integrated."""
        # Test info logging during batch parsing
        df = pd.DataFrame({
            'sample_id': ['1'],
            'catalog_content': ['Apple iPhone']
        })
        
        self.parser.batch_parse_catalog(df)
        
        # Should have called logger.info at least twice (start and end)
        self.assertGreaterEqual(self.logger.info.call_count, 2)


if __name__ == '__main__':
    unittest.main()