"""
High-precision Item Pack Quantity (IPQ) extraction system

This module implements a regex-based IPQ extractor with >90% precision requirement,
including validation test cases and unit normalization capabilities.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class IPQResult:
    """Result of IPQ extraction with confidence and normalization info"""
    raw_text: str
    extracted_value: Optional[str] = None
    normalized_value: Optional[float] = None
    unit: Optional[str] = None
    canonical_unit: Optional[str] = None
    confidence: float = 0.0
    extraction_method: Optional[str] = None


@dataclass
class ValidationCase:
    """Test case for IPQ extraction validation"""
    input_text: str
    expected_value: str
    expected_unit: str
    description: str


class IPQExtractor:
    """
    High-precision Item Pack Quantity extractor with >90% precision requirement.
    
    Implements regex-based extraction with comprehensive validation and unit normalization
    to convert ml/g/pcs to canonical units (grams/pieces).
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize IPQ extractor with precision patterns and unit mappings.
        
        Args:
            logger: Optional logger instance for tracking operations
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # High-precision IPQ extraction patterns (ordered by confidence)
        self.ipq_patterns = [
            # Pattern 1: Explicit pack/quantity statements (highest confidence)
            {
                'pattern': r'(?i)(?:pack\s+of\s+|quantity\s*:?\s*|qty\s*:?\s*)(\d+(?:\.\d+)?)\s*(pcs?|pieces?|units?|count|items?)?',
                'confidence': 0.95,
                'method': 'explicit_pack'
            },
            # Pattern 2: Count with units (high confidence)
            {
                'pattern': r'(?i)(\d+(?:\.\d+)?)\s*(pcs?|pieces?|units?|count|items?)\b',
                'confidence': 0.90,
                'method': 'count_with_units'
            },
            # Pattern 3: Weight/volume with units (high confidence)
            {
                'pattern': r'(?i)(\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?|oz|ounces?|lbs?|pounds?|ml|milliliters?|l|liters?|fl\s*oz)\b',
                'confidence': 0.88,
                'method': 'weight_volume'
            },
            # Pattern 4: Multiplication format (medium-high confidence)
            {
                'pattern': r'(?i)(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?|oz|ounces?|ml|milliliters?|l|liters?|pcs?|pieces?)?',
                'confidence': 0.85,
                'method': 'multiplication'
            },
            # Pattern 5: Size specifications (medium confidence)
            {
                'pattern': r'(?i)(?:size\s*:?\s*|pack\s+size\s*:?\s*)(\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?|oz|ounces?|ml|milliliters?|l|liters?|pcs?|pieces?)?',
                'confidence': 0.80,
                'method': 'size_spec'
            },
            # Pattern 6: Parenthetical quantities (lower confidence)
            {
                'pattern': r'\((\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?|oz|ounces?|ml|milliliters?|l|liters?|pcs?|pieces?)?\)',
                'confidence': 0.75,
                'method': 'parenthetical'
            }
        ]
        
        # Unit normalization mappings to canonical units
        self.unit_mappings = {
            # Weight units to grams
            'g': ('gram', 1.0),
            'gram': ('gram', 1.0),
            'grams': ('gram', 1.0),
            'kg': ('gram', 1000.0),
            'kilogram': ('gram', 1000.0),
            'kilograms': ('gram', 1000.0),
            'oz': ('gram', 28.3495),
            'ounce': ('gram', 28.3495),
            'ounces': ('gram', 28.3495),
            'lb': ('gram', 453.592),
            'lbs': ('gram', 453.592),
            'pound': ('gram', 453.592),
            'pounds': ('gram', 453.592),
            
            # Volume units to milliliters
            'ml': ('milliliter', 1.0),
            'milliliter': ('milliliter', 1.0),
            'milliliters': ('milliliter', 1.0),
            'l': ('milliliter', 1000.0),
            'liter': ('milliliter', 1000.0),
            'liters': ('milliliter', 1000.0),
            'fl oz': ('milliliter', 29.5735),
            'fl_oz': ('milliliter', 29.5735),
            
            # Count units to pieces
            'pc': ('piece', 1.0),
            'pcs': ('piece', 1.0),
            'piece': ('piece', 1.0),
            'pieces': ('piece', 1.0),
            'unit': ('piece', 1.0),
            'units': ('piece', 1.0),
            'count': ('piece', 1.0),
            'item': ('piece', 1.0),
            'items': ('piece', 1.0),
        }
        
        # Validation test cases for precision measurement
        self.validation_cases = self._create_validation_cases()
    
    def _create_validation_cases(self) -> List[ValidationCase]:
        """Create comprehensive validation test cases for precision measurement."""
        return [
            # Explicit pack statements (should have high success rate)
            ValidationCase("Pack of 12", "12", "piece", "Explicit pack statement"),
            ValidationCase("Quantity: 24", "24", "piece", "Quantity with colon"),
            ValidationCase("Qty 6", "6", "piece", "Abbreviated quantity"),
            ValidationCase("Pack of 8 pieces", "8", "piece", "Pack with pieces"),
            
            # Count specifications (should have high success rate)
            ValidationCase("12 pieces", "12", "piece", "Count with pieces"),
            ValidationCase("24 pcs", "24", "piece", "Count with pcs"),
            ValidationCase("6 units", "6", "piece", "Count with units"),
            ValidationCase("10 count", "10", "piece", "Count specification"),
            
            # Weight specifications
            ValidationCase("500g", "500", "gram", "Simple weight in grams"),
            ValidationCase("1kg", "1", "gram", "Weight in kilograms"),
            ValidationCase("750g pack", "750", "gram", "Weight with pack"),
            ValidationCase("2 kg", "2", "gram", "Weight with space"),
            
            # Volume specifications
            ValidationCase("250ml", "250", "milliliter", "Simple volume in ml"),
            ValidationCase("1l", "1", "milliliter", "Volume in liters"),
            ValidationCase("500ml bottle", "500", "milliliter", "Volume with bottle"),
            ValidationCase("2 liters", "2", "milliliter", "Volume with space"),
            
            # Multiplication format
            ValidationCase("6 x 100g", "6", "piece", "Multiplication with weight"),
            ValidationCase("12 x 50ml", "12", "piece", "Multiplication with volume"),
            
            # Size specifications
            ValidationCase("Size: 200g", "200", "gram", "Size specification"),
            ValidationCase("Pack size 12", "12", "piece", "Pack size"),
            
            # Parenthetical quantities
            ValidationCase("Product (100g)", "100", "gram", "Parenthetical weight"),
            ValidationCase("Bottle (250ml)", "250", "milliliter", "Parenthetical volume"),
        ]
    
    def extract_ipq_with_validation(self, catalog_content: str) -> IPQResult:
        """
        Extract Item Pack Quantity with validation and confidence scoring.
        
        Args:
            catalog_content: Raw catalog content text
            
        Returns:
            IPQResult with extracted value, unit, and confidence
        """
        if not catalog_content or pd.isna(catalog_content):
            return IPQResult(raw_text="")
        
        content = str(catalog_content).strip()
        result = IPQResult(raw_text=content)
        
        # Try each pattern in order of confidence
        for pattern_info in self.ipq_patterns:
            match = re.search(pattern_info['pattern'], content)
            if match:
                result.confidence = pattern_info['confidence']
                result.extraction_method = pattern_info['method']
                
                # Handle different pattern types
                if pattern_info['method'] == 'multiplication':
                    # For multiplication, extract count and unit
                    count = float(match.group(1))
                    value = float(match.group(2))
                    unit = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
                    
                    result.extracted_value = str(count)
                    result.unit = 'piece'  # Multiplication always represents count
                    
                    # For multiplication, the count is usually the pack quantity
                    result.normalized_value = count
                    result.canonical_unit = 'piece'
                    
                else:
                    # Standard extraction
                    value = float(match.group(1))
                    unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else 'piece'
                    
                    result.extracted_value = str(value)
                    result.unit = unit.lower().strip() if unit else 'piece'
                    
                    # Normalize to canonical units
                    normalized = self.normalize_units_to_canonical(value, result.unit)
                    result.normalized_value = normalized['value']
                    result.canonical_unit = normalized['unit']
                
                self.logger.debug(f"IPQ extracted: {result.extracted_value} {result.unit} "
                                f"(confidence: {result.confidence:.2f}, method: {result.extraction_method})")
                break
        
        return result
    
    def normalize_units_to_canonical(self, value: float, unit: str) -> Dict[str, Union[float, str]]:
        """
        Normalize units to canonical format (grams/milliliters/pieces).
        
        Args:
            value: Numerical value to normalize
            unit: Unit to normalize
            
        Returns:
            Dictionary with normalized value and canonical unit
        """
        if not unit:
            return {'value': value, 'unit': 'piece'}
        
        unit_clean = unit.lower().strip()
        
        if unit_clean in self.unit_mappings:
            canonical_unit, conversion_factor = self.unit_mappings[unit_clean]
            normalized_value = value * conversion_factor
            return {'value': normalized_value, 'unit': canonical_unit}
        
        # Default to pieces for unknown units
        self.logger.warning(f"Unknown unit '{unit}', defaulting to pieces")
        return {'value': value, 'unit': 'piece'}
    
    def validate_ipq_extraction_precision(self, test_samples: Optional[List[str]] = None) -> float:
        """
        Validate IPQ extraction precision on test samples.
        
        Args:
            test_samples: Optional list of test samples. If None, uses built-in validation cases.
            
        Returns:
            Precision score (0.0 to 1.0)
        """
        if test_samples is None:
            # Use built-in validation cases
            test_cases = self.validation_cases
            total_cases = len(test_cases)
            correct_extractions = 0
            
            self.logger.info(f"Running precision validation on {total_cases} test cases")
            
            for case in test_cases:
                result = self.extract_ipq_with_validation(case.input_text)
                
                # Check if extraction matches expected value and unit type
                if result.extracted_value and result.canonical_unit:
                    expected_canonical = self.normalize_units_to_canonical(
                        float(case.expected_value), case.expected_unit
                    )
                    
                    # For value comparison, be more flexible with extracted vs expected
                    extracted_value = float(result.extracted_value)
                    expected_value = float(case.expected_value)
                    
                    # Allow some tolerance for floating point comparisons
                    value_match = abs(extracted_value - expected_value) < 0.01
                    unit_match = result.canonical_unit == expected_canonical['unit']
                    
                    if value_match and unit_match:
                        correct_extractions += 1
                        self.logger.debug(f"✓ PASS: {case.description}")
                    else:
                        self.logger.debug(f"✗ FAIL: {case.description} - "
                                        f"Expected: {case.expected_value} {expected_canonical['unit']}, "
                                        f"Got: {result.extracted_value} {result.canonical_unit}")
                else:
                    self.logger.debug(f"✗ FAIL: {case.description} - No extraction")
            
            precision = correct_extractions / total_cases
            
        else:
            # Use provided test samples (would need ground truth labels)
            self.logger.warning("Custom test samples provided but no ground truth labels available")
            precision = 0.0
        
        self.logger.info(f"IPQ extraction precision: {precision:.3f} ({correct_extractions}/{total_cases})")
        
        if precision < 0.90:
            self.logger.warning(f"Precision {precision:.3f} is below required 90% threshold!")
        
        return precision
    
    def extract_quantity_features(self, catalog_content: str) -> Dict[str, float]:
        """
        Extract numerical quantity features from catalog content.
        
        Args:
            catalog_content: Raw catalog content
            
        Returns:
            Dictionary of numerical features
        """
        result = self.extract_ipq_with_validation(catalog_content)
        
        features = {
            'has_ipq': 1.0 if result.extracted_value else 0.0,
            'ipq_confidence': result.confidence,
            'ipq_value': result.normalized_value if result.normalized_value else 0.0,
            'is_weight_unit': 1.0 if result.canonical_unit == 'gram' else 0.0,
            'is_volume_unit': 1.0 if result.canonical_unit == 'milliliter' else 0.0,
            'is_count_unit': 1.0 if result.canonical_unit == 'piece' else 0.0,
        }
        
        # Add log-transformed values for better model performance
        if result.normalized_value and result.normalized_value > 0:
            features['ipq_value_log'] = np.log1p(result.normalized_value)
        else:
            features['ipq_value_log'] = 0.0
        
        return features
    
    def batch_extract_ipq(self, df: pd.DataFrame, 
                         content_column: str = 'catalog_content') -> pd.DataFrame:
        """
        Extract IPQ features for entire DataFrame.
        
        Args:
            df: DataFrame containing catalog content
            content_column: Name of the column containing catalog content
            
        Returns:
            DataFrame with additional IPQ feature columns
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        
        self.logger.info(f"Extracting IPQ features for {len(df)} samples")
        
        # Extract features for each row
        ipq_features = []
        extraction_details = []
        
        for idx, row in df.iterrows():
            try:
                # Extract IPQ result
                result = self.extract_ipq_with_validation(row[content_column])
                
                # Get numerical features
                features = self.extract_quantity_features(row[content_column])
                ipq_features.append(features)
                
                # Store extraction details
                extraction_details.append({
                    'ipq_raw_extraction': result.extracted_value,
                    'ipq_unit': result.unit,
                    'ipq_canonical_unit': result.canonical_unit,
                    'ipq_extraction_method': result.extraction_method,
                })
                
            except Exception as e:
                self.logger.error(f"Error extracting IPQ for row {idx}: {e}")
                # Add empty features for failed rows
                ipq_features.append({
                    'has_ipq': 0.0,
                    'ipq_confidence': 0.0,
                    'ipq_value': 0.0,
                    'is_weight_unit': 0.0,
                    'is_volume_unit': 0.0,
                    'is_count_unit': 0.0,
                    'ipq_value_log': 0.0,
                })
                extraction_details.append({
                    'ipq_raw_extraction': None,
                    'ipq_unit': None,
                    'ipq_canonical_unit': None,
                    'ipq_extraction_method': None,
                })
        
        # Create DataFrames from features and details
        features_df = pd.DataFrame(ipq_features)
        details_df = pd.DataFrame(extraction_details)
        
        # Combine with original DataFrame
        result_df = pd.concat([df, features_df, details_df], axis=1)
        
        # Log extraction statistics
        successful_extractions = features_df['has_ipq'].sum()
        extraction_rate = successful_extractions / len(df)
        
        self.logger.info(f"IPQ extraction completed. Success rate: {extraction_rate:.3f} "
                        f"({successful_extractions}/{len(df)})")
        
        return result_df