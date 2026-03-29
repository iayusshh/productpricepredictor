"""
Catalog Parser for structured information extraction

This module implements comprehensive parsing of product catalog content to extract
structured information including specifications, dimensions, materials, and brand identification.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict


@dataclass
class ProductSpecification:
    """Structured product specification data"""
    dimensions: Optional[Dict[str, float]] = None
    materials: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    specifications: Optional[Dict[str, Any]] = None
    numerical_features: Optional[Dict[str, float]] = None


class CatalogParser:
    """
    Comprehensive catalog parser for structured information extraction.
    
    Parses Item Pack Quantity, extracts product specifications, dimensions,
    materials, and implements brand name and category identification.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize catalog parser with extraction patterns.
        
        Args:
            logger: Optional logger instance for tracking operations
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize extraction patterns
        self.dimension_patterns = self._initialize_dimension_patterns()
        self.material_patterns = self._initialize_material_patterns()
        self.color_patterns = self._initialize_color_patterns()
        self.brand_patterns = self._initialize_brand_patterns()
        self.category_patterns = self._initialize_category_patterns()
        self.specification_patterns = self._initialize_specification_patterns()
        
        # Unit conversion mappings
        self.dimension_units = {
            'mm': 0.001,  # to meters
            'cm': 0.01,
            'm': 1.0,
            'in': 0.0254,
            'inch': 0.0254,
            'inches': 0.0254,
            'ft': 0.3048,
            'feet': 0.3048,
        }
        
        # Weight unit conversions (to grams)
        self.weight_units = {
            'g': 1.0,
            'gram': 1.0,
            'grams': 1.0,
            'kg': 1000.0,
            'kilogram': 1000.0,
            'kilograms': 1000.0,
            'oz': 28.3495,
            'ounce': 28.3495,
            'ounces': 28.3495,
            'lb': 453.592,
            'lbs': 453.592,
            'pound': 453.592,
            'pounds': 453.592,
        }
    
    def _initialize_dimension_patterns(self) -> List[Dict[str, Any]]:
        """Initialize dimension extraction patterns."""
        return [
            # Length x Width x Height patterns
            {
                'pattern': r'(?i)(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches|ft|feet)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches|ft|feet)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches|ft|feet)',
                'type': 'lwh',
                'groups': [1, 3, 5],  # length, width, height values
                'units': [2, 4, 6]    # corresponding units
            },
            # Length x Width patterns
            {
                'pattern': r'(?i)(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches|ft|feet)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches|ft|feet)',
                'type': 'lw',
                'groups': [1, 3],
                'units': [2, 4]
            },
            # Single dimension with explicit labels
            {
                'pattern': r'(?i)(?:length|width|height|depth|diameter)\s*:?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches|ft|feet)',
                'type': 'single',
                'groups': [1],
                'units': [2]
            },
            # Diameter patterns
            {
                'pattern': r'(?i)(?:diameter|dia\.?)\s*:?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)',
                'type': 'diameter',
                'groups': [1],
                'units': [2]
            },
        ]
    
    def _initialize_material_patterns(self) -> List[str]:
        """Initialize material detection patterns."""
        return [
            r'\b(?:cotton|polyester|wool|silk|linen|nylon|spandex|elastane)\b',
            r'\b(?:plastic|metal|steel|aluminum|wood|glass|ceramic|rubber)\b',
            r'\b(?:leather|suede|canvas|denim|fleece|velvet|satin)\b',
            r'\b(?:stainless steel|carbon fiber|titanium|brass|copper)\b',
            r'\b(?:organic|bamboo|hemp|recycled|eco-friendly)\b',
        ]
    
    def _initialize_color_patterns(self) -> List[str]:
        """Initialize color detection patterns."""
        return [
            r'\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey)\b',
            r'\b(?:navy|maroon|burgundy|teal|turquoise|magenta|cyan|lime|olive)\b',
            r'\b(?:beige|tan|cream|ivory|silver|gold|bronze|copper)\b',
            r'\b(?:dark|light|bright|pale|deep|vivid)\s+(?:red|blue|green|yellow|orange|purple|pink|brown|gray|grey)\b',
        ]
    
    def _initialize_brand_patterns(self) -> List[Dict[str, Any]]:
        """Initialize brand detection patterns with categories."""
        return [
            # Technology brands
            {'pattern': r'\b(apple|samsung|sony|lg|microsoft|google|amazon|dell|hp|lenovo|asus|acer)\b', 'category': 'technology'},
            {'pattern': r'\b(intel|amd|nvidia|qualcomm|broadcom|cisco|oracle)\b', 'category': 'technology'},
            
            # Fashion and apparel brands
            {'pattern': r'\b(nike|adidas|puma|under armour|reebok|new balance|converse)\b', 'category': 'sportswear'},
            {'pattern': r'\b(gucci|prada|louis vuitton|chanel|hermes|versace|armani)\b', 'category': 'luxury_fashion'},
            {'pattern': r'\b(zara|h&m|uniqlo|gap|old navy|target|walmart)\b', 'category': 'retail_fashion'},
            
            # Consumer goods brands
            {'pattern': r'\b(coca-cola|pepsi|nestle|unilever|procter|johnson|kraft|general mills)\b', 'category': 'consumer_goods'},
            {'pattern': r'\b(tide|dove|pantene|head shoulders|gillette|oral-b)\b', 'category': 'personal_care'},
            
            # Automotive brands
            {'pattern': r'\b(toyota|ford|bmw|mercedes|honda|volkswagen|audi|nissan|hyundai|kia)\b', 'category': 'automotive'},
            
            # Home and furniture brands
            {'pattern': r'\b(ikea|home depot|lowes|wayfair|pottery barn|west elm)\b', 'category': 'home_furniture'},
        ]
    
    def _initialize_category_patterns(self) -> List[Dict[str, Any]]:
        """Initialize product category detection patterns."""
        return [
            {'pattern': r'\b(?:phone|smartphone|tablet|laptop|computer|electronics?|gadget)\b', 'category': 'electronics', 'confidence': 0.9},
            {'pattern': r'\b(?:shirt|pants|dress|shoes|clothing|apparel|fashion|wear)\b', 'category': 'clothing', 'confidence': 0.85},
            {'pattern': r'\b(?:food|snack|beverage|drink|grocery|organic|nutrition)\b', 'category': 'food_beverage', 'confidence': 0.8},
            {'pattern': r'\b(?:book|magazine|novel|textbook|literature|reading)\b', 'category': 'books', 'confidence': 0.9},
            {'pattern': r'\b(?:toy|game|puzzle|doll|action figure|playset)\b', 'category': 'toys', 'confidence': 0.9},
            {'pattern': r'\b(?:furniture|chair|table|sofa|bed|home|decor)\b', 'category': 'furniture', 'confidence': 0.85},
            {'pattern': r'\b(?:beauty|cosmetic|skincare|makeup|shampoo|lotion)\b', 'category': 'beauty', 'confidence': 0.85},
            {'pattern': r'\b(?:tool|hardware|drill|hammer|screwdriver|wrench)\b', 'category': 'tools', 'confidence': 0.9},
            {'pattern': r'\b(?:car|auto|vehicle|tire|engine|automotive|motor)\b', 'category': 'automotive', 'confidence': 0.85},
            {'pattern': r'\b(?:health|vitamin|supplement|medicine|medical|wellness)\b', 'category': 'health', 'confidence': 0.8},
            {'pattern': r'\b(?:sport|fitness|exercise|gym|workout|athletic)\b', 'category': 'sports', 'confidence': 0.8},
            {'pattern': r'\b(?:kitchen|cooking|cookware|appliance|utensil)\b', 'category': 'kitchen', 'confidence': 0.85},
        ]
    
    def _initialize_specification_patterns(self) -> List[Dict[str, Any]]:
        """Initialize specification extraction patterns."""
        return [
            # Weight specifications
            {'pattern': r'(?i)(?:weight|wt\.?)\s*:?\s*(\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?|oz|ounces?|lbs?|pounds?)', 'type': 'weight'},
            
            # Capacity/Volume specifications
            {'pattern': r'(?i)(?:capacity|volume|size)\s*:?\s*(\d+(?:\.\d+)?)\s*(ml|milliliters?|l|liters?|fl\s*oz|cups?|pints?|quarts?|gallons?)', 'type': 'volume'},
            
            # Power specifications
            {'pattern': r'(?i)(?:power|wattage|watts?)\s*:?\s*(\d+(?:\.\d+)?)\s*(w|watts?|kw|kilowatts?)', 'type': 'power'},
            
            # Voltage specifications
            {'pattern': r'(?i)(?:voltage|volts?)\s*:?\s*(\d+(?:\.\d+)?)\s*(v|volts?|kv)', 'type': 'voltage'},
            
            # Temperature specifications
            {'pattern': r'(?i)(?:temperature|temp\.?)\s*:?\s*(\d+(?:\.\d+)?)\s*(°?[cf]|celsius|fahrenheit|degrees?)', 'type': 'temperature'},
            
            # Speed specifications
            {'pattern': r'(?i)(?:speed|rpm)\s*:?\s*(\d+(?:\.\d+)?)\s*(rpm|mph|kmh|km/h)', 'type': 'speed'},
            
            # Memory/Storage specifications
            {'pattern': r'(?i)(?:memory|ram|storage)\s*:?\s*(\d+(?:\.\d+)?)\s*(gb|mb|tb|gigabytes?|megabytes?|terabytes?)', 'type': 'memory'},
        ]
    
    def parse_item_pack_quantity(self, catalog_content: str) -> Dict[str, Any]:
        """
        Parse Item Pack Quantity and convert to numerical features with validation.
        
        Args:
            catalog_content: Raw catalog content
            
        Returns:
            Dictionary with IPQ information and numerical features
        """
        if not catalog_content or pd.isna(catalog_content):
            return self._empty_ipq_result()
        
        content = str(catalog_content).lower()
        
        # IPQ extraction patterns (ordered by precision)
        ipq_patterns = [
            r'(?:pack\s+of\s+|quantity\s*:?\s*|qty\s*:?\s*)(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:pcs?|pieces?|units?|count|items?)\b',
            r'(\d+)\s*x\s*(\d+(?:\.\d+)?)',  # multiplication format
            r'(?:size\s*:?\s*|pack\s+size\s*:?\s*)(\d+(?:\.\d+)?)',
        ]
        
        ipq_result = {
            'ipq_extracted': False,
            'ipq_value': 0.0,
            'ipq_confidence': 0.0,
            'ipq_method': None,
        }
        
        for i, pattern in enumerate(ipq_patterns):
            match = re.search(pattern, content)
            if match:
                if len(match.groups()) == 2:  # multiplication format
                    value = float(match.group(1))  # Use count, not individual size
                else:
                    value = float(match.group(1))
                
                ipq_result.update({
                    'ipq_extracted': True,
                    'ipq_value': value,
                    'ipq_confidence': 1.0 - (i * 0.1),  # Higher confidence for earlier patterns
                    'ipq_method': f'pattern_{i+1}',
                })
                break
        
        return ipq_result
    
    def extract_dimensions(self, catalog_content: str) -> Dict[str, Any]:
        """Extract product dimensions from catalog content."""
        if not catalog_content or pd.isna(catalog_content):
            return {'has_dimensions': False}
        
        content = str(catalog_content).lower()
        dimensions = {}
        
        for pattern_info in self.dimension_patterns:
            match = re.search(pattern_info['pattern'], content)
            if match:
                values = []
                units = []
                
                for group_idx in pattern_info['groups']:
                    values.append(float(match.group(group_idx)))
                
                for unit_idx in pattern_info['units']:
                    unit = match.group(unit_idx).lower()
                    units.append(unit)
                
                # Convert to standard units (meters)
                converted_values = []
                for value, unit in zip(values, units):
                    if unit in self.dimension_units:
                        converted_values.append(value * self.dimension_units[unit])
                    else:
                        converted_values.append(value)
                
                # Store dimensions based on pattern type
                if pattern_info['type'] == 'lwh':
                    dimensions.update({
                        'length': converted_values[0],
                        'width': converted_values[1],
                        'height': converted_values[2],
                        'volume': converted_values[0] * converted_values[1] * converted_values[2],
                    })
                elif pattern_info['type'] == 'lw':
                    dimensions.update({
                        'length': converted_values[0],
                        'width': converted_values[1],
                        'area': converted_values[0] * converted_values[1],
                    })
                elif pattern_info['type'] == 'diameter':
                    radius = converted_values[0] / 2
                    dimensions.update({
                        'diameter': converted_values[0],
                        'radius': radius,
                        'area': 3.14159 * radius * radius,
                    })
                
                break
        
        dimensions['has_dimensions'] = len(dimensions) > 1  # More than just 'has_dimensions'
        return dimensions
    
    def extract_materials(self, catalog_content: str) -> List[str]:
        """Extract materials from catalog content."""
        if not catalog_content or pd.isna(catalog_content):
            return []
        
        content = str(catalog_content).lower()
        materials = []
        
        for pattern in self.material_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            materials.extend(matches)
        
        # Remove duplicates and return
        return list(set(materials))
    
    def extract_colors(self, catalog_content: str) -> List[str]:
        """Extract colors from catalog content."""
        if not catalog_content or pd.isna(catalog_content):
            return []
        
        content = str(catalog_content).lower()
        colors = []
        
        for pattern in self.color_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            colors.extend(matches)
        
        # Remove duplicates and return
        return list(set(colors))
    
    def identify_brand(self, catalog_content: str) -> Dict[str, Any]:
        """Identify brand name and category using pattern matching."""
        if not catalog_content or pd.isna(catalog_content):
            return {'brand': None, 'brand_category': None, 'brand_confidence': 0.0}
        
        content = str(catalog_content).lower()
        
        for brand_info in self.brand_patterns:
            match = re.search(brand_info['pattern'], content, re.IGNORECASE)
            if match:
                return {
                    'brand': match.group(1),
                    'brand_category': brand_info['category'],
                    'brand_confidence': 0.9,  # High confidence for exact matches
                }
        
        return {'brand': None, 'brand_category': None, 'brand_confidence': 0.0}
    
    def identify_category(self, catalog_content: str) -> Dict[str, Any]:
        """Identify product category using pattern matching."""
        if not catalog_content or pd.isna(catalog_content):
            return {'category': None, 'category_confidence': 0.0}
        
        content = str(catalog_content).lower()
        best_match = {'category': None, 'category_confidence': 0.0}
        
        for category_info in self.category_patterns:
            if re.search(category_info['pattern'], content, re.IGNORECASE):
                if category_info['confidence'] > best_match['category_confidence']:
                    best_match = {
                        'category': category_info['category'],
                        'category_confidence': category_info['confidence'],
                    }
        
        return best_match
    
    def extract_specifications(self, catalog_content: str) -> Dict[str, Any]:
        """Extract product specifications like weight, power, etc."""
        if not catalog_content or pd.isna(catalog_content):
            return {}
        
        content = str(catalog_content).lower()
        specifications = {}
        
        for spec_info in self.specification_patterns:
            match = re.search(spec_info['pattern'], content)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower() if len(match.groups()) > 1 else ''
                
                # Convert to standard units
                if spec_info['type'] == 'weight' and unit in self.weight_units:
                    value = value * self.weight_units[unit]
                    unit = 'gram'
                
                specifications[spec_info['type']] = {
                    'value': value,
                    'unit': unit,
                    'raw_text': match.group(0),
                }
        
        return specifications
    
    def parse_catalog_content(self, catalog_content: str) -> ProductSpecification:
        """
        Comprehensive parsing of catalog content.
        
        Args:
            catalog_content: Raw catalog content
            
        Returns:
            ProductSpecification object with all extracted information
        """
        if not catalog_content or pd.isna(catalog_content):
            return ProductSpecification()
        
        # Extract all components
        dimensions = self.extract_dimensions(catalog_content)
        materials = self.extract_materials(catalog_content)
        colors = self.extract_colors(catalog_content)
        brand_info = self.identify_brand(catalog_content)
        category_info = self.identify_category(catalog_content)
        specifications = self.extract_specifications(catalog_content)
        ipq_info = self.parse_item_pack_quantity(catalog_content)
        
        # Create numerical features
        numerical_features = {
            # IPQ features
            'ipq_value': ipq_info['ipq_value'],
            'ipq_confidence': ipq_info['ipq_confidence'],
            'has_ipq': 1.0 if ipq_info['ipq_extracted'] else 0.0,
            
            # Dimension features
            'has_dimensions': 1.0 if dimensions.get('has_dimensions', False) else 0.0,
            'length': dimensions.get('length', 0.0),
            'width': dimensions.get('width', 0.0),
            'height': dimensions.get('height', 0.0),
            'volume': dimensions.get('volume', 0.0),
            'area': dimensions.get('area', 0.0),
            'diameter': dimensions.get('diameter', 0.0),
            
            # Material and color features
            'material_count': len(materials),
            'color_count': len(colors),
            'has_materials': 1.0 if materials else 0.0,
            'has_colors': 1.0 if colors else 0.0,
            
            # Brand and category features
            'has_brand': 1.0 if brand_info['brand'] else 0.0,
            'brand_confidence': brand_info['brand_confidence'],
            'has_category': 1.0 if category_info['category'] else 0.0,
            'category_confidence': category_info['category_confidence'],
            
            # Specification features
            'spec_count': len(specifications),
            'has_weight': 1.0 if 'weight' in specifications else 0.0,
            'has_volume_spec': 1.0 if 'volume' in specifications else 0.0,
            'has_power': 1.0 if 'power' in specifications else 0.0,
        }
        
        # Add specification values
        for spec_type, spec_data in specifications.items():
            numerical_features[f'{spec_type}_value'] = spec_data['value']
        
        return ProductSpecification(
            dimensions=dimensions if dimensions.get('has_dimensions') else None,
            materials=materials if materials else None,
            colors=colors if colors else None,
            brand=brand_info['brand'],
            category=category_info['category'],
            specifications=specifications if specifications else None,
            numerical_features=numerical_features,
        )
    
    def _empty_ipq_result(self) -> Dict[str, Any]:
        """Return empty IPQ result for missing content."""
        return {
            'ipq_extracted': False,
            'ipq_value': 0.0,
            'ipq_confidence': 0.0,
            'ipq_method': None,
        }
    
    def batch_parse_catalog(self, df: pd.DataFrame, 
                          content_column: str = 'catalog_content') -> pd.DataFrame:
        """
        Parse catalog content for entire DataFrame.
        
        Args:
            df: DataFrame containing catalog content
            content_column: Name of the column containing catalog content
            
        Returns:
            DataFrame with additional parsed feature columns
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        
        self.logger.info(f"Parsing catalog content for {len(df)} samples")
        
        # Parse each row
        parsed_features = []
        
        for idx, row in df.iterrows():
            try:
                spec = self.parse_catalog_content(row[content_column])
                parsed_features.append(spec.numerical_features)
                
            except Exception as e:
                self.logger.error(f"Error parsing row {idx}: {e}")
                # Add empty features for failed rows
                parsed_features.append(self._get_empty_numerical_features())
        
        # Create DataFrame from parsed features
        features_df = pd.DataFrame(parsed_features)
        
        # Combine with original DataFrame
        result_df = pd.concat([df, features_df], axis=1)
        
        # Log parsing statistics
        successful_parses = features_df['has_ipq'].sum()
        brand_detections = features_df['has_brand'].sum()
        category_detections = features_df['has_category'].sum()
        dimension_detections = features_df['has_dimensions'].sum()
        
        self.logger.info(f"Catalog parsing completed. "
                        f"IPQ: {successful_parses}/{len(df)}, "
                        f"Brands: {brand_detections}/{len(df)}, "
                        f"Categories: {category_detections}/{len(df)}, "
                        f"Dimensions: {dimension_detections}/{len(df)}")
        
        return result_df
    
    def _get_empty_numerical_features(self) -> Dict[str, float]:
        """Get empty numerical features for failed parsing."""
        return {
            'ipq_value': 0.0, 'ipq_confidence': 0.0, 'has_ipq': 0.0,
            'has_dimensions': 0.0, 'length': 0.0, 'width': 0.0, 'height': 0.0,
            'volume': 0.0, 'area': 0.0, 'diameter': 0.0,
            'material_count': 0.0, 'color_count': 0.0, 'has_materials': 0.0, 'has_colors': 0.0,
            'has_brand': 0.0, 'brand_confidence': 0.0, 'has_category': 0.0, 'category_confidence': 0.0,
            'spec_count': 0.0, 'has_weight': 0.0, 'has_volume_spec': 0.0, 'has_power': 0.0,
        }