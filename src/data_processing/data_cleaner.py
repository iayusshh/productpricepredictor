"""
DataCleaner for comprehensive preprocessing of ML Product Pricing Challenge 2025
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any, Set
try:
    from ..config import config
    from ..models.data_models import ProductSample
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config
    from models.data_models import ProductSample


class DataCleaningError(Exception):
    """Custom exception for data cleaning errors"""
    pass


class DataCleaner:
    """
    DataCleaner for comprehensive preprocessing
    
    Creates methods to handle missing values and invalid entries in catalog content.
    Implements text cleaning and standardization for catalog content.
    Adds validation for sample_id uniqueness and completeness.
    """
    
    def __init__(self, data_config=None):
        """Initialize DataCleaner with configuration"""
        self.config = data_config or config.data
        self.logger = logging.getLogger(__name__)
        
        # Text cleaning patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})')
        
        # Special characters to normalize
        self.special_chars = {
            '"': '"',  # Smart quotes
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '…': '...',  # Ellipsis
            '®': '(R)',  # Registered trademark
            '™': '(TM)',  # Trademark
            '©': '(C)',  # Copyright
        }
        
        # Cleaning statistics
        self.cleaning_stats = {
            'samples_processed': 0,
            'missing_catalog_content': 0,
            'empty_catalog_content': 0,
            'html_tags_removed': 0,
            'urls_removed': 0,
            'emails_removed': 0,
            'phones_removed': 0,
            'special_chars_normalized': 0,
            'whitespace_normalized': 0,
            'duplicate_sample_ids': 0,
            'invalid_sample_ids': 0
        }
    
    def clean_catalog_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize catalog content
        
        Args:
            df: DataFrame with catalog_content column to clean
            
        Returns:
            pd.DataFrame: DataFrame with cleaned catalog content
        """
        if 'catalog_content' not in df.columns:
            self.logger.warning("No 'catalog_content' column found in DataFrame")
            return df
        
        self.logger.info("Starting catalog content cleaning")
        df_cleaned = df.copy()
        
        # Reset cleaning statistics
        self.cleaning_stats['samples_processed'] = len(df)
        
        # Handle missing catalog content
        missing_mask = df_cleaned['catalog_content'].isna()
        missing_count = missing_mask.sum()
        if missing_count > 0:
            self.cleaning_stats['missing_catalog_content'] = missing_count
            # Replace missing with empty string for processing
            df_cleaned.loc[missing_mask, 'catalog_content'] = ''
            self.logger.warning(f"Found {missing_count} samples with missing catalog content")
        
        # Convert to string and handle empty content
        df_cleaned['catalog_content'] = df_cleaned['catalog_content'].astype(str)
        empty_mask = df_cleaned['catalog_content'].str.strip() == ''
        empty_count = empty_mask.sum()
        if empty_count > 0:
            self.cleaning_stats['empty_catalog_content'] = empty_count
            self.logger.warning(f"Found {empty_count} samples with empty catalog content")
        
        # Apply cleaning functions
        df_cleaned['catalog_content'] = df_cleaned['catalog_content'].apply(self._clean_single_content)
        
        self.logger.info(f"Catalog content cleaning completed. Processed {len(df)} samples")
        return df_cleaned
    
    def _clean_single_content(self, content: str) -> str:
        """
        Clean a single catalog content string
        
        Args:
            content: Raw catalog content string
            
        Returns:
            str: Cleaned catalog content
        """
        if not isinstance(content, str) or not content.strip():
            return content
        
        cleaned = content
        
        # Remove HTML tags
        if self.html_pattern.search(cleaned):
            cleaned = self.html_pattern.sub(' ', cleaned)
            self.cleaning_stats['html_tags_removed'] += 1
        
        # Remove URLs (but keep domain names that might be brand names)
        if self.url_pattern.search(cleaned):
            cleaned = self.url_pattern.sub(' [URL] ', cleaned)
            self.cleaning_stats['urls_removed'] += 1
        
        # Remove email addresses
        if self.email_pattern.search(cleaned):
            cleaned = self.email_pattern.sub(' [EMAIL] ', cleaned)
            self.cleaning_stats['emails_removed'] += 1
        
        # Remove phone numbers
        if self.phone_pattern.search(cleaned):
            cleaned = self.phone_pattern.sub(' [PHONE] ', cleaned)
            self.cleaning_stats['phones_removed'] += 1
        
        # Normalize special characters
        for special_char, replacement in self.special_chars.items():
            if special_char in cleaned:
                cleaned = cleaned.replace(special_char, replacement)
                self.cleaning_stats['special_chars_normalized'] += 1
        
        # Normalize whitespace
        original_length = len(cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())
        if len(cleaned) != original_length:
            self.cleaning_stats['whitespace_normalized'] += 1
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)  # Multiple exclamation marks
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)  # Multiple question marks
        cleaned = re.sub(r'[.]{3,}', '...', cleaned)  # Multiple dots
        
        # Clean up common formatting issues
        cleaned = re.sub(r'\s*:\s*', ': ', cleaned)  # Normalize colons
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)  # Normalize commas
        cleaned = re.sub(r'\s*;\s*', '; ', cleaned)  # Normalize semicolons
        
        # Remove leading/trailing punctuation and whitespace
        cleaned = cleaned.strip(' .,;:-')
        
        return cleaned
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and invalid entries in catalog content
        
        Args:
            df: DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df_handled = df.copy()
        
        # Handle missing catalog content
        if 'catalog_content' in df_handled.columns:
            missing_content = df_handled['catalog_content'].isna()
            if missing_content.any():
                # Strategy: Replace with placeholder text that indicates missing content
                placeholder = "No product description available"
                df_handled.loc[missing_content, 'catalog_content'] = placeholder
                self.logger.info(f"Replaced {missing_content.sum()} missing catalog content entries")
        
        # Handle missing image links
        if 'image_link' in df_handled.columns:
            missing_images = df_handled['image_link'].isna()
            if missing_images.any():
                # Strategy: Replace with placeholder URL
                placeholder_url = "https://placeholder.image/missing"
                df_handled.loc[missing_images, 'image_link'] = placeholder_url
                df_handled['has_image'] = ~missing_images  # Flag for missing images
                self.logger.info(f"Replaced {missing_images.sum()} missing image links")
        
        # Handle missing sample IDs (critical error)
        if 'sample_id' in df_handled.columns:
            missing_ids = df_handled['sample_id'].isna()
            if missing_ids.any():
                raise DataCleaningError(
                    f"Found {missing_ids.sum()} samples with missing sample_id. "
                    f"This is a critical error that must be fixed at the data source."
                )
        
        return df_handled
    
    def validate_sample_id_uniqueness(self, df: pd.DataFrame) -> bool:
        """
        Validate sample_id uniqueness and completeness
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            DataCleaningError: If validation fails
        """
        if 'sample_id' not in df.columns:
            raise DataCleaningError("No 'sample_id' column found in DataFrame")
        
        # Check for missing sample IDs
        missing_ids = df['sample_id'].isna()
        missing_count = missing_ids.sum()
        if missing_count > 0:
            self.cleaning_stats['invalid_sample_ids'] = missing_count
            raise DataCleaningError(
                f"Found {missing_count} samples with missing sample_id"
            )
        
        # Check for duplicate sample IDs
        duplicates = df['sample_id'].duplicated()
        duplicate_count = duplicates.sum()
        if duplicate_count > 0:
            self.cleaning_stats['duplicate_sample_ids'] = duplicate_count
            duplicate_ids = df[duplicates]['sample_id'].tolist()
            raise DataCleaningError(
                f"Found {duplicate_count} duplicate sample_ids: {duplicate_ids[:10]}..."
            )
        
        # Check for invalid sample ID formats
        invalid_format_mask = ~df['sample_id'].astype(str).str.match(r'^[a-zA-Z0-9_-]+$')
        invalid_format_count = invalid_format_mask.sum()
        if invalid_format_count > 0:
            invalid_ids = df[invalid_format_mask]['sample_id'].head(10).tolist()
            self.logger.warning(
                f"Found {invalid_format_count} sample_ids with potentially invalid format: {invalid_ids}"
            )
        
        # Check for empty sample IDs
        empty_ids = df['sample_id'].astype(str).str.strip() == ''
        empty_count = empty_ids.sum()
        if empty_count > 0:
            self.cleaning_stats['invalid_sample_ids'] += empty_count
            raise DataCleaningError(
                f"Found {empty_count} samples with empty sample_id"
            )
        
        self.logger.info(f"Sample ID validation passed. {len(df)} unique sample IDs found")
        return True
    
    def standardize_text_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize text formatting across catalog content
        
        Args:
            df: DataFrame with catalog content to standardize
            
        Returns:
            pd.DataFrame: DataFrame with standardized text format
        """
        if 'catalog_content' not in df.columns:
            return df
        
        df_standardized = df.copy()
        
        # Apply standardization
        df_standardized['catalog_content'] = df_standardized['catalog_content'].apply(
            self._standardize_single_text
        )
        
        return df_standardized
    
    def _standardize_single_text(self, text: str) -> str:
        """
        Standardize a single text string
        
        Args:
            text: Text to standardize
            
        Returns:
            str: Standardized text
        """
        if not isinstance(text, str):
            return str(text)
        
        standardized = text
        
        # Standardize common product description patterns
        # Item Name: -> Item Name:
        standardized = re.sub(r'Item\s+Name\s*:\s*', 'Item Name: ', standardized, flags=re.IGNORECASE)
        
        # Value: -> Value:
        standardized = re.sub(r'Value\s*:\s*', 'Value: ', standardized, flags=re.IGNORECASE)
        
        # Unit: -> Unit:
        standardized = re.sub(r'Unit\s*:\s*', 'Unit: ', standardized, flags=re.IGNORECASE)
        
        # Bullet Point -> Bullet Point
        standardized = re.sub(r'Bullet\s+Point\s+(\d+)\s*:\s*', r'Bullet Point \1: ', standardized, flags=re.IGNORECASE)
        
        # Standardize measurements
        # Convert common measurement abbreviations to full words
        measurement_replacements = {
            r'\boz\b': 'ounce',
            r'\blb\b': 'pound',
            r'\blbs\b': 'pounds',
            r'\bft\b': 'feet',
            r'\bin\b': 'inch',
            r'\bins\b': 'inches',
            r'\bml\b': 'milliliter',
            r'\bl\b': 'liter',
            r'\bkg\b': 'kilogram',
            r'\bg\b': 'gram',
            r'\bmm\b': 'millimeter',
            r'\bcm\b': 'centimeter',
            r'\bm\b': 'meter'
        }
        
        for pattern, replacement in measurement_replacements.items():
            standardized = re.sub(pattern, replacement, standardized, flags=re.IGNORECASE)
        
        # Standardize pack/package terminology
        standardized = re.sub(r'\bpack\s+of\s+(\d+)\b', r'(\1-pack)', standardized, flags=re.IGNORECASE)
        standardized = re.sub(r'\b(\d+)\s*-?\s*pack\b', r'(\1-pack)', standardized, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        standardized = re.sub(r'\s+', ' ', standardized.strip())
        
        return standardized
    
    def extract_structured_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract structured fields from catalog content
        
        Args:
            df: DataFrame with catalog content
            
        Returns:
            pd.DataFrame: DataFrame with extracted structured fields
        """
        if 'catalog_content' not in df.columns:
            return df
        
        df_extracted = df.copy()
        
        # Extract item names
        df_extracted['item_name'] = df_extracted['catalog_content'].apply(
            lambda x: self._extract_item_name(x)
        )
        
        # Extract values and units
        value_unit_data = df_extracted['catalog_content'].apply(
            lambda x: self._extract_value_unit(x)
        )
        df_extracted['extracted_value'] = [item[0] for item in value_unit_data]
        df_extracted['extracted_unit'] = [item[1] for item in value_unit_data]
        
        # Extract bullet points
        df_extracted['bullet_points'] = df_extracted['catalog_content'].apply(
            lambda x: self._extract_bullet_points(x)
        )
        
        # Count bullet points
        df_extracted['bullet_point_count'] = df_extracted['bullet_points'].apply(len)
        
        return df_extracted
    
    def _extract_item_name(self, content: str) -> Optional[str]:
        """Extract item name from catalog content"""
        if not isinstance(content, str):
            return None
        
        # Look for "Item Name:" pattern
        match = re.search(r'Item\s+Name\s*:\s*([^\n\r]+)', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: use first line if it looks like a product name
        lines = content.split('\n')
        if lines and len(lines[0].strip()) > 10:  # Reasonable product name length
            return lines[0].strip()
        
        return None
    
    def _extract_value_unit(self, content: str) -> tuple:
        """Extract value and unit from catalog content"""
        if not isinstance(content, str):
            return None, None
        
        # Look for "Value:" and "Unit:" patterns
        value_match = re.search(r'Value\s*:\s*([^\n\r]+)', content, re.IGNORECASE)
        unit_match = re.search(r'Unit\s*:\s*([^\n\r]+)', content, re.IGNORECASE)
        
        value = value_match.group(1).strip() if value_match else None
        unit = unit_match.group(1).strip() if unit_match else None
        
        # Try to convert value to numeric
        if value:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string if not numeric
        
        return value, unit
    
    def _extract_bullet_points(self, content: str) -> List[str]:
        """Extract bullet points from catalog content"""
        if not isinstance(content, str):
            return []
        
        # Look for "Bullet Point N:" patterns
        bullet_points = []
        matches = re.finditer(r'Bullet\s+Point\s+\d+\s*:\s*([^\n\r]+)', content, re.IGNORECASE)
        
        for match in matches:
            bullet_point = match.group(1).strip()
            if bullet_point:
                bullet_points.append(bullet_point)
        
        return bullet_points
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of cleaning operations performed
        
        Returns:
            Dict containing cleaning statistics and summary
        """
        summary = {
            'cleaning_statistics': self.cleaning_stats.copy(),
            'cleaning_rate': {},
            'recommendations': []
        }
        
        # Calculate cleaning rates
        if self.cleaning_stats['samples_processed'] > 0:
            total_samples = self.cleaning_stats['samples_processed']
            summary['cleaning_rate'] = {
                'missing_content_rate': (self.cleaning_stats['missing_catalog_content'] / total_samples) * 100,
                'empty_content_rate': (self.cleaning_stats['empty_catalog_content'] / total_samples) * 100,
                'html_cleaning_rate': (self.cleaning_stats['html_tags_removed'] / total_samples) * 100,
                'url_cleaning_rate': (self.cleaning_stats['urls_removed'] / total_samples) * 100,
                'special_char_rate': (self.cleaning_stats['special_chars_normalized'] / total_samples) * 100
            }
        
        # Generate recommendations
        if self.cleaning_stats['missing_catalog_content'] > 0:
            summary['recommendations'].append(
                f"High number of missing catalog content ({self.cleaning_stats['missing_catalog_content']}). "
                f"Consider data source quality improvement."
            )
        
        if self.cleaning_stats['html_tags_removed'] > total_samples * 0.1:
            summary['recommendations'].append(
                "High rate of HTML content detected. Consider preprocessing at data source."
            )
        
        if self.cleaning_stats['duplicate_sample_ids'] > 0:
            summary['recommendations'].append(
                f"Duplicate sample IDs found ({self.cleaning_stats['duplicate_sample_ids']}). "
                f"This indicates data quality issues."
            )
        
        return summary
    
    def reset_cleaning_stats(self):
        """Reset cleaning statistics for new dataset"""
        for key in self.cleaning_stats:
            self.cleaning_stats[key] = 0
        self.logger.info("Cleaning statistics reset")
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate cleaned data quality
        
        Args:
            df: Cleaned DataFrame to validate
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check catalog content quality
        if 'catalog_content' in df.columns:
            empty_content = (df['catalog_content'].str.strip() == '').sum()
            very_short_content = (df['catalog_content'].str.len() < 20).sum()
            
            validation_results['summary']['empty_content_count'] = empty_content
            validation_results['summary']['very_short_content_count'] = very_short_content
            
            if empty_content > 0:
                validation_results['warnings'].append(f"{empty_content} samples have empty catalog content")
            
            if very_short_content > len(df) * 0.1:  # >10% very short content
                validation_results['warnings'].append(
                    f"{very_short_content} samples have very short catalog content (<20 chars)"
                )
        
        # Check sample ID quality
        try:
            self.validate_sample_id_uniqueness(df)
            validation_results['summary']['sample_id_validation'] = 'passed'
        except DataCleaningError as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(str(e))
            validation_results['summary']['sample_id_validation'] = 'failed'
        
        return validation_results