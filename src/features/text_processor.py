"""
TextProcessor for catalog content parsing and text cleaning

This module implements text processing functionality for the ML Product Pricing Challenge,
focusing on extracting structured information from catalog content and cleaning text data.
"""

import re
import string
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from dataclasses import dataclass


@dataclass
class ParsedCatalogContent:
    """Structured representation of parsed catalog content"""
    title: Optional[str] = None
    description: Optional[str] = None
    item_pack_quantity: Optional[str] = None
    raw_content: str = ""
    cleaned_content: str = ""


class TextProcessor:
    """
    Handles catalog content parsing and text cleaning operations.
    
    This class provides methods to extract title, description, and Item Pack Quantity
    from catalog_content, along with comprehensive text cleaning and normalization.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize TextProcessor with optional logger.
        
        Args:
            logger: Optional logger instance for tracking operations
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Common patterns for catalog content parsing
        self.title_patterns = [
            r'^([^|]+)\|',  # Text before first pipe
            r'^([^-]+)-',   # Text before first dash
            r'^([^\n]+)',   # First line
        ]
        
        self.description_patterns = [
            r'\|([^|]+)$',  # Text after last pipe
            r'-([^-]+)$',   # Text after last dash
        ]
        
        self.ipq_patterns = [
            r'(?i)(?:pack|quantity|qty|count|size)\s*:?\s*([^|,\n]+)',
            r'(?i)(\d+\s*(?:pack|pcs?|pieces?|count|units?))',
            r'(?i)(pack\s+of\s+\d+)',
            r'(?i)(\d+\s*x\s*\d+)',
        ]
        
        # Text cleaning patterns
        self.special_char_pattern = re.compile(r'[^\w\s\-.,()&/]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.html_pattern = re.compile(r'<[^>]+>')
        
    def parse_catalog_content(self, catalog_content: str) -> ParsedCatalogContent:
        """
        Parse catalog content to extract title, description, and IPQ.
        
        Args:
            catalog_content: Raw catalog content string
            
        Returns:
            ParsedCatalogContent object with extracted components
        """
        if not catalog_content or pd.isna(catalog_content):
            return ParsedCatalogContent(raw_content="")
            
        content = str(catalog_content).strip()
        parsed = ParsedCatalogContent(raw_content=content)
        
        # Extract title using multiple patterns
        parsed.title = self._extract_title(content)
        
        # Extract description using multiple patterns
        parsed.description = self._extract_description(content)
        
        # Extract Item Pack Quantity
        parsed.item_pack_quantity = self._extract_ipq(content)
        
        # Clean the entire content
        parsed.cleaned_content = self.clean_text(content)
        
        self.logger.debug(f"Parsed catalog content: title='{parsed.title}', "
                         f"description='{parsed.description}', ipq='{parsed.item_pack_quantity}'")
        
        return parsed
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract product title from catalog content."""
        for pattern in self.title_patterns:
            match = re.search(pattern, content)
            if match:
                title = match.group(1).strip()
                if len(title) > 3:  # Minimum viable title length
                    return self.clean_text(title)
        
        # Fallback: use first 100 characters
        if len(content) > 10:
            return self.clean_text(content[:100])
        
        return None
    
    def _extract_description(self, content: str) -> Optional[str]:
        """Extract product description from catalog content."""
        for pattern in self.description_patterns:
            match = re.search(pattern, content)
            if match:
                description = match.group(1).strip()
                if len(description) > 10:  # Minimum viable description length
                    return self.clean_text(description)
        
        # Fallback: use content after first separator or full content
        separators = ['|', '-', '\n']
        for sep in separators:
            if sep in content:
                parts = content.split(sep, 1)
                if len(parts) > 1 and len(parts[1].strip()) > 10:
                    return self.clean_text(parts[1].strip())
        
        # Use full content as description if no title was extracted
        if len(content) > 10:
            return self.clean_text(content)
        
        return None
    
    def _extract_ipq(self, content: str) -> Optional[str]:
        """Extract Item Pack Quantity from catalog content."""
        for pattern in self.ipq_patterns:
            match = re.search(pattern, content)
            if match:
                ipq = match.group(1).strip()
                if ipq:
                    return self.clean_text(ipq)
        
        return None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Handle special characters
        text = self._handle_special_characters(text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip and normalize case
        text = text.strip()
        
        return text
    
    def _handle_special_characters(self, text: str) -> str:
        """Handle special characters in text."""
        # Replace common special characters with spaces or remove them
        replacements = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '\t': ' ',
            '\r': ' ',
            '\n': ' ',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove other special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,()&/\'"]', ' ', text)
        
        return text
    
    def standardize_text(self, text: str) -> str:
        """
        Standardize text for consistent processing.
        
        Args:
            text: Text to standardize
            
        Returns:
            Standardized text
        """
        if not text or pd.isna(text):
            return ""
        
        text = self.clean_text(text)
        
        # Convert to lowercase for standardization
        text = text.lower()
        
        # Standardize common terms
        standardizations = {
            r'\bpcs?\b': 'pieces',
            r'\bpkg?\b': 'package',
            r'\boz?\b': 'ounce',
            r'\blbs?\b': 'pound',
            r'\bkgs?\b': 'kilogram',
            r'\bgms?\b': 'gram',
            r'\bmls?\b': 'milliliter',
            r'\blts?\b': 'liter',
        }
        
        for pattern, replacement in standardizations.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def extract_product_attributes(self, catalog_content: str) -> Dict[str, Any]:
        """
        Extract comprehensive product attributes from catalog content.
        
        Args:
            catalog_content: Raw catalog content
            
        Returns:
            Dictionary of extracted attributes
        """
        parsed = self.parse_catalog_content(catalog_content)
        
        attributes = {
            'title': parsed.title,
            'description': parsed.description,
            'item_pack_quantity': parsed.item_pack_quantity,
            'cleaned_content': parsed.cleaned_content,
            'standardized_content': self.standardize_text(parsed.cleaned_content),
            'content_length': len(parsed.cleaned_content) if parsed.cleaned_content else 0,
            'word_count': len(parsed.cleaned_content.split()) if parsed.cleaned_content else 0,
            'has_title': parsed.title is not None,
            'has_description': parsed.description is not None,
            'has_ipq': parsed.item_pack_quantity is not None,
        }
        
        return attributes
    
    def batch_process_catalog_content(self, df: pd.DataFrame, 
                                    content_column: str = 'catalog_content') -> pd.DataFrame:
        """
        Process catalog content for entire DataFrame.
        
        Args:
            df: DataFrame containing catalog content
            content_column: Name of the column containing catalog content
            
        Returns:
            DataFrame with additional parsed columns
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        
        self.logger.info(f"Processing {len(df)} catalog content entries")
        
        # Process each row
        parsed_data = []
        for idx, row in df.iterrows():
            try:
                attributes = self.extract_product_attributes(row[content_column])
                parsed_data.append(attributes)
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
                # Add empty attributes for failed rows
                parsed_data.append({
                    'title': None,
                    'description': None,
                    'item_pack_quantity': None,
                    'cleaned_content': "",
                    'standardized_content': "",
                    'content_length': 0,
                    'word_count': 0,
                    'has_title': False,
                    'has_description': False,
                    'has_ipq': False,
                })
        
        # Create new DataFrame with parsed attributes
        parsed_df = pd.DataFrame(parsed_data)
        
        # Combine with original DataFrame
        result_df = pd.concat([df, parsed_df], axis=1)
        
        self.logger.info(f"Successfully processed catalog content. "
                        f"Extracted titles: {parsed_df['has_title'].sum()}, "
                        f"descriptions: {parsed_df['has_description'].sum()}, "
                        f"IPQ: {parsed_df['has_ipq'].sum()}")
        
        return result_df