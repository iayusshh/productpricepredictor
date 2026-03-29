"""
DataLoader with robust schema validation for ML Product Pricing Challenge 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
try:
    from ..interfaces import DataPreprocessorInterface
    from ..config import config
    from ..models.data_models import ProductSample
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from interfaces import DataPreprocessorInterface
    from config import config
    from models.data_models import ProductSample


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class DataLoader:
    """
    DataLoader with robust schema validation and comprehensive error handling
    
    Implements fail-fast schema and column type validation with human-readable errors.
    Provides methods to load train.csv and test.csv with comprehensive error handling.
    Creates data integrity checks and completeness validation.
    """
    
    def __init__(self, data_config=None):
        """Initialize DataLoader with configuration"""
        self.config = data_config or config.data
        self.logger = logging.getLogger(__name__)
        
        # Expected schemas for different datasets
        self.train_schema = {
            'sample_id': 'object',
            'catalog_content': 'object', 
            'image_link': 'object',
            'price': 'float64'
        }
        
        self.test_schema = {
            'sample_id': 'object',
            'catalog_content': 'object',
            'image_link': 'object'
        }
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load training data from dataset/train.csv with comprehensive validation
        
        Returns:
            pd.DataFrame: Validated training data
            
        Raises:
            DataValidationError: If data fails validation checks
        """
        try:
            self.logger.info(f"Loading training data from {self.config.train_file}")
            
            # Check if file exists
            if not Path(self.config.train_file).exists():
                raise DataValidationError(
                    f"Training file not found: {self.config.train_file}. "
                    f"Please ensure the file exists in the correct location."
                )
            
            # Load data with error handling
            try:
                df = pd.read_csv(self.config.train_file)
            except pd.errors.EmptyDataError:
                raise DataValidationError(
                    f"Training file is empty: {self.config.train_file}"
                )
            except pd.errors.ParserError as e:
                raise DataValidationError(
                    f"Failed to parse training file {self.config.train_file}: {str(e)}. "
                    f"Please check file format and encoding."
                )
            
            # Validate schema and types
            self._validate_schema_and_types(df, self.train_schema, "training")
            
            # Validate data integrity
            self._validate_data_integrity(df, is_training=True)
            
            self.logger.info(f"Successfully loaded {len(df)} training samples")
            return df
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            else:
                raise DataValidationError(f"Unexpected error loading training data: {str(e)}")
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data from dataset/test.csv with comprehensive validation
        
        Returns:
            pd.DataFrame: Validated test data
            
        Raises:
            DataValidationError: If data fails validation checks
        """
        try:
            self.logger.info(f"Loading test data from {self.config.test_file}")
            
            # Check if file exists
            if not Path(self.config.test_file).exists():
                raise DataValidationError(
                    f"Test file not found: {self.config.test_file}. "
                    f"Please ensure the file exists in the correct location."
                )
            
            # Load data with error handling
            try:
                df = pd.read_csv(self.config.test_file)
            except pd.errors.EmptyDataError:
                raise DataValidationError(
                    f"Test file is empty: {self.config.test_file}"
                )
            except pd.errors.ParserError as e:
                raise DataValidationError(
                    f"Failed to parse test file {self.config.test_file}: {str(e)}. "
                    f"Please check file format and encoding."
                )
            
            # Validate schema and types
            self._validate_schema_and_types(df, self.test_schema, "test")
            
            # Validate data integrity
            self._validate_data_integrity(df, is_training=False)
            
            self.logger.info(f"Successfully loaded {len(df)} test samples")
            return df
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            else:
                raise DataValidationError(f"Unexpected error loading test data: {str(e)}")
    
    def validate_schema_and_types(self, df: pd.DataFrame) -> bool:
        """
        Public interface for schema validation
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            DataValidationError: If validation fails
        """
        # Determine schema based on columns
        if 'price' in df.columns:
            schema = self.train_schema
            dataset_type = "training"
        else:
            schema = self.test_schema
            dataset_type = "test"
        
        return self._validate_schema_and_types(df, schema, dataset_type)
    
    def _validate_schema_and_types(self, df: pd.DataFrame, expected_schema: Dict[str, str], 
                                 dataset_type: str) -> bool:
        """
        Validate DataFrame schema and column types with fail-fast error handling
        
        Args:
            df: DataFrame to validate
            expected_schema: Expected column names and types
            dataset_type: Type of dataset for error messages
            
        Returns:
            bool: True if validation passes
            
        Raises:
            DataValidationError: If validation fails with human-readable error
        """
        # Check if DataFrame is empty
        if df.empty:
            raise DataValidationError(
                f"The {dataset_type} dataset is empty. "
                f"Please ensure the file contains data."
            )
        
        # Check for required columns
        missing_columns = set(expected_schema.keys()) - set(df.columns)
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns in {dataset_type} dataset: {list(missing_columns)}. "
                f"Expected columns: {list(expected_schema.keys())}. "
                f"Found columns: {list(df.columns)}."
            )
        
        # Check for extra columns (warn but don't fail)
        extra_columns = set(df.columns) - set(expected_schema.keys())
        if extra_columns:
            self.logger.warning(
                f"Extra columns found in {dataset_type} dataset: {list(extra_columns)}. "
                f"These will be ignored."
            )
        
        # Validate column types
        type_errors = []
        for column, expected_type in expected_schema.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                
                # Special handling for different type representations
                if expected_type == 'object' and actual_type not in ['object', 'string']:
                    # Try to convert to string if it's not already
                    try:
                        df[column] = df[column].astype('object')
                        self.logger.info(f"Converted {column} to object type")
                    except Exception:
                        type_errors.append(
                            f"Column '{column}': expected {expected_type}, got {actual_type}"
                        )
                
                elif expected_type == 'float64' and not pd.api.types.is_numeric_dtype(df[column]):
                    # Try to convert to numeric
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                        self.logger.info(f"Converted {column} to numeric type")
                    except Exception:
                        type_errors.append(
                            f"Column '{column}': expected {expected_type}, got {actual_type}"
                        )
        
        if type_errors:
            raise DataValidationError(
                f"Column type validation failed for {dataset_type} dataset:\n" +
                "\n".join(f"  - {error}" for error in type_errors) +
                f"\n\nPlease ensure all columns have the correct data types."
            )
        
        self.logger.info(f"Schema validation passed for {dataset_type} dataset")
        return True
    
    def _validate_data_integrity(self, df: pd.DataFrame, is_training: bool = True) -> bool:
        """
        Validate data integrity and completeness
        
        Args:
            df: DataFrame to validate
            is_training: Whether this is training data
            
        Returns:
            bool: True if validation passes
            
        Raises:
            DataValidationError: If critical integrity issues are found
        """
        integrity_issues = []
        warnings = []
        
        # Check for duplicate sample_ids
        duplicate_ids = df[df['sample_id'].duplicated()]
        if not duplicate_ids.empty:
            integrity_issues.append(
                f"Found {len(duplicate_ids)} duplicate sample_ids: "
                f"{duplicate_ids['sample_id'].tolist()[:10]}..."  # Show first 10
            )
        
        # Check for missing sample_ids
        missing_sample_ids = df['sample_id'].isna().sum()
        if missing_sample_ids > 0:
            integrity_issues.append(
                f"Found {missing_sample_ids} missing sample_ids"
            )
        
        # Check for empty catalog_content
        empty_catalog = df['catalog_content'].isna().sum()
        if empty_catalog > 0:
            warnings.append(
                f"Found {empty_catalog} samples with missing catalog_content"
            )
        
        # Check for empty image_links
        empty_images = df['image_link'].isna().sum()
        if empty_images > 0:
            warnings.append(
                f"Found {empty_images} samples with missing image_links"
            )
        
        # Training-specific checks
        if is_training:
            # Check for missing prices
            missing_prices = df['price'].isna().sum()
            if missing_prices > 0:
                warnings.append(
                    f"Found {missing_prices} samples with missing prices"
                )
            
            # Check for negative prices
            negative_prices = (df['price'] < 0).sum()
            if negative_prices > 0:
                integrity_issues.append(
                    f"Found {negative_prices} samples with negative prices"
                )
            
            # Check for zero prices (warning, not error)
            zero_prices = (df['price'] == 0).sum()
            if zero_prices > 0:
                warnings.append(
                    f"Found {zero_prices} samples with zero prices "
                    f"(will be handled according to strategy: {self.config.zero_price_strategy})"
                )
            
            # Check price distribution for outliers
            if not df['price'].empty:
                price_stats = df['price'].describe()
                q99 = df['price'].quantile(0.99)
                q01 = df['price'].quantile(0.01)
                extreme_high = (df['price'] > q99 * 10).sum()  # 10x the 99th percentile
                extreme_low = (df['price'] < q01 / 10).sum()   # 1/10th the 1st percentile
                
                if extreme_high > 0:
                    warnings.append(
                        f"Found {extreme_high} samples with extremely high prices (>10x 99th percentile)"
                    )
                if extreme_low > 0:
                    warnings.append(
                        f"Found {extreme_low} samples with extremely low prices (<1/10th 1st percentile)"
                    )
        
        # Log warnings
        for warning in warnings:
            self.logger.warning(warning)
        
        # Fail on critical integrity issues
        if integrity_issues:
            raise DataValidationError(
                f"Data integrity validation failed:\n" +
                "\n".join(f"  - {issue}" for issue in integrity_issues) +
                f"\n\nPlease fix these issues before proceeding."
            )
        
        self.logger.info("Data integrity validation passed")
        return True
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary of loaded data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dict containing data summary statistics
        """
        summary = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'missing_values': df.isna().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Add training-specific summary
        if 'price' in df.columns:
            price_stats = df['price'].describe().to_dict()
            summary['price_statistics'] = price_stats
            summary['zero_prices'] = (df['price'] == 0).sum()
            summary['negative_prices'] = (df['price'] < 0).sum()
        
        # Add text content summary
        if 'catalog_content' in df.columns:
            content_lengths = df['catalog_content'].str.len()
            summary['catalog_content_stats'] = {
                'avg_length': content_lengths.mean(),
                'min_length': content_lengths.min(),
                'max_length': content_lengths.max(),
                'empty_content': df['catalog_content'].isna().sum()
            }
        
        return summary
    
    def validate_sample_format(self, sample_dict: Dict[str, Any]) -> ProductSample:
        """
        Validate and convert a dictionary to ProductSample
        
        Args:
            sample_dict: Dictionary containing sample data
            
        Returns:
            ProductSample: Validated product sample
            
        Raises:
            DataValidationError: If sample format is invalid
        """
        required_fields = ['sample_id', 'catalog_content', 'image_link']
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in sample_dict]
        if missing_fields:
            raise DataValidationError(
                f"Missing required fields in sample: {missing_fields}"
            )
        
        # Validate field types and values
        if not isinstance(sample_dict['sample_id'], (str, int)):
            raise DataValidationError("sample_id must be string or integer")
        
        if not isinstance(sample_dict['catalog_content'], str):
            raise DataValidationError("catalog_content must be string")
        
        if not isinstance(sample_dict['image_link'], str):
            raise DataValidationError("image_link must be string")
        
        # Convert to ProductSample
        try:
            return ProductSample(
                sample_id=str(sample_dict['sample_id']),
                catalog_content=sample_dict['catalog_content'],
                image_link=sample_dict['image_link'],
                price=sample_dict.get('price')
            )
        except Exception as e:
            raise DataValidationError(f"Failed to create ProductSample: {str(e)}")