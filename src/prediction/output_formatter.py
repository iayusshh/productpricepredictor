"""
OutputFormatter for strict submission compliance in ML Product Pricing Challenge 2025

This module implements strict output formatting that exactly matches the sample_test_out.csv
format with comprehensive validation and compliance checks.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import csv

from ..config import config


class OutputFormatter:
    """
    Strict output formatter ensuring exact compliance with submission requirements.
    
    This class formats predictions exactly as sample_test_out.csv with rigorous
    validation of sample_id matching, row counts, and value formats.
    """
    
    def __init__(self, 
                 output_precision: int = None,
                 validate_format: bool = True,
                 strict_mode: bool = True):
        """
        Initialize OutputFormatter with configuration.
        
        Args:
            output_precision: Number of decimal places for price values
            validate_format: Whether to validate output format
            strict_mode: Whether to enforce strict compliance checks
        """
        self.output_precision = output_precision or config.prediction.output_precision
        self.validate_format = validate_format
        self.strict_mode = strict_mode
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load sample format for reference
        self.sample_format = self._load_sample_format()
        
        self.logger.info(f"OutputFormatter initialized with precision={self.output_precision}, "
                        f"strict_mode={strict_mode}")
    
    def format_predictions_exact(self, 
                               sample_ids: List[str], 
                               predictions: np.ndarray,
                               output_file: str = None) -> pd.DataFrame:
        """
        Format predictions exactly as sample_test_out.csv.
        
        Args:
            sample_ids: List of sample IDs from test.csv
            predictions: Prediction values
            output_file: Optional output file path
            
        Returns:
            pd.DataFrame: Formatted output DataFrame
            
        Raises:
            ValueError: If inputs don't match requirements
            RuntimeError: If formatting fails
        """
        if len(sample_ids) != len(predictions):
            raise ValueError(f"Sample ID count ({len(sample_ids)}) must match "
                           f"prediction count ({len(predictions)})")
        
        self.logger.info(f"Formatting {len(sample_ids)} predictions for submission")
        
        try:
            # Create DataFrame with exact column names and order
            output_df = pd.DataFrame({
                'sample_id': sample_ids,
                'price': predictions
            })
            
            # Ensure sample_id is string type (matching sample format)
            output_df['sample_id'] = output_df['sample_id'].astype(str)
            
            # Format price values with exact precision
            output_df['price'] = self._format_price_values(output_df['price'])
            
            # Validate format compliance
            if self.validate_format:
                self._validate_exact_format(output_df)
            
            # Save to file if specified
            if output_file:
                self.save_to_csv(output_df, output_file)
            
            self.logger.info(f"Successfully formatted output: {len(output_df)} rows")
            
            return output_df
            
        except Exception as e:
            self.logger.error(f"Output formatting failed: {str(e)}")
            raise RuntimeError(f"Failed to format output: {str(e)}")
    
    def validate_sample_id_exact_match(self, 
                                     output_df: pd.DataFrame, 
                                     test_df: pd.DataFrame) -> bool:
        """
        Validate exact sample_id matching between output and test.csv.
        
        Args:
            output_df: Formatted output DataFrame
            test_df: Original test DataFrame
            
        Returns:
            bool: True if sample_ids match exactly
        """
        self.logger.info("Validating exact sample_id matching")
        
        # Convert to sets for comparison
        output_ids = set(output_df['sample_id'].astype(str))
        test_ids = set(test_df['sample_id'].astype(str))
        
        # Check for exact match
        if output_ids == test_ids:
            self.logger.info("✓ Sample ID validation passed: exact match")
            return True
        
        # Detailed analysis of mismatches
        missing_in_output = test_ids - output_ids
        extra_in_output = output_ids - test_ids
        
        self.logger.error("✗ Sample ID validation failed:")
        
        if missing_in_output:
            self.logger.error(f"  Missing in output: {len(missing_in_output)} sample_ids")
            if len(missing_in_output) <= 10:
                self.logger.error(f"  Missing IDs: {sorted(list(missing_in_output))}")
            else:
                self.logger.error(f"  First 10 missing: {sorted(list(missing_in_output))[:10]}")
        
        if extra_in_output:
            self.logger.error(f"  Extra in output: {len(extra_in_output)} sample_ids")
            if len(extra_in_output) <= 10:
                self.logger.error(f"  Extra IDs: {sorted(list(extra_in_output))}")
            else:
                self.logger.error(f"  First 10 extra: {sorted(list(extra_in_output))[:10]}")
        
        return False
    
    def validate_row_count_exact_match(self, 
                                     output_df: pd.DataFrame, 
                                     test_df: pd.DataFrame) -> bool:
        """
        Validate exact row count matching between output and test.csv.
        
        Args:
            output_df: Formatted output DataFrame
            test_df: Original test DataFrame
            
        Returns:
            bool: True if row counts match exactly
        """
        output_count = len(output_df)
        test_count = len(test_df)
        
        if output_count == test_count:
            self.logger.info(f"✓ Row count validation passed: {output_count} rows")
            return True
        
        self.logger.error(f"✗ Row count validation failed:")
        self.logger.error(f"  Output rows: {output_count}")
        self.logger.error(f"  Test rows: {test_count}")
        self.logger.error(f"  Difference: {output_count - test_count}")
        
        return False
    
    def validate_positive_float_values(self, output_df: pd.DataFrame) -> bool:
        """
        Validate all predictions are positive float values.
        
        Args:
            output_df: Formatted output DataFrame
            
        Returns:
            bool: True if all price values are valid positive floats
        """
        self.logger.info("Validating positive float values")
        
        prices = output_df['price']
        
        # Check data type
        if not pd.api.types.is_numeric_dtype(prices):
            self.logger.error("✗ Price column is not numeric")
            return False
        
        # Check for null values
        null_count = prices.isnull().sum()
        if null_count > 0:
            self.logger.error(f"✗ Found {null_count} null price values")
            return False
        
        # Check for infinite values
        inf_count = np.isinf(prices).sum()
        if inf_count > 0:
            self.logger.error(f"✗ Found {inf_count} infinite price values")
            return False
        
        # Check for NaN values
        nan_count = np.isnan(prices).sum()
        if nan_count > 0:
            self.logger.error(f"✗ Found {nan_count} NaN price values")
            return False
        
        # Check for negative values
        negative_count = (prices < 0).sum()
        if negative_count > 0:
            self.logger.error(f"✗ Found {negative_count} negative price values")
            self.logger.error(f"  Min value: {prices.min()}")
            return False
        
        # Check for zero values (may be acceptable but worth noting)
        zero_count = (prices == 0).sum()
        if zero_count > 0:
            self.logger.warning(f"⚠ Found {zero_count} zero price values")
        
        self.logger.info("✓ All price values are valid positive floats")
        return True
    
    def save_to_csv(self, 
                   output_df: pd.DataFrame, 
                   output_file: str,
                   validate_after_save: bool = True) -> bool:
        """
        Save formatted output to CSV with exact format compliance.
        
        Args:
            output_df: Formatted output DataFrame
            output_file: Output file path
            validate_after_save: Whether to validate file after saving
            
        Returns:
            bool: True if save was successful
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving output to: {output_file}")
            
            # Save with exact format matching sample_test_out.csv
            output_df.to_csv(
                output_file,
                index=False,
                float_format=f'%.{self.output_precision}f' if self.output_precision else None,
                lineterminator='\n'
            )
            
            # Validate file after saving
            if validate_after_save:
                return self._validate_saved_file(output_file, output_df)
            
            self.logger.info(f"✓ Successfully saved {len(output_df)} rows to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed to save output file: {str(e)}")
            return False
    
    def create_submission_file(self, 
                             sample_ids: List[str], 
                             predictions: np.ndarray,
                             test_df: pd.DataFrame,
                             output_file: str = None) -> Tuple[pd.DataFrame, bool]:
        """
        Create complete submission file with full validation.
        
        Args:
            sample_ids: List of sample IDs
            predictions: Prediction values
            test_df: Original test DataFrame for validation
            output_file: Output file path (default: test_out.csv)
            
        Returns:
            Tuple[pd.DataFrame, bool]: (formatted_df, validation_passed)
        """
        output_file = output_file or config.prediction.output_file
        
        self.logger.info("Creating complete submission file with validation")
        
        # Format predictions
        output_df = self.format_predictions_exact(sample_ids, predictions)
        
        # Comprehensive validation
        validation_results = []
        
        # Sample ID matching
        validation_results.append(
            self.validate_sample_id_exact_match(output_df, test_df)
        )
        
        # Row count matching
        validation_results.append(
            self.validate_row_count_exact_match(output_df, test_df)
        )
        
        # Positive float values
        validation_results.append(
            self.validate_positive_float_values(output_df)
        )
        
        # Format compliance
        validation_results.append(
            self._validate_exact_format(output_df)
        )
        
        all_valid = all(validation_results)
        
        if all_valid:
            self.logger.info("✓ All validations passed")
            # Save the file
            save_success = self.save_to_csv(output_df, output_file)
            if save_success:
                self.logger.info(f"✓ Submission file created successfully: {output_file}")
            else:
                all_valid = False
        else:
            self.logger.error("✗ Validation failed - submission file may not be compliant")
        
        return output_df, all_valid
    
    def _format_price_values(self, prices: pd.Series) -> pd.Series:
        """
        Format price values with exact precision and type.
        
        Args:
            prices: Series of price values
            
        Returns:
            pd.Series: Formatted price values
        """
        # Convert to float64 for consistency
        formatted_prices = prices.astype(np.float64)
        
        # Round to specified precision
        if self.output_precision is not None:
            formatted_prices = formatted_prices.round(self.output_precision)
        
        return formatted_prices
    
    def _validate_exact_format(self, output_df: pd.DataFrame) -> bool:
        """
        Validate exact format compliance with sample_test_out.csv.
        
        Args:
            output_df: Output DataFrame to validate
            
        Returns:
            bool: True if format is exactly compliant
        """
        self.logger.info("Validating exact format compliance")
        
        # Check column names
        expected_columns = ['sample_id', 'price']
        if list(output_df.columns) != expected_columns:
            self.logger.error(f"✗ Column names mismatch. Expected: {expected_columns}, "
                            f"Got: {list(output_df.columns)}")
            return False
        
        # Check column order
        if output_df.columns.tolist() != expected_columns:
            self.logger.error("✗ Column order incorrect")
            return False
        
        # Check data types
        if not pd.api.types.is_string_dtype(output_df['sample_id']):
            self.logger.error("✗ sample_id column should be string type")
            return False
        
        if not pd.api.types.is_numeric_dtype(output_df['price']):
            self.logger.error("✗ price column should be numeric type")
            return False
        
        # Check for duplicates
        duplicate_count = output_df['sample_id'].duplicated().sum()
        if duplicate_count > 0:
            self.logger.error(f"✗ Found {duplicate_count} duplicate sample_ids")
            return False
        
        # Check for empty values
        empty_sample_ids = output_df['sample_id'].isin(['', None]).sum()
        if empty_sample_ids > 0:
            self.logger.error(f"✗ Found {empty_sample_ids} empty sample_ids")
            return False
        
        self.logger.info("✓ Format validation passed")
        return True
    
    def _validate_saved_file(self, file_path: str, original_df: pd.DataFrame) -> bool:
        """
        Validate saved file by reading it back and comparing.
        
        Args:
            file_path: Path to saved file
            original_df: Original DataFrame for comparison
            
        Returns:
            bool: True if saved file is valid
        """
        try:
            # Read the saved file
            saved_df = pd.read_csv(file_path)
            
            # Check shape
            if saved_df.shape != original_df.shape:
                self.logger.error(f"✗ Saved file shape mismatch: "
                                f"expected {original_df.shape}, got {saved_df.shape}")
                return False
            
            # Check columns
            if list(saved_df.columns) != list(original_df.columns):
                self.logger.error("✗ Saved file columns mismatch")
                return False
            
            # Check sample_ids
            if not saved_df['sample_id'].equals(original_df['sample_id'].astype(str)):
                self.logger.error("✗ Saved file sample_ids mismatch")
                return False
            
            # Check prices (with tolerance for floating point precision)
            price_diff = np.abs(saved_df['price'] - original_df['price'])
            max_diff = price_diff.max()
            tolerance = 10 ** (-self.output_precision) if self.output_precision else 1e-6
            
            if max_diff > tolerance:
                self.logger.error(f"✗ Saved file prices mismatch: max_diff={max_diff}, "
                                f"tolerance={tolerance}")
                return False
            
            self.logger.info("✓ Saved file validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed to validate saved file: {str(e)}")
            return False
    
    def _load_sample_format(self) -> Optional[pd.DataFrame]:
        """
        Load sample format for reference.
        
        Returns:
            Optional[pd.DataFrame]: Sample format DataFrame if available
        """
        try:
            sample_path = Path(config.data.sample_test_out_file)
            if sample_path.exists():
                sample_df = pd.read_csv(sample_path)
                self.logger.info(f"Loaded sample format: {sample_df.shape}")
                return sample_df
            else:
                self.logger.warning(f"Sample format file not found: {sample_path}")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to load sample format: {str(e)}")
            return None
    
    def get_format_summary(self) -> Dict[str, Any]:
        """
        Get summary of format requirements.
        
        Returns:
            Dict: Format requirements summary
        """
        return {
            'required_columns': ['sample_id', 'price'],
            'column_order': ['sample_id', 'price'],
            'sample_id_type': 'string',
            'price_type': 'float',
            'price_precision': self.output_precision,
            'no_duplicates': True,
            'no_null_values': True,
            'positive_prices_only': True,
            'exact_row_count_match': True,
            'exact_sample_id_match': True
        }