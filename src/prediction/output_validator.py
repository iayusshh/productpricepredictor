"""
Comprehensive output validation system for ML Product Pricing Challenge 2025

This module implements comprehensive validation of prediction outputs including
sample_id verification, prediction range validation, outlier detection, and
final quality assurance checks before submission.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import json
from datetime import datetime
import warnings

from ..config import config


class OutputValidator:
    """
    Comprehensive output validation system for submission compliance.
    
    This class performs extensive validation of prediction outputs to ensure
    compliance with all submission requirements and identify potential issues.
    """
    
    def __init__(self, 
                 strict_mode: bool = True,
                 outlier_detection: bool = True,
                 generate_report: bool = True):
        """
        Initialize OutputValidator with configuration.
        
        Args:
            strict_mode: Whether to enforce strict validation rules
            outlier_detection: Whether to perform outlier detection
            generate_report: Whether to generate validation reports
        """
        self.strict_mode = strict_mode
        self.outlier_detection = outlier_detection
        self.generate_report = generate_report
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validation results storage
        self.validation_results = {}
        self.validation_warnings = []
        self.validation_errors = []
        
        # Outlier detection parameters
        self.outlier_methods = ['iqr', 'zscore', 'isolation_forest']
        self.outlier_thresholds = {
            'iqr_multiplier': 3.0,
            'zscore_threshold': 3.0,
            'isolation_contamination': 0.1
        }
        
        self.logger.info(f"OutputValidator initialized with strict_mode={strict_mode}")
    
    def validate_complete_output(self, 
                                output_df: pd.DataFrame, 
                                test_df: pd.DataFrame,
                                predictions: np.ndarray = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of complete output.
        
        Args:
            output_df: Formatted output DataFrame
            test_df: Original test DataFrame
            predictions: Optional raw predictions for additional analysis
            
        Returns:
            Dict: Comprehensive validation results
        """
        self.logger.info("Starting comprehensive output validation")
        
        # Reset validation state
        self._reset_validation_state()
        
        # Core validations
        self.validation_results['sample_id_validation'] = self._validate_sample_ids(output_df, test_df)
        self.validation_results['row_count_validation'] = self._validate_row_counts(output_df, test_df)
        self.validation_results['data_type_validation'] = self._validate_data_types(output_df)
        self.validation_results['value_range_validation'] = self._validate_value_ranges(output_df)
        self.validation_results['format_compliance'] = self._validate_format_compliance(output_df)
        self.validation_results['duplicate_validation'] = self._validate_duplicates(output_df)
        self.validation_results['completeness_validation'] = self._validate_completeness(output_df)
        
        # Advanced validations
        if self.outlier_detection:
            self.validation_results['outlier_analysis'] = self._detect_outliers(output_df)
        
        self.validation_results['distribution_analysis'] = self._analyze_prediction_distribution(output_df)
        self.validation_results['consistency_checks'] = self._perform_consistency_checks(output_df, test_df)
        
        # Additional analysis if raw predictions provided
        if predictions is not None:
            self.validation_results['prediction_analysis'] = self._analyze_raw_predictions(predictions, output_df)
        
        # File integrity checks
        self.validation_results['file_integrity'] = self._validate_file_integrity(output_df)
        
        # Overall validation summary
        self.validation_results['overall_status'] = self._compute_overall_status()
        self.validation_results['validation_timestamp'] = datetime.now().isoformat()
        
        # Generate report if requested
        if self.generate_report:
            self._generate_validation_report()
        
        self.logger.info(f"Validation completed. Overall status: {self.validation_results['overall_status']['status']}")
        
        return self.validation_results
    
    def verify_exact_sample_id_match(self, 
                                   output_df: pd.DataFrame, 
                                   test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Verify output contains exactly the same sample_ids as test.csv.
        
        Args:
            output_df: Output DataFrame
            test_df: Test DataFrame
            
        Returns:
            Dict: Detailed sample_id validation results
        """
        self.logger.info("Verifying exact sample_id match")
        
        # Convert to string for consistent comparison
        output_ids = set(output_df['sample_id'].astype(str))
        test_ids = set(test_df['sample_id'].astype(str))
        
        # Find differences
        missing_in_output = test_ids - output_ids
        extra_in_output = output_ids - test_ids
        common_ids = output_ids & test_ids
        
        # Detailed analysis
        results = {
            'exact_match': len(missing_in_output) == 0 and len(extra_in_output) == 0,
            'total_test_ids': len(test_ids),
            'total_output_ids': len(output_ids),
            'common_ids_count': len(common_ids),
            'missing_in_output_count': len(missing_in_output),
            'extra_in_output_count': len(extra_in_output),
            'missing_ids': sorted(list(missing_in_output))[:100],  # Limit for logging
            'extra_ids': sorted(list(extra_in_output))[:100],
            'coverage_percentage': len(common_ids) / len(test_ids) * 100 if test_ids else 0
        }
        
        if results['exact_match']:
            self.logger.info("✓ Sample ID verification passed: exact match")
        else:
            self.logger.error("✗ Sample ID verification failed:")
            self.logger.error(f"  Missing: {results['missing_in_output_count']} IDs")
            self.logger.error(f"  Extra: {results['extra_in_output_count']} IDs")
            self.logger.error(f"  Coverage: {results['coverage_percentage']:.2f}%")
        
        return results
    
    def validate_prediction_ranges(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate prediction ranges and identify potential outliers.
        
        Args:
            output_df: Output DataFrame with predictions
            
        Returns:
            Dict: Range validation results
        """
        self.logger.info("Validating prediction ranges")
        
        prices = output_df['price'].values
        
        # Basic statistics
        stats = {
            'count': len(prices),
            'mean': float(np.mean(prices)),
            'std': float(np.std(prices)),
            'min': float(np.min(prices)),
            'max': float(np.max(prices)),
            'median': float(np.median(prices)),
            'q25': float(np.percentile(prices, 25)),
            'q75': float(np.percentile(prices, 75)),
            'iqr': float(np.percentile(prices, 75) - np.percentile(prices, 25))
        }
        
        # Range validations
        validations = {
            'all_positive': np.all(prices > 0),
            'no_zero_values': np.all(prices != 0),
            'no_negative_values': np.all(prices >= 0),
            'no_infinite_values': np.all(np.isfinite(prices)),
            'no_nan_values': not np.any(np.isnan(prices)),
            'within_reasonable_range': np.all((prices >= 0.01) & (prices <= 10000))  # Reasonable price range
        }
        
        # Count violations
        violations = {
            'negative_count': int(np.sum(prices < 0)),
            'zero_count': int(np.sum(prices == 0)),
            'infinite_count': int(np.sum(~np.isfinite(prices))),
            'nan_count': int(np.sum(np.isnan(prices))),
            'below_threshold_count': int(np.sum(prices < config.prediction.min_price_threshold)),
            'extremely_high_count': int(np.sum(prices > 1000))  # Potentially unrealistic prices
        }
        
        # Overall range validation
        range_valid = all(validations.values())
        
        results = {
            'range_valid': range_valid,
            'statistics': stats,
            'validations': validations,
            'violations': violations
        }
        
        if range_valid:
            self.logger.info("✓ Prediction range validation passed")
        else:
            self.logger.warning("⚠ Prediction range validation found issues:")
            for key, valid in validations.items():
                if not valid:
                    self.logger.warning(f"  {key}: FAILED")
        
        return results
    
    def create_submission_integrity_checks(self, 
                                         output_df: pd.DataFrame,
                                         test_df: pd.DataFrame,
                                         output_file: str = None) -> Dict[str, Any]:
        """
        Create comprehensive submission file integrity checks.
        
        Args:
            output_df: Output DataFrame
            test_df: Test DataFrame
            output_file: Path to output file for file-level checks
            
        Returns:
            Dict: Integrity check results
        """
        self.logger.info("Performing submission integrity checks")
        
        integrity_results = {}
        
        # DataFrame integrity
        integrity_results['dataframe_checks'] = {
            'not_empty': len(output_df) > 0,
            'has_required_columns': all(col in output_df.columns for col in ['sample_id', 'price']),
            'correct_column_count': len(output_df.columns) == 2,
            'correct_column_order': list(output_df.columns) == ['sample_id', 'price'],
            'no_duplicate_columns': len(output_df.columns) == len(set(output_df.columns))
        }
        
        # Data integrity
        integrity_results['data_checks'] = {
            'no_null_sample_ids': not output_df['sample_id'].isnull().any(),
            'no_null_prices': not output_df['price'].isnull().any(),
            'no_empty_sample_ids': not output_df['sample_id'].isin(['', ' ']).any(),
            'sample_ids_are_strings': pd.api.types.is_string_dtype(output_df['sample_id']) or pd.api.types.is_object_dtype(output_df['sample_id']),
            'prices_are_numeric': pd.api.types.is_numeric_dtype(output_df['price'])
        }
        
        # Relationship integrity
        integrity_results['relationship_checks'] = {
            'exact_sample_id_match': self.verify_exact_sample_id_match(output_df, test_df)['exact_match'],
            'exact_row_count_match': len(output_df) == len(test_df),
            'no_duplicate_sample_ids': not output_df['sample_id'].duplicated().any(),
            'sample_id_uniqueness': len(output_df['sample_id'].unique()) == len(output_df)
        }
        
        # File integrity (if file path provided)
        if output_file and Path(output_file).exists():
            integrity_results['file_checks'] = self._check_file_integrity(output_file, output_df)
        
        # Overall integrity status
        all_checks = []
        for category in integrity_results.values():
            if isinstance(category, dict):
                all_checks.extend(category.values())
        
        integrity_results['overall_integrity'] = {
            'all_checks_passed': all(all_checks),
            'total_checks': len(all_checks),
            'passed_checks': sum(all_checks),
            'failed_checks': len(all_checks) - sum(all_checks)
        }
        
        if integrity_results['overall_integrity']['all_checks_passed']:
            self.logger.info("✓ All submission integrity checks passed")
        else:
            failed = integrity_results['overall_integrity']['failed_checks']
            total = integrity_results['overall_integrity']['total_checks']
            self.logger.error(f"✗ Submission integrity checks failed: {failed}/{total} checks failed")
        
        return integrity_results
    
    def perform_final_quality_assurance(self, 
                                      output_df: pd.DataFrame,
                                      test_df: pd.DataFrame,
                                      output_file: str = None) -> Dict[str, Any]:
        """
        Perform final quality assurance checks before submission.
        
        Args:
            output_df: Final output DataFrame
            test_df: Original test DataFrame
            output_file: Path to output file
            
        Returns:
            Dict: Final QA results with pass/fail status
        """
        self.logger.info("Performing final quality assurance checks")
        
        qa_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'summary': {},
            'recommendations': []
        }
        
        # Critical checks (must pass)
        critical_checks = {
            'sample_id_exact_match': self.verify_exact_sample_id_match(output_df, test_df)['exact_match'],
            'row_count_match': len(output_df) == len(test_df),
            'no_null_values': not output_df.isnull().any().any(),
            'positive_prices': (output_df['price'] > 0).all(),
            'no_duplicates': not output_df['sample_id'].duplicated().any(),
            'correct_format': self._validate_submission_format(output_df)
        }
        
        # Important checks (should pass)
        important_checks = {
            'reasonable_price_range': ((output_df['price'] >= 0.01) & (output_df['price'] <= 1000)).all(),
            'no_extreme_outliers': self._check_extreme_outliers(output_df['price']),
            'consistent_precision': self._check_price_precision(output_df['price']),
            'file_readable': self._check_file_readability(output_file) if output_file else True
        }
        
        # Optional checks (nice to have)
        optional_checks = {
            'distribution_reasonable': self._check_distribution_reasonableness(output_df['price']),
            'no_repeated_values': len(output_df['price'].unique()) > len(output_df) * 0.5,
            'smooth_distribution': self._check_distribution_smoothness(output_df['price'])
        }
        
        qa_results['checks']['critical'] = critical_checks
        qa_results['checks']['important'] = important_checks
        qa_results['checks']['optional'] = optional_checks
        
        # Compute summary
        critical_passed = sum(critical_checks.values())
        important_passed = sum(important_checks.values())
        optional_passed = sum(optional_checks.values())
        
        qa_results['summary'] = {
            'critical_checks': {
                'total': len(critical_checks),
                'passed': critical_passed,
                'failed': len(critical_checks) - critical_passed,
                'pass_rate': critical_passed / len(critical_checks)
            },
            'important_checks': {
                'total': len(important_checks),
                'passed': important_passed,
                'failed': len(important_checks) - important_passed,
                'pass_rate': important_passed / len(important_checks)
            },
            'optional_checks': {
                'total': len(optional_checks),
                'passed': optional_passed,
                'failed': len(optional_checks) - optional_passed,
                'pass_rate': optional_passed / len(optional_checks)
            }
        }
        
        # Overall QA status
        all_critical_passed = qa_results['summary']['critical_checks']['pass_rate'] == 1.0
        most_important_passed = qa_results['summary']['important_checks']['pass_rate'] >= 0.8
        
        if all_critical_passed and most_important_passed:
            qa_status = 'PASS'
            qa_message = "All critical checks passed, ready for submission"
        elif all_critical_passed:
            qa_status = 'PASS_WITH_WARNINGS'
            qa_message = "Critical checks passed but some important checks failed"
        else:
            qa_status = 'FAIL'
            qa_message = "Critical checks failed, submission not recommended"
        
        qa_results['overall_status'] = qa_status
        qa_results['status_message'] = qa_message
        
        # Generate recommendations
        qa_results['recommendations'] = self._generate_qa_recommendations(qa_results['checks'])
        
        # Log results
        self.logger.info(f"Final QA Status: {qa_status}")
        self.logger.info(f"Critical: {critical_passed}/{len(critical_checks)} passed")
        self.logger.info(f"Important: {important_passed}/{len(important_checks)} passed")
        self.logger.info(f"Optional: {optional_passed}/{len(optional_checks)} passed")
        
        if qa_status == 'FAIL':
            self.logger.error("❌ Final QA FAILED - submission not recommended")
            for check, passed in critical_checks.items():
                if not passed:
                    self.logger.error(f"  Critical check failed: {check}")
        elif qa_status == 'PASS_WITH_WARNINGS':
            self.logger.warning("⚠️ Final QA PASSED WITH WARNINGS")
        else:
            self.logger.info("✅ Final QA PASSED - ready for submission")
        
        return qa_results
    
    def _validate_sample_ids(self, output_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate sample IDs comprehensively."""
        return self.verify_exact_sample_id_match(output_df, test_df)
    
    def _validate_row_counts(self, output_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate row counts match exactly."""
        output_count = len(output_df)
        test_count = len(test_df)
        
        return {
            'match': output_count == test_count,
            'output_count': output_count,
            'test_count': test_count,
            'difference': output_count - test_count
        }
    
    def _validate_data_types(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types are correct."""
        return {
            'sample_id_type_valid': pd.api.types.is_string_dtype(output_df['sample_id']) or pd.api.types.is_object_dtype(output_df['sample_id']),
            'price_type_valid': pd.api.types.is_numeric_dtype(output_df['price']),
            'sample_id_dtype': str(output_df['sample_id'].dtype),
            'price_dtype': str(output_df['price'].dtype)
        }
    
    def _validate_value_ranges(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate value ranges are appropriate."""
        return self.validate_prediction_ranges(output_df)
    
    def _validate_format_compliance(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate format compliance."""
        return {
            'correct_columns': list(output_df.columns) == ['sample_id', 'price'],
            'column_count': len(output_df.columns) == 2,
            'no_extra_columns': len(output_df.columns) <= 2,
            'column_order_correct': list(output_df.columns) == ['sample_id', 'price']
        }
    
    def _validate_duplicates(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate no duplicate entries."""
        duplicate_sample_ids = output_df['sample_id'].duplicated().sum()
        
        return {
            'no_duplicate_sample_ids': duplicate_sample_ids == 0,
            'duplicate_count': int(duplicate_sample_ids),
            'unique_sample_ids': len(output_df['sample_id'].unique()),
            'total_rows': len(output_df)
        }
    
    def _validate_completeness(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness."""
        return {
            'no_null_sample_ids': not output_df['sample_id'].isnull().any(),
            'no_null_prices': not output_df['price'].isnull().any(),
            'no_empty_sample_ids': not output_df['sample_id'].isin(['', ' ', None]).any(),
            'all_rows_complete': not output_df.isnull().any().any()
        }
    
    def _detect_outliers(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in predictions."""
        prices = output_df['price'].values
        outlier_results = {}
        
        # IQR method
        q25, q75 = np.percentile(prices, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - self.outlier_thresholds['iqr_multiplier'] * iqr
        upper_bound = q75 + self.outlier_thresholds['iqr_multiplier'] * iqr
        iqr_outliers = (prices < lower_bound) | (prices > upper_bound)
        
        outlier_results['iqr'] = {
            'outlier_count': int(np.sum(iqr_outliers)),
            'outlier_percentage': float(np.sum(iqr_outliers) / len(prices) * 100),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
        
        # Z-score method
        z_scores = np.abs((prices - np.mean(prices)) / np.std(prices))
        zscore_outliers = z_scores > self.outlier_thresholds['zscore_threshold']
        
        outlier_results['zscore'] = {
            'outlier_count': int(np.sum(zscore_outliers)),
            'outlier_percentage': float(np.sum(zscore_outliers) / len(prices) * 100),
            'threshold': self.outlier_thresholds['zscore_threshold']
        }
        
        return outlier_results
    
    def _analyze_prediction_distribution(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction distribution."""
        prices = output_df['price'].values
        
        return {
            'mean': float(np.mean(prices)),
            'std': float(np.std(prices)),
            'skewness': float(self._calculate_skewness(prices)),
            'kurtosis': float(self._calculate_kurtosis(prices)),
            'percentiles': {
                'p1': float(np.percentile(prices, 1)),
                'p5': float(np.percentile(prices, 5)),
                'p10': float(np.percentile(prices, 10)),
                'p25': float(np.percentile(prices, 25)),
                'p50': float(np.percentile(prices, 50)),
                'p75': float(np.percentile(prices, 75)),
                'p90': float(np.percentile(prices, 90)),
                'p95': float(np.percentile(prices, 95)),
                'p99': float(np.percentile(prices, 99))
            }
        }
    
    def _perform_consistency_checks(self, output_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform consistency checks."""
        return {
            'sample_id_format_consistent': self._check_sample_id_format_consistency(output_df, test_df),
            'price_precision_consistent': self._check_price_precision_consistency(output_df),
            'no_systematic_errors': self._check_systematic_errors(output_df)
        }
    
    def _analyze_raw_predictions(self, predictions: np.ndarray, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze raw predictions vs formatted output."""
        formatted_prices = output_df['price'].values
        
        return {
            'clamping_applied': not np.array_equal(predictions, formatted_prices),
            'clamped_count': int(np.sum(predictions != formatted_prices)),
            'original_negative_count': int(np.sum(predictions < 0)),
            'original_zero_count': int(np.sum(predictions == 0)),
            'transformation_summary': {
                'min_original': float(np.min(predictions)),
                'min_formatted': float(np.min(formatted_prices)),
                'max_original': float(np.max(predictions)),
                'max_formatted': float(np.max(formatted_prices))
            }
        }
    
    def _validate_file_integrity(self, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate file integrity."""
        return {
            'dataframe_not_empty': len(output_df) > 0,
            'memory_usage_reasonable': output_df.memory_usage(deep=True).sum() < 1e9,  # < 1GB
            'no_memory_issues': True  # Placeholder for memory-related checks
        }
    
    def _compute_overall_status(self) -> Dict[str, Any]:
        """Compute overall validation status."""
        critical_validations = [
            self.validation_results.get('sample_id_validation', {}).get('exact_match', False),
            self.validation_results.get('row_count_validation', {}).get('match', False),
            self.validation_results.get('value_range_validation', {}).get('range_valid', False),
            self.validation_results.get('format_compliance', {}).get('correct_columns', False),
            self.validation_results.get('duplicate_validation', {}).get('no_duplicate_sample_ids', False),
            self.validation_results.get('completeness_validation', {}).get('all_rows_complete', False)
        ]
        
        critical_passed = sum(critical_validations)
        total_critical = len(critical_validations)
        
        if critical_passed == total_critical:
            status = 'PASS'
            message = 'All critical validations passed'
        elif critical_passed >= total_critical * 0.8:
            status = 'PASS_WITH_WARNINGS'
            message = 'Most critical validations passed'
        else:
            status = 'FAIL'
            message = 'Critical validations failed'
        
        return {
            'status': status,
            'message': message,
            'critical_passed': critical_passed,
            'critical_total': total_critical,
            'pass_rate': critical_passed / total_critical
        }
    
    def _reset_validation_state(self):
        """Reset validation state for new validation run."""
        self.validation_results = {}
        self.validation_warnings = []
        self.validation_errors = []
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        if not self.generate_report:
            return
        
        report_path = Path(config.infrastructure.log_dir) / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {str(e)}")
    
    # Helper methods for specific checks
    def _validate_submission_format(self, output_df: pd.DataFrame) -> bool:
        """Validate submission format."""
        return (list(output_df.columns) == ['sample_id', 'price'] and
                len(output_df.columns) == 2 and
                not output_df.isnull().any().any())
    
    def _check_extreme_outliers(self, prices: pd.Series) -> bool:
        """Check for extreme outliers."""
        q99 = prices.quantile(0.99)
        q01 = prices.quantile(0.01)
        extreme_high = (prices > q99 * 10).sum()
        extreme_low = (prices < q01 / 10).sum()
        return extreme_high == 0 and extreme_low == 0
    
    def _check_price_precision(self, prices: pd.Series) -> bool:
        """Check price precision consistency."""
        # Check if all prices have reasonable precision (not too many decimal places)
        decimal_places = prices.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
        return decimal_places.max() <= 10  # Reasonable precision limit
    
    def _check_file_readability(self, file_path: str) -> bool:
        """Check if file is readable."""
        if not file_path or not Path(file_path).exists():
            return False
        
        try:
            test_df = pd.read_csv(file_path, nrows=5)
            return len(test_df) > 0
        except Exception:
            return False
    
    def _check_distribution_reasonableness(self, prices: pd.Series) -> bool:
        """Check if distribution is reasonable."""
        # Basic reasonableness checks
        cv = prices.std() / prices.mean()  # Coefficient of variation
        return 0.1 <= cv <= 2.0  # Reasonable variation
    
    def _check_distribution_smoothness(self, prices: pd.Series) -> bool:
        """Check distribution smoothness."""
        # Simple check for too many repeated values
        unique_ratio = len(prices.unique()) / len(prices)
        return unique_ratio > 0.1  # At least 10% unique values
    
    def _check_file_integrity(self, file_path: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Check file integrity."""
        try:
            file_df = pd.read_csv(file_path)
            return {
                'file_readable': True,
                'correct_shape': file_df.shape == original_df.shape,
                'correct_columns': list(file_df.columns) == list(original_df.columns),
                'file_size_reasonable': Path(file_path).stat().st_size < 100 * 1024 * 1024  # < 100MB
            }
        except Exception as e:
            return {
                'file_readable': False,
                'error': str(e)
            }
    
    def _generate_qa_recommendations(self, checks: Dict[str, Dict[str, bool]]) -> List[str]:
        """Generate QA recommendations based on check results."""
        recommendations = []
        
        # Check critical failures
        for check, passed in checks.get('critical', {}).items():
            if not passed:
                if 'sample_id' in check:
                    recommendations.append("Fix sample_id matching issues - ensure all test.csv sample_ids are present")
                elif 'row_count' in check:
                    recommendations.append("Fix row count mismatch - output must have same number of rows as test.csv")
                elif 'null' in check:
                    recommendations.append("Remove null values from output")
                elif 'positive' in check:
                    recommendations.append("Ensure all price predictions are positive")
                elif 'duplicate' in check:
                    recommendations.append("Remove duplicate sample_ids from output")
                elif 'format' in check:
                    recommendations.append("Fix output format - ensure columns are ['sample_id', 'price']")
        
        # Check important failures
        for check, passed in checks.get('important', {}).items():
            if not passed:
                if 'range' in check:
                    recommendations.append("Review price ranges - some predictions may be unrealistic")
                elif 'outlier' in check:
                    recommendations.append("Investigate extreme outliers in predictions")
                elif 'precision' in check:
                    recommendations.append("Check price precision - avoid excessive decimal places")
        
        if not recommendations:
            recommendations.append("All checks passed - output appears ready for submission")
        
        return recommendations
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _check_sample_id_format_consistency(self, output_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Check sample_id format consistency."""
        output_formats = output_df['sample_id'].astype(str).str.len().unique()
        test_formats = test_df['sample_id'].astype(str).str.len().unique()
        return len(set(output_formats) - set(test_formats)) == 0
    
    def _check_price_precision_consistency(self, output_df: pd.DataFrame) -> bool:
        """Check price precision consistency."""
        decimal_places = output_df['price'].apply(
            lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0
        )
        return decimal_places.std() < 2.0  # Reasonable consistency
    
    def _check_systematic_errors(self, output_df: pd.DataFrame) -> bool:
        """Check for systematic errors."""
        # Simple check for systematic patterns that might indicate errors
        prices = output_df['price'].values
        
        # Check for too many identical values
        most_common_count = pd.Series(prices).value_counts().iloc[0]
        identical_ratio = most_common_count / len(prices)
        
        # Check for arithmetic sequences (might indicate systematic error)
        sorted_prices = np.sort(prices)
        diffs = np.diff(sorted_prices)
        constant_diff_ratio = np.sum(np.abs(diffs - np.mean(diffs)) < 0.01) / len(diffs)
        
        return identical_ratio < 0.1 and constant_diff_ratio < 0.5