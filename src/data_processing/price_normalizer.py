"""
Advanced price normalization and cleaning for ML Product Pricing Challenge 2025
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
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


class PriceNormalizationError(Exception):
    """Custom exception for price normalization errors"""
    pass


class PriceNormalizer:
    """
    Advanced price normalization and cleaning component
    
    Creates price normalization functions to strip currency symbols and thousand separators.
    Implements zero price detection and documented handling strategy (drop/epsilon/special class).
    Adds anomaly logging and data quality reporting.
    """
    
    def __init__(self, data_config=None):
        """Initialize PriceNormalizer with configuration"""
        self.config = data_config or config.data
        self.logger = logging.getLogger(__name__)
        
        # Currency symbols and patterns to remove
        self.currency_patterns = [
            r'\$',           # Dollar sign
            r'€',            # Euro
            r'£',            # Pound
            r'¥',            # Yen
            r'₹',            # Rupee
            r'USD',          # USD text
            r'EUR',          # EUR text
            r'GBP',          # GBP text
            r'CAD',          # CAD text
            r'AUD',          # AUD text
        ]
        
        # Thousand separators to remove
        self.separator_patterns = [
            r',',            # Comma (1,000.00)
            r'\s',           # Space (1 000.00)
            r"'",            # Apostrophe (1'000.00)
        ]
        
        # Anomaly tracking
        self.anomalies = {
            'zero_prices': [],
            'negative_prices': [],
            'extreme_high_prices': [],
            'extreme_low_prices': [],
            'invalid_formats': [],
            'currency_symbols_found': [],
            'thousand_separators_found': []
        }
    
    def normalize_price_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize price formatting by stripping currency symbols and thousand separators
        
        Args:
            df: DataFrame with price column to normalize
            
        Returns:
            pd.DataFrame: DataFrame with normalized prices
        """
        if 'price' not in df.columns:
            self.logger.warning("No 'price' column found in DataFrame")
            return df
        
        self.logger.info("Starting price normalization")
        df_normalized = df.copy()
        
        # Convert price column to string for processing
        price_series = df_normalized['price'].astype(str)
        
        # Track original values for anomaly detection
        original_prices = price_series.copy()
        
        # Remove currency symbols
        for pattern in self.currency_patterns:
            matches_before = price_series.str.contains(pattern, regex=True, na=False).sum()
            if matches_before > 0:
                self.anomalies['currency_symbols_found'].append({
                    'pattern': pattern,
                    'count': matches_before,
                    'samples': price_series[price_series.str.contains(pattern, regex=True, na=False)].head(5).tolist()
                })
                price_series = price_series.str.replace(pattern, '', regex=True)
                self.logger.info(f"Removed currency pattern '{pattern}' from {matches_before} prices")
        
        # Remove thousand separators (but preserve decimal points)
        for pattern in self.separator_patterns:
            if pattern == r',':
                # Special handling for commas - only remove if not decimal separator
                # Remove commas that are followed by exactly 3 digits and then either another comma or end
                comma_pattern = r',(?=\d{3}(?:,|\s|$))'
                matches_before = price_series.str.contains(comma_pattern, regex=True, na=False).sum()
                if matches_before > 0:
                    self.anomalies['thousand_separators_found'].append({
                        'pattern': 'comma_thousands',
                        'count': matches_before,
                        'samples': price_series[price_series.str.contains(comma_pattern, regex=True, na=False)].head(5).tolist()
                    })
                    price_series = price_series.str.replace(comma_pattern, '', regex=True)
                    self.logger.info(f"Removed thousand separator commas from {matches_before} prices")
            else:
                matches_before = price_series.str.contains(pattern, regex=True, na=False).sum()
                if matches_before > 0:
                    self.anomalies['thousand_separators_found'].append({
                        'pattern': pattern,
                        'count': matches_before,
                        'samples': price_series[price_series.str.contains(pattern, regex=True, na=False)].head(5).tolist()
                    })
                    price_series = price_series.str.replace(pattern, '', regex=True)
                    self.logger.info(f"Removed separator pattern '{pattern}' from {matches_before} prices")
        
        # Convert to numeric, handling errors
        numeric_prices = pd.to_numeric(price_series, errors='coerce')
        
        # Track invalid formats
        invalid_mask = numeric_prices.isna() & price_series.notna()
        if invalid_mask.sum() > 0:
            invalid_samples = original_prices[invalid_mask].head(10).tolist()
            self.anomalies['invalid_formats'].extend(invalid_samples)
            self.logger.warning(f"Found {invalid_mask.sum()} prices with invalid format")
        
        # Update DataFrame
        df_normalized['price'] = numeric_prices
        
        self.logger.info(f"Price normalization completed. Processed {len(df)} samples")
        return df_normalized
    
    def handle_zero_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle zero prices with documented strategy
        
        Args:
            df: DataFrame with normalized prices
            
        Returns:
            pd.DataFrame: DataFrame with zero prices handled according to strategy
        """
        if 'price' not in df.columns:
            return df
        
        # Identify zero prices
        zero_mask = (df['price'] == 0) | (df['price'].isna())
        zero_count = zero_mask.sum()
        
        if zero_count == 0:
            self.logger.info("No zero prices found")
            return df
        
        self.logger.info(f"Found {zero_count} zero/missing prices")
        
        # Store zero price samples for anomaly tracking
        zero_samples = df[zero_mask]['sample_id'].tolist()
        self.anomalies['zero_prices'].extend(zero_samples)
        
        df_handled = df.copy()
        
        if self.config.zero_price_strategy == "drop":
            # Drop samples with zero prices
            df_handled = df_handled[~zero_mask]
            self.logger.info(f"Dropped {zero_count} samples with zero prices")
            
        elif self.config.zero_price_strategy == "epsilon":
            # Replace zero prices with small epsilon value
            epsilon = self.config.zero_price_epsilon
            df_handled.loc[zero_mask, 'price'] = epsilon
            self.logger.info(f"Replaced {zero_count} zero prices with epsilon value: {epsilon}")
            
        elif self.config.zero_price_strategy == "special_class":
            # Keep zero prices but mark them for special handling
            df_handled['is_zero_price'] = zero_mask
            self.logger.info(f"Marked {zero_count} zero prices for special handling")
            
        else:
            raise PriceNormalizationError(
                f"Unknown zero price strategy: {self.config.zero_price_strategy}. "
                f"Valid options: 'drop', 'epsilon', 'special_class'"
            )
        
        return df_handled
    
    def detect_price_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and log price anomalies for data quality reporting
        
        Args:
            df: DataFrame with prices to analyze
            
        Returns:
            Dict containing anomaly detection results
        """
        if 'price' not in df.columns or df['price'].empty:
            return {'anomalies_found': False, 'details': {}}
        
        anomaly_report = {
            'anomalies_found': False,
            'total_samples': len(df),
            'details': {}
        }
        
        prices = df['price'].dropna()
        if len(prices) == 0:
            return anomaly_report
        
        # Calculate price statistics
        price_stats = prices.describe()
        q01, q99 = prices.quantile([0.01, 0.99])
        iqr = price_stats['75%'] - price_stats['25%']
        
        # Detect negative prices
        negative_mask = prices < 0
        negative_count = negative_mask.sum()
        if negative_count > 0:
            negative_samples = df[df['price'] < 0]['sample_id'].tolist()
            self.anomalies['negative_prices'].extend(negative_samples)
            anomaly_report['details']['negative_prices'] = {
                'count': negative_count,
                'percentage': (negative_count / len(prices)) * 100,
                'samples': negative_samples[:10]  # First 10 samples
            }
            anomaly_report['anomalies_found'] = True
        
        # Detect extremely high prices (>10x 99th percentile)
        extreme_high_threshold = q99 * 10
        extreme_high_mask = prices > extreme_high_threshold
        extreme_high_count = extreme_high_mask.sum()
        if extreme_high_count > 0:
            extreme_high_samples = df[df['price'] > extreme_high_threshold]['sample_id'].tolist()
            self.anomalies['extreme_high_prices'].extend(extreme_high_samples)
            anomaly_report['details']['extreme_high_prices'] = {
                'count': extreme_high_count,
                'percentage': (extreme_high_count / len(prices)) * 100,
                'threshold': extreme_high_threshold,
                'max_price': prices.max(),
                'samples': extreme_high_samples[:10]
            }
            anomaly_report['anomalies_found'] = True
        
        # Detect extremely low prices (<1/10th 1st percentile, but not zero)
        extreme_low_threshold = max(q01 / 10, 0.001)  # At least 0.001
        extreme_low_mask = (prices > 0) & (prices < extreme_low_threshold)
        extreme_low_count = extreme_low_mask.sum()
        if extreme_low_count > 0:
            extreme_low_samples = df[(df['price'] > 0) & (df['price'] < extreme_low_threshold)]['sample_id'].tolist()
            self.anomalies['extreme_low_prices'].extend(extreme_low_samples)
            anomaly_report['details']['extreme_low_prices'] = {
                'count': extreme_low_count,
                'percentage': (extreme_low_count / len(prices)) * 100,
                'threshold': extreme_low_threshold,
                'min_price': prices[prices > 0].min(),
                'samples': extreme_low_samples[:10]
            }
            anomaly_report['anomalies_found'] = True
        
        # Detect outliers using IQR method
        lower_bound = price_stats['25%'] - 1.5 * iqr
        upper_bound = price_stats['75%'] + 1.5 * iqr
        outlier_mask = (prices < lower_bound) | (prices > upper_bound)
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            anomaly_report['details']['iqr_outliers'] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(prices)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        # Add price distribution summary
        anomaly_report['price_distribution'] = {
            'mean': price_stats['mean'],
            'median': price_stats['50%'],
            'std': price_stats['std'],
            'min': price_stats['min'],
            'max': price_stats['max'],
            'q01': q01,
            'q99': q99,
            'zero_count': (prices == 0).sum()
        }
        
        return anomaly_report
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing comprehensive data quality metrics
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_samples': len(df),
            'data_quality_score': 0.0,
            'issues_found': []
        }
        
        # Price quality analysis
        if 'price' in df.columns:
            price_analysis = self.detect_price_anomalies(df)
            report['price_anomalies'] = price_analysis
            
            # Calculate price quality score
            valid_prices = df['price'].notna().sum()
            positive_prices = (df['price'] > 0).sum()
            price_quality = (positive_prices / len(df)) * 100 if len(df) > 0 else 0
            report['price_quality_percentage'] = price_quality
            
            if price_quality < 95:
                report['issues_found'].append(f"Price quality below 95%: {price_quality:.1f}%")
        
        # Normalization summary
        report['normalization_summary'] = {
            'currency_symbols_removed': len(self.anomalies['currency_symbols_found']),
            'thousand_separators_removed': len(self.anomalies['thousand_separators_found']),
            'invalid_formats_found': len(self.anomalies['invalid_formats']),
            'zero_prices_handled': len(self.anomalies['zero_prices']),
            'negative_prices_found': len(self.anomalies['negative_prices'])
        }
        
        # Calculate overall data quality score
        quality_factors = []
        
        if 'price' in df.columns:
            # Price completeness (0-40 points)
            price_completeness = (df['price'].notna().sum() / len(df)) * 40
            quality_factors.append(price_completeness)
            
            # Price validity (0-30 points)
            valid_prices = ((df['price'] >= 0) & df['price'].notna()).sum()
            price_validity = (valid_prices / len(df)) * 30
            quality_factors.append(price_validity)
            
            # Anomaly penalty (0-20 points, higher is better)
            anomaly_rate = sum(len(anomalies) for anomalies in self.anomalies.values()) / len(df)
            anomaly_score = max(0, 20 - (anomaly_rate * 100))  # Penalty for high anomaly rate
            quality_factors.append(anomaly_score)
        
        # Data consistency (0-10 points)
        consistency_score = 10  # Start with full points
        if len(self.anomalies['invalid_formats']) > len(df) * 0.01:  # >1% invalid formats
            consistency_score -= 5
        quality_factors.append(consistency_score)
        
        report['data_quality_score'] = sum(quality_factors)
        
        # Add recommendations
        recommendations = []
        if report['data_quality_score'] < 80:
            recommendations.append("Data quality score is below 80. Consider additional cleaning.")
        if len(self.anomalies['zero_prices']) > len(df) * 0.05:  # >5% zero prices
            recommendations.append("High percentage of zero prices detected. Review pricing strategy.")
        if len(self.anomalies['extreme_high_prices']) > 0:
            recommendations.append("Extreme high prices detected. Verify these are not data entry errors.")
        if len(self.anomalies['invalid_formats']) > 0:
            recommendations.append("Invalid price formats found. Check data source quality.")
        
        report['recommendations'] = recommendations
        
        return report
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get summary of all detected anomalies
        
        Returns:
            Dict containing anomaly summary
        """
        return {
            'anomaly_counts': {key: len(value) for key, value in self.anomalies.items()},
            'anomaly_details': self.anomalies.copy(),
            'total_anomalies': sum(len(value) for value in self.anomalies.values())
        }
    
    def reset_anomaly_tracking(self):
        """Reset anomaly tracking for new dataset"""
        for key in self.anomalies:
            self.anomalies[key] = []
        self.logger.info("Anomaly tracking reset")
    
    def validate_price_range(self, df: pd.DataFrame, min_price: float = 0.01, 
                           max_price: Optional[float] = None) -> Tuple[bool, List[str]]:
        """
        Validate that all prices are within acceptable range
        
        Args:
            df: DataFrame with prices to validate
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price (optional)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if 'price' not in df.columns:
            return True, []
        
        issues = []
        prices = df['price'].dropna()
        
        if len(prices) == 0:
            issues.append("No valid prices found")
            return False, issues
        
        # Check minimum price
        below_min = (prices < min_price).sum()
        if below_min > 0:
            issues.append(f"{below_min} prices below minimum threshold {min_price}")
        
        # Check maximum price if specified
        if max_price is not None:
            above_max = (prices > max_price).sum()
            if above_max > 0:
                issues.append(f"{above_max} prices above maximum threshold {max_price}")
        
        # Check for negative prices
        negative = (prices < 0).sum()
        if negative > 0:
            issues.append(f"{negative} negative prices found")
        
        return len(issues) == 0, issues