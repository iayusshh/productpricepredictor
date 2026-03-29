"""
Comprehensive data preprocessing component that integrates all preprocessing steps
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from ..interfaces import DataPreprocessorInterface
    from ..config import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from interfaces import DataPreprocessorInterface
    from config import config
from .data_loader import DataLoader, DataValidationError
from .data_cleaner import DataCleaner, DataCleaningError
from .image_downloader import ImageDownloader, ImageDownloadError
from .price_normalizer import PriceNormalizer, PriceNormalizationError


class DataPreprocessor(DataPreprocessorInterface):
    """
    Comprehensive data preprocessing component
    
    Integrates DataLoader, DataCleaner, ImageDownloader, and PriceNormalizer
    to provide a complete data preprocessing pipeline.
    """
    
    def __init__(self, data_config=None):
        """Initialize DataPreprocessor with all components"""
        self.config = data_config or config.data
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = DataLoader(data_config)
        self.data_cleaner = DataCleaner(data_config)
        self.image_downloader = ImageDownloader(data_config)
        self.price_normalizer = PriceNormalizer(data_config)
        
        # Processing statistics
        self.processing_stats = {
            'total_samples_loaded': 0,
            'samples_after_cleaning': 0,
            'samples_after_price_handling': 0,
            'images_downloaded': 0,
            'processing_time_seconds': 0.0
        }
    
    def load_training_data(self) -> pd.DataFrame:
        """Load training data using DataLoader"""
        return self.data_loader.load_training_data()
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data using DataLoader"""
        return self.data_loader.load_test_data()
    
    def validate_schema_and_types(self, df: pd.DataFrame) -> bool:
        """Validate schema and types using DataLoader"""
        return self.data_loader.validate_schema_and_types(df)
    
    def normalize_price_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize price formatting using PriceNormalizer"""
        return self.price_normalizer.normalize_price_formatting(df)
    
    def handle_zero_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle zero prices using PriceNormalizer"""
        return self.price_normalizer.handle_zero_prices(df)
    
    def clean_catalog_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean catalog content using DataCleaner"""
        return self.data_cleaner.clean_catalog_content(df)
    
    def download_images(self, df: pd.DataFrame, image_dir: str) -> Dict[str, str]:
        """Download images using ImageDownloader"""
        return self.image_downloader.download_images(df, image_dir)
    
    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """Validate data integrity using DataCleaner"""
        try:
            return self.data_cleaner.validate_sample_id_uniqueness(df)
        except DataCleaningError:
            return False
    
    def preprocess_training_data(self, download_images: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for training data
        
        Args:
            download_images: Whether to download images
            
        Returns:
            pd.DataFrame: Fully preprocessed training data
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting complete training data preprocessing pipeline")
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading training data")
            df = self.load_training_data()
            self.processing_stats['total_samples_loaded'] = len(df)
            
            # Step 2: Clean catalog content
            self.logger.info("Step 2: Cleaning catalog content")
            df = self.clean_catalog_content(df)
            df = self.data_cleaner.handle_missing_values(df)
            df = self.data_cleaner.standardize_text_format(df)
            df = self.data_cleaner.extract_structured_fields(df)
            
            # Step 3: Normalize and handle prices
            self.logger.info("Step 3: Normalizing prices")
            df = self.normalize_price_formatting(df)
            df = self.handle_zero_prices(df)
            self.processing_stats['samples_after_price_handling'] = len(df)
            
            # Step 4: Download images if requested
            if download_images:
                self.logger.info("Step 4: Downloading images")
                image_results = self.download_images(df, self.config.image_dir)
                df['image_download_status'] = df['sample_id'].map(image_results)
                successful_downloads = sum(1 for status in image_results.values() 
                                         if not status.startswith('error_'))
                self.processing_stats['images_downloaded'] = successful_downloads
                self.logger.info(f"Downloaded {successful_downloads}/{len(df)} images successfully")
            
            # Step 5: Final validation
            self.logger.info("Step 5: Final validation")
            self.validate_data_integrity(df)
            self.processing_stats['samples_after_cleaning'] = len(df)
            
            # Calculate processing time
            self.processing_stats['processing_time_seconds'] = time.time() - start_time
            
            self.logger.info(f"Training data preprocessing completed successfully. "
                           f"Final dataset: {len(df)} samples")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Training data preprocessing failed: {str(e)}")
            raise
    
    def preprocess_test_data(self, download_images: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for test data
        
        Args:
            download_images: Whether to download images
            
        Returns:
            pd.DataFrame: Fully preprocessed test data
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting complete test data preprocessing pipeline")
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading test data")
            df = self.load_test_data()
            self.processing_stats['total_samples_loaded'] = len(df)
            
            # Step 2: Clean catalog content
            self.logger.info("Step 2: Cleaning catalog content")
            df = self.clean_catalog_content(df)
            df = self.data_cleaner.handle_missing_values(df)
            df = self.data_cleaner.standardize_text_format(df)
            df = self.data_cleaner.extract_structured_fields(df)
            
            # Step 3: Download images if requested
            if download_images:
                self.logger.info("Step 3: Downloading images")
                image_results = self.download_images(df, self.config.image_dir)
                df['image_download_status'] = df['sample_id'].map(image_results)
                successful_downloads = sum(1 for status in image_results.values() 
                                         if not status.startswith('error_'))
                self.processing_stats['images_downloaded'] = successful_downloads
                self.logger.info(f"Downloaded {successful_downloads}/{len(df)} images successfully")
            
            # Step 4: Final validation
            self.logger.info("Step 4: Final validation")
            self.validate_data_integrity(df)
            self.processing_stats['samples_after_cleaning'] = len(df)
            
            # Calculate processing time
            self.processing_stats['processing_time_seconds'] = time.time() - start_time
            
            self.logger.info(f"Test data preprocessing completed successfully. "
                           f"Final dataset: {len(df)} samples")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Test data preprocessing failed: {str(e)}")
            raise
    
    def generate_preprocessing_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive preprocessing report
        
        Returns:
            Dict containing preprocessing statistics and analysis
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'processing_statistics': self.processing_stats.copy(),
            'component_reports': {}
        }
        
        # Add component-specific reports
        try:
            report['component_reports']['data_cleaning'] = self.data_cleaner.get_cleaning_summary()
        except Exception as e:
            self.logger.warning(f"Failed to get cleaning summary: {str(e)}")
        
        try:
            report['component_reports']['price_normalization'] = self.price_normalizer.get_anomaly_summary()
        except Exception as e:
            self.logger.warning(f"Failed to get price normalization summary: {str(e)}")
        
        try:
            report['component_reports']['image_download'] = self.image_downloader.get_download_statistics()
        except Exception as e:
            self.logger.warning(f"Failed to get download statistics: {str(e)}")
        
        # Calculate overall quality metrics
        if self.processing_stats['total_samples_loaded'] > 0:
            report['quality_metrics'] = {
                'data_retention_rate': (self.processing_stats['samples_after_cleaning'] / 
                                      self.processing_stats['total_samples_loaded']) * 100,
                'image_success_rate': (self.processing_stats['images_downloaded'] / 
                                     self.processing_stats['total_samples_loaded']) * 100 
                                     if self.processing_stats['images_downloaded'] > 0 else 0,
                'processing_efficiency': (self.processing_stats['samples_after_cleaning'] / 
                                        max(self.processing_stats['processing_time_seconds'], 1))
            }
        
        return report
    
    def save_preprocessing_report(self, filepath: str):
        """
        Save preprocessing report to file
        
        Args:
            filepath: Path to save the report
        """
        import json
        
        report = self.generate_preprocessing_report()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Preprocessing report saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save preprocessing report: {str(e)}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    def reset_processing_statistics(self):
        """Reset processing statistics"""
        for key in self.processing_stats:
            if isinstance(self.processing_stats[key], (int, float)):
                self.processing_stats[key] = 0
        
        # Reset component statistics
        self.data_cleaner.reset_cleaning_stats()
        self.price_normalizer.reset_anomaly_tracking()
        self.image_downloader.reset_download_stats()
        
        self.logger.info("Processing statistics reset")
    
    def validate_preprocessing_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the quality of preprocessed data
        
        Args:
            df: Preprocessed DataFrame to validate
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'quality_score': 0.0
        }
        
        # Basic data validation
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("DataFrame is empty")
            return validation_results
        
        # Check required columns
        required_columns = ['sample_id', 'catalog_content', 'image_link']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check data quality
        quality_factors = []
        
        # Sample ID quality (25 points)
        try:
            self.data_cleaner.validate_sample_id_uniqueness(df)
            quality_factors.append(25)
        except DataCleaningError as e:
            validation_results['issues'].append(f"Sample ID validation failed: {str(e)}")
            quality_factors.append(0)
        
        # Catalog content quality (25 points)
        if 'catalog_content' in df.columns:
            empty_content = (df['catalog_content'].str.strip() == '').sum()
            content_quality = max(0, 25 - (empty_content / len(df)) * 25)
            quality_factors.append(content_quality)
            
            if empty_content > len(df) * 0.05:  # >5% empty content
                validation_results['warnings'].append(f"{empty_content} samples have empty catalog content")
        
        # Image availability (25 points)
        if 'image_download_status' in df.columns:
            successful_images = sum(1 for status in df['image_download_status'] 
                                  if not str(status).startswith('error_'))
            image_quality = (successful_images / len(df)) * 25
            quality_factors.append(image_quality)
            
            if image_quality < 20:  # <80% success rate
                validation_results['warnings'].append(f"Low image download success rate: {image_quality/25*100:.1f}%")
        
        # Price quality (25 points) - only for training data
        if 'price' in df.columns:
            valid_prices = (df['price'] > 0).sum()
            price_quality = (valid_prices / len(df)) * 25
            quality_factors.append(price_quality)
            
            if price_quality < 20:  # <80% valid prices
                validation_results['warnings'].append(f"Low price validity rate: {price_quality/25*100:.1f}%")
        else:
            quality_factors.append(25)  # Full points for test data without prices
        
        # Calculate overall quality score
        validation_results['quality_score'] = sum(quality_factors)
        
        # Add recommendations
        recommendations = []
        if validation_results['quality_score'] < 80:
            recommendations.append("Overall data quality is below 80%. Consider additional preprocessing.")
        if len(validation_results['issues']) > 0:
            recommendations.append("Critical issues found. These must be resolved before proceeding.")
        if len(validation_results['warnings']) > 3:
            recommendations.append("Multiple warnings detected. Review data quality.")
        
        validation_results['recommendations'] = recommendations
        
        return validation_results