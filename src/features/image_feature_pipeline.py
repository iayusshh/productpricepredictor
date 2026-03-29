"""
Integrated Image Feature Engineering Pipeline for ML Product Pricing Challenge 2025

This module integrates all image processing components into a unified pipeline:
- ImageProcessor for robust preprocessing
- ImageEmbeddingSystem for versioned CNN embeddings
- VisualFeatureExtractor for comprehensive visual features
- MissingImageHandler for robust fallback mechanisms
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

try:
    from ..config import config
    from .image_processor import ImageProcessor
    from .image_embedding_system import ImageEmbeddingSystem
    from .visual_feature_extractor import VisualFeatureExtractor
    from .missing_image_handler import MissingImageHandler
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config
    from features.image_processor import ImageProcessor
    from features.image_embedding_system import ImageEmbeddingSystem
    from features.visual_feature_extractor import VisualFeatureExtractor
    from features.missing_image_handler import MissingImageHandler


class ImageFeaturePipeline:
    """
    Integrated Image Feature Engineering Pipeline
    
    Combines all image processing components into a unified interface for:
    - Robust image preprocessing with fallback mechanisms
    - CNN feature extraction with versioning and caching
    - Traditional computer vision features (color, texture, edges)
    - Missing image handling with text-based fallbacks
    """
    
    def __init__(self, image_config=None):
        """
        Initialize ImageFeaturePipeline
        
        Args:
            image_config: ImageFeatureConfig instance or None for default
        """
        self.config = image_config or config.image_features
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.image_processor = ImageProcessor(self.config)
        self.embedding_system = ImageEmbeddingSystem(self.config)
        self.visual_extractor = VisualFeatureExtractor(self.config)
        self.missing_handler = MissingImageHandler(self.config)
        
        # Update missing handler with feature dimensions
        feature_dims = self.visual_extractor.get_feature_dimensions()
        self.missing_handler.update_feature_dimensions(feature_dims)
        
        # Pipeline statistics
        self.pipeline_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'missing_images_handled': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("Image Feature Pipeline initialized")
    
    def extract_features_single(self, image_path: str, sample_id: str, 
                              text_content: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract all image features for a single sample
        
        Args:
            image_path: Path to image file
            sample_id: Sample identifier
            text_content: Optional text content for fallback generation
            
        Returns:
            Dict containing all extracted features
        """
        start_time = time.time()
        
        try:
            # Check if image exists and is valid
            if not Path(image_path).exists():
                self.logger.warning(f"Image not found: {image_path}")
                return self._handle_missing_image(
                    sample_id, image_path, "file_not_found", text_content
                )
            
            # Validate image integrity
            if not self.image_processor.validate_image_integrity(image_path):
                self.logger.warning(f"Image integrity check failed: {image_path}")
                return self._handle_missing_image(
                    sample_id, image_path, "integrity_check_failed", text_content
                )
            
            # Extract all visual features
            features = self.visual_extractor.extract_all_visual_features(image_path, sample_id)
            
            # Validate extracted features
            if self._validate_extracted_features(features):
                self.pipeline_stats['successful_extractions'] += 1
                
                # Update missing handler's available features database
                self.missing_handler.update_available_features_database({sample_id: features})
                
                processing_time = time.time() - start_time
                self.pipeline_stats['total_processing_time'] += processing_time
                
                return features
            else:
                self.logger.warning(f"Feature validation failed for {sample_id}")
                return self._handle_missing_image(
                    sample_id, image_path, "feature_validation_failed", text_content
                )
        
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {sample_id}: {str(e)}")
            return self._handle_missing_image(
                sample_id, image_path, f"extraction_error_{str(e)}", text_content
            )
        
        finally:
            self.pipeline_stats['total_processed'] += 1
            if self.pipeline_stats['total_processed'] > 0:
                self.pipeline_stats['avg_processing_time'] = (
                    self.pipeline_stats['total_processing_time'] / 
                    self.pipeline_stats['total_processed']
                )
    
    def extract_features_batch(self, df: pd.DataFrame, 
                             image_path_column: str = 'image_path',
                             sample_id_column: str = 'sample_id',
                             text_content_column: Optional[str] = 'catalog_content') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract image features for batch of samples
        
        Args:
            df: DataFrame containing sample information
            image_path_column: Column name for image paths
            sample_id_column: Column name for sample IDs
            text_content_column: Column name for text content (optional)
            
        Returns:
            Dict mapping sample_id to feature dictionary
        """
        self.logger.info(f"Starting batch feature extraction for {len(df)} samples")
        
        batch_features = {}
        
        for _, row in df.iterrows():
            sample_id = row[sample_id_column]
            image_path = row[image_path_column]
            text_content = row[text_content_column] if text_content_column and text_content_column in row else None
            
            features = self.extract_features_single(image_path, sample_id, text_content)
            batch_features[sample_id] = features
        
        self.logger.info(f"Batch feature extraction completed: {len(batch_features)} samples processed")
        
        return batch_features
    
    def _handle_missing_image(self, sample_id: str, image_path: str, 
                            failure_reason: str, text_content: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Handle missing or failed image processing"""
        self.pipeline_stats['failed_extractions'] += 1
        self.pipeline_stats['missing_images_handled'] += 1
        
        # Get target feature types
        target_features = list(self.visual_extractor.get_feature_dimensions().keys())
        
        # Use missing image handler
        fallback_features = self.missing_handler.handle_missing_image(
            sample_id=sample_id,
            image_url=image_path,
            failure_reason=failure_reason,
            text_content=text_content,
            target_features=target_features
        )
        
        return fallback_features
    
    def _validate_extracted_features(self, features: Dict[str, np.ndarray]) -> bool:
        """Validate that extracted features are valid"""
        try:
            expected_dims = self.visual_extractor.get_feature_dimensions()
            
            for feature_name, feature_vector in features.items():
                if feature_name not in expected_dims:
                    self.logger.warning(f"Unexpected feature type: {feature_name}")
                    continue
                
                expected_dim = expected_dims[feature_name]
                
                # Check dimensions
                if len(feature_vector.shape) != 1 or feature_vector.shape[0] != expected_dim:
                    self.logger.warning(f"Invalid feature dimensions for {feature_name}: "
                                      f"expected {expected_dim}, got {feature_vector.shape}")
                    return False
                
                # Check for NaN or infinite values
                if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                    self.logger.warning(f"Invalid values in feature {feature_name}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Feature validation error: {str(e)}")
            return False
    
    def save_features_to_cache(self, features: Dict[str, Dict[str, np.ndarray]], 
                             version_suffix: Optional[str] = None) -> str:
        """
        Save extracted features to versioned cache
        
        Args:
            features: Dictionary of sample_id -> feature_dict
            version_suffix: Optional version suffix for cache file
            
        Returns:
            str: Path to saved cache file
        """
        try:
            # Prepare embeddings for CNN features
            cnn_embeddings = {}
            for sample_id, feature_dict in features.items():
                if 'cnn_features' in feature_dict:
                    cnn_embeddings[sample_id] = feature_dict['cnn_features']
            
            # Save CNN embeddings using embedding system
            if cnn_embeddings:
                embedding_file = self.embedding_system.save_versioned_embeddings(
                    cnn_embeddings, version_suffix=version_suffix
                )
                self.logger.info(f"Saved CNN embeddings to {embedding_file}")
            
            # Save all features as comprehensive cache
            cache_file = Path("embeddings") / f"all_features_{int(time.time())}.pkl"
            if version_suffix:
                cache_file = cache_file.with_name(f"all_features_{version_suffix}_{int(time.time())}.pkl")
            
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"Saved all features to {cache_file}")
            
            return str(cache_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save features to cache: {str(e)}")
            raise
    
    def load_features_from_cache(self, cache_file: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load features from cache file
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            Dict mapping sample_id to feature dictionary
        """
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                features = pickle.load(f)
            
            self.logger.info(f"Loaded {len(features)} feature sets from {cache_file}")
            
            # Update missing handler's database
            self.missing_handler.update_available_features_database(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to load features from cache: {str(e)}")
            raise
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = self.pipeline_stats.copy()
        
        # Add component statistics
        stats['image_processor'] = self.image_processor.get_processing_statistics()
        stats['embedding_system'] = self.embedding_system.get_extraction_statistics()
        stats['visual_extractor'] = self.visual_extractor.get_extraction_statistics()
        stats['missing_handler'] = self.missing_handler.get_handling_statistics()
        
        # Calculate derived statistics
        if stats['total_processed'] > 0:
            stats['success_rate'] = (stats['successful_extractions'] / stats['total_processed']) * 100
            stats['failure_rate'] = (stats['failed_extractions'] / stats['total_processed']) * 100
            stats['missing_image_rate'] = (stats['missing_images_handled'] / stats['total_processed']) * 100
        
        # Add feature information
        stats['feature_dimensions'] = self.visual_extractor.get_feature_dimensions()
        stats['total_feature_dimension'] = self.visual_extractor.get_total_feature_dimension()
        
        # Add configuration info
        stats['cnn_model'] = self.config.cnn_model
        stats['image_size'] = self.config.image_size
        stats['missing_image_strategy'] = self.config.missing_image_strategy
        stats['extract_color_features'] = self.config.extract_color_features
        stats['extract_texture_features'] = self.config.extract_texture_features
        
        return stats
    
    def validate_pipeline_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of the entire pipeline
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            'pipeline_valid': True,
            'component_validations': {},
            'issues': []
        }
        
        try:
            # Validate embedding system cache
            cache_validation = self.embedding_system.validate_cache_integrity()
            validation_results['component_validations']['embedding_system'] = cache_validation
            
            if cache_validation['invalid_files'] > 0:
                validation_results['issues'].append(f"Embedding cache has {cache_validation['invalid_files']} invalid files")
            
            # Validate missing image handler
            missing_analysis = self.missing_handler.get_missing_image_analysis()
            validation_results['component_validations']['missing_handler'] = missing_analysis
            
            # Validate feature dimensions consistency
            expected_dims = self.visual_extractor.get_feature_dimensions()
            handler_dims = self.missing_handler.feature_dimensions
            
            dim_mismatches = []
            for feature_name, expected_dim in expected_dims.items():
                if feature_name in handler_dims and handler_dims[feature_name] != expected_dim:
                    dim_mismatches.append(f"{feature_name}: {handler_dims[feature_name]} vs {expected_dim}")
            
            if dim_mismatches:
                validation_results['issues'].extend(dim_mismatches)
                validation_results['pipeline_valid'] = False
            
            # Check if any critical issues exist
            if validation_results['issues']:
                validation_results['pipeline_valid'] = False
            
            return validation_results
            
        except Exception as e:
            validation_results['pipeline_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            return validation_results
    
    def cleanup_pipeline_cache(self) -> Dict[str, int]:
        """
        Clean up pipeline caches
        
        Returns:
            Dict with cleanup results
        """
        cleanup_results = {
            'embedding_cache_cleaned': 0,
            'missing_records_cleaned': 0,
            'total_cleaned': 0
        }
        
        try:
            # Clean up embedding system cache
            embedding_cleaned = self.embedding_system.cleanup_invalid_cache()
            cleanup_results['embedding_cache_cleaned'] = embedding_cleaned
            
            # Clean up old missing image records
            missing_cleaned = self.missing_handler.cleanup_old_records(days_old=30)
            cleanup_results['missing_records_cleaned'] = missing_cleaned
            
            cleanup_results['total_cleaned'] = embedding_cleaned + missing_cleaned
            
            self.logger.info(f"Pipeline cleanup completed: {cleanup_results['total_cleaned']} items cleaned")
            
            return cleanup_results
            
        except Exception as e:
            self.logger.error(f"Pipeline cleanup failed: {str(e)}")
            return cleanup_results
    
    def reset_all_statistics(self):
        """Reset statistics for all components"""
        # Reset pipeline statistics
        for key in self.pipeline_stats:
            if isinstance(self.pipeline_stats[key], (int, float)):
                self.pipeline_stats[key] = 0
        
        # Reset component statistics
        self.image_processor.reset_statistics()
        self.embedding_system.reset_statistics()
        self.visual_extractor.reset_statistics()
        self.missing_handler.reset_statistics()
        
        self.logger.info("All pipeline statistics reset")