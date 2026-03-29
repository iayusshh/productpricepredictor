"""
Feature Engineering Pipeline for ML Product Pricing Challenge

This package provides comprehensive text and image processing and feature extraction 
capabilities for product catalog content and images, including:

Text Processing:
- TextProcessor: Catalog content parsing and text cleaning
- IPQExtractor: High-precision Item Pack Quantity extraction with >90% precision
- TextFeatureExtractor: Semantic embeddings and statistical text features
- CatalogParser: Structured information extraction (dimensions, materials, brands)

Image Processing:
- ImageProcessor: Robust image preprocessing with fallback mechanisms
- ImageEmbeddingSystem: Versioned CNN embeddings with metadata
- VisualFeatureExtractor: Comprehensive visual features (CNN, color, texture, edges)
- MissingImageHandler: Text-based fallbacks for missing images
- ImageFeaturePipeline: Integrated image processing pipeline

Feature Fusion:
- FeatureFusion: Multimodal feature combination with concatenation, weighted, and attention methods
- DimensionalityReducer: PCA, feature selection, and correlation analysis for dimension management
- AttentionFusion: Neural attention mechanism for learnable feature combination
"""

# Text processing imports
from .text_processor import TextProcessor, ParsedCatalogContent
from .ipq_extractor import IPQExtractor, IPQResult, ValidationCase
from .text_feature_extractor import TextFeatureExtractor, TextFeatures, ReadabilityCalculator
from .catalog_parser import CatalogParser, ProductSpecification

# Image processing imports
from .image_processor import ImageProcessor, ImageProcessingResult
from .image_embedding_system import ImageEmbeddingSystem, EmbeddingMetadata, EmbeddingCacheEntry
from .visual_feature_extractor import VisualFeatureExtractor
from .missing_image_handler import MissingImageHandler, MissingImageRecord, InterpolationResult
from .image_feature_pipeline import ImageFeaturePipeline

# Feature fusion imports
from .feature_fusion import FeatureFusion, DimensionalityReducer, AttentionFusion
from .feature_fusion import FusionResult, DimensionalityReductionResult
from .feature_fusion import fuse_features, reduce_feature_dimensions

__all__ = [
    # Text processing classes
    'TextProcessor',
    'IPQExtractor', 
    'TextFeatureExtractor',
    'CatalogParser',
    
    # Image processing classes
    'ImageProcessor',
    'ImageEmbeddingSystem',
    'VisualFeatureExtractor',
    'MissingImageHandler',
    'ImageFeaturePipeline',
    
    # Feature fusion classes
    'FeatureFusion',
    'DimensionalityReducer',
    'AttentionFusion',
    
    # Text data classes
    'ParsedCatalogContent',
    'IPQResult',
    'ValidationCase',
    'TextFeatures',
    'ProductSpecification',
    
    # Image data classes
    'ImageProcessingResult',
    'EmbeddingMetadata',
    'EmbeddingCacheEntry',
    'MissingImageRecord',
    'InterpolationResult',
    
    # Feature fusion data classes
    'FusionResult',
    'DimensionalityReductionResult',
    
    # Utility classes
    'ReadabilityCalculator',
    
    # Convenience functions
    'fuse_features',
    'reduce_feature_dimensions',
]

# Version information
__version__ = "1.0.0"
__author__ = "ML Product Pricing Challenge Team"
__description__ = "Enhanced text feature engineering pipeline for product pricing prediction"