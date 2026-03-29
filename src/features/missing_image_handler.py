"""
Robust Missing Image Handling System for ML Product Pricing Challenge 2025

This module implements default feature vectors for missing images based on text content,
feature interpolation strategies for unavailable images, and comprehensive logging
and tracking for image processing failures as required by task 4.4.
"""

import os
import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

try:
    from ..config import config
    from .text_processor import TextProcessor
    from .text_feature_extractor import TextFeatureExtractor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config
    from features.text_processor import TextProcessor
    from features.text_feature_extractor import TextFeatureExtractor


@dataclass
class MissingImageRecord:
    """Record for tracking missing image information"""
    sample_id: str
    image_url: str
    failure_reason: str
    text_content: Optional[str]
    fallback_strategy: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MissingImageRecord':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class InterpolationResult:
    """Result of feature interpolation"""
    success: bool
    interpolated_features: Optional[Dict[str, np.ndarray]] = None
    interpolation_method: Optional[str] = None
    confidence_score: Optional[float] = None
    similar_samples: Optional[List[str]] = None
    error_message: Optional[str] = None


class MissingImageHandler:
    """
    Robust Missing Image Handling System
    
    Implements:
    - Default feature vectors for missing images based on text content
    - Feature interpolation strategies for unavailable images
    - Comprehensive logging and tracking for image processing failures
    """
    
    def __init__(self, image_config=None, log_dir: Optional[str] = None):
        """
        Initialize MissingImageHandler
        
        Args:
            image_config: ImageFeatureConfig instance or None for default
            log_dir: Custom log directory or None for default
        """
        self.config = image_config or config.image_features
        self.logger = logging.getLogger(__name__)
        
        # Setup logging directory
        self.log_dir = Path(log_dir) if log_dir else Path(config.infrastructure.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text processing components
        self.text_processor = TextProcessor()
        self.text_feature_extractor = TextFeatureExtractor()
        
        # Missing image tracking
        self.missing_image_records = {}
        self.missing_image_log_file = self.log_dir / "missing_images.json"
        
        # Feature interpolation components
        self.available_features_db = {}  # sample_id -> features
        self.text_similarity_model = None
        self.feature_interpolator = None
        
        # Statistics
        self.handling_stats = {
            'total_missing_images': 0,
            'text_based_fallbacks': 0,
            'interpolated_features': 0,
            'zero_fallbacks': 0,
            'mean_fallbacks': 0,
            'successful_interpolations': 0,
            'failed_interpolations': 0,
            'total_processing_time': 0.0
        }
        
        # Feature dimensions (will be set based on actual feature extraction)
        self.feature_dimensions = {
            'cnn_features': 2048,
            'color_histogram': 192,  # 64 bins * 3 channels
            'dominant_colors': 15,   # 5 colors * 3 channels
            'texture_features': 13,
            'edge_features': 4,
            'shape_features': 7
        }
        
        # Load existing missing image records
        self._load_missing_image_records()
    
    def update_feature_dimensions(self, feature_dimensions: Dict[str, int]):
        """
        Update feature dimensions based on actual feature extractor
        
        Args:
            feature_dimensions: Dictionary of feature names to dimensions
        """
        self.feature_dimensions.update(feature_dimensions)
        self.logger.info(f"Updated feature dimensions: {self.feature_dimensions}")
    
    def create_text_based_fallback(self, sample_id: str, text_content: str, 
                                 target_features: List[str]) -> Dict[str, np.ndarray]:
        """
        Create default feature vectors based on text content
        
        Args:
            sample_id: Sample identifier
            text_content: Product catalog content
            target_features: List of feature types to generate
            
        Returns:
            Dict of feature name to fallback feature vector
        """
        start_time = time.time()
        fallback_features = {}
        
        try:
            # Process text content
            processed_text = self.text_processor.clean_and_preprocess_text(text_content)
            
            # Extract text features for guidance
            text_features = self.text_feature_extractor.extract_text_features(processed_text)
            
            # Create text-based hash for consistency
            text_hash = hashlib.md5(text_content.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16) % (2**32))
            
            for feature_name in target_features:
                if feature_name not in self.feature_dimensions:
                    self.logger.warning(f"Unknown feature type: {feature_name}")
                    continue
                
                feature_dim = self.feature_dimensions[feature_name]
                
                if feature_name == 'cnn_features':
                    # Create CNN-like features based on text
                    fallback_features[feature_name] = self._create_text_based_cnn_features(
                        text_features, feature_dim, text_hash
                    )
                
                elif feature_name == 'color_histogram':
                    # Create color histogram based on text keywords
                    fallback_features[feature_name] = self._create_text_based_color_features(
                        processed_text, feature_dim
                    )
                
                elif feature_name == 'dominant_colors':
                    # Create dominant colors based on text
                    fallback_features[feature_name] = self._create_text_based_dominant_colors(
                        processed_text
                    )
                
                elif feature_name in ['texture_features', 'edge_features', 'shape_features']:
                    # Create texture/edge/shape features based on text complexity
                    fallback_features[feature_name] = self._create_text_based_structural_features(
                        processed_text, feature_dim, feature_name
                    )
                
                else:
                    # Default to small random values
                    fallback_features[feature_name] = np.random.normal(
                        0, 0.1, feature_dim
                    ).astype(np.float32)
            
            processing_time = time.time() - start_time
            self.handling_stats['text_based_fallbacks'] += 1
            self.handling_stats['total_processing_time'] += processing_time
            
            self.logger.info(f"Created text-based fallback features for {sample_id}")
            
            return fallback_features
            
        except Exception as e:
            self.logger.error(f"Failed to create text-based fallback for {sample_id}: {str(e)}")
            
            # Return zero features as last resort
            fallback_features = {}
            for feature_name in target_features:
                if feature_name in self.feature_dimensions:
                    fallback_features[feature_name] = np.zeros(
                        self.feature_dimensions[feature_name], dtype=np.float32
                    )
            
            return fallback_features
    
    def _create_text_based_cnn_features(self, text_features: np.ndarray, 
                                      target_dim: int, text_hash: str) -> np.ndarray:
        """Create CNN-like features based on text features"""
        try:
            # Use text features as seed
            if len(text_features) > 0:
                # Expand or compress text features to target dimension
                if len(text_features) >= target_dim:
                    # Use PCA or truncation
                    cnn_features = text_features[:target_dim]
                else:
                    # Repeat and add noise
                    repeats = target_dim // len(text_features) + 1
                    expanded = np.tile(text_features, repeats)[:target_dim]
                    
                    # Add controlled noise based on text hash
                    np.random.seed(int(text_hash[:8], 16) % (2**32))
                    noise = np.random.normal(0, 0.1, target_dim)
                    cnn_features = expanded + noise
            else:
                # Fallback to hash-based features
                np.random.seed(int(text_hash[:8], 16) % (2**32))
                cnn_features = np.random.normal(0, 0.5, target_dim)
            
            # Normalize to typical CNN feature range
            cnn_features = np.clip(cnn_features, -3, 3)
            
            return cnn_features.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to create text-based CNN features: {str(e)}")
            return np.random.normal(0, 0.1, target_dim).astype(np.float32)
    
    def _create_text_based_color_features(self, text: str, target_dim: int) -> np.ndarray:
        """Create color histogram features based on text keywords"""
        try:
            # Color keywords mapping
            color_keywords = {
                'red': [1.0, 0.0, 0.0], 'blue': [0.0, 0.0, 1.0], 'green': [0.0, 1.0, 0.0],
                'yellow': [1.0, 1.0, 0.0], 'orange': [1.0, 0.5, 0.0], 'purple': [0.5, 0.0, 0.5],
                'pink': [1.0, 0.5, 0.5], 'brown': [0.6, 0.3, 0.1], 'black': [0.0, 0.0, 0.0],
                'white': [1.0, 1.0, 1.0], 'gray': [0.5, 0.5, 0.5], 'silver': [0.7, 0.7, 0.7],
                'gold': [1.0, 0.8, 0.0], 'beige': [0.9, 0.9, 0.7], 'navy': [0.0, 0.0, 0.5]
            }
            
            # Initialize color histogram
            bins_per_channel = target_dim // 3
            color_hist = np.zeros(target_dim)
            
            # Find color keywords in text
            text_lower = text.lower()
            found_colors = []
            
            for color, rgb in color_keywords.items():
                if color in text_lower:
                    found_colors.append(rgb)
            
            if found_colors:
                # Create histogram based on found colors
                for i, rgb in enumerate(found_colors):
                    for c, value in enumerate(rgb):
                        if c < 3:  # RGB channels
                            start_idx = c * bins_per_channel
                            bin_idx = int(value * (bins_per_channel - 1))
                            color_hist[start_idx + bin_idx] += 1.0 / len(found_colors)
            else:
                # Default neutral color distribution
                for c in range(3):
                    start_idx = c * bins_per_channel
                    mid_idx = bins_per_channel // 2
                    color_hist[start_idx + mid_idx] = 0.3
            
            # Normalize
            if np.sum(color_hist) > 0:
                color_hist = color_hist / np.sum(color_hist)
            
            return color_hist.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to create text-based color features: {str(e)}")
            return np.ones(target_dim, dtype=np.float32) / target_dim
    
    def _create_text_based_dominant_colors(self, text: str) -> np.ndarray:
        """Create dominant color features based on text"""
        try:
            # Same color keywords as above
            color_keywords = {
                'red': [1.0, 0.0, 0.0], 'blue': [0.0, 0.0, 1.0], 'green': [0.0, 1.0, 0.0],
                'yellow': [1.0, 1.0, 0.0], 'orange': [1.0, 0.5, 0.0], 'purple': [0.5, 0.0, 0.5],
                'pink': [1.0, 0.5, 0.5], 'brown': [0.6, 0.3, 0.1], 'black': [0.0, 0.0, 0.0],
                'white': [1.0, 1.0, 1.0], 'gray': [0.5, 0.5, 0.5], 'silver': [0.7, 0.7, 0.7],
                'gold': [1.0, 0.8, 0.0], 'beige': [0.9, 0.9, 0.7], 'navy': [0.0, 0.0, 0.5]
            }
            
            text_lower = text.lower()
            dominant_colors = []
            
            # Find up to 5 colors
            for color, rgb in color_keywords.items():
                if color in text_lower and len(dominant_colors) < 5:
                    dominant_colors.append(rgb)
            
            # Fill remaining slots with neutral colors
            while len(dominant_colors) < 5:
                dominant_colors.append([0.5, 0.5, 0.5])  # Gray
            
            # Flatten to feature vector
            dominant_color_features = np.array(dominant_colors).flatten()
            
            return dominant_color_features.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to create text-based dominant colors: {str(e)}")
            return np.full(15, 0.5, dtype=np.float32)  # 5 colors * 3 channels
    
    def _create_text_based_structural_features(self, text: str, target_dim: int, 
                                             feature_type: str) -> np.ndarray:
        """Create structural features based on text complexity"""
        try:
            # Text complexity metrics
            text_length = len(text)
            word_count = len(text.split())
            unique_words = len(set(text.lower().split()))
            avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
            
            # Normalize metrics
            complexity_score = min(1.0, text_length / 1000)  # Normalize by typical text length
            diversity_score = unique_words / word_count if word_count > 0 else 0
            word_length_score = min(1.0, avg_word_length / 10)  # Normalize by typical word length
            
            if feature_type == 'texture_features':
                # Map text complexity to texture-like features
                features = [
                    complexity_score,  # Overall complexity
                    diversity_score,   # Word diversity (like texture variation)
                    word_length_score, # Word length (like texture granularity)
                    min(1.0, text.count(',') / 10),  # Punctuation density
                    min(1.0, text.count(' ') / 100), # Space density
                ]
                
            elif feature_type == 'edge_features':
                # Map text structure to edge-like features
                features = [
                    min(1.0, text.count('.') / 5),   # Sentence boundaries
                    min(1.0, text.count('\n') / 3), # Line breaks
                    complexity_score,                # Overall structure
                    diversity_score,                 # Structural variation
                ]
                
            elif feature_type == 'shape_features':
                # Map text organization to shape-like features
                features = [
                    complexity_score,     # Overall size
                    diversity_score,      # Compactness
                    word_length_score,    # Aspect ratio
                    min(1.0, word_count / 50),  # Extent
                    min(1.0, unique_words / word_count) if word_count > 0 else 0,  # Solidity
                    min(1.0, text_length / (word_count * 10)) if word_count > 0 else 0,  # Eccentricity
                    min(1.0, text.count(' ') / text_length) if text_length > 0 else 0,  # Convexity
                ]
            
            else:
                features = [complexity_score, diversity_score, word_length_score]
            
            # Pad or truncate to target dimension
            if len(features) < target_dim:
                features.extend([0.0] * (target_dim - len(features)))
            else:
                features = features[:target_dim]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to create text-based {feature_type}: {str(e)}")
            return np.zeros(target_dim, dtype=np.float32)
    
    def interpolate_features_from_similar_samples(self, sample_id: str, text_content: str,
                                                target_features: List[str],
                                                k_neighbors: int = 5) -> InterpolationResult:
        """
        Interpolate features based on similar samples with available images
        
        Args:
            sample_id: Sample identifier
            text_content: Product catalog content
            target_features: List of feature types to interpolate
            k_neighbors: Number of similar samples to use
            
        Returns:
            InterpolationResult with interpolated features or error
        """
        try:
            if not self.available_features_db:
                return InterpolationResult(
                    success=False,
                    error_message="No available features database for interpolation"
                )
            
            # Find similar samples based on text content
            similar_samples = self._find_similar_samples(text_content, k_neighbors)
            
            if not similar_samples:
                return InterpolationResult(
                    success=False,
                    error_message="No similar samples found for interpolation"
                )
            
            # Interpolate features
            interpolated_features = {}
            confidence_scores = []
            
            for feature_name in target_features:
                if feature_name not in self.feature_dimensions:
                    continue
                
                # Collect features from similar samples
                similar_feature_vectors = []
                for similar_id, similarity_score in similar_samples:
                    if (similar_id in self.available_features_db and 
                        feature_name in self.available_features_db[similar_id]):
                        similar_feature_vectors.append(
                            (self.available_features_db[similar_id][feature_name], similarity_score)
                        )
                
                if similar_feature_vectors:
                    # Weighted average based on similarity scores
                    weighted_features = []
                    total_weight = 0
                    
                    for feature_vector, weight in similar_feature_vectors:
                        weighted_features.append(feature_vector * weight)
                        total_weight += weight
                    
                    if total_weight > 0:
                        interpolated_feature = np.sum(weighted_features, axis=0) / total_weight
                        interpolated_features[feature_name] = interpolated_feature.astype(np.float32)
                        confidence_scores.append(total_weight / len(similar_feature_vectors))
                    else:
                        # Fallback to simple average
                        feature_vectors = [fv for fv, _ in similar_feature_vectors]
                        interpolated_features[feature_name] = np.mean(feature_vectors, axis=0).astype(np.float32)
                        confidence_scores.append(0.5)
                else:
                    # No similar features available, use text-based fallback
                    text_fallback = self.create_text_based_fallback(
                        sample_id, text_content, [feature_name]
                    )
                    interpolated_features[feature_name] = text_fallback[feature_name]
                    confidence_scores.append(0.1)
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            self.handling_stats['interpolated_features'] += 1
            self.handling_stats['successful_interpolations'] += 1
            
            return InterpolationResult(
                success=True,
                interpolated_features=interpolated_features,
                interpolation_method="weighted_similarity",
                confidence_score=overall_confidence,
                similar_samples=[sid for sid, _ in similar_samples]
            )
            
        except Exception as e:
            self.logger.error(f"Feature interpolation failed for {sample_id}: {str(e)}")
            self.handling_stats['failed_interpolations'] += 1
            
            return InterpolationResult(
                success=False,
                error_message=str(e)
            )
    
    def _find_similar_samples(self, text_content: str, k: int) -> List[Tuple[str, float]]:
        """Find similar samples based on text content"""
        try:
            if not hasattr(self, '_text_vectors') or self._text_vectors is None:
                self._build_text_similarity_index()
            
            # Process query text
            processed_text = self.text_processor.clean_and_preprocess_text(text_content)
            
            # Extract text features for similarity comparison
            query_features = self.text_feature_extractor.extract_text_features(processed_text)
            
            if len(query_features) == 0:
                return []
            
            # Find similar samples using nearest neighbors
            if hasattr(self, '_similarity_model') and self._similarity_model is not None:
                # Reshape for sklearn
                query_vector = query_features.reshape(1, -1)
                
                # Find k nearest neighbors
                distances, indices = self._similarity_model.kneighbors(query_vector, n_neighbors=min(k, len(self._sample_ids)))
                
                # Convert distances to similarity scores
                similarities = 1.0 / (1.0 + distances[0])
                
                # Return sample IDs with similarity scores
                similar_samples = []
                for idx, similarity in zip(indices[0], similarities):
                    if idx < len(self._sample_ids):
                        similar_samples.append((self._sample_ids[idx], similarity))
                
                return similar_samples
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to find similar samples: {str(e)}")
            return []
    
    def _build_text_similarity_index(self):
        """Build text similarity index from available samples"""
        try:
            if not self.available_features_db:
                self.logger.warning("No available features database to build similarity index")
                return
            
            # This would be built from actual text content of available samples
            # For now, create a placeholder
            self._sample_ids = list(self.available_features_db.keys())
            self._text_vectors = None
            self._similarity_model = None
            
            self.logger.info("Text similarity index placeholder created")
            
        except Exception as e:
            self.logger.error(f"Failed to build text similarity index: {str(e)}")
    
    def update_available_features_database(self, sample_features: Dict[str, Dict[str, np.ndarray]]):
        """
        Update database of available features for interpolation
        
        Args:
            sample_features: Dict mapping sample_id to feature dictionary
        """
        self.available_features_db.update(sample_features)
        self.logger.info(f"Updated available features database with {len(sample_features)} samples")
        
        # Rebuild similarity index
        self._build_text_similarity_index()
    
    def handle_missing_image(self, sample_id: str, image_url: str, failure_reason: str,
                           text_content: Optional[str] = None,
                           target_features: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Handle missing image with comprehensive fallback strategies
        
        Args:
            sample_id: Sample identifier
            image_url: URL of missing image
            failure_reason: Reason for image processing failure
            text_content: Optional text content for fallback generation
            target_features: List of feature types to generate
            
        Returns:
            Dict of feature name to fallback feature vector
        """
        start_time = time.time()
        
        # Record missing image
        missing_record = MissingImageRecord(
            sample_id=sample_id,
            image_url=image_url,
            failure_reason=failure_reason,
            text_content=text_content,
            fallback_strategy="",
            timestamp=time.time()
        )
        
        # Default target features if not specified
        if target_features is None:
            target_features = list(self.feature_dimensions.keys())
        
        fallback_features = {}
        
        try:
            # Strategy 1: Try interpolation from similar samples if text is available
            if text_content and self.available_features_db:
                interpolation_result = self.interpolate_features_from_similar_samples(
                    sample_id, text_content, target_features
                )
                
                if interpolation_result.success and interpolation_result.confidence_score > 0.3:
                    fallback_features = interpolation_result.interpolated_features
                    missing_record.fallback_strategy = f"interpolation_confidence_{interpolation_result.confidence_score:.2f}"
                    self.logger.info(f"Used interpolation fallback for {sample_id}")
            
            # Strategy 2: Text-based fallback if text is available
            if not fallback_features and text_content:
                fallback_features = self.create_text_based_fallback(
                    sample_id, text_content, target_features
                )
                missing_record.fallback_strategy = "text_based"
                self.logger.info(f"Used text-based fallback for {sample_id}")
            
            # Strategy 3: Use configured missing image strategy
            if not fallback_features:
                if self.config.missing_image_strategy == "zero":
                    for feature_name in target_features:
                        if feature_name in self.feature_dimensions:
                            fallback_features[feature_name] = np.zeros(
                                self.feature_dimensions[feature_name], dtype=np.float32
                            )
                    missing_record.fallback_strategy = "zero"
                    self.handling_stats['zero_fallbacks'] += 1
                
                elif self.config.missing_image_strategy == "mean":
                    # Use mean values (would be computed from available samples)
                    for feature_name in target_features:
                        if feature_name in self.feature_dimensions:
                            fallback_features[feature_name] = np.full(
                                self.feature_dimensions[feature_name], 0.5, dtype=np.float32
                            )
                    missing_record.fallback_strategy = "mean"
                    self.handling_stats['mean_fallbacks'] += 1
                
                else:
                    # Default to zero
                    for feature_name in target_features:
                        if feature_name in self.feature_dimensions:
                            fallback_features[feature_name] = np.zeros(
                                self.feature_dimensions[feature_name], dtype=np.float32
                            )
                    missing_record.fallback_strategy = "zero_default"
                    self.handling_stats['zero_fallbacks'] += 1
            
            # Record the missing image
            self.missing_image_records[sample_id] = missing_record
            self.handling_stats['total_missing_images'] += 1
            
            processing_time = time.time() - start_time
            self.handling_stats['total_processing_time'] += processing_time
            
            # Save missing image log
            self._save_missing_image_records()
            
            return fallback_features
            
        except Exception as e:
            self.logger.error(f"Failed to handle missing image for {sample_id}: {str(e)}")
            
            # Last resort: return zero features
            fallback_features = {}
            for feature_name in target_features:
                if feature_name in self.feature_dimensions:
                    fallback_features[feature_name] = np.zeros(
                        self.feature_dimensions[feature_name], dtype=np.float32
                    )
            
            missing_record.fallback_strategy = "error_fallback"
            self.missing_image_records[sample_id] = missing_record
            
            return fallback_features
    
    def _load_missing_image_records(self):
        """Load existing missing image records"""
        if self.missing_image_log_file.exists():
            try:
                with open(self.missing_image_log_file, 'r') as f:
                    records_data = json.load(f)
                
                self.missing_image_records = {
                    sample_id: MissingImageRecord.from_dict(record_data)
                    for sample_id, record_data in records_data.items()
                }
                
                self.logger.info(f"Loaded {len(self.missing_image_records)} missing image records")
                
            except Exception as e:
                self.logger.warning(f"Failed to load missing image records: {str(e)}")
                self.missing_image_records = {}
        else:
            self.missing_image_records = {}
    
    def _save_missing_image_records(self):
        """Save missing image records to disk"""
        try:
            records_data = {
                sample_id: record.to_dict()
                for sample_id, record in self.missing_image_records.items()
            }
            
            with open(self.missing_image_log_file, 'w') as f:
                json.dump(records_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save missing image records: {str(e)}")
    
    def get_missing_image_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of missing images
        
        Returns:
            Dict containing missing image analysis
        """
        analysis = {
            'total_missing_images': len(self.missing_image_records),
            'failure_reasons': {},
            'fallback_strategies': {},
            'temporal_distribution': {},
            'handling_statistics': self.handling_stats.copy()
        }
        
        # Analyze failure reasons
        for record in self.missing_image_records.values():
            reason = record.failure_reason
            analysis['failure_reasons'][reason] = analysis['failure_reasons'].get(reason, 0) + 1
        
        # Analyze fallback strategies
        for record in self.missing_image_records.values():
            strategy = record.fallback_strategy
            analysis['fallback_strategies'][strategy] = analysis['fallback_strategies'].get(strategy, 0) + 1
        
        # Temporal distribution (by hour)
        for record in self.missing_image_records.values():
            hour = int(record.timestamp // 3600) * 3600  # Round to hour
            analysis['temporal_distribution'][hour] = analysis['temporal_distribution'].get(hour, 0) + 1
        
        # Calculate rates
        if analysis['total_missing_images'] > 0:
            for strategy, count in analysis['fallback_strategies'].items():
                analysis['fallback_strategies'][f"{strategy}_rate"] = (count / analysis['total_missing_images']) * 100
        
        return analysis
    
    def get_handling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive handling statistics"""
        stats = self.handling_stats.copy()
        
        # Add analysis data
        stats.update(self.get_missing_image_analysis())
        
        # Add configuration info
        stats['missing_image_strategy'] = self.config.missing_image_strategy
        stats['feature_dimensions'] = self.feature_dimensions.copy()
        stats['available_samples_count'] = len(self.available_features_db)
        
        return stats
    
    def reset_statistics(self):
        """Reset handling statistics"""
        for key in self.handling_stats:
            if isinstance(self.handling_stats[key], (int, float)):
                self.handling_stats[key] = 0
        self.logger.info("Missing image handling statistics reset")
    
    def cleanup_old_records(self, days_old: int = 30) -> int:
        """
        Clean up old missing image records
        
        Args:
            days_old: Remove records older than this many days
            
        Returns:
            int: Number of records removed
        """
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        old_records = [
            sample_id for sample_id, record in self.missing_image_records.items()
            if record.timestamp < cutoff_time
        ]
        
        for sample_id in old_records:
            del self.missing_image_records[sample_id]
        
        if old_records:
            self._save_missing_image_records()
            self.logger.info(f"Cleaned up {len(old_records)} old missing image records")
        
        return len(old_records)