"""
Versioned Image Embedding System with Metadata for ML Product Pricing Challenge 2025

This module implements embedding extraction using pre-trained CNN models (ResNet, EfficientNet),
versioned embedding storage with model name, checkpoint, and preprocessing metadata,
and embedding cache management with integrity validation as required by task 4.2.
"""

import os
import json
import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd

try:
    from ..config import config
    from .image_processor import ImageProcessor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config
    from features.image_processor import ImageProcessor


@dataclass
class EmbeddingMetadata:
    """Metadata for embedding storage"""
    model_name: str
    model_checkpoint: str
    preprocessing_steps: List[str]
    embedding_dimension: int
    creation_timestamp: float
    pytorch_version: str
    torchvision_version: str
    image_size: Tuple[int, int]
    normalization_mean: List[float]
    normalization_std: List[float]
    total_samples: int
    processing_time: float
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class EmbeddingCacheEntry:
    """Cache entry for embeddings"""
    sample_id: str
    embedding: np.ndarray
    metadata_hash: str
    creation_timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding embedding array)"""
        return {
            'sample_id': self.sample_id,
            'metadata_hash': self.metadata_hash,
            'creation_timestamp': self.creation_timestamp,
            'embedding_shape': self.embedding.shape,
            'embedding_dtype': str(self.embedding.dtype)
        }


class EmbeddingExtractionError(Exception):
    """Custom exception for embedding extraction errors"""
    pass


class ImageEmbeddingSystem:
    """
    Versioned Image Embedding System with comprehensive metadata management
    
    Implements:
    - Embedding extraction using pre-trained CNN models (ResNet, EfficientNet)
    - Versioned embedding storage with model name, checkpoint, and preprocessing metadata
    - Embedding cache management with integrity validation
    """
    
    def __init__(self, image_config=None, cache_dir: Optional[str] = None):
        """
        Initialize ImageEmbeddingSystem
        
        Args:
            image_config: ImageFeatureConfig instance or None for default
            cache_dir: Custom cache directory or None for default
        """
        self.config = image_config or config.image_features
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.cache_dir = Path(cache_dir) if cache_dir else Path("embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize image processor
        self.image_processor = ImageProcessor(self.config)
        
        # Model and embedding state
        self.current_model = None
        self.current_model_name = None
        self.current_metadata = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cache management
        self.embedding_cache = {}
        self.metadata_cache = {}
        
        # Statistics
        self.extraction_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'model_loads': 0,
            'extraction_time': 0.0,
            'cache_saves': 0,
            'cache_loads': 0
        }
        
        # Load existing cache
        self._load_cache_index()
    
    def _get_model_info(self) -> Dict[str, str]:
        """Get current PyTorch and torchvision versions"""
        try:
            import torchvision
            return {
                'pytorch_version': torch.__version__,
                'torchvision_version': torchvision.__version__
            }
        except Exception as e:
            self.logger.warning(f"Could not get version info: {str(e)}")
            return {
                'pytorch_version': 'unknown',
                'torchvision_version': 'unknown'
            }
    
    def _create_metadata(self, model_name: str, total_samples: int, processing_time: float) -> EmbeddingMetadata:
        """Create metadata for current embedding extraction"""
        version_info = self._get_model_info()
        
        preprocessing_steps = [
            f"resize_to_{self.config.image_size}",
            "convert_to_tensor",
            f"normalize_mean_{self.config.normalize_mean}",
            f"normalize_std_{self.config.normalize_std}"
        ]
        
        if self.config.use_augmentation:
            preprocessing_steps.extend([
                f"random_rotation_{self.config.rotation_range}",
                f"color_jitter_brightness_{self.config.brightness_range}",
                "random_horizontal_flip"
            ])
        
        return EmbeddingMetadata(
            model_name=model_name,
            model_checkpoint=self._get_model_checkpoint_info(model_name),
            preprocessing_steps=preprocessing_steps,
            embedding_dimension=self.config.feature_dim,
            creation_timestamp=time.time(),
            pytorch_version=version_info['pytorch_version'],
            torchvision_version=version_info['torchvision_version'],
            image_size=self.config.image_size,
            normalization_mean=self.config.normalize_mean,
            normalization_std=self.config.normalize_std,
            total_samples=total_samples,
            processing_time=processing_time
        )
    
    def _get_model_checkpoint_info(self, model_name: str) -> str:
        """Get model checkpoint information"""
        checkpoint_info = {
            'resnet50': 'torchvision.models.ResNet50_Weights.IMAGENET1K_V2',
            'resnet101': 'torchvision.models.ResNet101_Weights.IMAGENET1K_V2',
            'efficientnet-b0': 'torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1',
            'efficientnet-b3': 'torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1',
            'vit-base': 'torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1'
        }
        return checkpoint_info.get(model_name, f"default_{model_name}")
    
    def _load_model(self, model_name: str) -> nn.Module:
        """
        Load pre-trained CNN model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            nn.Module: Loaded model ready for feature extraction
        """
        self.logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        try:
            if model_name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                # Remove final classification layer
                model = nn.Sequential(*list(model.children())[:-1])
                self.config.feature_dim = 2048
                
            elif model_name == 'resnet101':
                model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
                model = nn.Sequential(*list(model.children())[:-1])
                self.config.feature_dim = 2048
                
            elif model_name == 'efficientnet-b0':
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                # Remove classifier
                model.classifier = nn.Identity()
                self.config.feature_dim = 1280
                
            elif model_name == 'efficientnet-b3':
                model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
                model.classifier = nn.Identity()
                self.config.feature_dim = 1536
                
            elif model_name == 'vit-base':
                model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
                # Remove classification head
                model.heads = nn.Identity()
                self.config.feature_dim = 768
                
            else:
                raise EmbeddingExtractionError(f"Unsupported model: {model_name}")
            
            # Move to device and set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            self.current_model = model
            self.current_model_name = model_name
            self.extraction_stats['model_loads'] += 1
            
            load_time = time.time() - start_time
            self.logger.info(f"Model {model_name} loaded in {load_time:.2f}s on {self.device}")
            
            return model
            
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to load model {model_name}: {str(e)}")
    
    def _extract_features_from_tensor(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features from preprocessed image tensor
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            np.ndarray: Feature vector
        """
        try:
            with torch.no_grad():
                # Add batch dimension if needed
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                # Move to device
                image_tensor = image_tensor.to(self.device)
                
                # Extract features
                features = self.current_model(image_tensor)
                
                # Flatten features
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
                # Convert to numpy
                features_np = features.cpu().numpy()
                
                # Remove batch dimension
                if features_np.shape[0] == 1:
                    features_np = features_np.squeeze(0)
                
                return features_np
                
        except Exception as e:
            raise EmbeddingExtractionError(f"Feature extraction failed: {str(e)}")
    
    def extract_embedding_from_image_path(self, image_path: str, sample_id: str) -> np.ndarray:
        """
        Extract embedding from image file
        
        Args:
            image_path: Path to image file
            sample_id: Sample identifier
            
        Returns:
            np.ndarray: Embedding vector
        """
        try:
            # Process image
            result = self.image_processor.load_and_preprocess_image(image_path)
            
            if result.success:
                # Convert to tensor
                image_tensor = torch.from_numpy(result.processed_image)
                
                # Extract features
                embedding = self._extract_features_from_tensor(image_tensor)
                
                return embedding
            else:
                # Use fallback for failed image processing
                self.logger.warning(f"Image processing failed for {sample_id}: {result.error_message}")
                fallback_image = self.image_processor.handle_missing_images(sample_id)
                image_tensor = torch.from_numpy(fallback_image)
                return self._extract_features_from_tensor(image_tensor)
                
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to extract embedding for {sample_id}: {str(e)}")
    
    def extract_embeddings_batch(self, image_paths: List[str], sample_ids: List[str], 
                                model_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for batch of images with caching
        
        Args:
            image_paths: List of image file paths
            sample_ids: List of sample identifiers
            model_name: Model to use (defaults to config)
            
        Returns:
            Dict mapping sample_id to embedding vector
        """
        if len(image_paths) != len(sample_ids):
            raise ValueError("image_paths and sample_ids must have same length")
        
        model_name = model_name or self.config.cnn_model
        
        # Load model if needed
        if self.current_model is None or self.current_model_name != model_name:
            self._load_model(model_name)
        
        start_time = time.time()
        embeddings = {}
        
        # Create metadata hash for cache validation
        temp_metadata = self._create_metadata(model_name, len(sample_ids), 0.0)
        metadata_hash = self._calculate_metadata_hash(temp_metadata)
        
        self.logger.info(f"Extracting embeddings for {len(sample_ids)} samples using {model_name}")
        
        for image_path, sample_id in zip(image_paths, sample_ids):
            try:
                # Check cache first
                cached_embedding = self._get_cached_embedding(sample_id, metadata_hash)
                if cached_embedding is not None:
                    embeddings[sample_id] = cached_embedding
                    self.extraction_stats['cache_hits'] += 1
                    continue
                
                # Extract new embedding
                embedding = self.extract_embedding_from_image_path(image_path, sample_id)
                embeddings[sample_id] = embedding
                
                # Cache the result
                self._cache_embedding(sample_id, embedding, metadata_hash)
                self.extraction_stats['cache_misses'] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to extract embedding for {sample_id}: {str(e)}")
                # Use zero vector as fallback
                embeddings[sample_id] = np.zeros(self.config.feature_dim, dtype=np.float32)
        
        processing_time = time.time() - start_time
        self.extraction_stats['total_extractions'] += len(sample_ids)
        self.extraction_stats['extraction_time'] += processing_time
        
        # Create and save metadata
        metadata = self._create_metadata(model_name, len(sample_ids), processing_time)
        self.current_metadata = metadata
        
        self.logger.info(f"Batch embedding extraction completed in {processing_time:.2f}s")
        
        return embeddings
    
    def save_versioned_embeddings(self, embeddings: Dict[str, np.ndarray], 
                                 metadata: Optional[EmbeddingMetadata] = None,
                                 version_suffix: Optional[str] = None) -> str:
        """
        Save embeddings with versioning and metadata
        
        Args:
            embeddings: Dictionary of sample_id -> embedding
            metadata: Optional metadata (uses current if None)
            version_suffix: Optional version suffix for filename
            
        Returns:
            str: Path to saved embedding file
        """
        metadata = metadata or self.current_metadata
        if metadata is None:
            raise ValueError("No metadata available. Run extract_embeddings_batch first.")
        
        # Create versioned filename
        timestamp = int(time.time())
        model_name_clean = metadata.model_name.replace('-', '_')
        
        if version_suffix:
            filename = f"embeddings_{model_name_clean}_{version_suffix}_{timestamp}"
        else:
            filename = f"embeddings_{model_name_clean}_{timestamp}"
        
        # Save embeddings
        embeddings_file = self.cache_dir / f"{filename}.pkl"
        metadata_file = self.cache_dir / f"{filename}_metadata.json"
        
        try:
            # Save embeddings as pickle
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata as JSON
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update cache index
            self._update_cache_index(filename, metadata, len(embeddings))
            
            self.extraction_stats['cache_saves'] += 1
            self.logger.info(f"Saved {len(embeddings)} embeddings to {embeddings_file}")
            
            return str(embeddings_file)
            
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to save embeddings: {str(e)}")
    
    def load_versioned_embeddings(self, embedding_file: str) -> Tuple[Dict[str, np.ndarray], EmbeddingMetadata]:
        """
        Load versioned embeddings with metadata validation
        
        Args:
            embedding_file: Path to embedding file
            
        Returns:
            Tuple of (embeddings_dict, metadata)
        """
        embedding_path = Path(embedding_file)
        
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        # Determine metadata file path
        if embedding_path.suffix == '.pkl':
            metadata_file = embedding_path.with_suffix('').with_suffix('_metadata.json')
        else:
            metadata_file = embedding_path.parent / f"{embedding_path.stem}_metadata.json"
        
        try:
            # Load embeddings
            with open(embedding_path, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Load metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = EmbeddingMetadata.from_dict(metadata_dict)
            else:
                self.logger.warning(f"Metadata file not found: {metadata_file}")
                metadata = None
            
            self.extraction_stats['cache_loads'] += 1
            self.logger.info(f"Loaded {len(embeddings)} embeddings from {embedding_file}")
            
            return embeddings, metadata
            
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to load embeddings: {str(e)}")
    
    def _calculate_metadata_hash(self, metadata: EmbeddingMetadata) -> str:
        """Calculate hash of metadata for cache validation"""
        # Create hash from key metadata fields
        hash_data = {
            'model_name': metadata.model_name,
            'model_checkpoint': metadata.model_checkpoint,
            'preprocessing_steps': metadata.preprocessing_steps,
            'image_size': metadata.image_size,
            'normalization_mean': metadata.normalization_mean,
            'normalization_std': metadata.normalization_std
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _get_cached_embedding(self, sample_id: str, metadata_hash: str) -> Optional[np.ndarray]:
        """Get embedding from cache if valid"""
        cache_key = f"{sample_id}_{metadata_hash}"
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key].embedding
        
        return None
    
    def _cache_embedding(self, sample_id: str, embedding: np.ndarray, metadata_hash: str):
        """Cache embedding with metadata hash"""
        cache_key = f"{sample_id}_{metadata_hash}"
        
        cache_entry = EmbeddingCacheEntry(
            sample_id=sample_id,
            embedding=embedding.copy(),
            metadata_hash=metadata_hash,
            creation_timestamp=time.time()
        )
        
        self.embedding_cache[cache_key] = cache_entry
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.metadata_cache = json.load(f)
                self.logger.info(f"Loaded cache index with {len(self.metadata_cache)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {str(e)}")
                self.metadata_cache = {}
        else:
            self.metadata_cache = {}
    
    def _update_cache_index(self, filename: str, metadata: EmbeddingMetadata, embedding_count: int):
        """Update cache index with new entry"""
        self.metadata_cache[filename] = {
            'metadata': metadata.to_dict(),
            'embedding_count': embedding_count,
            'file_path': str(self.cache_dir / f"{filename}.pkl"),
            'metadata_path': str(self.cache_dir / f"{filename}_metadata.json"),
            'creation_timestamp': time.time()
        }
        
        # Save updated index
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.metadata_cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to update cache index: {str(e)}")
    
    def validate_cache_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of cached embeddings
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            'total_cached_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'missing_files': 0,
            'missing_metadata': 0,
            'corrupted_files': [],
            'orphaned_files': []
        }
        
        # Check files in cache index
        for filename, cache_info in self.metadata_cache.items():
            validation_results['total_cached_files'] += 1
            
            embedding_file = Path(cache_info['file_path'])
            metadata_file = Path(cache_info['metadata_path'])
            
            # Check if files exist
            if not embedding_file.exists():
                validation_results['missing_files'] += 1
                validation_results['corrupted_files'].append(str(embedding_file))
                continue
            
            if not metadata_file.exists():
                validation_results['missing_metadata'] += 1
                validation_results['corrupted_files'].append(str(metadata_file))
                continue
            
            # Try to load files
            try:
                embeddings, metadata = self.load_versioned_embeddings(str(embedding_file))
                
                # Validate embedding count
                expected_count = cache_info['embedding_count']
                actual_count = len(embeddings)
                
                if expected_count != actual_count:
                    validation_results['invalid_files'] += 1
                    validation_results['corrupted_files'].append(
                        f"{embedding_file}: count mismatch ({actual_count} vs {expected_count})"
                    )
                else:
                    validation_results['valid_files'] += 1
                    
            except Exception as e:
                validation_results['invalid_files'] += 1
                validation_results['corrupted_files'].append(f"{embedding_file}: {str(e)}")
        
        # Check for orphaned files
        for file_path in self.cache_dir.glob("*.pkl"):
            filename = file_path.stem
            if filename not in self.metadata_cache:
                validation_results['orphaned_files'].append(str(file_path))
        
        return validation_results
    
    def cleanup_invalid_cache(self) -> int:
        """
        Clean up invalid cache entries
        
        Returns:
            int: Number of files cleaned up
        """
        validation_results = self.validate_cache_integrity()
        cleaned_count = 0
        
        # Remove corrupted files
        for corrupted_file in validation_results['corrupted_files']:
            if ':' in corrupted_file:
                file_path = corrupted_file.split(':')[0]
            else:
                file_path = corrupted_file
            
            try:
                Path(file_path).unlink(missing_ok=True)
                cleaned_count += 1
                self.logger.info(f"Removed corrupted file: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to remove {file_path}: {str(e)}")
        
        # Remove orphaned files
        for orphaned_file in validation_results['orphaned_files']:
            try:
                Path(orphaned_file).unlink()
                cleaned_count += 1
                self.logger.info(f"Removed orphaned file: {orphaned_file}")
            except Exception as e:
                self.logger.error(f"Failed to remove {orphaned_file}: {str(e)}")
        
        return cleaned_count
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        stats = self.extraction_stats.copy()
        
        # Calculate derived statistics
        if stats['total_extractions'] > 0:
            stats['cache_hit_rate'] = (stats['cache_hits'] / stats['total_extractions']) * 100
            stats['cache_miss_rate'] = (stats['cache_misses'] / stats['total_extractions']) * 100
        
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['avg_extraction_time'] = stats['extraction_time'] / (stats['cache_hits'] + stats['cache_misses'])
        
        # Add cache info
        stats['cached_entries'] = len(self.embedding_cache)
        stats['cached_files'] = len(self.metadata_cache)
        stats['current_model'] = self.current_model_name
        stats['device'] = str(self.device)
        
        return stats
    
    def reset_statistics(self):
        """Reset extraction statistics"""
        for key in self.extraction_stats:
            if isinstance(self.extraction_stats[key], (int, float)):
                self.extraction_stats[key] = 0
        self.logger.info("Extraction statistics reset")