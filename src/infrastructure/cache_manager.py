"""
Caching and artifact management system for ML product pricing.
Handles embedding cache, image cache, and model checkpoint management.
"""

import json
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import hashlib
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EmbeddingMetadata:
    """Metadata for cached embeddings."""
    version: str
    model_name: str
    model_checkpoint: str
    preprocessing_steps: List[str]
    feature_dim: int
    sample_count: int
    created_timestamp: str
    checksum: str
    file_size_bytes: int


@dataclass
class ImageCacheEntry:
    """Entry in image cache manifest."""
    sample_id: str
    image_url: str
    local_path: str
    download_status: str  # 'success', 'failed', 'pending'
    download_timestamp: Optional[str]
    file_size_bytes: Optional[int]
    checksum: Optional[str]
    error_message: Optional[str] = None


@dataclass
class ModelCheckpoint:
    """Model checkpoint metadata."""
    checkpoint_id: str
    model_type: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    file_path: str
    created_timestamp: str
    checksum: str
    file_size_bytes: int


class EmbeddingCache:
    """Manages versioned embedding cache with metadata."""
    
    def __init__(self, cache_dir: str = "embeddings", logger=None):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for embedding cache
            logger: LoggingManager instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logger
        self.metadata_file = self.cache_dir / "embedding_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, EmbeddingMetadata]:
        """Load embedding metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                return {k: EmbeddingMetadata(**v) for k, v in data.items()}
            except Exception as e:
                if self.logger:
                    self.logger.log_error("metadata_load_error", 
                                        f"Failed to load embedding metadata: {str(e)}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save embedding metadata to file."""
        try:
            data = {k: asdict(v) for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.log_error("metadata_save_error", 
                                    f"Failed to save embedding metadata: {str(e)}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            if self.logger:
                self.logger.log_error("checksum_error", 
                                    f"Failed to calculate checksum for {file_path}: {str(e)}")
            return ""
    
    def save_embeddings(self, embeddings: np.ndarray, version: str, 
                       model_name: str, model_checkpoint: str,
                       preprocessing_steps: List[str]) -> str:
        """
        Save embeddings with versioned metadata.
        
        Args:
            embeddings: Numpy array of embeddings
            version: Version identifier
            model_name: Name of the model used
            model_checkpoint: Model checkpoint identifier
            preprocessing_steps: List of preprocessing steps applied
            
        Returns:
            Path to saved embeddings file
        """
        # Create versioned filename
        safe_version = version.replace("/", "_").replace(":", "_")
        embedding_file = self.cache_dir / f"embeddings_{safe_version}.npy"
        
        try:
            # Save embeddings
            np.save(embedding_file, embeddings)
            
            # Calculate metadata
            file_size = embedding_file.stat().st_size
            checksum = self._calculate_checksum(str(embedding_file))
            
            # Create metadata
            metadata = EmbeddingMetadata(
                version=version,
                model_name=model_name,
                model_checkpoint=model_checkpoint,
                preprocessing_steps=preprocessing_steps,
                feature_dim=embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0],
                sample_count=embeddings.shape[0] if len(embeddings.shape) > 1 else 1,
                created_timestamp=datetime.now().isoformat(),
                checksum=checksum,
                file_size_bytes=file_size
            )
            
            # Store metadata
            self.metadata[version] = metadata
            self._save_metadata()
            
            if self.logger:
                self.logger.log_performance_metrics("embedding_cache_save", {
                    "version": version,
                    "model_name": model_name,
                    "feature_dim": metadata.feature_dim,
                    "sample_count": metadata.sample_count,
                    "file_size_mb": file_size / 1024**2,
                    "file_path": str(embedding_file)
                })
            
            return str(embedding_file)
            
        except Exception as e:
            if self.logger:
                self.logger.log_error("embedding_save_error", 
                                    f"Failed to save embeddings {version}: {str(e)}")
            raise
    
    def load_embeddings(self, version: str, validate_checksum: bool = True) -> Optional[np.ndarray]:
        """
        Load embeddings by version.
        
        Args:
            version: Version identifier
            validate_checksum: Whether to validate file integrity
            
        Returns:
            Loaded embeddings or None if not found
        """
        if version not in self.metadata:
            if self.logger:
                self.logger.log_error("embedding_not_found", 
                                    f"Embedding version {version} not found in cache")
            return None
        
        metadata = self.metadata[version]
        safe_version = version.replace("/", "_").replace(":", "_")
        embedding_file = self.cache_dir / f"embeddings_{safe_version}.npy"
        
        if not embedding_file.exists():
            if self.logger:
                self.logger.log_error("embedding_file_missing", 
                                    f"Embedding file {embedding_file} not found")
            return None
        
        try:
            # Validate checksum if requested
            if validate_checksum:
                current_checksum = self._calculate_checksum(str(embedding_file))
                if current_checksum != metadata.checksum:
                    if self.logger:
                        self.logger.log_error("embedding_checksum_mismatch", 
                                            f"Checksum mismatch for {version}")
                    return None
            
            # Load embeddings
            embeddings = np.load(embedding_file)
            
            if self.logger:
                self.logger.log_performance_metrics("embedding_cache_load", {
                    "version": version,
                    "model_name": metadata.model_name,
                    "feature_dim": metadata.feature_dim,
                    "sample_count": metadata.sample_count,
                    "checksum_validated": validate_checksum
                })
            
            return embeddings
            
        except Exception as e:
            if self.logger:
                self.logger.log_error("embedding_load_error", 
                                    f"Failed to load embeddings {version}: {str(e)}")
            return None
    
    def list_versions(self) -> List[str]:
        """List all available embedding versions."""
        return list(self.metadata.keys())
    
    def get_metadata(self, version: str) -> Optional[EmbeddingMetadata]:
        """Get metadata for specific version."""
        return self.metadata.get(version)
    
    def delete_version(self, version: str) -> bool:
        """Delete specific embedding version."""
        if version not in self.metadata:
            return False
        
        try:
            safe_version = version.replace("/", "_").replace(":", "_")
            embedding_file = self.cache_dir / f"embeddings_{safe_version}.npy"
            
            if embedding_file.exists():
                embedding_file.unlink()
            
            del self.metadata[version]
            self._save_metadata()
            
            if self.logger:
                self.logger.log_performance_metrics("embedding_cache_delete", {
                    "version": version,
                    "file_path": str(embedding_file)
                })
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log_error("embedding_delete_error", 
                                    f"Failed to delete embedding {version}: {str(e)}")
            return False
    
    def cleanup_old_versions(self, keep_latest: int = 5) -> List[str]:
        """Clean up old embedding versions, keeping only the latest N."""
        if len(self.metadata) <= keep_latest:
            return []
        
        # Sort by creation timestamp
        sorted_versions = sorted(
            self.metadata.items(),
            key=lambda x: x[1].created_timestamp,
            reverse=True
        )
        
        # Delete old versions
        deleted_versions = []
        for version, _ in sorted_versions[keep_latest:]:
            if self.delete_version(version):
                deleted_versions.append(version)
        
        if self.logger and deleted_versions:
            self.logger.log_performance_metrics("embedding_cache_cleanup", {
                "deleted_versions": deleted_versions,
                "kept_versions": keep_latest,
                "total_deleted": len(deleted_versions)
            })
        
        return deleted_versions


class ImageCache:
    """Manages image cache with manifest and checksum validation."""
    
    def __init__(self, cache_dir: str = "images", logger=None):
        """
        Initialize image cache.
        
        Args:
            cache_dir: Directory for image cache
            logger: LoggingManager instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logger
        self.manifest_file = self.cache_dir / "image_manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, ImageCacheEntry]:
        """Load image cache manifest."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    data = json.load(f)
                return {k: ImageCacheEntry(**v) for k, v in data.items()}
            except Exception as e:
                if self.logger:
                    self.logger.log_error("manifest_load_error", 
                                        f"Failed to load image manifest: {str(e)}")
                return {}
        return {}
    
    def _save_manifest(self) -> None:
        """Save image cache manifest."""
        try:
            data = {k: asdict(v) for k, v in self.manifest.items()}
            with open(self.manifest_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.log_error("manifest_save_error", 
                                    f"Failed to save image manifest: {str(e)}")
    
    def add_image(self, sample_id: str, image_url: str, local_path: str, 
                  download_status: str, error_message: Optional[str] = None) -> None:
        """Add image entry to cache manifest."""
        try:
            # Calculate file info if successful download
            file_size = None
            checksum = None
            
            if download_status == 'success' and Path(local_path).exists():
                file_size = Path(local_path).stat().st_size
                checksum = self._calculate_checksum(local_path)
            
            entry = ImageCacheEntry(
                sample_id=sample_id,
                image_url=image_url,
                local_path=local_path,
                download_status=download_status,
                download_timestamp=datetime.now().isoformat(),
                file_size_bytes=file_size,
                checksum=checksum,
                error_message=error_message
            )
            
            self.manifest[sample_id] = entry
            self._save_manifest()
            
            if self.logger:
                self.logger.log_performance_metrics("image_cache_add", {
                    "sample_id": sample_id,
                    "download_status": download_status,
                    "file_size_bytes": file_size,
                    "has_error": error_message is not None
                })
                
        except Exception as e:
            if self.logger:
                self.logger.log_error("image_cache_add_error", 
                                    f"Failed to add image {sample_id}: {str(e)}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of image file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def get_image_path(self, sample_id: str) -> Optional[str]:
        """Get local path for cached image."""
        if sample_id in self.manifest:
            entry = self.manifest[sample_id]
            if entry.download_status == 'success' and Path(entry.local_path).exists():
                return entry.local_path
        return None
    
    def is_image_cached(self, sample_id: str) -> bool:
        """Check if image is successfully cached."""
        return self.get_image_path(sample_id) is not None
    
    def validate_cache_integrity(self) -> Dict[str, bool]:
        """Validate integrity of all cached images."""
        results = {}
        
        for sample_id, entry in self.manifest.items():
            if entry.download_status != 'success':
                results[sample_id] = False
                continue
            
            # Check file exists
            if not Path(entry.local_path).exists():
                results[sample_id] = False
                if self.logger:
                    self.logger.log_error("image_file_missing", 
                                        f"Image file missing for {sample_id}: {entry.local_path}")
                continue
            
            # Validate checksum if available
            if entry.checksum:
                current_checksum = self._calculate_checksum(entry.local_path)
                if current_checksum != entry.checksum:
                    results[sample_id] = False
                    if self.logger:
                        self.logger.log_error("image_checksum_mismatch", 
                                            f"Checksum mismatch for {sample_id}")
                    continue
            
            results[sample_id] = True
        
        # Log summary
        if self.logger:
            valid_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            self.logger.log_performance_metrics("image_cache_validation", {
                "total_images": total_count,
                "valid_images": valid_count,
                "validation_success_rate": valid_count / total_count if total_count > 0 else 0
            })
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_images = len(self.manifest)
        successful_downloads = sum(1 for entry in self.manifest.values() 
                                 if entry.download_status == 'success')
        failed_downloads = sum(1 for entry in self.manifest.values() 
                             if entry.download_status == 'failed')
        
        total_size = sum(entry.file_size_bytes or 0 for entry in self.manifest.values() 
                        if entry.file_size_bytes)
        
        return {
            "total_images": total_images,
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "success_rate": successful_downloads / total_images if total_images > 0 else 0,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024**2
        }


class ModelCheckpointManager:
    """Manages model checkpoints with configuration tracking."""
    
    def __init__(self, checkpoint_dir: str = "models", logger=None):
        """
        Initialize model checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for model checkpoints
            logger: LoggingManager instance
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logger
        self.registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelCheckpoint]:
        """Load checkpoint registry."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                return {k: ModelCheckpoint(**v) for k, v in data.items()}
            except Exception as e:
                if self.logger:
                    self.logger.log_error("registry_load_error", 
                                        f"Failed to load checkpoint registry: {str(e)}")
                return {}
        return {}
    
    def _save_registry(self) -> None:
        """Save checkpoint registry."""
        try:
            data = {k: asdict(v) for k, v in self.registry.items()}
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.log_error("registry_save_error", 
                                    f"Failed to save checkpoint registry: {str(e)}")
    
    def save_checkpoint(self, model: Any, checkpoint_id: str, model_type: str,
                       model_config: Dict[str, Any], training_config: Dict[str, Any],
                       performance_metrics: Dict[str, float]) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: Model object to save
            checkpoint_id: Unique identifier for checkpoint
            model_type: Type of model (e.g., 'xgboost', 'pytorch', 'sklearn')
            model_config: Model configuration parameters
            training_config: Training configuration
            performance_metrics: Performance metrics (SMAPE, etc.)
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        try:
            # Save model based on type
            if TORCH_AVAILABLE and hasattr(model, 'state_dict'):
                # PyTorch model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': model_config,
                    'training_config': training_config
                }, checkpoint_file)
            else:
                # Scikit-learn or other pickle-able models
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'model_config': model_config,
                        'training_config': training_config
                    }, f)
            
            # Calculate metadata
            file_size = checkpoint_file.stat().st_size
            checksum = self._calculate_checksum(str(checkpoint_file))
            
            # Create checkpoint metadata
            checkpoint = ModelCheckpoint(
                checkpoint_id=checkpoint_id,
                model_type=model_type,
                model_config=model_config,
                training_config=training_config,
                performance_metrics=performance_metrics,
                file_path=str(checkpoint_file),
                created_timestamp=datetime.now().isoformat(),
                checksum=checksum,
                file_size_bytes=file_size
            )
            
            # Store in registry
            self.registry[checkpoint_id] = checkpoint
            self._save_registry()
            
            if self.logger:
                self.logger.log_performance_metrics("checkpoint_save", {
                    "checkpoint_id": checkpoint_id,
                    "model_type": model_type,
                    "file_size_mb": file_size / 1024**2,
                    "performance_metrics": performance_metrics
                })
            
            return str(checkpoint_file)
            
        except Exception as e:
            if self.logger:
                self.logger.log_error("checkpoint_save_error", 
                                    f"Failed to save checkpoint {checkpoint_id}: {str(e)}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str, validate_checksum: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier
            validate_checksum: Whether to validate file integrity
            
        Returns:
            Dictionary containing model and metadata
        """
        if checkpoint_id not in self.registry:
            if self.logger:
                self.logger.log_error("checkpoint_not_found", 
                                    f"Checkpoint {checkpoint_id} not found in registry")
            return None
        
        checkpoint = self.registry[checkpoint_id]
        checkpoint_file = Path(checkpoint.file_path)
        
        if not checkpoint_file.exists():
            if self.logger:
                self.logger.log_error("checkpoint_file_missing", 
                                    f"Checkpoint file {checkpoint_file} not found")
            return None
        
        try:
            # Validate checksum if requested
            if validate_checksum:
                current_checksum = self._calculate_checksum(str(checkpoint_file))
                if current_checksum != checkpoint.checksum:
                    if self.logger:
                        self.logger.log_error("checkpoint_checksum_mismatch", 
                                            f"Checksum mismatch for {checkpoint_id}")
                    return None
            
            # Load checkpoint
            if TORCH_AVAILABLE and checkpoint.model_type == 'pytorch':
                data = torch.load(checkpoint_file, map_location='cpu')
            else:
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
            
            # Add metadata
            data['checkpoint_metadata'] = checkpoint
            
            if self.logger:
                self.logger.log_performance_metrics("checkpoint_load", {
                    "checkpoint_id": checkpoint_id,
                    "model_type": checkpoint.model_type,
                    "checksum_validated": validate_checksum
                })
            
            return data
            
        except Exception as e:
            if self.logger:
                self.logger.log_error("checkpoint_load_error", 
                                    f"Failed to load checkpoint {checkpoint_id}: {str(e)}")
            return None
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of checkpoint file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return list(self.registry.keys())
    
    def get_best_checkpoint(self, metric: str = 'smape_mean', minimize: bool = True) -> Optional[str]:
        """Get checkpoint with best performance metric."""
        if not self.registry:
            return None
        
        valid_checkpoints = [
            (checkpoint_id, checkpoint) 
            for checkpoint_id, checkpoint in self.registry.items()
            if metric in checkpoint.performance_metrics
        ]
        
        if not valid_checkpoints:
            return None
        
        best_checkpoint_id, _ = min(valid_checkpoints, 
                                   key=lambda x: x[1].performance_metrics[metric]) if minimize else \
                               max(valid_checkpoints, 
                                   key=lambda x: x[1].performance_metrics[metric])
        
        return best_checkpoint_id
    
    def cleanup_old_checkpoints(self, keep_best: int = 3, metric: str = 'smape_mean') -> List[str]:
        """Clean up old checkpoints, keeping only the best N."""
        if len(self.registry) <= keep_best:
            return []
        
        # Sort by performance metric
        sorted_checkpoints = sorted(
            self.registry.items(),
            key=lambda x: x[1].performance_metrics.get(metric, float('inf'))
        )
        
        # Delete worst checkpoints
        deleted_checkpoints = []
        for checkpoint_id, checkpoint in sorted_checkpoints[keep_best:]:
            try:
                checkpoint_file = Path(checkpoint.file_path)
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                
                del self.registry[checkpoint_id]
                deleted_checkpoints.append(checkpoint_id)
                
            except Exception as e:
                if self.logger:
                    self.logger.log_error("checkpoint_cleanup_error", 
                                        f"Failed to delete checkpoint {checkpoint_id}: {str(e)}")
        
        if deleted_checkpoints:
            self._save_registry()
            
            if self.logger:
                self.logger.log_performance_metrics("checkpoint_cleanup", {
                    "deleted_checkpoints": deleted_checkpoints,
                    "kept_checkpoints": keep_best,
                    "total_deleted": len(deleted_checkpoints)
                })
        
        return deleted_checkpoints


class ArtifactManager:
    """Unified artifact management and storage optimization."""
    
    def __init__(self, base_dir: str = ".", logger=None):
        """
        Initialize artifact manager.
        
        Args:
            base_dir: Base directory for all artifacts
            logger: LoggingManager instance
        """
        self.base_dir = Path(base_dir)
        self.logger = logger
        
        # Initialize component managers
        self.embedding_cache = EmbeddingCache(
            cache_dir=str(self.base_dir / "embeddings"), 
            logger=logger
        )
        self.image_cache = ImageCache(
            cache_dir=str(self.base_dir / "images"), 
            logger=logger
        )
        self.checkpoint_manager = ModelCheckpointManager(
            checkpoint_dir=str(self.base_dir / "models"), 
            logger=logger
        )
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get comprehensive storage summary for all artifacts."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "base_directory": str(self.base_dir),
            "components": {}
        }
        
        # Embedding cache stats
        embedding_versions = self.embedding_cache.list_versions()
        embedding_size = 0
        for version in embedding_versions:
            metadata = self.embedding_cache.get_metadata(version)
            if metadata:
                embedding_size += metadata.file_size_bytes
        
        summary["components"]["embeddings"] = {
            "version_count": len(embedding_versions),
            "total_size_bytes": embedding_size,
            "total_size_mb": embedding_size / 1024**2
        }
        
        # Image cache stats
        image_stats = self.image_cache.get_cache_stats()
        summary["components"]["images"] = image_stats
        
        # Model checkpoint stats
        checkpoints = self.checkpoint_manager.list_checkpoints()
        checkpoint_size = 0
        for checkpoint_id in checkpoints:
            checkpoint = self.checkpoint_manager.registry.get(checkpoint_id)
            if checkpoint:
                checkpoint_size += checkpoint.file_size_bytes
        
        summary["components"]["models"] = {
            "checkpoint_count": len(checkpoints),
            "total_size_bytes": checkpoint_size,
            "total_size_mb": checkpoint_size / 1024**2
        }
        
        # Calculate totals
        total_size = embedding_size + image_stats["total_size_bytes"] + checkpoint_size
        summary["total_size_bytes"] = total_size
        summary["total_size_gb"] = total_size / 1024**3
        
        if self.logger:
            self.logger.log_performance_metrics("artifact_storage_summary", summary)
        
        return summary
    
    def optimize_storage(self, target_size_gb: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize storage by cleaning up old artifacts.
        
        Args:
            target_size_gb: Target storage size in GB (optional)
            
        Returns:
            Summary of optimization actions
        """
        initial_summary = self.get_storage_summary()
        initial_size_gb = initial_summary["total_size_gb"]
        
        optimization_actions = {
            "initial_size_gb": initial_size_gb,
            "target_size_gb": target_size_gb,
            "actions_taken": []
        }
        
        # Clean up old embeddings (keep latest 3)
        deleted_embeddings = self.embedding_cache.cleanup_old_versions(keep_latest=3)
        if deleted_embeddings:
            optimization_actions["actions_taken"].append({
                "action": "cleanup_old_embeddings",
                "deleted_versions": deleted_embeddings
            })
        
        # Clean up old checkpoints (keep best 3)
        deleted_checkpoints = self.checkpoint_manager.cleanup_old_checkpoints(keep_best=3)
        if deleted_checkpoints:
            optimization_actions["actions_taken"].append({
                "action": "cleanup_old_checkpoints", 
                "deleted_checkpoints": deleted_checkpoints
            })
        
        # Get final summary
        final_summary = self.get_storage_summary()
        final_size_gb = final_summary["total_size_gb"]
        
        optimization_actions["final_size_gb"] = final_size_gb
        optimization_actions["size_reduction_gb"] = initial_size_gb - final_size_gb
        optimization_actions["size_reduction_percent"] = (
            (initial_size_gb - final_size_gb) / initial_size_gb * 100 
            if initial_size_gb > 0 else 0
        )
        
        if self.logger:
            self.logger.log_performance_metrics("storage_optimization", optimization_actions)
        
        return optimization_actions
    
    def validate_all_artifacts(self) -> Dict[str, Any]:
        """Validate integrity of all cached artifacts."""
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "embeddings": {},
            "images": {},
            "models": {},
            "overall_status": "unknown"
        }
        
        # Validate embeddings
        embedding_versions = self.embedding_cache.list_versions()
        embedding_valid_count = 0
        for version in embedding_versions:
            embeddings = self.embedding_cache.load_embeddings(version, validate_checksum=True)
            is_valid = embeddings is not None
            validation_results["embeddings"][version] = is_valid
            if is_valid:
                embedding_valid_count += 1
        
        # Validate images
        image_validation = self.image_cache.validate_cache_integrity()
        validation_results["images"] = image_validation
        image_valid_count = sum(1 for v in image_validation.values() if v)
        
        # Validate model checkpoints
        checkpoints = self.checkpoint_manager.list_checkpoints()
        model_valid_count = 0
        for checkpoint_id in checkpoints:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(
                checkpoint_id, validate_checksum=True
            )
            is_valid = checkpoint_data is not None
            validation_results["models"][checkpoint_id] = is_valid
            if is_valid:
                model_valid_count += 1
        
        # Calculate overall status
        total_artifacts = len(embedding_versions) + len(image_validation) + len(checkpoints)
        valid_artifacts = embedding_valid_count + image_valid_count + model_valid_count
        
        if total_artifacts == 0:
            validation_results["overall_status"] = "no_artifacts"
        elif valid_artifacts == total_artifacts:
            validation_results["overall_status"] = "all_valid"
        elif valid_artifacts == 0:
            validation_results["overall_status"] = "all_invalid"
        else:
            validation_results["overall_status"] = "partially_valid"
        
        validation_results["summary"] = {
            "total_artifacts": total_artifacts,
            "valid_artifacts": valid_artifacts,
            "validation_success_rate": valid_artifacts / total_artifacts if total_artifacts > 0 else 0
        }
        
        if self.logger:
            self.logger.log_performance_metrics("artifact_validation", validation_results["summary"])
        
        return validation_results