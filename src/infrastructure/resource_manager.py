"""
Resource management for GPU, memory, and storage monitoring.
Handles resource allocation and monitoring for ML training and inference.
"""

import os
import psutil
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import subprocess

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""
    timestamp: str
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    gpu_info: Optional[Dict[str, Any]] = None


@dataclass
class StorageRequirement:
    """Storage requirement calculation."""
    component: str
    size_gb: float
    description: str
    path: str
    checksum: Optional[str] = None


class ResourceManager:
    """Manages system resources and monitors usage."""
    
    def __init__(self, logger=None):
        """
        Initialize resource manager.
        
        Args:
            logger: LoggingManager instance for structured logging
        """
        self.logger = logger
        self.resource_history = []
        self.storage_requirements = []
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information and usage."""
        gpu_info = {
            "available": False,
            "count": 0,
            "devices": []
        }
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_info["available"] = len(gpus) > 0
                gpu_info["count"] = len(gpus)
                
                for i, gpu in enumerate(gpus):
                    device_info = {
                        "id": i,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "gpu_percent": gpu.load * 100,
                        "temperature": gpu.temperature
                    }
                    gpu_info["devices"].append(device_info)
                    
            except Exception as e:
                if self.logger:
                    self.logger.log_error("gpu_info_error", str(e))
        
        # Try PyTorch GPU info as fallback
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if not gpu_info["available"]:
                gpu_info["available"] = True
                gpu_info["count"] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(i) / 1024**3
                    memory_total = props.total_memory / 1024**3
                    
                    device_info = {
                        "id": i,
                        "name": props.name,
                        "memory_total": memory_total,
                        "memory_allocated": memory_allocated,
                        "memory_cached": memory_cached,
                        "memory_free": memory_total - memory_cached,
                        "compute_capability": f"{props.major}.{props.minor}"
                    }
                    gpu_info["devices"].append(device_info)
        
        return gpu_info
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / 1024**3,
            "available_gb": memory.available / 1024**3,
            "used_gb": memory.used / 1024**3,
            "percent": memory.percent
        }
    
    def get_disk_info(self, path: str = ".") -> Dict[str, float]:
        """Get disk usage information for given path."""
        disk = psutil.disk_usage(path)
        return {
            "total_gb": disk.total / 1024**3,
            "used_gb": disk.used / 1024**3,
            "free_gb": disk.free / 1024**3,
            "percent": (disk.used / disk.total) * 100
        }
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information and usage."""
        return {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else None
        }
    
    def capture_resource_snapshot(self) -> ResourceUsage:
        """Capture current resource usage snapshot."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get system info
        memory_info = self.get_memory_info()
        disk_info = self.get_disk_info()
        cpu_info = self.get_cpu_info()
        gpu_info = self.get_gpu_info()
        
        snapshot = ResourceUsage(
            timestamp=timestamp,
            cpu_percent=cpu_info["percent"],
            memory_used_gb=memory_info["used_gb"],
            memory_total_gb=memory_info["total_gb"],
            memory_percent=memory_info["percent"],
            disk_used_gb=disk_info["used_gb"],
            disk_total_gb=disk_info["total_gb"],
            disk_percent=disk_info["percent"],
            gpu_info=gpu_info if gpu_info["available"] else None
        )
        
        self.resource_history.append(snapshot)
        
        if self.logger:
            self.logger.log_performance_metrics("resource_snapshot", {
                "cpu_percent": snapshot.cpu_percent,
                "memory_used_gb": snapshot.memory_used_gb,
                "memory_percent": snapshot.memory_percent,
                "disk_percent": snapshot.disk_percent,
                "gpu_available": gpu_info["available"],
                "gpu_count": gpu_info["count"]
            })
        
        return snapshot
    
    def check_gpu_requirements(self, required_memory_gb: float = 16.0) -> bool:
        """
        Check if GPU requirements are met.
        
        Args:
            required_memory_gb: Minimum required GPU memory in GB
            
        Returns:
            True if requirements are met
        """
        gpu_info = self.get_gpu_info()
        
        if not gpu_info["available"]:
            if self.logger:
                self.logger.log_error("gpu_requirement_check", 
                                    "No GPU available but GPU required")
            return False
        
        # Check if any GPU has sufficient memory
        for device in gpu_info["devices"]:
            available_memory = device.get("memory_free", 0)
            if "memory_total" in device:
                available_memory = device["memory_total"] - device.get("memory_used", 0)
            
            if available_memory >= required_memory_gb:
                if self.logger:
                    self.logger.log_performance_metrics("gpu_requirement_check", {
                        "required_memory_gb": required_memory_gb,
                        "available_memory_gb": available_memory,
                        "gpu_id": device["id"],
                        "gpu_name": device["name"],
                        "requirement_met": True
                    })
                return True
        
        if self.logger:
            self.logger.log_error("gpu_requirement_check", 
                                f"No GPU with {required_memory_gb}GB available memory found")
        
        return False
    
    def monitor_memory_usage(self, threshold_percent: float = 90.0) -> bool:
        """
        Monitor memory usage and warn if threshold exceeded.
        
        Args:
            threshold_percent: Memory usage threshold for warnings
            
        Returns:
            True if memory usage is below threshold
        """
        memory_info = self.get_memory_info()
        
        if memory_info["percent"] > threshold_percent:
            if self.logger:
                self.logger.log_error("memory_threshold_exceeded", 
                                    f"Memory usage {memory_info['percent']:.1f}% exceeds threshold {threshold_percent}%")
            return False
        
        return True
    
    def calculate_storage_requirements(self, base_path: str = ".") -> List[StorageRequirement]:
        """Calculate storage requirements for different components."""
        requirements = []
        base_path = Path(base_path)
        
        # Define expected storage components
        components = {
            "images": {"path": "images", "description": "Downloaded product images"},
            "embeddings": {"path": "embeddings", "description": "Cached embeddings and features"},
            "models": {"path": "models", "description": "Trained model checkpoints"},
            "cache": {"path": "cache", "description": "General cache files"},
            "logs": {"path": "logs", "description": "Structured logs and metrics"},
            "dataset": {"path": "dataset", "description": "Training and test datasets"}
        }
        
        for component, info in components.items():
            component_path = base_path / info["path"]
            
            if component_path.exists():
                size_bytes = sum(f.stat().st_size for f in component_path.rglob('*') if f.is_file())
                size_gb = size_bytes / 1024**3
            else:
                # Estimate based on component type
                size_gb = self._estimate_component_size(component)
            
            requirement = StorageRequirement(
                component=component,
                size_gb=size_gb,
                description=info["description"],
                path=str(component_path)
            )
            requirements.append(requirement)
        
        self.storage_requirements = requirements
        
        if self.logger:
            total_size = sum(req.size_gb for req in requirements)
            self.logger.log_performance_metrics("storage_requirements", {
                "total_size_gb": total_size,
                "components": {req.component: req.size_gb for req in requirements}
            })
        
        return requirements
    
    def _estimate_component_size(self, component: str) -> float:
        """Estimate storage size for component if not present."""
        estimates = {
            "images": 10.0,  # ~10GB for 75k images
            "embeddings": 5.0,  # ~5GB for embeddings
            "models": 2.0,  # ~2GB for model checkpoints
            "cache": 1.0,  # ~1GB for cache
            "logs": 0.5,  # ~500MB for logs
            "dataset": 0.1  # ~100MB for CSV files
        }
        return estimates.get(component, 1.0)
    
    def validate_storage_space(self, safety_margin_gb: float = 5.0) -> bool:
        """
        Validate sufficient storage space is available.
        
        Args:
            safety_margin_gb: Additional space to keep free
            
        Returns:
            True if sufficient space available
        """
        disk_info = self.get_disk_info()
        requirements = self.calculate_storage_requirements()
        
        total_required = sum(req.size_gb for req in requirements) + safety_margin_gb
        available_space = disk_info["free_gb"]
        
        is_sufficient = available_space >= total_required
        
        if self.logger:
            self.logger.log_performance_metrics("storage_validation", {
                "required_gb": total_required,
                "available_gb": available_space,
                "safety_margin_gb": safety_margin_gb,
                "is_sufficient": is_sufficient
            })
        
        if not is_sufficient:
            if self.logger:
                self.logger.log_error("insufficient_storage", 
                                    f"Need {total_required:.1f}GB but only {available_space:.1f}GB available")
        
        return is_sufficient
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary."""
        current_snapshot = self.capture_resource_snapshot()
        storage_reqs = self.calculate_storage_requirements()
        
        return {
            "timestamp": current_snapshot.timestamp,
            "cpu": {
                "percent": current_snapshot.cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "used_gb": current_snapshot.memory_used_gb,
                "total_gb": current_snapshot.memory_total_gb,
                "percent": current_snapshot.memory_percent
            },
            "disk": {
                "used_gb": current_snapshot.disk_used_gb,
                "total_gb": current_snapshot.disk_total_gb,
                "percent": current_snapshot.disk_percent
            },
            "gpu": current_snapshot.gpu_info,
            "storage_requirements": {
                "total_gb": sum(req.size_gb for req in storage_reqs),
                "components": {req.component: req.size_gb for req in storage_reqs}
            }
        }


class ChecksumValidator:
    """Validates file integrity using checksums."""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def calculate_file_checksum(self, file_path: str, algorithm: str = "md5") -> str:
        """Calculate checksum for a file."""
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            if self.logger:
                self.logger.log_error("checksum_calculation_error", 
                                    f"Failed to calculate checksum for {file_path}: {str(e)}")
            raise
    
    def validate_file_checksum(self, file_path: str, expected_checksum: str, 
                             algorithm: str = "md5") -> bool:
        """Validate file against expected checksum."""
        try:
            actual_checksum = self.calculate_file_checksum(file_path, algorithm)
            is_valid = actual_checksum == expected_checksum
            
            if self.logger:
                self.logger.log_performance_metrics("checksum_validation", {
                    "file_path": file_path,
                    "expected_checksum": expected_checksum,
                    "actual_checksum": actual_checksum,
                    "is_valid": is_valid,
                    "algorithm": algorithm
                })
            
            return is_valid
        except Exception as e:
            if self.logger:
                self.logger.log_error("checksum_validation_error", 
                                    f"Failed to validate checksum for {file_path}: {str(e)}")
            return False
    
    def create_checksum_manifest(self, directory: str, output_file: str = None) -> Dict[str, str]:
        """Create checksum manifest for all files in directory."""
        directory = Path(directory)
        manifest = {}
        
        if not directory.exists():
            if self.logger:
                self.logger.log_error("manifest_creation_error", 
                                    f"Directory {directory} does not exist")
            return manifest
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                try:
                    relative_path = str(file_path.relative_to(directory))
                    checksum = self.calculate_file_checksum(str(file_path))
                    manifest[relative_path] = checksum
                except Exception as e:
                    if self.logger:
                        self.logger.log_error("manifest_file_error", 
                                            f"Failed to process {file_path}: {str(e)}")
        
        # Save manifest if output file specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                if self.logger:
                    self.logger.log_performance_metrics("manifest_created", {
                        "directory": str(directory),
                        "output_file": output_file,
                        "file_count": len(manifest)
                    })
            except Exception as e:
                if self.logger:
                    self.logger.log_error("manifest_save_error", 
                                        f"Failed to save manifest: {str(e)}")
        
        return manifest
    
    def validate_directory_checksums(self, directory: str, manifest_file: str) -> Dict[str, bool]:
        """Validate all files in directory against manifest."""
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            if self.logger:
                self.logger.log_error("manifest_load_error", 
                                    f"Failed to load manifest {manifest_file}: {str(e)}")
            return {}
        
        directory = Path(directory)
        results = {}
        
        for relative_path, expected_checksum in manifest.items():
            file_path = directory / relative_path
            
            if not file_path.exists():
                results[relative_path] = False
                if self.logger:
                    self.logger.log_error("missing_file", 
                                        f"File {relative_path} missing from directory")
                continue
            
            is_valid = self.validate_file_checksum(str(file_path), expected_checksum)
            results[relative_path] = is_valid
        
        # Summary logging
        valid_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        if self.logger:
            self.logger.log_performance_metrics("directory_validation", {
                "directory": str(directory),
                "manifest_file": manifest_file,
                "total_files": total_count,
                "valid_files": valid_count,
                "validation_success_rate": valid_count / total_count if total_count > 0 else 0
            })
        
        return results