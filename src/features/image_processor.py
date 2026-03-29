"""
Enhanced ImageProcessor with robust preprocessing for ML Product Pricing Challenge 2025

This module implements image loading, resizing, normalization, augmentation, and fallback
mechanisms for corrupted or missing images as required by task 4.1.
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from torchvision import transforms
import torch
from dataclasses import dataclass

try:
    from ..config import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config


@dataclass
class ImageProcessingResult:
    """Result of image processing operation"""
    success: bool
    processed_image: Optional[np.ndarray] = None
    original_size: Optional[Tuple[int, int]] = None
    processed_size: Optional[Tuple[int, int]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


class ImageProcessor:
    """
    Enhanced ImageProcessor with robust preprocessing capabilities
    
    Implements:
    - Image loading, resizing, and normalization functions
    - Image augmentation techniques for training data enhancement  
    - Fallback mechanisms for corrupted or missing images
    """
    
    def __init__(self, image_config=None):
        """
        Initialize ImageProcessor with configuration
        
        Args:
            image_config: ImageFeatureConfig instance or None for default
        """
        self.config = image_config or config.image_features
        self.logger = logging.getLogger(__name__)
        
        # Initialize transforms
        self._setup_transforms()
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'corrupted_images': 0,
            'missing_images': 0,
            'augmented_images': 0,
            'fallback_used': 0,
            'total_processing_time': 0.0
        }
        
        # Cache for default fallback images
        self._fallback_cache = {}
    
    def _setup_transforms(self):
        """Setup image transformation pipelines"""
        # Basic preprocessing transform
        self.preprocess_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
        
        # Augmentation transforms for training
        if self.config.use_augmentation:
            self.augmentation_transform = transforms.Compose([
                transforms.RandomRotation(self.config.rotation_range),
                transforms.ColorJitter(brightness=self.config.brightness_range),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                )
            ])
        else:
            self.augmentation_transform = self.preprocess_transform
    
    def load_and_preprocess_image(self, image_path: str, apply_augmentation: bool = False) -> ImageProcessingResult:
        """
        Load and preprocess a single image with robust error handling
        
        Args:
            image_path: Path to image file
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            ImageProcessingResult with processed image or error information
        """
        import time
        start_time = time.time()
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                self.processing_stats['missing_images'] += 1
                return ImageProcessingResult(
                    success=False,
                    error_message=f"Image file not found: {image_path}",
                    processing_time=time.time() - start_time
                )
            
            # Load image with PIL
            try:
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    original_size = img.size
                    
                    # Verify image is not corrupted
                    img.verify()
                    
                    # Reload image after verify (verify() closes the file)
                    with Image.open(image_path) as img_reload:
                        if img_reload.mode != 'RGB':
                            img_reload = img_reload.convert('RGB')
                        
                        # Apply appropriate transform
                        if apply_augmentation and self.config.use_augmentation:
                            processed_tensor = self.augmentation_transform(img_reload)
                            self.processing_stats['augmented_images'] += 1
                        else:
                            processed_tensor = self.preprocess_transform(img_reload)
                        
                        # Convert to numpy array
                        processed_image = processed_tensor.numpy()
                        processed_size = processed_image.shape[1:3]  # H, W
                        
                        self.processing_stats['successful_loads'] += 1
                        processing_time = time.time() - start_time
                        self.processing_stats['total_processing_time'] += processing_time
                        
                        return ImageProcessingResult(
                            success=True,
                            processed_image=processed_image,
                            original_size=original_size,
                            processed_size=processed_size,
                            processing_time=processing_time
                        )
            
            except (IOError, OSError, Image.UnidentifiedImageError) as e:
                self.processing_stats['corrupted_images'] += 1
                return ImageProcessingResult(
                    success=False,
                    error_message=f"Corrupted or invalid image: {str(e)}",
                    processing_time=time.time() - start_time
                )
        
        except Exception as e:
            self.processing_stats['failed_loads'] += 1
            return ImageProcessingResult(
                success=False,
                error_message=f"Unexpected error processing image: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        finally:
            self.processing_stats['total_processed'] += 1
    
    def create_fallback_image(self, fallback_type: str = "default") -> np.ndarray:
        """
        Create fallback image for missing or corrupted images
        
        Args:
            fallback_type: Type of fallback image to create
            
        Returns:
            np.ndarray: Processed fallback image tensor
        """
        if fallback_type in self._fallback_cache:
            return self._fallback_cache[fallback_type].copy()
        
        try:
            if fallback_type == "default":
                # Create a neutral gray image
                fallback_img = Image.new('RGB', self.config.image_size, color=(128, 128, 128))
            elif fallback_type == "noise":
                # Create a noise pattern
                noise = np.random.randint(0, 256, (*self.config.image_size, 3), dtype=np.uint8)
                fallback_img = Image.fromarray(noise)
            elif fallback_type == "gradient":
                # Create a gradient pattern
                gradient = np.zeros((*self.config.image_size, 3), dtype=np.uint8)
                for i in range(self.config.image_size[0]):
                    gradient[i, :, :] = int(255 * i / self.config.image_size[0])
                fallback_img = Image.fromarray(gradient)
            else:
                # Default to gray
                fallback_img = Image.new('RGB', self.config.image_size, color=(128, 128, 128))
            
            # Process fallback image
            processed_tensor = self.preprocess_transform(fallback_img)
            processed_image = processed_tensor.numpy()
            
            # Cache the result
            self._fallback_cache[fallback_type] = processed_image.copy()
            
            return processed_image
        
        except Exception as e:
            self.logger.error(f"Failed to create fallback image: {str(e)}")
            # Return zero tensor as last resort
            return np.zeros((3, *self.config.image_size), dtype=np.float32)
    
    def handle_missing_images(self, sample_id: str, text_content: Optional[str] = None) -> np.ndarray:
        """
        Handle missing images with fallback mechanisms
        
        Args:
            sample_id: Sample identifier for logging
            text_content: Optional text content to influence fallback
            
        Returns:
            np.ndarray: Fallback image tensor
        """
        self.processing_stats['fallback_used'] += 1
        
        if self.config.missing_image_strategy == "zero":
            return np.zeros((3, *self.config.image_size), dtype=np.float32)
        
        elif self.config.missing_image_strategy == "mean":
            # Use mean values from normalization
            mean_tensor = torch.tensor(self.config.normalize_mean).view(3, 1, 1)
            return mean_tensor.expand(3, *self.config.image_size).numpy()
        
        elif self.config.missing_image_strategy == "text_based":
            # Create fallback based on text content
            if text_content:
                # Use text hash to create consistent fallback
                text_hash = hashlib.md5(text_content.encode()).hexdigest()
                fallback_type = "gradient" if int(text_hash[:2], 16) % 2 else "default"
            else:
                fallback_type = "default"
            
            fallback_image = self.create_fallback_image(fallback_type)
            self.logger.info(f"Using {fallback_type} fallback for sample {sample_id}")
            return fallback_image
        
        else:
            # Default fallback
            return self.create_fallback_image("default")
    
    def batch_process_images(self, image_paths: List[str], apply_augmentation: bool = False, 
                           sample_ids: Optional[List[str]] = None,
                           text_contents: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Process multiple images in batch with fallback handling
        
        Args:
            image_paths: List of image file paths
            apply_augmentation: Whether to apply data augmentation
            sample_ids: Optional list of sample IDs for logging
            text_contents: Optional list of text contents for fallback generation
            
        Returns:
            Dict mapping sample_id (or index) to processed image tensor
        """
        results = {}
        
        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(image_paths))]
        
        if text_contents is None:
            text_contents = [None] * len(image_paths)
        
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        
        for i, (image_path, sample_id, text_content) in enumerate(zip(image_paths, sample_ids, text_contents)):
            try:
                # Attempt to process image
                result = self.load_and_preprocess_image(image_path, apply_augmentation)
                
                if result.success:
                    results[sample_id] = result.processed_image
                else:
                    # Use fallback mechanism
                    self.logger.warning(f"Image processing failed for {sample_id}: {result.error_message}")
                    fallback_image = self.handle_missing_images(sample_id, text_content)
                    results[sample_id] = fallback_image
            
            except Exception as e:
                self.logger.error(f"Unexpected error processing image {sample_id}: {str(e)}")
                fallback_image = self.handle_missing_images(sample_id, text_content)
                results[sample_id] = fallback_image
        
        self.logger.info(f"Batch processing completed: {len(results)} images processed")
        return results
    
    def resize_image(self, image: Union[np.ndarray, Image.Image], target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size with proper aspect ratio handling
        
        Args:
            image: Input image as numpy array or PIL Image
            target_size: Target size as (width, height)
            
        Returns:
            np.ndarray: Resized image
        """
        try:
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                
                if len(image.shape) == 3 and image.shape[0] == 3:
                    # CHW format, convert to HWC
                    image = np.transpose(image, (1, 2, 0))
                
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Resize with high-quality resampling
            resized_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert back to numpy array
            return np.array(resized_image)
        
        except Exception as e:
            self.logger.error(f"Error resizing image: {str(e)}")
            # Return zero array as fallback
            return np.zeros((*target_size[::-1], 3), dtype=np.uint8)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using configured mean and std
        
        Args:
            image: Input image as numpy array (HWC format, 0-255)
            
        Returns:
            np.ndarray: Normalized image tensor (CHW format)
        """
        try:
            # Convert to float and normalize to [0, 1]
            image_float = image.astype(np.float32) / 255.0
            
            # Convert to CHW format
            if len(image_float.shape) == 3:
                image_float = np.transpose(image_float, (2, 0, 1))
            
            # Apply normalization
            mean = np.array(self.config.normalize_mean).reshape(3, 1, 1)
            std = np.array(self.config.normalize_std).reshape(3, 1, 1)
            
            normalized = (image_float - mean) / std
            
            return normalized
        
        except Exception as e:
            self.logger.error(f"Error normalizing image: {str(e)}")
            # Return zero tensor as fallback
            return np.zeros((3, *self.config.image_size), dtype=np.float32)
    
    def apply_augmentation(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Augmented image tensor
        """
        try:
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                
                if len(image.shape) == 3 and image.shape[0] == 3:
                    # CHW format, convert to HWC
                    image = np.transpose(image, (1, 2, 0))
                
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Apply augmentation transform
            augmented_tensor = self.augmentation_transform(pil_image)
            return augmented_tensor.numpy()
        
        except Exception as e:
            self.logger.error(f"Error applying augmentation: {str(e)}")
            # Fallback to basic preprocessing
            return self.preprocess_transform(pil_image).numpy()
    
    def validate_image_integrity(self, image_path: str) -> bool:
        """
        Validate image file integrity
        
        Args:
            image_path: Path to image file
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            if not os.path.exists(image_path):
                return False
            
            with Image.open(image_path) as img:
                img.verify()
                return True
        
        except Exception:
            return False
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics
        
        Returns:
            Dict containing processing statistics and analysis
        """
        stats = self.processing_stats.copy()
        
        # Calculate derived statistics
        if stats['total_processed'] > 0:
            stats['success_rate'] = (stats['successful_loads'] / stats['total_processed']) * 100
            stats['failure_rate'] = (stats['failed_loads'] / stats['total_processed']) * 100
            stats['corruption_rate'] = (stats['corrupted_images'] / stats['total_processed']) * 100
            stats['missing_rate'] = (stats['missing_images'] / stats['total_processed']) * 100
            stats['fallback_rate'] = (stats['fallback_used'] / stats['total_processed']) * 100
        
        if stats['successful_loads'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['successful_loads']
        
        # Add configuration info
        stats['image_size'] = self.config.image_size
        stats['augmentation_enabled'] = self.config.use_augmentation
        stats['missing_image_strategy'] = self.config.missing_image_strategy
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        for key in self.processing_stats:
            if isinstance(self.processing_stats[key], (int, float)):
                self.processing_stats[key] = 0
        self.logger.info("Processing statistics reset")
    
    def cleanup_cache(self):
        """Clean up cached fallback images"""
        self._fallback_cache.clear()
        self.logger.info("Fallback image cache cleared")