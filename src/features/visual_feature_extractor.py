"""
Visual Feature Extractor using Pre-trained Models for ML Product Pricing Challenge 2025

This module implements visual feature extraction using pre-trained CNN models with proper
licensing validation, color histogram and dominant color extraction, and texture analysis
and edge detection features as required by task 4.3.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import cv2
from PIL import Image, ImageStat
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from skimage import feature, filters, measure
from skimage.color import rgb2gray, rgb2hsv
from skimage.segmentation import slic
from scipy import ndimage
import pandas as pd

try:
    from ..config import config
    from .image_processor import ImageProcessor
    from .image_embedding_system import ImageEmbeddingSystem
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config
    from features.image_processor import ImageProcessor
    from features.image_embedding_system import ImageEmbeddingSystem


class LicenseValidationError(Exception):
    """Exception for license validation failures"""
    pass


class VisualFeatureExtractionError(Exception):
    """Exception for visual feature extraction failures"""
    pass


class VisualFeatureExtractor:
    """
    Visual Feature Extractor using pre-trained models and traditional computer vision
    
    Implements:
    - CNN models for deep visual features with proper licensing validation
    - Color histograms and dominant color features from product images
    - Texture analysis and edge detection features
    """
    
    # Licensed models and their licenses
    LICENSED_MODELS = {
        'resnet50': {
            'license': 'BSD-3-Clause',
            'source': 'torchvision',
            'weights': 'IMAGENET1K_V2',
            'max_params': '25.6M'
        },
        'resnet101': {
            'license': 'BSD-3-Clause', 
            'source': 'torchvision',
            'weights': 'IMAGENET1K_V2',
            'max_params': '44.5M'
        },
        'efficientnet-b0': {
            'license': 'Apache-2.0',
            'source': 'torchvision',
            'weights': 'IMAGENET1K_V1',
            'max_params': '5.3M'
        },
        'efficientnet-b3': {
            'license': 'Apache-2.0',
            'source': 'torchvision', 
            'weights': 'IMAGENET1K_V1',
            'max_params': '12.2M'
        },
        'vit-base': {
            'license': 'Apache-2.0',
            'source': 'torchvision',
            'weights': 'IMAGENET1K_V1',
            'max_params': '86.6M'
        }
    }
    
    def __init__(self, image_config=None):
        """
        Initialize VisualFeatureExtractor
        
        Args:
            image_config: ImageFeatureConfig instance or None for default
        """
        self.config = image_config or config.image_features
        self.logger = logging.getLogger(__name__)
        
        # Validate model license
        self._validate_model_license(self.config.cnn_model)
        
        # Initialize components
        self.image_processor = ImageProcessor(self.config)
        self.embedding_system = ImageEmbeddingSystem(self.config)
        
        # Feature extraction statistics
        self.extraction_stats = {
            'total_extractions': 0,
            'cnn_extractions': 0,
            'color_extractions': 0,
            'texture_extractions': 0,
            'edge_extractions': 0,
            'failed_extractions': 0,
            'total_extraction_time': 0.0,
            'avg_extraction_time': 0.0
        }
        
        # Feature dimensions
        self.feature_dimensions = {
            'cnn_features': self.config.feature_dim,
            'color_histogram': self.config.color_histogram_bins * 3,  # RGB
            'dominant_colors': 15,  # 5 colors * 3 channels
            'texture_features': 13,  # LBP + Haralick features
            'edge_features': 4,  # Edge statistics
            'shape_features': 7   # Shape and contour features
        }
    
    def _validate_model_license(self, model_name: str):
        """
        Validate that the model has proper licensing
        
        Args:
            model_name: Name of the model to validate
            
        Raises:
            LicenseValidationError: If model license is not valid
        """
        if model_name not in self.LICENSED_MODELS:
            raise LicenseValidationError(f"Model {model_name} is not in approved license list")
        
        model_info = self.LICENSED_MODELS[model_name]
        allowed_licenses = ['MIT', 'Apache-2.0', 'BSD-3-Clause']
        
        if model_info['license'] not in allowed_licenses:
            raise LicenseValidationError(
                f"Model {model_name} has license {model_info['license']} "
                f"which is not in allowed licenses: {allowed_licenses}"
            )
        
        self.logger.info(f"Model {model_name} license validated: {model_info['license']}")
    
    def extract_cnn_features(self, image_path: str, sample_id: str) -> np.ndarray:
        """
        Extract deep visual features using pre-trained CNN models
        
        Args:
            image_path: Path to image file
            sample_id: Sample identifier
            
        Returns:
            np.ndarray: CNN feature vector
        """
        try:
            # Use embedding system for CNN feature extraction
            embedding = self.embedding_system.extract_embedding_from_image_path(image_path, sample_id)
            self.extraction_stats['cnn_extractions'] += 1
            return embedding
            
        except Exception as e:
            self.logger.error(f"CNN feature extraction failed for {sample_id}: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.config.feature_dim, dtype=np.float32)
    
    def extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features from image
        
        Args:
            image: Image as numpy array (HWC format, 0-255)
            
        Returns:
            np.ndarray: Color histogram features
        """
        try:
            if len(image.shape) == 3 and image.shape[0] == 3:
                # Convert CHW to HWC
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Calculate histograms for each channel
            bins = self.config.color_histogram_bins
            
            hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
            
            # Normalize histograms
            hist_r = hist_r.flatten() / np.sum(hist_r)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_b = hist_b.flatten() / np.sum(hist_b)
            
            # Combine histograms
            color_histogram = np.concatenate([hist_r, hist_g, hist_b])
            
            self.extraction_stats['color_extractions'] += 1
            return color_histogram.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Color histogram extraction failed: {str(e)}")
            return np.zeros(self.feature_dimensions['color_histogram'], dtype=np.float32)
    
    def extract_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> np.ndarray:
        """
        Extract dominant color features using K-means clustering
        
        Args:
            image: Image as numpy array (HWC format, 0-255)
            n_colors: Number of dominant colors to extract
            
        Returns:
            np.ndarray: Dominant color features (flattened RGB values)
        """
        try:
            if len(image.shape) == 3 and image.shape[0] == 3:
                # Convert CHW to HWC
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3)
            
            # Remove pure black pixels (likely background)
            non_black_mask = np.sum(pixels, axis=1) > 10
            if np.sum(non_black_mask) > 100:  # Ensure we have enough pixels
                pixels = pixels[non_black_mask]
            
            # Apply K-means clustering
            if len(pixels) < n_colors:
                # Not enough pixels, pad with zeros
                dominant_colors = np.zeros((n_colors, 3))
                dominant_colors[:len(pixels)] = pixels[:n_colors]
            else:
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                dominant_colors = kmeans.cluster_centers_
            
            # Normalize to [0, 1]
            dominant_colors = dominant_colors / 255.0
            
            # Flatten to feature vector
            dominant_color_features = dominant_colors.flatten()
            
            return dominant_color_features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Dominant color extraction failed: {str(e)}")
            return np.zeros(self.feature_dimensions['dominant_colors'], dtype=np.float32)
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract texture features using Local Binary Patterns and Haralick features
        
        Args:
            image: Image as numpy array
            
        Returns:
            np.ndarray: Texture feature vector
        """
        try:
            if len(image.shape) == 3 and image.shape[0] == 3:
                # Convert CHW to HWC
                image = np.transpose(image, (1, 2, 0))
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = rgb2gray(image)
            else:
                gray = image
            
            # Ensure correct data type
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8)
            
            texture_features = []
            
            # Local Binary Pattern features
            try:
                radius = 3
                n_points = 8 * radius
                lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
                
                # Calculate LBP histogram
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                         range=(0, n_points + 2), density=True)
                
                # Use first 8 bins as features
                texture_features.extend(lbp_hist[:8])
                
            except Exception as e:
                self.logger.warning(f"LBP extraction failed: {str(e)}")
                texture_features.extend([0.0] * 8)
            
            # Gray-Level Co-occurrence Matrix (GLCM) features
            try:
                from skimage.feature import graycomatrix, graycoprops
                
                # Calculate GLCM
                distances = [1]
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                
                glcm = graycomatrix(gray, distances=distances, angles=angles, 
                                  levels=256, symmetric=True, normed=True)
                
                # Extract Haralick features
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                
                texture_features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
                
            except Exception as e:
                self.logger.warning(f"GLCM extraction failed: {str(e)}")
                texture_features.extend([0.0] * 5)
            
            # Pad or truncate to expected size
            expected_size = self.feature_dimensions['texture_features']
            if len(texture_features) < expected_size:
                texture_features.extend([0.0] * (expected_size - len(texture_features)))
            else:
                texture_features = texture_features[:expected_size]
            
            self.extraction_stats['texture_extractions'] += 1
            return np.array(texture_features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Texture feature extraction failed: {str(e)}")
            return np.zeros(self.feature_dimensions['texture_features'], dtype=np.float32)
    
    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract edge detection features
        
        Args:
            image: Image as numpy array
            
        Returns:
            np.ndarray: Edge feature vector
        """
        try:
            if len(image.shape) == 3 and image.shape[0] == 3:
                # Convert CHW to HWC
                image = np.transpose(image, (1, 2, 0))
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = rgb2gray(image)
            else:
                gray = image
            
            # Ensure correct data type
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8)
            
            edge_features = []
            
            # Canny edge detection
            try:
                edges = feature.canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
                
                # Edge statistics
                edge_density = np.mean(edges)  # Proportion of edge pixels
                edge_count = np.sum(edges)     # Total edge pixels
                
                edge_features.extend([edge_density, edge_count / (gray.shape[0] * gray.shape[1])])
                
            except Exception as e:
                self.logger.warning(f"Canny edge detection failed: {str(e)}")
                edge_features.extend([0.0, 0.0])
            
            # Sobel edge detection
            try:
                sobel_h = filters.sobel_h(gray)
                sobel_v = filters.sobel_v(gray)
                
                # Edge magnitude statistics
                edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
                mean_magnitude = np.mean(edge_magnitude)
                std_magnitude = np.std(edge_magnitude)
                
                edge_features.extend([mean_magnitude, std_magnitude])
                
            except Exception as e:
                self.logger.warning(f"Sobel edge detection failed: {str(e)}")
                edge_features.extend([0.0, 0.0])
            
            # Pad or truncate to expected size
            expected_size = self.feature_dimensions['edge_features']
            if len(edge_features) < expected_size:
                edge_features.extend([0.0] * (expected_size - len(edge_features)))
            else:
                edge_features = edge_features[:expected_size]
            
            self.extraction_stats['edge_extractions'] += 1
            return np.array(edge_features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Edge feature extraction failed: {str(e)}")
            return np.zeros(self.feature_dimensions['edge_features'], dtype=np.float32)
    
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract shape and contour features
        
        Args:
            image: Image as numpy array
            
        Returns:
            np.ndarray: Shape feature vector
        """
        try:
            if len(image.shape) == 3 and image.shape[0] == 3:
                # Convert CHW to HWC
                image = np.transpose(image, (1, 2, 0))
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = rgb2gray(image)
            else:
                gray = image
            
            # Ensure correct data type
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8)
            
            shape_features = []
            
            try:
                # Threshold image to create binary mask
                threshold = filters.threshold_otsu(gray)
                binary = gray > threshold
                
                # Find contours
                contours = measure.find_contours(binary, 0.5)
                
                if contours:
                    # Use largest contour
                    largest_contour = max(contours, key=len)
                    
                    # Calculate shape properties
                    props = measure.regionprops(binary.astype(int))[0] if measure.regionprops(binary.astype(int)) else None
                    
                    if props:
                        # Shape features
                        area = props.area / (gray.shape[0] * gray.shape[1])  # Normalized area
                        perimeter = props.perimeter / (2 * (gray.shape[0] + gray.shape[1]))  # Normalized perimeter
                        eccentricity = props.eccentricity
                        solidity = props.solidity
                        extent = props.extent
                        
                        # Aspect ratio
                        bbox = props.bbox
                        aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) if (bbox[3] - bbox[1]) > 0 else 1.0
                        
                        # Compactness
                        compactness = (4 * np.pi * props.area) / (props.perimeter ** 2) if props.perimeter > 0 else 0.0
                        
                        shape_features = [area, perimeter, eccentricity, solidity, extent, aspect_ratio, compactness]
                    else:
                        shape_features = [0.0] * 7
                else:
                    shape_features = [0.0] * 7
                    
            except Exception as e:
                self.logger.warning(f"Shape feature extraction failed: {str(e)}")
                shape_features = [0.0] * 7
            
            # Ensure correct size
            expected_size = self.feature_dimensions['shape_features']
            if len(shape_features) < expected_size:
                shape_features.extend([0.0] * (expected_size - len(shape_features)))
            else:
                shape_features = shape_features[:expected_size]
            
            return np.array(shape_features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Shape feature extraction failed: {str(e)}")
            return np.zeros(self.feature_dimensions['shape_features'], dtype=np.float32)
    
    def extract_all_visual_features(self, image_path: str, sample_id: str) -> Dict[str, np.ndarray]:
        """
        Extract all visual features from image
        
        Args:
            image_path: Path to image file
            sample_id: Sample identifier
            
        Returns:
            Dict containing all extracted features
        """
        start_time = time.time()
        features = {}
        
        try:
            # Load and preprocess image
            result = self.image_processor.load_and_preprocess_image(image_path)
            
            if result.success:
                processed_image = result.processed_image
                
                # Convert tensor format to image format for traditional CV
                if len(processed_image.shape) == 3 and processed_image.shape[0] == 3:
                    # Denormalize image for traditional CV features
                    mean = np.array(self.config.normalize_mean).reshape(3, 1, 1)
                    std = np.array(self.config.normalize_std).reshape(3, 1, 1)
                    denormalized = (processed_image * std) + mean
                    denormalized = np.clip(denormalized, 0, 1)
                    cv_image = (denormalized * 255).astype(np.uint8)
                else:
                    cv_image = processed_image
                
                # Extract CNN features
                features['cnn_features'] = self.extract_cnn_features(image_path, sample_id)
                
                # Extract traditional CV features
                if self.config.extract_color_features:
                    features['color_histogram'] = self.extract_color_histogram(cv_image)
                    features['dominant_colors'] = self.extract_dominant_colors(cv_image)
                
                if self.config.extract_texture_features:
                    features['texture_features'] = self.extract_texture_features(cv_image)
                    features['edge_features'] = self.extract_edge_features(cv_image)
                    features['shape_features'] = self.extract_shape_features(cv_image)
                
            else:
                # Use fallback for failed image processing
                self.logger.warning(f"Image processing failed for {sample_id}: {result.error_message}")
                
                # Create fallback features
                features['cnn_features'] = np.zeros(self.feature_dimensions['cnn_features'], dtype=np.float32)
                
                if self.config.extract_color_features:
                    features['color_histogram'] = np.zeros(self.feature_dimensions['color_histogram'], dtype=np.float32)
                    features['dominant_colors'] = np.zeros(self.feature_dimensions['dominant_colors'], dtype=np.float32)
                
                if self.config.extract_texture_features:
                    features['texture_features'] = np.zeros(self.feature_dimensions['texture_features'], dtype=np.float32)
                    features['edge_features'] = np.zeros(self.feature_dimensions['edge_features'], dtype=np.float32)
                    features['shape_features'] = np.zeros(self.feature_dimensions['shape_features'], dtype=np.float32)
                
                self.extraction_stats['failed_extractions'] += 1
            
            processing_time = time.time() - start_time
            self.extraction_stats['total_extractions'] += 1
            self.extraction_stats['total_extraction_time'] += processing_time
            
            if self.extraction_stats['total_extractions'] > 0:
                self.extraction_stats['avg_extraction_time'] = (
                    self.extraction_stats['total_extraction_time'] / 
                    self.extraction_stats['total_extractions']
                )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Visual feature extraction failed for {sample_id}: {str(e)}")
            self.extraction_stats['failed_extractions'] += 1
            
            # Return zero features as fallback
            fallback_features = {}
            for feature_name, dimension in self.feature_dimensions.items():
                fallback_features[feature_name] = np.zeros(dimension, dtype=np.float32)
            
            return fallback_features
    
    def extract_features_batch(self, image_paths: List[str], sample_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract visual features for batch of images
        
        Args:
            image_paths: List of image file paths
            sample_ids: List of sample identifiers
            
        Returns:
            Dict mapping sample_id to feature dictionary
        """
        if len(image_paths) != len(sample_ids):
            raise ValueError("image_paths and sample_ids must have same length")
        
        self.logger.info(f"Extracting visual features for {len(sample_ids)} samples")
        
        batch_features = {}
        
        for image_path, sample_id in zip(image_paths, sample_ids):
            batch_features[sample_id] = self.extract_all_visual_features(image_path, sample_id)
        
        self.logger.info(f"Batch visual feature extraction completed")
        
        return batch_features
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of all feature types"""
        return self.feature_dimensions.copy()
    
    def get_total_feature_dimension(self) -> int:
        """Get total dimension of all features combined"""
        total_dim = 0
        
        # CNN features
        total_dim += self.feature_dimensions['cnn_features']
        
        # Color features
        if self.config.extract_color_features:
            total_dim += self.feature_dimensions['color_histogram']
            total_dim += self.feature_dimensions['dominant_colors']
        
        # Texture features
        if self.config.extract_texture_features:
            total_dim += self.feature_dimensions['texture_features']
            total_dim += self.feature_dimensions['edge_features']
            total_dim += self.feature_dimensions['shape_features']
        
        return total_dim
    
    def get_licensed_models(self) -> Dict[str, Dict[str, str]]:
        """Get information about licensed models"""
        return self.LICENSED_MODELS.copy()
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        stats = self.extraction_stats.copy()
        
        # Add feature dimension info
        stats['feature_dimensions'] = self.get_feature_dimensions()
        stats['total_feature_dimension'] = self.get_total_feature_dimension()
        
        # Add configuration info
        stats['cnn_model'] = self.config.cnn_model
        stats['extract_color_features'] = self.config.extract_color_features
        stats['extract_texture_features'] = self.config.extract_texture_features
        stats['color_histogram_bins'] = self.config.color_histogram_bins
        
        # Calculate rates
        if stats['total_extractions'] > 0:
            stats['success_rate'] = ((stats['total_extractions'] - stats['failed_extractions']) / 
                                   stats['total_extractions']) * 100
            stats['failure_rate'] = (stats['failed_extractions'] / stats['total_extractions']) * 100
        
        return stats
    
    def reset_statistics(self):
        """Reset extraction statistics"""
        for key in self.extraction_stats:
            if isinstance(self.extraction_stats[key], (int, float)):
                self.extraction_stats[key] = 0
        self.logger.info("Visual feature extraction statistics reset")