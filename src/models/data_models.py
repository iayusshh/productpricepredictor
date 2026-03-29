"""
Core data models for ML Product Pricing Challenge 2025
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ProductSample:
    """Represents a single product sample from the dataset"""
    sample_id: str
    catalog_content: str
    image_link: str
    price: Optional[float] = None
    
    def __post_init__(self):
        """Validate required fields"""
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not self.catalog_content:
            raise ValueError("catalog_content cannot be empty")
        if not self.image_link:
            raise ValueError("image_link cannot be empty")


@dataclass
class ProcessedFeatures:
    """Represents processed features for a product sample"""
    sample_id: str
    text_features: np.ndarray
    image_features: np.ndarray
    combined_features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate feature arrays"""
        if self.text_features.size == 0:
            raise ValueError("text_features cannot be empty")
        if self.image_features.size == 0:
            raise ValueError("image_features cannot be empty")


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_config: Dict[str, Any]
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate model configuration"""
        if not self.model_type:
            raise ValueError("model_type cannot be empty")
        if not isinstance(self.hyperparameters, dict):
            raise ValueError("hyperparameters must be a dictionary")


@dataclass
class PredictionResult:
    """Represents a prediction result for a product sample"""
    sample_id: str
    predicted_price: float
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate prediction result"""
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if self.predicted_price < 0:
            raise ValueError("predicted_price cannot be negative")


@dataclass
class ExperimentMetadata:
    """Metadata for tracking experiments"""
    experiment_id: str
    timestamp: str
    model_config: ModelConfig
    cv_folds: int
    random_seed: int
    feature_dimensions: Dict[str, int]
    performance_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate experiment metadata"""
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")


@dataclass
class ValidationResult:
    """Results from model validation"""
    smape_mean: float
    smape_std: float
    smape_per_fold: list
    smape_per_quantile: Dict[str, float]
    additional_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate validation results"""
        if self.smape_mean < 0:
            raise ValueError("smape_mean cannot be negative")
        if len(self.smape_per_fold) == 0:
            raise ValueError("smape_per_fold cannot be empty")


@dataclass
class ImageProcessingResult:
    """Result from image processing operations"""
    sample_id: str
    image_path: Optional[str]
    features: np.ndarray
    processing_status: str  # "success", "missing", "corrupted", "failed"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate image processing result"""
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if self.processing_status not in ["success", "missing", "corrupted", "failed"]:
            raise ValueError("Invalid processing_status")


@dataclass
class TextProcessingResult:
    """Result from text processing operations"""
    sample_id: str
    cleaned_text: str
    extracted_attributes: Dict[str, Any]
    features: np.ndarray
    ipq_extraction: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate text processing result"""
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not self.cleaned_text:
            raise ValueError("cleaned_text cannot be empty")


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    model_name: str
    validation_results: ValidationResult
    feature_importance: Optional[np.ndarray] = None
    prediction_distribution: Optional[Dict[str, Any]] = None
    residual_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate evaluation report"""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")


@dataclass
class ComplianceReport:
    """Report on compliance and license tracking"""
    dependency_licenses: Dict[str, str]
    model_licenses: Dict[str, str]
    compliance_status: bool
    issues: list
    timestamp: str
    
    def __post_init__(self):
        """Validate compliance report"""
        if not isinstance(self.dependency_licenses, dict):
            raise ValueError("dependency_licenses must be a dictionary")
        if not isinstance(self.model_licenses, dict):
            raise ValueError("model_licenses must be a dictionary")