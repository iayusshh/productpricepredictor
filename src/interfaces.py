"""
Base interfaces and abstract classes for ML Product Pricing Challenge 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


class DataPreprocessorInterface(ABC):
    """Interface for data preprocessing components"""
    
    @abstractmethod
    def load_training_data(self) -> pd.DataFrame:
        """Load training data from dataset/train.csv"""
        pass
    
    @abstractmethod
    def load_test_data(self) -> pd.DataFrame:
        """Load test data from dataset/test.csv"""
        pass
    
    @abstractmethod
    def validate_schema_and_types(self, df: pd.DataFrame) -> bool:
        """Validate schema and column types with fail-fast error handling"""
        pass
    
    @abstractmethod
    def normalize_price_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize price formatting by stripping currency symbols and separators"""
        pass
    
    @abstractmethod
    def handle_zero_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle zero prices with documented strategy"""
        pass
    
    @abstractmethod
    def clean_catalog_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize catalog content"""
        pass
    
    @abstractmethod
    def download_images(self, df: pd.DataFrame, image_dir: str) -> Dict[str, str]:
        """Download images with retry logic and caching"""
        pass
    
    @abstractmethod
    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """Validate data integrity and completeness"""
        pass


class TextFeatureEngineerInterface(ABC):
    """Interface for text feature engineering components"""
    
    @abstractmethod
    def extract_product_attributes(self, catalog_content: str) -> Dict[str, Any]:
        """Extract product attributes from catalog content"""
        pass
    
    @abstractmethod
    def generate_text_embeddings(self, text: str) -> np.ndarray:
        """Generate text embeddings using pre-trained models"""
        pass
    
    @abstractmethod
    def extract_ipq_with_validation(self, catalog_content: str) -> Dict[str, Any]:
        """Extract Item Pack Quantity with >90% precision validation"""
        pass
    
    @abstractmethod
    def normalize_units_to_canonical(self, text: str) -> Dict[str, float]:
        """Normalize units to canonical format (grams/pieces)"""
        pass
    
    @abstractmethod
    def extract_quantity_features(self, catalog_content: str) -> Dict[str, float]:
        """Extract numerical quantity features"""
        pass
    
    @abstractmethod
    def create_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create comprehensive text feature matrix"""
        pass
    
    @abstractmethod
    def validate_ipq_extraction_precision(self, test_samples: List[str]) -> float:
        """Validate IPQ extraction precision on test samples"""
        pass


class ImageFeatureEngineerInterface(ABC):
    """Interface for image feature engineering components"""
    
    @abstractmethod
    def download_with_retry_and_cache(self, image_url: str, max_retries: int = 3) -> str:
        """Download image with retry logic and caching"""
        pass
    
    @abstractmethod
    def create_download_manifest(self, download_results: Dict) -> None:
        """Create manifest file for download tracking"""
        pass
    
    @abstractmethod
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for feature extraction"""
        pass
    
    @abstractmethod
    def extract_visual_features(self, image: np.ndarray) -> np.ndarray:
        """Extract visual features using CNN models"""
        pass
    
    @abstractmethod
    def save_versioned_embeddings(self, embeddings: np.ndarray, metadata: Dict) -> str:
        """Save embeddings with version and metadata"""
        pass
    
    @abstractmethod
    def handle_missing_images(self, sample_id: str) -> np.ndarray:
        """Handle missing images with fallback features"""
        pass
    
    @abstractmethod
    def create_image_features(self, image_paths: List[str]) -> np.ndarray:
        """Create comprehensive image feature matrix"""
        pass


class FeatureFusionInterface(ABC):
    """Interface for feature fusion components"""
    
    @abstractmethod
    def concatenate_features(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """Concatenate text and image features"""
        pass
    
    @abstractmethod
    def weighted_fusion(self, text_features: np.ndarray, image_features: np.ndarray, 
                       weights: Tuple[float, float]) -> np.ndarray:
        """Combine features using weighted averaging"""
        pass
    
    @abstractmethod
    def attention_fusion(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """Combine features using attention mechanism"""
        pass
    
    @abstractmethod
    def reduce_dimensions(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce feature dimensions while preserving information"""
        pass


class ModelTrainerInterface(ABC):
    """Interface for model training components"""
    
    @abstractmethod
    def set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility"""
        pass
    
    @abstractmethod
    def capture_experiment_metadata(self, config: Dict, cv_folds: int, seed: int) -> str:
        """Capture experiment metadata for tracking"""
        pass
    
    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> Any:
        """Train model with given configuration"""
        pass
    
    @abstractmethod
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune hyperparameters to minimize SMAPE"""
        pass
    
    @abstractmethod
    def validate_model_with_detailed_metrics(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Validate model with comprehensive metrics"""
        pass
    
    @abstractmethod
    def report_cv_results_with_statistics(self, cv_scores: List[float]) -> Dict:
        """Report cross-validation results with statistics"""
        pass
    
    @abstractmethod
    def calculate_per_quantile_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate SMAPE per price quantile"""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, model_path: str) -> None:
        """Save trained model to disk"""
        pass


class EvaluationInterface(ABC):
    """Interface for model evaluation components"""
    
    @abstractmethod
    def calculate_smape_with_validation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate SMAPE with validation against known examples"""
        pass
    
    @abstractmethod
    def test_smape_on_known_examples(self) -> bool:
        """Test SMAPE calculation on known examples"""
        pass
    
    @abstractmethod
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Generate comprehensive evaluation report"""
        pass
    
    @abstractmethod
    def create_distribution_plots(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Create distribution plots for predicted vs actual"""
        pass
    
    @abstractmethod
    def generate_residual_histograms(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Generate residual histograms for error analysis"""
        pass
    
    @abstractmethod
    def calculate_shap_feature_importance(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP feature importance"""
        pass


class PredictionGeneratorInterface(ABC):
    """Interface for prediction generation components"""
    
    @abstractmethod
    def predict(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions for test data"""
        pass
    
    @abstractmethod
    def clamp_predictions_to_threshold(self, predictions: np.ndarray, min_threshold: float = 0.01) -> np.ndarray:
        """Clamp predictions to minimum threshold"""
        pass
    
    @abstractmethod
    def ensemble_predict(self, models: List[Any], X_test: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions"""
        pass
    
    @abstractmethod
    def format_output(self, sample_ids: List[str], predictions: np.ndarray) -> pd.DataFrame:
        """Format predictions according to submission requirements"""
        pass
    
    @abstractmethod
    def validate_exact_sample_id_match(self, output_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate exact sample_id matching"""
        pass
    
    @abstractmethod
    def validate_row_count_match(self, output_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate row count matching"""
        pass
    
    @abstractmethod
    def validate_output(self, output_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Comprehensive output validation"""
        pass


class InfrastructureInterface(ABC):
    """Interface for infrastructure and logging components"""
    
    @abstractmethod
    def setup_structured_logging(self, log_dir: str) -> None:
        """Setup structured JSON logging"""
        pass
    
    @abstractmethod
    def log_experiment_metrics(self, metrics: Dict, timestamp: str) -> None:
        """Log experiment metrics with timestamp"""
        pass
    
    @abstractmethod
    def manage_gpu_resources(self, required_memory: str) -> bool:
        """Manage GPU resources and memory"""
        pass
    
    @abstractmethod
    def calculate_storage_requirements(self) -> Dict[str, str]:
        """Calculate total storage requirements"""
        pass
    
    @abstractmethod
    def setup_embedding_cache_with_checksums(self, cache_dir: str) -> None:
        """Setup embedding cache with checksum validation"""
        pass
    
    @abstractmethod
    def validate_inference_timing(self, start_time: float, sample_count: int) -> bool:
        """Validate inference timing constraints"""
        pass


class ComplianceInterface(ABC):
    """Interface for compliance and deliverable management"""
    
    @abstractmethod
    def track_dependency_licenses(self) -> Dict[str, str]:
        """Track and validate dependency licenses"""
        pass
    
    @abstractmethod
    def validate_model_licenses(self, model_checkpoints: List[str]) -> bool:
        """Validate model checkpoint licenses"""
        pass
    
    @abstractmethod
    def create_deliverable_structure(self) -> None:
        """Create required deliverable structure"""
        pass
    
    @abstractmethod
    def validate_submission_completeness(self) -> Dict[str, bool]:
        """Validate submission completeness"""
        pass
    
    @abstractmethod
    def generate_compliance_log(self) -> str:
        """Generate compliance log with license information"""
        pass
    
    @abstractmethod
    def create_reproduction_package(self) -> None:
        """Create complete reproduction package"""
        pass