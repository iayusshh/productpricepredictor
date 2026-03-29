"""
Configuration management for ML Product Pricing Challenge 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data processing"""
    train_file: str = "dataset/train.csv"
    test_file: str = "dataset/test.csv"
    sample_test_out_file: str = "dataset/sample_test_out.csv"
    image_dir: str = "images/"
    cache_dir: str = "cache/"
    
    # Data validation settings
    fail_fast_validation: bool = True
    zero_price_strategy: str = "drop"  # Options: "drop", "epsilon", "special_class"
    zero_price_epsilon: float = 0.01
    
    # Image download settings
    max_download_retries: int = 3
    download_timeout: int = 30
    batch_size: int = 100


@dataclass
class TextFeatureConfig:
    """Configuration for text feature engineering"""
    # Embedding models
    embedding_model: str = "bert-base-uncased"
    embedding_dim: int = 768
    max_sequence_length: int = 512
    
    # IPQ extraction settings
    ipq_precision_threshold: float = 0.90
    canonical_units: List[str] = field(default_factory=lambda: ["grams", "pieces", "ml"])
    
    # Text processing settings
    clean_special_chars: bool = True
    normalize_whitespace: bool = True
    extract_brands: bool = True
    extract_categories: bool = True
    
    # Statistical features
    include_length_features: bool = True
    include_readability_scores: bool = True
    include_word_count: bool = True


@dataclass
class ImageFeatureConfig:
    """Configuration for image feature engineering"""
    # CNN models
    cnn_model: str = "resnet50"  # Options: "resnet50", "efficientnet-b0", "vit-base"
    feature_dim: int = 2048
    
    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Augmentation settings
    use_augmentation: bool = True
    rotation_range: int = 15
    brightness_range: float = 0.2
    
    # Feature extraction settings
    extract_color_features: bool = True
    extract_texture_features: bool = True
    color_histogram_bins: int = 64
    
    # Missing image handling
    missing_image_strategy: str = "text_based"  # Options: "zero", "mean", "text_based"


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion"""
    fusion_method: str = "concatenation"  # Options: "concatenation", "weighted", "attention"
    text_weight: float = 0.6
    image_weight: float = 0.4
    
    # Dimensionality reduction
    use_dimensionality_reduction: bool = True
    target_dimensions: int = 1000
    reduction_method: str = "pca"  # Options: "pca", "feature_selection"
    variance_threshold: float = 0.95


@dataclass
class ModelConfig:
    """Configuration for model training"""
    # Model types to train (7 models for competition requirement of 5+)
    model_types: List[str] = field(default_factory=lambda: [
        "random_forest",
        "xgboost",
        "lightgbm",
        "extra_trees",
        "gradient_boosting",
        "ridge_regression",
        "neural_network",
    ])
    
    # Random Forest settings
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    
    # XGBoost settings
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    
    # LightGBM settings
    lgb_n_estimators: int = 100
    lgb_max_depth: int = -1
    lgb_learning_rate: float = 0.1
    lgb_num_leaves: int = 31
    
    # Extra Trees settings
    et_n_estimators: int = 200
    et_max_depth: Optional[int] = None

    # Gradient Boosting (sklearn) settings
    gbr_n_estimators: int = 150
    gbr_max_depth: int = 5
    gbr_learning_rate: float = 0.1
    gbr_subsample: float = 0.8

    # Ridge Regression settings
    ridge_alpha: float = 10.0

    # Neural network settings
    nn_hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    nn_dropout_rate: float = 0.3
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 100
    
    # Training settings
    random_seed: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Hyperparameter tuning
    use_hyperparameter_tuning: bool = True
    tuning_method: str = "grid_search"  # Options: "grid_search", "random_search", "bayesian"
    tuning_cv_folds: int = 3
    
    # Ensemble settings
    use_ensemble: bool = True
    ensemble_method: str = "weighted_average"  # Options: "voting", "weighted_average", "stacking"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    # SMAPE calculation settings
    smape_epsilon: float = 1e-8
    
    # Validation settings
    validation_split: float = 0.2
    stratify_by_price_quantiles: bool = True
    price_quantiles: int = 5
    
    # Reporting settings
    generate_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # Feature importance settings
    calculate_shap: bool = True
    shap_sample_size: int = 1000


@dataclass
class PredictionConfig:
    """Configuration for prediction generation"""
    # Prediction settings
    min_price_threshold: float = 0.01
    max_price_threshold: Optional[float] = None
    
    # Output formatting
    output_file: str = "test_out.csv"
    output_precision: int = 6
    
    # Validation settings
    validate_sample_ids: bool = True
    validate_row_count: bool = True
    validate_price_range: bool = True


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure and logging"""
    # Logging settings
    log_dir: str = "logs/"
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Resource management
    gpu_memory_limit: str = "16GB"
    max_cpu_cores: Optional[int] = None
    
    # Storage settings
    cache_embeddings: bool = True
    cache_images: bool = True
    validate_checksums: bool = True
    
    # Performance settings
    inference_timeout: int = 3600  # 1 hour for 75k samples
    batch_processing: bool = True
    batch_size: int = 1000


@dataclass
class ComplianceConfig:
    """Configuration for compliance and deliverables"""
    # License tracking
    allowed_licenses: List[str] = field(default_factory=lambda: ["MIT", "Apache-2.0", "BSD-3-Clause"])
    track_all_dependencies: bool = True
    
    # Deliverable settings
    methodology_file: str = "methodology_1page.pdf"
    readme_file: str = "README.md"
    requirements_file: str = "requirements.txt"
    environment_file: str = "environment.yml"
    
    # Reproduction settings
    create_run_script: bool = True
    run_script_name: str = "run_all.sh"
    include_test_coverage: bool = True
    min_test_coverage: float = 0.80


@dataclass
class MLPricingConfig:
    """Main configuration class combining all component configurations"""
    data: DataConfig = field(default_factory=DataConfig)
    text_features: TextFeatureConfig = field(default_factory=TextFeatureConfig)
    image_features: ImageFeatureConfig = field(default_factory=ImageFeatureConfig)
    feature_fusion: FeatureFusionConfig = field(default_factory=FeatureFusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    
    # Project settings
    project_name: str = "ml-product-pricing"
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Create necessary directories"""
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data.image_dir,
            self.data.cache_dir,
            self.infrastructure.log_dir,
            "models/",
            "embeddings/",
            "notebooks/",
            "tests/",
            "deliverables/"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLPricingConfig':
        """Create configuration from dictionary"""
        # This would implement loading from a configuration file
        # For now, return default configuration
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'text_features': self.text_features.__dict__,
            'image_features': self.image_features.__dict__,
            'feature_fusion': self.feature_fusion.__dict__,
            'model': self.model.__dict__,
            'evaluation': self.evaluation.__dict__,
            'prediction': self.prediction.__dict__,
            'infrastructure': self.infrastructure.__dict__,
            'compliance': self.compliance.__dict__,
            'project_name': self.project_name,
            'version': self.version
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MLPricingConfig':
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Global configuration instance
config = MLPricingConfig()