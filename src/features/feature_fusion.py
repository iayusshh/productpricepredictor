"""
Feature Fusion for Multimodal Feature Combination

This module implements comprehensive feature fusion strategies for combining text and image
features in the ML Product Pricing Challenge. It provides concatenation, weighted averaging,
and attention-based fusion methods with feature normalization and dimensionality management.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import warnings

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.feature_selection import VarianceThreshold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Feature fusion will be limited.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Attention-based fusion will be limited.")

try:
    from ..interfaces import FeatureFusionInterface
    from ..config import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from interfaces import FeatureFusionInterface
    from config import config


@dataclass
class FusionResult:
    """Container for fusion results with metadata"""
    fused_features: np.ndarray
    fusion_method: str
    original_dimensions: Tuple[int, int]  # (text_dim, image_dim)
    final_dimension: int
    normalization_method: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DimensionalityReductionResult:
    """Container for dimensionality reduction results"""
    reduced_features: np.ndarray
    original_dimension: int
    target_dimension: int
    method: str
    explained_variance_ratio: Optional[float] = None
    selected_features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class AttentionFusion(nn.Module):
    """
    Neural attention mechanism for feature fusion.
    
    Implements learnable attention weights to combine text and image features
    based on their relevance for the prediction task.
    """
    
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int = 128):
        super(AttentionFusion, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers to common dimension
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        
        # Attention mechanism
        self.attention_text = nn.Linear(hidden_dim, 1)
        self.attention_image = nn.Linear(hidden_dim, 1)
        
        # Final fusion layer
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention-based fusion.
        
        Args:
            text_features: Text features tensor [batch_size, text_dim]
            image_features: Image features tensor [batch_size, image_dim]
            
        Returns:
            Fused features tensor [batch_size, hidden_dim]
        """
        # Project to common dimension
        text_proj = torch.relu(self.text_projection(text_features))
        image_proj = torch.relu(self.image_projection(image_features))
        
        # Calculate attention weights
        text_attention = torch.sigmoid(self.attention_text(text_proj))
        image_attention = torch.sigmoid(self.attention_image(image_proj))
        
        # Normalize attention weights
        total_attention = text_attention + image_attention + 1e-8
        text_weight = text_attention / total_attention
        image_weight = image_attention / total_attention
        
        # Apply attention weights
        weighted_text = text_weight * text_proj
        weighted_image = image_weight * image_proj
        
        # Concatenate and fuse
        concatenated = torch.cat([weighted_text, weighted_image], dim=1)
        fused = torch.relu(self.fusion_layer(concatenated))
        
        return self.dropout(fused)


class FeatureFusion(FeatureFusionInterface):
    """
    Comprehensive feature fusion system for multimodal feature combination.
    
    Implements concatenation, weighted averaging, and attention-based fusion methods
    with feature normalization and scaling for different modalities. Handles mismatched
    feature dimensions and provides dimensionality reduction capabilities.
    """
    
    def __init__(self, 
                 normalization_method: str = 'standard',
                 handle_missing: str = 'zero',
                 attention_hidden_dim: int = 128,
                 random_state: int = 42):
        """
        Initialize FeatureFusion with configuration options.
        
        Args:
            normalization_method: Method for feature normalization ('standard', 'minmax', 'robust', 'none')
            handle_missing: How to handle missing features ('zero', 'mean', 'drop')
            attention_hidden_dim: Hidden dimension for attention mechanism
            random_state: Random state for reproducibility
        """
        self.normalization_method = normalization_method
        self.handle_missing = handle_missing
        self.attention_hidden_dim = attention_hidden_dim
        self.random_state = random_state
        
        # Initialize scalers
        self.text_scaler = None
        self.image_scaler = None
        self.combined_scaler = None
        
        # Initialize attention model
        self.attention_model = None
        self.attention_trained = False
        
        # Metadata tracking
        self.fusion_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def _validate_inputs(self, text_features: np.ndarray, image_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and preprocess input features.
        
        Args:
            text_features: Text features array [n_samples, text_dim]
            image_features: Image features array [n_samples, image_dim]
            
        Returns:
            Tuple of validated and preprocessed features
            
        Raises:
            ValueError: If inputs are invalid
        """
        if text_features is None or image_features is None:
            raise ValueError("Both text_features and image_features must be provided")
            
        if len(text_features.shape) != 2 or len(image_features.shape) != 2:
            raise ValueError("Features must be 2D arrays [n_samples, n_features]")
            
        if text_features.shape[0] != image_features.shape[0]:
            raise ValueError(f"Sample count mismatch: text={text_features.shape[0]}, image={image_features.shape[0]}")
            
        # Handle missing values
        if self.handle_missing == 'zero':
            text_features = np.nan_to_num(text_features, nan=0.0, posinf=0.0, neginf=0.0)
            image_features = np.nan_to_num(image_features, nan=0.0, posinf=0.0, neginf=0.0)
        elif self.handle_missing == 'mean':
            text_mean = np.nanmean(text_features, axis=0)
            image_mean = np.nanmean(image_features, axis=0)
            
            text_features = np.where(np.isnan(text_features), text_mean, text_features)
            image_features = np.where(np.isnan(image_features), image_mean, image_features)
        elif self.handle_missing == 'drop':
            # Find rows with any missing values
            text_missing = np.isnan(text_features).any(axis=1)
            image_missing = np.isnan(image_features).any(axis=1)
            missing_mask = text_missing | image_missing
            
            if missing_mask.sum() > 0:
                self.logger.warning(f"Dropping {missing_mask.sum()} samples with missing values")
                text_features = text_features[~missing_mask]
                image_features = image_features[~missing_mask]
        
        return text_features, image_features
    
    def _normalize_features(self, text_features: np.ndarray, image_features: np.ndarray, 
                          fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features using specified normalization method.
        
        Args:
            text_features: Text features to normalize
            image_features: Image features to normalize
            fit_scalers: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Tuple of normalized features
        """
        if self.normalization_method == 'none':
            return text_features, image_features
            
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. Skipping normalization.")
            return text_features, image_features
            
        # Initialize scalers if needed
        if fit_scalers or self.text_scaler is None:
            if self.normalization_method == 'standard':
                self.text_scaler = StandardScaler()
                self.image_scaler = StandardScaler()
            elif self.normalization_method == 'minmax':
                self.text_scaler = MinMaxScaler()
                self.image_scaler = MinMaxScaler()
            elif self.normalization_method == 'robust':
                self.text_scaler = RobustScaler()
                self.image_scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        # Fit and transform or just transform
        if fit_scalers:
            text_normalized = self.text_scaler.fit_transform(text_features)
            image_normalized = self.image_scaler.fit_transform(image_features)
        else:
            text_normalized = self.text_scaler.transform(text_features)
            image_normalized = self.image_scaler.transform(image_features)
            
        return text_normalized, image_normalized
    
    def concatenate_features(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """
        Concatenate text and image features with normalization.
        
        Args:
            text_features: Text features array [n_samples, text_dim]
            image_features: Image features array [n_samples, image_dim]
            
        Returns:
            Concatenated features array [n_samples, text_dim + image_dim]
        """
        # Validate inputs
        text_features, image_features = self._validate_inputs(text_features, image_features)
        
        # Normalize features
        text_normalized, image_normalized = self._normalize_features(text_features, image_features)
        
        # Concatenate
        concatenated = np.concatenate([text_normalized, image_normalized], axis=1)
        
        # Track fusion metadata
        fusion_metadata = {
            'method': 'concatenation',
            'text_dim': text_features.shape[1],
            'image_dim': image_features.shape[1],
            'output_dim': concatenated.shape[1],
            'normalization': self.normalization_method,
            'n_samples': concatenated.shape[0]
        }
        self.fusion_history.append(fusion_metadata)
        
        self.logger.info(f"Concatenated features: {text_features.shape[1]} + {image_features.shape[1]} = {concatenated.shape[1]} dimensions")
        
        return concatenated
    
    def weighted_fusion(self, text_features: np.ndarray, image_features: np.ndarray, 
                       weights: Tuple[float, float]) -> np.ndarray:
        """
        Combine features using weighted averaging after dimension alignment.
        
        Args:
            text_features: Text features array [n_samples, text_dim]
            image_features: Image features array [n_samples, image_dim]
            weights: Tuple of (text_weight, image_weight)
            
        Returns:
            Weighted combined features array [n_samples, max(text_dim, image_dim)]
        """
        # Validate inputs
        text_features, image_features = self._validate_inputs(text_features, image_features)
        
        # Validate weights
        text_weight, image_weight = weights
        if text_weight < 0 or image_weight < 0:
            raise ValueError("Weights must be non-negative")
        
        # Normalize weights
        total_weight = text_weight + image_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be positive")
        
        text_weight = text_weight / total_weight
        image_weight = image_weight / total_weight
        
        # Normalize features
        text_normalized, image_normalized = self._normalize_features(text_features, image_features)
        
        # Handle dimension mismatch by padding or truncating
        text_dim = text_normalized.shape[1]
        image_dim = image_normalized.shape[1]
        target_dim = max(text_dim, image_dim)
        
        # Pad smaller dimension with zeros
        if text_dim < target_dim:
            padding = np.zeros((text_normalized.shape[0], target_dim - text_dim))
            text_aligned = np.concatenate([text_normalized, padding], axis=1)
        else:
            text_aligned = text_normalized[:, :target_dim]
            
        if image_dim < target_dim:
            padding = np.zeros((image_normalized.shape[0], target_dim - image_dim))
            image_aligned = np.concatenate([image_normalized, padding], axis=1)
        else:
            image_aligned = image_normalized[:, :target_dim]
        
        # Weighted combination
        weighted_features = text_weight * text_aligned + image_weight * image_aligned
        
        # Track fusion metadata
        fusion_metadata = {
            'method': 'weighted_fusion',
            'text_dim': text_dim,
            'image_dim': image_dim,
            'output_dim': target_dim,
            'text_weight': text_weight,
            'image_weight': image_weight,
            'normalization': self.normalization_method,
            'n_samples': weighted_features.shape[0]
        }
        self.fusion_history.append(fusion_metadata)
        
        self.logger.info(f"Weighted fusion: text({text_dim}) * {text_weight:.3f} + image({image_dim}) * {image_weight:.3f} = {target_dim} dimensions")
        
        return weighted_features
    
    def attention_fusion(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """
        Combine features using learnable attention mechanism.
        
        Args:
            text_features: Text features array [n_samples, text_dim]
            image_features: Image features array [n_samples, image_dim]
            
        Returns:
            Attention-fused features array [n_samples, hidden_dim]
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available. Falling back to weighted fusion with equal weights.")
            return self.weighted_fusion(text_features, image_features, (0.5, 0.5))
        
        # Validate inputs
        text_features, image_features = self._validate_inputs(text_features, image_features)
        
        # Normalize features
        text_normalized, image_normalized = self._normalize_features(text_features, image_features)
        
        # Initialize attention model if needed
        if self.attention_model is None:
            text_dim = text_normalized.shape[1]
            image_dim = image_normalized.shape[1]
            self.attention_model = AttentionFusion(text_dim, image_dim, self.attention_hidden_dim)
        
        # Convert to tensors
        text_tensor = torch.FloatTensor(text_normalized)
        image_tensor = torch.FloatTensor(image_normalized)
        
        # Forward pass
        self.attention_model.eval()
        with torch.no_grad():
            fused_tensor = self.attention_model(text_tensor, image_tensor)
            fused_features = fused_tensor.numpy()
        
        # Track fusion metadata
        fusion_metadata = {
            'method': 'attention_fusion',
            'text_dim': text_normalized.shape[1],
            'image_dim': image_normalized.shape[1],
            'output_dim': fused_features.shape[1],
            'hidden_dim': self.attention_hidden_dim,
            'normalization': self.normalization_method,
            'n_samples': fused_features.shape[0]
        }
        self.fusion_history.append(fusion_metadata)
        
        self.logger.info(f"Attention fusion: text({text_normalized.shape[1]}) + image({image_normalized.shape[1]}) -> {fused_features.shape[1]} dimensions")
        
        return fused_features
    
    def train_attention_fusion(self, text_features: np.ndarray, image_features: np.ndarray, 
                             target_values: np.ndarray, epochs: int = 100, 
                             learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train the attention fusion model on target values.
        
        Args:
            text_features: Text features for training
            image_features: Image features for training
            target_values: Target values for supervised training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            Training history and metrics
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Cannot train attention fusion.")
        
        # Validate inputs
        text_features, image_features = self._validate_inputs(text_features, image_features)
        
        if len(target_values) != text_features.shape[0]:
            raise ValueError("Target values length must match number of samples")
        
        # Normalize features
        text_normalized, image_normalized = self._normalize_features(text_features, image_features)
        
        # Initialize model
        text_dim = text_normalized.shape[1]
        image_dim = image_normalized.shape[1]
        self.attention_model = AttentionFusion(text_dim, image_dim, self.attention_hidden_dim)
        
        # Add prediction head for training
        prediction_head = nn.Linear(self.attention_hidden_dim, 1)
        
        # Convert to tensors
        text_tensor = torch.FloatTensor(text_normalized)
        image_tensor = torch.FloatTensor(image_normalized)
        target_tensor = torch.FloatTensor(target_values.reshape(-1, 1))
        
        # Setup training
        optimizer = torch.optim.Adam(
            list(self.attention_model.parameters()) + list(prediction_head.parameters()),
            lr=learning_rate
        )
        criterion = nn.MSELoss()
        
        # Training loop
        training_history = []
        self.attention_model.train()
        prediction_head.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            fused_features = self.attention_model(text_tensor, image_tensor)
            predictions = prediction_head(fused_features)
            
            # Calculate loss
            loss = criterion(predictions, target_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track progress
            if epoch % 10 == 0:
                training_history.append({
                    'epoch': epoch,
                    'loss': loss.item()
                })
                self.logger.debug(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        self.attention_trained = True
        
        # Final evaluation
        self.attention_model.eval()
        prediction_head.eval()
        with torch.no_grad():
            final_fused = self.attention_model(text_tensor, image_tensor)
            final_predictions = prediction_head(final_fused)
            final_loss = criterion(final_predictions, target_tensor).item()
        
        training_result = {
            'final_loss': final_loss,
            'training_history': training_history,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'model_parameters': sum(p.numel() for p in self.attention_model.parameters())
        }
        
        self.logger.info(f"Attention fusion training completed. Final loss: {final_loss:.6f}")
        
        return training_result
    
    def reduce_dimensions(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Reduce feature dimensions while preserving information using PCA.
        
        Args:
            features: Input features array [n_samples, n_features]
            target_dim: Target number of dimensions
            
        Returns:
            Reduced features array [n_samples, target_dim]
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. Returning truncated features.")
            return features[:, :target_dim] if features.shape[1] > target_dim else features
        
        if features.shape[1] <= target_dim:
            self.logger.info(f"Features already have {features.shape[1]} dimensions, no reduction needed")
            return features
        
        # Apply PCA
        pca = PCA(n_components=target_dim, random_state=self.random_state)
        reduced_features = pca.fit_transform(features)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        
        self.logger.info(f"PCA reduction: {features.shape[1]} -> {target_dim} dimensions, "
                        f"explained variance: {explained_variance:.3f}")
        
        return reduced_features
    
    def get_fusion_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all fusion operations performed.
        
        Returns:
            List of fusion metadata dictionaries
        """
        return self.fusion_history.copy()
    
    def save_fusion_config(self, filepath: str) -> None:
        """
        Save fusion configuration and history to file.
        
        Args:
            filepath: Path to save configuration
        """
        config_data = {
            'normalization_method': self.normalization_method,
            'handle_missing': self.handle_missing,
            'attention_hidden_dim': self.attention_hidden_dim,
            'random_state': self.random_state,
            'attention_trained': self.attention_trained,
            'fusion_history': self.fusion_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Fusion configuration saved to {filepath}")
    
    def load_fusion_config(self, filepath: str) -> None:
        """
        Load fusion configuration from file.
        
        Args:
            filepath: Path to load configuration from
        """
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.normalization_method = config_data.get('normalization_method', 'standard')
        self.handle_missing = config_data.get('handle_missing', 'zero')
        self.attention_hidden_dim = config_data.get('attention_hidden_dim', 128)
        self.random_state = config_data.get('random_state', 42)
        self.attention_trained = config_data.get('attention_trained', False)
        self.fusion_history = config_data.get('fusion_history', [])
        
        self.logger.info(f"Fusion configuration loaded from {filepath}")


class DimensionalityReducer:
    """
    Comprehensive dimensionality reduction and feature selection system.
    
    Implements PCA, feature selection methods, correlation analysis, and feature
    importance ranking to manage high-dimensional features effectively.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DimensionalityReducer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.fitted_reducers = {}
        self.feature_importance_scores = {}
        self.correlation_matrix = None
        self.selected_features = None
        
        self.logger = logging.getLogger(__name__)
    
    def apply_pca(self, features: np.ndarray, target_dim: int, 
                  fit: bool = True) -> DimensionalityReductionResult:
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Args:
            features: Input features array [n_samples, n_features]
            target_dim: Target number of dimensions
            fit: Whether to fit PCA (True for training, False for inference)
            
        Returns:
            DimensionalityReductionResult with reduced features and metadata
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn not available for PCA")
        
        original_dim = features.shape[1]
        
        if original_dim <= target_dim:
            self.logger.info(f"Features already have {original_dim} dimensions, no PCA needed")
            return DimensionalityReductionResult(
                reduced_features=features,
                original_dimension=original_dim,
                target_dimension=original_dim,
                method='none',
                explained_variance_ratio=1.0
            )
        
        # Initialize or retrieve PCA
        if fit or 'pca' not in self.fitted_reducers:
            pca = PCA(n_components=target_dim, random_state=self.random_state)
            reduced_features = pca.fit_transform(features)
            self.fitted_reducers['pca'] = pca
        else:
            pca = self.fitted_reducers['pca']
            reduced_features = pca.transform(features)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        
        result = DimensionalityReductionResult(
            reduced_features=reduced_features,
            original_dimension=original_dim,
            target_dimension=target_dim,
            method='pca',
            explained_variance_ratio=explained_variance,
            metadata={
                'explained_variance_per_component': pca.explained_variance_ratio_.tolist(),
                'singular_values': pca.singular_values_.tolist(),
                'n_components': pca.n_components_
            }
        )
        
        self.logger.info(f"PCA applied: {original_dim} -> {target_dim} dimensions, "
                        f"explained variance: {explained_variance:.3f}")
        
        return result
    
    def apply_feature_selection(self, features: np.ndarray, target_values: np.ndarray, 
                              target_dim: int, method: str = 'f_regression',
                              fit: bool = True) -> DimensionalityReductionResult:
        """
        Apply feature selection based on statistical tests.
        
        Args:
            features: Input features array [n_samples, n_features]
            target_values: Target values for supervised selection
            target_dim: Target number of features to select
            method: Selection method ('f_regression', 'mutual_info')
            fit: Whether to fit selector (True for training, False for inference)
            
        Returns:
            DimensionalityReductionResult with selected features and metadata
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn not available for feature selection")
        
        original_dim = features.shape[1]
        
        if original_dim <= target_dim:
            self.logger.info(f"Features already have {original_dim} dimensions, no selection needed")
            return DimensionalityReductionResult(
                reduced_features=features,
                original_dimension=original_dim,
                target_dimension=original_dim,
                method='none',
                selected_features=np.arange(original_dim)
            )
        
        # Initialize selector
        if method == 'f_regression':
            score_func = f_regression
        elif method == 'mutual_info':
            score_func = mutual_info_regression
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        selector_key = f'selector_{method}'
        
        if fit or selector_key not in self.fitted_reducers:
            selector = SelectKBest(score_func=score_func, k=target_dim)
            reduced_features = selector.fit_transform(features, target_values)
            self.fitted_reducers[selector_key] = selector
            
            # Store feature importance scores
            self.feature_importance_scores[method] = selector.scores_
        else:
            selector = self.fitted_reducers[selector_key]
            reduced_features = selector.transform(features)
        
        selected_features = selector.get_support(indices=True)
        
        result = DimensionalityReductionResult(
            reduced_features=reduced_features,
            original_dimension=original_dim,
            target_dimension=target_dim,
            method=f'feature_selection_{method}',
            selected_features=selected_features,
            metadata={
                'feature_scores': selector.scores_.tolist(),
                'selected_indices': selected_features.tolist(),
                'score_function': method
            }
        )
        
        self.logger.info(f"Feature selection ({method}): {original_dim} -> {target_dim} features")
        
        return result
    
    def remove_low_variance_features(self, features: np.ndarray, 
                                   threshold: float = 0.01,
                                   fit: bool = True) -> DimensionalityReductionResult:
        """
        Remove features with low variance.
        
        Args:
            features: Input features array [n_samples, n_features]
            threshold: Variance threshold below which features are removed
            fit: Whether to fit selector (True for training, False for inference)
            
        Returns:
            DimensionalityReductionResult with filtered features
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn not available for variance filtering")
        
        original_dim = features.shape[1]
        
        if fit or 'variance_selector' not in self.fitted_reducers:
            selector = VarianceThreshold(threshold=threshold)
            reduced_features = selector.fit_transform(features)
            self.fitted_reducers['variance_selector'] = selector
        else:
            selector = self.fitted_reducers['variance_selector']
            reduced_features = selector.transform(features)
        
        selected_features = selector.get_support(indices=True)
        target_dim = len(selected_features)
        
        result = DimensionalityReductionResult(
            reduced_features=reduced_features,
            original_dimension=original_dim,
            target_dimension=target_dim,
            method='variance_threshold',
            selected_features=selected_features,
            metadata={
                'threshold': threshold,
                'variances': selector.variances_.tolist(),
                'selected_indices': selected_features.tolist()
            }
        )
        
        self.logger.info(f"Variance filtering: {original_dim} -> {target_dim} features "
                        f"(threshold: {threshold})")
        
        return result
    
    def analyze_feature_correlations(self, features: np.ndarray, 
                                   correlation_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Analyze feature correlations and identify highly correlated features.
        
        Args:
            features: Input features array [n_samples, n_features]
            correlation_threshold: Threshold above which features are considered highly correlated
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(features.T)
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        n_features = features.shape[1]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_value = abs(self.correlation_matrix[i, j])
                if corr_value > correlation_threshold:
                    high_corr_pairs.append({
                        'feature_1': i,
                        'feature_2': j,
                        'correlation': corr_value
                    })
        
        # Identify features to remove (keep first occurrence)
        features_to_remove = set()
        for pair in high_corr_pairs:
            features_to_remove.add(pair['feature_2'])
        
        features_to_keep = [i for i in range(n_features) if i not in features_to_remove]
        
        analysis_result = {
            'correlation_matrix_shape': self.correlation_matrix.shape,
            'high_correlation_pairs': high_corr_pairs,
            'features_to_remove': list(features_to_remove),
            'features_to_keep': features_to_keep,
            'n_original_features': n_features,
            'n_features_after_removal': len(features_to_keep),
            'correlation_threshold': correlation_threshold
        }
        
        self.logger.info(f"Correlation analysis: {len(high_corr_pairs)} highly correlated pairs found, "
                        f"{len(features_to_remove)} features recommended for removal")
        
        return analysis_result
    
    def remove_correlated_features(self, features: np.ndarray, 
                                 correlation_threshold: float = 0.95) -> DimensionalityReductionResult:
        """
        Remove highly correlated features.
        
        Args:
            features: Input features array [n_samples, n_features]
            correlation_threshold: Threshold above which features are considered highly correlated
            
        Returns:
            DimensionalityReductionResult with uncorrelated features
        """
        original_dim = features.shape[1]
        
        # Analyze correlations
        analysis = self.analyze_feature_correlations(features, correlation_threshold)
        features_to_keep = analysis['features_to_keep']
        
        # Select uncorrelated features
        reduced_features = features[:, features_to_keep]
        
        result = DimensionalityReductionResult(
            reduced_features=reduced_features,
            original_dimension=original_dim,
            target_dimension=len(features_to_keep),
            method='correlation_removal',
            selected_features=np.array(features_to_keep),
            metadata=analysis
        )
        
        self.logger.info(f"Correlation removal: {original_dim} -> {len(features_to_keep)} features")
        
        return result
    
    def rank_feature_importance(self, features: np.ndarray, target_values: np.ndarray,
                              method: str = 'f_regression') -> np.ndarray:
        """
        Rank features by importance using statistical tests.
        
        Args:
            features: Input features array [n_samples, n_features]
            target_values: Target values for supervised ranking
            method: Ranking method ('f_regression', 'mutual_info')
            
        Returns:
            Array of feature indices sorted by importance (descending)
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn not available for feature ranking")
        
        if method == 'f_regression':
            scores, _ = f_regression(features, target_values)
        elif method == 'mutual_info':
            scores = mutual_info_regression(features, target_values, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        # Handle NaN scores
        scores = np.nan_to_num(scores, nan=0.0)
        
        # Sort by importance (descending)
        importance_ranking = np.argsort(scores)[::-1]
        
        # Store scores
        self.feature_importance_scores[method] = scores
        
        self.logger.info(f"Feature importance ranking completed using {method}")
        
        return importance_ranking
    
    def select_top_features(self, features: np.ndarray, target_values: np.ndarray,
                          n_features: int, method: str = 'f_regression') -> DimensionalityReductionResult:
        """
        Select top N features based on importance ranking.
        
        Args:
            features: Input features array [n_samples, n_features]
            target_values: Target values for supervised selection
            n_features: Number of top features to select
            method: Ranking method ('f_regression', 'mutual_info')
            
        Returns:
            DimensionalityReductionResult with top features
        """
        original_dim = features.shape[1]
        
        if original_dim <= n_features:
            self.logger.info(f"Features already have {original_dim} dimensions, no selection needed")
            return DimensionalityReductionResult(
                reduced_features=features,
                original_dimension=original_dim,
                target_dimension=original_dim,
                method='none',
                selected_features=np.arange(original_dim)
            )
        
        # Rank features
        importance_ranking = self.rank_feature_importance(features, target_values, method)
        
        # Select top features
        top_features = importance_ranking[:n_features]
        reduced_features = features[:, top_features]
        
        result = DimensionalityReductionResult(
            reduced_features=reduced_features,
            original_dimension=original_dim,
            target_dimension=n_features,
            method=f'top_features_{method}',
            selected_features=top_features,
            metadata={
                'importance_scores': self.feature_importance_scores[method].tolist(),
                'importance_ranking': importance_ranking.tolist(),
                'selected_indices': top_features.tolist(),
                'ranking_method': method
            }
        )
        
        self.logger.info(f"Top feature selection ({method}): {original_dim} -> {n_features} features")
        
        return result
    
    def get_feature_importance_scores(self, method: str = None) -> Dict[str, np.ndarray]:
        """
        Get stored feature importance scores.
        
        Args:
            method: Specific method to retrieve, or None for all methods
            
        Returns:
            Dictionary of feature importance scores
        """
        if method is None:
            return self.feature_importance_scores.copy()
        elif method in self.feature_importance_scores:
            return {method: self.feature_importance_scores[method]}
        else:
            return {}
    
    def save_reducer_state(self, filepath: str) -> None:
        """
        Save dimensionality reducer state to file.
        
        Args:
            filepath: Path to save state
        """
        import pickle
        
        state_data = {
            'random_state': self.random_state,
            'fitted_reducers': self.fitted_reducers,
            'feature_importance_scores': self.feature_importance_scores,
            'correlation_matrix': self.correlation_matrix,
            'selected_features': self.selected_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
        
        self.logger.info(f"Reducer state saved to {filepath}")
    
    def load_reducer_state(self, filepath: str) -> None:
        """
        Load dimensionality reducer state from file.
        
        Args:
            filepath: Path to load state from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)
        
        self.random_state = state_data.get('random_state', 42)
        self.fitted_reducers = state_data.get('fitted_reducers', {})
        self.feature_importance_scores = state_data.get('feature_importance_scores', {})
        self.correlation_matrix = state_data.get('correlation_matrix', None)
        self.selected_features = state_data.get('selected_features', None)
        
        self.logger.info(f"Reducer state loaded from {filepath}")


# Convenience functions for easy usage
def fuse_features(text_features: np.ndarray, image_features: np.ndarray, 
                 method: str = 'concatenate', **kwargs) -> np.ndarray:
    """
    Convenience function for feature fusion.
    
    Args:
        text_features: Text features array
        image_features: Image features array
        method: Fusion method ('concatenate', 'weighted', 'attention')
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Fused features array
    """
    fusion = FeatureFusion()
    
    if method == 'concatenate':
        return fusion.concatenate_features(text_features, image_features)
    elif method == 'weighted':
        weights = kwargs.get('weights', (0.5, 0.5))
        return fusion.weighted_fusion(text_features, image_features, weights)
    elif method == 'attention':
        return fusion.attention_fusion(text_features, image_features)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def reduce_feature_dimensions(features: np.ndarray, target_dim: int, 
                            method: str = 'pca', **kwargs) -> np.ndarray:
    """
    Convenience function for dimensionality reduction.
    
    Args:
        features: Input features array
        target_dim: Target number of dimensions
        method: Reduction method ('pca', 'variance', 'correlation')
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Reduced features array
    """
    reducer = DimensionalityReducer()
    
    if method == 'pca':
        result = reducer.apply_pca(features, target_dim)
    elif method == 'variance':
        threshold = kwargs.get('threshold', 0.01)
        result = reducer.remove_low_variance_features(features, threshold)
    elif method == 'correlation':
        threshold = kwargs.get('correlation_threshold', 0.95)
        result = reducer.remove_correlated_features(features, threshold)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    return result.reduced_features