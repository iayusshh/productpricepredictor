"""
EnsembleManager implementation for ML Product Pricing Challenge 2025

This module implements voting, averaging, and stacking ensemble methods,
model weight optimization based on validation performance, and ensemble prediction generation.
"""

import logging
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize

from ..config import MLPricingConfig


class EnsembleManager:
    """
    Ensemble manager for model combination
    
    Implements voting, averaging, and stacking ensemble methods with
    model weight optimization based on validation performance.
    """
    
    def __init__(self, config: MLPricingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.trained_models = {}
        self.ensemble_models = {}
        self.model_weights = {}
        self.validation_scores = {}
        
        # Create necessary directories
        self.models_dir = Path("models")
        self.logs_dir = Path(config.infrastructure.log_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for ensemble management"""
        logger = logging.getLogger(f"{__name__}.EnsembleManager")
        logger.setLevel(getattr(logging, self.config.infrastructure.log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            log_file = self.logs_dir / "ensemble_management.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def add_models(self, models: Dict[str, Any], validation_scores: Dict[str, float] = None):
        """Add trained models to the ensemble"""
        self.trained_models.update(models)
        
        if validation_scores:
            self.validation_scores.update(validation_scores)
        
        self.logger.info(f"Added {len(models)} models to ensemble. Total models: {len(self.trained_models)}")
    
    def create_voting_ensemble(self, model_names: List[str] = None) -> VotingRegressor:
        """
        Create voting ensemble from selected models
        
        Args:
            model_names: List of model names to include. If None, uses all models.
            
        Returns:
            VotingRegressor ensemble
        """
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        self.logger.info(f"Creating voting ensemble with models: {model_names}")
        
        # Prepare models for VotingRegressor
        estimators = []
        for name in model_names:
            if name in self.trained_models:
                model = self.trained_models[name]
                # Skip neural network models for sklearn VotingRegressor
                if hasattr(model, 'predict') and not hasattr(model, 'forward'):
                    estimators.append((name, model))
                else:
                    self.logger.warning(f"Skipping {name} for voting ensemble (incompatible model type)")
        
        if not estimators:
            raise ValueError("No compatible models found for voting ensemble")
        
        voting_ensemble = VotingRegressor(estimators=estimators)
        
        self.ensemble_models['voting'] = voting_ensemble
        self.logger.info(f"Voting ensemble created with {len(estimators)} models")
        
        return voting_ensemble
    
    def create_weighted_average_ensemble(self, model_names: List[str] = None, 
                                       weights: List[float] = None) -> 'WeightedAverageEnsemble':
        """
        Create weighted average ensemble
        
        Args:
            model_names: List of model names to include
            weights: List of weights for each model. If None, optimizes weights.
            
        Returns:
            WeightedAverageEnsemble instance
        """
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        self.logger.info(f"Creating weighted average ensemble with models: {model_names}")
        
        selected_models = {name: self.trained_models[name] for name in model_names 
                          if name in self.trained_models}
        
        if not selected_models:
            raise ValueError("No valid models found for weighted average ensemble")
        
        # Optimize weights if not provided
        if weights is None:
            weights = self._optimize_ensemble_weights(selected_models)
        
        weighted_ensemble = WeightedAverageEnsemble(selected_models, weights)
        
        self.ensemble_models['weighted_average'] = weighted_ensemble
        self.model_weights['weighted_average'] = dict(zip(model_names, weights))
        
        self.logger.info(f"Weighted average ensemble created with weights: {dict(zip(model_names, weights))}")
        
        return weighted_ensemble
    
    def create_stacking_ensemble(self, model_names: List[str] = None, 
                               meta_learner: Any = None) -> 'StackingEnsemble':
        """
        Create stacking ensemble with meta-learner
        
        Args:
            model_names: List of model names to use as base learners
            meta_learner: Meta-learner model. If None, uses Ridge regression.
            
        Returns:
            StackingEnsemble instance
        """
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        if meta_learner is None:
            meta_learner = Ridge(alpha=1.0, random_state=self.config.model.random_seed)
        
        self.logger.info(f"Creating stacking ensemble with models: {model_names}")
        
        selected_models = {name: self.trained_models[name] for name in model_names 
                          if name in self.trained_models}
        
        if not selected_models:
            raise ValueError("No valid models found for stacking ensemble")
        
        stacking_ensemble = StackingEnsemble(selected_models, meta_learner)
        
        self.ensemble_models['stacking'] = stacking_ensemble
        
        self.logger.info(f"Stacking ensemble created with {len(selected_models)} base learners")
        
        return stacking_ensemble
    
    def _optimize_ensemble_weights(self, models: Dict[str, Any], 
                                 X_val: np.ndarray = None, y_val: np.ndarray = None) -> List[float]:
        """
        Optimize ensemble weights based on validation performance
        
        Args:
            models: Dictionary of models
            X_val: Validation features (if None, uses stored validation scores)
            y_val: Validation targets
            
        Returns:
            List of optimized weights
        """
        model_names = list(models.keys())
        n_models = len(model_names)
        
        if X_val is not None and y_val is not None:
            # Use provided validation data
            self.logger.info("Optimizing weights using provided validation data")
            
            # Get predictions from each model
            predictions = np.zeros((len(y_val), n_models))
            for i, (name, model) in enumerate(models.items()):
                predictions[:, i] = self._predict_with_model(model, X_val)
            
            # Objective function to minimize SMAPE
            def objective(weights):
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize weights
                
                ensemble_pred = np.dot(predictions, weights)
                smape = self._calculate_smape(y_val, ensemble_pred)
                return smape
            
            # Constraints: weights sum to 1 and are non-negative
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(n_models)]
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_models) / n_models
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = result.x.tolist()
                self.logger.info(f"Weight optimization successful. Final SMAPE: {result.fun:.4f}")
            else:
                self.logger.warning("Weight optimization failed. Using equal weights.")
                optimized_weights = [1.0 / n_models] * n_models
        
        else:
            # Use validation scores for weight optimization
            self.logger.info("Optimizing weights using validation scores")
            
            if not self.validation_scores:
                self.logger.warning("No validation scores available. Using equal weights.")
                return [1.0 / n_models] * n_models
            
            # Convert SMAPE scores to weights (lower SMAPE = higher weight)
            scores = []
            for name in model_names:
                if name in self.validation_scores:
                    scores.append(self.validation_scores[name])
                else:
                    scores.append(100.0)  # High SMAPE for missing scores
            
            scores = np.array(scores)
            
            # Inverse weighting: better models (lower SMAPE) get higher weights
            inverse_scores = 1.0 / (scores + 1e-8)  # Add small epsilon to avoid division by zero
            optimized_weights = inverse_scores / np.sum(inverse_scores)
            optimized_weights = optimized_weights.tolist()
        
        self.logger.info(f"Optimized weights: {dict(zip(model_names, optimized_weights))}")
        return optimized_weights
    
    def _predict_with_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with a model (handles different model types)"""
        try:
            # Handle PyTorch models
            if hasattr(model, 'forward'):
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    predictions = model(X_tensor).numpy().flatten()
            else:
                # Standard sklearn-like models
                predictions = model.predict(X)
            
            return predictions
        except Exception as e:
            self.logger.error(f"Error making predictions with model: {str(e)}")
            # Return zeros as fallback
            return np.zeros(len(X))
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        epsilon = self.config.evaluation.smape_epsilon
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        denominator = np.maximum(denominator, epsilon)
        
        smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
        return smape
    
    def evaluate_ensemble(self, ensemble_name: str, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Evaluate ensemble performance
        
        Args:
            ensemble_name: Name of the ensemble to evaluate
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        predictions = ensemble.predict(X_val)
        
        metrics = {
            'smape': self._calculate_smape(y_val, predictions),
            'mae': mean_absolute_error(y_val, predictions),
            'mse': mean_squared_error(y_val, predictions),
            'rmse': np.sqrt(mean_squared_error(y_val, predictions)),
            'r2': r2_score(y_val, predictions),
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions)
        }
        
        self.logger.info(f"{ensemble_name} ensemble SMAPE: {metrics['smape']:.4f}")
        
        return metrics
    
    def cross_validate_ensemble(self, ensemble_name: str, X: np.ndarray, y: np.ndarray, 
                              cv_folds: int = None) -> Dict:
        """
        Perform cross-validation on ensemble
        
        Args:
            ensemble_name: Name of the ensemble
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of CV results
        """
        if cv_folds is None:
            cv_folds = self.config.model.cv_folds
        
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        self.logger.info(f"Cross-validating {ensemble_name} ensemble with {cv_folds} folds")
        
        ensemble = self.ensemble_models[ensemble_name]
        
        # Custom SMAPE scorer
        def smape_scorer(estimator, X, y):
            predictions = estimator.predict(X)
            return -self._calculate_smape(y, predictions)  # Negative because CV maximizes
        
        cv_scores = cross_val_score(ensemble, X, y, cv=cv_folds, scoring=smape_scorer, n_jobs=-1)
        cv_scores = -cv_scores  # Convert back to positive SMAPE
        
        cv_results = {
            'mean_smape': np.mean(cv_scores),
            'std_smape': np.std(cv_scores),
            'min_smape': np.min(cv_scores),
            'max_smape': np.max(cv_scores),
            'cv_scores': cv_scores.tolist()
        }
        
        self.logger.info(f"{ensemble_name} CV SMAPE: {cv_results['mean_smape']:.4f} ± {cv_results['std_smape']:.4f}")
        
        return cv_results
    
    def generate_ensemble_predictions(self, ensemble_name: str, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions using specified ensemble
        
        Args:
            ensemble_name: Name of the ensemble
            X_test: Test features
            
        Returns:
            Array of predictions
        """
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        predictions = ensemble.predict(X_test)
        
        self.logger.info(f"Generated {len(predictions)} predictions using {ensemble_name} ensemble")
        
        return predictions
    
    def save_ensemble(self, ensemble_name: str, filepath: str):
        """Save ensemble model to disk"""
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble, f)
        
        # Save metadata
        metadata = {
            'ensemble_name': ensemble_name,
            'ensemble_type': type(ensemble).__name__,
            'saved_at': datetime.now().isoformat(),
            'model_weights': self.model_weights.get(ensemble_name, {}),
            'config': self.config.model.__dict__
        }
        
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Ensemble '{ensemble_name}' saved to {filepath}")
    
    def load_ensemble(self, filepath: str) -> Any:
        """Load ensemble model from disk"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Ensemble file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            ensemble = pickle.load(f)
        
        self.logger.info(f"Ensemble loaded from {filepath}")
        
        return ensemble
    
    def compare_ensembles(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Compare performance of all created ensembles
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of comparison results
        """
        if not self.ensemble_models:
            raise ValueError("No ensembles created yet")
        
        self.logger.info("Comparing ensemble performance")
        
        comparison_results = {}
        
        for ensemble_name in self.ensemble_models.keys():
            try:
                metrics = self.evaluate_ensemble(ensemble_name, X_val, y_val)
                comparison_results[ensemble_name] = metrics
            except Exception as e:
                self.logger.error(f"Error evaluating {ensemble_name}: {str(e)}")
                continue
        
        # Find best ensemble
        if comparison_results:
            best_ensemble = min(comparison_results.keys(), 
                              key=lambda x: comparison_results[x]['smape'])
            
            comparison_results['best_ensemble'] = {
                'name': best_ensemble,
                'smape': comparison_results[best_ensemble]['smape']
            }
            
            self.logger.info(f"Best ensemble: {best_ensemble} (SMAPE: {comparison_results[best_ensemble]['smape']:.4f})")
        
        # Save comparison results
        results_file = self.logs_dir / f"ensemble_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        return comparison_results


class WeightedAverageEnsemble:
    """Weighted average ensemble implementation"""
    
    def __init__(self, models: Dict[str, Any], weights: List[float]):
        self.models = models
        self.weights = np.array(weights)
        self.model_names = list(models.keys())
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted average predictions"""
        predictions = np.zeros((len(X), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            try:
                if hasattr(model, 'forward'):  # PyTorch model
                    import torch
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        pred = model(X_tensor).numpy().flatten()
                else:
                    pred = model.predict(X)
                
                predictions[:, i] = pred
            except Exception as e:
                # Use zeros for failed predictions
                predictions[:, i] = 0
        
        # Weighted average
        weighted_predictions = np.dot(predictions, self.weights)
        
        return weighted_predictions
    
    def get_weights(self) -> Dict[str, float]:
        """Get model weights"""
        return dict(zip(self.model_names, self.weights))


class StackingEnsemble:
    """Stacking ensemble implementation"""
    
    def __init__(self, base_models: Dict[str, Any], meta_learner: Any):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
        """Fit the stacking ensemble"""
        from sklearn.model_selection import cross_val_predict
        
        # Generate meta-features using cross-validation
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                if hasattr(model, 'forward'):  # Skip PyTorch models for now
                    meta_features[:, i] = 0
                else:
                    # Use cross-validation to generate out-of-fold predictions
                    cv_predictions = cross_val_predict(model, X, y, cv=cv_folds)
                    meta_features[:, i] = cv_predictions
            except Exception as e:
                meta_features[:, i] = 0
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacking predictions"""
        if not self.is_fitted:
            raise ValueError("Stacking ensemble must be fitted before making predictions")
        
        # Generate base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                if hasattr(model, 'forward'):  # PyTorch model
                    import torch
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        pred = model(X_tensor).numpy().flatten()
                else:
                    pred = model.predict(X)
                
                base_predictions[:, i] = pred
            except Exception as e:
                base_predictions[:, i] = 0
        
        # Meta-learner prediction
        final_predictions = self.meta_learner.predict(base_predictions)
        
        return final_predictions