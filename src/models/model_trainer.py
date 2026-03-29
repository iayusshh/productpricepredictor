"""
ModelTrainer implementation for ML Product Pricing Challenge 2025

This module implements reproducible model training with experiment tracking,
supporting Random Forest, XGBoost, LightGBM, Extra Trees, Gradient Boosting,
Ridge Regression, and neural network models (7 total).
"""

import os
import json
import pickle
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Neural network training will be disabled.")

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    logging.warning("scikit-optimize not available. Bayesian optimization will be disabled.")

from ..interfaces import ModelTrainerInterface
from ..config import MLPricingConfig


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for price prediction"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout_rate: float = 0.3):
        super(SimpleNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ModelTrainer(ModelTrainerInterface):
    """
    Reproducible model trainer with experiment tracking
    
    Implements training pipelines for Random Forest, XGBoost, LightGBM, and neural networks
    with comprehensive experiment tracking and hyperparameter tuning.
    """
    
    def __init__(self, config: MLPricingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.experiment_metadata = {}
        self.trained_models = {}
        
        # Create necessary directories
        self.models_dir = Path("models")
        self.logs_dir = Path(config.infrastructure.log_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for model training"""
        logger = logging.getLogger(f"{__name__}.ModelTrainer")
        logger.setLevel(getattr(logging, self.config.infrastructure.log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            log_file = self.logs_dir / "model_training.log"
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
    
    def set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility across all libraries"""
        self.logger.info(f"Setting random seeds to {seed}")
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch (if available)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Set environment variable for other libraries
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.logger.info("Random seeds set successfully")
    
    def capture_experiment_metadata(self, config: Dict, cv_folds: int, seed: int) -> str:
        """Capture comprehensive experiment metadata for tracking"""
        timestamp = datetime.now().isoformat()
        experiment_id = f"exp_{timestamp.replace(':', '-').replace('.', '-')}"
        
        metadata = {
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'config': config,
            'cv_folds': cv_folds,
            'random_seed': seed,
            'system_info': {
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'numpy_version': np.__version__,
                'torch_available': TORCH_AVAILABLE,
                'bayesian_opt_available': BAYESIAN_OPT_AVAILABLE
            },
            'data_info': {
                'feature_engineering_config': self.config.feature_fusion.__dict__,
                'text_config': self.config.text_features.__dict__,
                'image_config': self.config.image_features.__dict__
            }
        }
        
        # Save metadata to file
        metadata_file = self.logs_dir / f"{experiment_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.experiment_metadata[experiment_id] = metadata
        self.logger.info(f"Experiment metadata captured: {experiment_id}")
        
        return experiment_id
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> Any:
        """Train model with given configuration"""
        model_type = model_config.get('model_type', 'random_forest')
        self.logger.info(f"Training {model_type} model")
        
        if model_type == 'random_forest':
            return self._train_random_forest(X, y, model_config)
        elif model_type == 'xgboost':
            return self._train_xgboost(X, y, model_config)
        elif model_type == 'lightgbm':
            return self._train_lightgbm(X, y, model_config)
        elif model_type == 'neural_network':
            return self._train_neural_network(X, y, model_config)
        elif model_type == 'extra_trees':
            return self._train_extra_trees(X, y, model_config)
        elif model_type == 'gradient_boosting':
            return self._train_gradient_boosting(X, y, model_config)
        elif model_type == 'ridge_regression':
            return self._train_ridge_regression(X, y, model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> RandomForestRegressor:
        """Train Random Forest model"""
        params = {
            'n_estimators': model_config.get('n_estimators', self.config.model.rf_n_estimators),
            'max_depth': model_config.get('max_depth', self.config.model.rf_max_depth),
            'min_samples_split': model_config.get('min_samples_split', self.config.model.rf_min_samples_split),
            'random_state': self.config.model.random_seed,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X, y)
        
        self.logger.info(f"Random Forest trained with {params}")
        return model
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        params = {
            'n_estimators': model_config.get('n_estimators', self.config.model.xgb_n_estimators),
            'max_depth': model_config.get('max_depth', self.config.model.xgb_max_depth),
            'learning_rate': model_config.get('learning_rate', self.config.model.xgb_learning_rate),
            'subsample': model_config.get('subsample', self.config.model.xgb_subsample),
            'random_state': self.config.model.random_seed,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        
        self.logger.info(f"XGBoost trained with {params}")
        return model
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        params = {
            'n_estimators': model_config.get('n_estimators', self.config.model.lgb_n_estimators),
            'max_depth': model_config.get('max_depth', self.config.model.lgb_max_depth),
            'learning_rate': model_config.get('learning_rate', self.config.model.lgb_learning_rate),
            'num_leaves': model_config.get('num_leaves', self.config.model.lgb_num_leaves),
            'random_state': self.config.model.random_seed,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        
        self.logger.info(f"LightGBM trained with {params}")
        return model
    
    def _train_extra_trees(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> ExtraTreesRegressor:
        """Train Extra Trees Regressor — fast, low-variance ensemble of fully random trees"""
        params = {
            'n_estimators': model_config.get('n_estimators', self.config.model.et_n_estimators),
            'max_depth': model_config.get('max_depth', self.config.model.et_max_depth),
            'min_samples_split': model_config.get('min_samples_split', 2),
            'random_state': self.config.model.random_seed,
            'n_jobs': -1
        }
        model = ExtraTreesRegressor(**params)
        model.fit(X, y)
        self.logger.info(f"Extra Trees trained with {params}")
        return model

    def _train_gradient_boosting(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> GradientBoostingRegressor:
        """Train sklearn Gradient Boosting — sequential boosting, different from XGBoost"""
        params = {
            'n_estimators': model_config.get('n_estimators', self.config.model.gbr_n_estimators),
            'max_depth': model_config.get('max_depth', self.config.model.gbr_max_depth),
            'learning_rate': model_config.get('learning_rate', self.config.model.gbr_learning_rate),
            'subsample': model_config.get('subsample', self.config.model.gbr_subsample),
            'random_state': self.config.model.random_seed
        }
        model = GradientBoostingRegressor(**params)
        model.fit(X, y)
        self.logger.info(f"Gradient Boosting trained with {params}")
        return model

    def _train_ridge_regression(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> Ridge:
        """Train Ridge Regression — fast linear model with L2 regularization"""
        params = {
            'alpha': model_config.get('alpha', self.config.model.ridge_alpha),
        }
        model = Ridge(**params)
        model.fit(X, y)
        self.logger.info(f"Ridge Regression trained with {params}")
        return model

    def _train_neural_network(self, X: np.ndarray, y: np.ndarray, model_config: Dict) -> Optional[SimpleNeuralNetwork]:
        """Train neural network model"""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available. Cannot train neural network.")
            return None
        
        # Model parameters
        hidden_layers = model_config.get('hidden_layers', self.config.model.nn_hidden_layers)
        dropout_rate = model_config.get('dropout_rate', self.config.model.nn_dropout_rate)
        learning_rate = model_config.get('learning_rate', self.config.model.nn_learning_rate)
        batch_size = model_config.get('batch_size', self.config.model.nn_batch_size)
        epochs = model_config.get('epochs', self.config.model.nn_epochs)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create model
        model = SimpleNeuralNetwork(X.shape[1], hidden_layers, dropout_rate)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.logger.info(f"Neural network trained for {epochs} epochs")
        return model
    
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune hyperparameters to minimize SMAPE on validation data"""
        if not self.config.model.use_hyperparameter_tuning:
            self.logger.info("Hyperparameter tuning disabled")
            return {}
        
        tuning_method = self.config.model.tuning_method
        cv_folds = self.config.model.tuning_cv_folds
        
        self.logger.info(f"Starting hyperparameter tuning using {tuning_method}")
        
        best_params = {}
        
        for model_type in self.config.model.model_types:
            if model_type == 'neural_network':
                # Skip neural network tuning for now (would require custom implementation)
                continue
            
            self.logger.info(f"Tuning hyperparameters for {model_type}")
            
            if tuning_method == 'grid_search':
                best_params[model_type] = self._grid_search_tuning(X, y, model_type, cv_folds)
            elif tuning_method == 'random_search':
                best_params[model_type] = self._random_search_tuning(X, y, model_type, cv_folds)
            elif tuning_method == 'bayesian' and BAYESIAN_OPT_AVAILABLE:
                best_params[model_type] = self._bayesian_tuning(X, y, model_type, cv_folds)
            else:
                self.logger.warning(f"Tuning method {tuning_method} not available, using grid search")
                best_params[model_type] = self._grid_search_tuning(X, y, model_type, cv_folds)
        
        # Save tuning results
        tuning_results_file = self.logs_dir / "hyperparameter_tuning_results.json"
        with open(tuning_results_file, 'w') as f:
            json.dump(best_params, f, indent=2, default=str)
        
        self.logger.info("Hyperparameter tuning completed")
        return best_params
    
    def _grid_search_tuning(self, X: np.ndarray, y: np.ndarray, model_type: str, cv_folds: int) -> Dict:
        """Perform grid search hyperparameter tuning"""
        param_grids = self._get_param_grids()
        param_grid = param_grids.get(model_type, {})
        
        if not param_grid:
            return {}
        
        model = self._get_base_model(model_type)
        
        # Use custom SMAPE scorer
        def smape_scorer(y_true, y_pred):
            return -self._calculate_smape(y_true, y_pred)  # Negative because GridSearchCV maximizes
        
        from sklearn.metrics import make_scorer
        smape_score = make_scorer(smape_scorer, greater_is_better=True)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, scoring=smape_score, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_
    
    def _random_search_tuning(self, X: np.ndarray, y: np.ndarray, model_type: str, cv_folds: int) -> Dict:
        """Perform random search hyperparameter tuning"""
        param_distributions = self._get_param_distributions()
        param_dist = param_distributions.get(model_type, {})
        
        if not param_dist:
            return {}
        
        model = self._get_base_model(model_type)
        
        def smape_scorer(y_true, y_pred):
            return -self._calculate_smape(y_true, y_pred)
        
        from sklearn.metrics import make_scorer
        smape_score = make_scorer(smape_scorer, greater_is_better=True)
        
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=20, cv=cv_folds, scoring=smape_score, 
            n_jobs=-1, verbose=1, random_state=self.config.model.random_seed
        )
        
        random_search.fit(X, y)
        
        return random_search.best_params_
    
    def _bayesian_tuning(self, X: np.ndarray, y: np.ndarray, model_type: str, cv_folds: int) -> Dict:
        """Perform Bayesian optimization hyperparameter tuning"""
        search_spaces = self._get_bayesian_search_spaces()
        search_space = search_spaces.get(model_type, {})
        
        if not search_space:
            return {}
        
        model = self._get_base_model(model_type)
        
        def smape_scorer(y_true, y_pred):
            return -self._calculate_smape(y_true, y_pred)
        
        from sklearn.metrics import make_scorer
        smape_score = make_scorer(smape_scorer, greater_is_better=True)
        
        bayes_search = BayesSearchCV(
            model, search_space, n_iter=20, cv=cv_folds, scoring=smape_score,
            n_jobs=-1, verbose=1, random_state=self.config.model.random_seed
        )
        
        bayes_search.fit(X, y)
        
        return bayes_search.best_params_
    
    def _get_base_model(self, model_type: str):
        """Get base model for hyperparameter tuning"""
        if model_type == 'random_forest':
            return RandomForestRegressor(random_state=self.config.model.random_seed, n_jobs=-1)
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(random_state=self.config.model.random_seed, n_jobs=-1)
        elif model_type == 'lightgbm':
            return lgb.LGBMRegressor(random_state=self.config.model.random_seed, n_jobs=-1, verbose=-1)
        elif model_type == 'extra_trees':
            return ExtraTreesRegressor(random_state=self.config.model.random_seed, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(random_state=self.config.model.random_seed)
        elif model_type == 'ridge_regression':
            return Ridge()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for grid search"""
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'extra_trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ridge_regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        }
    
    def _get_param_distributions(self) -> Dict[str, Dict]:
        """Get parameter distributions for random search"""
        from scipy.stats import randint, uniform

        return {
            'random_forest': {
                'n_estimators': randint(50, 300),
                'max_depth': [None] + list(range(10, 50, 5)),
                'min_samples_split': randint(2, 20)
            },
            'xgboost': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4)
            },
            'lightgbm': {
                'n_estimators': randint(50, 300),
                'max_depth': [-1] + list(range(10, 50, 5)),
                'learning_rate': uniform(0.01, 0.3),
                'num_leaves': randint(20, 200)
            },
            'extra_trees': {
                'n_estimators': randint(50, 300),
                'max_depth': [None] + list(range(10, 50, 5)),
                'min_samples_split': randint(2, 20)
            },
            'gradient_boosting': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 8),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4)
            },
            'ridge_regression': {
                'alpha': uniform(0.001, 200.0)
            }
        }
    
    def _get_bayesian_search_spaces(self) -> Dict[str, Dict]:
        """Get search spaces for Bayesian optimization"""
        return {
            'random_forest': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(10, 50),
                'min_samples_split': Integer(2, 20)
            },
            'xgboost': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 15),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0)
            },
            'lightgbm': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(10, 50),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'num_leaves': Integer(20, 200)
            },
            'extra_trees': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(10, 50),
                'min_samples_split': Integer(2, 20)
            },
            'gradient_boosting': {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(3, 8),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0)
            },
            'ridge_regression': {
                'alpha': Real(0.001, 200.0, prior='log-uniform')
            }
        }
    
    def validate_model_with_detailed_metrics(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Validate model with comprehensive metrics"""
        predictions = self._predict_model(model, X_val)
        
        metrics = {
            'smape': self._calculate_smape(y_val, predictions),
            'mae': mean_absolute_error(y_val, predictions),
            'mse': mean_squared_error(y_val, predictions),
            'rmse': np.sqrt(mean_squared_error(y_val, predictions)),
            'r2': r2_score(y_val, predictions),
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions)
        }
        
        # Add per-quantile SMAPE
        quantile_smape = self.calculate_per_quantile_smape(y_val, predictions)
        metrics.update(quantile_smape)
        
        return metrics
    
    def _predict_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with model (handles different model types)"""
        if isinstance(model, SimpleNeuralNetwork):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = model(X_tensor).numpy().flatten()
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def report_cv_results_with_statistics(self, cv_scores: List[float]) -> Dict:
        """Report cross-validation results with comprehensive statistics"""
        cv_scores = np.array(cv_scores)
        
        results = {
            'mean_smape': np.mean(cv_scores),
            'std_smape': np.std(cv_scores),
            'min_smape': np.min(cv_scores),
            'max_smape': np.max(cv_scores),
            'median_smape': np.median(cv_scores),
            'cv_scores': cv_scores.tolist(),
            'confidence_interval_95': {
                'lower': np.percentile(cv_scores, 2.5),
                'upper': np.percentile(cv_scores, 97.5)
            }
        }
        
        self.logger.info(f"CV Results - Mean SMAPE: {results['mean_smape']:.4f} ± {results['std_smape']:.4f}")
        
        return results
    
    def calculate_per_quantile_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate SMAPE per price quantile"""
        n_quantiles = self.config.evaluation.price_quantiles
        quantiles = np.linspace(0, 1, n_quantiles + 1)
        
        quantile_results = {}
        
        for i in range(n_quantiles):
            lower_bound = np.percentile(y_true, quantiles[i] * 100)
            upper_bound = np.percentile(y_true, quantiles[i + 1] * 100)
            
            mask = (y_true >= lower_bound) & (y_true <= upper_bound)
            
            if np.sum(mask) > 0:
                quantile_smape = self._calculate_smape(y_true[mask], y_pred[mask])
                quantile_results[f'quantile_{i+1}_smape'] = quantile_smape
                quantile_results[f'quantile_{i+1}_count'] = np.sum(mask)
                quantile_results[f'quantile_{i+1}_range'] = f"{lower_bound:.2f}-{upper_bound:.2f}"
        
        return quantile_results
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        epsilon = self.config.evaluation.smape_epsilon
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        denominator = np.maximum(denominator, epsilon)  # Avoid division by zero
        
        smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
        return smape
    
    def save_model(self, model: Any, model_path: str) -> None:
        """Save trained model to disk"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(model, SimpleNeuralNetwork):
            torch.save(model.state_dict(), model_path)
            self.logger.info(f"Neural network model saved to {model_path}")
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Model saved to {model_path}")
        
        # Save model metadata
        metadata = {
            'model_type': type(model).__name__,
            'saved_at': datetime.now().isoformat(),
            'model_path': str(model_path),
            'config': self.config.model.__dict__
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_model(self, model_path: str, model_type: str = None) -> Any:
        """Load trained model from disk"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load metadata if available
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model_type = model_type or metadata.get('model_type')
        
        if model_type == 'SimpleNeuralNetwork':
            # Would need to reconstruct the model architecture
            # This is a simplified version
            model = SimpleNeuralNetwork(
                input_dim=1000,  # Would need to store this in metadata
                hidden_layers=self.config.model.nn_hidden_layers
            )
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        self.logger.info(f"Model loaded from {model_path}")
        return model
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all configured model types"""
        self.logger.info("Starting training for all configured models")
        
        # Set random seeds
        self.set_random_seeds(self.config.model.random_seed)
        
        # Capture experiment metadata
        experiment_id = self.capture_experiment_metadata(
            config=self.config.model.__dict__,
            cv_folds=self.config.model.cv_folds,
            seed=self.config.model.random_seed
        )
        
        trained_models = {}
        
        # Hyperparameter tuning (if enabled)
        best_params = {}
        if self.config.model.use_hyperparameter_tuning:
            best_params = self.tune_hyperparameters(X, y)
        
        # Train each model type
        for model_type in self.config.model.model_types:
            try:
                self.logger.info(f"Training {model_type} model")
                
                # Get model configuration
                model_config = {'model_type': model_type}
                if model_type in best_params:
                    model_config.update(best_params[model_type])
                
                # Train model
                model = self.train_model(X, y, model_config)
                
                if model is not None:
                    trained_models[model_type] = model
                    
                    # Save model
                    model_filename = f"{experiment_id}_{model_type}_model.pkl"
                    if model_type == 'neural_network':
                        model_filename = f"{experiment_id}_{model_type}_model.pth"
                    
                    model_path = self.models_dir / model_filename
                    self.save_model(model, model_path)
                    
                    self.logger.info(f"{model_type} model trained and saved successfully")
                else:
                    self.logger.error(f"Failed to train {model_type} model")
            
            except Exception as e:
                self.logger.error(f"Error training {model_type} model: {str(e)}")
                continue
        
        self.trained_models = trained_models
        self.logger.info(f"Training completed. {len(trained_models)} models trained successfully.")
        
        return trained_models