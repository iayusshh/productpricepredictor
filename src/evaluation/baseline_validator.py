"""
Baseline model validation framework.

Provides baseline model comparisons and performance analysis tools
for validating model performance against simple baselines.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from pathlib import Path
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .smape_calculator import SMAPECalculator


class BaselineValidator:
    """
    Baseline model validation framework.
    
    Provides baseline model comparisons (mean, median, simple regression)
    and performance analysis tools for model validation.
    """
    
    def __init__(self, output_dir: str = "logs/baseline_validation"):
        """
        Initialize baseline validator.
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.smape_calculator = SMAPECalculator(log_performance=False)
        
        # Store baseline models and results
        self.baseline_models = {}
        self.baseline_results = {}
    
    def create_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Create and train baseline models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained baseline models
        """
        self.logger.info("Creating baseline models...")
        
        # Simple statistical baselines
        mean_baseline = np.mean(y_train)
        median_baseline = np.median(y_train)
        
        # Simple linear regression baseline (only if features available)
        linear_baseline = None
        if X_train.shape[1] > 0:
            linear_baseline = LinearRegression()
            linear_baseline.fit(X_train, y_train)
        
        # Feature-based baselines (if features available)
        baselines = {
            'mean': {
                'model': lambda X: np.full(len(X), mean_baseline),
                'type': 'statistical',
                'description': f'Constant prediction using training mean: {mean_baseline:.4f}'
            },
            'median': {
                'model': lambda X: np.full(len(X), median_baseline),
                'type': 'statistical', 
                'description': f'Constant prediction using training median: {median_baseline:.4f}'
            }
        }
        
        # Add linear regression only if features are available
        if linear_baseline is not None:
            baselines['linear_regression'] = {
                'model': linear_baseline,
                'type': 'regression',
                'description': 'Simple linear regression on all features'
            }
        
        # Add feature-specific baselines if we have meaningful features
        if X_train.shape[1] > 0:
            # Single feature baselines (using first few features)
            n_features_to_test = min(3, X_train.shape[1])
            
            for i in range(n_features_to_test):
                single_feature_model = LinearRegression()
                single_feature_model.fit(X_train[:, i:i+1], y_train)
                
                baselines[f'single_feature_{i}'] = {
                    'model': single_feature_model,
                    'type': 'regression',
                    'description': f'Linear regression using only feature {i}'
                }
        
        self.baseline_models = baselines
        self.logger.info(f"Created {len(baselines)} baseline models")
        
        return baselines
    
    def evaluate_baselines(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all baseline models on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics for each baseline
        """
        if not self.baseline_models:
            raise ValueError("No baseline models created. Call create_baseline_models first.")
        
        self.logger.info("Evaluating baseline models...")
        
        results = {}
        
        for name, baseline_info in self.baseline_models.items():
            try:
                model = baseline_info['model']
                
                # Generate predictions
                if callable(model) and not hasattr(model, 'predict'):
                    # Statistical baseline (lambda function)
                    y_pred = model(X_test)
                elif hasattr(model, 'predict'):
                    # Sklearn model
                    if name.startswith('single_feature_'):
                        feature_idx = int(name.split('_')[-1])
                        y_pred = model.predict(X_test[:, feature_idx:feature_idx+1])
                    else:
                        y_pred = model.predict(X_test)
                else:
                    self.logger.error(f"Unknown model type for {name}")
                    continue
                
                # Calculate metrics
                metrics = self._calculate_baseline_metrics(y_test, y_pred)
                metrics['model_type'] = baseline_info['type']
                metrics['description'] = baseline_info['description']
                
                results[name] = metrics
                
                self.logger.info(f"{name}: SMAPE={metrics['smape']:.4f}, R²={metrics['r2']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating baseline {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.baseline_results = results
        return results
    
    def cross_validate_baselines(self, X: np.ndarray, y: np.ndarray, 
                                cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Perform cross-validation on baseline models.
        
        Args:
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results for each baseline
        """
        if not self.baseline_models:
            raise ValueError("No baseline models created. Call create_baseline_models first.")
        
        self.logger.info(f"Cross-validating baseline models with {cv_folds} folds...")
        
        cv_results = {}
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, baseline_info in self.baseline_models.items():
            try:
                model = baseline_info['model']
                
                if name in ['mean', 'median']:
                    # Statistical baselines - manual CV
                    cv_scores = self._cross_validate_statistical_baseline(
                        X, y, name, kfold
                    )
                elif hasattr(model, 'predict'):
                    # Sklearn models
                    if name.startswith('single_feature_'):
                        feature_idx = int(name.split('_')[-1])
                        X_single = X[:, feature_idx:feature_idx+1]
                        cv_scores = self._cross_validate_sklearn_model(
                            X_single, y, model, kfold
                        )
                    else:
                        cv_scores = self._cross_validate_sklearn_model(
                            X, y, model, kfold
                        )
                else:
                    self.logger.warning(f"Cannot cross-validate {name}")
                    continue
                
                cv_results[name] = {
                    'smape_scores': cv_scores['smape'],
                    'smape_mean': np.mean(cv_scores['smape']),
                    'smape_std': np.std(cv_scores['smape']),
                    'r2_scores': cv_scores['r2'],
                    'r2_mean': np.mean(cv_scores['r2']),
                    'r2_std': np.std(cv_scores['r2']),
                    'model_type': baseline_info['type'],
                    'description': baseline_info['description']
                }
                
                self.logger.info(
                    f"{name} CV: SMAPE={cv_results[name]['smape_mean']:.4f}±{cv_results[name]['smape_std']:.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in CV for baseline {name}: {str(e)}")
                cv_results[name] = {'error': str(e)}
        
        return cv_results
    
    def compare_with_model(self, model_results: Dict[str, float], 
                          model_name: str = "target_model") -> Dict[str, Any]:
        """
        Compare target model performance with baselines.
        
        Args:
            model_results: Dictionary with model metrics (must include 'smape', 'r2')
            model_name: Name of the target model
            
        Returns:
            Dictionary with comparison analysis
        """
        if not self.baseline_results:
            raise ValueError("No baseline results available. Run evaluate_baselines first.")
        
        self.logger.info(f"Comparing {model_name} with baseline models...")
        
        comparison = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'model_metrics': model_results,
            'baseline_comparison': {},
            'performance_ranking': {},
            'improvement_analysis': {}
        }
        
        # Extract metrics for comparison
        model_smape = model_results.get('smape', float('inf'))
        model_r2 = model_results.get('r2', -float('inf'))
        
        # Compare with each baseline
        smape_improvements = {}
        r2_improvements = {}
        
        for baseline_name, baseline_metrics in self.baseline_results.items():
            if 'error' in baseline_metrics:
                continue
                
            baseline_smape = baseline_metrics.get('smape', float('inf'))
            baseline_r2 = baseline_metrics.get('r2', -float('inf'))
            
            # Calculate improvements (negative means worse performance)
            smape_improvement = baseline_smape - model_smape  # Positive is better (lower SMAPE)
            r2_improvement = model_r2 - baseline_r2  # Positive is better (higher R²)
            
            smape_improvements[baseline_name] = smape_improvement
            r2_improvements[baseline_name] = r2_improvement
            
            comparison['baseline_comparison'][baseline_name] = {
                'baseline_smape': baseline_smape,
                'baseline_r2': baseline_r2,
                'smape_improvement': smape_improvement,
                'r2_improvement': r2_improvement,
                'smape_improvement_pct': (smape_improvement / baseline_smape * 100) if baseline_smape > 0 else 0,
                'r2_improvement_pct': (r2_improvement / abs(baseline_r2) * 100) if baseline_r2 != 0 else 0,
                'better_than_baseline': bool(smape_improvement > 0 and r2_improvement > 0)
            }
        
        # Performance ranking
        all_models = {model_name: {'smape': model_smape, 'r2': model_r2}}
        for name, metrics in self.baseline_results.items():
            if 'error' not in metrics:
                all_models[name] = {'smape': metrics['smape'], 'r2': metrics['r2']}
        
        # Rank by SMAPE (lower is better)
        smape_ranking = sorted(all_models.items(), key=lambda x: x[1]['smape'])
        r2_ranking = sorted(all_models.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        comparison['performance_ranking'] = {
            'smape_ranking': [(name, metrics['smape']) for name, metrics in smape_ranking],
            'r2_ranking': [(name, metrics['r2']) for name, metrics in r2_ranking],
            'model_smape_rank': next(i for i, (name, _) in enumerate(smape_ranking, 1) if name == model_name),
            'model_r2_rank': next(i for i, (name, _) in enumerate(r2_ranking, 1) if name == model_name)
        }
        
        # Improvement analysis
        comparison['improvement_analysis'] = {
            'beats_all_baselines': bool(all(comp['better_than_baseline'] 
                                          for comp in comparison['baseline_comparison'].values())),
            'best_smape_improvement': max(smape_improvements.values()) if smape_improvements else 0,
            'best_r2_improvement': max(r2_improvements.values()) if r2_improvements else 0,
            'worst_smape_improvement': min(smape_improvements.values()) if smape_improvements else 0,
            'worst_r2_improvement': min(r2_improvements.values()) if r2_improvements else 0,
            'avg_smape_improvement': np.mean(list(smape_improvements.values())) if smape_improvements else 0,
            'avg_r2_improvement': np.mean(list(r2_improvements.values())) if r2_improvements else 0
        }
        
        # Save comparison report
        comparison_path = self.output_dir / f"{model_name}_baseline_comparison.json"
        with open(comparison_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_comparison = self._convert_numpy_types(comparison)
            json.dump(json_comparison, f, indent=2)
        
        self.logger.info(f"Baseline comparison saved to {comparison_path}")
        return comparison
    
    def validate_model_consistency(self, cv_results: Dict[str, List[float]], 
                                 model_name: str = "target_model") -> Dict[str, Any]:
        """
        Validate model performance consistency across validation folds.
        
        Args:
            cv_results: Cross-validation results with lists of scores per fold
            model_name: Name of the model being validated
            
        Returns:
            Dictionary with consistency analysis
        """
        self.logger.info(f"Validating performance consistency for {model_name}")
        
        consistency_analysis = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'consistency_metrics': {},
            'stability_assessment': {},
            'outlier_detection': {}
        }
        
        for metric_name, scores in cv_results.items():
            if not isinstance(scores, (list, np.ndarray)) or len(scores) == 0:
                continue
                
            scores = np.array(scores)
            
            # Basic statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv_coefficient = std_score / abs(mean_score) if mean_score != 0 else float('inf')
            
            # Consistency metrics
            consistency_analysis['consistency_metrics'][metric_name] = {
                'mean': float(mean_score),
                'std': float(std_score),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'range': float(np.max(scores) - np.min(scores)),
                'coefficient_of_variation': float(cv_coefficient),
                'median': float(np.median(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75)),
                'iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25))
            }
            
            # Stability assessment
            is_stable = cv_coefficient < 0.1  # Less than 10% variation
            is_moderately_stable = cv_coefficient < 0.2  # Less than 20% variation
            
            consistency_analysis['stability_assessment'][metric_name] = {
                'is_stable': bool(is_stable),
                'is_moderately_stable': bool(is_moderately_stable),
                'stability_level': 'stable' if is_stable else 'moderate' if is_moderately_stable else 'unstable',
                'coefficient_of_variation': float(cv_coefficient)
            }
            
            # Outlier detection using IQR method
            q1, q3 = np.percentile(scores, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = scores[(scores < lower_bound) | (scores > upper_bound)]
            
            consistency_analysis['outlier_detection'][metric_name] = {
                'outlier_count': len(outliers),
                'outlier_percentage': float(len(outliers) / len(scores) * 100),
                'outlier_values': outliers.tolist(),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        
        # Overall consistency assessment
        smape_stable = consistency_analysis['stability_assessment'].get('smape', {}).get('is_stable', False)
        r2_stable = consistency_analysis['stability_assessment'].get('r2', {}).get('is_stable', False)
        
        consistency_analysis['overall_assessment'] = {
            'is_consistent': bool(smape_stable and r2_stable),
            'primary_concerns': [],
            'recommendations': []
        }
        
        # Add concerns and recommendations
        if not smape_stable:
            consistency_analysis['overall_assessment']['primary_concerns'].append(
                "SMAPE shows high variability across folds"
            )
            consistency_analysis['overall_assessment']['recommendations'].append(
                "Consider increasing regularization or using more stable model architecture"
            )
        
        if not r2_stable:
            consistency_analysis['overall_assessment']['primary_concerns'].append(
                "R² shows high variability across folds"
            )
            consistency_analysis['overall_assessment']['recommendations'].append(
                "Model may be overfitting - consider feature selection or regularization"
            )
        
        # Save consistency report
        consistency_path = self.output_dir / f"{model_name}_consistency_analysis.json"
        with open(consistency_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_consistency = self._convert_numpy_types(consistency_analysis)
            json.dump(json_consistency, f, indent=2)
        
        self.logger.info(f"Consistency analysis saved to {consistency_path}")
        return consistency_analysis
    
    def _calculate_baseline_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics for baseline evaluation."""
        metrics = {}
        
        # SMAPE
        metrics['smape'] = self.smape_calculator.calculate_smape(y_true, y_pred)
        
        # Standard regression metrics
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        
        # MAPE (if no zeros in y_true)
        mask = y_true != 0
        if np.sum(mask) > 0:
            mape = 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = None
        
        return metrics
    
    def _cross_validate_statistical_baseline(self, X: np.ndarray, y: np.ndarray, 
                                           baseline_name: str, kfold) -> Dict[str, List[float]]:
        """Cross-validate statistical baselines (mean, median)."""
        smape_scores = []
        r2_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            if baseline_name == 'mean':
                prediction = np.full(len(y_val_fold), np.mean(y_train_fold))
            elif baseline_name == 'median':
                prediction = np.full(len(y_val_fold), np.median(y_train_fold))
            
            smape = self.smape_calculator.calculate_smape(y_val_fold, prediction)
            r2 = r2_score(y_val_fold, prediction)
            
            smape_scores.append(smape)
            r2_scores.append(r2)
        
        return {'smape': smape_scores, 'r2': r2_scores}
    
    def _cross_validate_sklearn_model(self, X: np.ndarray, y: np.ndarray, 
                                    model, kfold) -> Dict[str, List[float]]:
        """Cross-validate sklearn models."""
        smape_scores = []
        r2_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create fresh model instance
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            
            prediction = fold_model.predict(X_val_fold)
            
            smape = self.smape_calculator.calculate_smape(y_val_fold, prediction)
            r2 = r2_score(y_val_fold, prediction)
            
            smape_scores.append(smape)
            r2_scores.append(r2)
        
        return {'smape': smape_scores, 'r2': r2_scores}
    
    def generate_baseline_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of baseline validation."""
        if not self.baseline_results:
            raise ValueError("No baseline results available. Run evaluate_baselines first.")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_baselines': len(self.baseline_results),
            'baseline_names': list(self.baseline_results.keys()),
            'best_baseline': {},
            'baseline_statistics': {}
        }
        
        # Find best baseline by SMAPE
        valid_baselines = {name: metrics for name, metrics in self.baseline_results.items() 
                          if 'error' not in metrics}
        
        if valid_baselines:
            best_smape_baseline = min(valid_baselines.items(), key=lambda x: x[1]['smape'])
            best_r2_baseline = max(valid_baselines.items(), key=lambda x: x[1]['r2'])
            
            summary['best_baseline'] = {
                'by_smape': {
                    'name': best_smape_baseline[0],
                    'smape': best_smape_baseline[1]['smape'],
                    'r2': best_smape_baseline[1]['r2']
                },
                'by_r2': {
                    'name': best_r2_baseline[0],
                    'smape': best_r2_baseline[1]['smape'],
                    'r2': best_r2_baseline[1]['r2']
                }
            }
            
            # Baseline statistics
            smape_values = [metrics['smape'] for metrics in valid_baselines.values()]
            r2_values = [metrics['r2'] for metrics in valid_baselines.values()]
            
            summary['baseline_statistics'] = {
                'smape': {
                    'min': float(np.min(smape_values)),
                    'max': float(np.max(smape_values)),
                    'mean': float(np.mean(smape_values)),
                    'std': float(np.std(smape_values))
                },
                'r2': {
                    'min': float(np.min(r2_values)),
                    'max': float(np.max(r2_values)),
                    'mean': float(np.mean(r2_values)),
                    'std': float(np.std(r2_values))
                }
            }
        
        # Save summary
        summary_path = self.output_dir / "baseline_validation_summary.json"
        with open(summary_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_summary = self._convert_numpy_types(summary)
            json.dump(json_summary, f, indent=2)
        
        return summary
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj