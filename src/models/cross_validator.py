"""
Cross-validation and reporting implementation for ML Product Pricing Challenge 2025

This module implements detailed k-fold cross-validation with comprehensive SMAPE calculation,
mean ± std SMAPE reporting across folds with per-price-quantile analysis, and holdout validation.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import MLPricingConfig


class CrossValidator:
    """
    Detailed cross-validation and reporting system
    
    Implements k-fold cross-validation with comprehensive SMAPE calculation,
    detailed reporting with statistics, and holdout validation.
    """
    
    def __init__(self, config: MLPricingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.cv_results = {}
        
        # Create necessary directories
        self.logs_dir = Path(config.infrastructure.log_dir)
        self.plots_dir = self.logs_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for cross-validation"""
        logger = logging.getLogger(f"{__name__}.CrossValidator")
        logger.setLevel(getattr(logging, self.config.infrastructure.log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            log_file = self.logs_dir / "cross_validation.log"
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
    
    def perform_kfold_cv(self, X: np.ndarray, y: np.ndarray, model_trainer, 
                        model_configs: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Perform k-fold cross-validation with comprehensive SMAPE calculation
        
        Args:
            X: Feature matrix
            y: Target values
            model_trainer: ModelTrainer instance
            model_configs: Dictionary of model configurations
            
        Returns:
            Dictionary containing CV results for each model
        """
        self.logger.info(f"Starting {self.config.model.cv_folds}-fold cross-validation")
        
        # Create stratified folds based on price quantiles if enabled
        if self.config.evaluation.stratify_by_price_quantiles:
            cv_splitter = self._create_stratified_cv(y)
        else:
            cv_splitter = KFold(
                n_splits=self.config.model.cv_folds,
                shuffle=True,
                random_state=self.config.model.random_seed
            )
        
        cv_results = {}
        
        for model_name, model_config in model_configs.items():
            self.logger.info(f"Cross-validating {model_name}")
            
            fold_results = []
            fold_predictions = []
            fold_actuals = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                self.logger.debug(f"Processing fold {fold_idx + 1}/{self.config.model.cv_folds}")
                
                # Split data
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Train model
                model = model_trainer.train_model(X_train_fold, y_train_fold, model_config)
                
                # Validate model
                fold_metrics = model_trainer.validate_model_with_detailed_metrics(
                    model, X_val_fold, y_val_fold
                )
                
                # Store predictions for ensemble analysis
                predictions = model_trainer._predict_model(model, X_val_fold)
                fold_predictions.extend(predictions)
                fold_actuals.extend(y_val_fold)
                
                fold_results.append(fold_metrics)
                
                self.logger.debug(f"Fold {fold_idx + 1} SMAPE: {fold_metrics['smape']:.4f}")
            
            # Aggregate results across folds
            cv_results[model_name] = self._aggregate_cv_results(fold_results, fold_predictions, fold_actuals)
            
            self.logger.info(f"{model_name} CV completed - Mean SMAPE: {cv_results[model_name]['mean_smape']:.4f} ± {cv_results[model_name]['std_smape']:.4f}")
        
        # Save CV results
        self._save_cv_results(cv_results)
        
        # Generate comparison report
        self._generate_cv_comparison_report(cv_results)
        
        self.cv_results = cv_results
        return cv_results
    
    def _create_stratified_cv(self, y: np.ndarray) -> StratifiedKFold:
        """Create stratified CV based on price quantiles"""
        n_quantiles = self.config.evaluation.price_quantiles
        quantiles = np.linspace(0, 1, n_quantiles + 1)
        
        # Create quantile labels
        quantile_labels = np.zeros(len(y), dtype=int)
        for i in range(n_quantiles):
            lower_bound = np.percentile(y, quantiles[i] * 100)
            upper_bound = np.percentile(y, quantiles[i + 1] * 100)
            
            if i == n_quantiles - 1:  # Last quantile includes upper bound
                mask = (y >= lower_bound) & (y <= upper_bound)
            else:
                mask = (y >= lower_bound) & (y < upper_bound)
            
            quantile_labels[mask] = i
        
        return StratifiedKFold(
            n_splits=self.config.model.cv_folds,
            shuffle=True,
            random_state=self.config.model.random_seed
        )
    
    def _aggregate_cv_results(self, fold_results: List[Dict], 
                            fold_predictions: List[float], 
                            fold_actuals: List[float]) -> Dict:
        """Aggregate results across CV folds"""
        
        # Extract metrics from each fold
        metrics_by_fold = {}
        for metric_name in fold_results[0].keys():
            if metric_name.startswith('quantile_') and metric_name.endswith('_count'):
                continue  # Skip count metrics for aggregation
            if metric_name.startswith('quantile_') and metric_name.endswith('_range'):
                continue  # Skip range metrics for aggregation
            
            values = [fold_result[metric_name] for fold_result in fold_results 
                     if metric_name in fold_result and isinstance(fold_result[metric_name], (int, float))]
            
            if values:
                metrics_by_fold[metric_name] = values
        
        # Calculate statistics
        aggregated_results = {}
        for metric_name, values in metrics_by_fold.items():
            values = np.array(values)
            aggregated_results[f'mean_{metric_name}'] = np.mean(values)
            aggregated_results[f'std_{metric_name}'] = np.std(values)
            aggregated_results[f'min_{metric_name}'] = np.min(values)
            aggregated_results[f'max_{metric_name}'] = np.max(values)
            aggregated_results[f'median_{metric_name}'] = np.median(values)
            aggregated_results[f'{metric_name}_folds'] = values.tolist()
        
        # Calculate overall metrics on all predictions
        fold_predictions = np.array(fold_predictions)
        fold_actuals = np.array(fold_actuals)
        
        aggregated_results['overall_smape'] = self._calculate_smape(fold_actuals, fold_predictions)
        aggregated_results['overall_mae'] = mean_absolute_error(fold_actuals, fold_predictions)
        aggregated_results['overall_mse'] = mean_squared_error(fold_actuals, fold_predictions)
        aggregated_results['overall_rmse'] = np.sqrt(mean_squared_error(fold_actuals, fold_predictions))
        aggregated_results['overall_r2'] = r2_score(fold_actuals, fold_predictions)
        
        # Calculate per-quantile SMAPE on overall predictions
        quantile_smape = self._calculate_per_quantile_smape(fold_actuals, fold_predictions)
        aggregated_results.update(quantile_smape)
        
        # Calculate confidence intervals
        if 'smape' in metrics_by_fold:
            smape_values = np.array(metrics_by_fold['smape'])
            aggregated_results['smape_confidence_interval_95'] = {
                'lower': np.percentile(smape_values, 2.5),
                'upper': np.percentile(smape_values, 97.5)
            }
        
        return aggregated_results
    
    def _calculate_per_quantile_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate SMAPE per price quantile"""
        n_quantiles = self.config.evaluation.price_quantiles
        quantiles = np.linspace(0, 1, n_quantiles + 1)
        
        quantile_results = {}
        
        for i in range(n_quantiles):
            lower_bound = np.percentile(y_true, quantiles[i] * 100)
            upper_bound = np.percentile(y_true, quantiles[i + 1] * 100)
            
            if i == n_quantiles - 1:  # Last quantile includes upper bound
                mask = (y_true >= lower_bound) & (y_true <= upper_bound)
            else:
                mask = (y_true >= lower_bound) & (y_true < upper_bound)
            
            if np.sum(mask) > 0:
                quantile_smape = self._calculate_smape(y_true[mask], y_pred[mask])
                quantile_results[f'overall_quantile_{i+1}_smape'] = quantile_smape
                quantile_results[f'overall_quantile_{i+1}_count'] = np.sum(mask)
                quantile_results[f'overall_quantile_{i+1}_range'] = f"{lower_bound:.2f}-{upper_bound:.2f}"
        
        return quantile_results
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        epsilon = self.config.evaluation.smape_epsilon
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        denominator = np.maximum(denominator, epsilon)  # Avoid division by zero
        
        smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
        return smape
    
    def perform_holdout_validation(self, X: np.ndarray, y: np.ndarray, model_trainer,
                                 model_configs: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Perform holdout validation mimicking test set structure
        
        Args:
            X: Feature matrix
            y: Target values
            model_trainer: ModelTrainer instance
            model_configs: Dictionary of model configurations
            
        Returns:
            Dictionary containing holdout validation results
        """
        self.logger.info("Starting holdout validation")
        
        # Split data
        test_size = self.config.evaluation.validation_split
        
        if self.config.evaluation.stratify_by_price_quantiles:
            # Create stratified split based on price quantiles
            n_quantiles = self.config.evaluation.price_quantiles
            quantiles = np.linspace(0, 1, n_quantiles + 1)
            
            quantile_labels = np.zeros(len(y), dtype=int)
            for i in range(n_quantiles):
                lower_bound = np.percentile(y, quantiles[i] * 100)
                upper_bound = np.percentile(y, quantiles[i + 1] * 100)
                
                if i == n_quantiles - 1:
                    mask = (y >= lower_bound) & (y <= upper_bound)
                else:
                    mask = (y >= lower_bound) & (y < upper_bound)
                
                quantile_labels[mask] = i
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, stratify=quantile_labels,
                random_state=self.config.model.random_seed
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=self.config.model.random_seed
            )
        
        self.logger.info(f"Holdout split: {len(X_train)} train, {len(X_val)} validation samples")
        
        holdout_results = {}
        
        for model_name, model_config in model_configs.items():
            self.logger.info(f"Holdout validation for {model_name}")
            
            # Train model
            model = model_trainer.train_model(X_train, y_train, model_config)
            
            # Validate model
            metrics = model_trainer.validate_model_with_detailed_metrics(model, X_val, y_val)
            
            holdout_results[model_name] = metrics
            
            self.logger.info(f"{model_name} holdout SMAPE: {metrics['smape']:.4f}")
        
        # Save holdout results
        self._save_holdout_results(holdout_results)
        
        return holdout_results
    
    def create_detailed_performance_tracking(self, cv_results: Dict[str, Dict],
                                           holdout_results: Dict[str, Dict] = None) -> Dict:
        """Create detailed model performance tracking and comparison utilities"""
        
        performance_summary = {
            'timestamp': datetime.now().isoformat(),
            'cv_summary': {},
            'model_comparison': {},
            'best_models': {}
        }
        
        # Summarize CV results
        for model_name, results in cv_results.items():
            performance_summary['cv_summary'][model_name] = {
                'mean_smape': results.get('mean_smape', 0),
                'std_smape': results.get('std_smape', 0),
                'overall_smape': results.get('overall_smape', 0),
                'confidence_interval': results.get('smape_confidence_interval_95', {}),
                'mean_r2': results.get('mean_r2', 0),
                'overall_r2': results.get('overall_r2', 0)
            }
        
        # Add holdout results if available
        if holdout_results:
            performance_summary['holdout_summary'] = {}
            for model_name, results in holdout_results.items():
                performance_summary['holdout_summary'][model_name] = {
                    'smape': results.get('smape', 0),
                    'r2': results.get('r2', 0),
                    'mae': results.get('mae', 0),
                    'rmse': results.get('rmse', 0)
                }
        
        # Model comparison
        model_names = list(cv_results.keys())
        comparison_matrix = {}
        
        for i, model1 in enumerate(model_names):
            comparison_matrix[model1] = {}
            for j, model2 in enumerate(model_names):
                if i != j:
                    smape1 = cv_results[model1].get('mean_smape', float('inf'))
                    smape2 = cv_results[model2].get('mean_smape', float('inf'))
                    comparison_matrix[model1][model2] = {
                        'smape_difference': smape1 - smape2,
                        'better': smape1 < smape2
                    }
        
        performance_summary['model_comparison'] = comparison_matrix
        
        # Identify best models
        best_cv_model = min(cv_results.keys(), 
                           key=lambda x: cv_results[x].get('mean_smape', float('inf')))
        performance_summary['best_models']['cv_best'] = best_cv_model
        
        if holdout_results:
            best_holdout_model = min(holdout_results.keys(),
                                   key=lambda x: holdout_results[x].get('smape', float('inf')))
            performance_summary['best_models']['holdout_best'] = best_holdout_model
        
        # Save performance summary
        summary_file = self.logs_dir / "performance_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(performance_summary, f, indent=2, default=str)
        
        self.logger.info(f"Performance summary saved to {summary_file}")
        
        return performance_summary
    
    def _save_cv_results(self, cv_results: Dict[str, Dict]):
        """Save cross-validation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.logs_dir / f"cv_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        
        self.logger.info(f"CV results saved to {results_file}")
    
    def _save_holdout_results(self, holdout_results: Dict[str, Dict]):
        """Save holdout validation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.logs_dir / f"holdout_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(holdout_results, f, indent=2, default=str)
        
        self.logger.info(f"Holdout results saved to {results_file}")
    
    def _generate_cv_comparison_report(self, cv_results: Dict[str, Dict]):
        """Generate comprehensive CV comparison report with visualizations"""
        if not self.config.evaluation.generate_plots:
            return
        
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Cross-Validation Results Comparison', fontsize=16)
            
            # Extract data for plotting
            model_names = list(cv_results.keys())
            mean_smapes = [cv_results[model]['mean_smape'] for model in model_names]
            std_smapes = [cv_results[model]['std_smape'] for model in model_names]
            
            # 1. Mean SMAPE comparison with error bars
            axes[0, 0].bar(model_names, mean_smapes, yerr=std_smapes, capsize=5)
            axes[0, 0].set_title('Mean SMAPE by Model')
            axes[0, 0].set_ylabel('SMAPE')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. SMAPE distribution across folds
            smape_data = []
            labels = []
            for model in model_names:
                if 'smape_folds' in cv_results[model]:
                    smape_data.extend(cv_results[model]['smape_folds'])
                    labels.extend([model] * len(cv_results[model]['smape_folds']))
            
            if smape_data:
                df_smape = pd.DataFrame({'Model': labels, 'SMAPE': smape_data})
                sns.boxplot(data=df_smape, x='Model', y='SMAPE', ax=axes[0, 1])
                axes[0, 1].set_title('SMAPE Distribution Across Folds')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. R² comparison
            if all('mean_r2' in cv_results[model] for model in model_names):
                mean_r2s = [cv_results[model]['mean_r2'] for model in model_names]
                std_r2s = [cv_results[model]['std_r2'] for model in model_names]
                
                axes[1, 0].bar(model_names, mean_r2s, yerr=std_r2s, capsize=5, color='green', alpha=0.7)
                axes[1, 0].set_title('Mean R² by Model')
                axes[1, 0].set_ylabel('R²')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Quantile SMAPE comparison (if available)
            quantile_data = {}
            for model in model_names:
                quantile_data[model] = []
                for key, value in cv_results[model].items():
                    if key.startswith('overall_quantile_') and key.endswith('_smape'):
                        quantile_data[model].append(value)
            
            if any(quantile_data.values()):
                x_pos = np.arange(len(model_names))
                width = 0.15
                n_quantiles = max(len(values) for values in quantile_data.values())
                
                for i in range(n_quantiles):
                    quantile_smapes = [quantile_data[model][i] if i < len(quantile_data[model]) else 0 
                                     for model in model_names]
                    axes[1, 1].bar(x_pos + i * width, quantile_smapes, width, 
                                  label=f'Quantile {i+1}', alpha=0.8)
                
                axes[1, 1].set_title('SMAPE by Price Quantile')
                axes[1, 1].set_ylabel('SMAPE')
                axes[1, 1].set_xticks(x_pos + width * (n_quantiles - 1) / 2)
                axes[1, 1].set_xticklabels(model_names, rotation=45)
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.plots_dir / f"cv_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.evaluation.plot_format}"
            plt.savefig(plot_file, dpi=self.config.evaluation.plot_dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"CV comparison plot saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating CV comparison plot: {str(e)}")
    
    def generate_detailed_report(self, cv_results: Dict[str, Dict], 
                               holdout_results: Dict[str, Dict] = None) -> str:
        """Generate detailed text report of validation results"""
        
        report_lines = [
            "=" * 80,
            "DETAILED CROSS-VALIDATION AND HOLDOUT VALIDATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"CV Folds: {self.config.model.cv_folds}",
            f"Random Seed: {self.config.model.random_seed}",
            ""
        ]
        
        # CV Results Summary
        report_lines.extend([
            "CROSS-VALIDATION RESULTS",
            "-" * 40
        ])
        
        for model_name, results in cv_results.items():
            report_lines.extend([
                f"\n{model_name.upper()}:",
                f"  Mean SMAPE: {results.get('mean_smape', 0):.4f} ± {results.get('std_smape', 0):.4f}",
                f"  Overall SMAPE: {results.get('overall_smape', 0):.4f}",
                f"  Mean R²: {results.get('mean_r2', 0):.4f} ± {results.get('std_r2', 0):.4f}",
                f"  Overall R²: {results.get('overall_r2', 0):.4f}"
            ])
            
            # Add confidence interval if available
            if 'smape_confidence_interval_95' in results:
                ci = results['smape_confidence_interval_95']
                report_lines.append(f"  95% CI: [{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}]")
            
            # Add quantile analysis
            quantile_info = []
            for key, value in results.items():
                if key.startswith('overall_quantile_') and key.endswith('_smape'):
                    quantile_num = key.split('_')[2]
                    range_key = f'overall_quantile_{quantile_num}_range'
                    count_key = f'overall_quantile_{quantile_num}_count'
                    
                    range_info = results.get(range_key, 'N/A')
                    count_info = results.get(count_key, 0)
                    
                    quantile_info.append(f"    Q{quantile_num}: {value:.4f} (range: {range_info}, n={count_info})")
            
            if quantile_info:
                report_lines.append("  Quantile SMAPE:")
                report_lines.extend(quantile_info)
        
        # Holdout Results
        if holdout_results:
            report_lines.extend([
                "\n\nHOLDOUT VALIDATION RESULTS",
                "-" * 40
            ])
            
            for model_name, results in holdout_results.items():
                report_lines.extend([
                    f"\n{model_name.upper()}:",
                    f"  SMAPE: {results.get('smape', 0):.4f}",
                    f"  R²: {results.get('r2', 0):.4f}",
                    f"  MAE: {results.get('mae', 0):.4f}",
                    f"  RMSE: {results.get('rmse', 0):.4f}"
                ])
        
        # Model Ranking
        report_lines.extend([
            "\n\nMODEL RANKING (by CV SMAPE)",
            "-" * 40
        ])
        
        sorted_models = sorted(cv_results.items(), key=lambda x: x[1].get('mean_smape', float('inf')))
        
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            smape = results.get('mean_smape', 0)
            std_smape = results.get('std_smape', 0)
            report_lines.append(f"{rank}. {model_name}: {smape:.4f} ± {std_smape:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.logs_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Detailed validation report saved to {report_file}")
        
        return report_content