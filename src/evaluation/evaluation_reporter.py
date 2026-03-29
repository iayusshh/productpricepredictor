"""
Comprehensive evaluation reporting system.

Provides detailed evaluation reports with distribution plots, residual analysis,
feature importance, and model diagnostic tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import os
from pathlib import Path
import json
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
from .smape_calculator import SMAPECalculator


class EvaluationReporter:
    """
    Comprehensive evaluation reporting system with visualizations and analysis.
    
    Generates detailed reports including distribution plots, residual analysis,
    feature importance, and model diagnostics.
    """
    
    def __init__(self, output_dir: str = "logs/evaluation_reports", 
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize evaluation reporter.
        
        Args:
            output_dir: Directory to save reports and plots
            figsize: Default figure size for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
        self.smape_calculator = SMAPECalculator(log_performance=False)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_comprehensive_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    model: Optional[Any] = None, 
                                    X_test: Optional[np.ndarray] = None,
                                    feature_names: Optional[List[str]] = None,
                                    model_name: str = "model",
                                    save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model: Trained model (for feature importance)
            X_test: Test features (for SHAP analysis)
            feature_names: Names of features
            model_name: Name of the model for reporting
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing all evaluation metrics and analysis
        """
        self.logger.info(f"Generating comprehensive evaluation report for {model_name}")
        
        # Basic validation
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Initialize report
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
            'metrics': {},
            'diagnostics': {},
            'plots_saved': []
        }
        
        # Calculate comprehensive metrics
        report['metrics'] = self._calculate_comprehensive_metrics(y_true, y_pred)
        
        # Generate distribution plots
        if save_plots:
            plot_paths = self._create_distribution_plots(y_true, y_pred, model_name)
            report['plots_saved'].extend(plot_paths)
        
        # Generate residual analysis
        if save_plots:
            residual_paths = self._create_residual_analysis(y_true, y_pred, model_name)
            report['plots_saved'].extend(residual_paths)
        
        # Model diagnostics
        report['diagnostics'] = self._perform_model_diagnostics(y_true, y_pred)
        
        # Feature importance analysis
        if model is not None:
            importance_analysis = self._analyze_feature_importance(
                model, X_test, feature_names, model_name, save_plots
            )
            report['feature_importance'] = importance_analysis
            if save_plots and 'plots' in importance_analysis:
                report['plots_saved'].extend(importance_analysis['plots'])
        
        # SHAP analysis if available
        if SHAP_AVAILABLE and model is not None and X_test is not None:
            shap_analysis = self._generate_shap_analysis(
                model, X_test, feature_names, model_name, save_plots
            )
            report['shap_analysis'] = shap_analysis
            if save_plots and 'plots' in shap_analysis:
                report['plots_saved'].extend(shap_analysis['plots'])
        
        # Save report to JSON
        report_path = self.output_dir / f"{model_name}_evaluation_report.json"
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_report = self._prepare_report_for_json(report.copy())
            json.dump(json_report, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {report_path}")
        return report
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # SMAPE metrics
        smape_details = self.smape_calculator.calculate_smape_with_details(y_true, y_pred)
        metrics['smape'] = smape_details
        
        # Quantile SMAPE
        quantile_smape = self.smape_calculator.calculate_quantile_smape(y_true, y_pred)
        metrics['quantile_smape'] = quantile_smape
        
        # Additional regression metrics
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        
        # Mean Absolute Percentage Error (MAPE)
        mask = y_true != 0
        if np.sum(mask) > 0:
            mape = 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = None
        
        # Prediction statistics
        metrics['prediction_stats'] = {
            'mean': float(np.mean(y_pred)),
            'std': float(np.std(y_pred)),
            'min': float(np.min(y_pred)),
            'max': float(np.max(y_pred)),
            'median': float(np.median(y_pred))
        }
        
        # Actual statistics
        metrics['actual_stats'] = {
            'mean': float(np.mean(y_true)),
            'std': float(np.std(y_true)),
            'min': float(np.min(y_true)),
            'max': float(np.max(y_true)),
            'median': float(np.median(y_true))
        }
        
        return metrics
    
    def _create_distribution_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str) -> List[str]:
        """Create distribution plots (predicted vs actual)."""
        plot_paths = []
        
        # 1. Scatter plot: Predicted vs Actual
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), 'g-', alpha=0.8, label=f'Trend Line (slope={z[0]:.3f})')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name}: Predicted vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{model_name}_predicted_vs_actual.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        # 2. Distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograms
        ax1.hist(y_true, bins=50, alpha=0.7, label='Actual', density=True)
        ax1.hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True)
        ax1.set_xlabel('Values')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{model_name}: Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(y_pred - y_true, dist="norm", plot=ax2)
        ax2.set_title(f'{model_name}: Q-Q Plot of Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{model_name}_distribution_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        return plot_paths
    
    def _create_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str) -> List[str]:
        """Create residual analysis plots."""
        plot_paths = []
        residuals = y_pred - y_true
        
        # Create comprehensive residual analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line for residuals
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        ax1.plot(y_pred, p(y_pred), 'g-', alpha=0.8, 
                label=f'Trend (slope={z[0]:.6f})')
        ax1.legend()
        
        # 2. Residual histogram
        ax2.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Density')
        ax2.set_title('Residual Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add normal distribution overlay
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
                label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        ax2.legend()
        
        # 3. Residuals vs Actual
        ax3.scatter(y_true, residuals, alpha=0.6, s=20)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals vs Actual Values')
        ax3.grid(True, alpha=0.3)
        
        # 4. Absolute residuals vs Predicted (for heteroscedasticity)
        ax4.scatter(y_pred, np.abs(residuals), alpha=0.6, s=20)
        ax4.set_xlabel('Predicted Values')
        ax4.set_ylabel('|Residuals|')
        ax4.set_title('Absolute Residuals vs Predicted')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line for absolute residuals
        z_abs = np.polyfit(y_pred, np.abs(residuals), 1)
        p_abs = np.poly1d(z_abs)
        ax4.plot(y_pred, p_abs(y_pred), 'g-', alpha=0.8,
                label=f'Trend (slope={z_abs[0]:.6f})')
        ax4.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{model_name}_residual_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        return plot_paths
    
    def _perform_model_diagnostics(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive model diagnostics."""
        diagnostics = {}
        residuals = y_pred - y_true
        
        # Residual statistics
        diagnostics['residual_stats'] = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals))
        }
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        diagnostics['normality_test'] = {
            'shapiro_wilk_statistic': float(shapiro_stat),
            'shapiro_wilk_p_value': float(shapiro_p),
            'is_normal': bool(shapiro_p > 0.05)
        }
        
        # Heteroscedasticity test (Breusch-Pagan-like)
        # Simple correlation between |residuals| and predicted values
        abs_residuals = np.abs(residuals)
        hetero_corr = np.corrcoef(y_pred, abs_residuals)[0, 1]
        diagnostics['heteroscedasticity'] = {
            'correlation_abs_residuals_predicted': float(hetero_corr),
            'potential_heteroscedasticity': bool(abs(hetero_corr) > 0.1)
        }
        
        # Outlier detection
        residual_threshold = 3 * np.std(residuals)
        outlier_mask = np.abs(residuals) > residual_threshold
        diagnostics['outliers'] = {
            'count': int(np.sum(outlier_mask)),
            'percentage': float(100 * np.sum(outlier_mask) / len(residuals)),
            'threshold': float(residual_threshold)
        }
        
        # Prediction range analysis
        diagnostics['prediction_range'] = {
            'actual_range': float(np.max(y_true) - np.min(y_true)),
            'predicted_range': float(np.max(y_pred) - np.min(y_pred)),
            'range_ratio': float((np.max(y_pred) - np.min(y_pred)) / (np.max(y_true) - np.min(y_true))),
            'negative_predictions': int(np.sum(y_pred < 0)),
            'zero_predictions': int(np.sum(y_pred == 0))
        }
        
        # Bias analysis
        diagnostics['bias_analysis'] = {
            'mean_bias': float(np.mean(residuals)),
            'median_bias': float(np.median(residuals)),
            'systematic_underestimation': bool(np.mean(residuals) < -0.01 * np.mean(y_true)),
            'systematic_overestimation': bool(np.mean(residuals) > 0.01 * np.mean(y_true))
        }
        
        return diagnostics
    
    def _analyze_feature_importance(self, model: Any, X_test: Optional[np.ndarray],
                                  feature_names: Optional[List[str]], 
                                  model_name: str, save_plots: bool) -> Dict[str, Any]:
        """Analyze feature importance using model-specific methods."""
        importance_analysis = {'plots': []}
        
        try:
            # Try to get feature importance from the model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                self.logger.warning(f"Model {type(model)} doesn't have feature_importances_ or coef_")
                return importance_analysis
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_analysis['feature_importance'] = {
                'top_10_features': importance_df.head(10).to_dict('records'),
                'all_features': importance_df.to_dict('records')
            }
            
            if save_plots:
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(12, 8))
                
                top_features = importance_df.head(20)
                bars = ax.barh(range(len(top_features)), top_features['importance'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{model_name}: Top 20 Feature Importances')
                ax.grid(True, alpha=0.3)
                
                # Color bars by importance
                colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                plt.tight_layout()
                plot_path = self.output_dir / f"{model_name}_feature_importance.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                importance_analysis['plots'].append(str(plot_path))
                
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {str(e)}")
            importance_analysis['error'] = str(e)
        
        return importance_analysis
    
    def _generate_shap_analysis(self, model: Any, X_test: np.ndarray,
                              feature_names: Optional[List[str]], 
                              model_name: str, save_plots: bool) -> Dict[str, Any]:
        """Generate SHAP analysis for model interpretability."""
        shap_analysis = {'plots': []}
        
        if not SHAP_AVAILABLE:
            shap_analysis['error'] = "SHAP not available. Install with: pip install shap"
            return shap_analysis
        
        try:
            # Limit samples for SHAP analysis (computational efficiency)
            max_samples = min(1000, len(X_test))
            X_sample = X_test[:max_samples]
            
            # Create SHAP explainer
            if hasattr(model, 'predict'):
                explainer = shap.Explainer(model, X_sample[:100])  # Use subset as background
                shap_values = explainer(X_sample)
            else:
                self.logger.warning("Model doesn't have predict method for SHAP analysis")
                return shap_analysis
            
            # SHAP summary statistics
            shap_analysis['summary_stats'] = {
                'mean_abs_shap': float(np.mean(np.abs(shap_values.values))),
                'max_shap': float(np.max(np.abs(shap_values.values))),
                'n_samples_analyzed': max_samples
            }
            
            if save_plots:
                # SHAP summary plot
                plt.figure(figsize=self.figsize)
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                show=False, max_display=20)
                plt.title(f'{model_name}: SHAP Summary Plot')
                plot_path = self.output_dir / f"{model_name}_shap_summary.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                shap_analysis['plots'].append(str(plot_path))
                
                # SHAP feature importance
                plt.figure(figsize=self.figsize)
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                plot_type="bar", show=False, max_display=20)
                plt.title(f'{model_name}: SHAP Feature Importance')
                plot_path = self.output_dir / f"{model_name}_shap_importance.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                shap_analysis['plots'].append(str(plot_path))
                
        except Exception as e:
            self.logger.error(f"Error in SHAP analysis: {str(e)}")
            shap_analysis['error'] = str(e)
        
        return shap_analysis
    
    def _prepare_report_for_json(self, report: Dict) -> Dict:
        """Prepare report for JSON serialization by converting numpy types."""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        return convert_numpy(report)
    
    def compare_models(self, model_reports: List[Dict], 
                      save_comparison: bool = True) -> Dict[str, Any]:
        """
        Compare multiple model evaluation reports.
        
        Args:
            model_reports: List of evaluation reports from different models
            save_comparison: Whether to save comparison plots
            
        Returns:
            Dictionary with model comparison analysis
        """
        if len(model_reports) < 2:
            raise ValueError("Need at least 2 model reports for comparison")
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'n_models': len(model_reports),
            'model_names': [report['model_name'] for report in model_reports],
            'metric_comparison': {},
            'plots_saved': []
        }
        
        # Extract metrics for comparison
        metrics_to_compare = ['smape', 'mae', 'mse', 'rmse', 'r2']
        
        for metric in metrics_to_compare:
            comparison['metric_comparison'][metric] = {}
            values = []
            
            for report in model_reports:
                if metric == 'smape':
                    value = report['metrics']['smape']['smape']
                else:
                    value = report['metrics'].get(metric, None)
                
                comparison['metric_comparison'][metric][report['model_name']] = value
                if value is not None:
                    values.append(value)
            
            if values:
                comparison['metric_comparison'][metric]['best_model'] = model_reports[
                    np.argmin(values) if metric in ['smape', 'mae', 'mse', 'rmse'] 
                    else np.argmax(values)
                ]['model_name']
        
        if save_comparison:
            # Create comparison plots
            plot_path = self._create_model_comparison_plot(model_reports)
            comparison['plots_saved'].append(plot_path)
        
        # Save comparison report
        comparison_path = self.output_dir / "model_comparison_report.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def _create_model_comparison_plot(self, model_reports: List[Dict]) -> str:
        """Create model comparison visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = [report['model_name'] for report in model_reports]
        
        # SMAPE comparison
        smape_values = [report['metrics']['smape']['smape'] for report in model_reports]
        bars1 = ax1.bar(model_names, smape_values)
        ax1.set_ylabel('SMAPE (%)')
        ax1.set_title('SMAPE Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Color bars by performance (lower is better for SMAPE)
        colors1 = plt.cm.RdYlGn_r(np.array(smape_values) / max(smape_values))
        for bar, color in zip(bars1, colors1):
            bar.set_color(color)
        
        # R² comparison
        r2_values = [report['metrics']['r2'] for report in model_reports]
        bars2 = ax2.bar(model_names, r2_values)
        ax2.set_ylabel('R²')
        ax2.set_title('R² Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color bars by performance (higher is better for R²)
        colors2 = plt.cm.RdYlGn(np.array(r2_values) / max(r2_values) if max(r2_values) > 0 else [0.5]*len(r2_values))
        for bar, color in zip(bars2, colors2):
            bar.set_color(color)
        
        # MAE comparison
        mae_values = [report['metrics']['mae'] for report in model_reports]
        bars3 = ax3.bar(model_names, mae_values)
        ax3.set_ylabel('MAE')
        ax3.set_title('MAE Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        rmse_values = [report['metrics']['rmse'] for report in model_reports]
        bars4 = ax4.bar(model_names, rmse_values)
        ax4.set_ylabel('RMSE')
        ax4.set_title('RMSE Comparison')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = self.output_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)