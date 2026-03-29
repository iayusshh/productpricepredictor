"""
Unit tests for Evaluation Reporter.

Tests for comprehensive evaluation reporting system including
distribution plots, residual analysis, and model diagnostics.
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
import json
from unittest.mock import Mock, patch

from src.evaluation.evaluation_reporter import EvaluationReporter


class TestEvaluationReporter(unittest.TestCase):
    """Test cases for Evaluation Reporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = EvaluationReporter(output_dir=self.temp_dir, figsize=(8, 6))
        
        # Create sample data
        np.random.seed(42)
        self.y_true = np.random.uniform(1, 100, 100)
        self.y_pred = self.y_true + np.random.normal(0, 5, 100)  # Add some noise
        
        # Ensure no negative predictions for realistic test
        self.y_pred = np.maximum(self.y_pred, 0.1)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test reporter initialization."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(self.reporter.figsize, (8, 6))
        self.assertIsNotNone(self.reporter.smape_calculator)
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = self.reporter._calculate_comprehensive_metrics(self.y_true, self.y_pred)
        
        # Check that all expected metrics are present
        expected_keys = ['smape', 'quantile_smape', 'mae', 'mse', 'rmse', 'r2', 
                        'mape', 'prediction_stats', 'actual_stats']
        
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check metric types and ranges
        self.assertIsInstance(metrics['smape'], dict)
        self.assertIsInstance(metrics['mae'], float)
        self.assertIsInstance(metrics['mse'], float)
        self.assertIsInstance(metrics['rmse'], float)
        self.assertIsInstance(metrics['r2'], float)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['mae'], 0)
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['rmse'], 0)
        self.assertLessEqual(metrics['r2'], 1.0)
    
    def test_model_diagnostics(self):
        """Test model diagnostics functionality."""
        diagnostics = self.reporter._perform_model_diagnostics(self.y_true, self.y_pred)
        
        # Check that all expected diagnostic categories are present
        expected_keys = ['residual_stats', 'normality_test', 'heteroscedasticity', 
                        'outliers', 'prediction_range', 'bias_analysis']
        
        for key in expected_keys:
            self.assertIn(key, diagnostics)
        
        # Check residual statistics
        residual_stats = diagnostics['residual_stats']
        self.assertIn('mean', residual_stats)
        self.assertIn('std', residual_stats)
        self.assertIn('skewness', residual_stats)
        self.assertIn('kurtosis', residual_stats)
        
        # Check normality test
        normality = diagnostics['normality_test']
        self.assertIn('shapiro_wilk_statistic', normality)
        self.assertIn('shapiro_wilk_p_value', normality)
        self.assertIn('is_normal', normality)
        self.assertIsInstance(normality['is_normal'], bool)
        
        # Check outlier detection
        outliers = diagnostics['outliers']
        self.assertIn('count', outliers)
        self.assertIn('percentage', outliers)
        self.assertGreaterEqual(outliers['count'], 0)
        self.assertGreaterEqual(outliers['percentage'], 0)
        self.assertLessEqual(outliers['percentage'], 100)
    
    def test_generate_comprehensive_report_basic(self):
        """Test basic comprehensive report generation."""
        report = self.reporter.generate_comprehensive_report(
            self.y_true, self.y_pred, model_name="test_model", save_plots=False
        )
        
        # Check report structure
        expected_keys = ['model_name', 'timestamp', 'n_samples', 'metrics', 
                        'diagnostics', 'plots_saved']
        
        for key in expected_keys:
            self.assertIn(key, report)
        
        # Check basic properties
        self.assertEqual(report['model_name'], 'test_model')
        self.assertEqual(report['n_samples'], len(self.y_true))
        self.assertIsInstance(report['timestamp'], str)
        self.assertIsInstance(report['plots_saved'], list)
        
        # Check that JSON report file was created
        report_file = Path(self.temp_dir) / "test_model_evaluation_report.json"
        self.assertTrue(report_file.exists())
        
        # Verify JSON can be loaded
        with open(report_file, 'r') as f:
            loaded_report = json.load(f)
        self.assertEqual(loaded_report['model_name'], 'test_model')
    
    def test_generate_comprehensive_report_with_plots(self):
        """Test comprehensive report generation with plots."""
        report = self.reporter.generate_comprehensive_report(
            self.y_true, self.y_pred, model_name="test_model_plots", save_plots=True
        )
        
        # Check that plots were saved
        self.assertGreater(len(report['plots_saved']), 0)
        
        # Verify plot files exist
        for plot_path in report['plots_saved']:
            self.assertTrue(Path(plot_path).exists())
            self.assertTrue(plot_path.endswith('.png'))
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        # Create mock model with feature_importances_
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.25])
        
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        importance_analysis = self.reporter._analyze_feature_importance(
            mock_model, None, feature_names, "test_model", save_plots=False
        )
        
        # Check analysis structure
        self.assertIn('feature_importance', importance_analysis)
        self.assertIn('top_10_features', importance_analysis['feature_importance'])
        self.assertIn('all_features', importance_analysis['feature_importance'])
        
        # Check that features are sorted by importance
        top_features = importance_analysis['feature_importance']['top_10_features']
        importances = [f['importance'] for f in top_features]
        self.assertEqual(importances, sorted(importances, reverse=True))
    
    def test_feature_importance_with_coef(self):
        """Test feature importance analysis with model coefficients."""
        # Create mock model with coef_ (like linear regression)
        mock_model = Mock()
        mock_model.coef_ = np.array([0.3, -0.2, 0.15, -0.1, 0.25])
        del mock_model.feature_importances_  # Remove feature_importances_
        
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        importance_analysis = self.reporter._analyze_feature_importance(
            mock_model, None, feature_names, "test_model", save_plots=False
        )
        
        # Check that absolute values of coefficients are used
        self.assertIn('feature_importance', importance_analysis)
        top_features = importance_analysis['feature_importance']['top_10_features']
        
        # Should use absolute values, so feature_1 (0.3) should be first
        self.assertEqual(top_features[0]['feature'], 'feature_1')
        self.assertEqual(top_features[0]['importance'], 0.3)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Create two mock reports
        report1 = self.reporter.generate_comprehensive_report(
            self.y_true, self.y_pred, model_name="model_1", save_plots=False
        )
        
        # Create slightly different predictions for second model
        y_pred_2 = self.y_pred + np.random.normal(0, 2, len(self.y_pred))
        y_pred_2 = np.maximum(y_pred_2, 0.1)
        
        report2 = self.reporter.generate_comprehensive_report(
            self.y_true, y_pred_2, model_name="model_2", save_plots=False
        )
        
        # Compare models
        comparison = self.reporter.compare_models([report1, report2], save_comparison=False)
        
        # Check comparison structure
        expected_keys = ['timestamp', 'n_models', 'model_names', 'metric_comparison']
        for key in expected_keys:
            self.assertIn(key, comparison)
        
        self.assertEqual(comparison['n_models'], 2)
        self.assertEqual(comparison['model_names'], ['model_1', 'model_2'])
        
        # Check metric comparison
        metric_comparison = comparison['metric_comparison']
        self.assertIn('smape', metric_comparison)
        self.assertIn('mae', metric_comparison)
        self.assertIn('r2', metric_comparison)
        
        # Check that best model is identified
        for metric_data in metric_comparison.values():
            if 'best_model' in metric_data:
                self.assertIn(metric_data['best_model'], ['model_1', 'model_2'])
    
    def test_input_validation(self):
        """Test input validation for report generation."""
        # Test shape mismatch
        y_true_wrong = np.array([1, 2, 3])
        y_pred_wrong = np.array([1, 2])
        
        with self.assertRaises(ValueError):
            self.reporter.generate_comprehensive_report(
                y_true_wrong, y_pred_wrong, save_plots=False
            )
    
    def test_empty_model_comparison(self):
        """Test model comparison with insufficient models."""
        report1 = self.reporter.generate_comprehensive_report(
            self.y_true, self.y_pred, model_name="single_model", save_plots=False
        )
        
        with self.assertRaises(ValueError):
            self.reporter.compare_models([report1], save_comparison=False)
    
    @patch('src.evaluation.evaluation_reporter.SHAP_AVAILABLE', False)
    def test_shap_not_available(self):
        """Test SHAP analysis when SHAP is not available."""
        mock_model = Mock()
        X_test = np.random.random((50, 5))
        
        shap_analysis = self.reporter._generate_shap_analysis(
            mock_model, X_test, None, "test_model", save_plots=False
        )
        
        self.assertIn('error', shap_analysis)
        self.assertIn('SHAP not available', shap_analysis['error'])
    
    def test_json_serialization(self):
        """Test that reports can be properly serialized to JSON."""
        # Create report with various numpy types
        report = self.reporter.generate_comprehensive_report(
            self.y_true, self.y_pred, model_name="json_test", save_plots=False
        )
        
        # Test that the prepared report can be serialized to JSON
        prepared_report = self.reporter._prepare_report_for_json(report.copy())
        json_str = json.dumps(prepared_report)
        
        # Test that it can be deserialized
        loaded_report = json.loads(json_str)
        self.assertEqual(loaded_report['model_name'], 'json_test')
    
    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_perfect = self.y_true.copy()
        
        report = self.reporter.generate_comprehensive_report(
            self.y_true, y_perfect, model_name="perfect_model", save_plots=False
        )
        
        # SMAPE should be 0 for perfect predictions
        self.assertAlmostEqual(report['metrics']['smape']['smape'], 0.0, places=6)
        
        # R² should be 1 for perfect predictions
        self.assertAlmostEqual(report['metrics']['r2'], 1.0, places=6)
        
        # MAE should be 0 for perfect predictions
        self.assertAlmostEqual(report['metrics']['mae'], 0.0, places=6)
    
    def test_extreme_predictions(self):
        """Test evaluation with extreme prediction values."""
        # Create predictions with some extreme values
        y_extreme = self.y_pred.copy()
        y_extreme[0] = 1000  # Very high prediction
        y_extreme[1] = 0.001  # Very low prediction
        
        # Should not raise errors
        report = self.reporter.generate_comprehensive_report(
            self.y_true, y_extreme, model_name="extreme_model", save_plots=False
        )
        
        # Check that outliers are detected
        self.assertGreater(report['diagnostics']['outliers']['count'], 0)


if __name__ == '__main__':
    unittest.main()