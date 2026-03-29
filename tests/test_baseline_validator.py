"""
Unit tests for Baseline Validator.

Tests for baseline model validation framework including
baseline model creation, evaluation, and comparison analysis.
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
import json
from unittest.mock import Mock, patch

from src.evaluation.baseline_validator import BaselineValidator


class TestBaselineValidator(unittest.TestCase):
    """Test cases for Baseline Validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.validator = BaselineValidator(output_dir=self.temp_dir)
        
        # Create sample data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5
        
        # Create realistic training data
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = (
            2 * self.X_train[:, 0] + 
            1.5 * self.X_train[:, 1] + 
            np.random.normal(0, 0.5, self.n_samples) + 10
        )
        
        # Create test data
        self.X_test = np.random.randn(50, self.n_features)
        self.y_test = (
            2 * self.X_test[:, 0] + 
            1.5 * self.X_test[:, 1] + 
            np.random.normal(0, 0.5, 50) + 10
        )
        
        # Ensure positive values (for realistic price data)
        self.y_train = np.maximum(self.y_train, 0.1)
        self.y_test = np.maximum(self.y_test, 0.1)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test validator initialization."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertIsNotNone(self.validator.smape_calculator)
        self.assertEqual(self.validator.baseline_models, {})
        self.assertEqual(self.validator.baseline_results, {})
    
    def test_create_baseline_models(self):
        """Test baseline model creation."""
        baselines = self.validator.create_baseline_models(self.X_train, self.y_train)
        
        # Check that expected baselines are created
        expected_baselines = ['mean', 'median', 'linear_regression']
        for baseline in expected_baselines:
            self.assertIn(baseline, baselines)
        
        # Check baseline properties
        self.assertEqual(baselines['mean']['type'], 'statistical')
        self.assertEqual(baselines['median']['type'], 'statistical')
        self.assertEqual(baselines['linear_regression']['type'], 'regression')
        
        # Check that models are callable/have predict method
        self.assertTrue(callable(baselines['mean']['model']))
        self.assertTrue(callable(baselines['median']['model']))
        self.assertTrue(hasattr(baselines['linear_regression']['model'], 'predict'))
        
        # Check single feature baselines are created
        single_feature_baselines = [name for name in baselines.keys() 
                                  if name.startswith('single_feature_')]
        self.assertGreater(len(single_feature_baselines), 0)
        self.assertLessEqual(len(single_feature_baselines), 3)  # Max 3 features tested
    
    def test_evaluate_baselines(self):
        """Test baseline model evaluation."""
        # First create baselines
        self.validator.create_baseline_models(self.X_train, self.y_train)
        
        # Evaluate baselines
        results = self.validator.evaluate_baselines(self.X_test, self.y_test)
        
        # Check that all baselines were evaluated
        self.assertGreater(len(results), 0)
        
        # Check that each result has expected metrics
        for name, metrics in results.items():
            if 'error' not in metrics:
                expected_metrics = ['smape', 'mae', 'mse', 'rmse', 'r2', 'model_type', 'description']
                for metric in expected_metrics:
                    self.assertIn(metric, metrics)
                
                # Check metric ranges
                self.assertGreaterEqual(metrics['smape'], 0)
                self.assertLessEqual(metrics['smape'], 200)  # SMAPE max is 200%
                self.assertGreaterEqual(metrics['mae'], 0)
                self.assertGreaterEqual(metrics['mse'], 0)
                self.assertGreaterEqual(metrics['rmse'], 0)
                self.assertLessEqual(metrics['r2'], 1.0)
    
    def test_evaluate_baselines_without_models(self):
        """Test that evaluation fails without created models."""
        with self.assertRaises(ValueError):
            self.validator.evaluate_baselines(self.X_test, self.y_test)
    
    def test_cross_validate_baselines(self):
        """Test cross-validation of baseline models."""
        # Create baselines
        self.validator.create_baseline_models(self.X_train, self.y_train)
        
        # Cross-validate
        cv_results = self.validator.cross_validate_baselines(self.X_train, self.y_train, cv_folds=3)
        
        # Check results structure
        self.assertGreater(len(cv_results), 0)
        
        for name, results in cv_results.items():
            if 'error' not in results:
                expected_keys = ['smape_scores', 'smape_mean', 'smape_std', 
                               'r2_scores', 'r2_mean', 'r2_std', 'model_type', 'description']
                for key in expected_keys:
                    self.assertIn(key, results)
                
                # Check that we have scores for each fold
                self.assertEqual(len(results['smape_scores']), 3)
                self.assertEqual(len(results['r2_scores']), 3)
                
                # Check that means and stds are reasonable
                self.assertAlmostEqual(results['smape_mean'], 
                                     np.mean(results['smape_scores']), places=6)
                self.assertAlmostEqual(results['r2_mean'], 
                                     np.mean(results['r2_scores']), places=6)
    
    def test_compare_with_model(self):
        """Test model comparison with baselines."""
        # Create and evaluate baselines
        self.validator.create_baseline_models(self.X_train, self.y_train)
        self.validator.evaluate_baselines(self.X_test, self.y_test)
        
        # Create mock model results
        model_results = {
            'smape': 15.0,
            'r2': 0.8,
            'mae': 2.0,
            'rmse': 3.0
        }
        
        # Compare with baselines
        comparison = self.validator.compare_with_model(model_results, "test_model")
        
        # Check comparison structure
        expected_keys = ['model_name', 'timestamp', 'model_metrics', 
                        'baseline_comparison', 'performance_ranking', 'improvement_analysis']
        for key in expected_keys:
            self.assertIn(key, comparison)
        
        self.assertEqual(comparison['model_name'], 'test_model')
        self.assertEqual(comparison['model_metrics'], model_results)
        
        # Check baseline comparisons
        self.assertGreater(len(comparison['baseline_comparison']), 0)
        
        for baseline_name, comp_metrics in comparison['baseline_comparison'].items():
            expected_comp_keys = ['baseline_smape', 'baseline_r2', 'smape_improvement', 
                                'r2_improvement', 'smape_improvement_pct', 'r2_improvement_pct', 
                                'better_than_baseline']
            for key in expected_comp_keys:
                self.assertIn(key, comp_metrics)
            
            self.assertIsInstance(comp_metrics['better_than_baseline'], bool)
        
        # Check performance ranking
        ranking = comparison['performance_ranking']
        self.assertIn('smape_ranking', ranking)
        self.assertIn('r2_ranking', ranking)
        self.assertIn('model_smape_rank', ranking)
        self.assertIn('model_r2_rank', ranking)
        
        # Check improvement analysis
        improvement = comparison['improvement_analysis']
        expected_improvement_keys = ['beats_all_baselines', 'best_smape_improvement', 
                                   'best_r2_improvement', 'avg_smape_improvement', 'avg_r2_improvement']
        for key in expected_improvement_keys:
            self.assertIn(key, improvement)
        
        self.assertIsInstance(improvement['beats_all_baselines'], bool)
    
    def test_compare_with_model_without_baselines(self):
        """Test that comparison fails without baseline results."""
        model_results = {'smape': 15.0, 'r2': 0.8}
        
        with self.assertRaises(ValueError):
            self.validator.compare_with_model(model_results)
    
    def test_validate_model_consistency(self):
        """Test model consistency validation."""
        # Create mock CV results
        cv_results = {
            'smape': [15.2, 14.8, 15.5, 14.9, 15.1],
            'r2': [0.82, 0.79, 0.81, 0.83, 0.80],
            'mae': [2.1, 2.0, 2.2, 1.9, 2.0]
        }
        
        consistency = self.validator.validate_model_consistency(cv_results, "test_model")
        
        # Check consistency structure
        expected_keys = ['model_name', 'timestamp', 'consistency_metrics', 
                        'stability_assessment', 'outlier_detection', 'overall_assessment']
        for key in expected_keys:
            self.assertIn(key, consistency)
        
        self.assertEqual(consistency['model_name'], 'test_model')
        
        # Check consistency metrics for each metric
        for metric_name in ['smape', 'r2', 'mae']:
            self.assertIn(metric_name, consistency['consistency_metrics'])
            
            metrics = consistency['consistency_metrics'][metric_name]
            expected_metric_keys = ['mean', 'std', 'min', 'max', 'range', 
                                  'coefficient_of_variation', 'median', 'q25', 'q75', 'iqr']
            for key in expected_metric_keys:
                self.assertIn(key, metrics)
            
            # Check that statistics are reasonable
            self.assertAlmostEqual(metrics['mean'], np.mean(cv_results[metric_name]), places=6)
            self.assertAlmostEqual(metrics['std'], np.std(cv_results[metric_name]), places=6)
            self.assertEqual(metrics['min'], min(cv_results[metric_name]))
            self.assertEqual(metrics['max'], max(cv_results[metric_name]))
        
        # Check stability assessment
        for metric_name in ['smape', 'r2', 'mae']:
            self.assertIn(metric_name, consistency['stability_assessment'])
            
            stability = consistency['stability_assessment'][metric_name]
            expected_stability_keys = ['is_stable', 'is_moderately_stable', 
                                     'stability_level', 'coefficient_of_variation']
            for key in expected_stability_keys:
                self.assertIn(key, stability)
            
            self.assertIsInstance(stability['is_stable'], bool)
            self.assertIsInstance(stability['is_moderately_stable'], bool)
            self.assertIn(stability['stability_level'], ['stable', 'moderate', 'unstable'])
        
        # Check outlier detection
        for metric_name in ['smape', 'r2', 'mae']:
            self.assertIn(metric_name, consistency['outlier_detection'])
            
            outliers = consistency['outlier_detection'][metric_name]
            expected_outlier_keys = ['outlier_count', 'outlier_percentage', 
                                   'outlier_values', 'lower_bound', 'upper_bound']
            for key in expected_outlier_keys:
                self.assertIn(key, outliers)
            
            self.assertGreaterEqual(outliers['outlier_count'], 0)
            self.assertGreaterEqual(outliers['outlier_percentage'], 0)
            self.assertLessEqual(outliers['outlier_percentage'], 100)
        
        # Check overall assessment
        overall = consistency['overall_assessment']
        expected_overall_keys = ['is_consistent', 'primary_concerns', 'recommendations']
        for key in expected_overall_keys:
            self.assertIn(key, overall)
        
        self.assertIsInstance(overall['is_consistent'], bool)
        self.assertIsInstance(overall['primary_concerns'], list)
        self.assertIsInstance(overall['recommendations'], list)
    
    def test_generate_baseline_summary(self):
        """Test baseline summary generation."""
        # Create and evaluate baselines
        self.validator.create_baseline_models(self.X_train, self.y_train)
        self.validator.evaluate_baselines(self.X_test, self.y_test)
        
        summary = self.validator.generate_baseline_summary()
        
        # Check summary structure
        expected_keys = ['timestamp', 'n_baselines', 'baseline_names', 
                        'best_baseline', 'baseline_statistics']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        self.assertGreater(summary['n_baselines'], 0)
        self.assertEqual(len(summary['baseline_names']), summary['n_baselines'])
        
        # Check best baseline info
        best_baseline = summary['best_baseline']
        self.assertIn('by_smape', best_baseline)
        self.assertIn('by_r2', best_baseline)
        
        for criterion in ['by_smape', 'by_r2']:
            self.assertIn('name', best_baseline[criterion])
            self.assertIn('smape', best_baseline[criterion])
            self.assertIn('r2', best_baseline[criterion])
        
        # Check baseline statistics
        stats = summary['baseline_statistics']
        for metric in ['smape', 'r2']:
            self.assertIn(metric, stats)
            for stat in ['min', 'max', 'mean', 'std']:
                self.assertIn(stat, stats[metric])
    
    def test_generate_baseline_summary_without_results(self):
        """Test that summary generation fails without results."""
        with self.assertRaises(ValueError):
            self.validator.generate_baseline_summary()
    
    def test_statistical_baseline_predictions(self):
        """Test that statistical baselines produce correct predictions."""
        # Create baselines
        baselines = self.validator.create_baseline_models(self.X_train, self.y_train)
        
        # Test mean baseline
        mean_model = baselines['mean']['model']
        mean_predictions = mean_model(self.X_test)
        expected_mean = np.mean(self.y_train)
        
        self.assertTrue(np.allclose(mean_predictions, expected_mean))
        
        # Test median baseline
        median_model = baselines['median']['model']
        median_predictions = median_model(self.X_test)
        expected_median = np.median(self.y_train)
        
        self.assertTrue(np.allclose(median_predictions, expected_median))
    
    def test_linear_regression_baseline(self):
        """Test linear regression baseline functionality."""
        # Create baselines
        baselines = self.validator.create_baseline_models(self.X_train, self.y_train)
        
        # Test linear regression baseline
        lr_model = baselines['linear_regression']['model']
        self.assertTrue(hasattr(lr_model, 'predict'))
        self.assertTrue(hasattr(lr_model, 'coef_'))
        self.assertTrue(hasattr(lr_model, 'intercept_'))
        
        # Make predictions
        predictions = lr_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check that predictions are reasonable (not all the same)
        self.assertGreater(np.std(predictions), 0)
    
    def test_single_feature_baselines(self):
        """Test single feature baseline functionality."""
        # Create baselines
        baselines = self.validator.create_baseline_models(self.X_train, self.y_train)
        
        # Find single feature baselines
        single_feature_baselines = {name: info for name, info in baselines.items() 
                                  if name.startswith('single_feature_')}
        
        self.assertGreater(len(single_feature_baselines), 0)
        
        for name, info in single_feature_baselines.items():
            model = info['model']
            self.assertTrue(hasattr(model, 'predict'))
            self.assertEqual(info['type'], 'regression')
            
            # Extract feature index
            feature_idx = int(name.split('_')[-1])
            self.assertLess(feature_idx, self.n_features)
            
            # Test prediction
            single_feature_data = self.X_test[:, feature_idx:feature_idx+1]
            predictions = model.predict(single_feature_data)
            self.assertEqual(len(predictions), len(self.X_test))
    
    def test_file_saving(self):
        """Test that results are properly saved to files."""
        # Create and evaluate baselines
        self.validator.create_baseline_models(self.X_train, self.y_train)
        self.validator.evaluate_baselines(self.X_test, self.y_test)
        
        # Test model comparison file saving
        model_results = {'smape': 15.0, 'r2': 0.8}
        comparison = self.validator.compare_with_model(model_results, "test_model")
        
        comparison_file = Path(self.temp_dir) / "test_model_baseline_comparison.json"
        self.assertTrue(comparison_file.exists())
        
        # Test consistency analysis file saving
        cv_results = {'smape': [15.0, 14.5, 15.5], 'r2': [0.8, 0.82, 0.78]}
        consistency = self.validator.validate_model_consistency(cv_results, "test_model")
        
        consistency_file = Path(self.temp_dir) / "test_model_consistency_analysis.json"
        self.assertTrue(consistency_file.exists())
        
        # Test summary file saving
        summary = self.validator.generate_baseline_summary()
        
        summary_file = Path(self.temp_dir) / "baseline_validation_summary.json"
        self.assertTrue(summary_file.exists())
        
        # Verify files contain valid JSON
        for file_path in [comparison_file, consistency_file, summary_file]:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.assertIsInstance(data, dict)
    
    def test_edge_case_empty_features(self):
        """Test baseline creation with empty features."""
        X_empty = np.empty((self.n_samples, 0))
        
        baselines = self.validator.create_baseline_models(X_empty, self.y_train)
        
        # Should still create statistical baselines but not linear regression
        expected_baselines = ['mean', 'median']
        for baseline in expected_baselines:
            self.assertIn(baseline, baselines)
        
        # Should not create linear regression with empty features
        self.assertNotIn('linear_regression', baselines)
        
        # Should not create single feature baselines
        single_feature_baselines = [name for name in baselines.keys() 
                                  if name.startswith('single_feature_')]
        self.assertEqual(len(single_feature_baselines), 0)
    
    def test_edge_case_single_feature(self):
        """Test baseline creation with single feature."""
        X_single = self.X_train[:, :1]
        
        baselines = self.validator.create_baseline_models(X_single, self.y_train)
        
        # Should create one single feature baseline
        single_feature_baselines = [name for name in baselines.keys() 
                                  if name.startswith('single_feature_')]
        self.assertEqual(len(single_feature_baselines), 1)
        self.assertIn('single_feature_0', baselines)


if __name__ == '__main__':
    unittest.main()