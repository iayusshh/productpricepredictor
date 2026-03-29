"""
Example usage of the comprehensive evaluation system.

This script demonstrates how to use the SMAPE calculator, evaluation reporter,
and baseline validator for ML model evaluation.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .smape_calculator import SMAPECalculator
from .evaluation_reporter import EvaluationReporter
from .baseline_validator import BaselineValidator


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some relationship to features
    y = (
        2 * X[:, 0] + 
        1.5 * X[:, 1] - 
        0.8 * X[:, 2] + 
        np.random.normal(0, 0.5, n_samples) + 10
    )
    
    # Ensure positive values (for realistic price data)
    y = np.maximum(y, 0.1)
    
    # Split into train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def demonstrate_smape_calculator():
    """Demonstrate SMAPE calculator usage."""
    print("=== SMAPE Calculator Demo ===")
    
    calculator = SMAPECalculator(log_performance=True)
    
    # Run validation tests
    print("Running validation tests...")
    validation_passed = calculator.run_validation_tests()
    print(f"Validation tests passed: {validation_passed}")
    
    # Calculate SMAPE on sample data
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 18, 32, 38, 52])
    
    smape = calculator.calculate_smape(y_true, y_pred)
    print(f"Sample SMAPE: {smape:.4f}%")
    
    # Get detailed results
    details = calculator.calculate_smape_with_details(y_true, y_pred)
    print(f"Detailed SMAPE results: {details['smape']:.4f}% ± {details['std_error']:.4f}%")
    
    # Calculate quantile SMAPE
    quantile_smape = calculator.calculate_quantile_smape(y_true, y_pred, n_quantiles=3)
    print("Quantile SMAPE:")
    for quantile, smape_val in quantile_smape.items():
        print(f"  {quantile}: {smape_val:.4f}%")
    
    print()


def demonstrate_baseline_validator():
    """Demonstrate baseline validator usage."""
    print("=== Baseline Validator Demo ===")
    
    X_train, X_test, y_train, y_test = create_sample_data()
    
    validator = BaselineValidator(output_dir="logs/demo_baseline_validation")
    
    # Create baseline models
    print("Creating baseline models...")
    baselines = validator.create_baseline_models(X_train, y_train)
    print(f"Created {len(baselines)} baseline models: {list(baselines.keys())}")
    
    # Evaluate baselines
    print("Evaluating baseline models...")
    baseline_results = validator.evaluate_baselines(X_test, y_test)
    
    print("Baseline Results:")
    for name, metrics in baseline_results.items():
        if 'error' not in metrics:
            print(f"  {name}: SMAPE={metrics['smape']:.4f}%, R²={metrics['r2']:.4f}")
    
    # Cross-validate baselines
    print("Cross-validating baseline models...")
    cv_results = validator.cross_validate_baselines(X_train, y_train, cv_folds=5)
    
    print("Cross-validation Results:")
    for name, results in cv_results.items():
        if 'error' not in results:
            print(f"  {name}: SMAPE={results['smape_mean']:.4f}±{results['smape_std']:.4f}%")
    
    # Train a simple model for comparison
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate model metrics
    calculator = SMAPECalculator(log_performance=False)
    model_results = {
        'smape': calculator.calculate_smape(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    print(f"Random Forest Model: SMAPE={model_results['smape']:.4f}%, R²={model_results['r2']:.4f}")
    
    # Compare with baselines
    print("Comparing model with baselines...")
    comparison = validator.compare_with_model(model_results, "RandomForest")
    
    print("Model vs Baselines:")
    print(f"  Beats all baselines: {comparison['improvement_analysis']['beats_all_baselines']}")
    print(f"  Best SMAPE improvement: {comparison['improvement_analysis']['best_smape_improvement']:.4f}%")
    print(f"  Model SMAPE rank: {comparison['performance_ranking']['model_smape_rank']}")
    
    # Validate model consistency
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_smape_scores = []
    cv_r2_scores = []
    
    for train_idx, val_idx in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        fold_model = RandomForestRegressor(n_estimators=50, random_state=42)
        fold_model.fit(X_train_fold, y_train_fold)
        y_val_pred = fold_model.predict(X_val_fold)
        
        cv_smape_scores.append(calculator.calculate_smape(y_val_fold, y_val_pred))
        cv_r2_scores.append(r2_score(y_val_fold, y_val_pred))
    
    cv_results_model = {
        'smape': cv_smape_scores,
        'r2': cv_r2_scores
    }
    
    print("Validating model consistency...")
    consistency = validator.validate_model_consistency(cv_results_model, "RandomForest")
    
    print("Model Consistency:")
    print(f"  Is consistent: {consistency['overall_assessment']['is_consistent']}")
    print(f"  SMAPE stability: {consistency['stability_assessment']['smape']['stability_level']}")
    print(f"  R² stability: {consistency['stability_assessment']['r2']['stability_level']}")
    
    print()


def demonstrate_evaluation_reporter():
    """Demonstrate evaluation reporter usage."""
    print("=== Evaluation Reporter Demo ===")
    
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Train a model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Generate comprehensive report
    reporter = EvaluationReporter(output_dir="logs/demo_evaluation_reports")
    
    print("Generating comprehensive evaluation report...")
    report = reporter.generate_comprehensive_report(
        y_test, y_pred, 
        model=model, 
        X_test=X_test,
        feature_names=feature_names,
        model_name="RandomForest_Demo",
        save_plots=True
    )
    
    print("Evaluation Report Summary:")
    print(f"  SMAPE: {report['metrics']['smape']['smape']:.4f}%")
    print(f"  R²: {report['metrics']['r2']:.4f}")
    print(f"  MAE: {report['metrics']['mae']:.4f}")
    print(f"  Plots saved: {len(report['plots_saved'])}")
    
    # Model diagnostics
    diagnostics = report['diagnostics']
    print("Model Diagnostics:")
    print(f"  Residual mean: {diagnostics['residual_stats']['mean']:.6f}")
    print(f"  Residual std: {diagnostics['residual_stats']['std']:.4f}")
    print(f"  Normality test p-value: {diagnostics['normality_test']['shapiro_wilk_p_value']:.4f}")
    print(f"  Outliers: {diagnostics['outliers']['count']} ({diagnostics['outliers']['percentage']:.1f}%)")
    
    # Feature importance
    if 'feature_importance' in report:
        top_features = report['feature_importance']['feature_importance']['top_10_features'][:3]
        print("Top 3 Features:")
        for i, feature in enumerate(top_features, 1):
            print(f"  {i}. {feature['feature']}: {feature['importance']:.4f}")
    
    print()


def main():
    """Run all demonstrations."""
    print("ML Product Pricing - Evaluation System Demo")
    print("=" * 50)
    
    demonstrate_smape_calculator()
    demonstrate_baseline_validator()
    demonstrate_evaluation_reporter()
    
    print("Demo completed! Check the logs/ directory for saved reports and plots.")


if __name__ == "__main__":
    main()