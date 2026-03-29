"""
Example usage of the ML Product Pricing training framework

This script demonstrates how to use the ModelTrainer, CrossValidator, 
EnsembleManager, and TrainingPipeline components.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ..config import MLPricingConfig
from .training_pipeline import TrainingPipeline
from .model_trainer import ModelTrainer
from .cross_validator import CrossValidator
from .ensemble_manager import EnsembleManager


def generate_sample_data(n_samples: int = 1000, n_features: int = 100) -> tuple:
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some relationship to features
    true_coefficients = np.random.randn(n_features) * 0.1
    noise = np.random.randn(n_samples) * 0.1
    y = X @ true_coefficients + noise + 10  # Add baseline price
    y = np.abs(y)  # Ensure positive prices
    
    return X, y


def example_individual_training():
    """Example of training individual models"""
    print("=" * 60)
    print("EXAMPLE: Individual Model Training")
    print("=" * 60)
    
    # Create configuration
    config = MLPricingConfig()
    config.model.model_types = ['random_forest', 'xgboost', 'lightgbm']
    config.model.use_hyperparameter_tuning = False  # Disable for quick demo
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=500, n_features=50)
    print(f"Generated sample data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train all models
    trained_models = trainer.train_all_models(X, y)
    
    print(f"Successfully trained {len(trained_models)} models:")
    for model_name in trained_models.keys():
        print(f"  - {model_name}")
    
    return trained_models, X, y, config


def example_cross_validation():
    """Example of cross-validation"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Cross-Validation")
    print("=" * 60)
    
    # Get models from previous example
    trained_models, X, y, config = example_individual_training()
    
    # Initialize cross-validator
    cv = CrossValidator(config)
    trainer = ModelTrainer(config)
    
    # Prepare model configurations
    model_configs = {}
    for model_name in trained_models.keys():
        model_configs[model_name] = {'model_type': model_name}
    
    # Perform cross-validation
    cv_results = cv.perform_kfold_cv(X, y, trainer, model_configs)
    
    print("\nCross-validation results:")
    for model_name, results in cv_results.items():
        mean_smape = results.get('mean_smape', 0)
        std_smape = results.get('std_smape', 0)
        print(f"  {model_name}: {mean_smape:.4f} ± {std_smape:.4f} SMAPE")
    
    return cv_results


def example_ensemble_training():
    """Example of ensemble training"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Ensemble Training")
    print("=" * 60)
    
    # Get models from previous example
    trained_models, X, y, config = example_individual_training()
    
    # Initialize ensemble manager
    ensemble_manager = EnsembleManager(config)
    
    # Add models to ensemble
    validation_scores = {
        'random_forest': 15.2,
        'xgboost': 14.8,
        'lightgbm': 15.0
    }
    ensemble_manager.add_models(trained_models, validation_scores)
    
    # Create different ensemble types
    try:
        # Weighted average ensemble
        weighted_ensemble = ensemble_manager.create_weighted_average_ensemble()
        print("Created weighted average ensemble")
        
        # Voting ensemble (sklearn models only)
        voting_ensemble = ensemble_manager.create_voting_ensemble()
        print("Created voting ensemble")
        
        # Stacking ensemble
        stacking_ensemble = ensemble_manager.create_stacking_ensemble()
        stacking_ensemble.fit(X, y)
        print("Created and fitted stacking ensemble")
        
    except Exception as e:
        print(f"Error creating ensembles: {e}")
    
    # Compare ensemble performance
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        comparison_results = ensemble_manager.compare_ensembles(X_val, y_val)
        
        print("\nEnsemble comparison results:")
        for ensemble_name, metrics in comparison_results.items():
            if ensemble_name != 'best_ensemble' and isinstance(metrics, dict):
                smape = metrics.get('smape', 0)
                print(f"  {ensemble_name}: {smape:.4f} SMAPE")
        
        if 'best_ensemble' in comparison_results:
            best = comparison_results['best_ensemble']
            print(f"\nBest ensemble: {best['name']} ({best['smape']:.4f} SMAPE)")
    
    except Exception as e:
        print(f"Error comparing ensembles: {e}")


def example_complete_pipeline():
    """Example of complete training pipeline"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Complete Training Pipeline")
    print("=" * 60)
    
    # Create configuration
    config = MLPricingConfig()
    config.model.model_types = ['random_forest', 'xgboost']  # Reduced for demo
    config.model.use_hyperparameter_tuning = False
    config.model.use_ensemble = True
    config.model.ensemble_method = 'weighted_average'
    config.model.cv_folds = 3  # Reduced for demo
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=300, n_features=30)
    print(f"Generated sample data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config)
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_training_pipeline(X, y)
        
        print("\nPipeline completed successfully!")
        print(f"Duration: {results['pipeline_info']['duration_seconds']:.2f} seconds")
        
        # Get best model
        best_name, best_model, best_smape = pipeline.get_best_model()
        print(f"Best model: {best_name} (SMAPE: {best_smape:.4f})")
        
        # Save best model
        model_path = pipeline.save_best_model()
        print(f"Best model saved to: {model_path}")
        
        return results
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return None


def main():
    """Run all examples"""
    print("ML Product Pricing Training Framework Examples")
    print("=" * 60)
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    try:
        # Run individual examples
        example_individual_training()
        example_cross_validation()
        example_ensemble_training()
        
        # Run complete pipeline
        results = example_complete_pipeline()
        
        if results:
            print("\n" + "=" * 60)
            print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\nCheck the 'logs/' directory for detailed reports and results.")
            print("Check the 'models/' directory for saved models.")
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()