"""
Main entry point for ML Product Pricing Challenge 2025
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import MLPricingConfig
from src.training_pipeline import IntegratedTrainingPipeline
from src.prediction_pipeline import IntegratedPredictionPipeline


class MLPricingPipeline:
    """Main pipeline orchestrating training and prediction workflows"""
    
    def __init__(self, config: MLPricingConfig):
        self.config = config
        self.training_pipeline = IntegratedTrainingPipeline(config)
        self.prediction_pipeline = IntegratedPredictionPipeline(config)
    
    def run_training_pipeline(self) -> dict:
        """Execute the complete training pipeline"""
        return self.training_pipeline.run_complete_training_pipeline()
    
    def run_prediction_pipeline(self, model_path: Optional[str] = None) -> dict:
        """Execute the prediction pipeline for test data"""
        return self.prediction_pipeline.run_complete_prediction_pipeline(model_path)
    
    def run_full_pipeline(self) -> dict:
        """Execute both training and prediction pipelines"""
        print("Starting full ML Product Pricing pipeline...")
        
        # Run training pipeline
        print("Phase 1: Training pipeline")
        training_results = self.run_training_pipeline()
        
        # Determine best model for prediction
        best_model_path = self._get_best_model_path(training_results)
        
        # Run prediction pipeline
        print("Phase 2: Prediction pipeline")
        prediction_results = self.run_prediction_pipeline(best_model_path)
        
        # Compile final results
        final_results = {
            'training_results': training_results,
            'prediction_results': prediction_results,
            'best_model_used': best_model_path,
            'pipeline_status': 'completed'
        }
        
        print("Full pipeline completed successfully!")
        return final_results
    
    def _get_best_model_path(self, training_results: dict) -> str:
        """Determine the best model path from training results"""
        experiment_id = training_results['experiment_id']
        
        # Check if ensemble was used and performed better
        if (training_results.get('model_performance', {}).get('ensemble_smape') and
            training_results['model_performance']['ensemble_improvement'] > 0):
            return f"models/ensemble_{experiment_id}.pkl"
        
        # Otherwise, find the best individual model
        model_training = training_results.get('detailed_metrics', {}).get('model_training', {})
        best_model_type = None
        best_smape = float('inf')
        
        for model_type, metrics in model_training.items():
            if isinstance(metrics, dict) and 'validation_smape' in metrics:
                if metrics['validation_smape'] < best_smape:
                    best_smape = metrics['validation_smape']
                    best_model_type = model_type
        
        if best_model_type:
            return f"models/{best_model_type}_{experiment_id}.pkl"
        
        # Fallback to first available model
        model_files = list(Path("models").glob("*.pkl"))
        if model_files:
            return str(model_files[0])
        
        raise ValueError("No trained models found")


def main():
    """Main entry point with command line argument support"""
    parser = argparse.ArgumentParser(description='ML Product Pricing Challenge 2025')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                       help='Pipeline mode: train only, predict only, or full pipeline')
    parser.add_argument('--model-path', type=str, help='Path to trained model for prediction mode')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = MLPricingConfig.load_from_file(args.config)
        else:
            config = MLPricingConfig()
        
        # Initialize pipeline
        pipeline = MLPricingPipeline(config)
        
        # Run specified mode
        if args.mode == 'train':
            results = pipeline.run_training_pipeline()
            print(f"Training completed. Experiment ID: {results['experiment_id']}")
            print(f"Best SMAPE: {results['model_performance']['best_individual_smape']:.4f}")
            
        elif args.mode == 'predict':
            if not args.model_path:
                print("Error: --model-path required for prediction mode")
                sys.exit(1)
            results = pipeline.run_prediction_pipeline(args.model_path)
            print(f"Prediction completed. Output saved to: {results['output_file']}")
            
        else:  # full pipeline
            results = pipeline.run_full_pipeline()
            print(f"Full pipeline completed successfully!")
            
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()