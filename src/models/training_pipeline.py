"""
Training Pipeline integration for ML Product Pricing Challenge 2025

This module provides a comprehensive training pipeline that integrates
ModelTrainer, CrossValidator, and EnsembleManager for end-to-end model training.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from .model_trainer import ModelTrainer
from .cross_validator import CrossValidator
from .ensemble_manager import EnsembleManager
from ..config import MLPricingConfig


class TrainingPipeline:
    """
    Comprehensive training pipeline integrating all model training components
    
    Provides end-to-end training workflow from individual models to ensembles
    with comprehensive evaluation and reporting.
    """
    
    def __init__(self, config: MLPricingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.model_trainer = ModelTrainer(config)
        self.cross_validator = CrossValidator(config)
        self.ensemble_manager = EnsembleManager(config)
        
        # Results storage
        self.training_results = {}
        self.cv_results = {}
        self.ensemble_results = {}
        
        # Create directories
        self.logs_dir = Path(config.infrastructure.log_dir)
        self.models_dir = Path("models")
        self.logs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for training pipeline"""
        logger = logging.getLogger(f"{__name__}.TrainingPipeline")
        logger.setLevel(getattr(logging, self.config.infrastructure.log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            log_file = self.logs_dir / "training_pipeline.log"
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
    
    def run_complete_training_pipeline(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary containing all training results
        """
        self.logger.info("Starting complete training pipeline")
        pipeline_start_time = datetime.now()
        
        try:
            # Step 1: Train individual models
            self.logger.info("Step 1: Training individual models")
            trained_models = self.model_trainer.train_all_models(X, y)
            self.training_results['individual_models'] = trained_models
            
            if not trained_models:
                raise ValueError("No models were successfully trained")
            
            # Step 2: Cross-validation
            self.logger.info("Step 2: Performing cross-validation")
            model_configs = self._prepare_model_configs_for_cv(trained_models)
            cv_results = self.cross_validator.perform_kfold_cv(X, y, self.model_trainer, model_configs)
            self.cv_results = cv_results
            
            # Step 3: Holdout validation
            self.logger.info("Step 3: Performing holdout validation")
            holdout_results = self.cross_validator.perform_holdout_validation(
                X, y, self.model_trainer, model_configs
            )
            
            # Step 4: Create performance tracking
            self.logger.info("Step 4: Creating performance tracking")
            performance_summary = self.cross_validator.create_detailed_performance_tracking(
                cv_results, holdout_results
            )
            
            # Step 5: Create ensembles
            self.logger.info("Step 5: Creating ensemble models")
            ensemble_results = self._create_and_evaluate_ensembles(X, y, trained_models, cv_results)
            self.ensemble_results = ensemble_results
            
            # Step 6: Final evaluation and reporting
            self.logger.info("Step 6: Generating final reports")
            final_report = self._generate_final_report(
                trained_models, cv_results, holdout_results, ensemble_results, performance_summary
            )
            
            pipeline_end_time = datetime.now()
            pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            # Compile final results
            final_results = {
                'pipeline_info': {
                    'start_time': pipeline_start_time.isoformat(),
                    'end_time': pipeline_end_time.isoformat(),
                    'duration_seconds': pipeline_duration,
                    'success': True
                },
                'individual_models': trained_models,
                'cv_results': cv_results,
                'holdout_results': holdout_results,
                'ensemble_results': ensemble_results,
                'performance_summary': performance_summary,
                'final_report': final_report
            }
            
            # Save complete results
            self._save_pipeline_results(final_results)
            
            self.logger.info(f"Training pipeline completed successfully in {pipeline_duration:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            
            pipeline_end_time = datetime.now()
            pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            error_results = {
                'pipeline_info': {
                    'start_time': pipeline_start_time.isoformat(),
                    'end_time': pipeline_end_time.isoformat(),
                    'duration_seconds': pipeline_duration,
                    'success': False,
                    'error': str(e)
                },
                'individual_models': self.training_results.get('individual_models', {}),
                'cv_results': self.cv_results,
                'ensemble_results': self.ensemble_results
            }
            
            self._save_pipeline_results(error_results)
            raise
    
    def _prepare_model_configs_for_cv(self, trained_models: Dict[str, Any]) -> Dict[str, Dict]:
        """Prepare model configurations for cross-validation"""
        model_configs = {}
        
        for model_name in trained_models.keys():
            model_configs[model_name] = {'model_type': model_name}
        
        return model_configs
    
    def _create_and_evaluate_ensembles(self, X: np.ndarray, y: np.ndarray, 
                                     trained_models: Dict[str, Any],
                                     cv_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create and evaluate ensemble models"""
        ensemble_results = {}
        
        # Extract validation scores for ensemble weighting
        validation_scores = {}
        for model_name, results in cv_results.items():
            validation_scores[model_name] = results.get('mean_smape', 100.0)
        
        # Add models to ensemble manager
        self.ensemble_manager.add_models(trained_models, validation_scores)
        
        # Create different ensemble types
        ensemble_types = []
        
        if self.config.model.use_ensemble:
            ensemble_method = self.config.model.ensemble_method
            
            if ensemble_method in ['voting', 'all']:
                try:
                    voting_ensemble = self.ensemble_manager.create_voting_ensemble()
                    ensemble_types.append('voting')
                    self.logger.info("Voting ensemble created successfully")
                except Exception as e:
                    self.logger.error(f"Failed to create voting ensemble: {str(e)}")
            
            if ensemble_method in ['weighted_average', 'all']:
                try:
                    weighted_ensemble = self.ensemble_manager.create_weighted_average_ensemble()
                    ensemble_types.append('weighted_average')
                    self.logger.info("Weighted average ensemble created successfully")
                except Exception as e:
                    self.logger.error(f"Failed to create weighted average ensemble: {str(e)}")
            
            if ensemble_method in ['stacking', 'all']:
                try:
                    stacking_ensemble = self.ensemble_manager.create_stacking_ensemble()
                    # Fit stacking ensemble
                    stacking_ensemble.fit(X, y, cv_folds=self.config.model.cv_folds)
                    ensemble_types.append('stacking')
                    self.logger.info("Stacking ensemble created successfully")
                except Exception as e:
                    self.logger.error(f"Failed to create stacking ensemble: {str(e)}")
        
        # Evaluate ensembles using holdout validation
        if ensemble_types:
            from sklearn.model_selection import train_test_split
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.evaluation.validation_split,
                random_state=self.config.model.random_seed
            )
            
            # Compare ensemble performance
            try:
                comparison_results = self.ensemble_manager.compare_ensembles(X_val, y_val)
                ensemble_results['comparison'] = comparison_results
                
                # Cross-validate ensembles
                for ensemble_name in ensemble_types:
                    try:
                        cv_results_ensemble = self.ensemble_manager.cross_validate_ensemble(
                            ensemble_name, X, y
                        )
                        ensemble_results[f'{ensemble_name}_cv'] = cv_results_ensemble
                    except Exception as e:
                        self.logger.error(f"Failed to cross-validate {ensemble_name} ensemble: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate ensembles: {str(e)}")
        
        return ensemble_results
    
    def _generate_final_report(self, trained_models: Dict[str, Any], 
                             cv_results: Dict[str, Dict],
                             holdout_results: Dict[str, Dict],
                             ensemble_results: Dict[str, Any],
                             performance_summary: Dict) -> str:
        """Generate comprehensive final report"""
        
        report_lines = [
            "=" * 100,
            "COMPREHENSIVE TRAINING PIPELINE REPORT",
            "=" * 100,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {self.config.project_name} v{self.config.version}",
            ""
        ]
        
        # Individual Models Summary
        report_lines.extend([
            "INDIVIDUAL MODELS TRAINED",
            "-" * 50,
            f"Total models trained: {len(trained_models)}",
            f"Model types: {', '.join(trained_models.keys())}",
            ""
        ])
        
        # Cross-Validation Results
        report_lines.extend([
            "CROSS-VALIDATION RESULTS",
            "-" * 50
        ])
        
        # Sort models by CV performance
        sorted_cv_models = sorted(cv_results.items(), 
                                key=lambda x: x[1].get('mean_smape', float('inf')))
        
        for rank, (model_name, results) in enumerate(sorted_cv_models, 1):
            mean_smape = results.get('mean_smape', 0)
            std_smape = results.get('std_smape', 0)
            overall_smape = results.get('overall_smape', 0)
            
            report_lines.extend([
                f"{rank}. {model_name.upper()}:",
                f"   CV SMAPE: {mean_smape:.4f} ± {std_smape:.4f}",
                f"   Overall SMAPE: {overall_smape:.4f}",
                ""
            ])
        
        # Holdout Validation Results
        if holdout_results:
            report_lines.extend([
                "HOLDOUT VALIDATION RESULTS",
                "-" * 50
            ])
            
            sorted_holdout_models = sorted(holdout_results.items(),
                                         key=lambda x: x[1].get('smape', float('inf')))
            
            for rank, (model_name, results) in enumerate(sorted_holdout_models, 1):
                smape = results.get('smape', 0)
                r2 = results.get('r2', 0)
                
                report_lines.extend([
                    f"{rank}. {model_name.upper()}:",
                    f"   Holdout SMAPE: {smape:.4f}",
                    f"   Holdout R²: {r2:.4f}",
                    ""
                ])
        
        # Ensemble Results
        if ensemble_results:
            report_lines.extend([
                "ENSEMBLE RESULTS",
                "-" * 50
            ])
            
            if 'comparison' in ensemble_results:
                comparison = ensemble_results['comparison']
                if 'best_ensemble' in comparison:
                    best = comparison['best_ensemble']
                    report_lines.extend([
                        f"Best Ensemble: {best['name']}",
                        f"Best Ensemble SMAPE: {best['smape']:.4f}",
                        ""
                    ])
                
                # Individual ensemble performance
                for ensemble_name, metrics in comparison.items():
                    if ensemble_name != 'best_ensemble' and isinstance(metrics, dict):
                        smape = metrics.get('smape', 0)
                        r2 = metrics.get('r2', 0)
                        report_lines.extend([
                            f"{ensemble_name.upper()} Ensemble:",
                            f"   SMAPE: {smape:.4f}",
                            f"   R²: {r2:.4f}",
                            ""
                        ])
        
        # Performance Summary
        if 'best_models' in performance_summary:
            report_lines.extend([
                "BEST MODELS SUMMARY",
                "-" * 50
            ])
            
            best_models = performance_summary['best_models']
            if 'cv_best' in best_models:
                report_lines.append(f"Best CV Model: {best_models['cv_best']}")
            if 'holdout_best' in best_models:
                report_lines.append(f"Best Holdout Model: {best_models['holdout_best']}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 50
        ])
        
        # Find the overall best performing model/ensemble
        best_individual = min(cv_results.keys(), 
                            key=lambda x: cv_results[x].get('mean_smape', float('inf')))
        best_individual_smape = cv_results[best_individual].get('mean_smape', float('inf'))
        
        best_ensemble_smape = float('inf')
        best_ensemble_name = None
        
        if ensemble_results and 'comparison' in ensemble_results:
            comparison = ensemble_results['comparison']
            if 'best_ensemble' in comparison:
                best_ensemble_smape = comparison['best_ensemble']['smape']
                best_ensemble_name = comparison['best_ensemble']['name']
        
        if best_ensemble_name and best_ensemble_smape < best_individual_smape:
            report_lines.extend([
                f"RECOMMENDED MODEL: {best_ensemble_name.upper()} ENSEMBLE",
                f"Expected SMAPE: {best_ensemble_smape:.4f}",
                f"Improvement over best individual model: {best_individual_smape - best_ensemble_smape:.4f}",
            ])
        else:
            report_lines.extend([
                f"RECOMMENDED MODEL: {best_individual.upper()}",
                f"Expected SMAPE: {best_individual_smape:.4f}",
            ])
        
        report_lines.extend([
            "",
            "=" * 100
        ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.logs_dir / f"final_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Final training report saved to {report_file}")
        
        return report_content
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.logs_dir / f"complete_pipeline_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Complete pipeline results saved to {results_file}")
    
    def get_best_model(self) -> Tuple[str, Any, float]:
        """
        Get the best performing model based on cross-validation results
        
        Returns:
            Tuple of (model_name, model_object, cv_smape)
        """
        if not self.cv_results:
            raise ValueError("No cross-validation results available")
        
        best_model_name = min(self.cv_results.keys(),
                            key=lambda x: self.cv_results[x].get('mean_smape', float('inf')))
        
        best_cv_smape = self.cv_results[best_model_name].get('mean_smape', float('inf'))
        
        # Check if ensemble is better
        if self.ensemble_results and 'comparison' in self.ensemble_results:
            comparison = self.ensemble_results['comparison']
            if 'best_ensemble' in comparison:
                best_ensemble_smape = comparison['best_ensemble']['smape']
                best_ensemble_name = comparison['best_ensemble']['name']
                
                if best_ensemble_smape < best_cv_smape:
                    ensemble_model = self.ensemble_manager.ensemble_models.get(best_ensemble_name)
                    if ensemble_model:
                        return best_ensemble_name, ensemble_model, best_ensemble_smape
        
        # Return best individual model
        individual_models = self.training_results.get('individual_models', {})
        best_model = individual_models.get(best_model_name)
        
        return best_model_name, best_model, best_cv_smape
    
    def save_best_model(self, filepath: str = None) -> str:
        """
        Save the best performing model
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where the model was saved
        """
        best_name, best_model, best_smape = self.get_best_model()
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.models_dir / f"best_model_{best_name}_{timestamp}.pkl"
        
        if 'ensemble' in best_name.lower():
            self.ensemble_manager.save_ensemble(best_name, filepath)
        else:
            self.model_trainer.save_model(best_model, filepath)
        
        self.logger.info(f"Best model ({best_name}, SMAPE: {best_smape:.4f}) saved to {filepath}")
        
        return str(filepath)