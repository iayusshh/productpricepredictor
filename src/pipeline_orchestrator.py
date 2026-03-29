"""
Pipeline Orchestrator for ML Product Pricing Challenge 2025

This module provides comprehensive pipeline orchestration with error handling,
recovery mechanisms, and experiment tracking across both training and prediction
workflows.
"""

import logging
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from src.config import MLPricingConfig
from src.training_pipeline import IntegratedTrainingPipeline, TrainingPipelineError
from src.prediction_pipeline import IntegratedPredictionPipeline, PredictionPipelineError
from src.infrastructure import LoggingManager, ResourceManager
from src.compliance import ComplianceManager


@dataclass
class PipelineCheckpoint:
    """Data class for pipeline checkpoints"""
    checkpoint_id: str
    timestamp: str
    pipeline_type: str
    step_completed: str
    state_data: Dict[str, Any]
    artifacts_created: List[str]


class PipelineOrchestratorError(Exception):
    """Custom exception for pipeline orchestrator errors"""
    pass


class PipelineOrchestrator:
    """
    Comprehensive pipeline orchestrator that manages both training and prediction
    workflows with error handling, recovery, and experiment tracking.
    """
    
    def __init__(self, config: MLPricingConfig):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: Complete configuration for all pipelines
        """
        self.config = config
        self.orchestrator_id = self._generate_orchestrator_id()
        
        # Setup infrastructure
        self.logger = self._setup_logging()
        self.resource_manager = ResourceManager(config.infrastructure)
        self.compliance_manager = ComplianceManager(config.compliance)
        
        # Pipeline instances
        self.training_pipeline = None
        self.prediction_pipeline = None
        
        # Orchestrator state
        self.orchestrator_state = {
            'orchestrator_id': self.orchestrator_id,
            'start_time': None,
            'end_time': None,
            'status': 'initialized',
            'current_pipeline': None,
            'pipelines_completed': [],
            'checkpoints': [],
            'errors': [],
            'recovery_attempts': [],
            'final_results': {}
        }
        
        self.logger.info(f"Initialized pipeline orchestrator with ID: {self.orchestrator_id}")
    
    def _generate_orchestrator_id(self) -> str:
        """Generate unique orchestrator ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"orchestrator_{timestamp}"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the orchestrator"""
        logging_manager = LoggingManager(self.config.infrastructure)
        logger = logging_manager.setup_structured_logging(
            name=f"{self.config.project_name}.pipeline_orchestrator",
            experiment_id=self.orchestrator_id
        )
        return logger
    
    def run_full_pipeline_with_recovery(self, enable_recovery: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline (training + prediction) with error recovery.
        
        Args:
            enable_recovery: Whether to enable automatic error recovery
            
        Returns:
            Dict containing comprehensive results from both pipelines
        """
        self.orchestrator_state['start_time'] = time.time()
        self.orchestrator_state['status'] = 'running'
        
        try:
            self.logger.info("Starting full pipeline orchestration")
            
            # Phase 1: Training Pipeline
            training_results = self._execute_training_phase_with_recovery(enable_recovery)
            
            # Phase 2: Prediction Pipeline
            prediction_results = self._execute_prediction_phase_with_recovery(
                training_results, enable_recovery
            )
            
            # Phase 3: Final Integration and Validation
            integration_results = self._execute_final_integration_phase()
            
            # Compile comprehensive results
            final_results = self._compile_orchestrator_results(
                training_results, prediction_results, integration_results
            )
            
            self.orchestrator_state['status'] = 'completed'
            self.orchestrator_state['end_time'] = time.time()
            self.orchestrator_state['final_results'] = final_results
            
            self.logger.info("Full pipeline orchestration completed successfully")
            return final_results
            
        except Exception as e:
            self.orchestrator_state['status'] = 'failed'
            self.orchestrator_state['end_time'] = time.time()
            self.orchestrator_state['errors'].append({
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.error(f"Pipeline orchestration failed: {str(e)}")
            
            # Attempt final recovery if enabled
            if enable_recovery:
                recovery_results = self._attempt_final_recovery()
                if recovery_results['recovery_successful']:
                    return recovery_results['results']
            
            raise PipelineOrchestratorError(f"Pipeline orchestration failed: {str(e)}")
    
    def _execute_training_phase_with_recovery(self, enable_recovery: bool) -> Dict[str, Any]:
        """Execute training phase with error recovery"""
        self.orchestrator_state['current_pipeline'] = 'training'
        self.logger.info("Phase 1: Executing training pipeline")
        
        max_retries = 3 if enable_recovery else 1
        
        for attempt in range(max_retries):
            try:
                # Create checkpoint before training
                self._create_checkpoint('training', 'pre_training', {})
                
                # Initialize training pipeline
                self.training_pipeline = IntegratedTrainingPipeline(self.config)
                
                # Execute training pipeline
                training_results = self.training_pipeline.run_complete_training_pipeline()
                
                # Create checkpoint after successful training
                self._create_checkpoint('training', 'post_training', training_results)
                
                self.orchestrator_state['pipelines_completed'].append('training')
                self.logger.info("Training phase completed successfully")
                
                return training_results
                
            except TrainingPipelineError as e:
                self.logger.error(f"Training pipeline attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1 and enable_recovery:
                    # Attempt recovery
                    recovery_success = self._attempt_training_recovery(attempt + 1)
                    if recovery_success:
                        continue
                
                # Final attempt failed
                raise PipelineOrchestratorError(f"Training phase failed after {attempt + 1} attempts: {str(e)}")
    
    def _execute_prediction_phase_with_recovery(self, training_results: Dict[str, Any], 
                                              enable_recovery: bool) -> Dict[str, Any]:
        """Execute prediction phase with error recovery"""
        self.orchestrator_state['current_pipeline'] = 'prediction'
        self.logger.info("Phase 2: Executing prediction pipeline")
        
        max_retries = 3 if enable_recovery else 1
        
        # Determine best model from training results
        best_model_path = self._determine_best_model_path(training_results)
        
        for attempt in range(max_retries):
            try:
                # Create checkpoint before prediction
                self._create_checkpoint('prediction', 'pre_prediction', {'model_path': best_model_path})
                
                # Initialize prediction pipeline
                self.prediction_pipeline = IntegratedPredictionPipeline(self.config)
                
                # Execute prediction pipeline
                prediction_results = self.prediction_pipeline.run_complete_prediction_pipeline(best_model_path)
                
                # Create checkpoint after successful prediction
                self._create_checkpoint('prediction', 'post_prediction', prediction_results)
                
                self.orchestrator_state['pipelines_completed'].append('prediction')
                self.logger.info("Prediction phase completed successfully")
                
                return prediction_results
                
            except PredictionPipelineError as e:
                self.logger.error(f"Prediction pipeline attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1 and enable_recovery:
                    # Attempt recovery
                    recovery_success = self._attempt_prediction_recovery(attempt + 1, best_model_path)
                    if recovery_success:
                        continue
                
                # Final attempt failed
                raise PipelineOrchestratorError(f"Prediction phase failed after {attempt + 1} attempts: {str(e)}")
    
    def _execute_final_integration_phase(self) -> Dict[str, Any]:
        """Execute final integration and validation phase"""
        self.orchestrator_state['current_pipeline'] = 'integration'
        self.logger.info("Phase 3: Executing final integration")
        
        try:
            integration_results = {}
            
            # Validate all required artifacts exist
            self.logger.info("Validating pipeline artifacts")
            artifact_validation = self._validate_pipeline_artifacts()
            integration_results['artifact_validation'] = artifact_validation
            
            # Generate comprehensive compliance report
            self.logger.info("Generating compliance report")
            compliance_report = self.compliance_manager.generate_compliance_log()
            integration_results['compliance_report'] = compliance_report
            
            # Create final deliverable structure
            self.logger.info("Creating deliverable structure")
            self.compliance_manager.create_deliverable_structure()
            integration_results['deliverable_structure_created'] = True
            
            # Generate reproduction package
            self.logger.info("Creating reproduction package")
            self.compliance_manager.create_reproduction_package()
            integration_results['reproduction_package_created'] = True
            
            # Final resource usage summary
            self.logger.info("Generating resource usage summary")
            resource_summary = self.resource_manager.calculate_storage_requirements()
            integration_results['resource_summary'] = resource_summary
            
            # Create final checkpoint
            self._create_checkpoint('integration', 'final_integration', integration_results)
            
            self.orchestrator_state['pipelines_completed'].append('integration')
            self.logger.info("Final integration phase completed successfully")
            
            return integration_results
            
        except Exception as e:
            self.logger.error(f"Final integration phase failed: {str(e)}")
            raise PipelineOrchestratorError(f"Final integration failed: {str(e)}")
    
    def _determine_best_model_path(self, training_results: Dict[str, Any]) -> str:
        """Determine the best model path from training results"""
        try:
            experiment_id = training_results['experiment_id']
            
            # Check if ensemble was used and performed better
            model_performance = training_results.get('model_performance', {})
            if (model_performance.get('ensemble_smape') and 
                model_performance.get('ensemble_improvement', 0) > 0):
                return f"models/ensemble_{experiment_id}.pkl"
            
            # Find best individual model
            detailed_metrics = training_results.get('detailed_metrics', {})
            model_training = detailed_metrics.get('model_training', {})
            
            best_model_type = None
            best_smape = float('inf')
            
            for model_type, metrics in model_training.items():
                if isinstance(metrics, dict) and 'validation_smape' in metrics:
                    if metrics['validation_smape'] < best_smape:
                        best_smape = metrics['validation_smape']
                        best_model_type = model_type
            
            if best_model_type:
                return f"models/{best_model_type}_{experiment_id}.pkl"
            
            # Fallback to any available model
            model_files = list(Path("models").glob("*.pkl"))
            if model_files:
                return str(model_files[0])
            
            raise PipelineOrchestratorError("No trained models found")
            
        except Exception as e:
            self.logger.error(f"Failed to determine best model path: {str(e)}")
            raise
    
    def _create_checkpoint(self, pipeline_type: str, step: str, state_data: Dict[str, Any]):
        """Create a pipeline checkpoint"""
        try:
            checkpoint = PipelineCheckpoint(
                checkpoint_id=f"{pipeline_type}_{step}_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                pipeline_type=pipeline_type,
                step_completed=step,
                state_data=state_data,
                artifacts_created=self._list_current_artifacts()
            )
            
            self.orchestrator_state['checkpoints'].append(asdict(checkpoint))
            
            # Save checkpoint to disk
            checkpoint_path = Path("logs") / f"checkpoint_{checkpoint.checkpoint_id}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2, default=str)
            
            self.logger.info(f"Created checkpoint: {checkpoint.checkpoint_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create checkpoint: {str(e)}")
    
    def _list_current_artifacts(self) -> List[str]:
        """List currently available artifacts"""
        artifacts = []
        
        # Check for model files
        model_files = list(Path("models").glob("*.pkl"))
        artifacts.extend([str(f) for f in model_files])
        
        # Check for log files
        log_files = list(Path("logs").glob("*.json"))
        artifacts.extend([str(f) for f in log_files])
        
        # Check for output files
        if Path(self.config.prediction.output_file).exists():
            artifacts.append(self.config.prediction.output_file)
        
        # Check for embedding caches
        embedding_files = list(Path("embeddings").glob("*.npy"))
        artifacts.extend([str(f) for f in embedding_files])
        
        return artifacts
    
    def _attempt_training_recovery(self, attempt: int) -> bool:
        """Attempt to recover from training pipeline failure"""
        self.logger.info(f"Attempting training recovery (attempt {attempt})")
        
        try:
            recovery_actions = []
            
            # Clear any corrupted model files
            corrupted_models = list(Path("models").glob("*.pkl"))
            for model_file in corrupted_models:
                try:
                    # Try to load the model to check if it's corrupted
                    import pickle
                    with open(model_file, 'rb') as f:
                        pickle.load(f)
                except:
                    model_file.unlink()
                    recovery_actions.append(f"Removed corrupted model: {model_file}")
            
            # Clear cache if needed
            if attempt > 1:
                cache_files = list(Path(self.config.data.cache_dir).glob("*"))
                for cache_file in cache_files:
                    if cache_file.is_file():
                        cache_file.unlink()
                recovery_actions.append("Cleared cache files")
            
            # Reduce model complexity for retry
            if attempt > 2:
                self.config.model.model_types = ["random_forest"]  # Use simpler model
                recovery_actions.append("Reduced to simpler model types")
            
            self.orchestrator_state['recovery_attempts'].append({
                'attempt': attempt,
                'pipeline': 'training',
                'actions': recovery_actions,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Training recovery actions: {recovery_actions}")
            return True
            
        except Exception as e:
            self.logger.error(f"Training recovery failed: {str(e)}")
            return False
    
    def _attempt_prediction_recovery(self, attempt: int, model_path: str) -> bool:
        """Attempt to recover from prediction pipeline failure"""
        self.logger.info(f"Attempting prediction recovery (attempt {attempt})")
        
        try:
            recovery_actions = []
            
            # Try alternative model if available
            if attempt > 1:
                model_files = list(Path("models").glob("*.pkl"))
                alternative_models = [f for f in model_files if str(f) != model_path]
                if alternative_models:
                    model_path = str(alternative_models[0])
                    recovery_actions.append(f"Switched to alternative model: {model_path}")
            
            # Reduce batch size for memory issues
            if attempt > 2:
                self.config.infrastructure.batch_size = min(100, self.config.infrastructure.batch_size // 2)
                recovery_actions.append(f"Reduced batch size to: {self.config.infrastructure.batch_size}")
            
            self.orchestrator_state['recovery_attempts'].append({
                'attempt': attempt,
                'pipeline': 'prediction',
                'actions': recovery_actions,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Prediction recovery actions: {recovery_actions}")
            return True
            
        except Exception as e:
            self.logger.error(f"Prediction recovery failed: {str(e)}")
            return False
    
    def _attempt_final_recovery(self) -> Dict[str, Any]:
        """Attempt final recovery by using any available artifacts"""
        self.logger.info("Attempting final recovery")
        
        try:
            recovery_results = {'recovery_successful': False, 'results': {}}
            
            # Check if we have any usable models and test data
            model_files = list(Path("models").glob("*.pkl"))
            test_data_exists = Path(self.config.data.test_file).exists()
            
            if model_files and test_data_exists:
                self.logger.info("Found models and test data, attempting minimal prediction")
                
                # Try minimal prediction pipeline
                try:
                    prediction_pipeline = IntegratedPredictionPipeline(self.config)
                    prediction_results = prediction_pipeline.run_complete_prediction_pipeline(str(model_files[0]))
                    
                    recovery_results['recovery_successful'] = True
                    recovery_results['results'] = {
                        'orchestrator_id': self.orchestrator_id,
                        'recovery_mode': True,
                        'prediction_results': prediction_results,
                        'recovery_note': 'Partial recovery - prediction only'
                    }
                    
                    self.logger.info("Final recovery successful - prediction completed")
                    
                except Exception as e:
                    self.logger.error(f"Final recovery prediction failed: {str(e)}")
            
            return recovery_results
            
        except Exception as e:
            self.logger.error(f"Final recovery attempt failed: {str(e)}")
            return {'recovery_successful': False, 'results': {}}
    
    def _validate_pipeline_artifacts(self) -> Dict[str, bool]:
        """Validate that all required pipeline artifacts exist"""
        validation_results = {}
        
        # Check for trained models
        model_files = list(Path("models").glob("*.pkl"))
        validation_results['models_exist'] = len(model_files) > 0
        
        # Check for output file
        validation_results['output_file_exists'] = Path(self.config.prediction.output_file).exists()
        
        # Check for log files
        log_files = list(Path("logs").glob("*.json"))
        validation_results['log_files_exist'] = len(log_files) > 0
        
        # Check for embeddings cache
        embedding_files = list(Path("embeddings").glob("*"))
        validation_results['embeddings_cached'] = len(embedding_files) > 0
        
        # Check for compliance artifacts
        validation_results['compliance_log_exists'] = Path("compliance_log.txt").exists()
        
        return validation_results
    
    def _compile_orchestrator_results(self, training_results: Dict[str, Any], 
                                    prediction_results: Dict[str, Any],
                                    integration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive orchestrator results"""
        try:
            total_time = self.orchestrator_state['end_time'] - self.orchestrator_state['start_time']
            
            final_results = {
                'orchestrator_id': self.orchestrator_id,
                'orchestration_status': self.orchestrator_state['status'],
                'total_execution_time': total_time,
                'pipelines_completed': self.orchestrator_state['pipelines_completed'],
                'checkpoints_created': len(self.orchestrator_state['checkpoints']),
                'recovery_attempts': len(self.orchestrator_state['recovery_attempts']),
                'training_results': training_results,
                'prediction_results': prediction_results,
                'integration_results': integration_results,
                'final_artifacts': self._list_current_artifacts(),
                'orchestrator_state': self.orchestrator_state
            }
            
            # Save comprehensive results
            results_path = Path("logs") / f"orchestrator_results_{self.orchestrator_id}.json"
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            self.logger.info(f"Orchestrator results saved to: {results_path}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Failed to compile orchestrator results: {str(e)}")
            raise
    
    def get_orchestrator_state(self) -> Dict[str, Any]:
        """Get current orchestrator state"""
        return self.orchestrator_state.copy()
    
    def save_orchestrator_state(self, filepath: Optional[str] = None) -> str:
        """Save orchestrator state to file"""
        if filepath is None:
            filepath = f"logs/orchestrator_state_{self.orchestrator_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.orchestrator_state, f, indent=2, default=str)
        
        return filepath


def main():
    """Main function for running the pipeline orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Product Pricing Pipeline Orchestrator')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--no-recovery', action='store_true', help='Disable automatic error recovery')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = MLPricingConfig.load_from_file(args.config)
        else:
            config = MLPricingConfig()
        
        # Initialize and run orchestrator
        orchestrator = PipelineOrchestrator(config)
        results = orchestrator.run_full_pipeline_with_recovery(enable_recovery=not args.no_recovery)
        
        print(f"Pipeline orchestration completed successfully!")
        print(f"Orchestrator ID: {results['orchestrator_id']}")
        print(f"Pipelines completed: {results['pipelines_completed']}")
        print(f"Total execution time: {results['total_execution_time']:.2f} seconds")
        
        if results.get('training_results'):
            training = results['training_results']
            print(f"Best training SMAPE: {training['model_performance']['best_individual_smape']:.4f}")
        
        if results.get('prediction_results'):
            prediction = results['prediction_results']
            print(f"Predictions generated: {prediction['prediction_summary']['total_predictions']}")
            print(f"Output file: {prediction['output_file']}")
        
    except Exception as e:
        print(f"Pipeline orchestration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()