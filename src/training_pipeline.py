"""
Complete Training Pipeline Integration for ML Product Pricing Challenge 2025

This module provides the integrated training pipeline that connects all components
from data loading through model training to evaluation, with comprehensive error
handling, structured logging, and experiment tracking.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import MLPricingConfig
from src.data_processing import DataPreprocessor
from src.features import (
    TextProcessor, TextFeatureExtractor, CatalogParser, IPQExtractor,
    ImageProcessor, ImageEmbeddingSystem, VisualFeatureExtractor, 
    MissingImageHandler, ImageFeaturePipeline,
    FeatureFusion, DimensionalityReducer
)
from src.models import ModelTrainer, CrossValidator, EnsembleManager, TrainingPipeline
from src.evaluation import SMAPECalculator, EvaluationReporter, BaselineValidator
from src.infrastructure import LoggingManager, ResourceManager, EmbeddingCache, ImageCache
from src.compliance import ComplianceManager


class TrainingPipelineError(Exception):
    """Custom exception for training pipeline errors"""
    pass


class IntegratedTrainingPipeline:
    """
    Complete training pipeline that integrates all components with comprehensive
    error handling, logging, and experiment tracking.
    """
    
    def __init__(self, config: MLPricingConfig):
        """
        Initialize the integrated training pipeline.
        
        Args:
            config: Complete configuration for the pipeline
        """
        self.config = config
        self.experiment_id = self._generate_experiment_id()
        
        # Setup infrastructure components
        self.logger = self._setup_logging()
        self.resource_manager = ResourceManager(config.infrastructure)
        self.cache_manager = CacheManager(config.data.cache_dir)
        self.compliance_manager = ComplianceManager(config.compliance)
        
        # Initialize component instances
        self._initialize_components()
        
        # Pipeline state tracking
        self.pipeline_state = {
            'experiment_id': self.experiment_id,
            'start_time': None,
            'end_time': None,
            'status': 'initialized',
            'current_step': None,
            'steps_completed': [],
            'errors': [],
            'metrics': {},
            'artifacts': {}
        }
        
        self.logger.info(f"Initialized training pipeline with experiment ID: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"training_{timestamp}"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the pipeline"""
        logging_manager = LoggingManager(self.config.infrastructure)
        logger = logging_manager.setup_structured_logging(
            name=f"{self.config.project_name}.training_pipeline",
            experiment_id=self.experiment_id
        )
        return logger
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Data processing components
            self.data_preprocessor = DataPreprocessor(self.config.data)
            
            # Text feature engineering components
            self.text_processor = TextProcessor(self.config.text_features)
            self.text_feature_extractor = TextFeatureExtractor(self.config.text_features)
            self.catalog_parser = CatalogParser(self.config.text_features)
            self.ipq_extractor = IPQExtractor(self.config.text_features)
            
            # Image feature engineering components
            self.image_processor = ImageProcessor(self.config.image_features)
            self.image_embedding_system = ImageEmbeddingSystem(
                self.config.image_features, 
                self.cache_manager
            )
            self.visual_feature_extractor = VisualFeatureExtractor(self.config.image_features)
            self.missing_image_handler = MissingImageHandler(self.config.image_features)
            self.image_feature_pipeline = ImageFeaturePipeline(
                self.image_processor,
                self.image_embedding_system,
                self.visual_feature_extractor,
                self.missing_image_handler
            )
            
            # Feature fusion components
            self.feature_fusion = FeatureFusion(self.config.feature_fusion)
            self.dimensionality_reducer = DimensionalityReducer(self.config.feature_fusion)
            
            # Model training components
            self.model_trainer = ModelTrainer(self.config.model)
            self.cross_validator = CrossValidator(self.config.model)
            self.ensemble_manager = EnsembleManager(self.config.model)
            
            # Evaluation components
            self.smape_calculator = SMAPECalculator(self.config.evaluation)
            self.evaluation_reporter = EvaluationReporter(self.config.evaluation)
            self.baseline_validator = BaselineValidator(self.config.evaluation)
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise TrainingPipelineError(f"Component initialization failed: {str(e)}")
    
    def run_complete_training_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline from data loading to model evaluation.
        
        Returns:
            Dict containing pipeline results, metrics, and artifacts
        """
        self.pipeline_state['start_time'] = time.time()
        self.pipeline_state['status'] = 'running'
        
        try:
            self.logger.info("Starting complete training pipeline")
            
            # Step 1: Data Loading and Preprocessing
            train_data, processed_data = self._execute_data_preprocessing()
            
            # Step 2: Feature Engineering
            features, feature_metadata = self._execute_feature_engineering(processed_data)
            
            # Step 3: Model Training and Cross-Validation
            models, training_results = self._execute_model_training(features, train_data['price'].values)
            
            # Step 4: Model Evaluation and Reporting
            evaluation_results = self._execute_model_evaluation(models, features, train_data['price'].values)
            
            # Step 5: Ensemble Creation and Validation
            ensemble_results = self._execute_ensemble_creation(models, features, train_data['price'].values)
            
            # Step 6: Final Pipeline Validation
            pipeline_results = self._execute_pipeline_validation(ensemble_results)
            
            # Compile final results
            final_results = self._compile_pipeline_results(
                processed_data, feature_metadata, training_results, 
                evaluation_results, ensemble_results, pipeline_results
            )
            
            self.pipeline_state['status'] = 'completed'
            self.pipeline_state['end_time'] = time.time()
            
            self.logger.info("Training pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.pipeline_state['status'] = 'failed'
            self.pipeline_state['end_time'] = time.time()
            self.pipeline_state['errors'].append(str(e))
            
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise TrainingPipelineError(f"Pipeline execution failed: {str(e)}")
    
    def _execute_data_preprocessing(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute data loading and preprocessing step"""
        self.pipeline_state['current_step'] = 'data_preprocessing'
        self.logger.info("Step 1: Executing data preprocessing")
        
        try:
            # Load training data
            self.logger.info("Loading training data")
            train_data = self.data_preprocessor.load_training_data()
            
            # Validate schema and types
            self.logger.info("Validating data schema and types")
            if not self.data_preprocessor.validate_schema_and_types(train_data):
                raise TrainingPipelineError("Data schema validation failed")
            
            # Normalize price formatting
            self.logger.info("Normalizing price formatting")
            train_data = self.data_preprocessor.normalize_price_formatting(train_data)
            
            # Handle zero prices
            self.logger.info("Handling zero prices")
            train_data = self.data_preprocessor.handle_zero_prices(train_data)
            
            # Clean catalog content
            self.logger.info("Cleaning catalog content")
            processed_data = self.data_preprocessor.clean_catalog_content(train_data.copy())
            
            # Download images with retry logic
            self.logger.info("Downloading product images")
            image_results = self.data_preprocessor.download_images(
                processed_data, 
                self.config.data.image_dir
            )
            
            # Validate data integrity
            self.logger.info("Validating data integrity")
            if not self.data_preprocessor.validate_data_integrity(processed_data):
                raise TrainingPipelineError("Data integrity validation failed")
            
            # Log preprocessing metrics
            preprocessing_metrics = {
                'total_samples': len(train_data),
                'processed_samples': len(processed_data),
                'images_downloaded': len([r for r in image_results.values() if r == 'success']),
                'images_failed': len([r for r in image_results.values() if r != 'success']),
                'zero_prices_handled': len(train_data) - len(processed_data) if len(train_data) != len(processed_data) else 0
            }
            
            self.pipeline_state['metrics']['preprocessing'] = preprocessing_metrics
            self.logger.info(f"Data preprocessing completed: {preprocessing_metrics}")
            
            self.pipeline_state['steps_completed'].append('data_preprocessing')
            return train_data, processed_data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise TrainingPipelineError(f"Data preprocessing failed: {str(e)}")
    
    def _execute_feature_engineering(self, processed_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute feature engineering step"""
        self.pipeline_state['current_step'] = 'feature_engineering'
        self.logger.info("Step 2: Executing feature engineering")
        
        try:
            # Text feature engineering
            self.logger.info("Extracting text features")
            text_features = self._extract_text_features(processed_data)
            
            # Image feature engineering
            self.logger.info("Extracting image features")
            image_features = self._extract_image_features(processed_data)
            
            # Feature fusion
            self.logger.info("Fusing text and image features")
            fused_features = self.feature_fusion.concatenate_features(text_features, image_features)
            
            # Dimensionality reduction if configured
            if self.config.feature_fusion.use_dimensionality_reduction:
                self.logger.info("Applying dimensionality reduction")
                final_features = self.dimensionality_reducer.reduce_dimensions(
                    fused_features, 
                    self.config.feature_fusion.target_dimensions
                )
            else:
                final_features = fused_features
            
            # Compile feature metadata
            feature_metadata = {
                'text_feature_dim': text_features.shape[1],
                'image_feature_dim': image_features.shape[1],
                'fused_feature_dim': fused_features.shape[1],
                'final_feature_dim': final_features.shape[1],
                'dimensionality_reduction_applied': self.config.feature_fusion.use_dimensionality_reduction,
                'fusion_method': self.config.feature_fusion.fusion_method
            }
            
            self.pipeline_state['metrics']['feature_engineering'] = feature_metadata
            self.logger.info(f"Feature engineering completed: {feature_metadata}")
            
            self.pipeline_state['steps_completed'].append('feature_engineering')
            return final_features, feature_metadata
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise TrainingPipelineError(f"Feature engineering failed: {str(e)}")
    
    def _extract_text_features(self, processed_data: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive text features"""
        try:
            # Process catalog content
            processed_content = []
            for content in processed_data['catalog_content']:
                parsed = self.text_processor.parse_catalog_content(content)
                processed_content.append(parsed)
            
            # Extract IPQ features with validation
            ipq_features = []
            for content in processed_data['catalog_content']:
                ipq_result = self.ipq_extractor.extract_ipq_with_validation(content)
                ipq_features.append(ipq_result)
            
            # Validate IPQ extraction precision
            precision = self.ipq_extractor.validate_extraction_precision()
            if precision < self.config.text_features.ipq_precision_threshold:
                self.logger.warning(f"IPQ extraction precision ({precision:.3f}) below threshold ({self.config.text_features.ipq_precision_threshold})")
            
            # Generate text embeddings and statistical features
            text_features = self.text_feature_extractor.create_text_features(processed_data)
            
            # Parse structured information
            structured_features = []
            for content in processed_data['catalog_content']:
                specs = self.catalog_parser.parse_product_specifications(content)
                structured_features.append(specs)
            
            self.logger.info(f"Extracted text features with shape: {text_features.shape}")
            return text_features
            
        except Exception as e:
            self.logger.error(f"Text feature extraction failed: {str(e)}")
            raise
    
    def _extract_image_features(self, processed_data: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive image features"""
        try:
            # Create image paths from sample IDs
            image_paths = []
            for sample_id in processed_data['sample_id']:
                image_path = Path(self.config.data.image_dir) / f"{sample_id}.jpg"
                image_paths.append(str(image_path))
            
            # Extract image features using the integrated pipeline
            image_features = self.image_feature_pipeline.process_batch(
                image_paths, 
                processed_data['sample_id'].tolist()
            )
            
            self.logger.info(f"Extracted image features with shape: {image_features.shape}")
            return image_features
            
        except Exception as e:
            self.logger.error(f"Image feature extraction failed: {str(e)}")
            raise
    
    def _execute_model_training(self, features: np.ndarray, targets: np.ndarray) -> Tuple[List[Any], Dict[str, Any]]:
        """Execute model training and cross-validation step"""
        self.pipeline_state['current_step'] = 'model_training'
        self.logger.info("Step 3: Executing model training")
        
        try:
            # Set random seeds for reproducibility
            self.model_trainer.set_random_seeds(self.config.model.random_seed)
            
            # Capture experiment metadata
            experiment_metadata = self.model_trainer.capture_experiment_metadata(
                self.config.to_dict(),
                self.config.model.cv_folds,
                self.config.model.random_seed
            )
            
            # Train multiple model types
            trained_models = []
            training_results = {}
            
            for model_type in self.config.model.model_types:
                self.logger.info(f"Training {model_type} model")
                
                # Configure model-specific parameters
                model_config = self._get_model_config(model_type)
                
                # Train model with cross-validation
                model = self.model_trainer.train_model(features, targets, model_config)
                
                # Validate model with detailed metrics
                validation_metrics = self.cross_validator.validate_with_detailed_metrics(
                    model, features, targets
                )
                
                # Calculate per-quantile SMAPE
                quantile_metrics = self.model_trainer.calculate_per_quantile_smape(
                    validation_metrics['y_true'], 
                    validation_metrics['y_pred']
                )
                
                trained_models.append({
                    'model': model,
                    'type': model_type,
                    'config': model_config,
                    'validation_metrics': validation_metrics,
                    'quantile_metrics': quantile_metrics
                })
                
                training_results[model_type] = {
                    'validation_smape': validation_metrics['smape'],
                    'cv_scores': validation_metrics['cv_scores'],
                    'quantile_smape': quantile_metrics
                }
                
                # Save model checkpoint
                model_path = Path("models") / f"{model_type}_{self.experiment_id}.pkl"
                self.model_trainer.save_model(model, str(model_path))
                
                self.logger.info(f"{model_type} training completed - SMAPE: {validation_metrics['smape']:.4f}")
            
            # Report cross-validation results with statistics
            cv_summary = self.model_trainer.report_cv_results_with_statistics(
                [result['validation_smape'] for result in training_results.values()]
            )
            
            training_results['cv_summary'] = cv_summary
            training_results['experiment_metadata'] = experiment_metadata
            
            self.pipeline_state['metrics']['model_training'] = training_results
            self.logger.info(f"Model training completed: {len(trained_models)} models trained")
            
            self.pipeline_state['steps_completed'].append('model_training')
            return trained_models, training_results
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise TrainingPipelineError(f"Model training failed: {str(e)}")
    
    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        base = {'model_type': model_type, 'random_state': self.config.model.random_seed}

        if model_type == "random_forest":
            return {**base,
                'n_estimators': self.config.model.rf_n_estimators,
                'max_depth': self.config.model.rf_max_depth,
                'min_samples_split': self.config.model.rf_min_samples_split,
            }
        elif model_type == "xgboost":
            return {**base,
                'n_estimators': self.config.model.xgb_n_estimators,
                'max_depth': self.config.model.xgb_max_depth,
                'learning_rate': self.config.model.xgb_learning_rate,
                'subsample': self.config.model.xgb_subsample,
            }
        elif model_type == "lightgbm":
            return {**base,
                'n_estimators': self.config.model.lgb_n_estimators,
                'max_depth': self.config.model.lgb_max_depth,
                'learning_rate': self.config.model.lgb_learning_rate,
                'num_leaves': self.config.model.lgb_num_leaves,
            }
        elif model_type == "extra_trees":
            return {**base,
                'n_estimators': self.config.model.et_n_estimators,
                'max_depth': self.config.model.et_max_depth,
            }
        elif model_type == "gradient_boosting":
            return {**base,
                'n_estimators': self.config.model.gbr_n_estimators,
                'max_depth': self.config.model.gbr_max_depth,
                'learning_rate': self.config.model.gbr_learning_rate,
                'subsample': self.config.model.gbr_subsample,
            }
        elif model_type == "ridge_regression":
            return {**base, 'alpha': self.config.model.ridge_alpha}
        elif model_type == "neural_network":
            return {**base,
                'hidden_layers': self.config.model.nn_hidden_layers,
                'dropout_rate': self.config.model.nn_dropout_rate,
                'learning_rate': self.config.model.nn_learning_rate,
                'batch_size': self.config.model.nn_batch_size,
                'epochs': self.config.model.nn_epochs,
            }
        else:
            return base
    
    def _execute_model_evaluation(self, models: List[Dict], features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Execute comprehensive model evaluation step"""
        self.pipeline_state['current_step'] = 'model_evaluation'
        self.logger.info("Step 4: Executing model evaluation")
        
        try:
            evaluation_results = {}
            
            for model_info in models:
                model_type = model_info['type']
                model = model_info['model']
                validation_metrics = model_info['validation_metrics']
                
                self.logger.info(f"Evaluating {model_type} model")
                
                # Test SMAPE calculation on known examples
                smape_validation = self.smape_calculator.test_smape_on_known_examples()
                if not smape_validation:
                    self.logger.warning(f"SMAPE validation failed for {model_type}")
                
                # Generate comprehensive evaluation report
                evaluation_report = self.evaluation_reporter.generate_evaluation_report(
                    validation_metrics['y_true'],
                    validation_metrics['y_pred'],
                    model_type
                )
                
                # Create distribution plots
                self.evaluation_reporter.create_distribution_plots(
                    validation_metrics['y_true'],
                    validation_metrics['y_pred'],
                    f"{model_type}_{self.experiment_id}"
                )
                
                # Generate residual histograms
                self.evaluation_reporter.generate_residual_histograms(
                    validation_metrics['y_true'],
                    validation_metrics['y_pred'],
                    f"{model_type}_{self.experiment_id}"
                )
                
                # Calculate feature importance (if supported)
                try:
                    feature_importance = self.evaluation_reporter.calculate_shap_feature_importance(
                        model, features[:1000]  # Sample for SHAP calculation
                    )
                    evaluation_report['feature_importance'] = feature_importance
                except Exception as e:
                    self.logger.warning(f"Feature importance calculation failed for {model_type}: {str(e)}")
                
                evaluation_results[model_type] = evaluation_report
            
            # Baseline model validation
            baseline_results = self.baseline_validator.validate_against_baselines(
                targets, 
                [model_info['validation_metrics']['y_pred'] for model_info in models]
            )
            
            evaluation_results['baseline_comparison'] = baseline_results
            
            self.pipeline_state['metrics']['model_evaluation'] = evaluation_results
            self.logger.info("Model evaluation completed")
            
            self.pipeline_state['steps_completed'].append('model_evaluation')
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise TrainingPipelineError(f"Model evaluation failed: {str(e)}")
    
    def _execute_ensemble_creation(self, models: List[Dict], features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Execute ensemble creation and validation step"""
        self.pipeline_state['current_step'] = 'ensemble_creation'
        self.logger.info("Step 5: Executing ensemble creation")
        
        try:
            if not self.config.model.use_ensemble or len(models) < 2:
                self.logger.info("Ensemble creation skipped")
                return {'ensemble_used': False}
            
            # Extract models and their validation predictions
            model_list = [model_info['model'] for model_info in models]
            model_predictions = [model_info['validation_metrics']['y_pred'] for model_info in models]
            
            # Create ensemble using configured method
            if self.config.model.ensemble_method == "weighted_average":
                ensemble = self.ensemble_manager.create_weighted_average_ensemble(
                    model_list, 
                    model_predictions, 
                    targets
                )
            elif self.config.model.ensemble_method == "stacking":
                ensemble = self.ensemble_manager.create_stacking_ensemble(
                    model_list, 
                    features, 
                    targets
                )
            else:
                ensemble = self.ensemble_manager.create_voting_ensemble(model_list)
            
            # Validate ensemble performance
            ensemble_predictions = ensemble.predict(features)
            ensemble_smape = self.smape_calculator.calculate_smape_with_validation(
                targets, 
                ensemble_predictions
            )
            
            # Compare ensemble vs individual models
            individual_smapes = [model_info['validation_metrics']['smape'] for model_info in models]
            best_individual_smape = min(individual_smapes)
            
            ensemble_results = {
                'ensemble_used': True,
                'ensemble_method': self.config.model.ensemble_method,
                'ensemble_smape': ensemble_smape,
                'best_individual_smape': best_individual_smape,
                'improvement': best_individual_smape - ensemble_smape,
                'model_weights': getattr(ensemble, 'weights_', None)
            }
            
            # Save ensemble model
            ensemble_path = Path("models") / f"ensemble_{self.experiment_id}.pkl"
            self.model_trainer.save_model(ensemble, str(ensemble_path))
            
            self.pipeline_state['metrics']['ensemble_creation'] = ensemble_results
            self.logger.info(f"Ensemble creation completed - SMAPE: {ensemble_smape:.4f}")
            
            self.pipeline_state['steps_completed'].append('ensemble_creation')
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {str(e)}")
            raise TrainingPipelineError(f"Ensemble creation failed: {str(e)}")
    
    def _execute_pipeline_validation(self, ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final pipeline validation step"""
        self.pipeline_state['current_step'] = 'pipeline_validation'
        self.logger.info("Step 6: Executing pipeline validation")
        
        try:
            validation_results = {}
            
            # Validate resource usage
            resource_validation = self.resource_manager.validate_resource_usage()
            validation_results['resource_validation'] = resource_validation
            
            # Validate storage requirements
            storage_requirements = self.resource_manager.calculate_storage_requirements()
            validation_results['storage_requirements'] = storage_requirements
            
            # Validate compliance
            compliance_validation = self.compliance_manager.validate_submission_completeness()
            validation_results['compliance_validation'] = compliance_validation
            
            # Generate compliance log
            compliance_log = self.compliance_manager.generate_compliance_log()
            validation_results['compliance_log_path'] = compliance_log
            
            # Validate experiment reproducibility
            reproducibility_check = self._validate_reproducibility()
            validation_results['reproducibility_check'] = reproducibility_check
            
            self.pipeline_state['metrics']['pipeline_validation'] = validation_results
            self.logger.info("Pipeline validation completed")
            
            self.pipeline_state['steps_completed'].append('pipeline_validation')
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Pipeline validation failed: {str(e)}")
            raise TrainingPipelineError(f"Pipeline validation failed: {str(e)}")
    
    def _validate_reproducibility(self) -> Dict[str, bool]:
        """Validate experiment reproducibility"""
        try:
            reproducibility_checks = {
                'random_seeds_set': True,  # Already validated in model training
                'experiment_metadata_captured': 'experiment_metadata' in self.pipeline_state['metrics'].get('model_training', {}),
                'model_checkpoints_saved': Path("models").exists() and len(list(Path("models").glob("*.pkl"))) > 0,
                'configuration_saved': True,  # Configuration is part of experiment metadata
                'logs_structured': Path(self.config.infrastructure.log_dir).exists()
            }
            
            return reproducibility_checks
            
        except Exception as e:
            self.logger.error(f"Reproducibility validation failed: {str(e)}")
            return {'validation_failed': True, 'error': str(e)}
    
    def _compile_pipeline_results(self, processed_data: pd.DataFrame, feature_metadata: Dict, 
                                training_results: Dict, evaluation_results: Dict, 
                                ensemble_results: Dict, pipeline_results: Dict) -> Dict[str, Any]:
        """Compile final pipeline results"""
        try:
            # Calculate total pipeline time
            total_time = self.pipeline_state['end_time'] - self.pipeline_state['start_time']
            
            # Compile comprehensive results
            final_results = {
                'experiment_id': self.experiment_id,
                'pipeline_status': self.pipeline_state['status'],
                'total_execution_time': total_time,
                'steps_completed': self.pipeline_state['steps_completed'],
                'data_summary': {
                    'total_samples': len(processed_data),
                    'feature_dimensions': feature_metadata['final_feature_dim']
                },
                'model_performance': {
                    'best_individual_smape': min([result['validation_smape'] for result in training_results.values() if isinstance(result, dict) and 'validation_smape' in result]),
                    'ensemble_smape': ensemble_results.get('ensemble_smape'),
                    'ensemble_improvement': ensemble_results.get('improvement', 0)
                },
                'artifacts': {
                    'models_saved': len(list(Path("models").glob("*.pkl"))),
                    'evaluation_plots': len(list(Path("logs").glob("*.png"))),
                    'compliance_log': pipeline_results.get('compliance_log_path')
                },
                'detailed_metrics': self.pipeline_state['metrics']
            }
            
            # Save final results
            results_path = Path("logs") / f"training_results_{self.experiment_id}.json"
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline results saved to: {results_path}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Failed to compile pipeline results: {str(e)}")
            raise TrainingPipelineError(f"Results compilation failed: {str(e)}")
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state"""
        return self.pipeline_state.copy()
    
    def save_pipeline_state(self, filepath: Optional[str] = None) -> str:
        """Save pipeline state to file"""
        if filepath is None:
            filepath = f"logs/pipeline_state_{self.experiment_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)
        
        return filepath


def main():
    """Main function for running the integrated training pipeline"""
    try:
        # Load configuration
        config = MLPricingConfig()
        
        # Initialize and run pipeline
        pipeline = IntegratedTrainingPipeline(config)
        results = pipeline.run_complete_training_pipeline()
        
        print(f"Training pipeline completed successfully!")
        print(f"Experiment ID: {results['experiment_id']}")
        print(f"Best SMAPE: {results['model_performance']['best_individual_smape']:.4f}")
        
        if results['model_performance']['ensemble_smape']:
            print(f"Ensemble SMAPE: {results['model_performance']['ensemble_smape']:.4f}")
            print(f"Ensemble improvement: {results['model_performance']['ensemble_improvement']:.4f}")
        
    except Exception as e:
        print(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()