"""
Complete Prediction Pipeline Integration for ML Product Pricing Challenge 2025

This module provides the integrated prediction pipeline that processes test data
from loading through feature engineering to final submission file generation,
with comprehensive validation and quality assurance checks.
"""

import logging
import json
import time
import pickle
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
from src.prediction import PredictionGenerator, OutputFormatter, OutputValidator
from src.infrastructure import LoggingManager, ResourceManager
from src.compliance import ComplianceManager


class PredictionPipelineError(Exception):
    """Custom exception for prediction pipeline errors"""
    pass


class IntegratedPredictionPipeline:
    """
    Complete prediction pipeline that integrates test data processing, feature
    engineering, and prediction generation with comprehensive validation.
    """
    
    def __init__(self, config: MLPricingConfig):
        """
        Initialize the integrated prediction pipeline.
        
        Args:
            config: Complete configuration for the pipeline
        """
        self.config = config
        self.prediction_id = self._generate_prediction_id()
        
        # Setup infrastructure components
        self.logger = self._setup_logging()
        self.resource_manager = ResourceManager(config.infrastructure)
        self.compliance_manager = ComplianceManager(config.compliance)
        
        # Initialize component instances
        self._initialize_components()
        
        # Pipeline state tracking
        self.pipeline_state = {
            'prediction_id': self.prediction_id,
            'start_time': None,
            'end_time': None,
            'status': 'initialized',
            'current_step': None,
            'steps_completed': [],
            'errors': [],
            'metrics': {},
            'artifacts': {}
        }
        
        self.logger.info(f"Initialized prediction pipeline with ID: {self.prediction_id}")
    
    def _generate_prediction_id(self) -> str:
        """Generate unique prediction ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"prediction_{timestamp}"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the pipeline"""
        logging_manager = LoggingManager(self.config.infrastructure)
        logger = logging_manager.setup_structured_logging(
            name=f"{self.config.project_name}.prediction_pipeline",
            experiment_id=self.prediction_id
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
                cache_dir=self.config.data.cache_dir
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
            
            # Prediction components
            self.prediction_generator = PredictionGenerator(self.config.prediction)
            self.output_formatter = OutputFormatter(self.config.prediction)
            self.output_validator = OutputValidator(self.config.prediction)
            
            self.logger.info("All prediction pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise PredictionPipelineError(f"Component initialization failed: {str(e)}")
    
    def run_complete_prediction_pipeline(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete prediction pipeline from test data to submission file.
        
        Args:
            model_path: Path to trained model file. If None, will find latest model.
            
        Returns:
            Dict containing pipeline results, metrics, and output file path
        """
        self.pipeline_state['start_time'] = time.time()
        self.pipeline_state['status'] = 'running'
        
        try:
            self.logger.info("Starting complete prediction pipeline")
            
            # Step 1: Load and validate trained model
            model = self._load_trained_model(model_path)
            
            # Step 2: Test data loading and preprocessing
            test_data, processed_test_data = self._execute_test_data_preprocessing()
            
            # Step 3: Feature engineering for test data
            test_features, feature_metadata = self._execute_test_feature_engineering(processed_test_data)
            
            # Step 4: Prediction generation
            predictions = self._execute_prediction_generation(model, test_features)
            
            # Step 5: Output formatting and validation
            output_results = self._execute_output_formatting_and_validation(
                test_data, predictions
            )
            
            # Step 6: Final quality assurance and compliance checks
            qa_results = self._execute_final_quality_assurance(
                test_data, output_results['output_df']
            )
            
            # Compile final results
            final_results = self._compile_prediction_results(
                test_data, feature_metadata, predictions, output_results, qa_results
            )
            
            self.pipeline_state['status'] = 'completed'
            self.pipeline_state['end_time'] = time.time()
            
            self.logger.info("Prediction pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.pipeline_state['status'] = 'failed'
            self.pipeline_state['end_time'] = time.time()
            self.pipeline_state['errors'].append(str(e))
            
            self.logger.error(f"Prediction pipeline failed: {str(e)}")
            raise PredictionPipelineError(f"Pipeline execution failed: {str(e)}")
    
    def _load_trained_model(self, model_path: Optional[str] = None) -> Any:
        """Load trained model from file"""
        self.logger.info("Loading trained model")
        
        try:
            if model_path is None:
                # Find the most recent model file
                model_files = list(Path("models").glob("*.pkl"))
                if not model_files:
                    raise PredictionPipelineError("No trained models found in models/ directory")
                
                # Sort by modification time and get the most recent
                model_path = str(max(model_files, key=lambda p: p.stat().st_mtime))
                self.logger.info(f"Using most recent model: {model_path}")
            
            # Validate model file exists
            if not Path(model_path).exists():
                raise PredictionPipelineError(f"Model file not found: {model_path}")
            
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.pipeline_state['artifacts']['model_path'] = model_path
            self.logger.info(f"Successfully loaded model from: {model_path}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise PredictionPipelineError(f"Model loading failed: {str(e)}")
    
    def _execute_test_data_preprocessing(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute test data loading and preprocessing step"""
        self.pipeline_state['current_step'] = 'test_data_preprocessing'
        self.logger.info("Step 1: Executing test data preprocessing")
        
        try:
            # Load test data
            self.logger.info("Loading test data")
            test_data = self.data_preprocessor.load_test_data()
            
            # Validate schema and types
            self.logger.info("Validating test data schema and types")
            if not self.data_preprocessor.validate_schema_and_types(test_data):
                raise PredictionPipelineError("Test data schema validation failed")
            
            # Clean catalog content (no price normalization needed for test data)
            self.logger.info("Cleaning test catalog content")
            processed_test_data = self.data_preprocessor.clean_catalog_content(test_data.copy())
            
            # Download test images with retry logic
            self.logger.info("Downloading test product images")
            image_results = self.data_preprocessor.download_images(
                processed_test_data, 
                self.config.data.image_dir
            )
            
            # Validate data integrity
            self.logger.info("Validating test data integrity")
            if not self.data_preprocessor.validate_data_integrity(processed_test_data):
                raise PredictionPipelineError("Test data integrity validation failed")
            
            # Log preprocessing metrics
            preprocessing_metrics = {
                'total_test_samples': len(test_data),
                'processed_test_samples': len(processed_test_data),
                'test_images_downloaded': len([r for r in image_results.values() if r == 'success']),
                'test_images_failed': len([r for r in image_results.values() if r != 'success'])
            }
            
            self.pipeline_state['metrics']['test_preprocessing'] = preprocessing_metrics
            self.logger.info(f"Test data preprocessing completed: {preprocessing_metrics}")
            
            self.pipeline_state['steps_completed'].append('test_data_preprocessing')
            return test_data, processed_test_data
            
        except Exception as e:
            self.logger.error(f"Test data preprocessing failed: {str(e)}")
            raise PredictionPipelineError(f"Test data preprocessing failed: {str(e)}")
    
    def _execute_test_feature_engineering(self, processed_test_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute feature engineering for test data"""
        self.pipeline_state['current_step'] = 'test_feature_engineering'
        self.logger.info("Step 2: Executing test feature engineering")
        
        try:
            # Text feature engineering
            self.logger.info("Extracting text features from test data")
            test_text_features = self._extract_test_text_features(processed_test_data)
            
            # Image feature engineering
            self.logger.info("Extracting image features from test data")
            test_image_features = self._extract_test_image_features(processed_test_data)
            
            # Feature fusion (using same method as training)
            self.logger.info("Fusing test text and image features")
            fused_test_features = self.feature_fusion.concatenate_features(
                test_text_features, test_image_features
            )
            
            # Apply dimensionality reduction if configured (using fitted reducer from training)
            if self.config.feature_fusion.use_dimensionality_reduction:
                self.logger.info("Applying dimensionality reduction to test features")
                # Note: In practice, the dimensionality reducer should be fitted during training
                # and saved/loaded here. For now, we'll apply the same reduction.
                final_test_features = self.dimensionality_reducer.reduce_dimensions(
                    fused_test_features, 
                    self.config.feature_fusion.target_dimensions
                )
            else:
                final_test_features = fused_test_features
            
            # Compile feature metadata
            feature_metadata = {
                'test_text_feature_dim': test_text_features.shape[1],
                'test_image_feature_dim': test_image_features.shape[1],
                'test_fused_feature_dim': fused_test_features.shape[1],
                'test_final_feature_dim': final_test_features.shape[1],
                'dimensionality_reduction_applied': self.config.feature_fusion.use_dimensionality_reduction,
                'fusion_method': self.config.feature_fusion.fusion_method
            }
            
            self.pipeline_state['metrics']['test_feature_engineering'] = feature_metadata
            self.logger.info(f"Test feature engineering completed: {feature_metadata}")
            
            self.pipeline_state['steps_completed'].append('test_feature_engineering')
            return final_test_features, feature_metadata
            
        except Exception as e:
            self.logger.error(f"Test feature engineering failed: {str(e)}")
            raise PredictionPipelineError(f"Test feature engineering failed: {str(e)}")
    
    def _extract_test_text_features(self, processed_test_data: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive text features from test data"""
        try:
            # Process catalog content
            processed_content = []
            for content in processed_test_data['catalog_content']:
                parsed = self.text_processor.parse_catalog_content(content)
                processed_content.append(parsed)
            
            # Extract IPQ features
            ipq_features = []
            for content in processed_test_data['catalog_content']:
                ipq_result = self.ipq_extractor.extract_ipq_with_validation(content)
                ipq_features.append(ipq_result)
            
            # Generate text embeddings and statistical features
            test_text_features = self.text_feature_extractor.create_text_features(processed_test_data)
            
            # Parse structured information
            structured_features = []
            for content in processed_test_data['catalog_content']:
                specs = self.catalog_parser.parse_product_specifications(content)
                structured_features.append(specs)
            
            self.logger.info(f"Extracted test text features with shape: {test_text_features.shape}")
            return test_text_features
            
        except Exception as e:
            self.logger.error(f"Test text feature extraction failed: {str(e)}")
            raise
    
    def _extract_test_image_features(self, processed_test_data: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive image features from test data"""
        try:
            # Create image paths from sample IDs
            image_paths = []
            for sample_id in processed_test_data['sample_id']:
                image_path = Path(self.config.data.image_dir) / f"{sample_id}.jpg"
                image_paths.append(str(image_path))
            
            # Extract image features using the integrated pipeline
            test_image_features = self.image_feature_pipeline.process_batch(
                image_paths, 
                processed_test_data['sample_id'].tolist()
            )
            
            self.logger.info(f"Extracted test image features with shape: {test_image_features.shape}")
            return test_image_features
            
        except Exception as e:
            self.logger.error(f"Test image feature extraction failed: {str(e)}")
            raise
    
    def _execute_prediction_generation(self, model: Any, test_features: np.ndarray) -> np.ndarray:
        """Execute prediction generation step"""
        self.pipeline_state['current_step'] = 'prediction_generation'
        self.logger.info("Step 3: Executing prediction generation")
        
        try:
            # Record inference start time for timing validation
            inference_start_time = time.time()
            
            # Generate predictions
            self.logger.info(f"Generating predictions for {len(test_features)} test samples")
            raw_predictions = self.prediction_generator.predict(model, test_features)
            
            # Clamp predictions to minimum threshold
            self.logger.info("Clamping predictions to minimum threshold")
            clamped_predictions = self.prediction_generator.clamp_predictions_to_threshold(
                raw_predictions, 
                self.config.prediction.min_price_threshold
            )
            
            # Validate inference timing
            inference_time = time.time() - inference_start_time
            timing_valid = self.resource_manager.validate_inference_timing(
                inference_start_time, 
                len(test_features)
            )
            
            # Log prediction metrics
            prediction_metrics = {
                'total_predictions': len(clamped_predictions),
                'inference_time_seconds': inference_time,
                'timing_constraint_met': timing_valid,
                'min_prediction': float(np.min(clamped_predictions)),
                'max_prediction': float(np.max(clamped_predictions)),
                'mean_prediction': float(np.mean(clamped_predictions)),
                'predictions_clamped': int(np.sum(raw_predictions != clamped_predictions))
            }
            
            self.pipeline_state['metrics']['prediction_generation'] = prediction_metrics
            self.logger.info(f"Prediction generation completed: {prediction_metrics}")
            
            if not timing_valid:
                self.logger.warning("Inference timing constraint not met")
            
            self.pipeline_state['steps_completed'].append('prediction_generation')
            return clamped_predictions
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {str(e)}")
            raise PredictionPipelineError(f"Prediction generation failed: {str(e)}")
    
    def _execute_output_formatting_and_validation(self, test_data: pd.DataFrame, 
                                                predictions: np.ndarray) -> Dict[str, Any]:
        """Execute output formatting and validation step"""
        self.pipeline_state['current_step'] = 'output_formatting_validation'
        self.logger.info("Step 4: Executing output formatting and validation")
        
        try:
            # Format output according to submission requirements
            self.logger.info("Formatting output according to submission requirements")
            output_df = self.output_formatter.format_output(
                test_data['sample_id'].tolist(), 
                predictions
            )
            
            # Validate exact sample_id matching
            self.logger.info("Validating exact sample_id matching")
            sample_id_match = self.output_validator.validate_exact_sample_id_match(
                output_df, test_data
            )
            
            if not sample_id_match:
                raise PredictionPipelineError("Sample ID matching validation failed")
            
            # Validate row count matching
            self.logger.info("Validating row count matching")
            row_count_match = self.output_validator.validate_row_count_match(
                output_df, test_data
            )
            
            if not row_count_match:
                raise PredictionPipelineError("Row count matching validation failed")
            
            # Comprehensive output validation
            self.logger.info("Performing comprehensive output validation")
            output_valid = self.output_validator.validate_output(output_df, test_data)
            
            if not output_valid:
                raise PredictionPipelineError("Comprehensive output validation failed")
            
            # Save output file
            output_file_path = self.config.prediction.output_file
            self.logger.info(f"Saving output to: {output_file_path}")
            output_df.to_csv(output_file_path, index=False)
            
            # Log formatting and validation metrics
            formatting_metrics = {
                'output_rows': len(output_df),
                'test_data_rows': len(test_data),
                'sample_id_match': sample_id_match,
                'row_count_match': row_count_match,
                'output_validation_passed': output_valid,
                'output_file_path': output_file_path,
                'all_predictions_positive': bool(np.all(output_df['price'] > 0)),
                'output_file_size_mb': Path(output_file_path).stat().st_size / (1024 * 1024)
            }
            
            self.pipeline_state['metrics']['output_formatting_validation'] = formatting_metrics
            self.logger.info(f"Output formatting and validation completed: {formatting_metrics}")
            
            self.pipeline_state['steps_completed'].append('output_formatting_validation')
            return {
                'output_df': output_df,
                'output_file_path': output_file_path,
                'validation_results': formatting_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Output formatting and validation failed: {str(e)}")
            raise PredictionPipelineError(f"Output formatting and validation failed: {str(e)}")
    
    def _execute_final_quality_assurance(self, test_data: pd.DataFrame, 
                                       output_df: pd.DataFrame) -> Dict[str, Any]:
        """Execute final quality assurance and compliance checks"""
        self.pipeline_state['current_step'] = 'final_quality_assurance'
        self.logger.info("Step 5: Executing final quality assurance")
        
        try:
            qa_results = {}
            
            # Final data consistency checks
            self.logger.info("Performing final data consistency checks")
            consistency_checks = {
                'sample_ids_identical': set(test_data['sample_id']) == set(output_df['sample_id']),
                'no_missing_predictions': not output_df['price'].isna().any(),
                'all_prices_positive': (output_df['price'] > 0).all(),
                'price_range_reasonable': (output_df['price'] <= 10000).all(),  # Sanity check
                'output_format_correct': list(output_df.columns) == ['sample_id', 'price']
            }
            
            qa_results['consistency_checks'] = consistency_checks
            
            # Prediction distribution analysis
            self.logger.info("Analyzing prediction distribution")
            price_stats = {
                'min_price': float(output_df['price'].min()),
                'max_price': float(output_df['price'].max()),
                'mean_price': float(output_df['price'].mean()),
                'median_price': float(output_df['price'].median()),
                'std_price': float(output_df['price'].std()),
                'price_quantiles': {
                    'q25': float(output_df['price'].quantile(0.25)),
                    'q75': float(output_df['price'].quantile(0.75)),
                    'q95': float(output_df['price'].quantile(0.95))
                }
            }
            
            qa_results['price_statistics'] = price_stats
            
            # Resource usage validation
            self.logger.info("Validating resource usage")
            resource_validation = self.resource_manager.validate_resource_usage()
            qa_results['resource_validation'] = resource_validation
            
            # Compliance validation
            self.logger.info("Performing compliance validation")
            compliance_validation = self.compliance_manager.validate_submission_completeness()
            qa_results['compliance_validation'] = compliance_validation
            
            # Check for any critical failures
            critical_failures = []
            if not all(consistency_checks.values()):
                critical_failures.append("Data consistency check failed")
            if not resource_validation.get('within_constraints', True):
                critical_failures.append("Resource constraints violated")
            if not compliance_validation.get('all_requirements_met', True):
                critical_failures.append("Compliance requirements not met")
            
            qa_results['critical_failures'] = critical_failures
            qa_results['qa_passed'] = len(critical_failures) == 0
            
            if critical_failures:
                self.logger.error(f"Critical QA failures: {critical_failures}")
                raise PredictionPipelineError(f"Quality assurance failed: {critical_failures}")
            
            self.pipeline_state['metrics']['final_quality_assurance'] = qa_results
            self.logger.info("Final quality assurance completed successfully")
            
            self.pipeline_state['steps_completed'].append('final_quality_assurance')
            return qa_results
            
        except Exception as e:
            self.logger.error(f"Final quality assurance failed: {str(e)}")
            raise PredictionPipelineError(f"Final quality assurance failed: {str(e)}")
    
    def _compile_prediction_results(self, test_data: pd.DataFrame, feature_metadata: Dict,
                                  predictions: np.ndarray, output_results: Dict, 
                                  qa_results: Dict) -> Dict[str, Any]:
        """Compile final prediction pipeline results"""
        try:
            # Calculate total pipeline time
            total_time = self.pipeline_state['end_time'] - self.pipeline_state['start_time']
            
            # Compile comprehensive results
            final_results = {
                'prediction_id': self.prediction_id,
                'pipeline_status': self.pipeline_state['status'],
                'total_execution_time': total_time,
                'steps_completed': self.pipeline_state['steps_completed'],
                'test_data_summary': {
                    'total_test_samples': len(test_data),
                    'feature_dimensions': feature_metadata['test_final_feature_dim']
                },
                'prediction_summary': {
                    'total_predictions': len(predictions),
                    'min_prediction': float(np.min(predictions)),
                    'max_prediction': float(np.max(predictions)),
                    'mean_prediction': float(np.mean(predictions))
                },
                'output_file': output_results['output_file_path'],
                'quality_assurance': {
                    'qa_passed': qa_results['qa_passed'],
                    'critical_failures': qa_results['critical_failures']
                },
                'artifacts': {
                    'model_used': self.pipeline_state['artifacts']['model_path'],
                    'output_file': output_results['output_file_path'],
                    'output_file_size_mb': qa_results.get('price_statistics', {}).get('output_file_size_mb', 0)
                },
                'detailed_metrics': self.pipeline_state['metrics']
            }
            
            # Save final results
            results_path = Path("logs") / f"prediction_results_{self.prediction_id}.json"
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            self.logger.info(f"Prediction results saved to: {results_path}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Failed to compile prediction results: {str(e)}")
            raise PredictionPipelineError(f"Results compilation failed: {str(e)}")
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state"""
        return self.pipeline_state.copy()
    
    def save_pipeline_state(self, filepath: Optional[str] = None) -> str:
        """Save pipeline state to file"""
        if filepath is None:
            filepath = f"logs/prediction_pipeline_state_{self.prediction_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)
        
        return filepath


def main():
    """Main function for running the integrated prediction pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Product Pricing Prediction Pipeline')
    parser.add_argument('--model-path', type=str, help='Path to trained model file')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = MLPricingConfig.load_from_file(args.config)
        else:
            config = MLPricingConfig()
        
        # Initialize and run pipeline
        pipeline = IntegratedPredictionPipeline(config)
        results = pipeline.run_complete_prediction_pipeline(args.model_path)
        
        print(f"Prediction pipeline completed successfully!")
        print(f"Prediction ID: {results['prediction_id']}")
        print(f"Output file: {results['output_file']}")
        print(f"Total predictions: {results['prediction_summary']['total_predictions']}")
        print(f"QA passed: {results['quality_assurance']['qa_passed']}")
        
    except Exception as e:
        print(f"Prediction pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()