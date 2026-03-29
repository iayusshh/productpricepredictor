"""
Integration tests for Deliverable Structure and Submission Compliance.

Tests for deliverable completeness validation, submission format compliance,
and overall project structure integrity.
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import json
import yaml
import subprocess

from src.compliance.deliverable_manager import DeliverableManager
from src.compliance.integration_validator import IntegrationValidator
from src.config import config


class TestDeliverableComplianceIntegration(unittest.TestCase):
    """Integration test cases for Deliverable Compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create complete project structure
        self.create_project_structure()
        self.create_test_deliverables()
        
        # Mock config
        self.original_config = config
        self.mock_config()
        
        # Initialize managers
        self.deliverable_manager = DeliverableManager()
        self.integration_validator = IntegrationValidator()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original config
        config.__dict__.update(self.original_config.__dict__)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_project_structure(self):
        """Create complete project directory structure."""
        # Main directories
        directories = [
            "src", "src/data_processing", "src/features", "src/models", 
            "src/evaluation", "src/prediction", "src/compliance", "src/infrastructure",
            "notebooks", "tests", "dataset", "images", "cache", "models", 
            "embeddings", "logs", "deliverables"
        ]
        
        for directory in directories:
            (self.temp_path / directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_dirs = ["src", "src/data_processing", "src/features", "src/models", 
                    "src/evaluation", "src/prediction", "src/compliance", "src/infrastructure", "tests"]
        for directory in init_dirs:
            (self.temp_path / directory / "__init__.py").touch()
    
    def create_test_deliverables(self):
        """Create test deliverable files."""
        # Create test_out.csv
        test_out_data = pd.DataFrame({
            'sample_id': [f'test_{i}' for i in range(1, 101)],
            'price': np.random.uniform(5.0, 100.0, 100)
        })
        test_out_data.to_csv(self.temp_path / "deliverables" / "test_out.csv", index=False)
        
        # Create methodology PDF (mock)
        methodology_path = self.temp_path / "deliverables" / "methodology_1page.pdf"
        methodology_path.write_bytes(b'%PDF-1.4\n%Mock PDF content for testing')
        
        # Create requirements.txt
        requirements_content = """
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
torch==2.0.1
transformers==4.30.2
requests==2.31.0
tqdm==4.65.0
matplotlib==3.7.1
seaborn==0.12.2
"""
        (self.temp_path / "requirements.txt").write_text(requirements_content.strip())
        
        # Create environment.yml
        environment_content = """
name: ml-product-pricing
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - pandas=1.5.3
  - numpy=1.24.3
  - scikit-learn=1.3.0
  - pytorch=2.0.1
  - transformers=4.30.2
  - requests=2.31.0
  - tqdm=4.65.0
  - matplotlib=3.7.1
  - seaborn=0.12.2
  - pip
  - pip:
    - some-pip-only-package==1.0.0
"""
        (self.temp_path / "environment.yml").write_text(environment_content.strip())
        
        # Create README.md
        readme_content = """
# ML Product Pricing Challenge 2025

## Overview
This project implements a machine learning solution for product price prediction.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: Place train.csv and test.csv in dataset/
3. Run preprocessing: `python src/main.py --preprocess`
4. Train model: `python src/main.py --train`
5. Generate predictions: `python src/main.py --predict`

## Reproduction Steps
Run the complete pipeline:
```bash
bash run_all.sh
```

## Project Structure
- src/: Source code
- notebooks/: Jupyter notebooks for EDA
- tests/: Unit and integration tests
- dataset/: Training and test data
- models/: Saved model checkpoints
- deliverables/: Final submission files
"""
        (self.temp_path / "README.md").write_text(readme_content.strip())
        
        # Create run_all.sh
        run_all_content = """#!/bin/bash
set -e

echo "Starting ML Product Pricing Pipeline..."

# Data preprocessing
echo "Step 1: Data preprocessing..."
python src/main.py --preprocess

# Feature engineering
echo "Step 2: Feature engineering..."
python src/main.py --extract-features

# Model training
echo "Step 3: Model training..."
python src/main.py --train

# Generate predictions
echo "Step 4: Generate predictions..."
python src/main.py --predict

echo "Pipeline completed successfully!"
echo "Results saved to deliverables/test_out.csv"
"""
        run_all_path = self.temp_path / "run_all.sh"
        run_all_path.write_text(run_all_content.strip())
        run_all_path.chmod(0o755)  # Make executable
        
        # Create sample source files
        main_py_content = """
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='ML Product Pricing Pipeline')
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--extract-features', action='store_true', help='Extract features')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    
    args = parser.parse_args()
    
    if args.preprocess:
        print("Running data preprocessing...")
    elif args.extract_features:
        print("Extracting features...")
    elif args.train:
        print("Training models...")
    elif args.predict:
        print("Generating predictions...")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
"""
        (self.temp_path / "src" / "main.py").write_text(main_py_content.strip())
        
        # Create sample notebook
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Exploratory Data Analysis\n", "This notebook contains EDA for the ML Product Pricing Challenge."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import pandas as pd\n", "import numpy as np\n", "import matplotlib.pyplot as plt"]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(self.temp_path / "notebooks" / "eda.ipynb", 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        # Create logs directory with sample logs
        (self.temp_path / "logs" / "training.log").write_text("Sample training log")
        (self.temp_path / "logs" / "cv_results.json").write_text('{"fold_1": 25.5, "fold_2": 24.8, "fold_3": 26.1}')
        
        # Create models directory with sample model files
        (self.temp_path / "models" / "best_model.pkl").write_bytes(b"mock model data")
        (self.temp_path / "models" / "model_metadata.json").write_text('{"model_type": "ensemble", "cv_score": 25.5}')
        
        # Create embeddings directory
        (self.temp_path / "embeddings" / "text_embeddings.npy").write_bytes(b"mock embedding data")
        (self.temp_path / "embeddings" / "image_embeddings.npy").write_bytes(b"mock embedding data")
    
    def mock_config(self):
        """Mock configuration for testing."""
        # Set paths to temp directory
        config.project_root = str(self.temp_path)
        config.output.deliverables_dir = str(self.temp_path / "deliverables")
        config.output.predictions_file = str(self.temp_path / "deliverables" / "test_out.csv")
        config.model.model_dir = str(self.temp_path / "models")
        config.logging.log_dir = str(self.temp_path / "logs")
    
    def test_complete_deliverable_structure_validation(self):
        """Test complete deliverable structure validation."""
        # Validate deliverable structure
        validation_results = self.deliverable_manager.validate_deliverable_structure()
        
        # Should pass validation
        self.assertIsInstance(validation_results, dict)
        self.assertIn('is_valid', validation_results)
        self.assertTrue(validation_results['is_valid'])
        
        # Check specific validations
        self.assertIn('required_files', validation_results)
        self.assertIn('directory_structure', validation_results)
        self.assertIn('file_formats', validation_results)
        
        # Verify required files are present
        required_files = validation_results['required_files']
        self.assertTrue(required_files['test_out_csv'])
        self.assertTrue(required_files['methodology_pdf'])
        self.assertTrue(required_files['requirements_txt'])
        self.assertTrue(required_files['readme_md'])
        self.assertTrue(required_files['run_all_sh'])
    
    def test_submission_format_compliance(self):
        """Test submission format compliance."""
        # Load test_out.csv
        test_out_path = self.temp_path / "deliverables" / "test_out.csv"
        test_out_df = pd.read_csv(test_out_path)
        
        # Validate format compliance
        compliance_results = self.deliverable_manager.validate_submission_format(test_out_df)
        
        # Should pass compliance checks
        self.assertIsInstance(compliance_results, dict)
        self.assertIn('is_compliant', compliance_results)
        self.assertTrue(compliance_results['is_compliant'])
        
        # Check specific format requirements
        self.assertIn('column_names', compliance_results)
        self.assertIn('column_types', compliance_results)
        self.assertIn('data_quality', compliance_results)
        
        # Verify column compliance
        self.assertTrue(compliance_results['column_names']['correct'])
        self.assertTrue(compliance_results['column_types']['correct'])
        self.assertTrue(compliance_results['data_quality']['no_missing_values'])
        self.assertTrue(compliance_results['data_quality']['positive_prices'])
    
    def test_dependency_validation(self):
        """Test dependency and environment validation."""
        # Validate requirements.txt
        requirements_validation = self.deliverable_manager.validate_requirements_file()
        
        self.assertIsInstance(requirements_validation, dict)
        self.assertIn('is_valid', requirements_validation)
        self.assertTrue(requirements_validation['is_valid'])
        
        # Check for key dependencies
        self.assertIn('dependencies', requirements_validation)
        dependencies = requirements_validation['dependencies']
        
        # Should have core ML dependencies
        dep_names = [dep['name'] for dep in dependencies]
        required_deps = ['pandas', 'numpy', 'scikit-learn']
        for dep in required_deps:
            self.assertIn(dep, dep_names)
        
        # Validate environment.yml if present
        env_yml_path = self.temp_path / "environment.yml"
        if env_yml_path.exists():
            env_validation = self.deliverable_manager.validate_environment_file()
            self.assertIsInstance(env_validation, dict)
            self.assertIn('is_valid', env_validation)
    
    def test_code_structure_validation(self):
        """Test code structure and organization validation."""
        # Validate source code structure
        code_validation = self.integration_validator.validate_code_structure()
        
        self.assertIsInstance(code_validation, dict)
        self.assertIn('is_valid', code_validation)
        self.assertTrue(code_validation['is_valid'])
        
        # Check directory structure
        self.assertIn('src_directory', code_validation)
        self.assertTrue(code_validation['src_directory']['exists'])
        
        # Check for key modules
        self.assertIn('key_modules', code_validation)
        key_modules = code_validation['key_modules']
        
        expected_modules = ['data_processing', 'features', 'models', 'evaluation', 'prediction']
        for module in expected_modules:
            self.assertIn(module, key_modules)
            self.assertTrue(key_modules[module]['exists'])
    
    def test_reproducibility_validation(self):
        """Test reproducibility package validation."""
        # Validate reproducibility components
        repro_validation = self.integration_validator.validate_reproducibility()
        
        self.assertIsInstance(repro_validation, dict)
        self.assertIn('is_valid', repro_validation)
        self.assertTrue(repro_validation['is_valid'])
        
        # Check reproducibility components
        self.assertIn('readme_instructions', repro_validation)
        self.assertTrue(repro_validation['readme_instructions']['exists'])
        
        self.assertIn('run_script', repro_validation)
        self.assertTrue(repro_validation['run_script']['exists'])
        self.assertTrue(repro_validation['run_script']['executable'])
        
        self.assertIn('dependency_files', repro_validation)
        self.assertTrue(repro_validation['dependency_files']['requirements_txt'])
    
    def test_model_artifacts_validation(self):
        """Test model artifacts and checkpoints validation."""
        # Validate model artifacts
        model_validation = self.integration_validator.validate_model_artifacts()
        
        self.assertIsInstance(model_validation, dict)
        self.assertIn('is_valid', model_validation)
        self.assertTrue(model_validation['is_valid'])
        
        # Check model files
        self.assertIn('model_files', model_validation)
        model_files = model_validation['model_files']
        
        self.assertTrue(model_files['model_checkpoint']['exists'])
        self.assertTrue(model_files['model_metadata']['exists'])
        
        # Check embeddings
        self.assertIn('embeddings', model_validation)
        embeddings = model_validation['embeddings']
        
        self.assertTrue(embeddings['directory_exists'])
        self.assertGreater(embeddings['file_count'], 0)
    
    def test_logging_and_metrics_validation(self):
        """Test logging and metrics validation."""
        # Validate logging structure
        logging_validation = self.integration_validator.validate_logging_structure()
        
        self.assertIsInstance(logging_validation, dict)
        self.assertIn('is_valid', logging_validation)
        self.assertTrue(logging_validation['is_valid'])
        
        # Check log directory
        self.assertIn('log_directory', logging_validation)
        self.assertTrue(logging_validation['log_directory']['exists'])
        
        # Check for log files
        self.assertIn('log_files', logging_validation)
        log_files = logging_validation['log_files']
        
        self.assertGreater(log_files['count'], 0)
        
        # Check for CV results
        if 'cv_results' in logging_validation:
            self.assertTrue(logging_validation['cv_results']['exists'])
    
    def test_notebook_validation(self):
        """Test notebook structure and content validation."""
        # Validate notebooks
        notebook_validation = self.integration_validator.validate_notebooks()
        
        self.assertIsInstance(notebook_validation, dict)
        self.assertIn('is_valid', notebook_validation)
        self.assertTrue(notebook_validation['is_valid'])
        
        # Check notebook directory
        self.assertIn('notebook_directory', notebook_validation)
        self.assertTrue(notebook_validation['notebook_directory']['exists'])
        
        # Check for notebooks
        self.assertIn('notebooks', notebook_validation)
        notebooks = notebook_validation['notebooks']
        
        self.assertGreater(notebooks['count'], 0)
        
        # Validate notebook format
        for notebook_info in notebooks['files']:
            self.assertTrue(notebook_info['valid_format'])
    
    def test_test_coverage_validation(self):
        """Test test coverage and structure validation."""
        # Validate test structure
        test_validation = self.integration_validator.validate_test_structure()
        
        self.assertIsInstance(test_validation, dict)
        self.assertIn('is_valid', test_validation)
        self.assertTrue(test_validation['is_valid'])
        
        # Check test directory
        self.assertIn('test_directory', test_validation)
        self.assertTrue(test_validation['test_directory']['exists'])
        
        # Check for test files
        self.assertIn('test_files', test_validation)
        test_files = test_validation['test_files']
        
        self.assertGreater(test_files['count'], 0)
        
        # Check test categories
        if 'test_categories' in test_validation:
            categories = test_validation['test_categories']
            self.assertIn('unit_tests', categories)
            self.assertIn('integration_tests', categories)
    
    def test_complete_integration_validation(self):
        """Test complete integration validation."""
        # Run complete integration validation
        integration_results = self.integration_validator.run_complete_validation()
        
        self.assertIsInstance(integration_results, dict)
        self.assertIn('overall_valid', integration_results)
        self.assertTrue(integration_results['overall_valid'])
        
        # Check all validation categories
        expected_categories = [
            'deliverable_structure', 'submission_format', 'code_structure',
            'reproducibility', 'model_artifacts', 'logging_structure'
        ]
        
        for category in expected_categories:
            self.assertIn(category, integration_results)
            self.assertTrue(integration_results[category]['is_valid'])
        
        # Check validation summary
        self.assertIn('validation_summary', integration_results)
        summary = integration_results['validation_summary']
        
        self.assertIn('total_checks', summary)
        self.assertIn('passed_checks', summary)
        self.assertIn('failed_checks', summary)
        
        self.assertEqual(summary['failed_checks'], 0)
        self.assertGreater(summary['total_checks'], 0)
        self.assertEqual(summary['passed_checks'], summary['total_checks'])
    
    def test_submission_package_creation(self):
        """Test creation of complete submission package."""
        # Create submission package
        package_results = self.deliverable_manager.create_submission_package()
        
        self.assertIsInstance(package_results, dict)
        self.assertIn('package_created', package_results)
        self.assertTrue(package_results['package_created'])
        
        # Check package contents
        self.assertIn('package_path', package_results)
        self.assertIn('included_files', package_results)
        
        package_path = Path(package_results['package_path'])
        self.assertTrue(package_path.exists())
        
        # Verify key files are included
        included_files = package_results['included_files']
        required_files = ['test_out.csv', 'methodology_1page.pdf', 'README.md', 'requirements.txt']
        
        for required_file in required_files:
            self.assertTrue(any(required_file in file_path for file_path in included_files))
    
    def test_compliance_report_generation(self):
        """Test compliance report generation."""
        # Generate compliance report
        compliance_report = self.deliverable_manager.generate_compliance_report()
        
        self.assertIsInstance(compliance_report, dict)
        self.assertIn('compliance_status', compliance_report)
        self.assertEqual(compliance_report['compliance_status'], 'COMPLIANT')
        
        # Check report sections
        expected_sections = [
            'deliverable_validation', 'format_compliance', 'dependency_validation',
            'reproducibility_check', 'quality_metrics'
        ]
        
        for section in expected_sections:
            self.assertIn(section, compliance_report)
        
        # Check quality metrics
        quality_metrics = compliance_report['quality_metrics']
        self.assertIn('overall_score', quality_metrics)
        self.assertGreaterEqual(quality_metrics['overall_score'], 80)  # Should be high quality
    
    def test_error_handling_missing_files(self):
        """Test error handling when required files are missing."""
        # Remove a required file
        (self.temp_path / "deliverables" / "test_out.csv").unlink()
        
        # Validation should fail gracefully
        validation_results = self.deliverable_manager.validate_deliverable_structure()
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn('is_valid', validation_results)
        self.assertFalse(validation_results['is_valid'])
        
        # Should identify missing file
        self.assertIn('missing_files', validation_results)
        missing_files = validation_results['missing_files']
        self.assertIn('test_out.csv', missing_files)
    
    def test_error_handling_invalid_format(self):
        """Test error handling with invalid file formats."""
        # Create invalid test_out.csv (wrong columns)
        invalid_data = pd.DataFrame({
            'wrong_column': ['1', '2', '3'],
            'another_wrong_column': [10.0, 20.0, 30.0]
        })
        invalid_data.to_csv(self.temp_path / "deliverables" / "test_out.csv", index=False)
        
        # Format validation should fail
        test_out_df = pd.read_csv(self.temp_path / "deliverables" / "test_out.csv")
        compliance_results = self.deliverable_manager.validate_submission_format(test_out_df)
        
        self.assertIsInstance(compliance_results, dict)
        self.assertIn('is_compliant', compliance_results)
        self.assertFalse(compliance_results['is_compliant'])
        
        # Should identify format issues
        self.assertIn('format_errors', compliance_results)
        format_errors = compliance_results['format_errors']
        self.assertGreater(len(format_errors), 0)
    
    def test_performance_validation(self):
        """Test performance and resource validation."""
        # Validate performance requirements
        performance_validation = self.integration_validator.validate_performance_requirements()
        
        self.assertIsInstance(performance_validation, dict)
        self.assertIn('is_valid', performance_validation)
        
        # Check performance metrics
        if 'performance_metrics' in performance_validation:
            metrics = performance_validation['performance_metrics']
            
            # Check inference timing if available
            if 'inference_time' in metrics:
                self.assertIsInstance(metrics['inference_time'], (int, float))
                self.assertGreater(metrics['inference_time'], 0)
            
            # Check memory usage if available
            if 'memory_usage' in metrics:
                self.assertIsInstance(metrics['memory_usage'], (int, float))
                self.assertGreater(metrics['memory_usage'], 0)
    
    def test_license_compliance_validation(self):
        """Test license compliance validation."""
        # Validate license compliance
        license_validation = self.integration_validator.validate_license_compliance()
        
        self.assertIsInstance(license_validation, dict)
        self.assertIn('is_compliant', license_validation)
        
        # Check license tracking
        if 'license_summary' in license_validation:
            license_summary = license_validation['license_summary']
            
            self.assertIn('total_dependencies', license_summary)
            self.assertIn('compliant_licenses', license_summary)
            
            # Should have some dependencies
            self.assertGreater(license_summary['total_dependencies'], 0)


if __name__ == '__main__':
    unittest.main()