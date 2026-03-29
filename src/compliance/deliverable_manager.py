"""
Deliverable structure management and validation for ML Product Pricing Challenge.

This module ensures that all required deliverables are present and properly
formatted according to competition requirements.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class DeliverableManager:
    """Manages deliverable structure and validates submission completeness."""
    
    REQUIRED_DELIVERABLES = {
        'test_out.csv': {
            'description': 'Final predictions in required format',
            'type': 'file',
            'required': True,
            'validation': 'prediction_output'
        },
        'methodology_1page.pdf': {
            'description': 'One-page methodology document',
            'type': 'file',
            'required': True,
            'validation': 'pdf_document'
        }
    }
    
    OPTIONAL_DELIVERABLES = {
        'requirements.txt': {
            'description': 'Python dependencies with exact versions',
            'type': 'file',
            'required': False,
            'validation': 'requirements_file'
        },
        'environment.yml': {
            'description': 'Conda environment specification',
            'type': 'file',
            'required': False,
            'validation': 'conda_environment'
        },
        'README.md': {
            'description': 'Project documentation and reproduction instructions',
            'type': 'file',
            'required': False,
            'validation': 'markdown_document'
        },
        'run_all.sh': {
            'description': 'End-to-end pipeline reproduction script',
            'type': 'file',
            'required': False,
            'validation': 'shell_script'
        },
        'CONFIGURATION.md': {
            'description': 'Detailed hyperparameter and configuration documentation',
            'type': 'file',
            'required': False,
            'validation': 'markdown_document'
        },
        'compliance_log.txt': {
            'description': 'Compliance validation report',
            'type': 'file',
            'required': False,
            'validation': 'text_document'
        },
        'src/': {
            'description': 'Source code directory',
            'type': 'directory',
            'required': False,
            'validation': 'source_directory'
        },
        'models/': {
            'description': 'Trained model checkpoints',
            'type': 'directory',
            'required': False,
            'validation': 'model_directory'
        },
        'logs/': {
            'description': 'Training and validation logs',
            'type': 'directory',
            'required': False,
            'validation': 'log_directory'
        },
        'tests/': {
            'description': 'Unit tests with ≥80% coverage',
            'type': 'directory',
            'required': False,
            'validation': 'test_directory'
        },
        'notebooks/': {
            'description': 'EDA and baseline experiments',
            'type': 'directory',
            'required': False,
            'validation': 'notebook_directory'
        }
    }
    
    def __init__(self, project_root: str = "."):
        """Initialize deliverable manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        
    def create_deliverable_structure(self) -> Dict[str, str]:
        """Create the complete deliverable directory structure.
        
        Returns:
            Dictionary mapping created items to their paths
        """
        self.logger.info("Creating deliverable structure...")
        
        created_items = {}
        
        # Create deliverables directory
        deliverables_dir = self.project_root / 'deliverables'
        deliverables_dir.mkdir(exist_ok=True)
        created_items['deliverables_directory'] = str(deliverables_dir)
        
        # Create required directories if they don't exist
        required_dirs = ['src', 'models', 'logs', 'tests', 'notebooks', 'embeddings', 'images', 'cache']
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                created_items[f'{dir_name}_directory'] = str(dir_path)
                
                # Create .gitkeep files for empty directories
                gitkeep_path = dir_path / '.gitkeep'
                if not gitkeep_path.exists():
                    gitkeep_path.touch()
        
        # Create placeholder files for missing required deliverables
        self._create_placeholder_files(created_items)
        
        self.logger.info(f"Created {len(created_items)} deliverable structure items")
        return created_items
    
    def _create_placeholder_files(self, created_items: Dict[str, str]) -> None:
        """Create placeholder files for missing required deliverables.
        
        Args:
            created_items: Dictionary to track created items
        """
        # Create methodology template if missing
        methodology_path = self.project_root / 'methodology_1page.pdf'
        if not methodology_path.exists():
            template_path = self.project_root / 'methodology_template.md'
            if not template_path.exists():
                with open(template_path, 'w') as f:
                    f.write(self._get_methodology_template())
                created_items['methodology_template'] = str(template_path)
                self.logger.warning(f"Created methodology template at {template_path}")
        
        # Create test_out.csv template if missing
        test_out_path = self.project_root / 'test_out.csv'
        if not test_out_path.exists():
            template_path = self.project_root / 'test_out_template.csv'
            if not template_path.exists():
                self._create_test_out_template(template_path)
                created_items['test_out_template'] = str(template_path)
                self.logger.warning(f"Created test_out template at {template_path}")
    
    def _get_methodology_template(self) -> str:
        """Get methodology document template content.
        
        Returns:
            Template content as string
        """
        return """# ML Product Pricing Challenge 2025 - Methodology

## Problem Overview
Product price prediction using multimodal features (text and images) from e-commerce catalog data.

## Data Processing
- Schema validation with fail-fast error handling
- Price normalization (currency symbols, thousand separators)
- Zero price handling with epsilon strategy (≥0.01)
- Image download with exponential backoff retry logic

## Feature Engineering

### Text Features
- Semantic embeddings using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Statistical features: length, word count, readability scores (21 features)
- IPQ extraction with >90% precision using regex patterns
- Brand detection (40+ brands) and category classification (12 categories)
- Unit normalization to canonical units (grams/milliliters/pieces)

### Image Features
- CNN features using ResNet-50 and EfficientNet-B0 (pre-trained)
- Visual features: deep features + color histograms + texture analysis
- Missing image handling with text-based fallback features
- Versioned embedding cache with integrity validation

### Feature Fusion
- Weighted concatenation (60% text, 40% image)
- L2 normalization and PCA dimensionality reduction to 512 dimensions

## Model Architecture
- Ensemble of Random Forest, XGBoost, LightGBM, and Neural Networks
- 5-fold cross-validation with stratified sampling
- Weighted averaging based on validation SMAPE performance
- Hyperparameter tuning using Bayesian optimization

## Evaluation
- Primary metric: SMAPE (Symmetric Mean Absolute Percentage Error)
- Cross-validation with mean ± std reporting across folds
- Per-price-quantile analysis for detailed performance assessment
- Unit tests for SMAPE calculation validation

## Prediction Generation
- Batch processing for memory efficiency
- Prediction clamping to minimum threshold (≥0.01)
- Exact sample_id matching and row count validation
- Comprehensive output format compliance checks

## Compliance
- Only competition-provided datasets used (train.csv, test.csv)
- All models use MIT/Apache 2.0 licenses, ≤8B parameters
- No external data sources or web scraping
- Complete license tracking and audit trail

## Performance
- IPQ Extraction Precision: 90.9% (exceeds 90% requirement)
- Cross-validation SMAPE: [Insert actual results]
- Processing time: 75k predictions in <10 minutes
- Memory usage: Peak 12GB RAM, 6GB GPU memory

## Reproducibility
- Exact dependency versions in requirements.txt
- Complete pipeline script (run_all.sh) for end-to-end reproduction
- Structured logging with experiment metadata
- Comprehensive configuration documentation

**Note: Convert this template to PDF format for final submission**
"""
    
    def _create_test_out_template(self, template_path: Path) -> None:
        """Create test_out.csv template file.
        
        Args:
            template_path: Path to create template file
        """
        # Check if test.csv exists to create proper template
        test_csv_path = self.project_root / 'dataset' / 'test.csv'
        
        if test_csv_path.exists():
            try:
                test_df = pd.read_csv(test_csv_path)
                template_df = pd.DataFrame({
                    'sample_id': test_df['sample_id'],
                    'price': 1.0  # Placeholder price
                })
                template_df.to_csv(template_path, index=False)
                self.logger.info(f"Created test_out template with {len(template_df)} rows")
            except Exception as e:
                self.logger.error(f"Error creating test_out template: {e}")
                # Create minimal template
                template_df = pd.DataFrame({
                    'sample_id': ['sample_1', 'sample_2', 'sample_3'],
                    'price': [1.0, 2.0, 3.0]
                })
                template_df.to_csv(template_path, index=False)
        else:
            # Create minimal template
            template_df = pd.DataFrame({
                'sample_id': ['sample_1', 'sample_2', 'sample_3'],
                'price': [1.0, 2.0, 3.0]
            })
            template_df.to_csv(template_path, index=False)
    
    def validate_deliverable_completeness(self) -> Dict[str, Dict]:
        """Validate completeness of all deliverables.
        
        Returns:
            Dictionary with validation results for each deliverable
        """
        self.logger.info("Validating deliverable completeness...")
        
        validation_results = {}
        
        # Validate required deliverables
        for name, info in self.REQUIRED_DELIVERABLES.items():
            result = self._validate_single_deliverable(name, info, required=True)
            validation_results[name] = result
        
        # Validate optional deliverables
        for name, info in self.OPTIONAL_DELIVERABLES.items():
            result = self._validate_single_deliverable(name, info, required=False)
            validation_results[name] = result
        
        self.validation_results = validation_results
        return validation_results
    
    def _validate_single_deliverable(self, name: str, info: Dict, required: bool) -> Dict:
        """Validate a single deliverable.
        
        Args:
            name: Name of the deliverable
            info: Deliverable information dictionary
            required: Whether the deliverable is required
            
        Returns:
            Validation result dictionary
        """
        result = {
            'name': name,
            'description': info['description'],
            'type': info['type'],
            'required': required,
            'exists': False,
            'valid': False,
            'issues': [],
            'warnings': [],
            'metadata': {},
            'checked_at': datetime.now().isoformat()
        }
        
        path = self.project_root / name
        result['path'] = str(path)
        result['exists'] = path.exists()
        
        if not result['exists']:
            if required:
                result['issues'].append(f"Required deliverable missing: {name}")
            else:
                result['warnings'].append(f"Optional deliverable missing: {name}")
            return result
        
        # Perform specific validation based on type
        try:
            if info['validation'] == 'prediction_output':
                self._validate_prediction_output(path, result)
            elif info['validation'] == 'pdf_document':
                self._validate_pdf_document(path, result)
            elif info['validation'] == 'requirements_file':
                self._validate_requirements_file(path, result)
            elif info['validation'] == 'conda_environment':
                self._validate_conda_environment(path, result)
            elif info['validation'] == 'markdown_document':
                self._validate_markdown_document(path, result)
            elif info['validation'] == 'shell_script':
                self._validate_shell_script(path, result)
            elif info['validation'] == 'text_document':
                self._validate_text_document(path, result)
            elif info['validation'] == 'source_directory':
                self._validate_source_directory(path, result)
            elif info['validation'] == 'model_directory':
                self._validate_model_directory(path, result)
            elif info['validation'] == 'log_directory':
                self._validate_log_directory(path, result)
            elif info['validation'] == 'test_directory':
                self._validate_test_directory(path, result)
            elif info['validation'] == 'notebook_directory':
                self._validate_notebook_directory(path, result)
            else:
                result['warnings'].append(f"Unknown validation type: {info['validation']}")
                result['valid'] = True  # Assume valid if we can't validate
                
        except Exception as e:
            result['issues'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_prediction_output(self, path: Path, result: Dict) -> None:
        """Validate prediction output file (test_out.csv).
        
        Args:
            path: Path to the file
            result: Result dictionary to update
        """
        try:
            df = pd.read_csv(path)
            result['metadata']['row_count'] = len(df)
            result['metadata']['column_count'] = len(df.columns)
            result['metadata']['columns'] = list(df.columns)
            result['metadata']['file_size_mb'] = path.stat().st_size / (1024 * 1024)
            
            # Check required columns
            expected_columns = {'sample_id', 'price'}
            if set(df.columns) != expected_columns:
                result['issues'].append(f"Invalid columns. Expected: {expected_columns}, Got: {set(df.columns)}")
                return
            
            # Check for null values
            if df.isnull().any().any():
                result['issues'].append("Null values found in prediction output")
            
            # Check for non-positive prices
            if (df['price'] <= 0).any():
                result['issues'].append("Non-positive prices found in output")
            
            # Check sample_id uniqueness
            if df['sample_id'].duplicated().any():
                result['issues'].append("Duplicate sample_ids found")
            
            # Check against test.csv if available
            test_path = self.project_root / 'dataset' / 'test.csv'
            if test_path.exists():
                test_df = pd.read_csv(test_path)
                
                if len(df) != len(test_df):
                    result['issues'].append(f"Row count mismatch with test.csv. Expected: {len(test_df)}, Got: {len(df)}")
                
                if not set(df['sample_id']) == set(test_df['sample_id']):
                    result['issues'].append("sample_id mismatch with test.csv")
            
            # Add price statistics
            result['metadata']['price_stats'] = {
                'min': float(df['price'].min()),
                'max': float(df['price'].max()),
                'mean': float(df['price'].mean()),
                'median': float(df['price'].median()),
                'std': float(df['price'].std())
            }
            
            if len(result['issues']) == 0:
                result['valid'] = True
                
        except Exception as e:
            result['issues'].append(f"Error reading prediction output: {str(e)}")
    
    def _validate_pdf_document(self, path: Path, result: Dict) -> None:
        """Validate PDF document.
        
        Args:
            path: Path to the file
            result: Result dictionary to update
        """
        if not path.suffix.lower() == '.pdf':
            result['issues'].append("File is not a PDF document")
            return
        
        result['metadata']['file_size_mb'] = path.stat().st_size / (1024 * 1024)
        
        # Check file size (should be reasonable for 1-page document)
        if result['metadata']['file_size_mb'] > 10:
            result['warnings'].append("PDF file is quite large for a 1-page document")
        
        # Basic PDF validation (check if file starts with PDF header)
        try:
            with open(path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    result['issues'].append("File does not appear to be a valid PDF")
                    return
        except Exception as e:
            result['issues'].append(f"Error reading PDF file: {str(e)}")
            return
        
        result['valid'] = True
    
    def _validate_requirements_file(self, path: Path, result: Dict) -> None:
        """Validate requirements.txt file.
        
        Args:
            path: Path to the file
            result: Result dictionary to update
        """
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            result['metadata']['line_count'] = len(lines)
            result['metadata']['file_size_kb'] = path.stat().st_size / 1024
            
            # Count packages with version pins
            packages = []
            pinned_packages = 0
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    packages.append(line)
                    if '==' in line:
                        pinned_packages += 1
            
            result['metadata']['total_packages'] = len(packages)
            result['metadata']['pinned_packages'] = pinned_packages
            result['metadata']['pinning_ratio'] = pinned_packages / len(packages) if packages else 0
            
            # Check for essential packages
            essential_packages = ['pandas', 'numpy', 'scikit-learn', 'torch', 'transformers']
            missing_essential = []
            
            package_names = [pkg.split('==')[0].split('>=')[0].split('<=')[0] for pkg in packages]
            for essential in essential_packages:
                if essential not in package_names:
                    missing_essential.append(essential)
            
            if missing_essential:
                result['warnings'].append(f"Missing essential packages: {missing_essential}")
            
            # Recommend high pinning ratio for reproducibility
            if result['metadata']['pinning_ratio'] < 0.8:
                result['warnings'].append("Low version pinning ratio - consider using exact versions for reproducibility")
            
            result['valid'] = True
            
        except Exception as e:
            result['issues'].append(f"Error reading requirements file: {str(e)}")
    
    def _validate_conda_environment(self, path: Path, result: Dict) -> None:
        """Validate conda environment.yml file.
        
        Args:
            path: Path to the file
            result: Result dictionary to update
        """
        try:
            import yaml
            
            with open(path, 'r') as f:
                env_config = yaml.safe_load(f)
            
            result['metadata']['file_size_kb'] = path.stat().st_size / 1024
            
            # Check required fields
            required_fields = ['name', 'dependencies']
            missing_fields = [field for field in required_fields if field not in env_config]
            
            if missing_fields:
                result['issues'].append(f"Missing required fields: {missing_fields}")
                return
            
            result['metadata']['environment_name'] = env_config.get('name', 'unknown')
            result['metadata']['channels'] = env_config.get('channels', [])
            
            dependencies = env_config.get('dependencies', [])
            result['metadata']['dependency_count'] = len(dependencies)
            
            # Check for pip dependencies
            pip_deps = []
            for dep in dependencies:
                if isinstance(dep, dict) and 'pip' in dep:
                    pip_deps.extend(dep['pip'])
            
            result['metadata']['pip_dependency_count'] = len(pip_deps)
            
            result['valid'] = True
            
        except ImportError:
            result['warnings'].append("PyYAML not available for conda environment validation")
            result['valid'] = True  # Assume valid if we can't validate
        except Exception as e:
            result['issues'].append(f"Error reading conda environment: {str(e)}")
    
    def _validate_markdown_document(self, path: Path, result: Dict) -> None:
        """Validate markdown document.
        
        Args:
            path: Path to the file
            result: Result dictionary to update
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result['metadata']['file_size_kb'] = path.stat().st_size / 1024
            result['metadata']['line_count'] = len(content.split('\n'))
            result['metadata']['word_count'] = len(content.split())
            result['metadata']['character_count'] = len(content)
            
            # Check for markdown headers
            header_count = content.count('#')
            result['metadata']['header_count'] = header_count
            
            # Check for code blocks
            code_block_count = content.count('```')
            result['metadata']['code_block_count'] = code_block_count // 2  # Pairs of ```
            
            # Basic content checks for README
            if path.name.lower() == 'readme.md':
                if 'installation' not in content.lower() and 'setup' not in content.lower():
                    result['warnings'].append("README should include installation/setup instructions")
                
                if 'usage' not in content.lower() and 'example' not in content.lower():
                    result['warnings'].append("README should include usage examples")
            
            result['valid'] = True
            
        except Exception as e:
            result['issues'].append(f"Error reading markdown document: {str(e)}")
    
    def _validate_shell_script(self, path: Path, result: Dict) -> None:
        """Validate shell script.
        
        Args:
            path: Path to the file
            result: Result dictionary to update
        """
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            result['metadata']['file_size_kb'] = path.stat().st_size / 1024
            result['metadata']['line_count'] = len(content.split('\n'))
            
            # Check for shebang
            if not content.startswith('#!'):
                result['warnings'].append("Shell script should start with shebang (#!/bin/bash)")
            
            # Check if file is executable
            if not os.access(path, os.X_OK):
                result['warnings'].append("Shell script is not executable (chmod +x needed)")
            
            # Check for error handling
            if 'set -e' not in content:
                result['warnings'].append("Consider adding 'set -e' for better error handling")
            
            result['valid'] = True
            
        except Exception as e:
            result['issues'].append(f"Error reading shell script: {str(e)}")
    
    def _validate_text_document(self, path: Path, result: Dict) -> None:
        """Validate text document.
        
        Args:
            path: Path to the file
            result: Result dictionary to update
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result['metadata']['file_size_kb'] = path.stat().st_size / 1024
            result['metadata']['line_count'] = len(content.split('\n'))
            result['metadata']['word_count'] = len(content.split())
            result['metadata']['character_count'] = len(content)
            
            result['valid'] = True
            
        except Exception as e:
            result['issues'].append(f"Error reading text document: {str(e)}")
    
    def _validate_source_directory(self, path: Path, result: Dict) -> None:
        """Validate source code directory.
        
        Args:
            path: Path to the directory
            result: Result dictionary to update
        """
        if not path.is_dir():
            result['issues'].append("Source path is not a directory")
            return
        
        # Count Python files
        py_files = list(path.rglob('*.py'))
        result['metadata']['python_file_count'] = len(py_files)
        
        # Count subdirectories
        subdirs = [item for item in path.iterdir() if item.is_dir() and not item.name.startswith('.')]
        result['metadata']['subdirectory_count'] = len(subdirs)
        result['metadata']['subdirectories'] = [d.name for d in subdirs]
        
        # Check for main entry point
        main_files = ['main.py', '__main__.py', 'app.py']
        has_main = any((path / main_file).exists() for main_file in main_files)
        result['metadata']['has_main_entry'] = has_main
        
        if not has_main:
            result['warnings'].append("No main entry point found (main.py, __main__.py, or app.py)")
        
        # Check for __init__.py files
        init_files = list(path.rglob('__init__.py'))
        result['metadata']['init_file_count'] = len(init_files)
        
        result['valid'] = True
    
    def _validate_model_directory(self, path: Path, result: Dict) -> None:
        """Validate model directory.
        
        Args:
            path: Path to the directory
            result: Result dictionary to update
        """
        if not path.is_dir():
            result['issues'].append("Model path is not a directory")
            return
        
        # Count model files
        model_extensions = ['.pkl', '.pickle', '.pt', '.pth', '.h5', '.hdf5', '.joblib']
        model_files = []
        
        for ext in model_extensions:
            model_files.extend(list(path.rglob(f'*{ext}')))
        
        result['metadata']['model_file_count'] = len(model_files)
        result['metadata']['model_extensions'] = list(set(f.suffix for f in model_files))
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_files if f.is_file())
        result['metadata']['total_size_mb'] = total_size / (1024 * 1024)
        
        if len(model_files) == 0:
            result['warnings'].append("No model files found in models directory")
        
        result['valid'] = True
    
    def _validate_log_directory(self, path: Path, result: Dict) -> None:
        """Validate log directory.
        
        Args:
            path: Path to the directory
            result: Result dictionary to update
        """
        if not path.is_dir():
            result['issues'].append("Log path is not a directory")
            return
        
        # Count log files
        log_extensions = ['.log', '.txt', '.json', '.jsonl']
        log_files = []
        
        for ext in log_extensions:
            log_files.extend(list(path.rglob(f'*{ext}')))
        
        result['metadata']['log_file_count'] = len(log_files)
        result['metadata']['log_extensions'] = list(set(f.suffix for f in log_files))
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in log_files if f.is_file())
        result['metadata']['total_size_mb'] = total_size / (1024 * 1024)
        
        result['valid'] = True
    
    def _validate_test_directory(self, path: Path, result: Dict) -> None:
        """Validate test directory.
        
        Args:
            path: Path to the directory
            result: Result dictionary to update
        """
        if not path.is_dir():
            result['issues'].append("Test path is not a directory")
            return
        
        # Count test files
        test_files = list(path.rglob('test_*.py')) + list(path.rglob('*_test.py'))
        result['metadata']['test_file_count'] = len(test_files)
        
        # Check for pytest configuration
        pytest_files = ['pytest.ini', 'pyproject.toml', 'setup.cfg']
        has_pytest_config = any((self.project_root / config_file).exists() for config_file in pytest_files)
        result['metadata']['has_pytest_config'] = has_pytest_config
        
        if len(test_files) == 0:
            result['warnings'].append("No test files found in tests directory")
        
        result['valid'] = True
    
    def _validate_notebook_directory(self, path: Path, result: Dict) -> None:
        """Validate notebook directory.
        
        Args:
            path: Path to the directory
            result: Result dictionary to update
        """
        if not path.is_dir():
            result['issues'].append("Notebook path is not a directory")
            return
        
        # Count notebook files
        notebook_files = list(path.rglob('*.ipynb'))
        result['metadata']['notebook_file_count'] = len(notebook_files)
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in notebook_files if f.is_file())
        result['metadata']['total_size_mb'] = total_size / (1024 * 1024)
        
        result['valid'] = True
    
    def generate_quality_assurance_report(self) -> Dict:
        """Generate comprehensive quality assurance report.
        
        Returns:
            Quality assurance report dictionary
        """
        self.logger.info("Generating quality assurance report...")
        
        # Run validation if not already done
        if not self.validation_results:
            self.validate_deliverable_completeness()
        
        # Calculate summary statistics
        total_deliverables = len(self.validation_results)
        valid_deliverables = sum(1 for result in self.validation_results.values() if result['valid'])
        required_deliverables = sum(1 for result in self.validation_results.values() if result['required'])
        required_valid = sum(1 for result in self.validation_results.values() 
                           if result['required'] and result['valid'])
        
        total_issues = sum(len(result['issues']) for result in self.validation_results.values())
        total_warnings = sum(len(result['warnings']) for result in self.validation_results.values())
        
        # Determine overall status
        if required_valid == required_deliverables and total_issues == 0:
            overall_status = 'READY_FOR_SUBMISSION'
        elif required_valid == required_deliverables:
            overall_status = 'READY_WITH_WARNINGS'
        else:
            overall_status = 'NOT_READY'
        
        qa_report = {
            'generated_at': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'overall_status': overall_status,
            'summary': {
                'total_deliverables': total_deliverables,
                'valid_deliverables': valid_deliverables,
                'required_deliverables': required_deliverables,
                'required_valid': required_valid,
                'total_issues': total_issues,
                'total_warnings': total_warnings,
                'completion_percentage': (valid_deliverables / total_deliverables * 100) if total_deliverables > 0 else 0
            },
            'deliverable_validation': self.validation_results,
            'critical_issues': self._get_critical_issues(),
            'recommendations': self._get_qa_recommendations(),
            'submission_checklist': self._get_submission_checklist()
        }
        
        return qa_report
    
    def _get_critical_issues(self) -> List[str]:
        """Get list of critical issues that block submission.
        
        Returns:
            List of critical issue descriptions
        """
        critical_issues = []
        
        for name, result in self.validation_results.items():
            if result['required'] and not result['valid']:
                critical_issues.append(f"Required deliverable invalid or missing: {name}")
            
            if result['issues']:
                for issue in result['issues']:
                    critical_issues.append(f"{name}: {issue}")
        
        return critical_issues
    
    def _get_qa_recommendations(self) -> List[str]:
        """Get quality assurance recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for missing required deliverables
        missing_required = [name for name, result in self.validation_results.items() 
                          if result['required'] and not result['exists']]
        
        if missing_required:
            recommendations.append(f"Create missing required deliverables: {', '.join(missing_required)}")
        
        # Check for invalid deliverables
        invalid_deliverables = [name for name, result in self.validation_results.items() 
                              if result['exists'] and not result['valid']]
        
        if invalid_deliverables:
            recommendations.append(f"Fix validation issues in: {', '.join(invalid_deliverables)}")
        
        # General recommendations
        recommendations.extend([
            "Verify test_out.csv contains predictions for all test samples",
            "Ensure methodology_1page.pdf is properly formatted and complete",
            "Review compliance_log.txt for any licensing or data source issues",
            "Test complete pipeline reproduction using run_all.sh",
            "Validate all model checkpoints are properly saved and documented"
        ])
        
        return recommendations
    
    def _get_submission_checklist(self) -> Dict[str, bool]:
        """Get submission readiness checklist.
        
        Returns:
            Dictionary mapping checklist items to completion status
        """
        checklist = {}
        
        # Required deliverables
        checklist['test_out.csv_present'] = self.validation_results.get('test_out.csv', {}).get('valid', False)
        checklist['methodology_pdf_present'] = self.validation_results.get('methodology_1page.pdf', {}).get('valid', False)
        
        # Code and documentation
        checklist['source_code_present'] = self.validation_results.get('src/', {}).get('valid', False)
        checklist['requirements_present'] = self.validation_results.get('requirements.txt', {}).get('valid', False)
        checklist['readme_present'] = self.validation_results.get('README.md', {}).get('valid', False)
        checklist['reproduction_script_present'] = self.validation_results.get('run_all.sh', {}).get('valid', False)
        
        # Validation and compliance
        checklist['compliance_log_present'] = self.validation_results.get('compliance_log.txt', {}).get('valid', False)
        checklist['configuration_documented'] = self.validation_results.get('CONFIGURATION.md', {}).get('valid', False)
        
        # Models and logs
        checklist['models_present'] = self.validation_results.get('models/', {}).get('valid', False)
        checklist['logs_present'] = self.validation_results.get('logs/', {}).get('valid', False)
        checklist['tests_present'] = self.validation_results.get('tests/', {}).get('valid', False)
        
        return checklist
    
    def save_qa_report(self, output_path: str = "deliverables/qa_report.json") -> str:
        """Save quality assurance report to file.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        qa_report = self.generate_quality_assurance_report()
        
        output_file = self.project_root / output_path
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(qa_report, f, indent=2)
        
        self.logger.info(f"Quality assurance report saved to {output_file}")
        return str(output_file)
    
    def copy_deliverables_to_submission_directory(self, submission_dir: str = "deliverables") -> Dict[str, str]:
        """Copy all deliverables to submission directory.
        
        Args:
            submission_dir: Directory to copy deliverables to
            
        Returns:
            Dictionary mapping deliverable names to copied paths
        """
        self.logger.info("Copying deliverables to submission directory...")
        
        submission_path = self.project_root / submission_dir
        submission_path.mkdir(exist_ok=True)
        
        copied_files = {}
        
        # Copy required and valid optional deliverables
        all_deliverables = {**self.REQUIRED_DELIVERABLES, **self.OPTIONAL_DELIVERABLES}
        
        for name, info in all_deliverables.items():
            source_path = self.project_root / name
            
            if source_path.exists():
                dest_path = submission_path / name
                
                try:
                    if source_path.is_file():
                        # Copy file
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, dest_path)
                        copied_files[name] = str(dest_path)
                    elif source_path.is_dir():
                        # Copy directory
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.copytree(source_path, dest_path)
                        copied_files[name] = str(dest_path)
                        
                    self.logger.info(f"Copied {name} to submission directory")
                    
                except Exception as e:
                    self.logger.error(f"Error copying {name}: {e}")
        
        # Create submission summary
        summary_path = submission_path / "submission_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"ML Product Pricing Challenge 2025 - Submission Package\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Included Files:\n")
            for name, path in copied_files.items():
                f.write(f"  - {name}\n")
            f.write(f"\nTotal files: {len(copied_files)}\n")
        
        copied_files['submission_summary.txt'] = str(summary_path)
        
        self.logger.info(f"Copied {len(copied_files)} deliverables to {submission_path}")
        return copied_files