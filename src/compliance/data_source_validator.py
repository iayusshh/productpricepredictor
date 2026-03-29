"""
Data source validation for ML Product Pricing Challenge.

This module ensures that only allowed data sources are used in the solution,
maintaining compliance with competition rules.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


class DataSourceValidator:
    """Validates that only allowed data sources are used in the solution."""
    
    ALLOWED_DATA_FILES = {
        'dataset/train.csv',
        'dataset/test.csv',
        'dataset/sample_test_out.csv'
    }
    
    ALLOWED_DATA_DIRECTORIES = {
        'dataset',
        'images',
        'cache',
        'embeddings',
        'models',
        'logs'
    }
    
    PROHIBITED_PATTERNS = [
        'external_data',
        'scraped_data',
        'api_data',
        'web_data',
        'crawled_data',
        'third_party_data'
    ]
    
    def __init__(self, project_root: str = "."):
        """Initialize data source validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        self.validation_log = []
        
    def validate_data_files(self) -> Dict[str, Dict]:
        """Validate all data files in the project.
        
        Returns:
            Dictionary with validation results for each data file
        """
        validation_results = {}
        
        # Find all data files
        data_extensions = ['.csv', '.json', '.parquet', '.pkl', '.pickle']
        data_files = []
        
        for ext in data_extensions:
            data_files.extend(self.project_root.rglob(f'*{ext}'))
        
        for data_file in data_files:
            relative_path = data_file.relative_to(self.project_root)
            is_allowed, reason = self._validate_single_file(relative_path)
            
            file_info = {
                'path': str(relative_path),
                'size_mb': data_file.stat().st_size / (1024 * 1024),
                'is_allowed': is_allowed,
                'reason': reason,
                'checked_at': datetime.now().isoformat()
            }
            
            # Additional checks for CSV files
            if data_file.suffix.lower() == '.csv':
                csv_info = self._analyze_csv_file(data_file)
                file_info.update(csv_info)
            
            validation_results[str(relative_path)] = file_info
            
            # Log validation entry
            self.validation_log.append({
                'type': 'data_file_validation',
                'file': str(relative_path),
                'is_allowed': is_allowed,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            })
        
        return validation_results
    
    def _validate_single_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate a single data file.
        
        Args:
            file_path: Path to the file (relative to project root)
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        file_str = str(file_path).replace('\\', '/')
        
        # Check if it's an explicitly allowed file
        if file_str in self.ALLOWED_DATA_FILES:
            return True, "Explicitly allowed competition data file"
        
        # Check if it's in an allowed directory
        parts = file_path.parts
        if parts and parts[0] in self.ALLOWED_DATA_DIRECTORIES:
            # Additional checks for generated/cached files
            if parts[0] in ['cache', 'embeddings', 'models', 'logs']:
                return True, f"Generated/cached file in allowed directory: {parts[0]}"
            elif parts[0] == 'images':
                return True, "Downloaded product images (allowed)"
            elif parts[0] == 'dataset':
                return True, "File in dataset directory"
        
        # Check for prohibited patterns
        for pattern in self.PROHIBITED_PATTERNS:
            if pattern in file_str.lower():
                return False, f"Contains prohibited pattern: {pattern}"
        
        # Check file naming patterns that might indicate external data
        filename = file_path.name.lower()
        suspicious_names = [
            'external', 'scraped', 'crawled', 'api_data', 'web_data',
            'third_party', 'additional_data', 'extra_data'
        ]
        
        for suspicious in suspicious_names:
            if suspicious in filename:
                return False, f"Suspicious filename pattern: {suspicious}"
        
        # If it's a small file in src/ or tests/, it's probably code-related
        if parts and parts[0] in ['src', 'tests', 'notebooks']:
            return True, "Code-related data file"
        
        # Unknown file - flag for manual review
        return False, "Unknown data file requires manual review"
    
    def _analyze_csv_file(self, csv_path: Path) -> Dict:
        """Analyze a CSV file for additional validation.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Dictionary with CSV analysis results
        """
        analysis = {
            'row_count': 0,
            'column_count': 0,
            'columns': [],
            'has_sample_id': False,
            'has_price': False,
            'analysis_error': None
        }
        
        try:
            # Read just the header and first few rows for analysis
            df = pd.read_csv(csv_path, nrows=5)
            
            analysis['row_count'] = len(pd.read_csv(csv_path))  # Get actual row count
            analysis['column_count'] = len(df.columns)
            analysis['columns'] = list(df.columns)
            analysis['has_sample_id'] = 'sample_id' in df.columns
            analysis['has_price'] = 'price' in df.columns
            
            # Check if it matches expected competition data structure
            if set(df.columns) == {'sample_id', 'catalog_content', 'image_link', 'price'}:
                analysis['file_type'] = 'training_data'
            elif set(df.columns) == {'sample_id', 'catalog_content', 'image_link'}:
                analysis['file_type'] = 'test_data'
            elif set(df.columns) == {'sample_id', 'price'}:
                analysis['file_type'] = 'prediction_output'
            else:
                analysis['file_type'] = 'unknown'
                
        except Exception as e:
            analysis['analysis_error'] = str(e)
            self.logger.warning(f"Error analyzing CSV file {csv_path}: {e}")
        
        return analysis
    
    def check_data_loading_code(self) -> Dict[str, List[str]]:
        """Check code files for data loading patterns.
        
        Returns:
            Dictionary mapping files to suspicious data loading patterns
        """
        suspicious_files = {}
        
        # Find all Python files
        python_files = list(self.project_root.rglob('*.py'))
        
        # Patterns that might indicate external data usage
        external_patterns = [
            'requests.get',
            'urllib.request',
            'wget',
            'curl',
            'api_key',
            'api_token',
            'scrape',
            'crawl',
            'external_url',
            'download_from',
            'fetch_data',
            'web_scraping'
        ]
        
        # Patterns for reading non-competition data
        data_patterns = [
            'pd.read_csv.*external',
            'pd.read_csv.*additional',
            'pd.read_csv.*extra',
            'load.*external.*data',
            'open.*external',
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                found_patterns = []
                
                # Check for external data patterns
                for pattern in external_patterns:
                    if pattern in content.lower():
                        found_patterns.append(f"external_pattern: {pattern}")
                
                # Check for suspicious data loading patterns
                import re
                for pattern in data_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        found_patterns.append(f"data_pattern: {pattern}")
                
                if found_patterns:
                    relative_path = py_file.relative_to(self.project_root)
                    suspicious_files[str(relative_path)] = found_patterns
                    
                    # Log validation entry
                    self.validation_log.append({
                        'type': 'code_validation',
                        'file': str(relative_path),
                        'patterns_found': found_patterns,
                        'requires_review': True,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error checking file {py_file}: {e}")
        
        return suspicious_files
    
    def validate_dataset_integrity(self) -> Dict[str, Dict]:
        """Validate the integrity of competition datasets.
        
        Returns:
            Dictionary with dataset validation results
        """
        validation_results = {}
        
        # Check training data
        train_path = self.project_root / 'dataset' / 'train.csv'
        if train_path.exists():
            train_validation = self._validate_competition_dataset(train_path, 'training')
            validation_results['train.csv'] = train_validation
        else:
            validation_results['train.csv'] = {
                'exists': False,
                'error': 'Training dataset not found'
            }
        
        # Check test data
        test_path = self.project_root / 'dataset' / 'test.csv'
        if test_path.exists():
            test_validation = self._validate_competition_dataset(test_path, 'test')
            validation_results['test.csv'] = test_validation
        else:
            validation_results['test.csv'] = {
                'exists': False,
                'error': 'Test dataset not found'
            }
        
        # Check sample output
        sample_path = self.project_root / 'dataset' / 'sample_test_out.csv'
        if sample_path.exists():
            sample_validation = self._validate_competition_dataset(sample_path, 'sample_output')
            validation_results['sample_test_out.csv'] = sample_validation
        else:
            validation_results['sample_test_out.csv'] = {
                'exists': False,
                'error': 'Sample output not found'
            }
        
        return validation_results
    
    def _validate_competition_dataset(self, dataset_path: Path, dataset_type: str) -> Dict:
        """Validate a competition dataset file.
        
        Args:
            dataset_path: Path to dataset file
            dataset_type: Type of dataset ('training', 'test', 'sample_output')
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'exists': True,
            'path': str(dataset_path),
            'dataset_type': dataset_type,
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'checked_at': datetime.now().isoformat()
        }
        
        try:
            df = pd.read_csv(dataset_path)
            validation['row_count'] = len(df)
            validation['column_count'] = len(df.columns)
            validation['columns'] = list(df.columns)
            
            # Validate based on dataset type
            if dataset_type == 'training':
                expected_columns = {'sample_id', 'catalog_content', 'image_link', 'price'}
                if set(df.columns) != expected_columns:
                    validation['errors'].append(f"Invalid columns. Expected: {expected_columns}, Got: {set(df.columns)}")
                else:
                    validation['is_valid'] = True
                    
                # Check for reasonable data ranges
                if 'price' in df.columns:
                    price_stats = df['price'].describe()
                    validation['price_stats'] = price_stats.to_dict()
                    
                    if price_stats['min'] < 0:
                        validation['warnings'].append("Negative prices found")
                    if price_stats['max'] > 10000:  # Arbitrary high threshold
                        validation['warnings'].append("Very high prices found")
                        
            elif dataset_type == 'test':
                expected_columns = {'sample_id', 'catalog_content', 'image_link'}
                if set(df.columns) != expected_columns:
                    validation['errors'].append(f"Invalid columns. Expected: {expected_columns}, Got: {set(df.columns)}")
                else:
                    validation['is_valid'] = True
                    
            elif dataset_type == 'sample_output':
                expected_columns = {'sample_id', 'price'}
                if set(df.columns) != expected_columns:
                    validation['errors'].append(f"Invalid columns. Expected: {expected_columns}, Got: {set(df.columns)}")
                else:
                    validation['is_valid'] = True
            
            # Common validations
            if 'sample_id' in df.columns:
                if df['sample_id'].duplicated().any():
                    validation['errors'].append("Duplicate sample_ids found")
                if df['sample_id'].isnull().any():
                    validation['errors'].append("Null sample_ids found")
                    
        except Exception as e:
            validation['errors'].append(f"Error reading dataset: {str(e)}")
        
        return validation
    
    def generate_audit_trail(self) -> Dict:
        """Generate comprehensive audit trail of all data sources.
        
        Returns:
            Complete audit trail dictionary
        """
        self.logger.info("Generating data source audit trail...")
        
        audit_trail = {
            'generated_at': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'data_file_validation': self.validate_data_files(),
            'code_validation': self.check_data_loading_code(),
            'dataset_integrity': self.validate_dataset_integrity(),
            'validation_log': self.validation_log,
            'summary': {}
        }
        
        # Generate summary
        data_files = audit_trail['data_file_validation']
        allowed_files = sum(1 for f in data_files.values() if f['is_allowed'])
        total_files = len(data_files)
        
        suspicious_code_files = len(audit_trail['code_validation'])
        
        dataset_issues = sum(1 for d in audit_trail['dataset_integrity'].values() 
                           if not d.get('is_valid', False))
        
        audit_trail['summary'] = {
            'total_data_files': total_files,
            'allowed_data_files': allowed_files,
            'disallowed_data_files': total_files - allowed_files,
            'suspicious_code_files': suspicious_code_files,
            'dataset_validation_issues': dataset_issues,
            'compliance_status': 'PASS' if (total_files == allowed_files and 
                                          suspicious_code_files == 0 and 
                                          dataset_issues == 0) else 'REQUIRES_REVIEW'
        }
        
        return audit_trail
    
    def save_audit_trail(self, output_path: str = "data_source_audit.json") -> str:
        """Save audit trail to file.
        
        Args:
            output_path: Path to save the audit trail
            
        Returns:
            Path to saved audit trail
        """
        audit_trail = self.generate_audit_trail()
        
        output_file = self.project_root / output_path
        with open(output_file, 'w') as f:
            json.dump(audit_trail, f, indent=2)
        
        self.logger.info(f"Data source audit trail saved to {output_file}")
        return str(output_file)