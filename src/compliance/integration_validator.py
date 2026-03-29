"""
Integration validator for ML Product Pricing Challenge.

This module provides comprehensive validation of the entire solution,
ensuring all components work together correctly and meet competition requirements.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .compliance_manager import ComplianceManager
from .deliverable_manager import DeliverableManager


class IntegrationValidator:
    """Comprehensive integration validator for the complete solution."""
    
    def __init__(self, project_root: str = "."):
        """Initialize integration validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # Initialize component validators
        self.compliance_manager = ComplianceManager(project_root)
        self.deliverable_manager = DeliverableManager(project_root)
        
        self.validation_results = {}
        
    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive validation of the entire solution.
        
        Returns:
            Complete validation report
        """
        self.logger.info("Starting comprehensive solution validation...")
        
        validation_report = {
            'generated_at': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'validation_version': '1.0',
            'stages': {},
            'overall_summary': {},
            'recommendations': [],
            'submission_readiness': {}
        }
        
        # Run validation stages
        stages = [
            ('environment', self._validate_environment),
            ('datasets', self._validate_datasets),
            ('core_functionality', self._validate_core_functionality),
            ('compliance', self._validate_compliance),
            ('deliverables', self._validate_deliverables),
            ('integration', self._validate_integration),
            ('performance', self._validate_performance)
        ]
        
        for stage_name, stage_func in stages:
            self.logger.info(f"Running {stage_name} validation...")
            try:
                stage_result = stage_func()
                validation_report['stages'][stage_name] = stage_result
                self.validation_results[stage_name] = stage_result
            except Exception as e:
                self.logger.error(f"Error in {stage_name} validation: {e}")
                validation_report['stages'][stage_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Generate overall summary
        validation_report['overall_summary'] = self._generate_overall_summary()
        validation_report['recommendations'] = self._generate_recommendations()
        validation_report['submission_readiness'] = self._assess_submission_readiness()
        
        self.logger.info("Comprehensive validation completed")
        return validation_report
    
    def _validate_environment(self) -> Dict:
        """Validate Python environment and dependencies.
        
        Returns:
            Environment validation results
        """
        result = {
            'status': 'UNKNOWN',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check Python version
            python_version = sys.version_info
            result['checks']['python_version'] = {
                'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'valid': python_version >= (3, 8),
                'requirement': '>=3.8'
            }
            
            # Check virtual environment
            result['checks']['virtual_env'] = {
                'active': hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
                'path': getattr(sys, 'prefix', 'unknown')
            }
            
            # Check critical dependencies
            critical_deps = ['pandas', 'numpy', 'torch', 'transformers', 'scikit-learn']
            dependency_results = {}
            
            for dep in critical_deps:
                try:
                    __import__(dep)
                    dependency_results[dep] = {'available': True, 'error': None}
                except ImportError as e:
                    dependency_results[dep] = {'available': False, 'error': str(e)}
            
            result['checks']['dependencies'] = dependency_results
            
            # Check GPU availability
            try:
                import torch
                result['checks']['gpu'] = {
                    'cuda_available': torch.cuda.is_available(),
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
                }
            except:
                result['checks']['gpu'] = {'cuda_available': False, 'error': 'PyTorch not available'}
            
            # Determine overall status
            all_deps_available = all(dep['available'] for dep in dependency_results.values())
            python_valid = result['checks']['python_version']['valid']
            
            if python_valid and all_deps_available:
                result['status'] = 'PASS'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _validate_datasets(self) -> Dict:
        """Validate dataset structure and integrity.
        
        Returns:
            Dataset validation results
        """
        result = {
            'status': 'UNKNOWN',
            'datasets': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            import pandas as pd
            
            # Validate training data
            train_path = self.project_root / 'dataset' / 'train.csv'
            if train_path.exists():
                train_df = pd.read_csv(train_path)
                expected_cols = {'sample_id', 'catalog_content', 'image_link', 'price'}
                
                result['datasets']['train.csv'] = {
                    'exists': True,
                    'row_count': len(train_df),
                    'columns': list(train_df.columns),
                    'valid_structure': set(train_df.columns) == expected_cols,
                    'has_nulls': train_df.isnull().any().any(),
                    'price_stats': {
                        'min': float(train_df['price'].min()),
                        'max': float(train_df['price'].max()),
                        'mean': float(train_df['price'].mean()),
                        'zero_count': int((train_df['price'] == 0).sum())
                    }
                }
            else:
                result['datasets']['train.csv'] = {'exists': False}
            
            # Validate test data
            test_path = self.project_root / 'dataset' / 'test.csv'
            if test_path.exists():
                test_df = pd.read_csv(test_path)
                expected_cols = {'sample_id', 'catalog_content', 'image_link'}
                
                result['datasets']['test.csv'] = {
                    'exists': True,
                    'row_count': len(test_df),
                    'columns': list(test_df.columns),
                    'valid_structure': set(test_df.columns) == expected_cols,
                    'has_nulls': test_df.isnull().any().any()
                }
            else:
                result['datasets']['test.csv'] = {'exists': False}
            
            # Validate sample output
            sample_path = self.project_root / 'dataset' / 'sample_test_out.csv'
            if sample_path.exists():
                sample_df = pd.read_csv(sample_path)
                expected_cols = {'sample_id', 'price'}
                
                result['datasets']['sample_test_out.csv'] = {
                    'exists': True,
                    'row_count': len(sample_df),
                    'columns': list(sample_df.columns),
                    'valid_structure': set(sample_df.columns) == expected_cols
                }
            else:
                result['datasets']['sample_test_out.csv'] = {'exists': False}
            
            # Determine overall status
            all_exist = all(ds.get('exists', False) for ds in result['datasets'].values())
            all_valid = all(ds.get('valid_structure', False) for ds in result['datasets'].values() if ds.get('exists'))
            
            if all_exist and all_valid:
                result['status'] = 'PASS'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _validate_core_functionality(self) -> Dict:
        """Validate core functionality components.
        
        Returns:
            Core functionality validation results
        """
        result = {
            'status': 'UNKNOWN',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Test IPQ extraction
            try:
                sys.path.append(str(self.project_root / 'src'))
                from features import IPQExtractor
                
                extractor = IPQExtractor()
                precision = extractor.validate_ipq_extraction_precision()
                
                result['components']['ipq_extraction'] = {
                    'available': True,
                    'precision': precision,
                    'meets_requirement': precision >= 0.90,
                    'requirement': '>=0.90'
                }
            except Exception as e:
                result['components']['ipq_extraction'] = {
                    'available': False,
                    'error': str(e)
                }
            
            # Test SMAPE calculation
            try:
                from evaluation import SMAPECalculator
                
                calc = SMAPECalculator()
                unit_test_passed = calc.test_smape_on_known_examples()
                
                result['components']['smape_calculation'] = {
                    'available': True,
                    'unit_tests_passed': unit_test_passed
                }
            except Exception as e:
                result['components']['smape_calculation'] = {
                    'available': False,
                    'error': str(e)
                }
            
            # Test text processing
            try:
                from features import TextProcessor
                
                processor = TextProcessor()
                test_result = processor.parse_catalog_content("Test product | 1kg | Premium quality")
                
                result['components']['text_processing'] = {
                    'available': True,
                    'test_successful': test_result is not None
                }
            except Exception as e:
                result['components']['text_processing'] = {
                    'available': False,
                    'error': str(e)
                }
            
            # Test image processing
            try:
                from features import ImageProcessor
                
                processor = ImageProcessor()
                # Just test initialization
                result['components']['image_processing'] = {
                    'available': True,
                    'initialized': True
                }
            except Exception as e:
                result['components']['image_processing'] = {
                    'available': False,
                    'error': str(e)
                }
            
            # Determine overall status
            all_available = all(comp.get('available', False) for comp in result['components'].values())
            ipq_meets_req = result['components'].get('ipq_extraction', {}).get('meets_requirement', False)
            smape_valid = result['components'].get('smape_calculation', {}).get('unit_tests_passed', False)
            
            if all_available and ipq_meets_req and smape_valid:
                result['status'] = 'PASS'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _validate_compliance(self) -> Dict:
        """Validate compliance requirements.
        
        Returns:
            Compliance validation results
        """
        try:
            compliance_report = self.compliance_manager.run_full_compliance_check()
            is_ready, issues = self.compliance_manager.validate_submission_readiness()
            
            return {
                'status': 'PASS' if is_ready else 'FAIL',
                'submission_ready': is_ready,
                'blocking_issues': issues,
                'compliance_summary': compliance_report['overall_summary'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_deliverables(self) -> Dict:
        """Validate deliverable structure and completeness.
        
        Returns:
            Deliverable validation results
        """
        try:
            validation_results = self.deliverable_manager.validate_deliverable_completeness()
            qa_report = self.deliverable_manager.generate_quality_assurance_report()
            
            return {
                'status': qa_report['overall_status'],
                'summary': qa_report['summary'],
                'critical_issues': qa_report['critical_issues'],
                'submission_checklist': qa_report['submission_checklist'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_integration(self) -> Dict:
        """Validate integration between components.
        
        Returns:
            Integration validation results
        """
        result = {
            'status': 'UNKNOWN',
            'tests': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Test end-to-end pipeline components
            sys.path.append(str(self.project_root / 'src'))
            
            # Test data loading -> feature extraction pipeline
            try:
                from data_processing import DataLoader
                from features import TextProcessor, IPQExtractor
                
                # Test basic pipeline
                loader = DataLoader()
                processor = TextProcessor()
                extractor = IPQExtractor()
                
                # Test with sample data
                sample_text = "Premium Coffee Beans | 1kg pack | Organic"
                parsed = processor.parse_catalog_content(sample_text)
                ipq_result = extractor.extract_ipq_with_validation(sample_text)
                
                result['tests']['data_to_features'] = {
                    'successful': True,
                    'components_integrated': ['DataLoader', 'TextProcessor', 'IPQExtractor']
                }
            except Exception as e:
                result['tests']['data_to_features'] = {
                    'successful': False,
                    'error': str(e)
                }
            
            # Test feature extraction -> model pipeline
            try:
                from features import TextFeatureExtractor
                from models import ModelTrainer
                
                # Test basic integration
                feature_extractor = TextFeatureExtractor()
                model_trainer = ModelTrainer()
                
                result['tests']['features_to_model'] = {
                    'successful': True,
                    'components_integrated': ['TextFeatureExtractor', 'ModelTrainer']
                }
            except Exception as e:
                result['tests']['features_to_model'] = {
                    'successful': False,
                    'error': str(e)
                }
            
            # Test model -> prediction pipeline
            try:
                from prediction import PredictionGenerator, OutputFormatter
                
                # Test basic integration
                pred_generator = PredictionGenerator()
                output_formatter = OutputFormatter()
                
                result['tests']['model_to_prediction'] = {
                    'successful': True,
                    'components_integrated': ['PredictionGenerator', 'OutputFormatter']
                }
            except Exception as e:
                result['tests']['model_to_prediction'] = {
                    'successful': False,
                    'error': str(e)
                }
            
            # Determine overall status
            all_successful = all(test.get('successful', False) for test in result['tests'].values())
            result['status'] = 'PASS' if all_successful else 'FAIL'
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _validate_performance(self) -> Dict:
        """Validate performance requirements.
        
        Returns:
            Performance validation results
        """
        result = {
            'status': 'UNKNOWN',
            'benchmarks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check IPQ precision benchmark
            ipq_precision = self.validation_results.get('core_functionality', {}).get('components', {}).get('ipq_extraction', {}).get('precision', 0)
            result['benchmarks']['ipq_precision'] = {
                'value': ipq_precision,
                'requirement': 0.90,
                'meets_requirement': ipq_precision >= 0.90
            }
            
            # Check file sizes and storage requirements
            models_dir = self.project_root / 'models'
            if models_dir.exists():
                model_files = list(models_dir.rglob('*'))
                total_model_size = sum(f.stat().st_size for f in model_files if f.is_file())
                result['benchmarks']['model_storage'] = {
                    'size_mb': total_model_size / (1024 * 1024),
                    'file_count': len([f for f in model_files if f.is_file()])
                }
            
            # Check embeddings cache
            embeddings_dir = self.project_root / 'embeddings'
            if embeddings_dir.exists():
                embedding_files = list(embeddings_dir.rglob('*'))
                total_embedding_size = sum(f.stat().st_size for f in embedding_files if f.is_file())
                result['benchmarks']['embedding_storage'] = {
                    'size_mb': total_embedding_size / (1024 * 1024),
                    'file_count': len([f for f in embedding_files if f.is_file()])
                }
            
            # Check image cache
            images_dir = self.project_root / 'images'
            if images_dir.exists():
                image_files = list(images_dir.rglob('*'))
                total_image_size = sum(f.stat().st_size for f in image_files if f.is_file())
                result['benchmarks']['image_storage'] = {
                    'size_mb': total_image_size / (1024 * 1024),
                    'file_count': len([f for f in image_files if f.is_file()])
                }
            
            # Determine overall status based on key benchmarks
            ipq_meets_req = result['benchmarks']['ipq_precision']['meets_requirement']
            result['status'] = 'PASS' if ipq_meets_req else 'FAIL'
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _generate_overall_summary(self) -> Dict:
        """Generate overall validation summary.
        
        Returns:
            Overall summary dictionary
        """
        stage_statuses = [result.get('status', 'UNKNOWN') for result in self.validation_results.values()]
        
        passed_stages = sum(1 for status in stage_statuses if status == 'PASS')
        failed_stages = sum(1 for status in stage_statuses if status == 'FAIL')
        error_stages = sum(1 for status in stage_statuses if status == 'ERROR')
        total_stages = len(stage_statuses)
        
        # Determine overall status
        if error_stages > 0:
            overall_status = 'ERROR'
        elif failed_stages > 0:
            overall_status = 'FAIL'
        elif passed_stages == total_stages:
            overall_status = 'PASS'
        else:
            overall_status = 'PARTIAL'
        
        return {
            'overall_status': overall_status,
            'total_stages': total_stages,
            'passed_stages': passed_stages,
            'failed_stages': failed_stages,
            'error_stages': error_stages,
            'success_rate': (passed_stages / total_stages * 100) if total_stages > 0 else 0,
            'stage_breakdown': {stage: result.get('status', 'UNKNOWN') 
                              for stage, result in self.validation_results.items()}
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate validation recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Environment recommendations
        env_result = self.validation_results.get('environment', {})
        if env_result.get('status') != 'PASS':
            recommendations.append("Fix environment issues: ensure Python >=3.8 and all dependencies are installed")
        
        # Dataset recommendations
        dataset_result = self.validation_results.get('datasets', {})
        if dataset_result.get('status') != 'PASS':
            recommendations.append("Fix dataset issues: ensure all required datasets are present with correct structure")
        
        # Core functionality recommendations
        core_result = self.validation_results.get('core_functionality', {})
        if core_result.get('status') != 'PASS':
            ipq_comp = core_result.get('components', {}).get('ipq_extraction', {})
            if not ipq_comp.get('meets_requirement', False):
                recommendations.append("Improve IPQ extraction precision to meet >90% requirement")
            
            smape_comp = core_result.get('components', {}).get('smape_calculation', {})
            if not smape_comp.get('unit_tests_passed', False):
                recommendations.append("Fix SMAPE calculation to pass unit tests")
        
        # Compliance recommendations
        compliance_result = self.validation_results.get('compliance', {})
        if compliance_result.get('status') != 'PASS':
            blocking_issues = compliance_result.get('blocking_issues', [])
            if blocking_issues:
                recommendations.append(f"Fix compliance issues: {'; '.join(blocking_issues[:3])}")
        
        # Deliverable recommendations
        deliverable_result = self.validation_results.get('deliverables', {})
        if deliverable_result.get('status') not in ['READY_FOR_SUBMISSION', 'READY_WITH_WARNINGS']:
            critical_issues = deliverable_result.get('critical_issues', [])
            if critical_issues:
                recommendations.append(f"Fix deliverable issues: {'; '.join(critical_issues[:3])}")
        
        # Integration recommendations
        integration_result = self.validation_results.get('integration', {})
        if integration_result.get('status') != 'PASS':
            recommendations.append("Fix component integration issues to ensure end-to-end pipeline works")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All validation checks passed - solution appears ready for submission")
        else:
            recommendations.append("Address all validation issues before submission")
            recommendations.append("Run validation again after making fixes")
        
        return recommendations
    
    def _assess_submission_readiness(self) -> Dict:
        """Assess overall submission readiness.
        
        Returns:
            Submission readiness assessment
        """
        overall_summary = self._generate_overall_summary()
        
        # Critical requirements for submission
        critical_checks = {
            'environment_valid': self.validation_results.get('environment', {}).get('status') == 'PASS',
            'datasets_valid': self.validation_results.get('datasets', {}).get('status') == 'PASS',
            'ipq_precision_met': (
                self.validation_results.get('core_functionality', {})
                .get('components', {})
                .get('ipq_extraction', {})
                .get('meets_requirement', False)
            ),
            'smape_valid': (
                self.validation_results.get('core_functionality', {})
                .get('components', {})
                .get('smape_calculation', {})
                .get('unit_tests_passed', False)
            ),
            'compliance_passed': self.validation_results.get('compliance', {}).get('submission_ready', False),
            'required_deliverables_present': (
                self.validation_results.get('deliverables', {}).get('status') 
                in ['READY_FOR_SUBMISSION', 'READY_WITH_WARNINGS']
            )
        }
        
        # Calculate readiness score
        passed_critical = sum(1 for check in critical_checks.values() if check)
        total_critical = len(critical_checks)
        readiness_score = (passed_critical / total_critical * 100) if total_critical > 0 else 0
        
        # Determine readiness status
        if readiness_score == 100:
            readiness_status = 'READY'
        elif readiness_score >= 80:
            readiness_status = 'MOSTLY_READY'
        elif readiness_score >= 60:
            readiness_status = 'PARTIALLY_READY'
        else:
            readiness_status = 'NOT_READY'
        
        return {
            'readiness_status': readiness_status,
            'readiness_score': readiness_score,
            'critical_checks': critical_checks,
            'blocking_issues': [
                check_name for check_name, passed in critical_checks.items() if not passed
            ],
            'overall_validation_status': overall_summary['overall_status']
        }
    
    def save_integration_report(self, output_path: str = "integration_validation_report.json") -> str:
        """Save comprehensive integration validation report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        report = self.run_comprehensive_validation()
        
        output_file = self.project_root / output_path
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Integration validation report saved to {output_file}")
        return str(output_file)