"""
License Compliance and Dependency Validation Tests.

Tests for license tracking, dependency validation, compliance checking,
and audit trail generation for all dependencies and model checkpoints.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from src.compliance.license_tracker import LicenseTracker
from src.compliance.data_source_validator import DataSourceValidator
from src.compliance.compliance_manager import ComplianceManager
from src.config import config


class TestLicenseComplianceValidation(unittest.TestCase):
    """Test cases for License Compliance and Dependency Validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        (self.temp_path / "models").mkdir()
        (self.temp_path / "logs").mkdir()
        (self.temp_path / "compliance").mkdir()
        
        # Create test files
        self.create_test_dependency_files()
        self.create_test_model_files()
        
        # Mock config
        self.original_config = config
        self.mock_config()
        
        # Initialize components
        self.license_tracker = LicenseTracker()
        self.data_source_validator = DataSourceValidator()
        self.compliance_manager = ComplianceManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original config
        config.__dict__.update(self.original_config.__dict__)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dependency_files(self):
        """Create test dependency files."""
        # Create requirements.txt
        requirements_content = """
# Core ML dependencies
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0

# Deep learning
torch==2.0.1
transformers==4.30.2

# Data processing
requests==2.31.0
tqdm==4.65.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Testing
pytest==7.4.0
unittest-xml-reporting==3.2.0
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
    
    def create_test_model_files(self):
        """Create test model files with metadata."""
        # Create model metadata with license information
        model_metadata = {
            'model_name': 'bert-base-uncased',
            'model_source': 'huggingface',
            'license': 'Apache-2.0',
            'model_size': '440MB',
            'parameters': '110M',
            'download_url': 'https://huggingface.co/bert-base-uncased',
            'license_url': 'https://www.apache.org/licenses/LICENSE-2.0',
            'attribution': 'Google Research',
            'usage_restrictions': 'None for Apache-2.0'
        }
        
        model_metadata_file = self.temp_path / "models" / "model_metadata.json"
        with open(model_metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Create additional model files
        (self.temp_path / "models" / "pytorch_model.bin").write_bytes(b"mock model weights")
        (self.temp_path / "models" / "config.json").write_text('{"model_type": "bert"}')
        (self.temp_path / "models" / "tokenizer.json").write_text('{"tokenizer_type": "bert"}')
    
    def mock_config(self):
        """Mock configuration for testing."""
        config.project_root = str(self.temp_path)
        config.model.model_dir = str(self.temp_path / "models")
        config.compliance.log_dir = str(self.temp_path / "compliance")
        config.compliance.allowed_licenses = [
            'MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'ISC'
        ]
        config.compliance.prohibited_licenses = [
            'GPL-3.0', 'AGPL-3.0', 'LGPL-3.0', 'SSPL-1.0'
        ]
    
    def test_dependency_license_tracking(self):
        """Test tracking of dependency licenses."""
        # Track licenses from requirements.txt
        license_info = self.license_tracker.track_dependency_licenses()
        
        self.assertIsInstance(license_info, dict)
        self.assertIn('dependencies', license_info)
        self.assertIn('total_count', license_info)
        self.assertIn('license_summary', license_info)
        
        dependencies = license_info['dependencies']
        self.assertGreater(len(dependencies), 0)
        
        # Check dependency structure
        for dep in dependencies:
            self.assertIn('name', dep)
            self.assertIn('version', dep)
            self.assertIn('license', dep)
            self.assertIn('license_source', dep)
            
            # Name and version should be non-empty
            self.assertGreater(len(dep['name']), 0)
            self.assertGreater(len(dep['version']), 0)
        
        # Check license summary
        license_summary = license_info['license_summary']
        self.assertIsInstance(license_summary, dict)
        
        # Should have some license types
        total_licenses = sum(license_summary.values())
        self.assertEqual(total_licenses, license_info['total_count'])
    
    def test_model_license_validation(self):
        """Test validation of model licenses."""
        # Validate model licenses
        model_validation = self.license_tracker.validate_model_licenses()
        
        self.assertIsInstance(model_validation, dict)
        self.assertIn('is_compliant', model_validation)
        self.assertIn('models', model_validation)
        self.assertIn('compliance_summary', model_validation)
        
        models = model_validation['models']
        self.assertGreater(len(models), 0)
        
        # Check model license information
        for model in models:
            self.assertIn('model_name', model)
            self.assertIn('license', model)
            self.assertIn('is_allowed', model)
            self.assertIn('license_source', model)
            
            # Should have license information
            self.assertIsNotNone(model['license'])
            self.assertIsInstance(model['is_allowed'], bool)
        
        # Check compliance summary
        compliance_summary = model_validation['compliance_summary']
        self.assertIn('total_models', compliance_summary)
        self.assertIn('compliant_models', compliance_summary)
        self.assertIn('non_compliant_models', compliance_summary)
        
        # Counts should be consistent
        self.assertEqual(
            compliance_summary['compliant_models'] + compliance_summary['non_compliant_models'],
            compliance_summary['total_models']
        )
    
    def test_license_compatibility_checking(self):
        """Test license compatibility checking."""
        # Test individual license compatibility
        test_licenses = [
            ('MIT', True),
            ('Apache-2.0', True),
            ('BSD-3-Clause', True),
            ('GPL-3.0', False),
            ('AGPL-3.0', False),
            ('Unknown-License', False)
        ]
        
        for license_name, expected_allowed in test_licenses:
            with self.subTest(license=license_name):
                is_allowed = self.license_tracker.is_license_allowed(license_name)
                self.assertEqual(is_allowed, expected_allowed)
        
        # Test license compatibility matrix
        compatibility_matrix = self.license_tracker.get_license_compatibility_matrix()
        
        self.assertIsInstance(compatibility_matrix, dict)
        self.assertIn('allowed_licenses', compatibility_matrix)
        self.assertIn('prohibited_licenses', compatibility_matrix)
        self.assertIn('unknown_licenses', compatibility_matrix)
        
        # Should have some allowed licenses
        self.assertGreater(len(compatibility_matrix['allowed_licenses']), 0)
    
    def test_data_source_validation(self):
        """Test validation that no external data sources are used."""
        # Test data source validation
        validation_results = self.data_source_validator.validate_data_sources()
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn('is_compliant', validation_results)
        self.assertIn('data_sources', validation_results)
        self.assertIn('violations', validation_results)
        
        # Should be compliant (no external data sources)
        self.assertTrue(validation_results['is_compliant'])
        
        # Check data sources
        data_sources = validation_results['data_sources']
        self.assertIsInstance(data_sources, list)
        
        # All data sources should be allowed
        for source in data_sources:
            self.assertIn('source_type', source)
            self.assertIn('source_path', source)
            self.assertIn('is_allowed', source)
            self.assertTrue(source['is_allowed'])
        
        # Should have no violations
        violations = validation_results['violations']
        self.assertEqual(len(violations), 0)
    
    def test_external_data_detection(self):
        """Test detection of prohibited external data sources."""
        # Create mock external data references
        external_references = [
            'https://external-api.com/prices',
            'ftp://external-server.com/data',
            '/external/database/connection'
        ]
        
        # Test detection
        for reference in external_references:
            with self.subTest(reference=reference):
                is_external = self.data_source_validator.is_external_data_source(reference)
                self.assertTrue(is_external)
        
        # Test allowed internal references
        internal_references = [
            'dataset/train.csv',
            './data/test.csv',
            '../models/model.pkl',
            '/tmp/cache/embeddings.npy'
        ]
        
        for reference in internal_references:
            with self.subTest(reference=reference):
                is_external = self.data_source_validator.is_external_data_source(reference)
                self.assertFalse(is_external)
    
    def test_api_usage_detection(self):
        """Test detection of external API usage."""
        # Mock code content with API calls
        code_with_apis = """
import requests
import urllib.request

# This should be detected as external API usage
response = requests.get('https://api.external-service.com/data')
data = urllib.request.urlopen('https://another-api.com/prices')

# This should be allowed (local/internal)
local_response = requests.get('http://localhost:8080/internal-api')
"""
        
        # Create test file
        test_file = self.temp_path / "test_code.py"
        test_file.write_text(code_with_apis)
        
        # Detect API usage
        api_usage = self.data_source_validator.detect_api_usage(str(test_file))
        
        self.assertIsInstance(api_usage, dict)
        self.assertIn('has_external_apis', api_usage)
        self.assertIn('api_calls', api_usage)
        
        # Should detect external API usage
        self.assertTrue(api_usage['has_external_apis'])
        
        api_calls = api_usage['api_calls']
        self.assertGreater(len(api_calls), 0)
        
        # Check API call details
        for api_call in api_calls:
            self.assertIn('url', api_call)
            self.assertIn('is_external', api_call)
            self.assertIn('line_number', api_call)
    
    def test_web_scraping_detection(self):
        """Test detection of web scraping activities."""
        # Mock code with web scraping
        scraping_code = """
from bs4 import BeautifulSoup
import scrapy
from selenium import webdriver

# Web scraping code
soup = BeautifulSoup(html_content, 'html.parser')
driver = webdriver.Chrome()
driver.get('https://example.com/prices')

class PriceSpider(scrapy.Spider):
    name = 'prices'
    start_urls = ['https://ecommerce-site.com/products']
"""
        
        # Create test file
        test_file = self.temp_path / "scraping_code.py"
        test_file.write_text(scraping_code)
        
        # Detect web scraping
        scraping_detection = self.data_source_validator.detect_web_scraping(str(test_file))
        
        self.assertIsInstance(scraping_detection, dict)
        self.assertIn('has_scraping', scraping_detection)
        self.assertIn('scraping_indicators', scraping_detection)
        
        # Should detect web scraping
        self.assertTrue(scraping_detection['has_scraping'])
        
        indicators = scraping_detection['scraping_indicators']
        self.assertGreater(len(indicators), 0)
        
        # Check scraping indicators
        for indicator in indicators:
            self.assertIn('type', indicator)
            self.assertIn('line_number', indicator)
            self.assertIn('code_snippet', indicator)
    
    def test_compliance_audit_trail(self):
        """Test generation of compliance audit trail."""
        # Generate audit trail
        audit_trail = self.compliance_manager.generate_audit_trail()
        
        self.assertIsInstance(audit_trail, dict)
        self.assertIn('timestamp', audit_trail)
        self.assertIn('compliance_status', audit_trail)
        self.assertIn('audit_sections', audit_trail)
        
        # Check audit sections
        audit_sections = audit_trail['audit_sections']
        expected_sections = [
            'dependency_licenses', 'model_licenses', 'data_sources',
            'external_apis', 'web_scraping'
        ]
        
        for section in expected_sections:
            self.assertIn(section, audit_sections)
            self.assertIn('status', audit_sections[section])
            self.assertIn('details', audit_sections[section])
        
        # Check overall compliance status
        compliance_status = audit_trail['compliance_status']
        self.assertIn(compliance_status, ['COMPLIANT', 'NON_COMPLIANT', 'WARNING'])
    
    def test_compliance_report_generation(self):
        """Test generation of comprehensive compliance report."""
        # Generate compliance report
        compliance_report = self.compliance_manager.generate_compliance_report()
        
        self.assertIsInstance(compliance_report, dict)
        self.assertIn('report_metadata', compliance_report)
        self.assertIn('executive_summary', compliance_report)
        self.assertIn('detailed_findings', compliance_report)
        self.assertIn('recommendations', compliance_report)
        
        # Check report metadata
        metadata = compliance_report['report_metadata']
        self.assertIn('generated_at', metadata)
        self.assertIn('report_version', metadata)
        self.assertIn('project_name', metadata)
        
        # Check executive summary
        summary = compliance_report['executive_summary']
        self.assertIn('overall_status', summary)
        self.assertIn('total_dependencies', summary)
        self.assertIn('compliant_dependencies', summary)
        self.assertIn('total_models', summary)
        self.assertIn('compliant_models', summary)
        
        # Check detailed findings
        findings = compliance_report['detailed_findings']
        self.assertIn('license_analysis', findings)
        self.assertIn('data_source_analysis', findings)
        self.assertIn('risk_assessment', findings)
    
    def test_license_file_parsing(self):
        """Test parsing of license files and metadata."""
        # Create mock license file
        license_content = """
MIT License

Copyright (c) 2024 Test Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
        
        license_file = self.temp_path / "LICENSE"
        license_file.write_text(license_content.strip())
        
        # Parse license file
        license_info = self.license_tracker.parse_license_file(str(license_file))
        
        self.assertIsInstance(license_info, dict)
        self.assertIn('license_type', license_info)
        self.assertIn('copyright_holder', license_info)
        self.assertIn('copyright_year', license_info)
        self.assertIn('is_recognized', license_info)
        
        # Should recognize MIT license
        self.assertEqual(license_info['license_type'], 'MIT')
        self.assertTrue(license_info['is_recognized'])
        self.assertIn('2024', license_info['copyright_year'])
    
    def test_dependency_vulnerability_scanning(self):
        """Test scanning for known vulnerabilities in dependencies."""
        # Mock vulnerability database
        mock_vulnerabilities = {
            'requests': {
                '2.25.0': ['CVE-2021-33503'],
                '2.26.0': []
            },
            'urllib3': {
                '1.26.0': ['CVE-2021-33503'],
                '1.26.5': []
            }
        }
        
        with patch.object(self.license_tracker, '_get_vulnerability_database', return_value=mock_vulnerabilities):
            # Scan for vulnerabilities
            vuln_scan = self.license_tracker.scan_vulnerabilities()
            
            self.assertIsInstance(vuln_scan, dict)
            self.assertIn('has_vulnerabilities', vuln_scan)
            self.assertIn('vulnerable_packages', vuln_scan)
            self.assertIn('scan_summary', vuln_scan)
            
            # Check scan results
            if vuln_scan['has_vulnerabilities']:
                vulnerable_packages = vuln_scan['vulnerable_packages']
                self.assertGreater(len(vulnerable_packages), 0)
                
                for package in vulnerable_packages:
                    self.assertIn('name', package)
                    self.assertIn('version', package)
                    self.assertIn('vulnerabilities', package)
                    self.assertGreater(len(package['vulnerabilities']), 0)
    
    def test_license_compatibility_conflicts(self):
        """Test detection of license compatibility conflicts."""
        # Mock dependencies with conflicting licenses
        mock_dependencies = [
            {'name': 'package1', 'license': 'MIT'},
            {'name': 'package2', 'license': 'Apache-2.0'},
            {'name': 'package3', 'license': 'GPL-3.0'},  # Potentially conflicting
            {'name': 'package4', 'license': 'BSD-3-Clause'}
        ]
        
        # Check for conflicts
        conflicts = self.license_tracker.detect_license_conflicts(mock_dependencies)
        
        self.assertIsInstance(conflicts, dict)
        self.assertIn('has_conflicts', conflicts)
        self.assertIn('conflicts', conflicts)
        self.assertIn('conflict_summary', conflicts)
        
        # Should detect GPL conflict with permissive licenses
        if conflicts['has_conflicts']:
            conflict_list = conflicts['conflicts']
            self.assertGreater(len(conflict_list), 0)
            
            for conflict in conflict_list:
                self.assertIn('license1', conflict)
                self.assertIn('license2', conflict)
                self.assertIn('conflict_type', conflict)
                self.assertIn('severity', conflict)
    
    def test_automated_compliance_checking(self):
        """Test automated compliance checking workflow."""
        # Run automated compliance check
        compliance_check = self.compliance_manager.run_automated_compliance_check()
        
        self.assertIsInstance(compliance_check, dict)
        self.assertIn('overall_compliant', compliance_check)
        self.assertIn('check_results', compliance_check)
        self.assertIn('compliance_score', compliance_check)
        
        # Check individual compliance checks
        check_results = compliance_check['check_results']
        expected_checks = [
            'dependency_licenses', 'model_licenses', 'data_sources',
            'external_apis', 'vulnerability_scan'
        ]
        
        for check_name in expected_checks:
            if check_name in check_results:
                check_result = check_results[check_name]
                self.assertIn('passed', check_result)
                self.assertIn('details', check_result)
                self.assertIsInstance(check_result['passed'], bool)
        
        # Compliance score should be between 0 and 100
        compliance_score = compliance_check['compliance_score']
        self.assertGreaterEqual(compliance_score, 0)
        self.assertLessEqual(compliance_score, 100)
    
    def test_compliance_log_generation(self):
        """Test generation of detailed compliance log."""
        # Generate compliance log
        compliance_log = self.compliance_manager.generate_detailed_compliance_log()
        
        self.assertIsInstance(compliance_log, dict)
        self.assertIn('log_metadata', compliance_log)
        self.assertIn('dependency_details', compliance_log)
        self.assertIn('model_details', compliance_log)
        self.assertIn('validation_results', compliance_log)
        
        # Check log metadata
        metadata = compliance_log['log_metadata']
        self.assertIn('generated_at', metadata)
        self.assertIn('log_version', metadata)
        self.assertIn('validation_timestamp', metadata)
        
        # Check dependency details
        dependency_details = compliance_log['dependency_details']
        self.assertIsInstance(dependency_details, list)
        
        for dep in dependency_details:
            self.assertIn('name', dep)
            self.assertIn('version', dep)
            self.assertIn('license', dep)
            self.assertIn('license_url', dep)
            self.assertIn('is_compliant', dep)
        
        # Save compliance log to file
        log_file = self.temp_path / "compliance" / "compliance_log.json"
        with open(log_file, 'w') as f:
            json.dump(compliance_log, f, indent=2)
        
        # Verify file was created and is valid JSON
        self.assertTrue(log_file.exists())
        
        with open(log_file, 'r') as f:
            loaded_log = json.load(f)
        
        self.assertEqual(loaded_log['log_metadata']['log_version'], 
                        compliance_log['log_metadata']['log_version'])


if __name__ == '__main__':
    unittest.main()