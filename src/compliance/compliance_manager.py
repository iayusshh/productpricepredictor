"""
Main compliance management system for ML Product Pricing Challenge.

This module orchestrates all compliance checks and generates comprehensive
compliance reports for the competition submission.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .data_source_validator import DataSourceValidator
from .license_tracker import LicenseTracker


class ComplianceManager:
    """Main compliance manager that orchestrates all compliance checks."""
    
    def __init__(self, project_root: str = "."):
        """Initialize compliance manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # Initialize component validators
        self.license_tracker = LicenseTracker(project_root)
        self.data_validator = DataSourceValidator(project_root)
        
    def run_full_compliance_check(self) -> Dict:
        """Run comprehensive compliance check.
        
        Returns:
            Complete compliance report
        """
        self.logger.info("Starting full compliance check...")
        
        # Run all compliance checks
        license_report = self.license_tracker.generate_compliance_report()
        data_audit = self.data_validator.generate_audit_trail()
        
        # Generate overall compliance report
        compliance_report = {
            'generated_at': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'compliance_version': '1.0',
            'license_compliance': license_report,
            'data_source_compliance': data_audit,
            'overall_summary': self._generate_overall_summary(license_report, data_audit),
            'recommendations': self._generate_compliance_recommendations(license_report, data_audit),
            'deliverable_checklist': self._check_deliverable_requirements()
        }
        
        self.logger.info("Full compliance check completed")
        return compliance_report
    
    def _generate_overall_summary(self, license_report: Dict, data_audit: Dict) -> Dict:
        """Generate overall compliance summary.
        
        Args:
            license_report: License compliance report
            data_audit: Data source audit report
            
        Returns:
            Overall summary dictionary
        """
        license_status = license_report['summary']['compliance_status']
        data_status = data_audit['summary']['compliance_status']
        
        # Determine overall status
        if license_status == 'PASS' and data_status == 'PASS':
            overall_status = 'COMPLIANT'
        elif license_status == 'REQUIRES_REVIEW' or data_status == 'REQUIRES_REVIEW':
            overall_status = 'REQUIRES_REVIEW'
        else:
            overall_status = 'NON_COMPLIANT'
        
        summary = {
            'overall_compliance_status': overall_status,
            'license_compliance_status': license_status,
            'data_source_compliance_status': data_status,
            'total_issues_found': (
                license_report['summary']['invalid_dependencies'] +
                license_report['summary']['invalid_models'] +
                license_report['summary']['external_sources_found'] +
                data_audit['summary']['disallowed_data_files'] +
                data_audit['summary']['suspicious_code_files'] +
                data_audit['summary']['dataset_validation_issues']
            ),
            'critical_issues': [],
            'warnings': []
        }
        
        # Identify critical issues
        if license_report['summary']['invalid_dependencies'] > 0:
            summary['critical_issues'].append(
                f"{license_report['summary']['invalid_dependencies']} dependencies with invalid licenses"
            )
        
        if license_report['summary']['invalid_models'] > 0:
            summary['critical_issues'].append(
                f"{license_report['summary']['invalid_models']} models with invalid licenses"
            )
        
        if data_audit['summary']['disallowed_data_files'] > 0:
            summary['critical_issues'].append(
                f"{data_audit['summary']['disallowed_data_files']} disallowed data files found"
            )
        
        if data_audit['summary']['dataset_validation_issues'] > 0:
            summary['critical_issues'].append(
                f"{data_audit['summary']['dataset_validation_issues']} dataset validation issues"
            )
        
        # Identify warnings
        if license_report['summary']['external_sources_found'] > 0:
            summary['warnings'].append(
                f"{license_report['summary']['external_sources_found']} files with potential external data patterns"
            )
        
        if data_audit['summary']['suspicious_code_files'] > 0:
            summary['warnings'].append(
                f"{data_audit['summary']['suspicious_code_files']} code files with suspicious patterns"
            )
        
        return summary
    
    def _generate_compliance_recommendations(self, license_report: Dict, data_audit: Dict) -> List[str]:
        """Generate compliance recommendations.
        
        Args:
            license_report: License compliance report
            data_audit: Data source audit report
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # License recommendations
        if license_report['summary']['invalid_dependencies'] > 0:
            recommendations.append(
                "Review and replace dependencies with invalid licenses before submission"
            )
        
        if license_report['summary']['invalid_models'] > 0:
            recommendations.append(
                "Verify model licenses and replace any models with restrictive licenses"
            )
        
        # Data source recommendations
        if data_audit['summary']['disallowed_data_files'] > 0:
            recommendations.append(
                "Remove or justify any disallowed data files in the project"
            )
        
        if data_audit['summary']['suspicious_code_files'] > 0:
            recommendations.append(
                "Review code files flagged for potential external data usage"
            )
        
        if data_audit['summary']['dataset_validation_issues'] > 0:
            recommendations.append(
                "Fix dataset validation issues to ensure proper competition data format"
            )
        
        # General recommendations
        recommendations.extend([
            "Ensure all model checkpoints are properly documented with license information",
            "Verify that only competition-provided datasets are used for training and testing",
            "Document any external libraries or pre-trained models used in the solution",
            "Create clear audit trail of all data processing and model training steps"
        ])
        
        if not any([
            license_report['summary']['invalid_dependencies'],
            license_report['summary']['invalid_models'],
            data_audit['summary']['disallowed_data_files'],
            data_audit['summary']['dataset_validation_issues']
        ]):
            recommendations.insert(0, "All compliance checks passed - solution appears ready for submission")
        
        return recommendations
    
    def _check_deliverable_requirements(self) -> Dict[str, Dict]:
        """Check if all required deliverables are present.
        
        Returns:
            Dictionary with deliverable check results
        """
        deliverables = {
            'test_out.csv': {
                'required': True,
                'path': 'test_out.csv',
                'exists': False,
                'description': 'Final predictions in required format'
            },
            'methodology_1page.pdf': {
                'required': True,
                'path': 'methodology_1page.pdf',
                'exists': False,
                'description': 'One-page methodology document'
            },
            'requirements.txt': {
                'required': True,
                'path': 'requirements.txt',
                'exists': False,
                'description': 'Python dependencies'
            },
            'README.md': {
                'required': True,
                'path': 'README.md',
                'exists': False,
                'description': 'Project documentation and reproduction instructions'
            },
            'src/': {
                'required': True,
                'path': 'src',
                'exists': False,
                'description': 'Source code directory'
            },
            'models/': {
                'required': False,
                'path': 'models',
                'exists': False,
                'description': 'Trained model checkpoints'
            },
            'logs/': {
                'required': False,
                'path': 'logs',
                'exists': False,
                'description': 'Training and validation logs'
            }
        }
        
        # Check if each deliverable exists
        for name, info in deliverables.items():
            path = self.project_root / info['path']
            info['exists'] = path.exists()
            
            if info['exists'] and path.is_file():
                info['size_mb'] = path.stat().st_size / (1024 * 1024)
            elif info['exists'] and path.is_dir():
                # Count files in directory
                files = list(path.rglob('*'))
                info['file_count'] = len([f for f in files if f.is_file()])
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                info['total_size_mb'] = total_size / (1024 * 1024)
        
        return deliverables
    
    def generate_compliance_summary_text(self) -> str:
        """Generate human-readable compliance summary.
        
        Returns:
            Formatted compliance summary text
        """
        report = self.run_full_compliance_check()
        summary = report['overall_summary']
        
        text = f"""
ML Product Pricing Challenge - Compliance Report
===============================================

Generated: {report['generated_at']}
Project: {report['project_root']}

OVERALL STATUS: {summary['overall_compliance_status']}

Summary:
- License Compliance: {summary['license_compliance_status']}
- Data Source Compliance: {summary['data_source_compliance_status']}
- Total Issues Found: {summary['total_issues_found']}

"""
        
        if summary['critical_issues']:
            text += "CRITICAL ISSUES:\n"
            for issue in summary['critical_issues']:
                text += f"  ❌ {issue}\n"
            text += "\n"
        
        if summary['warnings']:
            text += "WARNINGS:\n"
            for warning in summary['warnings']:
                text += f"  ⚠️  {warning}\n"
            text += "\n"
        
        text += "RECOMMENDATIONS:\n"
        for rec in report['recommendations']:
            text += f"  • {rec}\n"
        
        text += "\nDELIVERABLE CHECKLIST:\n"
        for name, info in report['deliverable_checklist'].items():
            status = "✅" if info['exists'] else "❌"
            required = "(Required)" if info['required'] else "(Optional)"
            text += f"  {status} {name} {required}\n"
        
        return text
    
    def save_compliance_report(self, output_dir: str = "compliance") -> Dict[str, str]:
        """Save comprehensive compliance report.
        
        Args:
            output_dir: Directory to save compliance reports
            
        Returns:
            Dictionary mapping report types to file paths
        """
        output_path = self.project_root / output_dir
        output_path.mkdir(exist_ok=True)
        
        # Generate full report
        full_report = self.run_full_compliance_check()
        
        # Save full JSON report
        json_path = output_path / "compliance_report.json"
        with open(json_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        # Save human-readable summary
        summary_path = output_path / "compliance_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(self.generate_compliance_summary_text())
        
        # Save individual component reports
        license_path = output_path / "license_report.json"
        with open(license_path, 'w') as f:
            json.dump(full_report['license_compliance'], f, indent=2)
        
        data_path = output_path / "data_audit.json"
        with open(data_path, 'w') as f:
            json.dump(full_report['data_source_compliance'], f, indent=2)
        
        saved_files = {
            'full_report': str(json_path),
            'summary': str(summary_path),
            'license_report': str(license_path),
            'data_audit': str(data_path)
        }
        
        self.logger.info(f"Compliance reports saved to {output_path}")
        return saved_files
    
    def validate_submission_readiness(self) -> Tuple[bool, List[str]]:
        """Validate if the solution is ready for submission.
        
        Returns:
            Tuple of (is_ready, list_of_blocking_issues)
        """
        report = self.run_full_compliance_check()
        blocking_issues = []
        
        # Check overall compliance status
        if report['overall_summary']['overall_compliance_status'] == 'NON_COMPLIANT':
            blocking_issues.append("Solution has critical compliance issues")
        
        # Check required deliverables
        for name, info in report['deliverable_checklist'].items():
            if info['required'] and not info['exists']:
                blocking_issues.append(f"Missing required deliverable: {name}")
        
        # Check for critical license issues
        if report['license_compliance']['summary']['invalid_dependencies'] > 0:
            blocking_issues.append("Dependencies with invalid licenses found")
        
        if report['license_compliance']['summary']['invalid_models'] > 0:
            blocking_issues.append("Models with invalid licenses found")
        
        # Check for disallowed data files
        if report['data_source_compliance']['summary']['disallowed_data_files'] > 0:
            blocking_issues.append("Disallowed data files found")
        
        # Check dataset integrity
        if report['data_source_compliance']['summary']['dataset_validation_issues'] > 0:
            blocking_issues.append("Dataset validation issues found")
        
        is_ready = len(blocking_issues) == 0
        return is_ready, blocking_issues