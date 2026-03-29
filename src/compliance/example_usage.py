"""
Example usage of the compliance validation system.

This script demonstrates how to use the compliance components to validate
the solution against competition requirements.
"""

import logging
from pathlib import Path

from .compliance_manager import ComplianceManager
from .data_source_validator import DataSourceValidator
from .license_tracker import LicenseTracker


def setup_logging():
    """Set up logging for compliance checking."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_license_tracking():
    """Example of license tracking functionality."""
    print("=== License Tracking Example ===")
    
    tracker = LicenseTracker()
    
    # Track all dependencies
    print("Tracking dependency licenses...")
    dependencies = tracker.track_all_dependencies()
    
    print(f"Found {len(dependencies)} dependencies")
    
    # Show some examples
    for name, info in list(dependencies.items())[:5]:
        status = "✅" if info['is_valid'] else "❌"
        print(f"{status} {name} ({info['version']}): {info['license']}")
    
    # Generate compliance report
    print("\nGenerating compliance report...")
    report = tracker.generate_compliance_report()
    
    print(f"Compliance Status: {report['summary']['compliance_status']}")
    print(f"Valid Dependencies: {report['summary']['valid_dependencies']}/{report['summary']['total_dependencies']}")
    
    # Save report
    report_path = tracker.save_compliance_report("compliance_report.json")
    print(f"Report saved to: {report_path}")


def example_data_source_validation():
    """Example of data source validation functionality."""
    print("\n=== Data Source Validation Example ===")
    
    validator = DataSourceValidator()
    
    # Validate data files
    print("Validating data files...")
    data_files = validator.validate_data_files()
    
    allowed_files = sum(1 for f in data_files.values() if f['is_allowed'])
    total_files = len(data_files)
    
    print(f"Data files: {allowed_files}/{total_files} allowed")
    
    # Show some examples
    for path, info in list(data_files.items())[:5]:
        status = "✅" if info['is_allowed'] else "❌"
        print(f"{status} {path}: {info['reason']}")
    
    # Check code for suspicious patterns
    print("\nChecking code for external data patterns...")
    suspicious_files = validator.check_data_loading_code()
    
    if suspicious_files:
        print(f"Found {len(suspicious_files)} files with suspicious patterns:")
        for file_path, patterns in suspicious_files.items():
            print(f"  {file_path}: {patterns}")
    else:
        print("No suspicious patterns found in code")
    
    # Validate competition datasets
    print("\nValidating competition datasets...")
    dataset_validation = validator.validate_dataset_integrity()
    
    for dataset, info in dataset_validation.items():
        status = "✅" if info.get('is_valid', False) else "❌"
        print(f"{status} {dataset}: {info.get('row_count', 'N/A')} rows")
    
    # Save audit trail
    audit_path = validator.save_audit_trail("data_audit.json")
    print(f"Audit trail saved to: {audit_path}")


def example_full_compliance_check():
    """Example of full compliance check using ComplianceManager."""
    print("\n=== Full Compliance Check Example ===")
    
    manager = ComplianceManager()
    
    # Run full compliance check
    print("Running comprehensive compliance check...")
    report = manager.run_full_compliance_check()
    
    # Display summary
    summary = report['overall_summary']
    print(f"\nOverall Status: {summary['overall_compliance_status']}")
    print(f"Total Issues: {summary['total_issues_found']}")
    
    if summary['critical_issues']:
        print("\nCritical Issues:")
        for issue in summary['critical_issues']:
            print(f"  ❌ {issue}")
    
    if summary['warnings']:
        print("\nWarnings:")
        for warning in summary['warnings']:
            print(f"  ⚠️  {warning}")
    
    # Check deliverables
    print("\nDeliverable Checklist:")
    for name, info in report['deliverable_checklist'].items():
        status = "✅" if info['exists'] else "❌"
        required = "(Required)" if info['required'] else "(Optional)"
        print(f"  {status} {name} {required}")
    
    # Show recommendations
    print("\nRecommendations:")
    for rec in report['recommendations'][:5]:  # Show first 5
        print(f"  • {rec}")
    
    # Save comprehensive report
    saved_files = manager.save_compliance_report("compliance")
    print(f"\nReports saved:")
    for report_type, path in saved_files.items():
        print(f"  {report_type}: {path}")
    
    # Check submission readiness
    is_ready, blocking_issues = manager.validate_submission_readiness()
    print(f"\nSubmission Ready: {'✅ Yes' if is_ready else '❌ No'}")
    
    if blocking_issues:
        print("Blocking Issues:")
        for issue in blocking_issues:
            print(f"  • {issue}")


def example_generate_compliance_log():
    """Example of generating a compliance log for submission."""
    print("\n=== Generating Compliance Log ===")
    
    manager = ComplianceManager()
    
    # Generate human-readable summary
    summary_text = manager.generate_compliance_summary_text()
    
    # Save to compliance log file
    compliance_log_path = Path("compliance_log.txt")
    with open(compliance_log_path, 'w') as f:
        f.write(summary_text)
    
    print(f"Compliance log saved to: {compliance_log_path}")
    
    # Display first part of the log
    print("\nCompliance Log Preview:")
    print("=" * 50)
    lines = summary_text.split('\n')
    for line in lines[:20]:  # Show first 20 lines
        print(line)
    if len(lines) > 20:
        print("... (truncated)")


if __name__ == "__main__":
    setup_logging()
    
    try:
        # Run all examples
        example_license_tracking()
        example_data_source_validation()
        example_full_compliance_check()
        example_generate_compliance_log()
        
        print("\n" + "="*60)
        print("All compliance examples completed successfully!")
        print("Check the generated files for detailed compliance information.")
        
    except Exception as e:
        print(f"Error running compliance examples: {e}")
        import traceback
        traceback.print_exc()