#!/usr/bin/env python3
"""
Test runner script for ML Product Pricing Challenge 2025.

Runs all unit tests, integration tests, and performance validation tests
with coverage reporting and detailed test results.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json
import time


def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        end_time = time.time()
        
        print(f"✅ SUCCESS ({end_time - start_time:.2f}s)")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        return True, result
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        
        print(f"❌ FAILED ({end_time - start_time:.2f}s)")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        
        return False, e


def run_unit_tests(coverage=True):
    """Run unit tests with optional coverage."""
    print("\n🧪 Running Unit Tests...")
    
    if coverage:
        # Run with coverage
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_ipq_extractor.py",
            "tests/test_price_normalizer.py", 
            "tests/test_image_downloader.py",
            "tests/test_catalog_parser.py",
            "tests/test_data_loader.py",
            "tests/test_smape_calculator.py",
            "tests/test_baseline_validator.py",
            "tests/test_evaluation_reporter.py",
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-fail-under=80",
            "-v",
            "--tb=short"
        ]
    else:
        # Run without coverage
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_ipq_extractor.py",
            "tests/test_price_normalizer.py", 
            "tests/test_image_downloader.py",
            "tests/test_catalog_parser.py",
            "tests/test_data_loader.py",
            "tests/test_smape_calculator.py",
            "tests/test_baseline_validator.py",
            "tests/test_evaluation_reporter.py",
            "-v",
            "--tb=short"
        ]
    
    return run_command(command, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running Integration Tests...")
    
    command = [
        sys.executable, "-m", "pytest",
        "tests/test_integration_training_pipeline.py",
        "tests/test_integration_prediction_pipeline.py",
        "tests/test_integration_deliverable_compliance.py",
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure for integration tests
    ]
    
    return run_command(command, "Integration Tests")


def run_performance_tests():
    """Run performance and resource validation tests."""
    print("\n⚡ Running Performance Tests...")
    
    command = [
        sys.executable, "-m", "pytest",
        "tests/test_performance_validation.py",
        "tests/test_license_compliance_validation.py",
        "-v",
        "--tb=short"
    ]
    
    return run_command(command, "Performance and Compliance Tests")


def check_test_coverage():
    """Check and report test coverage."""
    print("\n📊 Checking Test Coverage...")
    
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        print("❌ Coverage file not found. Run tests with coverage first.")
        return False
    
    try:
        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)
        
        total_coverage = coverage_data['totals']['percent_covered']
        
        print(f"\n📈 Overall Coverage: {total_coverage:.1f}%")
        
        # Check coverage by module
        files = coverage_data['files']
        critical_modules = [
            'src/features/ipq_extractor.py',
            'src/data_processing/price_normalizer.py',
            'src/data_processing/image_downloader.py',
            'src/features/catalog_parser.py',
            'src/data_processing/data_loader.py',
            'src/evaluation/smape_calculator.py'
        ]
        
        print("\n📋 Coverage by Critical Module:")
        for module in critical_modules:
            if module in files:
                module_coverage = files[module]['summary']['percent_covered']
                status = "✅" if module_coverage >= 80 else "❌"
                print(f"  {status} {module}: {module_coverage:.1f}%")
            else:
                print(f"  ❓ {module}: Not found in coverage report")
        
        # Overall coverage check
        if total_coverage >= 80:
            print(f"\n✅ Coverage requirement met: {total_coverage:.1f}% >= 80%")
            return True
        else:
            print(f"\n❌ Coverage requirement not met: {total_coverage:.1f}% < 80%")
            return False
            
    except Exception as e:
        print(f"❌ Error reading coverage data: {e}")
        return False


def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n📄 Generating Test Report...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_categories': {
            'unit_tests': 'Tests for individual modules and functions',
            'integration_tests': 'Tests for component integration and pipelines',
            'performance_tests': 'Tests for performance and resource validation',
            'compliance_tests': 'Tests for license compliance and data validation'
        },
        'coverage_requirement': '≥80% for critical modules',
        'test_files': [
            'tests/test_ipq_extractor.py',
            'tests/test_price_normalizer.py',
            'tests/test_image_downloader.py',
            'tests/test_catalog_parser.py',
            'tests/test_data_loader.py',
            'tests/test_smape_calculator.py',
            'tests/test_baseline_validator.py',
            'tests/test_evaluation_reporter.py',
            'tests/test_integration_training_pipeline.py',
            'tests/test_integration_prediction_pipeline.py',
            'tests/test_integration_deliverable_compliance.py',
            'tests/test_performance_validation.py',
            'tests/test_license_compliance_validation.py'
        ]
    }
    
    # Save report
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Test report saved to test_report.json")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run ML Product Pricing tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--no-coverage', action='store_true', help='Skip coverage reporting')
    parser.add_argument('--fast', action='store_true', help='Run fast tests only (skip integration)')
    
    args = parser.parse_args()
    
    print("🚀 ML Product Pricing Challenge 2025 - Test Suite")
    print("=" * 60)
    
    success = True
    
    # Determine which tests to run
    if args.unit:
        success &= run_unit_tests(coverage=not args.no_coverage)[0]
    elif args.integration:
        success &= run_integration_tests()[0]
    elif args.performance:
        success &= run_performance_tests()[0]
    elif args.fast:
        # Run unit tests and performance tests (skip integration)
        success &= run_unit_tests(coverage=not args.no_coverage)[0]
        success &= run_performance_tests()[0]
    else:
        # Run all tests
        success &= run_unit_tests(coverage=not args.no_coverage)[0]
        success &= run_integration_tests()[0]
        success &= run_performance_tests()[0]
    
    # Check coverage if it was generated
    if not args.no_coverage and not args.integration and not args.performance:
        success &= check_test_coverage()
    
    # Generate test report
    generate_test_report()
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Test suite completed successfully")
        if not args.no_coverage:
            print("📊 Coverage requirements met")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED!")
        print("❌ Please check the output above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()