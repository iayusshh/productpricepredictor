"""
Performance and Resource Validation Tests.

Tests for GPU memory usage, inference timing constraints, storage requirements,
cache integrity, and system resource validation.
"""

import unittest
import tempfile
import shutil
import time
import psutil
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import json

from src.infrastructure.resource_manager import ResourceManager
from src.infrastructure.cache_manager import CacheManager
from src.infrastructure.logging_manager import LoggingManager
from src.config import config


class TestPerformanceValidation(unittest.TestCase):
    """Test cases for Performance and Resource Validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        (self.temp_path / "cache").mkdir()
        (self.temp_path / "logs").mkdir()
        (self.temp_path / "models").mkdir()
        
        # Mock config
        self.original_config = config
        self.mock_config()
        
        # Initialize managers
        self.resource_manager = ResourceManager()
        self.cache_manager = CacheManager()
        self.logging_manager = LoggingManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original config
        config.__dict__.update(self.original_config.__dict__)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def mock_config(self):
        """Mock configuration for testing."""
        config.infrastructure.cache_dir = str(self.temp_path / "cache")
        config.infrastructure.log_dir = str(self.temp_path / "logs")
        config.infrastructure.model_dir = str(self.temp_path / "models")
        config.infrastructure.max_memory_gb = 16
        config.infrastructure.max_inference_time_seconds = 300
        config.infrastructure.max_storage_gb = 50
    
    def test_gpu_memory_usage_validation(self):
        """Test GPU memory usage monitoring and validation."""
        # Test GPU memory monitoring
        gpu_info = self.resource_manager.get_gpu_info()
        
        self.assertIsInstance(gpu_info, dict)
        self.assertIn('gpu_available', gpu_info)
        
        if gpu_info['gpu_available']:
            # If GPU is available, test memory monitoring
            self.assertIn('gpu_count', gpu_info)
            self.assertIn('gpu_memory', gpu_info)
            
            for gpu_id, memory_info in gpu_info['gpu_memory'].items():
                self.assertIn('total', memory_info)
                self.assertIn('used', memory_info)
                self.assertIn('free', memory_info)
                
                # Memory values should be non-negative
                self.assertGreaterEqual(memory_info['total'], 0)
                self.assertGreaterEqual(memory_info['used'], 0)
                self.assertGreaterEqual(memory_info['free'], 0)
                
                # Used + free should approximately equal total
                self.assertAlmostEqual(
                    memory_info['used'] + memory_info['free'],
                    memory_info['total'],
                    delta=memory_info['total'] * 0.1  # 10% tolerance
                )
        else:
            # If no GPU, should handle gracefully
            self.assertEqual(gpu_info['gpu_count'], 0)
    
    def test_memory_usage_monitoring(self):
        """Test system memory usage monitoring."""
        memory_info = self.resource_manager.get_memory_info()
        
        self.assertIsInstance(memory_info, dict)
        self.assertIn('total', memory_info)
        self.assertIn('available', memory_info)
        self.assertIn('used', memory_info)
        self.assertIn('percentage', memory_info)
        
        # Memory values should be positive
        self.assertGreater(memory_info['total'], 0)
        self.assertGreaterEqual(memory_info['available'], 0)
        self.assertGreaterEqual(memory_info['used'], 0)
        
        # Percentage should be between 0 and 100
        self.assertGreaterEqual(memory_info['percentage'], 0)
        self.assertLessEqual(memory_info['percentage'], 100)
        
        # Used + available should approximately equal total
        self.assertAlmostEqual(
            memory_info['used'] + memory_info['available'],
            memory_info['total'],
            delta=memory_info['total'] * 0.1  # 10% tolerance
        )
    
    def test_memory_constraint_validation(self):
        """Test memory constraint validation."""
        # Test memory requirement validation
        required_memory_gb = 8
        can_allocate = self.resource_manager.validate_memory_requirements(required_memory_gb)
        
        self.assertIsInstance(can_allocate, bool)
        
        # Get current memory info to verify logic
        memory_info = self.resource_manager.get_memory_info()
        available_gb = memory_info['available'] / (1024**3)  # Convert to GB
        
        if available_gb >= required_memory_gb:
            self.assertTrue(can_allocate)
        else:
            self.assertFalse(can_allocate)
        
        # Test with unreasonably high memory requirement
        unreasonable_memory_gb = 1000  # 1TB
        can_allocate_unreasonable = self.resource_manager.validate_memory_requirements(unreasonable_memory_gb)
        self.assertFalse(can_allocate_unreasonable)
    
    def test_inference_timing_validation(self):
        """Test inference timing constraints validation."""
        # Simulate inference timing
        test_sample_count = 1000
        
        def mock_inference_function():
            """Mock inference function with controlled timing."""
            time.sleep(0.1)  # Simulate 0.1 seconds of processing
            return np.random.random(test_sample_count)
        
        # Measure inference timing
        timing_results = self.resource_manager.measure_inference_timing(
            mock_inference_function, test_sample_count
        )
        
        self.assertIsInstance(timing_results, dict)
        self.assertIn('total_time', timing_results)
        self.assertIn('samples_per_second', timing_results)
        self.assertIn('time_per_sample', timing_results)
        self.assertIn('meets_constraints', timing_results)
        
        # Verify timing calculations
        self.assertGreater(timing_results['total_time'], 0)
        self.assertGreater(timing_results['samples_per_second'], 0)
        self.assertGreater(timing_results['time_per_sample'], 0)
        
        # Check constraint validation
        expected_samples_per_second = test_sample_count / timing_results['total_time']
        self.assertAlmostEqual(
            timing_results['samples_per_second'],
            expected_samples_per_second,
            places=2
        )
    
    def test_large_dataset_inference_timing(self):
        """Test inference timing with large dataset (75k samples)."""
        large_sample_count = 75000
        
        def fast_mock_inference():
            """Fast mock inference for large dataset."""
            # Simulate very fast inference (vectorized operations)
            return np.random.random(large_sample_count)
        
        # Measure timing for large dataset
        timing_results = self.resource_manager.measure_inference_timing(
            fast_mock_inference, large_sample_count
        )
        
        # Should complete within reasonable time
        max_allowed_time = config.infrastructure.max_inference_time_seconds
        self.assertLessEqual(timing_results['total_time'], max_allowed_time)
        
        # Should process at reasonable rate
        min_samples_per_second = 100  # At least 100 samples per second
        self.assertGreaterEqual(timing_results['samples_per_second'], min_samples_per_second)
    
    def test_storage_requirements_calculation(self):
        """Test storage requirements calculation and validation."""
        # Create test files to calculate storage
        test_files = {
            'model.pkl': 50 * 1024 * 1024,  # 50MB
            'embeddings.npy': 100 * 1024 * 1024,  # 100MB
            'cache.json': 10 * 1024 * 1024,  # 10MB
        }
        
        for filename, size in test_files.items():
            file_path = self.temp_path / filename
            file_path.write_bytes(b'0' * size)
        
        # Calculate storage requirements
        storage_info = self.resource_manager.calculate_storage_requirements(str(self.temp_path))
        
        self.assertIsInstance(storage_info, dict)
        self.assertIn('total_size_bytes', storage_info)
        self.assertIn('total_size_gb', storage_info)
        self.assertIn('file_breakdown', storage_info)
        self.assertIn('meets_constraints', storage_info)
        
        # Verify calculations
        expected_total_bytes = sum(test_files.values())
        self.assertAlmostEqual(
            storage_info['total_size_bytes'],
            expected_total_bytes,
            delta=1024  # 1KB tolerance
        )
        
        expected_total_gb = expected_total_bytes / (1024**3)
        self.assertAlmostEqual(
            storage_info['total_size_gb'],
            expected_total_gb,
            places=3
        )
        
        # Check file breakdown
        file_breakdown = storage_info['file_breakdown']
        self.assertEqual(len(file_breakdown), len(test_files))
        
        for file_info in file_breakdown:
            self.assertIn('filename', file_info)
            self.assertIn('size_bytes', file_info)
            self.assertIn('size_mb', file_info)
    
    def test_cache_integrity_validation(self):
        """Test cache integrity and validation."""
        # Create test cache files
        cache_data = {
            'embeddings': {
                'text_embeddings': np.random.random((100, 768)),
                'image_embeddings': np.random.random((100, 512))
            },
            'metadata': {
                'version': '1.0',
                'created_at': time.time(),
                'sample_count': 100
            }
        }
        
        # Save cache data
        cache_file = self.temp_path / "cache" / "test_cache.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'metadata': cache_data['metadata'],
                'embeddings_shape': {
                    'text': list(cache_data['embeddings']['text_embeddings'].shape),
                    'image': list(cache_data['embeddings']['image_embeddings'].shape)
                }
            }, f)
        
        # Save embedding arrays
        np.save(self.temp_path / "cache" / "text_embeddings.npy", cache_data['embeddings']['text_embeddings'])
        np.save(self.temp_path / "cache" / "image_embeddings.npy", cache_data['embeddings']['image_embeddings'])
        
        # Validate cache integrity
        integrity_results = self.cache_manager.validate_cache_integrity()
        
        self.assertIsInstance(integrity_results, dict)
        self.assertIn('is_valid', integrity_results)
        self.assertTrue(integrity_results['is_valid'])
        
        self.assertIn('cache_files', integrity_results)
        self.assertIn('total_size', integrity_results)
        self.assertIn('validation_details', integrity_results)
        
        # Check cache files
        cache_files = integrity_results['cache_files']
        self.assertGreater(len(cache_files), 0)
        
        for cache_file_info in cache_files:
            self.assertIn('filename', cache_file_info)
            self.assertIn('size_bytes', cache_file_info)
            self.assertIn('is_valid', cache_file_info)
            self.assertTrue(cache_file_info['is_valid'])
    
    def test_cache_checksum_validation(self):
        """Test cache checksum validation."""
        # Create test file with known content
        test_content = b"test cache content for checksum validation"
        test_file = self.temp_path / "cache" / "test_file.bin"
        test_file.write_bytes(test_content)
        
        # Calculate and store checksum
        checksum = self.cache_manager.calculate_checksum(test_file)
        
        # Validate checksum
        is_valid = self.cache_manager.validate_checksum(test_file, checksum)
        self.assertTrue(is_valid)
        
        # Test with wrong checksum
        wrong_checksum = "wrong_checksum_value"
        is_valid_wrong = self.cache_manager.validate_checksum(test_file, wrong_checksum)
        self.assertFalse(is_valid_wrong)
        
        # Test with corrupted file
        test_file.write_bytes(b"corrupted content")
        is_valid_corrupted = self.cache_manager.validate_checksum(test_file, checksum)
        self.assertFalse(is_valid_corrupted)
    
    def test_structured_logging_format_validation(self):
        """Test structured logging format and completeness."""
        # Generate test log entries
        test_log_entries = [
            {
                'timestamp': time.time(),
                'level': 'INFO',
                'message': 'Test log message 1',
                'module': 'test_module',
                'metrics': {'accuracy': 0.85, 'loss': 0.15}
            },
            {
                'timestamp': time.time(),
                'level': 'ERROR',
                'message': 'Test error message',
                'module': 'test_module',
                'error_details': {'error_type': 'ValueError', 'error_message': 'Invalid input'}
            }
        ]
        
        # Write test logs
        log_file = self.temp_path / "logs" / "test.log"
        with open(log_file, 'w') as f:
            for entry in test_log_entries:
                f.write(json.dumps(entry) + '\n')
        
        # Validate log format
        log_validation = self.logging_manager.validate_log_format(log_file)
        
        self.assertIsInstance(log_validation, dict)
        self.assertIn('is_valid', log_validation)
        self.assertTrue(log_validation['is_valid'])
        
        self.assertIn('total_entries', log_validation)
        self.assertEqual(log_validation['total_entries'], len(test_log_entries))
        
        self.assertIn('valid_entries', log_validation)
        self.assertEqual(log_validation['valid_entries'], len(test_log_entries))
        
        self.assertIn('format_errors', log_validation)
        self.assertEqual(len(log_validation['format_errors']), 0)
    
    def test_logging_completeness_validation(self):
        """Test logging completeness and required fields."""
        # Create logs with missing required fields
        incomplete_log_entries = [
            {
                'timestamp': time.time(),
                'level': 'INFO',
                'message': 'Complete log entry',
                'module': 'test_module'
            },
            {
                'level': 'ERROR',  # Missing timestamp
                'message': 'Incomplete log entry',
                'module': 'test_module'
            },
            {
                'timestamp': time.time(),
                'message': 'Missing level',  # Missing level
                'module': 'test_module'
            }
        ]
        
        # Write incomplete logs
        log_file = self.temp_path / "logs" / "incomplete.log"
        with open(log_file, 'w') as f:
            for entry in incomplete_log_entries:
                f.write(json.dumps(entry) + '\n')
        
        # Validate completeness
        completeness_validation = self.logging_manager.validate_log_completeness(log_file)
        
        self.assertIsInstance(completeness_validation, dict)
        self.assertIn('is_complete', completeness_validation)
        self.assertFalse(completeness_validation['is_complete'])  # Should fail due to missing fields
        
        self.assertIn('missing_fields', completeness_validation)
        missing_fields = completeness_validation['missing_fields']
        self.assertGreater(len(missing_fields), 0)
        
        # Should identify missing timestamps and levels
        self.assertTrue(any('timestamp' in error for error in missing_fields))
        self.assertTrue(any('level' in error for error in missing_fields))
    
    def test_concurrent_resource_monitoring(self):
        """Test concurrent resource monitoring and thread safety."""
        results = []
        
        def monitor_resources():
            """Monitor resources in separate thread."""
            for _ in range(5):
                memory_info = self.resource_manager.get_memory_info()
                results.append(memory_info)
                time.sleep(0.1)
        
        # Start multiple monitoring threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=monitor_resources)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 15)  # 3 threads × 5 measurements each
        
        for memory_info in results:
            self.assertIsInstance(memory_info, dict)
            self.assertIn('total', memory_info)
            self.assertIn('available', memory_info)
            self.assertIn('used', memory_info)
    
    def test_resource_limit_enforcement(self):
        """Test resource limit enforcement and warnings."""
        # Test memory limit enforcement
        memory_limit_gb = 1  # Very low limit for testing
        
        # Mock high memory usage
        with patch.object(psutil, 'virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16 * 1024**3,  # 16GB total
                available=0.5 * 1024**3,  # 0.5GB available (below limit)
                used=15.5 * 1024**3,  # 15.5GB used
                percent=96.9
            )
            
            # Check memory constraint
            can_allocate = self.resource_manager.validate_memory_requirements(memory_limit_gb)
            self.assertFalse(can_allocate)
            
            # Get memory warnings
            warnings = self.resource_manager.get_resource_warnings()
            self.assertIsInstance(warnings, list)
            self.assertGreater(len(warnings), 0)
            
            # Should have memory warning
            memory_warnings = [w for w in warnings if 'memory' in w.lower()]
            self.assertGreater(len(memory_warnings), 0)
    
    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        # Create baseline performance metrics
        baseline_metrics = {
            'inference_time_per_sample': 0.001,  # 1ms per sample
            'memory_usage_mb': 512,
            'throughput_samples_per_second': 1000
        }
        
        # Save baseline
        baseline_file = self.temp_path / "performance_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_metrics, f)
        
        # Test current performance (simulated)
        current_metrics = {
            'inference_time_per_sample': 0.0015,  # 1.5ms per sample (50% slower)
            'memory_usage_mb': 600,  # 17% more memory
            'throughput_samples_per_second': 667  # 33% slower throughput
        }
        
        # Detect regression
        regression_results = self.resource_manager.detect_performance_regression(
            current_metrics, baseline_file
        )
        
        self.assertIsInstance(regression_results, dict)
        self.assertIn('has_regression', regression_results)
        self.assertTrue(regression_results['has_regression'])
        
        self.assertIn('regressions', regression_results)
        regressions = regression_results['regressions']
        
        # Should detect regressions in all metrics
        self.assertGreater(len(regressions), 0)
        
        for regression in regressions:
            self.assertIn('metric', regression)
            self.assertIn('baseline_value', regression)
            self.assertIn('current_value', regression)
            self.assertIn('change_percent', regression)
            self.assertIn('is_regression', regression)
            self.assertTrue(regression['is_regression'])
    
    def test_disk_space_validation(self):
        """Test disk space validation and monitoring."""
        # Get disk usage for temp directory
        disk_info = self.resource_manager.get_disk_usage(str(self.temp_path))
        
        self.assertIsInstance(disk_info, dict)
        self.assertIn('total', disk_info)
        self.assertIn('used', disk_info)
        self.assertIn('free', disk_info)
        self.assertIn('percentage', disk_info)
        
        # Values should be positive
        self.assertGreater(disk_info['total'], 0)
        self.assertGreaterEqual(disk_info['used'], 0)
        self.assertGreaterEqual(disk_info['free'], 0)
        
        # Percentage should be reasonable
        self.assertGreaterEqual(disk_info['percentage'], 0)
        self.assertLessEqual(disk_info['percentage'], 100)
        
        # Test disk space requirement validation
        required_space_gb = 1  # 1GB
        has_space = self.resource_manager.validate_disk_space_requirements(
            str(self.temp_path), required_space_gb
        )
        
        self.assertIsInstance(has_space, bool)
        
        # Should have space for 1GB (unless disk is very full)
        free_gb = disk_info['free'] / (1024**3)
        if free_gb >= required_space_gb:
            self.assertTrue(has_space)
        else:
            self.assertFalse(has_space)
    
    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring."""
        # Get CPU information
        cpu_info = self.resource_manager.get_cpu_info()
        
        self.assertIsInstance(cpu_info, dict)
        self.assertIn('cpu_count', cpu_info)
        self.assertIn('cpu_percent', cpu_info)
        self.assertIn('load_average', cpu_info)
        
        # CPU count should be positive
        self.assertGreater(cpu_info['cpu_count'], 0)
        
        # CPU percentage should be between 0 and 100
        self.assertGreaterEqual(cpu_info['cpu_percent'], 0)
        self.assertLessEqual(cpu_info['cpu_percent'], 100)
        
        # Load average should be non-negative
        if 'load_average' in cpu_info and cpu_info['load_average'] is not None:
            for load in cpu_info['load_average']:
                self.assertGreaterEqual(load, 0)
    
    def test_comprehensive_resource_validation(self):
        """Test comprehensive resource validation."""
        # Run complete resource validation
        validation_results = self.resource_manager.run_comprehensive_validation()
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn('overall_valid', validation_results)
        
        # Check individual validation categories
        expected_categories = [
            'memory_validation', 'disk_validation', 'cpu_validation',
            'performance_validation', 'cache_validation'
        ]
        
        for category in expected_categories:
            if category in validation_results:
                self.assertIn('is_valid', validation_results[category])
                self.assertIsInstance(validation_results[category]['is_valid'], bool)
        
        # Check validation summary
        self.assertIn('validation_summary', validation_results)
        summary = validation_results['validation_summary']
        
        self.assertIn('total_checks', summary)
        self.assertIn('passed_checks', summary)
        self.assertIn('failed_checks', summary)
        self.assertIn('warnings', summary)
        
        # Counts should be non-negative
        self.assertGreaterEqual(summary['total_checks'], 0)
        self.assertGreaterEqual(summary['passed_checks'], 0)
        self.assertGreaterEqual(summary['failed_checks'], 0)
        
        # Passed + failed should equal total
        self.assertEqual(
            summary['passed_checks'] + summary['failed_checks'],
            summary['total_checks']
        )


if __name__ == '__main__':
    unittest.main()