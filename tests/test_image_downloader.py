"""
Unit tests for Image Downloader.

Comprehensive tests covering retry logic, caching, batch downloading,
and error handling for image download functionality.
"""

import unittest
import tempfile
import shutil
import json
import time
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import requests

from src.data_processing.image_downloader import ImageDownloader, ImageDownloadError


class TestImageDownloader(unittest.TestCase):
    """Test cases for Image Downloader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.image_dir = str(self.image_dir)
        self.mock_config.cache_dir = str(self.cache_dir)
        self.mock_config.download_timeout = 30
        self.mock_config.max_download_retries = 3
        self.mock_config.batch_size = 10
        
        self.downloader = ImageDownloader(data_config=self.mock_config)
        
        # Test URLs
        self.test_urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.png",
            "https://example.com/image3.gif"
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test image downloader initialization."""
        self.assertTrue(self.image_dir.exists())
        self.assertTrue(self.cache_dir.exists())
        self.assertTrue(self.downloader.manifest_file.exists())
        self.assertIsInstance(self.downloader.manifest, dict)
        self.assertIsInstance(self.downloader.download_stats, dict)
        
        # Check download stats initialization
        expected_stats = ['total_requested', 'successful_downloads', 'cached_hits', 
                         'failed_downloads', 'retry_attempts', 'total_size_bytes', 
                         'download_time_seconds']
        for stat in expected_stats:
            self.assertIn(stat, self.downloader.download_stats)
            self.assertEqual(self.downloader.download_stats[stat], 0)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        url = "https://example.com/image.jpg"
        cache_key = self.downloader._generate_cache_key(url)
        
        # Should be MD5 hash
        expected_key = hashlib.md5(url.encode()).hexdigest()
        self.assertEqual(cache_key, expected_key)
        
        # Same URL should generate same key
        cache_key2 = self.downloader._generate_cache_key(url)
        self.assertEqual(cache_key, cache_key2)
        
        # Different URLs should generate different keys
        different_url = "https://example.com/different.jpg"
        different_key = self.downloader._generate_cache_key(different_url)
        self.assertNotEqual(cache_key, different_key)
    
    def test_generate_filename(self):
        """Test filename generation from URLs."""
        test_cases = [
            ("https://example.com/image.jpg", "image.jpg"),
            ("https://example.com/path/to/photo.png", "photo.png"),
            ("https://example.com/file.gif", "file.gif"),
        ]
        
        for url, expected_filename in test_cases:
            with self.subTest(url=url):
                filename = self.downloader._generate_filename(url)
                self.assertEqual(filename, expected_filename)
        
        # Test URL without filename
        url_no_filename = "https://example.com/"
        filename = self.downloader._generate_filename(url_no_filename)
        self.assertTrue(filename.startswith("image_"))
        self.assertTrue(filename.endswith(".jpg"))
    
    def test_calculate_checksum(self):
        """Test checksum calculation."""
        # Create a test file
        test_file = self.image_dir / "test.txt"
        test_content = b"test content for checksum"
        
        with open(test_file, 'wb') as f:
            f.write(test_content)
        
        checksum = self.downloader._calculate_checksum(test_file)
        
        # Verify checksum
        expected_checksum = hashlib.md5(test_content).hexdigest()
        self.assertEqual(checksum, expected_checksum)
        
        # Test with non-existent file
        non_existent = self.image_dir / "non_existent.txt"
        checksum_empty = self.downloader._calculate_checksum(non_existent)
        self.assertEqual(checksum_empty, "")
    
    def test_verify_checksum(self):
        """Test checksum verification."""
        # Create a test file
        test_file = self.image_dir / "test.txt"
        test_content = b"test content"
        
        with open(test_file, 'wb') as f:
            f.write(test_content)
        
        correct_checksum = hashlib.md5(test_content).hexdigest()
        wrong_checksum = "wrong_checksum"
        
        # Test correct checksum
        self.assertTrue(self.downloader._verify_checksum(test_file, correct_checksum))
        
        # Test wrong checksum
        self.assertFalse(self.downloader._verify_checksum(test_file, wrong_checksum))
        
        # Test with None checksum
        self.assertFalse(self.downloader._verify_checksum(test_file, None))
        
        # Test with non-existent file
        non_existent = self.image_dir / "non_existent.txt"
        self.assertFalse(self.downloader._verify_checksum(non_existent, correct_checksum))
    
    @patch('requests.Session.get')
    def test_download_single_image_success(self, mock_get):
        """Test successful single image download."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake image data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        url = "https://example.com/image.jpg"
        local_path = self.image_dir / "image.jpg"
        
        success, message = self.downloader._download_single_image(url, local_path)
        
        self.assertTrue(success)
        self.assertEqual(message, "Success")
        self.assertTrue(local_path.exists())
        
        # Verify file content
        with open(local_path, 'rb') as f:
            content = f.read()
        self.assertEqual(content, b'fake image data')
    
    @patch('requests.Session.get')
    def test_download_single_image_invalid_content_type(self, mock_get):
        """Test download with invalid content type."""
        # Mock response with invalid content type
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_get.return_value = mock_response
        
        url = "https://example.com/not_image.html"
        local_path = self.image_dir / "not_image.html"
        
        success, message = self.downloader._download_single_image(url, local_path)
        
        self.assertFalse(success)
        self.assertIn("Invalid content type", message)
    
    @patch('requests.Session.get')
    def test_download_single_image_timeout(self, mock_get):
        """Test download timeout handling."""
        mock_get.side_effect = requests.exceptions.Timeout()
        
        url = "https://example.com/image.jpg"
        local_path = self.image_dir / "image.jpg"
        
        success, message = self.downloader._download_single_image(url, local_path)
        
        self.assertFalse(success)
        self.assertEqual(message, "Timeout")
    
    @patch('requests.Session.get')
    def test_download_single_image_connection_error(self, mock_get):
        """Test download connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        url = "https://example.com/image.jpg"
        local_path = self.image_dir / "image.jpg"
        
        success, message = self.downloader._download_single_image(url, local_path)
        
        self.assertFalse(success)
        self.assertEqual(message, "Connection error")
    
    @patch('requests.Session.get')
    def test_download_single_image_http_error(self, mock_get):
        """Test download HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        url = "https://example.com/image.jpg"
        local_path = self.image_dir / "image.jpg"
        
        success, message = self.downloader._download_single_image(url, local_path)
        
        self.assertFalse(success)
        self.assertIn("HTTP error: 404", message)
    
    @patch.object(ImageDownloader, '_download_single_image')
    def test_download_with_retry_success_first_attempt(self, mock_download):
        """Test successful download on first attempt."""
        mock_download.return_value = (True, "Success")
        
        url = "https://example.com/image.jpg"
        result = self.downloader.download_with_retry_and_cache(url, max_retries=3)
        
        # Should return local path
        self.assertTrue(result.endswith("image.jpg"))
        self.assertEqual(mock_download.call_count, 1)
        
        # Check manifest update
        cache_key = self.downloader._generate_cache_key(url)
        self.assertIn(cache_key, self.downloader.manifest)
        self.assertEqual(self.downloader.manifest[cache_key]['status'], 'success')
    
    @patch.object(ImageDownloader, '_download_single_image')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_download_with_retry_success_after_retries(self, mock_sleep, mock_download):
        """Test successful download after retries."""
        # Fail first two attempts, succeed on third
        mock_download.side_effect = [
            (False, "Connection error"),
            (False, "Timeout"),
            (True, "Success")
        ]
        
        url = "https://example.com/image.jpg"
        result = self.downloader.download_with_retry_and_cache(url, max_retries=3)
        
        # Should return local path
        self.assertTrue(result.endswith("image.jpg"))
        self.assertEqual(mock_download.call_count, 3)
        
        # Check that sleep was called for retries
        self.assertEqual(mock_sleep.call_count, 2)
        
        # Check retry stats
        self.assertEqual(self.downloader.download_stats['retry_attempts'], 2)
    
    @patch.object(ImageDownloader, '_download_single_image')
    @patch('time.sleep')
    def test_download_with_retry_all_attempts_fail(self, mock_sleep, mock_download):
        """Test download failure after all retry attempts."""
        mock_download.return_value = (False, "Connection error")
        
        url = "https://example.com/image.jpg"
        result = self.downloader.download_with_retry_and_cache(url, max_retries=2)
        
        # Should return error status
        self.assertTrue(result.startswith("error_download_failed"))
        self.assertEqual(mock_download.call_count, 3)  # Initial + 2 retries
        
        # Check manifest update
        cache_key = self.downloader._generate_cache_key(url)
        self.assertIn(cache_key, self.downloader.manifest)
        self.assertEqual(self.downloader.manifest[cache_key]['status'], 'failed')
    
    def test_download_with_retry_invalid_url(self):
        """Test download with invalid URL."""
        invalid_urls = [None, "", "not_a_url", 123]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                result = self.downloader.download_with_retry_and_cache(url)
                self.assertEqual(result, "error_invalid_url")
    
    @patch.object(ImageDownloader, '_download_single_image')
    def test_download_with_cache_hit(self, mock_download):
        """Test cache hit scenario."""
        url = "https://example.com/image.jpg"
        
        # First download
        mock_download.return_value = (True, "Success")
        result1 = self.downloader.download_with_retry_and_cache(url)
        
        # Second download should hit cache
        result2 = self.downloader.download_with_retry_and_cache(url)
        
        self.assertEqual(result1, result2)
        self.assertEqual(mock_download.call_count, 1)  # Only called once
        self.assertEqual(self.downloader.download_stats['cached_hits'], 1)
    
    @patch.object(ImageDownloader, '_download_single_image')
    def test_download_with_cache_checksum_mismatch(self, mock_download):
        """Test cache with checksum mismatch."""
        url = "https://example.com/image.jpg"
        cache_key = self.downloader._generate_cache_key(url)
        
        # Create fake cache entry with wrong checksum
        local_path = self.image_dir / "image.jpg"
        local_path.write_bytes(b"fake content")
        
        self.downloader.manifest[cache_key] = {
            'status': 'success',
            'checksum': 'wrong_checksum'
        }
        
        # Mock successful re-download
        mock_download.return_value = (True, "Success")
        
        result = self.downloader.download_with_retry_and_cache(url)
        
        # Should re-download due to checksum mismatch
        self.assertEqual(mock_download.call_count, 1)
        self.assertNotIn(cache_key, self.downloader.manifest)  # Invalid entry removed
    
    def test_download_images_interface(self):
        """Test download_images interface method."""
        df = pd.DataFrame({
            'sample_id': ['1', '2', '3'],
            'image_link': self.test_urls
        })
        
        with patch.object(self.downloader, 'batch_download_with_progress') as mock_batch:
            mock_batch.return_value = {'1': 'path1', '2': 'path2', '3': 'path3'}
            
            result = self.downloader.download_images(df, str(self.image_dir))
            
            self.assertEqual(len(result), 3)
            mock_batch.assert_called_once()
            
            # Check that download list was prepared correctly
            call_args = mock_batch.call_args[0][0]
            self.assertEqual(len(call_args), 3)
            self.assertEqual(call_args[0]['sample_id'], '1')
            self.assertEqual(call_args[0]['image_url'], self.test_urls[0])
    
    def test_download_images_missing_columns(self):
        """Test download_images with missing required columns."""
        # Missing image_link column
        df1 = pd.DataFrame({
            'sample_id': ['1', '2'],
            'other_col': ['a', 'b']
        })
        
        with self.assertRaises(ImageDownloadError):
            self.downloader.download_images(df1, str(self.image_dir))
        
        # Missing sample_id column
        df2 = pd.DataFrame({
            'image_link': self.test_urls[:2],
            'other_col': ['a', 'b']
        })
        
        with self.assertRaises(ImageDownloadError):
            self.downloader.download_images(df2, str(self.image_dir))
    
    @patch.object(ImageDownloader, '_download_with_retry_wrapper')
    def test_batch_download_with_progress(self, mock_wrapper):
        """Test batch download with progress tracking."""
        download_list = [
            {'sample_id': '1', 'image_url': self.test_urls[0]},
            {'sample_id': '2', 'image_url': self.test_urls[1]},
            {'sample_id': '3', 'image_url': self.test_urls[2]}
        ]
        
        mock_wrapper.side_effect = ['path1', 'path2', 'path3']
        
        with patch('multiprocessing.Pool') as mock_pool:
            mock_pool_instance = Mock()
            mock_pool.return_value.__enter__.return_value = mock_pool_instance
            mock_pool_instance.map.return_value = ['path1', 'path2', 'path3']
            
            result = self.downloader.batch_download_with_progress(download_list)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result['1'], 'path1')
        self.assertEqual(result['2'], 'path2')
        self.assertEqual(result['3'], 'path3')
        
        # Check stats update
        self.assertEqual(self.downloader.download_stats['total_requested'], 3)
    
    @patch.object(ImageDownloader, '_download_with_retry_wrapper')
    def test_batch_download_fallback_to_sequential(self, mock_wrapper):
        """Test batch download fallback to sequential processing."""
        download_list = [
            {'sample_id': '1', 'image_url': self.test_urls[0]},
            {'sample_id': '2', 'image_url': self.test_urls[1]}
        ]
        
        mock_wrapper.side_effect = ['path1', 'path2']
        
        # Mock multiprocessing to raise exception
        with patch('multiprocessing.Pool', side_effect=Exception("Pool error")):
            result = self.downloader.batch_download_with_progress(download_list)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result['1'], 'path1')
        self.assertEqual(result['2'], 'path2')
    
    def test_create_download_manifest(self):
        """Test download manifest creation."""
        download_results = {
            '1': 'path1',
            '2': 'error_download_failed',
            '3': 'path3'
        }
        
        self.downloader.create_download_manifest(download_results)
        
        # Check that summary file was created
        summary_file = self.cache_dir / "download_summary.json"
        self.assertTrue(summary_file.exists())
        
        # Check summary content
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        self.assertEqual(summary['total_downloads'], 3)
        self.assertEqual(summary['successful_downloads'], 2)
        self.assertEqual(summary['failed_downloads'], 1)
        self.assertEqual(summary['results'], download_results)
    
    def test_get_download_statistics(self):
        """Test download statistics calculation."""
        # Set some test stats
        self.downloader.download_stats.update({
            'total_requested': 10,
            'successful_downloads': 8,
            'cached_hits': 2,
            'failed_downloads': 2,
            'download_time_seconds': 16.0,
            'total_size_bytes': 8000
        })
        
        stats = self.downloader.get_download_statistics()
        
        # Check derived statistics
        self.assertEqual(stats['success_rate'], 80.0)
        self.assertEqual(stats['cache_hit_rate'], 20.0)
        self.assertEqual(stats['failure_rate'], 20.0)
        self.assertEqual(stats['avg_download_time'], 2.0)
        self.assertEqual(stats['avg_file_size'], 1000.0)
    
    def test_get_failure_analysis(self):
        """Test failure analysis generation."""
        # Add some failed entries to manifest
        self.downloader.manifest.update({
            'key1': {
                'url': 'url1',
                'status': 'failed',
                'error': 'timeout_error',
                'attempts': 3
            },
            'key2': {
                'url': 'url2',
                'status': 'failed',
                'error': 'connection_error',
                'attempts': 2
            },
            'key3': {
                'url': 'url3',
                'status': 'success'
            }
        })
        
        analysis = self.downloader.get_failure_analysis()
        
        self.assertEqual(analysis['total_failures'], 2)
        self.assertIn('timeout', analysis['failure_types'])
        self.assertIn('connection', analysis['failure_types'])
        self.assertEqual(len(analysis['high_attempt_failures']), 1)  # Only key1 has >2 attempts
    
    def test_cleanup_failed_downloads(self):
        """Test cleanup of failed download files."""
        # Create some test files
        failed_file1 = self.image_dir / "failed1.jpg"
        failed_file2 = self.image_dir / "failed2.jpg"
        success_file = self.image_dir / "success.jpg"
        
        failed_file1.write_bytes(b"fake content")
        failed_file2.write_bytes(b"fake content")
        success_file.write_bytes(b"fake content")
        
        # Add manifest entries
        self.downloader.manifest.update({
            'key1': {
                'status': 'failed',
                'local_path': str(failed_file1)
            },
            'key2': {
                'status': 'failed',
                'local_path': str(failed_file2)
            },
            'key3': {
                'status': 'success',
                'local_path': str(success_file)
            }
        })
        
        cleaned_count = self.downloader.cleanup_failed_downloads()
        
        self.assertEqual(cleaned_count, 2)
        self.assertFalse(failed_file1.exists())
        self.assertFalse(failed_file2.exists())
        self.assertTrue(success_file.exists())  # Success file should remain
    
    def test_reset_download_stats(self):
        """Test resetting download statistics."""
        # Set some test stats
        self.downloader.download_stats.update({
            'total_requested': 10,
            'successful_downloads': 8,
            'failed_downloads': 2,
            'download_time_seconds': 16.0
        })
        
        self.downloader.reset_download_stats()
        
        # All numeric stats should be reset to 0
        for key, value in self.downloader.download_stats.items():
            if isinstance(value, (int, float)):
                self.assertEqual(value, 0)
    
    def test_manifest_persistence(self):
        """Test manifest loading and saving."""
        # Add entry to manifest
        test_entry = {
            'url': 'test_url',
            'status': 'success',
            'checksum': 'test_checksum'
        }
        self.downloader.manifest['test_key'] = test_entry
        
        # Save manifest
        self.downloader._save_manifest()
        
        # Create new downloader instance (should load existing manifest)
        new_downloader = ImageDownloader(data_config=self.mock_config)
        
        # Check that manifest was loaded
        self.assertIn('test_key', new_downloader.manifest)
        self.assertEqual(new_downloader.manifest['test_key'], test_entry)
    
    def test_manifest_loading_error_handling(self):
        """Test manifest loading with corrupted file."""
        # Create corrupted manifest file
        with open(self.downloader.manifest_file, 'w') as f:
            f.write("invalid json content")
        
        # Create new downloader (should handle corrupted manifest gracefully)
        new_downloader = ImageDownloader(data_config=self.mock_config)
        
        # Should have empty manifest
        self.assertEqual(new_downloader.manifest, {})
    
    def test_download_with_retry_wrapper(self):
        """Test download wrapper for multiprocessing."""
        item = {
            'sample_id': '1',
            'image_url': 'https://example.com/image.jpg'
        }
        
        with patch.object(self.downloader, 'download_with_retry_and_cache') as mock_download:
            mock_download.return_value = 'test_path'
            
            result = self.downloader._download_with_retry_wrapper(item, 3)
            
            self.assertEqual(result, 'test_path')
            mock_download.assert_called_once_with('https://example.com/image.jpg', 3)


if __name__ == '__main__':
    unittest.main()