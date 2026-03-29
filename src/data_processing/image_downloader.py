"""
Enhanced ImageDownloader with retry logic and caching for ML Product Pricing Challenge 2025
"""

import os
import time
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from functools import partial
from tqdm import tqdm
import pandas as pd
import requests
import urllib.request
from urllib.parse import urlparse
try:
    from ..config import config
    try:
        from ..utils import download_image as original_download_image
    except ImportError:
        original_download_image = None
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config
    try:
        from utils import download_image as original_download_image
    except ImportError:
        original_download_image = None


class ImageDownloadError(Exception):
    """Custom exception for image download errors"""
    pass


class ImageDownloader:
    """
    Enhanced ImageDownloader with retry logic and caching
    
    Implements exponential backoff retry mechanism with configurable attempts.
    Creates manifest-based caching system with download status and checksums.
    Adds batch downloading with progress tracking and comprehensive failure logging.
    """
    
    def __init__(self, data_config=None):
        """Initialize ImageDownloader with configuration"""
        self.config = data_config or config.data
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        self.image_dir = Path(self.config.image_dir)
        self.cache_dir = Path(self.config.cache_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Manifest file for tracking downloads
        self.manifest_file = self.cache_dir / "image_download_manifest.json"
        self.manifest = self._load_manifest()
        
        # Download statistics
        self.download_stats = {
            'total_requested': 0,
            'successful_downloads': 0,
            'cached_hits': 0,
            'failed_downloads': 0,
            'retry_attempts': 0,
            'total_size_bytes': 0,
            'download_time_seconds': 0.0
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def download_images(self, df: pd.DataFrame, image_dir: str) -> Dict[str, str]:
        """
        Download images with retry logic and caching (interface compatibility)
        
        Args:
            df: DataFrame containing image_link column
            image_dir: Directory to save images
            
        Returns:
            Dict mapping sample_id to local image path or error status
        """
        if 'image_link' not in df.columns:
            raise ImageDownloadError("DataFrame must contain 'image_link' column")
        
        if 'sample_id' not in df.columns:
            raise ImageDownloadError("DataFrame must contain 'sample_id' column")
        
        # Update image directory if different
        if image_dir != str(self.image_dir):
            self.image_dir = Path(image_dir)
            self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare download list
        download_list = []
        for _, row in df.iterrows():
            download_list.append({
                'sample_id': row['sample_id'],
                'image_url': row['image_link']
            })
        
        return self.batch_download_with_progress(download_list)
    
    def download_with_retry_and_cache(self, image_url: str, max_retries: int = 3) -> str:
        """
        Download image with retry logic and caching
        
        Args:
            image_url: URL of image to download
            max_retries: Maximum number of retry attempts
            
        Returns:
            str: Local path to downloaded image or error status
        """
        if not image_url or not isinstance(image_url, str):
            return "error_invalid_url"
        
        # Generate filename from URL
        filename = self._generate_filename(image_url)
        local_path = self.image_dir / filename
        
        # Check cache first
        cache_key = self._generate_cache_key(image_url)
        if cache_key in self.manifest:
            cached_entry = self.manifest[cache_key]
            if cached_entry['status'] == 'success' and local_path.exists():
                # Verify checksum if available
                if self._verify_checksum(local_path, cached_entry.get('checksum')):
                    self.download_stats['cached_hits'] += 1
                    return str(local_path)
                else:
                    self.logger.warning(f"Checksum mismatch for cached image: {image_url}")
                    # Remove invalid cache entry
                    del self.manifest[cache_key]
        
        # Download with retry logic
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                    self.logger.info(f"Retrying download after {wait_time}s: {image_url}")
                    time.sleep(wait_time)
                    self.download_stats['retry_attempts'] += 1
                
                # Attempt download
                success, result = self._download_single_image(image_url, local_path)
                
                if success:
                    # Calculate checksum
                    checksum = self._calculate_checksum(local_path)
                    file_size = local_path.stat().st_size
                    
                    # Update manifest
                    self.manifest[cache_key] = {
                        'url': image_url,
                        'local_path': str(local_path),
                        'status': 'success',
                        'checksum': checksum,
                        'file_size': file_size,
                        'download_timestamp': time.time(),
                        'attempts': attempt + 1
                    }
                    
                    self.download_stats['successful_downloads'] += 1
                    self.download_stats['total_size_bytes'] += file_size
                    
                    return str(local_path)
                else:
                    self.logger.warning(f"Download attempt {attempt + 1} failed: {result}")
                    
            except Exception as e:
                self.logger.error(f"Download attempt {attempt + 1} error: {str(e)}")
                result = str(e)
        
        # All attempts failed
        self.manifest[cache_key] = {
            'url': image_url,
            'local_path': None,
            'status': 'failed',
            'error': result,
            'download_timestamp': time.time(),
            'attempts': max_retries + 1
        }
        
        self.download_stats['failed_downloads'] += 1
        return f"error_download_failed_{result}"
    
    def _download_single_image(self, image_url: str, local_path: Path) -> Tuple[bool, str]:
        """
        Download a single image
        
        Args:
            image_url: URL to download
            local_path: Local path to save image
            
        Returns:
            Tuple of (success, result_message)
        """
        try:
            start_time = time.time()
            
            # Use requests for better control
            response = self.session.get(
                image_url, 
                timeout=self.config.download_timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
                return False, f"Invalid content type: {content_type}"
            
            # Write file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            download_time = time.time() - start_time
            self.download_stats['download_time_seconds'] += download_time
            
            # Verify file was written correctly
            if not local_path.exists() or local_path.stat().st_size == 0:
                return False, "File not written or empty"
            
            return True, "Success"
            
        except requests.exceptions.Timeout:
            return False, "Timeout"
        except requests.exceptions.ConnectionError:
            return False, "Connection error"
        except requests.exceptions.HTTPError as e:
            return False, f"HTTP error: {e.response.status_code}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def batch_download_with_progress(self, download_list: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Download images in batches with progress tracking
        
        Args:
            download_list: List of dicts with 'sample_id' and 'image_url'
            
        Returns:
            Dict mapping sample_id to local path or error status
        """
        self.logger.info(f"Starting batch download of {len(download_list)} images")
        start_time = time.time()
        
        self.download_stats['total_requested'] = len(download_list)
        results = {}
        
        # Use multiprocessing for parallel downloads
        download_func = partial(
            self._download_with_retry_wrapper,
            max_retries=self.config.max_download_retries
        )
        
        try:
            with multiprocessing.Pool(processes=min(50, len(download_list))) as pool:
                # Create progress bar
                with tqdm(total=len(download_list), desc="Downloading images") as pbar:
                    # Process in batches to avoid memory issues
                    batch_size = self.config.batch_size
                    for i in range(0, len(download_list), batch_size):
                        batch = download_list[i:i + batch_size]
                        
                        # Download batch
                        batch_results = pool.map(download_func, batch)
                        
                        # Process results
                        for item, result in zip(batch, batch_results):
                            results[item['sample_id']] = result
                            pbar.update(1)
                        
                        # Save manifest periodically
                        if i % (batch_size * 5) == 0:  # Every 5 batches
                            self._save_manifest()
        
        except Exception as e:
            self.logger.error(f"Batch download error: {str(e)}")
            # Fallback to sequential download
            self.logger.info("Falling back to sequential download")
            for item in tqdm(download_list, desc="Downloading images (sequential)"):
                result = self._download_with_retry_wrapper(item, self.config.max_download_retries)
                results[item['sample_id']] = result
        
        # Final manifest save
        self._save_manifest()
        
        total_time = time.time() - start_time
        self.logger.info(f"Batch download completed in {total_time:.2f}s")
        
        return results
    
    def _download_with_retry_wrapper(self, item: Dict[str, str], max_retries: int) -> str:
        """
        Wrapper for download_with_retry_and_cache for multiprocessing
        
        Args:
            item: Dict with 'sample_id' and 'image_url'
            max_retries: Maximum retry attempts
            
        Returns:
            str: Local path or error status
        """
        return self.download_with_retry_and_cache(item['image_url'], max_retries)
    
    def create_download_manifest(self, download_results: Dict) -> None:
        """
        Create manifest file for download tracking (interface compatibility)
        
        Args:
            download_results: Results from download operations
        """
        # This method is for interface compatibility
        # The manifest is automatically maintained by other methods
        self._save_manifest()
        
        # Create summary manifest
        summary_manifest = {
            'timestamp': time.time(),
            'total_downloads': len(download_results),
            'successful_downloads': sum(1 for v in download_results.values() if not v.startswith('error_')),
            'failed_downloads': sum(1 for v in download_results.values() if v.startswith('error_')),
            'download_stats': self.download_stats.copy(),
            'results': download_results
        }
        
        summary_file = self.cache_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_manifest, f, indent=2)
        
        self.logger.info(f"Download manifest created: {summary_file}")
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load existing manifest or create new one"""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    manifest = json.load(f)
                self.logger.info(f"Loaded existing manifest with {len(manifest)} entries")
                return manifest
            except Exception as e:
                self.logger.warning(f"Failed to load manifest: {str(e)}")
        
        return {}
    
    def _save_manifest(self):
        """Save manifest to disk"""
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save manifest: {str(e)}")
    
    def _generate_cache_key(self, image_url: str) -> str:
        """Generate cache key from image URL"""
        return hashlib.md5(image_url.encode()).hexdigest()
    
    def _generate_filename(self, image_url: str) -> str:
        """Generate filename from image URL"""
        try:
            parsed = urlparse(image_url)
            filename = Path(parsed.path).name
            
            # If no filename or extension, generate one
            if not filename or '.' not in filename:
                url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
                filename = f"image_{url_hash}.jpg"
            
            # Ensure safe filename
            filename = "".join(c for c in filename if c.isalnum() or c in '._-')
            
            return filename
        except Exception:
            # Fallback to hash-based filename
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
            return f"image_{url_hash}.jpg"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate checksum for {file_path}: {str(e)}")
            return ""
    
    def _verify_checksum(self, file_path: Path, expected_checksum: Optional[str]) -> bool:
        """Verify file checksum"""
        if not expected_checksum or not file_path.exists():
            return False
        
        actual_checksum = self._calculate_checksum(file_path)
        return actual_checksum == expected_checksum
    
    def get_download_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive download statistics
        
        Returns:
            Dict containing download statistics and analysis
        """
        stats = self.download_stats.copy()
        
        # Calculate derived statistics
        if stats['total_requested'] > 0:
            stats['success_rate'] = (stats['successful_downloads'] / stats['total_requested']) * 100
            stats['cache_hit_rate'] = (stats['cached_hits'] / stats['total_requested']) * 100
            stats['failure_rate'] = (stats['failed_downloads'] / stats['total_requested']) * 100
        
        if stats['successful_downloads'] > 0:
            stats['avg_download_time'] = stats['download_time_seconds'] / stats['successful_downloads']
            stats['avg_file_size'] = stats['total_size_bytes'] / stats['successful_downloads']
        
        # Add manifest statistics
        stats['manifest_entries'] = len(self.manifest)
        stats['cached_successes'] = sum(1 for entry in self.manifest.values() if entry['status'] == 'success')
        stats['cached_failures'] = sum(1 for entry in self.manifest.values() if entry['status'] == 'failed')
        
        return stats
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """
        Analyze download failures for debugging
        
        Returns:
            Dict containing failure analysis
        """
        failures = {entry['url']: entry for entry in self.manifest.values() if entry['status'] == 'failed'}
        
        if not failures:
            return {'total_failures': 0, 'failure_types': {}}
        
        # Categorize failure types
        failure_types = {}
        for entry in failures.values():
            error = entry.get('error', 'unknown')
            error_type = error.split('_')[0] if '_' in error else error
            failure_types[error_type] = failure_types.get(error_type, 0) + 1
        
        # Find most problematic URLs (multiple attempts)
        high_attempt_failures = [
            {'url': entry['url'], 'attempts': entry['attempts'], 'error': entry.get('error')}
            for entry in failures.values()
            if entry.get('attempts', 0) > 2
        ]
        
        return {
            'total_failures': len(failures),
            'failure_types': failure_types,
            'high_attempt_failures': high_attempt_failures[:10],  # Top 10
            'failure_rate_by_type': {
                error_type: (count / len(failures)) * 100
                for error_type, count in failure_types.items()
            }
        }
    
    def cleanup_failed_downloads(self) -> int:
        """
        Clean up files from failed downloads
        
        Returns:
            int: Number of files cleaned up
        """
        cleaned_count = 0
        
        for entry in self.manifest.values():
            if entry['status'] == 'failed' and entry.get('local_path'):
                local_path = Path(entry['local_path'])
                if local_path.exists():
                    try:
                        local_path.unlink()
                        cleaned_count += 1
                        self.logger.info(f"Cleaned up failed download: {local_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up {local_path}: {str(e)}")
        
        return cleaned_count
    
    def reset_download_stats(self):
        """Reset download statistics"""
        for key in self.download_stats:
            if isinstance(self.download_stats[key], (int, float)):
                self.download_stats[key] = 0
        self.logger.info("Download statistics reset")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self._save_manifest()
            if hasattr(self, 'session'):
                self.session.close()
        except Exception:
            pass  # Ignore cleanup errors