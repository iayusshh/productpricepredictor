#!/usr/bin/env python3
"""
Basic Image Downloader using only standard library
For ML Product Pricing Challenge 2025
"""

import csv
import json
import time
import hashlib
import sqlite3
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from urllib.error import URLError, HTTPError
import threading
from concurrent.futures import ThreadPoolExecutor


class BasicImageDownloader:
    """Basic image downloader using only standard library."""
    
    def __init__(self, base_dir="images", db_path="image_tracking.db"):
        self.base_dir = Path(base_dir)
        self.db_path = db_path
        
        # Create directories
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "train").mkdir(exist_ok=True)
        (self.base_dir / "test").mkdir(exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Thread-safe counters
        self.lock = threading.Lock()
        self.stats = {
            'downloaded': 0,
            'cached': 0,
            'failed': 0,
            'duplicates': 0
        }
    
    def init_database(self):
        """Initialize SQLite database for tracking downloads."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_downloads (
                sample_id TEXT PRIMARY KEY,
                dataset_type TEXT NOT NULL,
                image_url TEXT NOT NULL,
                url_hash TEXT NOT NULL,
                local_path TEXT,
                file_size INTEGER,
                download_status TEXT NOT NULL,
                download_time REAL,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS url_deduplication (
                url_hash TEXT PRIMARY KEY,
                original_url TEXT NOT NULL,
                first_sample_id TEXT NOT NULL,
                reference_count INTEGER DEFAULT 1,
                master_file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_url_hash(self, url):
        """Generate hash for URL deduplication."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def get_safe_filename(self, url, sample_id):
        """Generate safe filename from URL and sample_id."""
        try:
            parsed = urlparse(url)
            original_name = Path(parsed.path).name
            
            if not original_name or '.' not in original_name:
                extension = '.jpg'
            else:
                extension = Path(original_name).suffix
                if not extension:
                    extension = '.jpg'
            
            safe_name = f"{sample_id}{extension}"
            return safe_name
            
        except Exception:
            return f"{sample_id}.jpg"
    
    def download_single_image(self, sample_id, url, dataset_type, timeout=30, max_retries=3):
        """Download a single image with comprehensive tracking."""
        
        # Check if already downloaded
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT local_path FROM image_downloads 
            WHERE sample_id = ? AND download_status = 'success'
        ''', (sample_id,))
        
        result = cursor.fetchone()
        if result and Path(result[0]).exists():
            conn.close()
            with self.lock:
                self.stats['cached'] += 1
            return result[0], 'cached'
        
        conn.close()
        
        # Prepare for download
        url_hash = self.get_url_hash(url)
        filename = self.get_safe_filename(url, sample_id)
        local_path = self.base_dir / dataset_type / filename
        
        # Download with retries
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(min(2 ** attempt, 10))
                
                start_time = time.time()
                
                # Create request with headers
                req = Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                with urlopen(req, timeout=timeout) as response:
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
                        raise ValueError(f"Invalid content type: {content_type}")
                    
                    # Download file
                    with open(local_path, 'wb') as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                
                download_time = time.time() - start_time
                file_size = local_path.stat().st_size
                
                if file_size == 0:
                    raise ValueError("Downloaded file is empty")
                
                # Record successful download
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO image_downloads 
                    (sample_id, dataset_type, image_url, url_hash, local_path, 
                     file_size, download_status, download_time)
                    VALUES (?, ?, ?, ?, ?, ?, 'success', ?)
                ''', (sample_id, dataset_type, url, url_hash, str(local_path), 
                      file_size, download_time))
                
                conn.commit()
                conn.close()
                
                with self.lock:
                    self.stats['downloaded'] += 1
                
                return str(local_path), 'success'
                
            except (URLError, HTTPError, ValueError) as e:
                error_msg = str(e)
                if attempt == max_retries - 1:
                    # Record failed download
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO image_downloads 
                        (sample_id, dataset_type, image_url, url_hash, download_status, error_message)
                        VALUES (?, ?, ?, ?, 'failed', ?)
                    ''', (sample_id, dataset_type, url, url_hash, error_msg))
                    
                    conn.commit()
                    conn.close()
                    
                    with self.lock:
                        self.stats['failed'] += 1
                    
                    return None, f'failed: {error_msg}'
        
        return None, 'max_retries_exceeded'
    
    def load_csv_data(self, csv_file):
        """Load CSV data using standard library."""
        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('sample_id') and row.get('image_link'):
                    data.append({
                        'sample_id': row['sample_id'],
                        'image_link': row['image_link']
                    })
        return data
    
    def download_dataset_images(self, csv_file, dataset_type, max_workers=10, max_samples=None):
        """Download all images for a dataset."""
        print(f"\n📥 Loading dataset from {csv_file}...")
        data = self.load_csv_data(csv_file)
        
        if max_samples:
            data = data[:max_samples]
            print(f"📊 Limited to {len(data)} samples for testing")
        else:
            print(f"📊 Loaded {len(data)} samples for {dataset_type} dataset")
        
        print(f"🎯 Starting download using {max_workers} workers...")
        
        results = {}
        completed = 0
        
        def download_and_track(item):
            sample_id = item['sample_id']
            image_url = item['image_link']
            
            local_path, status = self.download_single_image(sample_id, image_url, dataset_type)
            
            nonlocal completed
            completed += 1
            
            if completed % 100 == 0:
                success_rate = (self.stats['downloaded'] + self.stats['cached']) / completed * 100
                print(f"Progress: {completed}/{len(data)} - Success: {success_rate:.1f}% "
                      f"(Downloaded: {self.stats['downloaded']}, Cached: {self.stats['cached']}, "
                      f"Failed: {self.stats['failed']})")
            
            return sample_id, {'local_path': local_path, 'status': status}
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_and_track, item) for item in data]
            
            for future in futures:
                try:
                    sample_id, result = future.result()
                    results[sample_id] = result
                except Exception as e:
                    print(f"Error processing future: {e}")
        
        return results
    
    def save_sample_mapping(self, dataset_type):
        """Save sample_id to image_path mapping."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT sample_id, local_path 
            FROM image_downloads 
            WHERE dataset_type = ? AND download_status = 'success'
            AND local_path IS NOT NULL
        ''', (dataset_type,))
        
        mapping = {}
        for sample_id, local_path in cursor.fetchall():
            if Path(local_path).exists():
                mapping[sample_id] = local_path
        
        conn.close()
        
        # Save mapping to JSON
        mapping_file = self.base_dir / f"{dataset_type}_image_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"💾 Saved {len(mapping)} image mappings to {mapping_file}")
        return mapping_file
    
    def print_summary(self):
        """Print download summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT dataset_type, download_status, COUNT(*) 
            FROM image_downloads 
            GROUP BY dataset_type, download_status
        ''')
        
        print("\n" + "="*60)
        print("📊 DOWNLOAD SUMMARY")
        print("="*60)
        
        stats_by_dataset = {}
        for dataset_type, status, count in cursor.fetchall():
            if dataset_type not in stats_by_dataset:
                stats_by_dataset[dataset_type] = {}
            stats_by_dataset[dataset_type][status] = count
        
        for dataset_type, dataset_stats in stats_by_dataset.items():
            print(f"\n📁 {dataset_type.upper()} DATASET:")
            total = sum(dataset_stats.values())
            successful = dataset_stats.get('success', 0)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            print(f"  Total samples: {total:,}")
            print(f"  Successful: {successful:,} ({success_rate:.1f}%)")
            print(f"  Failed: {dataset_stats.get('failed', 0):,}")
        
        conn.close()


def main():
    """Main execution function."""
    import sys
    
    # Parse simple command line arguments
    dataset = 'train'  # Default
    max_samples = None
    workers = 10
    
    if len(sys.argv) > 1:
        if 'test' in sys.argv:
            dataset = 'test'
        elif 'both' in sys.argv:
            dataset = 'both'
        
        # Look for max_samples argument
        for i, arg in enumerate(sys.argv):
            if arg.startswith('--max='):
                max_samples = int(arg.split('=')[1])
            elif arg.startswith('--workers='):
                workers = int(arg.split('=')[1])
    
    print("🚀 Basic Image Downloader for ML Product Pricing Challenge")
    print("="*60)
    
    downloader = BasicImageDownloader()
    start_time = time.time()
    
    try:
        if dataset in ['train', 'both']:
            print(f"\n🏋️  DOWNLOADING TRAINING DATASET")
            train_results = downloader.download_dataset_images(
                'dataset/train.csv', 'train', max_workers=workers, max_samples=max_samples
            )
            downloader.save_sample_mapping('train')
        
        if dataset in ['test', 'both']:
            print(f"\n🧪 DOWNLOADING TEST DATASET")
            test_results = downloader.download_dataset_images(
                'dataset/test.csv', 'test', max_workers=workers, max_samples=max_samples
            )
            downloader.save_sample_mapping('test')
        
        total_time = time.time() - start_time
        downloader.print_summary()
        
        print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"📁 Images saved to: {downloader.base_dir}")
        print(f"🗄️  Database saved to: {downloader.db_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        downloader.print_summary()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        downloader.print_summary()


if __name__ == "__main__":
    main()