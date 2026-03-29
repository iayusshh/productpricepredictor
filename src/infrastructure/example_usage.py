"""
Example usage of infrastructure components for ML product pricing system.
Demonstrates logging, resource management, and caching capabilities.
"""

import numpy as np
import time
from pathlib import Path

from .logging_manager import LoggingManager, ExperimentMetrics, TimingValidator
from .resource_manager import ResourceManager, ChecksumValidator
from .cache_manager import EmbeddingCache, ImageCache, ModelCheckpointManager, ArtifactManager


def example_logging_usage():
    """Example of structured logging usage."""
    print("=== Logging Manager Example ===")
    
    # Initialize logging manager
    logger = LoggingManager(log_dir="logs", experiment_name="example_experiment")
    
    # Log experiment start
    config = {
        "model_type": "xgboost",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100
    }
    logger.log_experiment_start("exp_001", config)
    
    # Log data processing
    logger.log_data_processing("data_loading", {
        "samples_loaded": 75000,
        "features_extracted": 512,
        "processing_time": 45.2
    })
    
    # Log feature engineering
    logger.log_feature_engineering("text_embeddings", {
        "model_used": "bert-base-uncased",
        "embedding_dim": 768,
        "samples_processed": 75000
    })
    
    # Log model training
    logger.log_model_training("xgboost", {
        "training_samples": 60000,
        "validation_samples": 15000,
        "training_time": 120.5,
        "best_smape": 0.234
    })
    
    # Log experiment metrics
    metrics = ExperimentMetrics(
        experiment_id="exp_001",
        timestamp="2025-01-11T10:30:00",
        model_type="xgboost",
        cv_folds=5,
        seed=42,
        smape_mean=0.234,
        smape_std=0.012,
        hyperparameters=config,
        feature_config={"text_features": True, "image_features": True},
        training_time=120.5,
        validation_scores=[0.231, 0.238, 0.229, 0.241, 0.236]
    )
    logger.log_experiment_metrics(metrics)
    
    # Save experiment summary
    summary_file = logger.save_experiment_summary()
    print(f"Experiment summary saved to: {summary_file}")
    
    # Get best experiment
    best_exp = logger.get_best_experiment()
    if best_exp:
        print(f"Best experiment: {best_exp.experiment_id} with SMAPE: {best_exp.smape_mean}")


def example_timing_validation():
    """Example of timing validation usage."""
    print("\n=== Timing Validation Example ===")
    
    logger = LoggingManager()
    timing_validator = TimingValidator(logger)
    
    # Start timing an operation
    timing_id = timing_validator.start_timing("model_inference")
    
    # Simulate some work
    time.sleep(0.1)
    
    # End timing
    duration = timing_validator.end_timing(timing_id)
    print(f"Operation took: {duration:.3f} seconds")
    
    # Validate inference timing for 75k samples
    sample_count = 75000
    simulated_duration = 750.0  # 10 seconds per 1000 samples
    is_valid = timing_validator.validate_inference_timing(
        sample_count, simulated_duration, max_time_per_sample=0.02
    )
    print(f"Inference timing valid: {is_valid}")
    
    # Get timing summary
    summary = timing_validator.get_timing_summary()
    print(f"Timing summary: {summary}")


def example_resource_management():
    """Example of resource management usage."""
    print("\n=== Resource Management Example ===")
    
    logger = LoggingManager()
    resource_manager = ResourceManager(logger)
    
    # Get resource summary
    summary = resource_manager.get_resource_summary()
    print(f"CPU usage: {summary['cpu']['percent']:.1f}%")
    print(f"Memory usage: {summary['memory']['used_gb']:.1f}GB / {summary['memory']['total_gb']:.1f}GB")
    print(f"Disk usage: {summary['disk']['percent']:.1f}%")
    
    if summary['gpu']:
        print(f"GPU available: {summary['gpu']['available']}")
        print(f"GPU count: {summary['gpu']['count']}")
    
    # Check GPU requirements
    gpu_ok = resource_manager.check_gpu_requirements(required_memory_gb=16.0)
    print(f"GPU requirements met: {gpu_ok}")
    
    # Monitor memory usage
    memory_ok = resource_manager.monitor_memory_usage(threshold_percent=90.0)
    print(f"Memory usage acceptable: {memory_ok}")
    
    # Calculate storage requirements
    storage_reqs = resource_manager.calculate_storage_requirements()
    total_storage = sum(req.size_gb for req in storage_reqs)
    print(f"Total storage required: {total_storage:.1f}GB")
    
    # Validate storage space
    storage_ok = resource_manager.validate_storage_space(safety_margin_gb=5.0)
    print(f"Storage space sufficient: {storage_ok}")


def example_checksum_validation():
    """Example of checksum validation usage."""
    print("\n=== Checksum Validation Example ===")
    
    logger = LoggingManager()
    checksum_validator = ChecksumValidator(logger)
    
    # Create a test file
    test_file = Path("test_checksum_file.txt")
    test_file.write_text("This is a test file for checksum validation.")
    
    try:
        # Calculate checksum
        checksum = checksum_validator.calculate_file_checksum(str(test_file))
        print(f"File checksum: {checksum}")
        
        # Validate checksum
        is_valid = checksum_validator.validate_file_checksum(str(test_file), checksum)
        print(f"Checksum validation: {is_valid}")
        
        # Create manifest for current directory
        manifest = checksum_validator.create_checksum_manifest(".", "test_manifest.json")
        print(f"Created manifest with {len(manifest)} files")
        
        # Validate directory checksums
        if Path("test_manifest.json").exists():
            validation_results = checksum_validator.validate_directory_checksums(".", "test_manifest.json")
            valid_count = sum(1 for v in validation_results.values() if v)
            print(f"Directory validation: {valid_count}/{len(validation_results)} files valid")
    
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        if Path("test_manifest.json").exists():
            Path("test_manifest.json").unlink()


def example_embedding_cache():
    """Example of embedding cache usage."""
    print("\n=== Embedding Cache Example ===")
    
    logger = LoggingManager()
    embedding_cache = EmbeddingCache(cache_dir="embeddings", logger=logger)
    
    # Create sample embeddings
    sample_embeddings = np.random.rand(1000, 768)  # 1000 samples, 768 dimensions
    
    # Save embeddings with metadata
    version = "bert-base-v1.0"
    embedding_file = embedding_cache.save_embeddings(
        embeddings=sample_embeddings,
        version=version,
        model_name="bert-base-uncased",
        model_checkpoint="pytorch_model.bin",
        preprocessing_steps=["tokenization", "padding", "attention_mask"]
    )
    print(f"Embeddings saved to: {embedding_file}")
    
    # List available versions
    versions = embedding_cache.list_versions()
    print(f"Available versions: {versions}")
    
    # Get metadata
    metadata = embedding_cache.get_metadata(version)
    if metadata:
        print(f"Embedding metadata: {metadata.feature_dim} dims, {metadata.sample_count} samples")
    
    # Load embeddings
    loaded_embeddings = embedding_cache.load_embeddings(version, validate_checksum=True)
    if loaded_embeddings is not None:
        print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
        print(f"Embeddings match: {np.array_equal(sample_embeddings, loaded_embeddings)}")
    
    # Cleanup old versions (keep latest 2)
    deleted_versions = embedding_cache.cleanup_old_versions(keep_latest=2)
    print(f"Deleted old versions: {deleted_versions}")


def example_image_cache():
    """Example of image cache usage."""
    print("\n=== Image Cache Example ===")
    
    logger = LoggingManager()
    image_cache = ImageCache(cache_dir="images", logger=logger)
    
    # Add sample image entries
    image_cache.add_image(
        sample_id="sample_001",
        image_url="https://example.com/image1.jpg",
        local_path="images/sample_001.jpg",
        download_status="success"
    )
    
    image_cache.add_image(
        sample_id="sample_002",
        image_url="https://example.com/image2.jpg",
        local_path="images/sample_002.jpg",
        download_status="failed",
        error_message="HTTP 404 Not Found"
    )
    
    # Check if images are cached
    is_cached_1 = image_cache.is_image_cached("sample_001")
    is_cached_2 = image_cache.is_image_cached("sample_002")
    print(f"Sample 001 cached: {is_cached_1}")
    print(f"Sample 002 cached: {is_cached_2}")
    
    # Get cache statistics
    stats = image_cache.get_cache_stats()
    print(f"Cache stats: {stats['successful_downloads']}/{stats['total_images']} successful")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Total size: {stats['total_size_mb']:.1f}MB")
    
    # Validate cache integrity
    validation_results = image_cache.validate_cache_integrity()
    valid_count = sum(1 for v in validation_results.values() if v)
    print(f"Cache validation: {valid_count}/{len(validation_results)} files valid")


def example_model_checkpoint_manager():
    """Example of model checkpoint management."""
    print("\n=== Model Checkpoint Manager Example ===")
    
    logger = LoggingManager()
    checkpoint_manager = ModelCheckpointManager(checkpoint_dir="models", logger=logger)
    
    # Create a dummy model (in practice, this would be your trained model)
    class DummyModel:
        def __init__(self):
            self.weights = np.random.rand(100, 50)
    
    model = DummyModel()
    
    # Save checkpoint
    checkpoint_id = "xgboost_v1_20250111"
    model_config = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
    training_config = {"cv_folds": 5, "seed": 42, "validation_split": 0.2}
    performance_metrics = {"smape_mean": 0.234, "smape_std": 0.012, "r2_score": 0.78}
    
    checkpoint_file = checkpoint_manager.save_checkpoint(
        model=model,
        checkpoint_id=checkpoint_id,
        model_type="sklearn",
        model_config=model_config,
        training_config=training_config,
        performance_metrics=performance_metrics
    )
    print(f"Checkpoint saved to: {checkpoint_file}")
    
    # List checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"Available checkpoints: {checkpoints}")
    
    # Load checkpoint
    loaded_data = checkpoint_manager.load_checkpoint(checkpoint_id, validate_checksum=True)
    if loaded_data:
        print(f"Loaded checkpoint with model type: {loaded_data['checkpoint_metadata'].model_type}")
        print(f"Performance: SMAPE = {loaded_data['checkpoint_metadata'].performance_metrics['smape_mean']}")
    
    # Get best checkpoint
    best_checkpoint = checkpoint_manager.get_best_checkpoint(metric="smape_mean", minimize=True)
    print(f"Best checkpoint: {best_checkpoint}")
    
    # Cleanup old checkpoints (keep best 3)
    deleted_checkpoints = checkpoint_manager.cleanup_old_checkpoints(keep_best=3)
    print(f"Deleted old checkpoints: {deleted_checkpoints}")


def example_artifact_manager():
    """Example of unified artifact management."""
    print("\n=== Artifact Manager Example ===")
    
    logger = LoggingManager()
    artifact_manager = ArtifactManager(base_dir=".", logger=logger)
    
    # Get storage summary
    storage_summary = artifact_manager.get_storage_summary()
    print(f"Total storage used: {storage_summary['total_size_gb']:.2f}GB")
    
    for component, stats in storage_summary["components"].items():
        if "total_size_mb" in stats:
            print(f"  {component}: {stats['total_size_mb']:.1f}MB")
    
    # Validate all artifacts
    validation_results = artifact_manager.validate_all_artifacts()
    print(f"Overall validation status: {validation_results['overall_status']}")
    print(f"Validation success rate: {validation_results['summary']['validation_success_rate']:.1%}")
    
    # Optimize storage
    optimization_results = artifact_manager.optimize_storage(target_size_gb=10.0)
    print(f"Storage optimization:")
    print(f"  Initial size: {optimization_results['initial_size_gb']:.2f}GB")
    print(f"  Final size: {optimization_results['final_size_gb']:.2f}GB")
    print(f"  Reduction: {optimization_results['size_reduction_gb']:.2f}GB ({optimization_results['size_reduction_percent']:.1f}%)")
    print(f"  Actions taken: {len(optimization_results['actions_taken'])}")


def main():
    """Run all infrastructure examples."""
    print("Infrastructure Components Example Usage")
    print("=" * 50)
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("embeddings").mkdir(exist_ok=True)
    Path("images").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    try:
        example_logging_usage()
        example_timing_validation()
        example_resource_management()
        example_checksum_validation()
        example_embedding_cache()
        example_image_cache()
        example_model_checkpoint_manager()
        example_artifact_manager()
        
        print("\n" + "=" * 50)
        print("All infrastructure examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        raise


if __name__ == "__main__":
    main()