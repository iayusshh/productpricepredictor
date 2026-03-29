"""
Structured logging manager for ML product pricing system.
Provides JSON-based logging with timestamps and experiment tracking.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class ExperimentMetrics:
    """Structure for experiment metrics logging."""
    experiment_id: str
    timestamp: str
    model_type: str
    cv_folds: int
    seed: int
    smape_mean: float
    smape_std: float
    hyperparameters: Dict[str, Any]
    feature_config: Dict[str, Any]
    training_time: float
    validation_scores: list


class LoggingManager:
    """Manages structured JSON logging with timestamps."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "ml_pricing"):
        """
        Initialize logging manager.
        
        Args:
            log_dir: Directory to save log files
            experiment_name: Name prefix for log files
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logger
        self.logger = self._setup_logger()
        
        # Metrics storage
        self.experiment_metrics = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured JSON logger."""
        logger = logging.getLogger(f"{self.experiment_name}_structured")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": %(message)s}'
        )
        
        # File handler for structured logs
        log_file = self.log_dir / f"{self.experiment_name}_structured.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)
        
        return logger
    
    def log_experiment_start(self, experiment_id: str, config: Dict[str, Any]) -> None:
        """Log experiment start with configuration."""
        message = {
            "event": "experiment_start",
            "experiment_id": experiment_id,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.info(json.dumps(message))
    
    def log_experiment_metrics(self, metrics: ExperimentMetrics) -> None:
        """Log experiment metrics in structured format."""
        message = {
            "event": "experiment_metrics",
            **asdict(metrics)
        }
        self.logger.info(json.dumps(message))
        self.experiment_metrics.append(metrics)
    
    def log_data_processing(self, stage: str, details: Dict[str, Any]) -> None:
        """Log data processing stage with details."""
        message = {
            "event": "data_processing",
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            **details
        }
        self.logger.info(json.dumps(message))
    
    def log_feature_engineering(self, feature_type: str, details: Dict[str, Any]) -> None:
        """Log feature engineering progress."""
        message = {
            "event": "feature_engineering",
            "feature_type": feature_type,
            "timestamp": datetime.now().isoformat(),
            **details
        }
        self.logger.info(json.dumps(message))
    
    def log_model_training(self, model_type: str, details: Dict[str, Any]) -> None:
        """Log model training progress."""
        message = {
            "event": "model_training",
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            **details
        }
        self.logger.info(json.dumps(message))
    
    def log_prediction_generation(self, details: Dict[str, Any]) -> None:
        """Log prediction generation progress."""
        message = {
            "event": "prediction_generation",
            "timestamp": datetime.now().isoformat(),
            **details
        }
        self.logger.info(json.dumps(message))
    
    def log_error(self, error_type: str, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log errors with context."""
        message = {
            "event": "error",
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        if details:
            message.update(details)
        self.logger.error(json.dumps(message))
    
    def log_performance_metrics(self, stage: str, metrics: Dict[str, Any]) -> None:
        """Log performance metrics for different stages."""
        message = {
            "event": "performance_metrics",
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.logger.info(json.dumps(message))
    
    def save_experiment_summary(self) -> str:
        """Save experiment summary to JSON file."""
        summary_file = self.log_dir / f"{self.experiment_name}_experiment_summary.json"
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_experiments": len(self.experiment_metrics),
            "timestamp": datetime.now().isoformat(),
            "experiments": [asdict(exp) for exp in self.experiment_metrics]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_file)
    
    def get_best_experiment(self) -> Optional[ExperimentMetrics]:
        """Get experiment with lowest SMAPE."""
        if not self.experiment_metrics:
            return None
        
        return min(self.experiment_metrics, key=lambda x: x.smape_mean)


class TimingValidator:
    """Validates inference timing for large datasets."""
    
    def __init__(self, logger: LoggingManager):
        self.logger = logger
        self.timing_records = []
    
    def start_timing(self, operation: str) -> str:
        """Start timing an operation."""
        timing_id = f"{operation}_{int(time.time())}"
        start_time = time.time()
        
        self.timing_records.append({
            "timing_id": timing_id,
            "operation": operation,
            "start_time": start_time,
            "end_time": None,
            "duration": None
        })
        
        self.logger.log_performance_metrics("timing_start", {
            "timing_id": timing_id,
            "operation": operation
        })
        
        return timing_id
    
    def end_timing(self, timing_id: str) -> float:
        """End timing and return duration."""
        end_time = time.time()
        
        # Find timing record
        record = None
        for r in self.timing_records:
            if r["timing_id"] == timing_id:
                record = r
                break
        
        if not record:
            raise ValueError(f"Timing ID {timing_id} not found")
        
        duration = end_time - record["start_time"]
        record["end_time"] = end_time
        record["duration"] = duration
        
        self.logger.log_performance_metrics("timing_end", {
            "timing_id": timing_id,
            "operation": record["operation"],
            "duration_seconds": duration
        })
        
        return duration
    
    def validate_inference_timing(self, sample_count: int, duration: float, 
                                max_time_per_sample: float = 0.1) -> bool:
        """
        Validate inference timing meets requirements.
        
        Args:
            sample_count: Number of samples processed
            duration: Time taken in seconds
            max_time_per_sample: Maximum allowed time per sample
            
        Returns:
            True if timing is acceptable
        """
        time_per_sample = duration / sample_count if sample_count > 0 else float('inf')
        is_valid = time_per_sample <= max_time_per_sample
        
        self.logger.log_performance_metrics("inference_timing_validation", {
            "sample_count": sample_count,
            "total_duration": duration,
            "time_per_sample": time_per_sample,
            "max_allowed_per_sample": max_time_per_sample,
            "is_valid": is_valid
        })
        
        if not is_valid:
            self.logger.log_error("timing_violation", 
                                f"Inference too slow: {time_per_sample:.4f}s per sample")
        
        return is_valid
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get summary of all timing records."""
        completed_records = [r for r in self.timing_records if r["duration"] is not None]
        
        if not completed_records:
            return {"total_operations": 0}
        
        total_time = sum(r["duration"] for r in completed_records)
        avg_time = total_time / len(completed_records)
        
        operations = {}
        for record in completed_records:
            op = record["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(record["duration"])
        
        operation_stats = {}
        for op, times in operations.items():
            operation_stats[op] = {
                "count": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }
        
        return {
            "total_operations": len(completed_records),
            "total_time": total_time,
            "average_time": avg_time,
            "operations": operation_stats
        }