# training_status_tracker.py

"""Training status tracker for monitoring training progress."""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from utils.db_utils import update_training_history_status

logger = logging.getLogger(__name__)


class TrainingStatusTracker:
    """Tracks and manages training status across different training methods."""
    
    def __init__(self):
        self.training_status = {
            "is_training": False,
            "progress": 0,
            "status_message": "Ready",
            "current_document": None,
            "training_id": None,
            "start_time": None,
            "estimated_completion": None,
            # Add fields for real metrics
            "current_loss": None,
            "current_accuracy": None,
            "learning_rate": None,
            "cpu_usage": None,
            "memory_usage": None,
            "metrics_updated_at": None,
            "metrics_history": {
                "timestamps": [],
                "loss": [],
                "accuracy": [],
                "learning_rate": [],
                "gpu_usage": [],
                "memory_usage": []
            }
        }
        self._lock = threading.Lock()
        self._metrics_history_limit = 100  # Maximum number of data points to store
    
    def update_status(
        self,
        is_training: bool,
        progress: int,
        message: str,
        current_doc: str = None,
        training_id: str = None,
        metrics: Dict[str, Any] = None
    ):
        """Update training status with thread safety."""
        with self._lock:
            self.training_status.update({
                "is_training": is_training,
                "progress": progress,
                "status_message": message,
                "current_document": current_doc or self.training_status.get("current_document"),
                "training_id": training_id or self.training_status.get("training_id"),
            })
            
            # Update metrics if provided
            if metrics:
                self.update_metrics(metrics)
                
            if is_training and not self.training_status.get("start_time"):
                self.training_status["start_time"] = datetime.now()
                
                # Reset metrics history on new training
                self.training_status["metrics_history"] = {
                    "timestamps": [],
                    "loss": [],
                    "accuracy": [],
                    "learning_rate": [],
                    "gpu_usage": [],
                    "memory_usage": []
                }
            elif not is_training:
                self.training_status["start_time"] = None
                self.training_status["estimated_completion"] = None
        
        logger.info(f"Training status: {progress}% - {message}")
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics with thread safety."""
        with self._lock:
            # Update current metrics
            if "loss" in metrics:
                self.training_status["current_loss"] = metrics["loss"]
            if "accuracy" in metrics:
                self.training_status["current_accuracy"] = metrics["accuracy"]
            if "learning_rate" in metrics:
                self.training_status["learning_rate"] = metrics["learning_rate"]
            if "gpu_usage" in metrics:
                self.training_status["gpu_usage"] = metrics["gpu_usage"]
            if "memory_usage" in metrics:
                self.training_status["memory_usage"] = metrics["memory_usage"]
            
            # Record timestamp of metrics update
            current_time = datetime.now().timestamp()
            self.training_status["metrics_updated_at"] = current_time
            
            # Add to metrics history
            history = self.training_status["metrics_history"]
            history["timestamps"].append(current_time)
            history["loss"].append(metrics.get("loss", None))
            history["accuracy"].append(metrics.get("accuracy", None))
            history["learning_rate"].append(metrics.get("learning_rate", None))
            history["gpu_usage"].append(metrics.get("gpu_usage", None))
            history["memory_usage"].append(metrics.get("memory_usage", None))
            
            # Limit history size
            if len(history["timestamps"]) > self._metrics_history_limit:
                for key in history:
                    history[key] = history[key][-self._metrics_history_limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        with self._lock:
            return self.training_status.copy()
    
    def get_metrics_history(self) -> Dict[str, List]:
        """Get historical training metrics."""
        with self._lock:
            return self.training_status.get("metrics_history", {}).copy()
    
    def start_training(self, training_id: str, document_name: str = None):
        """Mark training as started."""
        self.update_status(
            is_training=True,
            progress=0,
            message="Training started",
            current_doc=document_name,
            training_id=training_id
        )
    
    def complete_training(self, training_id: str, message: str = "Training completed"):
        """Mark training as completed."""
        self.update_status(
            is_training=False,
            progress=100,
            message=message,
            training_id=training_id
        )
        
        try:
            update_training_history_status(training_id, "completed")
        except Exception as e:
            logger.error(f"Failed to update training history: {e}")
    
    def fail_training(self, training_id: str, error_message: str):
        """Mark training as failed."""
        self.update_status(
            is_training=False,
            progress=0,
            message=f"Training failed: {error_message}",
            training_id=training_id
        )
        
        try:
            update_training_history_status(training_id, "failed", error_message)
        except Exception as e:
            logger.error(f"Failed to update training history: {e}")
    
    def estimate_completion_time(self, current_progress: int, total_steps: int = 100):
        """Estimate completion time based on current progress."""
        if not self.training_status.get("start_time") or current_progress <= 0:
            return None
        
        elapsed_time = datetime.now() - self.training_status["start_time"]
        remaining_progress = total_steps - current_progress
        
        if remaining_progress <= 0:
            return datetime.now()
        
        time_per_step = elapsed_time.total_seconds() / current_progress
        estimated_remaining_seconds = remaining_progress * time_per_step
        estimated_completion = datetime.now().timestamp() + estimated_remaining_seconds
        
        with self._lock:
            self.training_status["estimated_completion"] = estimated_completion
        
        return estimated_completion
    
    def get_progress_percentage(self) -> int:
        """Get current progress as percentage."""
        return self.training_status.get("progress", 0)
    
    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        return self.training_status.get("is_training", False)
    
    def get_latest_training_name(self) -> Optional[str]:
        """
        Return the name of the current document being trained on,
        or the training_id if no document name is set.
        """
        with self._lock:
            name = self.training_status.get("current_document")
            if name:
                return name
            return self.training_status.get("training_id")
        
    def get_latest_training_id(self) -> Optional[str]:
        """
        Return the UUID of the most‐recent training run.
        """
        with self._lock:
            # We assume `self.training_status["training_id"]` was set when training started.
            return self.training_status.get("training_id")



class AzureJobMonitor:
    """Monitors Azure ML job progress with real-time metrics."""
    
    def __init__(self, azure_ml_client, status_tracker: TrainingStatusTracker):
        self.azure_ml_client = azure_ml_client
        self.status_tracker = status_tracker
        self._monitoring_threads = {}
        self._stop_events = {}
        self._metrics_history = {}
    
    def start_monitoring(self, job_name: str, training_id: str, completion_callback=None):
        """
        Start monitoring an Azure ML job.
        
        Args:
            job_name: The Azure ML job name to monitor
            training_id: The local training ID to associate with this job
            completion_callback: Optional callback function called when job completes
        """
        if job_name in self._monitoring_threads:
            logger.warning(f"Already monitoring job {job_name}")
            return
        
        # Create a stop event for this monitor thread
        stop_event = threading.Event()
        self._stop_events[job_name] = stop_event
        
        # Initialize metrics history for this job
        self._metrics_history[job_name] = {
            "timestamps": [],
            "loss": [],
            "accuracy": [],
            "learning_rate": [],
            "cpu_usage": [],
            "memory_usage": [],
        }
        
        monitor_thread = threading.Thread(
            target=self._monitor_job,
            args=(job_name, training_id, stop_event, completion_callback),
            daemon=True,
            name=f"azure_monitor_{job_name}"
        )
        
        self._monitoring_threads[job_name] = monitor_thread
        monitor_thread.start()
        logger.info(f"Started monitoring Azure job {job_name}")
    
    def stop_monitoring(self, job_name: str):
        """Stop monitoring a specific job."""
        if job_name in self._stop_events:
            # Signal the thread to stop
            self._stop_events[job_name].set()
            logger.info(f"Signaled monitor thread for job {job_name} to stop")
            
        if job_name in self._monitoring_threads:
            # Remove from tracking
            del self._monitoring_threads[job_name]
            logger.info(f"Stopped monitoring Azure job {job_name}")
            
        # Clean up the stop event
        if job_name in self._stop_events:
            del self._stop_events[job_name]
            
        # Keep metrics history for later retrieval
    
    def get_job_metrics(self, job_name: str) -> dict:
        """Get collected metrics for a specific job."""
        return self._metrics_history.get(job_name, {})
    
    def _update_job_metrics(self, job_name: str, job_status: dict, training_id: str):
        """Extract and store real metrics from job status."""
        try:
            # Get current timestamp
            current_time = datetime.now().timestamp()
            
            # Extract metrics from job status if available
            metrics = job_status.get("metrics", {})
            
            # Get actual metrics or use reasonable defaults (not random)
            loss = metrics.get("loss", self._get_last_value(job_name, "loss", 2.0))
            accuracy = metrics.get("accuracy", self._get_last_value(job_name, "accuracy", 0.5))
            learning_rate = metrics.get("learning_rate", self._get_last_value(job_name, "learning_rate", 0.001))
            
            # Try to get resource usage from job status
            cpu_usage = metrics.get("cpu_usage", self._get_last_value(job_name, "cpu_usage", 50.0))
            memory_usage = metrics.get("memory_usage", self._get_last_value(job_name, "memory_usage", 40.0))
            
            # Update metrics history
            if job_name in self._metrics_history:
                self._metrics_history[job_name]["timestamps"].append(current_time)
                self._metrics_history[job_name]["loss"].append(loss)
                self._metrics_history[job_name]["accuracy"].append(accuracy)
                self._metrics_history[job_name]["learning_rate"].append(learning_rate)
                self._metrics_history[job_name]["cpu_usage"].append(cpu_usage)
                self._metrics_history[job_name]["memory_usage"].append(memory_usage)
                
                # Keep only last 100 data points
                for key in self._metrics_history[job_name]:
                    if len(self._metrics_history[job_name][key]) > 100:
                        self._metrics_history[job_name][key] = self._metrics_history[job_name][key][-100:]
            
            # Update training status with real metrics
            metrics_update = {
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": learning_rate,
                "gpu_usage": cpu_usage,  # Use CPU usage as GPU proxy
                "memory_usage": memory_usage
            }
            self.status_tracker.update_metrics(metrics_update)
                
        except Exception as e:
            logger.warning(f"Error updating metrics for job {job_name}: {e}")
    
    def _get_last_value(self, job_name: str, metric_name: str, default_value: float) -> float:
        """Get the last recorded value for a metric or return default."""
        if job_name in self._metrics_history and metric_name in self._metrics_history[job_name]:
            values = self._metrics_history[job_name][metric_name]
            if values:
                return values[-1]
        return default_value
    
    def _get_job_phase(self, status: str, progress_metrics: dict) -> tuple:
        """
        Determine the current job phase and corresponding progress percentage.
        Returns (phase_name, progress_percentage)
        """
        if status in ["NotStarted", "Queued"]:
            return "queued", 10
        elif status == "Starting":
            return "starting", 20
        elif status == "Preparing":
            return "preparing", 30
        elif status == "Running":
            # Try to extract progress from metrics
            if "progress" in progress_metrics:
                # Scale progress from 0-100 to 40-90 range
                scaled_progress = 40 + (progress_metrics["progress"] * 0.5)
                return "training", min(90, int(scaled_progress))
            else:
                # Default progress for running state
                return "training", 65
        elif status == "Completed":
            return "completed", 100
        elif status == "Failed":
            return "failed", 0
        elif status == "Canceled":
            return "canceled", 0
        else:
            return "unknown", 50
    
    def _monitor_job(self, job_name: str, training_id: str, stop_event, completion_callback=None):
        """Monitor Azure ML job progress with real metrics collection."""
        try:
            # Start with a shorter poll interval and increase it for longer-running jobs
            poll_interval = 15  # seconds
            min_poll_interval = 15
            max_poll_interval = 60
            poll_count = 0
            last_status = None
            
            # Continue monitoring until explicitly stopped or job completes/fails
            while not stop_event.is_set() and self.status_tracker.is_training_active():
                try:
                    job_status = self.azure_ml_client.get_job_status(job_name)
                    
                    if not job_status:
                        logger.warning(f"Could not get status for job {job_name}")
                        time.sleep(poll_interval)
                        continue
                    
                    status = job_status.get("status", "Unknown")
                    
                    # Only log when status changes
                    if status != last_status:
                        logger.info(f"Azure job {job_name} status: {status}")
                        last_status = status
                    
                    # Extract real metrics from the job status
                    self._update_job_metrics(job_name, job_status, training_id)
                    
                    # Get job phase and corresponding progress
                    phase, progress = self._get_job_phase(status, job_status.get("metrics", {}))
                    
                    if status == "Running":
                        self.status_tracker.update_status(
                            True, progress, f"Azure job training: {phase}", training_id=training_id
                        )
                    elif status == "Completed":
                        self.status_tracker.complete_training(training_id, "Azure training completed successfully")
                        
                        # Call completion callback if provided
                        if completion_callback:
                            try:
                                completion_callback(job_name, training_id, status)
                            except Exception as cb_error:
                                logger.error(f"Error in completion callback: {cb_error}")
                        
                        break
                    elif status in ["Failed", "Canceled"]:
                        # Get detailed error information
                        error_details = job_status.get("error_details", "No details available")
                        error_msg = f"Azure job {status.lower()}: {error_details}"
                        self.status_tracker.fail_training(training_id, error_msg)
                        
                        # Call completion callback with failure status
                        if completion_callback:
                            try:
                                completion_callback(job_name, training_id, status)
                            except Exception as cb_error:
                                logger.error(f"Error in completion callback: {cb_error}")
                        
                        break
                    elif status in ["Queued", "Starting", "Preparing"]:
                        self.status_tracker.update_status(
                            True, progress, f"Azure job {phase}: {job_name}", training_id=training_id
                        )
                    
                    # Adjust poll interval based on job state
                    poll_count += 1
                    if poll_count > 10:
                        # Gradually increase poll interval for long-running jobs
                        poll_interval = min(max_poll_interval, poll_interval + 5)
                    
                    # Wait for next poll or until stopped
                    stop_event.wait(poll_interval)
                    
                except Exception as poll_error:
                    logger.error(f"Error polling job status: {poll_error}")
                    # Back off on errors but keep trying
                    stop_event.wait(min(poll_interval * 2, 120))
            
            # Clean up
            if job_name in self._monitoring_threads:
                del self._monitoring_threads[job_name]
                
            logger.info(f"Stopped monitoring thread for job {job_name}")

        except Exception as e:
            logger.error(f"Error monitoring Azure job {job_name}: {e}")
            self.status_tracker.fail_training(training_id, f"Monitoring failed: {str(e)}")
    
    def get_active_jobs(self) -> list:
        """Get list of currently monitored jobs."""
        return list(self._monitoring_threads.keys())


class TrainingProgressCallback:
    """Callback class for training progress updates with metrics support."""
    
    def __init__(self, status_tracker: TrainingStatusTracker, training_id: str):
        self.status_tracker = status_tracker
        self.training_id = training_id
    
    def __call__(
        self,
        is_training: bool,
        progress: int,
        message: str,
        current_doc: str = None,
        metrics: Dict[str, Any] = None
    ):
        """
        Callback function for status and metrics updates.
        
        Args:
            is_training: Whether training is active
            progress: Progress percentage (0-100)
            message: Status message
            current_doc: Current document name (optional)
            metrics: Dictionary of metrics (optional)
        """
        self.status_tracker.update_status(
            is_training=is_training,
            progress=progress,
            message=message,
            current_doc=current_doc,
            training_id=self.training_id,
            metrics=metrics
        )
        
        # Estimate completion time if training is active
        if is_training and progress > 0:
            self.status_tracker.estimate_completion_time(progress)