# training_status_tracker.py

"""Training status tracker for monitoring training progress."""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
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
        }
        self._lock = threading.Lock()
    
    def update_status(
        self,
        is_training: bool,
        progress: int,
        message: str,
        current_doc: str = None,
        training_id: str = None
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
            
            if is_training and not self.training_status.get("start_time"):
                self.training_status["start_time"] = datetime.now()
            elif not is_training:
                self.training_status["start_time"] = None
                self.training_status["estimated_completion"] = None
        
        logger.info(f"Training status: {progress}% - {message}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        with self._lock:
            return self.training_status.copy()
    
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
    """Monitors Azure ML job progress."""
    
    def __init__(self, azure_ml_client, status_tracker: TrainingStatusTracker):
        self.azure_ml_client = azure_ml_client
        self.status_tracker = status_tracker
        self._monitoring_threads = {}
    
    def start_monitoring(self, job_name: str, training_id: str):
        """Start monitoring an Azure ML job."""
        if job_name in self._monitoring_threads:
            logger.warning(f"Already monitoring job {job_name}")
            return
        
        monitor_thread = threading.Thread(
            target=self._monitor_job,
            args=(job_name, training_id),
            daemon=True,
            name=f"azure_monitor_{job_name}"
        )
        
        self._monitoring_threads[job_name] = monitor_thread
        monitor_thread.start()
        logger.info(f"Started monitoring Azure job {job_name}")
    
    def stop_monitoring(self, job_name: str):
        """Stop monitoring a specific job."""
        if job_name in self._monitoring_threads:
            # Note: We can't directly stop a thread, but we can remove it from tracking
            del self._monitoring_threads[job_name]
            logger.info(f"Stopped monitoring Azure job {job_name}")
    
    def _monitor_job(self, job_name: str, training_id: str):
        """Monitor Azure ML job progress."""
        try:
            poll_interval = 30  # seconds
            
            while self.status_tracker.is_training_active():
                try:
                    job_status = self.azure_ml_client.get_job_status(job_name)
                    
                    if not job_status:
                        logger.warning(f"Could not get status for job {job_name}")
                        time.sleep(poll_interval)
                        continue
                    
                    status = job_status.get("status", "Unknown")
                    
                    if status == "Running":
                        self.status_tracker.update_status(
                            True, 80, f"Azure job running: {job_name}", training_id=training_id
                        )
                    elif status == "Completed":
                        self.status_tracker.complete_training(training_id, "Azure training completed")
                        break
                    elif status in ["Failed", "Canceled"]:
                        error_msg = f"Azure job {status.lower()}"
                        self.status_tracker.fail_training(training_id, error_msg)
                        break
                    elif status in ["Queued", "Starting", "Preparing"]:
                        self.status_tracker.update_status(
                            True, 65, f"Azure job {status.lower()}: {job_name}", training_id=training_id
                        )
                    
                    time.sleep(poll_interval)
                    
                except Exception as poll_error:
                    logger.error(f"Error polling job status: {poll_error}")
                    time.sleep(poll_interval * 2)  # Back off on errors
            
            # Clean up
            if job_name in self._monitoring_threads:
                del self._monitoring_threads[job_name]

        except Exception as e:
            logger.error(f"Error monitoring Azure job {job_name}: {e}")
            self.status_tracker.fail_training(training_id, f"Monitoring failed: {str(e)}")
    
    def get_active_jobs(self) -> list:
        """Get list of currently monitored jobs."""
        return list(self._monitoring_threads.keys())


class TrainingProgressCallback:
    """Callback class for training progress updates."""
    
    def __init__(self, status_tracker: TrainingStatusTracker, training_id: str):
        self.status_tracker = status_tracker
        self.training_id = training_id
    
    def __call__(
        self,
        is_training: bool,
        progress: int,
        message: str,
        current_doc: str = None
    ):
        """Callback function for status updates."""
        self.status_tracker.update_status(
            is_training=is_training,
            progress=progress,
            message=message,
            current_doc=current_doc,
            training_id=self.training_id
        )
        
        # Estimate completion time if training is active
        if is_training and progress > 0:
            self.status_tracker.estimate_completion_time(progress)
