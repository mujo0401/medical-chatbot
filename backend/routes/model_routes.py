# routes/model_routes.py

from flask import Blueprint, jsonify
import logging
import os

model_bp = Blueprint("model_bp", __name__)
logger = logging.getLogger(__name__)

# Global trainer variable for lazy loading
_trainer = None

def get_trainer():
    """Get or create trainer instance (lazy loading)"""
    global _trainer
    if _trainer is None:
        from training.medical_trainer import MedicalChatbotTrainer
        _trainer = MedicalChatbotTrainer()
    return _trainer

def get_azure_available():
    """Get AZURE_AVAILABLE flag - FIXED VERSION"""
    
    # Check if Azure ML is disabled via environment variable
    if os.getenv("AZURE_ML_DISABLED", "").lower() in ["true", "1", "yes"]:
        return False
    
    # Check if Azure ML SDK is installed
    try:
        from azure.ai.ml import MLClient
        return True
    except ImportError:
        return False

@model_bp.route("/model/status", methods=["GET"])
def model_status():
    """Get detailed model status"""
    try:
        trainer = get_trainer()
        azure_available = get_azure_available()
        
        # Check OpenAI availability
        openai_available = trainer.openai_client is not None
        
        # Check Azure ML availability and configuration
        azure_status = {
            "available": False,
            "configured": False,
            "compute_targets": [],
            "error": None
        }
        
        if azure_available:
            if trainer.ml_client:
                try:
                    # Test Azure connection and get compute targets
                    compute_targets = []
                    try:
                        for compute in trainer.ml_client.compute.list():
                            if compute.state == "Running" or compute.state == "Idle":
                                compute_targets.append({
                                    "name": compute.name,
                                    "type": compute.type,
                                    "state": compute.state,
                                    "vm_size": getattr(compute, 'size', 'Unknown')
                                })
                    except Exception as e:
                        logger.warning(f"Could not fetch compute targets: {e}")
                        # Provide default compute targets
                        compute_targets = [
                            {"name": "cpu-cluster", "type": "AmlCompute", "state": "Unknown", "vm_size": "Standard_DS3_v2"},
                            {"name": "gpu-cluster", "type": "AmlCompute", "state": "Unknown", "vm_size": "Standard_NC6"}
                        ]
                    
                    azure_status.update({
                        "available": True,
                        "configured": True,
                        "compute_targets": compute_targets
                    })
                except Exception as e:
                    azure_status.update({
                        "available": True,
                        "configured": False,
                        "error": f"Azure configuration error: {str(e)}"
                    })
            else:
                azure_status.update({
                    "available": True,
                    "configured": False,
                    "error": "Azure credentials not configured"
                })
        else:
            azure_status["error"] = "Azure ML SDK not installed"
        
        # Check local model availability
        local_available = trainer.model is not None and trainer.tokenizer is not None
        
        return jsonify({
            "openai": {
                "available": openai_available,
                "model": "GPT-4 Turbo",
                "configured": openai_available
            },
            "azure": azure_status,
            "local": {
                "available": local_available,
                "model_name": "DialoGPT-medium",
                "trained": local_available
            },
            "current_preference": trainer.model_preference
        })
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route("/azure/jobs", methods=["GET"])
def get_azure_jobs():
    """Get recent Azure ML jobs"""
    trainer = get_trainer()
    if not trainer.ml_client:
        return jsonify({"error": "Azure not configured"}), 400
    
    try:
        jobs = []
        for job in trainer.ml_client.jobs.list(max_results=10):
            jobs.append({
                "name": job.name,
                "status": job.status,
                "creation_time": job.creation_context.created_at.isoformat() if job.creation_context.created_at else None,
                "experiment": getattr(job, 'experiment_name', 'unknown')
            })
        
        return jsonify({"jobs": jobs})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch Azure jobs: {str(e)}"}), 500

@model_bp.route("/azure/jobs/<job_name>/status", methods=["GET"])
def get_azure_job_status(job_name):
    """Get status of a specific Azure ML job"""
    trainer = get_trainer()
    if not trainer.ml_client:
        return jsonify({"error": "Azure not configured"}), 400
    
    try:
        job = trainer.ml_client.jobs.get(job_name)
        
        return jsonify({
            "name": job.name,
            "status": job.status,
            "creation_time": job.creation_context.created_at.isoformat() if job.creation_context.created_at else None,
            "start_time": getattr(job, 'start_time', None),
            "end_time": getattr(job, 'end_time', None),
            "experiment": getattr(job, 'experiment_name', 'unknown'),
            "properties": getattr(job, 'properties', {})
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get job status: {str(e)}"}), 500