# routes/train_routes.py

from flask import Blueprint, request, jsonify
from flask_cors import CORS
import uuid
import threading
import logging
import os
import time
from datetime import datetime, timedelta

from utils.db_utils import (
    get_documents_info,
    insert_training_history,
    get_training_history,
    update_training_history_status,
)
from utils.pdf_utils import extract_text_from_pdf, read_text_file
from training.medical_trainer import MedicalChatbotTrainer

def check_azure_availability(trainer):
    """Safely check Azure ML availability with proper exception handling"""
    try:
        if not getattr(trainer, "azure_ml_client", None):
            return False, "Azure ML client not initialized"
        
        # This is where the 'training_id' attribute error happens
        # Let's be more specific about catching it
        try:
            is_available = trainer.azure_ml_client.is_available()
            if not is_available:
                return False, "Azure ML not available"
        except AttributeError as attr_err:
            # This catches the 'training_id' attribute error specifically
            if "training_id" in str(attr_err):
                return False, "Azure ML client configuration error"
            else:
                return False, f"Azure ML attribute error: {str(attr_err)}"
        
        return True, None
    except Exception as e:
        return False, f"Azure ML error: {str(e)}"

train_bp = Blueprint("train_bp", __name__)
# ── Enable CORS on this entire blueprint (allows requests from any origin)
CORS(train_bp)  # ── ADDED to allow browser calls from React at http://localhost:3000

logger = logging.getLogger(__name__)

# Singleton trainer instance
_trainer = None


def get_trainer():
    """Get or create a MedicalChatbotTrainer instance (lazy loading)."""
    global _trainer
    if _trainer is None:
        _trainer = MedicalChatbotTrainer()
    return _trainer


@train_bp.route("/train", methods=["POST"])
def train():
    """
    Start a new training run (local or Azure) on selected document IDs.
    Expects JSON:
      {
        "document_ids": [1, 2, 3],
        "use_azure": false,
        "compute_target": "gpu-cluster"
      }
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    doc_ids = data.get("document_ids", [])
    use_azure = data.get("use_azure", False)
    compute_target = data.get("compute_target", "gpu-cluster")

    if not isinstance(doc_ids, list) or len(doc_ids) == 0:
        return jsonify({"error": "No documents selected"}), 400

    trainer = get_trainer()

    # Prevent starting if training already in progress
    try:
        status = trainer.get_training_status()
        if status.get("is_training"):
            return jsonify({"error": "Training already in progress", "status": status}), 400
    except Exception:
        pass

    # If Azure requested, ensure Azure client is configured with enhanced validation
    if use_azure:
        if not getattr(trainer, "azure_ml_client", None):
            return jsonify({
                "error": "Azure ML client not available",
                "fallback": "Use local training",
                "help": "Check Azure ML configuration and credentials"
            }), 400

        if not trainer.azure_ml_client.is_available():
            return jsonify({
                "error": "Azure ML not properly configured",
                "fallback": "Use local training",
                "help": "Verify Azure credentials and workspace settings"
            }), 400

        # Validate the specific compute target and override if necessary
        try:
            validation = trainer.azure_ml_client.validate_compute_training(doc_ids)
            if not validation.get("valid"):
                return jsonify({
                    "error": f"Azure training validation failed: {validation.get('reason', 'Unknown error')}",
                    "fallback": "Use local training",
                    "recommendation": validation.get("recommendation"),
                    "help_url": validation.get("help_url")
                }), 400

            # ── UPDATED: override compute_target with whatever Azure validation returns
            validated_compute = validation.get("compute_target", compute_target)
            compute_target = validated_compute

        except Exception as e:
            logger.error(f"Azure validation error: {e}", exc_info=True)
            return jsonify({
                "error": f"Azure validation failed: {str(e)}",
                    "fallback": "Use local training"
            }), 400

    # Fetch document info from DB
    docs = get_documents_info(doc_ids)
    if not docs:
        return jsonify({"error": "No valid processed documents found"}), 400

    texts = []
    names = []
    total_size = 0

    for doc in docs:
        name = doc.get("original_name")
        path = doc.get("file_path")
        file_type = doc.get("file_type")
        names.append(name)

        try:
            if file_type == "pdf":
                text = extract_text_from_pdf(path)
            else:
                text = read_text_file(path)
            texts.append(text)
            total_size += len(text)
        except Exception as e:
            logger.error(f"Error reading document {name}: {e}", exc_info=True)
            return jsonify({"error": f"Failed to read document {name}: {str(e)}"}), 400

    combined_name = ", ".join(names)
    platform = "azure" if use_azure else "local"
    compute = compute_target if use_azure else "local"

    # Create a new training_id and insert a "started" record
    training_id = str(uuid.uuid4())

    # Get estimated cost for Azure training
    estimated_cost = None
    if use_azure:
        try:
            validation = trainer.azure_ml_client.validate_compute_training(doc_ids)
            estimated_cost = validation.get("estimated_cost", 0.0)
        except Exception:
            estimated_cost = 0.0

    insert_training_history(
        training_id=training_id,
        doc_name=combined_name,
        doc_type="mixed",
        training_type=platform,
        status="started",
        error_msg=None,                      # matches your original param name
        compute_target=compute,
        estimated_cost=estimated_cost,
        index_path=None,
        metadata_path=None,
        model_path=None,
        total_documents=len(texts),
    )

    def _background():
        start_time = time.time()
        try:
            logger.info(
                f"Starting {platform} training for {len(texts)} documents "
                f"(total size: {total_size:,} chars)"
            )

            # Kick off actual training in MedicalChatbotTrainer
            result_id = trainer.train_on_documents(
                texts, combined_name, use_azure, compute_target
            )

            # On success, calculate duration and cost
            duration_minutes = int((time.time() - start_time) / 60)

            # Calculate actual cost for Azure or estimated cost for local
            if use_azure and estimated_cost:
                actual_cost = estimated_cost * (
                    duration_minutes /
                    (validation.get("estimated_duration_minutes", 60))
                )
            else:
                actual_cost = round(duration_minutes * 0.10, 2)  # $0.10/minute for local

            # Model path for local training
            model_dir = f"models/local_model_{training_id}"

            # Update DB record as "completed"
            update_training_history_status(
                training_id=training_id,
                status="completed",
                error_message=None,
                actual_duration=duration_minutes,
                compute_target=compute if use_azure else None,
                estimated_cost=actual_cost,
                index_path=f"models/indexes/{training_id}/faiss.index",
                metadata_path=f"models/indexes/{training_id}/chunks_metadata.pkl",
                model_path=model_dir,
                total_documents=len(texts),
            )

            logger.info(f"Training {training_id} completed successfully!")
            logger.info(f"Duration: {duration_minutes} minutes")
            logger.info(f"Cost: ${actual_cost}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training {training_id} failed: {error_msg}", exc_info=True)
            # Update DB record as "failed"
            update_training_history_status(
                training_id=training_id,
                status="failed",
                error_message=error_msg,
            )

    # Run in background thread
    thread = threading.Thread(target=_background, daemon=True)
    thread.start()

    return (
        jsonify(
            {
                "message": "Training started successfully",
                "training_id": training_id,
                "documents": names,
                "training_type": platform,
                "status": "started",
                "compute_target": compute,
                "estimated_cost": estimated_cost,
                "total_documents": len(texts),
                "total_size_chars": total_size,
            }
        ),
        202,
    )


@train_bp.route("/azure/validate-training", methods=["POST"])
def validate_azure_training():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}", "can_train": False}), 400

    doc_ids = data.get("document_ids", [])
    if not isinstance(doc_ids, list) or len(doc_ids) == 0:
        return jsonify({"error": "No documents selected", "can_train": False}), 200

    trainer = get_trainer()

    azure_available, azure_error = check_azure_availability(trainer)
    if not azure_available:
        return jsonify({
            "error": azure_error,
            "can_train": False,
            "reason": "Azure ML configuration issues",
            "help": "Check Azure ML configuration and credentials"
        }), 200

    try:
        # (…your total‐size calculation here…)

        validation_result = trainer.azure_ml_client.validate_compute_training(doc_ids)

        # translate to what the front‐end expects:
        response = {
            "can_train": validation_result.get("valid", False),
            "compute_target": validation_result.get("compute_target"),
            "estimated_cost": validation_result.get("estimated_cost"),
            "error": None if validation_result.get("valid") else validation_result.get("reason", "Unknown error")
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Azure validation error: {e}", exc_info=True)
        return jsonify({
            "error": f"Azure validation failed: {str(e)}",
            "can_train": False,
            "reason": "Internal validation error",
            "help": "Check Azure ML workspace connectivity"
        }), 500


@train_bp.route("/azure/billing-info", methods=["GET"])
def get_azure_billing_info():
    trainer = get_trainer()
    
    # FIXED: Safe Azure availability check - KEEP 400 STATUS
    azure_available, azure_error = check_azure_availability(trainer)
    if not azure_available:
        return jsonify({
            "available": False,
            "error": azure_error,
            "help": "Install azure-ai-ml and configure environment variables"
        }), 200

    try:
        logger.info("Fetching real Azure billing information.")
        billing_data = trainer.azure_ml_client.get_billing_information()
        return jsonify(billing_data), 200
        
    except Exception as e:
        logger.error(f"Error getting Azure billing info: {e}", exc_info=True)
        return jsonify({
            "available": False,
            "error": f"Failed to get billing info: {str(e)}",
            "help": "Check Azure permissions and API availability"
        }), 500  

@train_bp.route("/azure/cost-data", methods=["GET"])
def get_azure_cost_data():
    """
    Get detailed Azure cost data for the current month.
    """
    trainer = get_trainer()

    if not getattr(trainer, "azure_ml_client", None) or not trainer.azure_ml_client.is_available():
        return jsonify({
            "error": "Azure ML not available",
            "help": "Configure Azure ML first"
        }), 400

    try:
        logger.info("Fetching Azure cost data.")
        cost_data = trainer.azure_ml_client.get_cost_data()

        if "error" in cost_data:
            return jsonify({
                "error": cost_data["error"],
                "help": "Install azure-mgmt-consumption and grant Cost Management Reader permissions"
            }), 400

        logger.info(f"Retrieved cost data: ${cost_data.get('current_month', {}).get('total_cost', 0):.2f}")
        return jsonify(cost_data), 200

    except Exception as e:
        logger.error(f"Error getting cost data: {e}", exc_info=True)
        return jsonify({
            "error": f"Failed to get cost data: {str(e)}"
        }), 500


@train_bp.route("/training/azure/workspace-info", methods=["GET"])
def get_azure_workspace_info():
    trainer = get_trainer()
    
    # FIXED: Safe Azure availability check - KEEP 200 STATUS but with clear error  
    azure_available, azure_error = check_azure_availability(trainer)
    if not azure_available:
        return jsonify({
            "error": azure_error,
            "available": False,
            "name": "Azure ML workspace",
            "compute_targets": [],
            "status": "unavailable"
        }), 200

    try:
        # Because MLClient was already constructed with subscription, resource_group, workspace_name,
        # we can call get() with no arguments to retrieve that workspace.
        workspace = trainer.azure_ml_client.ml_client.workspaces.get()

        # Collect all compute targets
        compute_targets = []
        for compute in trainer.azure_ml_client.ml_client.compute.list():
            compute_info = {
                "name": compute.name,
                "type": getattr(compute, "type", "Unknown"),
                "state": getattr(compute, "provisioning_state", "Unknown"),
                "vm_size": getattr(compute, "size", "Unknown"),
                "location": getattr(compute, "location", "Unknown"),
            }
            # If scale_settings exist, include node counts
            if hasattr(compute, "scale_settings"):
                scale_settings = compute.scale_settings
                compute_info.update({
                    "current_nodes": getattr(compute, "current_node_count", 0),
                    "max_nodes": getattr(scale_settings, "max_instances", 1),
                    "min_nodes": getattr(scale_settings, "min_instances", 0),
                })
            compute_targets.append(compute_info)

        workspace_info = {
            "name": workspace.name,
            "resource_group": workspace.resource_group,
            "location": workspace.location,
            "subscription_id": (
                workspace.id.split("/")[2] if "/subscriptions/" in workspace.id else None
            ),
            "workspace_id": workspace.id,
            "compute_targets": compute_targets,
            "timestamp": time.time()
        }

        logger.info(f"Retrieved workspace info: {workspace.name} ({len(compute_targets)} compute targets)")
        return jsonify(workspace_info), 200

    except Exception as e:
        logger.error(f"Failed to get workspace info: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get workspace info: {str(e)}"}), 500  # KEEP 500!


@train_bp.route("/training/status", methods=["GET"])
def training_status_route():
    """
    Return current in‐progress training status, if any.
    Enhanced with real-time Azure job monitoring.
    """
    trainer = get_trainer()
    try:
        status = trainer.get_training_status()

        # If Azure training is active, get real Azure job status
        if (status.get("is_training") and
            hasattr(trainer, "azure_monitor") and
            trainer.azure_monitor):

            active_jobs = trainer.azure_monitor.get_active_jobs()
            if active_jobs:
                # Get status of the most recent job
                latest_job = active_jobs[0]
                try:
                    azure_status = trainer.azure_ml_client.get_job_status(latest_job)
                    if azure_status:
                        status.update({
                            "azure_job_name": azure_status["name"],
                            "azure_job_status": azure_status["status"],
                            "azure_compute": azure_status.get("compute_target"),
                            "duration_minutes": azure_status.get("duration_minutes"),
                        })
                except Exception as e:
                    logger.warning(f"Could not get Azure job status: {e}", exc_info=True)

        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error fetching training status: {e}", exc_info=True)
        return jsonify({
            "is_training": False,
            "progress": 0,
            "status_message": f"Error: {str(e)}"
        }), 500


@train_bp.route("/training/history", methods=["GET"])
def training_history_route():
    """
    Fetch the full training history with enhanced metadata.
    """
    try:
        history = get_training_history()

        # Enhance history with additional computed fields
        for item in history:
            # Calculate success rate and other metrics
            if item.get("status") == "completed" and item.get("actual_duration"):
                item["success"] = True
                item["cost_per_minute"] = (
                    item.get("estimated_cost", 0) /
                    max(item.get("actual_duration", 1), 1)
                )
            else:
                item["success"] = False

            # Format timestamps for better display
            if item.get("started_at"):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(item["started_at"])
                    item["started_at_formatted"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

        return jsonify(history), 200
    except Exception as e:
        logger.error(f"Error fetching training history: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch training history: {str(e)}"}), 500


# ══════════════════════════════════════════════════════════════════════════════════
# NEW ROUTES FOR DocumentsTab Integration
# ══════════════════════════════════════════════════════════════════════════════════

@train_bp.route("/training/azure/jobs", methods=["GET"])
def get_azure_jobs():
    """
    Get list of Azure ML jobs with status and details.
    """
    trainer = get_trainer()
    
    if not getattr(trainer, "azure_ml_client", None) or not trainer.azure_ml_client.is_available():
        return jsonify({
            "jobs": [],
            "error": "Azure ML not available"
        }), 200

    try:
        jobs = []
        
        # Get jobs from Azure ML workspace
        for job in trainer.azure_ml_client.ml_client.jobs.list(max_results=50):
            job_info = {
                "name": job.name,
                "display_name": getattr(job, "display_name", job.name),
                "status": job.status,
                "creation_time": job.creation_time.isoformat() if hasattr(job, "creation_time") and job.creation_time else None,
                "start_time": job.start_time.isoformat() if hasattr(job, "start_time") and job.start_time else None,
                "end_time": job.end_time.isoformat() if hasattr(job, "end_time") and job.end_time else None,
                "compute_target": getattr(job, "compute", "Unknown"),
                "experiment_name": getattr(job, "experiment_name", "Unknown"),
            }
            
            # Add tags if they exist
            if hasattr(job, "tags") and job.tags:
                job_info["tags"] = job.tags
            
            jobs.append(job_info)
        
        logger.info(f"Retrieved {len(jobs)} Azure ML jobs")
        return jsonify({"jobs": jobs}), 200
        
    except Exception as e:
        logger.error(f"Error fetching Azure jobs: {e}", exc_info=True)
        return jsonify({
            "jobs": [],
            "error": f"Failed to fetch Azure jobs: {str(e)}"
        }), 200


@train_bp.route("/training/azure/jobs/<job_name>/logs", methods=["GET"])
def get_azure_job_logs(job_name):
    """
    Get logs for a specific Azure ML job.
    """
    trainer = get_trainer()
    
    if not getattr(trainer, "azure_ml_client", None) or not trainer.azure_ml_client.is_available():
        return jsonify({
            "logs": ["Azure ML not available"]
        }), 400

    try:
        logs = trainer.azure_ml_client.get_job_logs(job_name)
        return jsonify({"logs": logs}), 200
        
    except Exception as e:
        logger.error(f"Error fetching job logs for {job_name}: {e}", exc_info=True)
        return jsonify({
            "logs": [f"Failed to fetch logs: {str(e)}"]
        }), 500


@train_bp.route("/training/azure/jobs/<job_name>/cancel", methods=["POST"])
def cancel_azure_job(job_name):
    """
    Cancel a specific Azure ML job.
    """
    trainer = get_trainer()
    
    if not getattr(trainer, "azure_ml_client", None) or not trainer.azure_ml_client.is_available():
        return jsonify({"error": "Azure ML not available"}), 400

    try:
        # Get the job and cancel it
        job = trainer.azure_ml_client.ml_client.jobs.get(job_name)
        if job.status in ["Completed", "Failed", "Canceled"]:
            return jsonify({
                "error": f"Job {job_name} is already {job.status.lower()} and cannot be canceled"
            }), 400
        
        # Cancel the job
        trainer.azure_ml_client.ml_client.jobs.cancel(job_name)
        
        logger.info(f"Canceled Azure ML job: {job_name}")
        return jsonify({
            "message": f"Job {job_name} cancellation requested",
            "status": "canceling"
        }), 200
        
    except Exception as e:
        logger.error(f"Error canceling job {job_name}: {e}", exc_info=True)
        return jsonify({
            "error": f"Failed to cancel job: {str(e)}"
        }), 500


@train_bp.route('/training/metrics', methods=['GET'])
def get_training_metrics():
    try:
        history = get_training_history()
        
        # Handle None values in all sum calculations:
        total_documents = sum((h.get("total_documents") or 0) for h in history)
        total_chunks = sum((h.get("total_chunks") or 0) for h in history)
        total_epochs = sum((h.get("epochs") or 0) for h in history)
        
        # Handle training time calculations safely:
        training_times = [
            h.get("training_time", 0) for h in history 
            if h.get("training_time") is not None
        ]
        avg_training_time = sum(training_times) / len(training_times) if training_times else 0
        
        # Handle loss calculations safely:
        losses = [
            h.get("final_loss", 0) for h in history 
            if h.get("final_loss") is not None and h.get("final_loss") > 0
        ]
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Get latest training info safely:
        latest_training = history[0] if history else {}
        latest_loss = latest_training.get("final_loss") if latest_training.get("final_loss") is not None else 0
        latest_accuracy = latest_training.get("accuracy") if latest_training.get("accuracy") is not None else 0
        
        return jsonify({
            "total_trainings": len(history),
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_epochs": total_epochs,
            "avg_training_time": round(avg_training_time, 2),
            "avg_loss": round(avg_loss, 4),
            "latest_loss": round(latest_loss, 4),
            "latest_accuracy": round(latest_accuracy, 4),
            "success_rate": len([h for h in history if h.get("status") == "completed"]) / len(history) * 100 if history else 0
        })
        
    except Exception as e:
        print(f"Error calculating training metrics: {str(e)}")
        # Return safe defaults if calculation fails:
        return jsonify({
            "total_trainings": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "total_epochs": 0,
            "avg_training_time": 0,
            "avg_loss": 0,
            "latest_loss": 0,
            "latest_accuracy": 0,
            "success_rate": 0
        })


@train_bp.route("/training/analytics", methods=["GET"])
def get_training_analytics():
    """
    Get training analytics data for charts and trends.
    """
    try:
        # Get training history
        history = get_training_history()
        
        # Calculate analytics for last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        daily_sessions = {}
        daily_costs = {}
        accuracy_trend = []
        
        # Initialize daily data
        for i in range(7):
            date = start_date + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            daily_sessions[date_str] = 0
            daily_costs[date_str] = 0.0
        
        # Process history data
        for h in history:
            if h.get("started_at"):
                try:
                    started = datetime.fromisoformat(h["started_at"])
                    if start_date <= started <= end_date:
                        date_str = started.strftime("%Y-%m-%d")
                        if date_str in daily_sessions:
                            daily_sessions[date_str] += 1
                            daily_costs[date_str] += h.get("estimated_cost", 0)
                except Exception:
                    pass
        
        # Generate mock accuracy trend (replace with real data when available)
        for i in range(7):
            date = start_date + timedelta(days=i)
            accuracy_trend.append({
                "date": date.strftime("%Y-%m-%d"),
                "accuracy": 85 + (i * 2) + ((-1) ** i * 3)  # Mock trending accuracy
            })
        
        analytics = {
            "sessions_per_day": [
                {"date": date, "sessions": count}
                for date, count in daily_sessions.items()
            ],
            "cost_per_day": [
                {"date": date, "cost": cost}
                for date, cost in daily_costs.items()
            ],
            "accuracy_trend": accuracy_trend,
            "duration_trend": [
                {"date": date, "avg_duration": 25 + (i * 2)}  # Mock data
                for i, date in enumerate(daily_sessions.keys())
            ],
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "last_updated": datetime.now().isoformat(),
        }
        
        return jsonify(analytics), 200
        
    except Exception as e:
        logger.error(f"Error calculating training analytics: {e}", exc_info=True)
        return jsonify({
            "error": f"Failed to calculate analytics: {str(e)}",
            "sessions_per_day": [],
            "cost_per_day": [],
            "accuracy_trend": [],
            "duration_trend": [],
        }), 200


@train_bp.route("/training/notifications", methods=["GET"])
def get_training_notifications():
    """
    Get recent training notifications and updates.
    """
    try:
        # Get recent training history
        history = get_training_history()
        
        notifications = []
        now = datetime.now()
        
        # Generate notifications from recent training events
        for h in history[:10]:  # Last 10 training sessions
            if h.get("started_at"):
                try:
                    started = datetime.fromisoformat(h["started_at"])
                    time_ago = now - started
                    
                    # Only include recent notifications (last 24 hours)
                    if time_ago.days == 0:
                        hours_ago = time_ago.seconds // 3600
                        minutes_ago = (time_ago.seconds % 3600) // 60
                        
                        if hours_ago > 0:
                            time_str = f"{hours_ago}h ago"
                        else:
                            time_str = f"{minutes_ago}m ago"
                        
                        if h.get("status") == "completed":
                            notifications.append({
                                "id": h.get("training_id", ""),
                                "type": "success",
                                "message": f"Training completed for {h.get('documents', 'documents')}",
                                "time": time_str,
                                "details": {
                                    "duration": h.get("actual_duration", 0),
                                    "cost": h.get("estimated_cost", 0),
                                    "platform": h.get("platform", "local")
                                }
                            })
                        elif h.get("status") == "failed":
                            notifications.append({
                                "id": h.get("training_id", ""),
                                "type": "warning",
                                "message": f"Training failed for {h.get('documents', 'documents')}",
                                "time": time_str,
                                "details": {
                                    "error": h.get("error_message", "Unknown error"),
                                    "platform": h.get("platform", "local")
                                }
                            })
                        elif h.get("status") == "started":
                            notifications.append({
                                "id": h.get("training_id", ""),
                                "type": "info",
                                "message": f"Training started for {h.get('documents', 'documents')}",
                                "time": time_str,
                                "details": {
                                    "platform": h.get("platform", "local"),
                                    "compute": h.get("compute_target", "local")
                                }
                            })
                except Exception:
                    pass
        
        return jsonify({
            "notifications": notifications[:5],  # Return only 5 most recent
            "last_updated": now.isoformat(),
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching training notifications: {e}", exc_info=True)
        return jsonify({
            "notifications": [],
            "error": f"Failed to fetch notifications: {str(e)}"
        }), 200

@train_bp.route("/training/azure/jobs/<job_name>/download", methods=["POST"])
def download_azure_job_outputs(job_name):
    """
    Manually download outputs from a completed Azure ML job.
    """
    try:
        data = request.get_json() or {}
        training_id = data.get("training_id")
        
        if not training_id:
            return jsonify({"error": "training_id is required"}), 400
        
        trainer = get_trainer()
        
        if not (trainer.azure_ml_client and trainer.azure_ml_client.is_available()):
            return jsonify({"error": "Azure ML not available"}), 400
        
        # Check job status
        job_status = trainer.azure_ml_client.get_job_status(job_name)
        if not job_status or job_status.get("status") != "Completed":
            return jsonify({
                "error": f"Job {job_name} is not completed yet",
                "current_status": job_status.get("status") if job_status else "Unknown"
            }), 400
        
        # Download outputs
        logger.info(f"Manual download requested for job {job_name}")
        trainer._download_azure_outputs(job_name, training_id)
        
        return jsonify({
            "message": f"Download initiated for job {job_name}",
            "training_id": training_id,
            "status": "success"
        }), 200
        
    except Exception as e:
        logger.error(f"Error downloading job outputs: {e}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500
    

@train_bp.route("/training/azure/download-outputs", methods=["POST"])
def manual_download_outputs():
    """
    Manually download outputs from a completed Azure ML job - for testing.
    """
    try:
        data = request.get_json() or {}
        job_name = data.get("job_name")
        training_id = data.get("training_id")
        
        if not job_name or not training_id:
            return jsonify({
                "error": "Both job_name and training_id are required",
                "example": {
                    "job_name": "medical-training-b1e3643c-1733520455",
                    "training_id": "b1e3643c-c03a-41c3-aee4-aceb2349c279"
                }
            }), 400
        
        trainer = get_trainer()
        
        if not (trainer.azure_ml_client and trainer.azure_ml_client.is_available()):
            return jsonify({"error": "Azure ML not available"}), 400
        
        # Check job status first
        job_status = trainer.azure_ml_client.get_job_status(job_name)
        if not job_status:
            return jsonify({
                "error": f"Could not find job: {job_name}",
                "suggestion": "Check the job name and try again"
            }), 400
            
        if job_status.get("status") != "Completed":
            return jsonify({
                "error": f"Job {job_name} is not completed yet",
                "current_status": job_status.get("status"),
                "suggestion": "Wait for the job to complete before downloading"
            }), 400
        
        # Download outputs
        logger.info(f"Manual download requested for job {job_name}")
        trainer._download_azure_outputs(job_name, training_id)
        
        return jsonify({
            "message": f"Download completed for job {job_name}",
            "training_id": training_id,
            "status": "success",
            "job_status": job_status.get("status")
        }), 200
        
    except Exception as e:
        logger.error(f"Error downloading job outputs: {e}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500
    
@train_bp.route("/training/azure/list-jobs", methods=["GET"])
def list_recent_azure_jobs():
    """
    List recent Azure ML jobs to help find the correct job name.
    """
    try:
        trainer = get_trainer()
        
        if not (trainer.azure_ml_client and trainer.azure_ml_client.is_available()):
            return jsonify({"error": "Azure ML not available"}), 400
        
        jobs = []
        
        # Get recent jobs from Azure ML workspace
        for job in trainer.azure_ml_client.ml_client.jobs.list(max_results=20):
            job_info = {
                "name": job.name,
                "display_name": getattr(job, "display_name", job.name),
                "status": job.status,
                "creation_time": job.creation_time.isoformat() if hasattr(job, "creation_time") and job.creation_time else None,
                "start_time": job.start_time.isoformat() if hasattr(job, "start_time") and job.start_time else None,
                "end_time": job.end_time.isoformat() if hasattr(job, "end_time") and job.end_time else None,
                "compute_target": getattr(job, "compute", "Unknown"),
                "experiment_name": getattr(job, "experiment_name", "Unknown"),
            }
            
            # Add tags if they exist (this will help us find training_id)
            if hasattr(job, "tags") and job.tags:
                job_info["tags"] = job.tags
            
            jobs.append(job_info)
        
        # Filter for medical chatbot training jobs
        medical_jobs = [j for j in jobs if "medical-training" in j.get("name", "").lower() or 
                       j.get("experiment_name") == "medical-chatbot-training"]
        
        logger.info(f"Found {len(jobs)} total jobs, {len(medical_jobs)} medical training jobs")
        
        return jsonify({
            "total_jobs": len(jobs),
            "medical_training_jobs": medical_jobs,
            "all_jobs": jobs[:10]  # Show first 10 for debugging
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing Azure jobs: {e}")
        return jsonify({"error": f"Failed to list jobs: {str(e)}"}), 500

@train_bp.route("/training/azure/find-job/<training_id>", methods=["GET"])
def find_job_by_training_id(training_id):
    """
    Find Azure job name by training_id from database.
    """
    try:
        from utils.db_utils import get_training_history_by_id
        
        # Get training record from database
        training_record = get_training_history_by_id(training_id)
        
        if not training_record:
            return jsonify({
                "error": f"Training ID {training_id} not found in database"
            }), 404
        
        azure_job_name = training_record.get("azure_job_name")
        
        if not azure_job_name:
            return jsonify({
                "error": f"No Azure job name found for training ID {training_id}",
                "training_record": training_record
            }), 400
        
        # Also check if the job exists in Azure
        trainer = get_trainer()
        if trainer.azure_ml_client and trainer.azure_ml_client.is_available():
            job_status = trainer.azure_ml_client.get_job_status(azure_job_name)
            
            return jsonify({
                "training_id": training_id,
                "azure_job_name": azure_job_name,
                "job_status": job_status,
                "training_record": training_record
            }), 200
        else:
            return jsonify({
                "training_id": training_id,
                "azure_job_name": azure_job_name,
                "training_record": training_record,
                "note": "Azure ML not available to check job status"
            }), 200
        
    except Exception as e:
        logger.error(f"Error finding job for training ID {training_id}: {e}")
        return jsonify({"error": f"Failed to find job: {str(e)}"}), 500

@train_bp.route("/training/stop", methods=["POST"])
def stop_training():
    """
    Stop the current training process immediately.
    Works for both local and Azure training.
    """
    try:
        trainer = get_trainer()
        
        # Get current training status
        status = trainer.get_training_status()
        if not status.get("is_training"):
            return jsonify({
                "error": "No training is currently in progress",
                "status": "not_training"
            }), 400

        # Stop the training
        success = trainer.stop_training()
        
        if success:
            logger.info("Training stopped successfully by user request")
            return jsonify({
                "message": "Training stopped successfully",
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                "error": "Failed to stop training",
                "status": "error"
            }), 500

    except Exception as e:
        logger.error(f"Error stopping training: {e}", exc_info=True)
        return jsonify({
            "error": f"Failed to stop training: {str(e)}",
            "status": "error"
        }), 500

@train_bp.route("/training/pause", methods=["POST"])
def pause_training():
    """
    Pause the current training process.
    Only works for local training (Azure jobs cannot be paused).
    """
    try:
        trainer = get_trainer()
        
        # Get current training status
        status = trainer.get_training_status()
        if not status.get("is_training"):
            return jsonify({
                "error": "No training is currently in progress",
                "status": "not_training"
            }), 400

        if status.get("is_paused"):
            return jsonify({
                "error": "Training is already paused",
                "status": "already_paused"
            }), 400

        # Check if this is Azure training
        if status.get("training_type") == "azure":
            return jsonify({
                "error": "Azure ML jobs cannot be paused",
                "status": "unsupported",
                "suggestion": "Use stop instead to cancel the Azure job"
            }), 400

        # Pause the training
        success = trainer.pause_training()
        
        if success:
            logger.info("Training paused successfully by user request")
            return jsonify({
                "message": "Training paused successfully",
                "status": "paused",
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                "error": "Failed to pause training",
                "status": "error"
            }), 500

    except Exception as e:
        logger.error(f"Error pausing training: {e}", exc_info=True)
        return jsonify({
            "error": f"Failed to pause training: {str(e)}",
            "status": "error"
        }), 500

@train_bp.route("/training/resume", methods=["POST"])
def resume_training():
    """
    Resume a paused training process.
    Only works for local training.
    """
    try:
        trainer = get_trainer()
        
        # Get current training status
        status = trainer.get_training_status()
        if not status.get("is_training"):
            return jsonify({
                "error": "No training is currently in progress",
                "status": "not_training"
            }), 400

        if not status.get("is_paused"):
            return jsonify({
                "error": "Training is not currently paused",
                "status": "not_paused"
            }), 400

        # Resume the training
        success = trainer.resume_training()
        
        if success:
            logger.info("Training resumed successfully by user request")
            return jsonify({
                "message": "Training resumed successfully",
                "status": "resumed",
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                "error": "Failed to resume training",
                "status": "error"
            }), 500

    except Exception as e:
        logger.error(f"Error resuming training: {e}", exc_info=True)
        return jsonify({
            "error": f"Failed to resume training: {str(e)}",
            "status": "error"
        }), 500
    
@train_bp.route("/documents/<int:doc_id>/azure-info", methods=["GET"])
def get_document_azure_info(doc_id):
    """Get Azure ML specific information for a document."""
    try:
        docs = get_documents_info([doc_id])
        if not docs:
            return jsonify({"error": "Document not found"}), 404
        
        doc = docs[0]
        trainer = get_trainer()
        
        # Check if Azure ML is available
        azure_available, azure_error = check_azure_availability(trainer)
        if not azure_available:
            return jsonify({
                "error": "Azure ML not available",
                "status": "unavailable",
                "reason": azure_error
            }), 200
        
        # Get document processing information
        document_info = {
            "document_id": doc_id,
            "document_name": doc.get("original_name"),
            "file_size": doc.get("file_size", 0),
            "file_type": doc.get("file_type"),
            "status": "processed" if doc.get("file_path") else "pending",
        }
        
        # Calculate Azure-specific metrics
        try:
            file_path = doc.get("file_path")
            if file_path and os.path.exists(file_path):
                file_type = doc.get("file_type", "")
                if file_type == "pdf":
                    text = extract_text_from_pdf(file_path)
                else:
                    text = read_text_file(file_path)
                
                text_length = len(text)
                word_count = len(text.split())
                estimated_chunks = max(1, word_count // 500)
                base_cost = (text_length / 1000) * 0.001
                estimated_cost = round(base_cost, 4)
                
                # Recommend compute target based on document size
                if text_length > 100000:
                    recommended_compute = "Standard_NC6s_v3"  # GPU
                elif text_length > 50000:
                    recommended_compute = "Standard_D4s_v3"   # CPU
                else:
                    recommended_compute = "Standard_D2s_v3"   # Small CPU
                
                document_info.update({
                    "text_length": text_length,
                    "word_count": word_count,
                    "chunk_count": estimated_chunks,
                    "estimated_cost": estimated_cost,
                    "recommended_compute": recommended_compute,
                    "processing_time_estimate": f"{max(1, estimated_chunks // 10)} minutes",
                })
                
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            document_info.update({
                "error": f"Processing error: {str(e)}",
                "chunk_count": 0,
                "estimated_cost": 0.0,
                "recommended_compute": "Standard_D2s_v3",
            })
        
        # Get training history for this document
        try:
            history = get_training_history()
            doc_name = doc.get("original_name", "")
            document_training_history = []
            
            for h in history:
                if doc_name in (h.get("doc_name", "") or ""):
                    document_training_history.append({
                        "training_id": h.get("training_id"),
                        "date": h.get("started_at", "").split("T")[0] if h.get("started_at") else "Unknown",
                        "status": h.get("status", "unknown"),
                        "training_type": h.get("training_type", "local"),
                        "duration": h.get("actual_duration", 0),
                        "cost": h.get("estimated_cost", 0.0),
                    })
            
            document_training_history.sort(key=lambda x: x["date"], reverse=True)
            document_info["training_history"] = document_training_history[:5]
            
        except Exception as e:
            document_info["training_history"] = []
        
        return jsonify(document_info), 200
        
    except Exception as e:
        logger.error(f"Error getting Azure ML info for document {doc_id}: {e}")
        return jsonify({
            "error": f"Failed to get document information: {str(e)}",
            "document_id": doc_id
        }), 500
    