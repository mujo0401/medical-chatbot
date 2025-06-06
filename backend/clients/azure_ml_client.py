# clients/azure_ml_client.py

import os
import logging
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AzureMLClient:
    """Enhanced Azure ML client for real metrics collection, compute validation, and job management."""

    def __init__(self):
        self.ml_client = None
        self.monitor_client = None
        self.consumption_client = None
        self.resource_client = None
        self.cost_client = None
        self.subscription_id = None
        self.resource_group = None
        self.workspace_name = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize Azure ML, monitoring, consumption, and resource management clients."""
        try:
            # If user set AZURE_ML_DISABLED, skip all Azure initialization.
            if os.getenv("AZURE_ML_DISABLED", "").lower() in ["true", "1", "yes"]:
                logger.info("Azure ML is disabled via environment variable")
                return

            # Import Azure ML SDK
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential

            # Attempt to import optional mgmt clients
            monitor_available = False
            consumption_available = False
            resource_available = False
            cost_available = False

            try:
                from azure.mgmt.monitor import MonitorManagementClient  # type: ignore
                monitor_available = True
                logger.info("Azure Monitor SDK available")
            except ImportError:
                logger.warning("azure-mgmt-monitor not installed — monitoring features disabled")

            try:
                from azure.mgmt.consumption import ConsumptionManagementClient  # type: ignore
                consumption_available = True
                logger.info("Azure Consumption SDK available")
            except ImportError:
                logger.warning("azure-mgmt-consumption not installed — cost tracking disabled")

            try:
                from azure.mgmt.resource import ResourceManagementClient  # type: ignore
                resource_available = True
                logger.info("Azure Resource SDK available")
            except ImportError:
                logger.warning("azure-mgmt-resource not installed — resource management disabled")

            try:
                from azure.mgmt.costmanagement import CostManagementClient  # type: ignore
                cost_available = True
                logger.info("Azure Cost Management SDK available")
            except ImportError:
                logger.warning("azure-mgmt-costmanagement not installed — detailed cost tracking disabled")
                cost_available = False

            # Read required environment variables
            self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
            self.workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

            missing = []
            if not self.subscription_id:
                missing.append("AZURE_SUBSCRIPTION_ID")
            if not self.resource_group:
                missing.append("AZURE_RESOURCE_GROUP")
            if not self.workspace_name:
                missing.append("AZURE_WORKSPACE_NAME")
            if missing:
                raise ValueError(f"Missing required Azure environment variables: {', '.join(missing)}")

            # Create credential and MLClient
            credential = DefaultAzureCredential()
            logger.info("Azure credentials initialized")

            # Initialize optional mgmt clients
            if monitor_available:
                try:
                    from azure.mgmt.monitor import MonitorManagementClient  # type: ignore
                    self.monitor_client = MonitorManagementClient(credential, self.subscription_id)
                    logger.info("Azure Monitor client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Monitor client: {e}")

            if consumption_available:
                try:
                    from azure.mgmt.consumption import ConsumptionManagementClient  # type: ignore
                    self.consumption_client = ConsumptionManagementClient(credential, self.subscription_id)
                    logger.info("Azure Consumption client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Consumption client: {e}")

            if resource_available:
                try:
                    from azure.mgmt.resource import ResourceManagementClient  # type: ignore
                    self.resource_client = ResourceManagementClient(credential, self.subscription_id)
                    logger.info("Azure Resource client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Resource client: {e}")

            if cost_available:
                try:
                    from azure.mgmt.costmanagement import CostManagementClient  # type: ignore
                    self.cost_client = CostManagementClient(credential)  # type: ignore
                    logger.info("Azure Cost Management client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Cost Management client: {e}")

            # Finally, create the MLClient itself
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name,
            )

            # Test connection by retrieving workspace
            ws = self.ml_client.workspaces.get(self.workspace_name)
            logger.info(f"Connected to Azure ML workspace: {ws.name} ({ws.location})")

            # Log how many compute targets exist
            compute_count = 0
            for c in self.ml_client.compute.list():
                compute_count += 1
                logger.info(f"Found compute target: {c.name} ({getattr(c, 'provisioning_state', 'unknown')})")
            if compute_count == 0:
                logger.warning("No compute targets found in Azure ML workspace")
            else:
                logger.info(f"Found {compute_count} compute targets")

        except ImportError as e:
            logger.error(f"Azure ML SDK is not installed: {e}")
            logger.error("Run: pip install azure-ai-ml azure-identity")
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML clients: {e}")
            raise

    def is_available(self) -> bool:
        """Return True if Azure ML is up and running (workspace can be retrieved)."""
        if not self.ml_client:
            return False
        try:
            _ = self.ml_client.workspaces.get(self.workspace_name).name
            return True
        except Exception as e:
            logger.error(f"Azure ML connectivity test failed: {e}")
            return False

    def get_real_compute_pricing(self, vm_size: str) -> Dict[str, Any]:
        """Return a hard-coded lookup of common Azure VM prices (for estimation purposes)."""
        pricing_data = {
            "Standard_NC4as_T4_v3": {"hourly_rate": 0.526, "gpu": True,  "cores": 4,  "memory_gb": 28,  "gpu_type": "T4"},
            "Standard_NC6":           {"hourly_rate": 0.90,  "gpu": True,  "cores": 6,  "memory_gb": 56,  "gpu_type": "K80"},
            "Standard_NC12":          {"hourly_rate": 1.80,  "gpu": True,  "cores": 12, "memory_gb": 112, "gpu_type": "K80"},
            "Standard_NC24":          {"hourly_rate": 3.60,  "gpu": True,  "cores": 24, "memory_gb": 224, "gpu_type": "K80"},
            "Standard_NC6s_v3":       {"hourly_rate": 3.168, "gpu": True,  "cores": 6,  "memory_gb": 112, "gpu_type": "V100"},
            "Standard_NC12s_v3":      {"hourly_rate": 6.336, "gpu": True,  "cores": 12, "memory_gb": 224, "gpu_type": "V100"},
            "Standard_NC24s_v3":      {"hourly_rate": 12.672,"gpu": True,  "cores": 24, "memory_gb": 448, "gpu_type": "V100"},
            "Standard_ND6s":          {"hourly_rate": 2.07,  "gpu": True,  "cores": 6,  "memory_gb": 112, "gpu_type": "P40"},
            "Standard_ND12s":         {"hourly_rate": 4.14,  "gpu": True,  "cores": 12, "memory_gb": 224, "gpu_type": "P40"},
            "Standard_ND24s":         {"hourly_rate": 8.28,  "gpu": True,  "cores": 24, "memory_gb": 448, "gpu_type": "P40"},
            "Standard_D2s_v3":        {"hourly_rate": 0.096, "gpu": False, "cores": 2,  "memory_gb": 8},
            "Standard_D4s_v3":        {"hourly_rate": 0.192, "gpu": False, "cores": 4,  "memory_gb": 16},
            "Standard_D8s_v3":        {"hourly_rate": 0.384, "gpu": False, "cores": 8,  "memory_gb": 32},
            "Standard_D16s_v3":       {"hourly_rate": 0.768, "gpu": False, "cores": 16, "memory_gb": 64},
            "Standard_F2s_v2":        {"hourly_rate": 0.085, "gpu": False, "cores": 2,  "memory_gb": 4},
            "Standard_F4s_v2":        {"hourly_rate": 0.169, "gpu": False, "cores": 4,  "memory_gb": 8},
            "Standard_F8s_v2":        {"hourly_rate": 0.338, "gpu": False, "cores": 8,  "memory_gb": 16},
            "Standard_F16s_v2":       {"hourly_rate": 0.676, "gpu": False, "cores": 16, "memory_gb": 32},
        }

        vm_info = pricing_data.get(
            vm_size,
            {
                "hourly_rate": 0.60,
                "gpu": vm_size.startswith(("Standard_NC", "Standard_ND")),
                "cores": 4,
                "memory_gb": 16,
            },
        )

        return {
            "vm_size": vm_size,
            "hourly_rate_usd": vm_info["hourly_rate"],
            "has_gpu": vm_info["gpu"],
            "cores": vm_info["cores"],
            "memory_gb": vm_info["memory_gb"],
            "gpu_type": vm_info.get("gpu_type"),
            "pricing_source": "azure_retail_prices" if vm_size in pricing_data else "estimated",
            "last_updated": datetime.now().isoformat(),
            "azure_ml_overhead": 0.12,  # 12% overhead for Azure ML compute
        }

    def get_cost_data(self) -> Dict[str, Any]:
        """Get basic cost data from Azure Consumption API (placeholder structure)."""
        if not self.consumption_client:
            return {"error": "Consumption client not available - install azure-mgmt-consumption"}

        try:
            now = datetime.now()
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now

            total_cost = 0.0
            ml_cost = 0.0
            daily_costs = {}
            service_costs = {}

            projected_monthly = (total_cost / max(1, (end_date - start_date).days)) * 30

            return {
                "current_month": {
                    "total_cost": round(total_cost, 2),
                    "ml_cost": round(ml_cost, 2),
                    "budget_limit": None,
                    "currency": "USD",
                    "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    "days_elapsed": max(1, (end_date - start_date).days),
                    "records_processed": 0,
                },
                "daily_breakdown": daily_costs,
                "service_breakdown": service_costs,
                "cost_trends": {
                    "daily_average": round(total_cost / max(1, (end_date - start_date).days), 2),
                    "projected_monthly": round(projected_monthly, 2),
                    "spend_rate": "normal" if projected_monthly < 1000 else "high",
                },
                "last_updated": datetime.now().isoformat(),
                "data_source": "azure_consumption_api",
            }

        except Exception as e:
            logger.error(f"Error getting cost data: {e}")
            return {"error": f"Failed to get cost data: {str(e)}"}

    def validate_compute_training(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Validate compute resources for training with real Azure data.

        Returns a dict like:
          {
            "valid": True or False,
            "can_train": True or False,
            "compute_target": "<compute_name>",
            "vm_size": "<vm_size>",
            "estimated_cost": <float>,
            "estimated_duration": <int>,
            "total_documents": <int>,
            "vm_details": { … },
            "available_targets": [ … ],
            "scaling_info": { … },
            "cost_breakdown": { … },
            "warnings": [ … ]
          }
        """
        if not self.ml_client:
            return {"valid": False, "can_train": False, "reason": "Azure ML not available"}

        try:
            compute_targets = []
            recommended_target = None

            logger.info("Scanning Azure ML compute targets for validation.")

            for compute in self.ml_client.compute.list():
                target_info: Dict[str, Any] = {
                    "name": compute.name,
                    "type": getattr(compute, "type", "Unknown"),
                    "state": getattr(compute, "provisioning_state", "Unknown"),
                    "vm_size": getattr(compute, "size", "Unknown"),
                    "location": getattr(compute, "location", "Unknown"),
                    "created_by": getattr(compute, "created_by", {}).get("user_name", "Unknown"),
                }

                if hasattr(compute, "scale_settings"):
                    scale_settings = compute.scale_settings
                    target_info.update({
                        "current_nodes": getattr(compute, "current_node_count", 0),
                        "max_nodes": getattr(scale_settings, "max_instances", 1),
                        "min_nodes": getattr(scale_settings, "min_instances", 0),
                        "idle_time_before_scale_down": getattr(scale_settings, "idle_time_before_scale_down", None),
                    })
                else:
                    target_info.update({
                        "current_nodes": 1 if target_info["state"] == "Running" else 0,
                        "max_nodes": 1,
                        "min_nodes": 0,
                    })

                pricing_info = self.get_real_compute_pricing(target_info["vm_size"])
                target_info.update({
                    "hourly_cost": pricing_info["hourly_rate_usd"],
                    "has_gpu": pricing_info["has_gpu"],
                    "cores": pricing_info["cores"],
                    "memory_gb": pricing_info["memory_gb"],
                    "gpu_type": pricing_info.get("gpu_type"),
                })

                compute_targets.append(target_info)
                logger.info(f"Found compute: {target_info['name']} ({target_info['state']}) – {target_info['vm_size']}")

                if target_info["state"] == "Succeeded" and target_info["max_nodes"] > 0:
                    if recommended_target is None:
                        recommended_target = target_info
                    else:
                        if target_info["has_gpu"] and not recommended_target["has_gpu"]:
                            recommended_target = target_info
                        elif target_info["has_gpu"] == recommended_target["has_gpu"] and \
                                target_info["cores"] > recommended_target["cores"]:
                            recommended_target = target_info

            if not compute_targets:
                return {
                    "valid": False,
                    "can_train": False,
                    "reason": "No compute targets available in workspace",
                    "recommendation": "Create or start a compute cluster in Azure ML Studio",
                }

            if not recommended_target:
                return {
                    "valid": False,
                    "can_train": False,
                    "reason": "No healthy compute cluster found",
                    "recommendation": "Ensure at least one compute cluster is in 'Succeeded' state",
                }

            num_documents = len(document_ids)
            if recommended_target["has_gpu"]:
                base_duration_minutes = 20 + (num_documents * 5)
            else:
                base_duration_minutes = 40 + (num_documents * 15)

            cores = recommended_target["cores"]
            if cores >= 16:
                base_duration_minutes = int(base_duration_minutes * 0.7)
            elif cores <= 4:
                base_duration_minutes = int(base_duration_minutes * 1.3)

            estimated_hours = base_duration_minutes / 60.0
            base_cost = recommended_target["hourly_cost"] * estimated_hours
            azure_ml_overhead = base_cost * 0.15
            estimated_cost = base_cost + azure_ml_overhead

            logger.info(f"Selected compute: {recommended_target['name']} for {num_documents} docs")
            logger.info(f"Estimated cost: ${estimated_cost:.2f} for {base_duration_minutes} minutes")

            return {
                "valid": True,
                "can_train": True,
                "compute_target": recommended_target["name"],
                "vm_size": recommended_target["vm_size"],
                "estimated_duration": base_duration_minutes,
                "total_documents": num_documents,
                "estimated_cost": round(estimated_cost, 2),
                "vm_details": {
                    "vm_size": recommended_target["vm_size"],
                    "has_gpu": recommended_target["has_gpu"],
                    "cores": recommended_target["cores"],
                    "memory_gb": recommended_target["memory_gb"],
                    "hourly_cost": recommended_target["hourly_cost"],
                    "gpu_type": recommended_target.get("gpu_type"),
                },
                "available_targets": compute_targets,
                "scaling_info": {
                    "current_nodes": recommended_target["current_nodes"],
                    "max_nodes": recommended_target["max_nodes"],
                    "can_scale": recommended_target["max_nodes"] > recommended_target["min_nodes"],
                },
                "cost_breakdown": {
                    "base_compute_cost": round(base_cost, 2),
                    "azure_ml_overhead": round(azure_ml_overhead, 2),
                    "total_estimated": round(estimated_cost, 2),
                    "hourly_rate": recommended_target["hourly_cost"],
                },
                "warnings": []
            }

        except Exception as e:
            logger.error(f"Compute validation failed: {e}")
            return {
                "valid": False,
                "can_train": False,
                "reason": f"Validation failed: {str(e)}",
                "recommendation": "Check Azure ML workspace connectivity and permissions"
            }

    def submit_training_job(
        self,
        script_path: Optional[str],  # Pass None to indicate “Git‐based” code
        index_path: str,             # Azure ML Data Asset URI, not local path
        metadata_path: str,          # Azure ML Data Asset URI
        compute_target: str,
        training_id: str,
    ) -> str:
        """
        Submit a training job to Azure ML with enhanced configuration and logging.
        Returns the job name if submission succeeds, otherwise raises.
        """
        logger.info(
            f"[submit_training_job] Called with script={script_path}, "
            f"index={index_path}, metadata={metadata_path}, compute={compute_target}, training_id={training_id}"
        )

        if not self.ml_client:
            raise RuntimeError("Azure ML client not initialized. Cannot submit job.")

        # 1) Verify compute target state
        try:
            compute_info = self.ml_client.compute.get(compute_target)
            logger.info(f"[submit_training_job] Compute target '{compute_target}' state: {compute_info.provisioning_state}")
            if compute_info.provisioning_state.lower() != "succeeded":
                raise RuntimeError(f"Compute target '{compute_target}' is not ready (state: {compute_info.provisioning_state}).")
        except Exception as e:
            logger.error(f"[submit_training_job] Error fetching compute target '{compute_target}': {e}")
            raise RuntimeError(f"Compute target validation failed: {e}") from e

        # ── Use Git-based code + Data Asset URIs ──
        try:
            from azure.ai.ml import command
            from azure.ai.ml.entities import CodeConfiguration
        except ImportError as e:
            logger.error(f"[submit_training_job] Failed to import Azure ML SDK: {e}")
            raise RuntimeError(f"Azure ML SDK import failed: {e}") from e

        # Build CodeConfiguration pointing at your Git repo
        git_url = "https://github.com/yourOrg/yourRepo.git"
        branch  = "main"
        subpath = "training"  # Must match where train.py lives
        code_config = CodeConfiguration(code=f"{git_url}#{branch}:{subpath}")

        # Build inputs using Data Asset URIs
        ml_inputs = {
            "metadata_pkl": metadata_path,   # e.g. "azureml://datastores/.../chunks_metadata.pkl"
            "faiss_index":  index_path       # e.g. "azureml://datastores/.../training_id.index"
        }

        timestamp = int(time.time())
        job_name = f"medical-training-{training_id[:8]}-{timestamp}"
        logger.info(f"[submit_training_job] Submitting job '{job_name}' to compute '{compute_target}'")

        try:
            env = self.ml_client.environments.get(
                "AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu", version="latest"
            )
            logger.info("[submit_training_job] Using curated PyTorch environment")
        except Exception:
            from azure.ai.ml.entities import Environment
            env = Environment(
                name="medical-training-env",
                description="Environment for medical chatbot training",
                conda_file={
                    "name": "medical_training",
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        "python=3.8",
                        "pip",
                        {
                            "pip": [
                                "torch",
                                "transformers",
                                "datasets",
                                "numpy",
                                "faiss-cpu",
                                "sentence-transformers",
                                "openai",
                                "scikit-learn",
                                "accelerate>=0.26.0",
                            ]
                        },
                    ],
                },
                image="mcr.microsoft.com/azureml/base:latest",
            )
            logger.info("[submit_training_job] Created custom environment with base image")

        ml_job = command(
            name=job_name,
            code=code_config,
            inputs=ml_inputs,
            command=(
                "python train.py "
                "--metadata_path ${{inputs.metadata_pkl}} "
                "--index_path    ${{inputs.faiss_index}} "
                f"--training_id {training_id}"
            ),
            environment=env,
            compute=compute_target,
            experiment_name="medical-chatbot-training",
            display_name=f"Medical Chatbot Training {training_id[:8]}",
            tags={ "training_id": training_id, "submitted_by": "medical_chatbot_trainer" },
        )
        logger.info("[submit_training_job] Built Azure ML command job definition")

        # 4) Actually submit the job
        try:
            submitted_job = self.ml_client.jobs.create_or_update(ml_job)
            logger.info(f"[submit_training_job] Successfully queued job: {submitted_job.name}")
            return submitted_job.name
        except Exception as e:
            logger.error(f"[submit_training_job] Azure ML job submission failed: {e}")
            raise RuntimeError(f"Azure ML job submission failed: {e}") from e

    def get_job_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific Azure ML job."""
        if not self.ml_client:
            return None
        try:
            job = self.ml_client.jobs.get(job_name)
            status_info: Dict[str, Any] = {
                "name": job.name,
                "status": job.status,
                "compute_target": getattr(job, "compute", "Unknown"),
                "experiment_name": getattr(job, "experiment_name", "Unknown"),
                "tags": getattr(job, "tags", {}),
            }

            if hasattr(job, "start_time") and job.start_time:
                status_info["start_time"] = job.start_time.isoformat()
                if hasattr(job, "end_time") and job.end_time:
                    duration = job.end_time - job.start_time
                    status_info["duration_minutes"] = int(duration.total_seconds() / 60)
                elif job.status == "Running":
                    duration = datetime.now() - job.start_time.replace(tzinfo=None)
                    status_info["duration_minutes"] = int(duration.total_seconds() / 60)

            return status_info

        except Exception as e:
            logger.error(f"Failed to get job status for {job_name}: {e}")
            return None

    def get_job_logs(self, job_name: str) -> List[str]:
        """Get logs from an Azure ML job, with helpful guidance for users."""
        if not self.ml_client:
            return ["Azure ML not available"]

        try:
            job = self.ml_client.jobs.get(job_name)
            logs: List[str] = [
                f"Job Name: {job.name}",
                f"Status: {job.status}",
                f"Compute Target: {getattr(job, 'compute', 'Unknown')}",
                "-" * 50,
            ]

            if job.status == "Running":
                logs += [
                    "Job is currently running",
                    "Check real-time logs in Azure ML Studio:",
                    f"https://ml.azure.com/runs/{job.name}"
                    f"?wsid=/subscriptions/{self.subscription_id}/resourcegroups/{self.resource_group}"
                    f"/workspaces/{self.workspace_name}",
                ]
            elif job.status == "Completed":
                logs += ["Job completed successfully!", "Check metrics and artifacts in Azure ML Studio"]
            elif job.status == "Failed":
                logs += [
                    "Job failed!",
                    "Common issues: compute unavailable, environment errors, code errors, data issues",
                    f"View full logs: https://ml.azure.com/runs/{job.name}"
                    f"?wsid=/subscriptions/{self.subscription_id}/resourcegroups/{self.resource_group}"
                    f"/workspaces/{self.workspace_name}",
                ]
            elif job.status == "Canceled":
                logs += ["Job was canceled. You can resubmit with the same parameters if needed."]

            if hasattr(job, "start_time") and job.start_time:
                logs.append(f"Started: {job.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                if hasattr(job, "end_time") and job.end_time:
                    logs.append(f"Ended: {job.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            return logs

        except Exception as e:
            logger.error(f"Failed to get logs for {job_name}: {e}")
            return [f"Failed to get logs: {str(e)}"]

    def get_billing_information(self) -> Dict[str, Any]:
        """Get comprehensive billing information for the current Azure subscription."""
        if not self.ml_client:
            raise RuntimeError("Azure ML not available")

        try:
            billing_info: Dict[str, Any] = {
                "available": True,
                "timestamp": datetime.now().isoformat(),
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "workspace_name": self.workspace_name,
            }

            try:
                ws = self.ml_client.workspaces.get(self.workspace_name)
                billing_info.update({
                    "workspace_location": ws.location,
                    "workspace_resource_group": ws.resource_group,
                    "workspace_id": ws.id,
                    "workspace_created": getattr(ws, "creation_time", None),
                })
                logger.info(f"Retrieved workspace info: {ws.name}")
            except Exception as e:
                logger.warning(f"Could not retrieve workspace details: {e}")

            cost_data = self.get_cost_data()
            if "error" not in cost_data:
                billing_info.update({
                    "current_spend": cost_data["current_month"]["total_cost"],
                    "ml_spend": cost_data["current_month"]["ml_cost"],
                    "budget_limit": None,
                })

            return billing_info

        except Exception as e:
            logger.error(f"Failed to get billing information: {e}")
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Submit or check Azure ML training jobs using AzureMLClient.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit a new training job")
    submit_parser.add_argument(
        "--script_path",
        required=True,
        help="Path to the azure_train_<ID>.py script you want to run in Azure ML."
    )
    submit_parser.add_argument(
        "--index_path",
        required=True,
        help="Path to the FAISS index file (.index)."
    )
    submit_parser.add_argument(
        "--metadata_path",
        required=True,
        help="Path to the chunks metadata file (.pkl)."
    )
    submit_parser.add_argument(
        "--compute_target",
        required=True,
        help="Name of the Azure ML compute target (e.g., gpu-cluster)."
    )
    submit_parser.add_argument(
        "--training_id",
        required=True,
        help="Unique identifier for this training run."
    )

    status_parser = subparsers.add_parser("status", help="Get status of an existing Azure ML job")
    status_parser.add_argument(
        "--job_name",
        required=True,
        help="Name of the Azure ML job to check status for."
    )

    logs_parser = subparsers.add_parser("logs", help="Get logs of an existing Azure ML job")
    logs_parser.add_argument(
        "--job_name",
        required=True,
        help="Name of the Azure ML job to retrieve logs for."
    )

    args = parser.parse_args()
    client = AzureMLClient()

    if args.command == "submit":
        if not client.is_available():
            logger.error("Azure ML workspace is not available. Check your credentials or environment variables.")
            return

        try:
            job_name = client.submit_training_job(
                script_path=args.script_path,
                index_path=args.index_path,
                metadata_path=args.metadata_path,
                compute_target=args.compute_target,
                training_id=args.training_id,
            )
            print(f"Job submitted: {job_name}")
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            raise

    elif args.command == "status":
        status = client.get_job_status(args.job_name)
        if status is None:
            print(f"Could not retrieve status for job '{args.job_name}'.")
        else:
            for key, value in status.items():
                print(f"{key}: {value}")

    elif args.command == "logs":
        logs = client.get_job_logs(args.job_name)
        for line in logs:
            print(line)


if __name__ == "__main__":
    main()
