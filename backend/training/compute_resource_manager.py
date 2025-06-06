# training/compute_resource_manager.py

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ComputeResourceManager:
    """Manages Azure compute resources and validation."""
    
    def __init__(self, azure_ml_client):
        self.azure_ml_client = azure_ml_client
    
    def validate_compute_resources(self, data_size: int) -> Dict[str, Any]:
        """
        Validate if Azure compute resources are available for training.
        
        Args:
            data_size: Size of training data in characters
            
        Returns:
            Dict containing validation results
        """
        logger.info(f"Validating compute resources for data size: {data_size}")
        
        if not self.azure_ml_client or not self.azure_ml_client.is_available():
            return {
                "available": False,
                "reason": "Azure ML client not available",
                "recommended_target": None,
                "vm_size": None,
                "hourly_cost": 0.0
            }
        
        if not self.azure_ml_client.ml_client:
            return {
                "available": False,
                "reason": "Azure ML not configured properly",
                "recommended_target": None,
                "vm_size": None,
                "hourly_cost": 0.0
            }
        
        try:
            # Get available compute targets
            compute_targets = self.azure_ml_client.list_compute_targets()
            
            if not compute_targets:
                return {
                    "available": False,
                    "reason": "No compute targets available",
                    "recommended_target": None,
                    "vm_size": None,
                    "hourly_cost": 0.0
                }
            
            # Determine recommended compute target based on data size
            recommended_target, vm_size, hourly_cost = self._get_recommended_compute(
                data_size, compute_targets
            )
            
            # Check if recommended compute is available
            target_info = next(
                (ct for ct in compute_targets if ct["name"] == recommended_target), 
                None
            )
            
            if not target_info:
                return {
                    "available": False,
                    "reason": f"Recommended compute target {recommended_target} not found",
                    "recommended_target": recommended_target,
                    "vm_size": vm_size,
                    "hourly_cost": hourly_cost
                }
            
            # Check compute state
            compute_state = target_info.get("state", "Unknown")
            if compute_state in ["Failed", "Deleting"]:
                return {
                    "available": False,
                    "reason": f"Compute target {recommended_target} is in {compute_state} state",
                    "recommended_target": recommended_target,
                    "vm_size": vm_size,
                    "hourly_cost": hourly_cost
                }
            
            return {
                "available": True,
                "reason": "Compute resources validated successfully",
                "recommended_target": recommended_target,
                "vm_size": vm_size,
                "hourly_cost": hourly_cost,
                "compute_state": compute_state,
                "min_instances": target_info.get("min_instances", 0),
                "max_instances": target_info.get("max_instances", 1)
            }
            
        except Exception as e:
            logger.error(f"Error validating compute resources: {e}")
            return {
                "available": False,
                "reason": f"Validation error: {str(e)}",
                "recommended_target": None,
                "vm_size": None,
                "hourly_cost": 0.0
            }
    
    def _get_recommended_compute(self, data_size: int, compute_targets: list) -> tuple:
        """
        Get recommended compute target based on data size.
        
        Args:
            data_size: Size of training data in characters
            compute_targets: List of available compute targets
            
        Returns:
            Tuple of (target_name, vm_size, hourly_cost)
        """
        # Convert data size to approximate complexity
        # Small: < 100K characters
        # Medium: 100K - 1M characters  
        # Large: > 1M characters
        
        if data_size < 100_000:
            # Small datasets - CPU is sufficient
            cpu_targets = [ct for ct in compute_targets if "cpu" in ct["name"].lower()]
            if cpu_targets:
                target = cpu_targets[0]
                return target["name"], target.get("vm_size", "Standard_DS3_v2"), 0.50
        
        elif data_size < 1_000_000:
            # Medium datasets - prefer GPU but CPU acceptable
            gpu_targets = [ct for ct in compute_targets if "gpu" in ct["name"].lower()]
            if gpu_targets:
                target = gpu_targets[0]
                return target["name"], target.get("vm_size", "Standard_NC6"), 2.50
            
            # Fallback to CPU
            cpu_targets = [ct for ct in compute_targets if "cpu" in ct["name"].lower()]
            if cpu_targets:
                target = cpu_targets[0]
                return target["name"], target.get("vm_size", "Standard_DS3_v2"), 0.50
        
        else:
            # Large datasets - GPU recommended
            gpu_targets = [ct for ct in compute_targets if "gpu" in ct["name"].lower()]
            if gpu_targets:
                target = gpu_targets[0]
                return target["name"], target.get("vm_size", "Standard_NC6"), 2.50
        
        # Default fallback
        if compute_targets:
            target = compute_targets[0]
            return target["name"], target.get("vm_size", "Standard_DS3_v2"), 0.50
        
        # Last resort
        return "cpu-cluster", "Standard_DS3_v2", 0.50
    
    def activate_compute_cluster(self, compute_target: str) -> Dict[str, Any]:
        """
        Activate an Azure compute cluster.
        
        Args:
            compute_target: Name of the compute target to activate
            
        Returns:
            Dict containing activation results
        """
        if not self.azure_ml_client or not self.azure_ml_client.ml_client:
            return {
                "success": False,
                "error": "Azure ML client not available"
            }
        
        try:
            # Get compute target info
            compute_targets = self.azure_ml_client.list_compute_targets()
            target_info = next(
                (ct for ct in compute_targets if ct["name"] == compute_target), 
                None
            )
            
            if not target_info:
                return {
                    "success": False,
                    "error": f"Compute target {compute_target} not found"
                }
            
            current_state = target_info.get("state", "Unknown")
            
            # Check if already running
            if current_state == "Running":
                return {
                    "success": True,
                    "message": f"Compute target {compute_target} is already running",
                    "state": current_state
                }
            
            # Check if in a state that can be activated
            if current_state in ["Failed", "Deleting"]:
                return {
                    "success": False,
                    "error": f"Compute target {compute_target} is in {current_state} state and cannot be activated"
                }
            
            # For AmlCompute, activation typically happens automatically when jobs are submitted
            # This is a placeholder for actual activation logic
            logger.info(f"Activating compute target {compute_target}")
            
            return {
                "success": True,
                "message": f"Compute target {compute_target} activation initiated",
                "state": "Starting",
                "vm_size": target_info.get("vm_size"),
                "max_instances": target_info.get("max_instances", 1)
            }
            
        except Exception as e:
            logger.error(f"Error activating compute cluster {compute_target}: {e}")
            return {
                "success": False,
                "error": f"Failed to activate compute cluster: {str(e)}"
            }
    
    def get_compute_usage_info(self, compute_target: str) -> Dict[str, Any]:
        """
        Get usage information for a compute target.
        
        Args:
            compute_target: Name of the compute target
            
        Returns:
            Dict containing usage information
        """
        if not self.azure_ml_client or not self.azure_ml_client.ml_client:
            return {
                "available": False,
                "error": "Azure ML client not available"
            }
        
        try:
            compute_targets = self.azure_ml_client.list_compute_targets()
            target_info = next(
                (ct for ct in compute_targets if ct["name"] == compute_target), 
                None
            )
            
            if not target_info:
                return {
                    "available": False,
                    "error": f"Compute target {compute_target} not found"
                }
            
            return {
                "available": True,
                "name": compute_target,
                "state": target_info.get("state", "Unknown"),
                "vm_size": target_info.get("vm_size", "Unknown"),
                "current_instances": 0,  # Would need actual API call to get this
                "min_instances": target_info.get("min_instances", 0),
                "max_instances": target_info.get("max_instances", 1),
                "estimated_cost_per_hour": self._estimate_hourly_cost(target_info.get("vm_size"))
            }
            
        except Exception as e:
            logger.error(f"Error getting compute usage info for {compute_target}: {e}")
            return {
                "available": False,
                "error": f"Failed to get usage info: {str(e)}"
            }
    
    def _estimate_hourly_cost(self, vm_size: str) -> float:
        """
        Estimate hourly cost for a VM size.
        
        Args:
            vm_size: Azure VM size name
            
        Returns:
            Estimated hourly cost in USD
        """
        # Simplified cost estimation - in practice would use Azure pricing API
        cost_map = {
            "Standard_DS3_v2": 0.50,
            "Standard_DS4_v2": 1.00,
            "Standard_NC6": 2.50,
            "Standard_NC12": 5.00,
            "Standard_NC24": 10.00
        }
        
        return cost_map.get(vm_size, 1.00)  # Default to $1.00/hour if unknown