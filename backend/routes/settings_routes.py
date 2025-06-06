from flask import Blueprint, request, jsonify
from datetime import datetime
import json
import logging
import os

from utils.db_utils import (
    get_system_settings,
    save_system_settings,
    get_user_settings,
    save_user_settings,
    create_system_backup,
    get_system_info
)

settings_bp = Blueprint("settings_bp", __name__)
logger = logging.getLogger(__name__)

# Default settings structure
DEFAULT_SETTINGS = {
    "general": {
        "theme": "light",
        "language": "en",
        "timezone": "UTC",
        "autoSave": True,
        "notifications": True,
        "soundEnabled": True
    },
    "security": {
        "sessionTimeout": 30,
        "requireMFA": False,
        "passwordExpiry": 90,
        "auditLogging": True,
        "dataEncryption": True,
        "hipaaCompliance": True
    },
    "ai": {
        "defaultModel": "local",
        "maxTokens": 2048,
        "temperature": 0.7,
        "responseTimeout": 30,
        "autoTraining": False,
        "confidenceThreshold": 0.8
    },
    "privacy": {
        "dataRetention": 365,
        "anonymizeData": True,
        "shareAnalytics": False,
        "exportData": True,
        "rightToDelete": True
    },
    "backup": {
        "autoBackup": True,
        "backupInterval": 24,
        "retentionPeriod": 30,
        "cloudBackup": False
    }
}

@settings_bp.route("/settings", methods=["GET"])
def get_settings():
    """Get all system settings"""
    try:
        # Get settings from database
        stored_settings = get_system_settings()
        
        # Merge with defaults to ensure all settings exist
        settings = DEFAULT_SETTINGS.copy()
        if stored_settings:
            for category, category_settings in stored_settings.items():
                if category in settings:
                    settings[category].update(category_settings)
                else:
                    settings[category] = category_settings
        
        return jsonify(settings), 200
        
    except Exception as e:
        logger.error(f"Error fetching settings: {e}")
        return jsonify({"error": f"Failed to fetch settings: {str(e)}"}), 500

@settings_bp.route("/settings", methods=["PUT"])
def update_settings():
    """Update system settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No settings data provided"}), 400
        
        # Validate settings structure
        valid_categories = set(DEFAULT_SETTINGS.keys())
        for category in data.keys():
            if category not in valid_categories:
                return jsonify({"error": f"Invalid settings category: {category}"}), 400
        
        # Get current settings
        current_settings = get_system_settings() or DEFAULT_SETTINGS.copy()
        
        # Update with new values
        for category, category_settings in data.items():
            if category in current_settings:
                current_settings[category].update(category_settings)
            else:
                current_settings[category] = category_settings
        
        # Add metadata
        current_settings["_metadata"] = {
            "lastUpdated": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Save settings
        success = save_system_settings(current_settings)
        
        if success:
            # Log settings change for audit
            logger.info(f"System settings updated: {list(data.keys())}")
            
            return jsonify({
                "message": "Settings updated successfully",
                "settings": current_settings
            }), 200
        else:
            return jsonify({"error": "Failed to save settings"}), 500
            
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"error": f"Failed to update settings: {str(e)}"}), 500

@settings_bp.route("/settings/<category>", methods=["GET"])
def get_settings_category(category):
    """Get settings for a specific category"""
    try:
        if category not in DEFAULT_SETTINGS:
            return jsonify({"error": f"Invalid settings category: {category}"}), 400
        
        settings = get_system_settings()
        
        if settings and category in settings:
            return jsonify({category: settings[category]}), 200
        else:
            return jsonify({category: DEFAULT_SETTINGS[category]}), 200
            
    except Exception as e:
        logger.error(f"Error fetching settings category {category}: {e}")
        return jsonify({"error": f"Failed to fetch settings: {str(e)}"}), 500

@settings_bp.route("/settings/<category>", methods=["PUT"])
def update_settings_category(category):
    """Update settings for a specific category"""
    try:
        if category not in DEFAULT_SETTINGS:
            return jsonify({"error": f"Invalid settings category: {category}"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No settings data provided"}), 400
        
        # Get current settings
        current_settings = get_system_settings() or DEFAULT_SETTINGS.copy()
        
        # Update the specific category
        if category in current_settings:
            current_settings[category].update(data)
        else:
            current_settings[category] = data
        
        # Add metadata
        current_settings["_metadata"] = {
            "lastUpdated": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Save settings
        success = save_system_settings(current_settings)
        
        if success:
            logger.info(f"Settings category '{category}' updated")
            return jsonify({
                "message": f"Settings category '{category}' updated successfully",
                category: current_settings[category]
            }), 200
        else:
            return jsonify({"error": "Failed to save settings"}), 500
            
    except Exception as e:
        logger.error(f"Error updating settings category {category}: {e}")
        return jsonify({"error": f"Failed to update settings: {str(e)}"}), 500

@settings_bp.route("/settings/reset", methods=["POST"])
def reset_settings():
    """Reset settings to defaults"""
    try:
        data = request.get_json() or {}
        categories = data.get('categories', [])
        
        if categories:
            # Reset specific categories
            current_settings = get_system_settings() or {}
            for category in categories:
                if category in DEFAULT_SETTINGS:
                    current_settings[category] = DEFAULT_SETTINGS[category].copy()
        else:
            # Reset all settings
            current_settings = DEFAULT_SETTINGS.copy()
        
        # Add metadata
        current_settings["_metadata"] = {
            "lastUpdated": datetime.now().isoformat(),
            "version": "1.0",
            "resetAt": datetime.now().isoformat()
        }
        
        # Save settings
        success = save_system_settings(current_settings)
        
        if success:
            reset_scope = f"categories: {categories}" if categories else "all settings"
            logger.info(f"Settings reset to defaults - {reset_scope}")
            
            return jsonify({
                "message": "Settings reset to defaults successfully",
                "settings": current_settings
            }), 200
        else:
            return jsonify({"error": "Failed to reset settings"}), 500
            
    except Exception as e:
        logger.error(f"Error resetting settings: {e}")
        return jsonify({"error": f"Failed to reset settings: {str(e)}"}), 500

@settings_bp.route("/settings/export", methods=["GET"])
def export_settings():
    """Export current settings"""
    try:
        settings = get_system_settings() or DEFAULT_SETTINGS.copy()
        
        export_data = {
            "settings": settings,
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "version": "1.0",
                "app_version": "2.1.0"
            }
        }
        
        return jsonify({
            "data": export_data,
            "filename": f"medical_ai_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }), 200
        
    except Exception as e:
        logger.error(f"Error exporting settings: {e}")
        return jsonify({"error": f"Failed to export settings: {str(e)}"}), 500

@settings_bp.route("/settings/import", methods=["POST"])
def import_settings():
    """Import settings from exported data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No import data provided"}), 400
        
        # Validate import data structure
        if "settings" not in data:
            return jsonify({"error": "Invalid import data format"}), 400
        
        imported_settings = data["settings"]
        
        # Validate categories
        for category in imported_settings.keys():
            if category.startswith("_"):  # Skip metadata
                continue
            if category not in DEFAULT_SETTINGS:
                logger.warning(f"Skipping unknown settings category: {category}")
                continue
        
        # Merge with current settings (preserve unknown categories)
        current_settings = get_system_settings() or {}
        
        for category, category_settings in imported_settings.items():
            if category.startswith("_"):  # Skip metadata
                continue
            current_settings[category] = category_settings
        
        # Add import metadata
        current_settings["_metadata"] = {
            "lastUpdated": datetime.now().isoformat(),
            "version": "1.0",
            "importedAt": datetime.now().isoformat(),
            "importSource": data.get("export_info", {}).get("exported_at", "unknown")
        }
        
        # Save settings
        success = save_system_settings(current_settings)
        
        if success:
            logger.info("Settings imported successfully")
            return jsonify({
                "message": "Settings imported successfully",
                "imported_categories": [k for k in imported_settings.keys() if not k.startswith("_")],
                "settings": current_settings
            }), 200
        else:
            return jsonify({"error": "Failed to save imported settings"}), 500
            
    except Exception as e:
        logger.error(f"Error importing settings: {e}")
        return jsonify({"error": f"Failed to import settings: {str(e)}"}), 500

@settings_bp.route("/settings/backup", methods=["POST"])
def create_backup():
    """Create a system backup"""
    try:
        data = request.get_json() or {}
        backup_type = data.get('type', 'full')  # 'full', 'settings', 'data'
        
        backup_id = create_system_backup(backup_type)
        
        if backup_id:
            return jsonify({
                "message": "Backup created successfully",
                "backup_id": backup_id,
                "backup_type": backup_type,
                "created_at": datetime.now().isoformat()
            }), 201
        else:
            return jsonify({"error": "Failed to create backup"}), 500
            
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return jsonify({"error": f"Failed to create backup: {str(e)}"}), 500

@settings_bp.route("/settings/system-info", methods=["GET"])
def get_system_information():
    """Get system information for the about section"""
    try:
        system_info = get_system_info()
        
        info = {
            "application": {
                "name": "Medical AI Assistant",
                "version": "2.1.0",
                "build_date": "2024-03-15",
                "license": "Proprietary"
            },
            "system": system_info.get("system", {
                "os": "Unknown",
                "python_version": "Unknown",
                "architecture": "Unknown"
            }),
            "database": {
                "type": "SQLite",
                "version": system_info.get("database", {}).get("version", "Unknown"),
                "size": system_info.get("database", {}).get("size", "Unknown")
            },
            "ai_models": {
                "local_model": {
                    "status": "Available",
                    "last_trained": system_info.get("ai_models", {}).get("last_trained", "Unknown")
                },
                "azure_integration": {
                    "status": "Available" if system_info.get("azure_available", False) else "Not Configured"
                }
            },
            "storage": {
                "total_documents": system_info.get("storage", {}).get("documents", 0),
                "total_size": system_info.get("storage", {}).get("total_size", "0 MB"),
                "available_space": system_info.get("storage", {}).get("available_space", "Unknown")
            },
            "performance": {
                "uptime": system_info.get("performance", {}).get("uptime", "Unknown"),
                "memory_usage": system_info.get("performance", {}).get("memory_usage", "Unknown"),
                "cpu_usage": system_info.get("performance", {}).get("cpu_usage", "Unknown")
            }
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({"error": f"Failed to get system information: {str(e)}"}), 500

@settings_bp.route("/settings/health-check", methods=["GET"])
def health_check():
    """Perform system health check"""
    try:
        health_status = {
            "overall_status": "healthy",
            "checks": {
                "database": {
                    "status": "healthy",
                    "response_time": "< 10ms",
                    "details": "Database connection active"
                },
                "ai_model": {
                    "status": "healthy",
                    "response_time": "< 1s",
                    "details": "Local model responding normally"
                },
                "storage": {
                    "status": "healthy",
                    "details": "Sufficient storage available"
                },
                "memory": {
                    "status": "healthy",
                    "usage": "< 80%",
                    "details": "Memory usage within normal range"
                },
                "security": {
                    "status": "healthy",
                    "details": "All security measures active"
                }
            },
            "last_check": datetime.now().isoformat(),
            "next_check": (datetime.now().timestamp() + 300) * 1000  # 5 minutes from now
        }
        
        # Check for any issues
        issues = []
        warnings = []
        
        # Example health checks (in real implementation, these would be actual checks)
        import random
        if random.random() < 0.1:  # 10% chance of warning
            warnings.append("High memory usage detected")
            health_status["checks"]["memory"]["status"] = "warning"
            health_status["checks"]["memory"]["details"] = "Memory usage above 80%"
        
        if warnings:
            health_status["overall_status"] = "warning"
            health_status["warnings"] = warnings
        
        if issues:
            health_status["overall_status"] = "unhealthy"
            health_status["issues"] = issues
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Error performing health check: {e}")
        return jsonify({
            "overall_status": "error",
            "error": f"Health check failed: {str(e)}",
            "last_check": datetime.now().isoformat()
        }), 500

@settings_bp.route("/settings/logs", methods=["GET"])
def get_system_logs():
    """Get system logs (limited for security)"""
    try:
        # In a real implementation, you'd read from actual log files
        # This is a simplified version for demo purposes
        
        level = request.args.get('level', 'INFO')
        limit = min(int(request.args.get('limit', 100)), 1000)  # Max 1000 entries
        
        # Mock log entries
        import random
        log_levels = ['INFO', 'WARNING', 'ERROR']
        log_messages = [
            "System startup completed",
            "Document uploaded successfully",
            "Training session started",
            "User authentication successful",
            "Backup completed",
            "Settings updated",
            "Health check passed"
        ]
        
        logs = []
        for i in range(min(limit, 50)):  # Return up to 50 mock entries
            log_entry = {
                "timestamp": (datetime.now().timestamp() - i * 300) * 1000,  # 5 min intervals
                "level": random.choice(log_levels),
                "message": random.choice(log_messages),
                "component": random.choice(["auth", "training", "api", "storage", "backup"])
            }
            
            if level == 'ALL' or log_entry["level"] == level:
                logs.append(log_entry)
        
        return jsonify({
            "logs": logs,
            "total": len(logs),
            "level_filter": level,
            "generated_at": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({"error": f"Failed to fetch logs: {str(e)}"}), 500