# routes/enhanced_model_routes.py

from flask import Blueprint, request, jsonify
import logging
from datetime import datetime
from typing import Dict, Any

# Get the global trainer instance from app.py
def get_trainer():
    """Get the global Enhanced MedicalChatbotTrainer instance from app.py"""
    try:
        from __main__ import trainer
        return trainer
    except (ImportError, AttributeError):
        # Fallback: create a new trainer instance
        try:
            from training.medical_trainer import MedicalChatbotTrainer
            return MedicalChatbotTrainer()
        except Exception as e:
            logging.error(f"Failed to create enhanced trainer: {e}")
            return None

model_bp = Blueprint("model_bp", __name__)
logger = logging.getLogger(__name__)


@model_bp.route("/models/available", methods=["GET"])
def get_available_models():
    """Get information about all available models."""
    try:
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available",
                "models": {}
            }), 500

        models_info = trainer.get_available_models()
        
        return jsonify({
            "models": models_info,
            "current_preference": trainer.model_preference,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return jsonify({
            "error": str(e),
            "models": {}
        }), 500


@model_bp.route("/models/health", methods=["GET"])
def get_model_health():
    """Get health status of all models."""
    try:
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available",
                "health": {}
            }), 500

        health_status = trainer.get_model_health_status()
        
        return jsonify({
            "health": health_status,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting model health: {e}")
        return jsonify({
            "error": str(e),
            "health": {}
        }), 500


@model_bp.route("/models/preference", methods=["GET"])
def get_model_preference():
    """Get current model preference."""
    try:
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available"
            }), 500

        return jsonify({
            "current_preference": trainer.model_preference,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting model preference: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@model_bp.route("/models/preference", methods=["POST"])
def set_model_preference():
    """Set model preference."""
    try:
        data = request.get_json()
        if not data or "preference" not in data:
            return jsonify({
                "error": "Missing 'preference' in request body"
            }), 400

        preference = data["preference"]
        
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available"
            }), 500

        # Validate preference
        valid_preferences = [
            "local_trained", "eleuther", "openai", 
            "hybrid_local_eleuther", "hybrid_all"
        ]
        
        if preference not in valid_preferences:
            return jsonify({
                "error": f"Invalid preference. Must be one of: {valid_preferences}",
                "valid_preferences": valid_preferences
            }), 400

        # Check if the requested model is available
        available_models = trainer.get_available_models()
        if preference not in available_models or not available_models[preference].get("available", False):
            return jsonify({
                "error": f"Model '{preference}' is not available",
                "available_models": {k: v for k, v in available_models.items() if v.get("available", False)}
            }), 400

        # Set the preference
        trainer.switch_model_preference(preference)
        
        return jsonify({
            "message": f"Model preference set to: {preference}",
            "previous_preference": trainer.model_preference,
            "new_preference": preference,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error setting model preference: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@model_bp.route("/models/test", methods=["POST"])
def test_model():
    """Test a specific model with a test message."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Missing request body"
            }), 400

        model_name = data.get("model", "current")
        test_message = data.get("message", "Hello, this is a test message.")
        
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available"
            }), 500

        # Use current preference if "current" is specified
        if model_name == "current":
            model_name = trainer.model_preference

        # Test the model
        test_session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response = trainer.generate_response(
            message=test_message,
            session_id=test_session_id,
            preferred_model=model_name
        )
        
        return jsonify({
            "model_tested": model_name,
            "test_message": test_message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@model_bp.route("/models/switch", methods=["POST"])
def switch_model():
    """Switch model dynamically (alias for set_preference)."""
    return set_model_preference()


@model_bp.route("/models/compare", methods=["POST"])
def compare_models():
    """Compare responses from multiple models."""
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({
                "error": "Missing 'message' in request body"
            }), 400

        message = data["message"]
        models_to_compare = data.get("models", ["local_trained", "eleuther", "openai"])
        
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available"
            }), 500

        # Get available models
        available_models = trainer.get_available_models()
        
        # Filter to only available models
        available_to_compare = [
            model for model in models_to_compare 
            if model in available_models and available_models[model].get("available", False)
        ]
        
        if not available_to_compare:
            return jsonify({
                "error": "No requested models are available",
                "requested": models_to_compare,
                "available": list(available_models.keys())
            }), 400

        # Generate responses from each model
        comparison_session_id = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        responses = {}
        
        for model_name in available_to_compare:
            try:
                response = trainer.generate_response(
                    message=message,
                    session_id=f"{comparison_session_id}_{model_name}",
                    preferred_model=model_name
                )
                responses[model_name] = response
            except Exception as e:
                logger.error(f"Error getting response from {model_name}: {e}")
                responses[model_name] = {
                    "error": str(e),
                    "reply": f"Error generating response from {model_name}"
                }

        return jsonify({
            "message": message,
            "responses": responses,
            "models_compared": available_to_compare,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@model_bp.route("/models/config", methods=["GET"])
def get_model_config():
    """Get configuration for all models."""
    try:
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available"
            }), 500

        # Get system status which includes model configurations
        system_status = trainer.get_system_status()
        
        return jsonify({
            "config": system_status,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@model_bp.route("/models/performance", methods=["GET"])
def get_model_performance():
    """Get performance metrics for available models."""
    try:
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available"
            }), 500

        # This would be expanded to include actual performance metrics
        # For now, return basic information
        available_models = trainer.get_available_models()
        health_status = trainer.get_model_health_status()
        
        performance_info = {}
        for model_name, model_info in available_models.items():
            if model_info.get("available", False):
                performance_info[model_name] = {
                    "available": True,
                    "healthy": health_status.get(model_name, {}).get("healthy", False),
                    "last_check": health_status.get(model_name, {}).get("last_check"),
                    "response_time": "N/A",  # Would be populated with actual metrics
                    "accuracy": "N/A",      # Would be populated with actual metrics
                    "resource_usage": model_info.get("memory_usage", "N/A")
                }
            else:
                performance_info[model_name] = {
                    "available": False,
                    "reason": model_info.get("error", "Model not loaded")
                }

        return jsonify({
            "performance": performance_info,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@model_bp.route("/models/reload", methods=["POST"])
def reload_models():
    """Reload/refresh all models."""
    try:
        data = request.get_json() or {}
        model_name = data.get("model", "all")
        
        trainer = get_trainer()
        if not trainer:
            return jsonify({
                "error": "Trainer not available"
            }), 500

        if model_name == "all":
            # Refresh all models
            if hasattr(trainer, 'response_generator') and trainer.response_generator:
                trainer.response_generator.refresh_available_models()
                
            # Reload local trainer if specified
            if hasattr(trainer, 'local_trainer') and trainer.local_trainer:
                trainer.local_trainer._is_loaded = False
                
            # Reload EleutherAI client if specified
            if hasattr(trainer, 'eleuther_client') and trainer.eleuther_client:
                trainer.eleuther_client._is_loaded = False
                
            message = "All models refreshed"
        else:
            # Reload specific model
            if model_name == "local_trained" and hasattr(trainer, 'local_trainer'):
                trainer.local_trainer._is_loaded = False
                message = f"Local trained model refreshed"
            elif model_name == "eleuther" and hasattr(trainer, 'eleuther_client'):
                trainer.eleuther_client._is_loaded = False
                message = f"EleutherAI model refreshed"
            else:
                return jsonify({
                    "error": f"Cannot reload model: {model_name}"
                }), 400

        # Get updated model status
        available_models = trainer.get_available_models()
        
        return jsonify({
            "message": message,
            "models": available_models,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return jsonify({
            "error": str(e)
        }), 500


# Error handlers for the model blueprint
@model_bp.errorhandler(404)
def model_not_found(error):
    return jsonify({
        "error": "Model endpoint not found",
        "available_endpoints": [
            "/models/available",
            "/models/health", 
            "/models/preference",
            "/models/test",
            "/models/compare",
            "/models/config",
            "/models/performance",
            "/models/reload"
        ]
    }), 404


@model_bp.errorhandler(500)
def model_internal_error(error):
    return jsonify({
        "error": "Internal server error in model management",
        "message": "Please check server logs for details"
    }), 500