"""Model response generator for handling responses across different models."""

import logging
import time
from typing import Optional
from enum import Enum

from config import MODEL_PREFERENCE

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of available model types."""
    OPENAI = "openai"
    LOCAL = "local"
    AZURE_TRAINED = "azure_trained"


class ModelResponseGenerator:
    """Handles response generation across different model types."""
    
    def __init__(self, openai_client=None, local_trainer=None):
        self.openai_client = openai_client
        self.local_trainer = local_trainer
        self.model_preference = MODEL_PREFERENCE
        
        # Track available models
        self.available_models = self._check_available_models()
    
    def _check_available_models(self) -> dict:
        """Check which models are available."""
        available = {
            ModelType.OPENAI: False,
            ModelType.LOCAL: False,
            ModelType.AZURE_TRAINED: False,
        }
        
        if self.openai_client and self.openai_client.is_available():
            available[ModelType.OPENAI] = True
        
        if self.local_trainer and self.local_trainer.is_available():
            available[ModelType.LOCAL] = True
        
        # Azure trained models would be loaded as local models
        # This is a placeholder for future implementation
        available[ModelType.AZURE_TRAINED] = False
        
        return available
    
    def generate_response(self, message: str, session_id: str, 
                         preferred_model: Optional[str] = None) -> str:
        """Generate response using preferred model with fallbacks."""
        try:
            # Determine which model to use
            model_to_use = preferred_model or self.model_preference
            
            # Try the preferred model first
            if model_to_use == "openai" and self.available_models[ModelType.OPENAI]:
                return self._generate_openai_response(message, session_id)
            elif model_to_use == "local" and self.available_models[ModelType.LOCAL]:
                return self._generate_local_response(message, session_id)
            
            # Fallback logic
            return self._generate_fallback_response(message, session_id)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_error_response()
    
    def _generate_openai_response(self, message: str, session_id: str) -> str:
        """Generate response using OpenAI API."""
        try:
            return self.openai_client.generate_response(message, session_id)
        except Exception as e:
            logger.error(f"OpenAI response generation failed: {e}")
            raise
    
    def _generate_local_response(self, message: str, session_id: str) -> str:
        """Generate response using local model."""
        try:
            return self.local_trainer.generate_response(message, session_id)
        except Exception as e:
            logger.error(f"Local response generation failed: {e}")
            raise
    
    def _generate_fallback_response(self, message: str, session_id: str) -> str:
        """Generate response using fallback models."""
        # Try local model if OpenAI failed
        if self.available_models[ModelType.LOCAL]:
            try:
                logger.info("Falling back to local model")
                return self._generate_local_response(message, session_id)
            except Exception as e:
                logger.error(f"Local fallback failed: {e}")
        
        # Try OpenAI if local failed and OpenAI is available
        if self.available_models[ModelType.OPENAI]:
            try:
                logger.info("Falling back to OpenAI model")
                return self._generate_openai_response(message, session_id)
            except Exception as e:
                logger.error(f"OpenAI fallback failed: {e}")
        
        # If all models fail, return error response
        return self._get_error_response()
    
    def _get_error_response(self) -> str:
        """Get a user-friendly error response."""
        return (
            "I apologize, but I'm having trouble processing your request right now. "
            "Please try again later."
        )
    
    def get_available_models(self) -> dict:
        """Get information about available models."""
        model_info = {}
        
        for model_type, is_available in self.available_models.items():
            model_info[model_type.value] = {
                "available": is_available,
                "description": self._get_model_description(model_type),
            }
            
            # Add specific model information if available
            if model_type == ModelType.OPENAI and is_available:
                try:
                    openai_config = self.openai_client.validate_configuration()
                    model_info[model_type.value].update(openai_config)
                except Exception as e:
                    model_info[model_type.value]["error"] = str(e)
            
            elif model_type == ModelType.LOCAL and is_available:
                try:
                    local_info = self.local_trainer.get_model_info()
                    model_info[model_type.value].update(local_info)
                except Exception as e:
                    model_info[model_type.value]["error"] = str(e)
        
        return model_info
    
    def _get_model_description(self, model_type: ModelType) -> str:
        """Get description for model type."""
        descriptions = {
            ModelType.OPENAI: "OpenAI GPT model accessed via API",
            ModelType.LOCAL: "Local DialoGPT model running on server",
            ModelType.AZURE_TRAINED: "Custom trained model via Azure ML",
        }
        return descriptions.get(model_type, "Unknown model type")
    
    def set_model_preference(self, preference: str):
        """Set the preferred model for response generation."""
        valid_preferences = ["openai", "local", "azure_trained"]
        if preference in valid_preferences:
            self.model_preference = preference
            logger.info(f"Model preference set to: {preference}")
        else:
            raise ValueError(f"Invalid model preference. Must be one of: {valid_preferences}")
    
    def test_model_performance(self, test_messages: list, model_type: str = None) -> dict:
        """Test model performance with a set of test messages."""
        model_to_test = model_type or self.model_preference
        
        results = {
            "model_type": model_to_test,
            "total_tests": len(test_messages),
            "successful_responses": 0,
            "failed_responses": 0,
            "average_response_time": 0,
            "responses": []
        }
        
        import time
        total_time = 0
        
        for i, message in enumerate(test_messages):
            start_time = time.time()
            
            try:
                response = self.generate_response(
                    message, 
                    f"test_session_{i}", 
                    preferred_model=model_to_test
                )
                
                response_time = time.time() - start_time
                total_time += response_time
                
                results["responses"].append({
                    "input": message,
                    "output": response,
                    "response_time": response_time,
                    "success": True
                })
                results["successful_responses"] += 1
                
            except Exception as e:
                response_time = time.time() - start_time
                total_time += response_time
                
                results["responses"].append({
                    "input": message,
                    "error": str(e),
                    "response_time": response_time,
                    "success": False
                })
                results["failed_responses"] += 1
        
        if results["total_tests"] > 0:
            results["average_response_time"] = total_time / results["total_tests"]
        
        return results
    
    def refresh_available_models(self):
        """Refresh the list of available models."""
        self.available_models = self._check_available_models()
        logger.info(f"Available models refreshed: {[k.value for k, v in self.available_models.items() if v]}")
    
    def get_model_health_status(self) -> dict:
        """Get health status of all models."""
        health_status = {}
        
        for model_type in ModelType:
            status = {
                "available": self.available_models[model_type],
                "healthy": False,
                "last_check": None,
            }
            
            if self.available_models[model_type]:
                try:
                    # Test with a simple message
                    test_response = self.generate_response(
                        "Hello", 
                        "health_check", 
                        preferred_model=model_type.value
                    )
                    status["healthy"] = bool(test_response and len(test_response) > 0)
                    status["test_response"] = test_response[:50] + "..." if len(test_response) > 50 else test_response
                except Exception as e:
                    status["healthy"] = False
                    status["error"] = str(e)
                
                status["last_check"] = time.time()
            
            health_status[model_type.value] = status
        
        return health_status