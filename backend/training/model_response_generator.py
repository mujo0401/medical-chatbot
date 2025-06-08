"""Enhanced Model response generator for handling responses across different models including hybrid approaches."""

import logging
import time
import asyncio
import concurrent.futures
from typing import Optional, List, Dict, Any
from enum import Enum

from config import MODEL_PREFERENCE

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of available model types."""
    OPENAI = "openai"
    LOCAL_TRAINED = "local_trained"
    ELEUTHER = "eleuther"
    HYBRID_LOCAL_ELEUTHER = "hybrid_local_eleuther"
    HYBRID_ALL = "hybrid_all"


class ModelResponseGenerator:
    """Handles response generation across different model types including hybrid approaches."""
    
    def __init__(self, openai_client=None, local_trainer=None, eleuther_client=None):
        self.openai_client = openai_client
        self.local_trainer = local_trainer
        self.eleuther_client = eleuther_client
        self.model_preference = MODEL_PREFERENCE
        
        # Track available models
        self.available_models = self._check_available_models()
        
        # Hybrid configuration
        self.hybrid_weights = {
            "local_trained": 0.6,
            "eleuther": 0.4,
            "openai": 0.5  # Used in triple hybrid
        }
        
        # Response quality scoring
        self.response_scorer = ResponseQualityScorer()
        
    def _check_available_models(self) -> dict:
        """Check which models are available."""
        available = {
            ModelType.OPENAI: False,
            ModelType.LOCAL_TRAINED: False,
            ModelType.ELEUTHER: False,
            ModelType.HYBRID_LOCAL_ELEUTHER: False,
            ModelType.HYBRID_ALL: False,
        }
        
        if self.openai_client and self.openai_client.is_available():
            available[ModelType.OPENAI] = True
        
        if self.local_trainer and self.local_trainer.is_available():
            available[ModelType.LOCAL_TRAINED] = True
        
        if self.eleuther_client and self.eleuther_client.is_available():
            available[ModelType.ELEUTHER] = True
        
        # Hybrid models available if constituent models are available
        if available[ModelType.LOCAL_TRAINED] and available[ModelType.ELEUTHER]:
            available[ModelType.HYBRID_LOCAL_ELEUTHER] = True
        
        if available[ModelType.LOCAL_TRAINED] and available[ModelType.ELEUTHER] and available[ModelType.OPENAI]:
            available[ModelType.HYBRID_ALL] = True
        
        return available
    
    def generate_response(
        self, 
        message: str, 
        session_id: str, 
        preferred_model: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using preferred model with fallbacks and hybrid support.
        
        Returns a dict with:
        - reply: The final response text
        - model_used: Which model(s) were used
        - confidence: Confidence score (0-1)
        - source_info: Information about how the response was generated
        """
        try:
            # Determine which model to use
            model_to_use = preferred_model or self.model_preference
            
            logger.info(f"Generating response using model: {model_to_use}")
            
            # Convert string model preference to enum
            try:
                model_enum = ModelType(model_to_use.lower())
            except ValueError:
                logger.warning(f"Unknown model preference: {model_to_use}, falling back to best available")
                model_enum = self._get_best_available_model()
            
            # Format messages for model consumption
            messages = self._format_message_with_context(message, context, session_id)
            
            # Generate response based on model type
            if model_enum == ModelType.HYBRID_LOCAL_ELEUTHER:
                return self._generate_hybrid_local_eleuther_response(messages, session_id)
            elif model_enum == ModelType.HYBRID_ALL:
                return self._generate_hybrid_all_response(messages, session_id)
            else:
                return self._generate_single_model_response(model_enum, messages, session_id)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "reply": self._get_error_response(),
                "model_used": "error",
                "confidence": 0.0,
                "source_info": {"error": str(e)}
            }
    
    def _format_message_with_context(self, message: str, context: Optional[str], session_id: str) -> List[Dict[str, str]]:
        """Format the message with context for model consumption."""
        messages = []
        
        # Add system message
        system_prompt = (
            "You are a helpful medical assistant. "
            "Provide accurate, evidence-based information. "
            "Keep responses clear and under 200 words. "
            "Always recommend consulting healthcare professionals for specific medical advice."
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # Add context if available
        if context:
            context_message = f"Context from uploaded documents:\n{context}\n\nUser question: {message}"
            messages.append({"role": "user", "content": context_message})
        else:
            messages.append({"role": "user", "content": message})
        
        return messages
    
    def _generate_single_model_response(
        self, 
        model_enum: ModelType, 
        messages: List[Dict[str, str]], 
        session_id: str
    ) -> Dict[str, Any]:
        """Generate response using a single model."""
        try:
            if model_enum == ModelType.OPENAI and self.available_models[ModelType.OPENAI]:
                response = self._generate_openai_response(messages, session_id)
                return {
                    "reply": response,
                    "model_used": "openai",
                    "confidence": 0.85,
                    "source_info": {"single_model": "openai"}
                }
            
            elif model_enum == ModelType.LOCAL_TRAINED and self.available_models[ModelType.LOCAL_TRAINED]:
                response = self._generate_local_trained_response(messages, session_id)
                return {
                    "reply": response,
                    "model_used": "local_trained",
                    "confidence": 0.80,
                    "source_info": {"single_model": "local_trained"}
                }
            
            elif model_enum == ModelType.ELEUTHER and self.available_models[ModelType.ELEUTHER]:
                response = self._generate_eleuther_response(messages, session_id)
                return {
                    "reply": response,
                    "model_used": "eleuther",
                    "confidence": 0.75,
                    "source_info": {"single_model": "eleuther"}
                }
            
            else:
                # Fallback to best available
                return self._generate_fallback_response(messages, session_id)
                
        except Exception as e:
            logger.error(f"Single model generation failed: {e}")
            raise
    
    def _generate_hybrid_local_eleuther_response(
        self, 
        messages: List[Dict[str, str]], 
        session_id: str
    ) -> Dict[str, Any]:
        """Generate hybrid response using local trained model and EleutherAI."""
        try:
            logger.info("Generating hybrid response: Local Trained + EleutherAI")
            
            responses = {}
            errors = {}
            
            # Generate responses concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                
                if self.available_models[ModelType.LOCAL_TRAINED]:
                    futures["local"] = executor.submit(
                        self._generate_local_trained_response, messages, session_id
                    )
                
                if self.available_models[ModelType.ELEUTHER]:
                    futures["eleuther"] = executor.submit(
                        self._generate_eleuther_response, messages, session_id
                    )
                
                # Collect results
                for model_name, future in futures.items():
                    try:
                        responses[model_name] = future.result(timeout=30)
                    except Exception as e:
                        errors[model_name] = str(e)
                        logger.error(f"Error from {model_name} model: {e}")
            
            # Combine responses
            if len(responses) >= 2:
                # Both models responded
                combined_response = self._combine_responses(
                    responses, 
                    weights={"local": self.hybrid_weights["local_trained"], "eleuther": self.hybrid_weights["eleuther"]}
                )
                confidence = 0.90
                source_info = {
                    "hybrid_type": "local_eleuther",
                    "models_used": list(responses.keys()),
                    "combination_method": "weighted_selection"
                }
            elif len(responses) == 1:
                # Only one model responded
                model_name, response = list(responses.items())[0]
                combined_response = response
                confidence = 0.70
                source_info = {
                    "hybrid_type": "local_eleuther",
                    "models_used": [model_name],
                    "combination_method": "single_fallback",
                    "errors": errors
                }
            else:
                # No models responded
                raise Exception(f"All hybrid models failed: {errors}")
            
            return {
                "reply": combined_response,
                "model_used": "hybrid_local_eleuther",
                "confidence": confidence,
                "source_info": source_info
            }
            
        except Exception as e:
            logger.error(f"Hybrid local+eleuther generation failed: {e}")
            return self._generate_fallback_response(messages, session_id)
    
    def _generate_hybrid_all_response(
        self, 
        messages: List[Dict[str, str]], 
        session_id: str
    ) -> Dict[str, Any]:
        """Generate hybrid response using all available models."""
        try:
            logger.info("Generating hybrid response: All models")
            
            responses = {}
            errors = {}
            
            # Generate responses concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                if self.available_models[ModelType.LOCAL_TRAINED]:
                    futures["local"] = executor.submit(
                        self._generate_local_trained_response, messages, session_id
                    )
                
                if self.available_models[ModelType.ELEUTHER]:
                    futures["eleuther"] = executor.submit(
                        self._generate_eleuther_response, messages, session_id
                    )
                
                if self.available_models[ModelType.OPENAI]:
                    futures["openai"] = executor.submit(
                        self._generate_openai_response, messages, session_id
                    )
                
                # Collect results
                for model_name, future in futures.items():
                    try:
                        responses[model_name] = future.result(timeout=30)
                    except Exception as e:
                        errors[model_name] = str(e)
                        logger.error(f"Error from {model_name} model: {e}")
            
            # Combine responses intelligently
            if len(responses) >= 2:
                combined_response = self._combine_responses_intelligently(responses, messages)
                confidence = min(0.95, 0.75 + (len(responses) * 0.1))
                source_info = {
                    "hybrid_type": "all_models",
                    "models_used": list(responses.keys()),
                    "combination_method": "intelligent_selection",
                    "response_count": len(responses)
                }
            elif len(responses) == 1:
                model_name, response = list(responses.items())[0]
                combined_response = response
                confidence = 0.70
                source_info = {
                    "hybrid_type": "all_models",
                    "models_used": [model_name],
                    "combination_method": "single_fallback",
                    "errors": errors
                }
            else:
                raise Exception(f"All models failed: {errors}")
            
            return {
                "reply": combined_response,
                "model_used": "hybrid_all",
                "confidence": confidence,
                "source_info": source_info
            }
            
        except Exception as e:
            logger.error(f"Hybrid all models generation failed: {e}")
            return self._generate_fallback_response(messages, session_id)
    
    def _combine_responses(
        self, 
        responses: Dict[str, str], 
        weights: Dict[str, float]
    ) -> str:
        """Combine multiple responses using weighted selection."""
        try:
            # Score each response
            scored_responses = []
            for model_name, response in responses.items():
                score = self.response_scorer.score_response(response)
                weight = weights.get(model_name, 0.5)
                final_score = score * weight
                scored_responses.append((final_score, response, model_name))
            
            # Sort by score and return the best
            scored_responses.sort(reverse=True, key=lambda x: x[0])
            best_response = scored_responses[0][1]
            
            logger.info(f"Selected response from {scored_responses[0][2]} (score: {scored_responses[0][0]:.3f})")
            return best_response
            
        except Exception as e:
            logger.error(f"Error combining responses: {e}")
            # Fallback to first available response
            return list(responses.values())[0]
    
    def _combine_responses_intelligently(
        self, 
        responses: Dict[str, str], 
        messages: List[Dict[str, str]]
    ) -> str:
        """Intelligently combine responses using multiple criteria."""
        try:
            # Extract the user question for context analysis
            user_message = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            # Score responses on multiple criteria
            scored_responses = []
            for model_name, response in responses.items():
                scores = self.response_scorer.comprehensive_score(response, user_message)
                
                # Apply model-specific bonuses
                model_bonus = 0.0
                if "medical" in user_message.lower() or "health" in user_message.lower():
                    if model_name == "local":  # Trained on medical data
                        model_bonus = 0.1
                elif "general" in user_message.lower() or "explain" in user_message.lower():
                    if model_name == "eleuther":  # Better general knowledge
                        model_bonus = 0.1
                elif model_name == "openai":  # Generally high quality
                    model_bonus = 0.05
                
                final_score = scores["overall"] + model_bonus
                scored_responses.append((final_score, response, model_name, scores))
            
            # Sort by score
            scored_responses.sort(reverse=True, key=lambda x: x[0])
            
            # Log scoring details
            for score, response, model_name, detailed_scores in scored_responses:
                logger.info(f"{model_name}: {score:.3f} (length: {detailed_scores['length']:.2f}, "
                           f"relevance: {detailed_scores['relevance']:.2f}, quality: {detailed_scores['quality']:.2f})")
            
            return scored_responses[0][1]
            
        except Exception as e:
            logger.error(f"Error in intelligent combination: {e}")
            return self._combine_responses(responses, {"local": 0.4, "eleuther": 0.3, "openai": 0.3})
    
    def _generate_openai_response(self, messages: List[Dict[str, str]], session_id: str) -> str:
        """Generate response using OpenAI API."""
        try:
            return self.openai_client.generate_response(messages, session_id)
        except Exception as e:
            logger.error(f"OpenAI response generation failed: {e}")
            raise
    
    def _generate_local_trained_response(self, messages: List[Dict[str, str]], session_id: str) -> str:
        """Generate response using local trained model."""
        try:
            return self.local_trainer.generate_with_local_model(messages, session_id)
        except Exception as e:
            logger.error(f"Local trained response generation failed: {e}")
            raise
    
    def _generate_eleuther_response(self, messages: List[Dict[str, str]], session_id: str) -> str:
        """Generate response using EleutherAI model."""
        try:
            return self.eleuther_client.generate_response(messages, session_id)
        except Exception as e:
            logger.error(f"EleutherAI response generation failed: {e}")
            raise
    
    def _generate_fallback_response(self, messages: List[Dict[str, str]], session_id: str) -> Dict[str, Any]:
        """Generate response using fallback models."""
        # Try models in order of preference
        fallback_order = [ModelType.OPENAI, ModelType.LOCAL_TRAINED, ModelType.ELEUTHER]
        
        for model_type in fallback_order:
            if self.available_models[model_type]:
                try:
                    logger.info(f"Falling back to {model_type.value}")
                    return self._generate_single_model_response(model_type, messages, session_id)
                except Exception as e:
                    logger.error(f"{model_type.value} fallback failed: {e}")
                    continue
        
        # If all models fail, return error response
        return {
            "reply": self._get_error_response(),
            "model_used": "error",
            "confidence": 0.0,
            "source_info": {"error": "All models failed"}
        }
    
    def _get_best_available_model(self) -> ModelType:
        """Get the best available model based on current availability."""
        # Preference order: hybrid_all > hybrid_local_eleuther > openai > local_trained > eleuther
        if self.available_models[ModelType.HYBRID_ALL]:
            return ModelType.HYBRID_ALL
        elif self.available_models[ModelType.HYBRID_LOCAL_ELEUTHER]:
            return ModelType.HYBRID_LOCAL_ELEUTHER
        elif self.available_models[ModelType.OPENAI]:
            return ModelType.OPENAI
        elif self.available_models[ModelType.LOCAL_TRAINED]:
            return ModelType.LOCAL_TRAINED
        elif self.available_models[ModelType.ELEUTHER]:
            return ModelType.ELEUTHER
        else:
            return ModelType.OPENAI  # Default, will fail gracefully
    
    def _get_error_response(self) -> str:
        """Get a user-friendly error response."""
        return (
            "I apologize, but I'm having trouble processing your request right now. "
            "Please try again later or rephrase your question."
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
            if model_type == ModelType.OPENAI and is_available and self.openai_client:
                try:
                    openai_info = self.openai_client.get_model_info()
                    model_info[model_type.value].update(openai_info)
                except Exception as e:
                    model_info[model_type.value]["error"] = str(e)
            
            elif model_type == ModelType.LOCAL_TRAINED and is_available and self.local_trainer:
                try:
                    local_info = self.local_trainer.get_model_info()
                    model_info[model_type.value].update(local_info)
                except Exception as e:
                    model_info[model_type.value]["error"] = str(e)
            
            elif model_type == ModelType.ELEUTHER and is_available and self.eleuther_client:
                try:
                    eleuther_info = self.eleuther_client.get_model_info()
                    model_info[model_type.value].update(eleuther_info)
                except Exception as e:
                    model_info[model_type.value]["error"] = str(e)
        
        return model_info
    
    def _get_model_description(self, model_type: ModelType) -> str:
        """Get description for model type."""
        descriptions = {
            ModelType.OPENAI: "OpenAI GPT model accessed via API",
            ModelType.LOCAL_TRAINED: "Local fine-tuned model using trained medical data",
            ModelType.ELEUTHER: "EleutherAI GPT-Neo-2.7B model for general language tasks",
            ModelType.HYBRID_LOCAL_ELEUTHER: "Hybrid combining local trained model with EleutherAI",
            ModelType.HYBRID_ALL: "Hybrid combining all available models for best results",
        }
        return descriptions.get(model_type, "Unknown model type")
    
    def set_model_preference(self, preference: str):
        """Set the preferred model for response generation."""
        valid_preferences = [model_type.value for model_type in ModelType]
        if preference in valid_preferences:
            self.model_preference = preference
            logger.info(f"Model preference set to: {preference}")
        else:
            raise ValueError(f"Invalid model preference. Must be one of: {valid_preferences}")
    
    def refresh_available_models(self):
        """Refresh the list of available models."""
        self.available_models = self._check_available_models()
        logger.info(f"Available models refreshed: {[k.value for k, v in self.available_models.items() if v]}")
    
    def get_model_health_status(self) -> dict:
        """Get health status of all models."""
        health_status = {}
        
        for model_type in ModelType:
            if model_type in [ModelType.HYBRID_LOCAL_ELEUTHER, ModelType.HYBRID_ALL]:
                # Skip direct testing of hybrid models
                continue
                
            status = {
                "available": self.available_models[model_type],
                "healthy": False,
                "last_check": None,
            }
            
            if self.available_models[model_type]:
                try:
                    # Test with a simple message
                    test_messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"}
                    ]
                    
                    result = self._generate_single_model_response(model_type, test_messages, "health_check")
                    status["healthy"] = bool(result.get("reply") and len(result["reply"]) > 0)
                    status["test_response"] = result.get("reply", "")[:50] + "..." if len(result.get("reply", "")) > 50 else result.get("reply", "")
                except Exception as e:
                    status["healthy"] = False
                    status["error"] = str(e)
                
                status["last_check"] = time.time()
            
            health_status[model_type.value] = status
        
        return health_status


class ResponseQualityScorer:
    """Scores response quality based on multiple criteria."""
    
    def score_response(self, response: str) -> float:
        """Basic response scoring."""
        if not response or not response.strip():
            return 0.0
        
        score = 0.0
        
        # Length scoring (prefer moderate length)
        length = len(response.strip())
        if 50 <= length <= 300:
            score += 0.3
        elif 20 <= length < 50 or 300 < length <= 500:
            score += 0.2
        elif length > 500:
            score += 0.1
        
        # Content quality indicators
        if any(word in response.lower() for word in ["recommend", "suggest", "advise", "consider"]):
            score += 0.2
        
        if any(word in response.lower() for word in ["medical", "health", "doctor", "physician", "treatment"]):
            score += 0.2
        
        # Proper sentence structure
        if response.strip().endswith(('.', '!', '?')):
            score += 0.1
        
        # Avoid repetitive content
        words = response.lower().split()
        if len(set(words)) / len(words) > 0.7:  # Lexical diversity
            score += 0.2
        
        return min(score, 1.0)
    
    def comprehensive_score(self, response: str, user_question: str) -> Dict[str, float]:
        """Comprehensive response scoring with detailed breakdown."""
        scores = {
            "length": 0.0,
            "relevance": 0.0,
            "quality": 0.0,
            "overall": 0.0
        }
        
        if not response or not response.strip():
            return scores
        
        # Length scoring
        length = len(response.strip())
        if 100 <= length <= 250:
            scores["length"] = 1.0
        elif 50 <= length < 100 or 250 < length <= 400:
            scores["length"] = 0.8
        elif 20 <= length < 50 or 400 < length <= 600:
            scores["length"] = 0.6
        else:
            scores["length"] = 0.3
        
        # Relevance scoring (based on keyword overlap)
        user_words = set(user_question.lower().split())
        response_words = set(response.lower().split())
        overlap = len(user_words & response_words) / max(len(user_words), 1)
        scores["relevance"] = min(overlap * 2, 1.0)
        
        # Quality scoring
        quality_indicators = [
            ("medical", 0.2),
            ("recommend", 0.15),
            ("however", 0.1),
            ("therefore", 0.1),
            ("important", 0.15),
            ("consider", 0.1),
            ("consult", 0.2)
        ]
        
        for indicator, weight in quality_indicators:
            if indicator in response.lower():
                scores["quality"] += weight
        
        scores["quality"] = min(scores["quality"], 1.0)
        
        # Overall score
        scores["overall"] = (scores["length"] * 0.3 + scores["relevance"] * 0.4 + scores["quality"] * 0.3)
        
        return scores