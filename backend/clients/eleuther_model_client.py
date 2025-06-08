# clients/eleuther_model_client.py

import logging
import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    pipeline
)
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class EleutherModelClient:
    """Client for EleutherAI GPT-Neo-2.7B model operations."""

    def __init__(self, model_name: str = "EleutherAI/gpt-neo-2.7B"):
        self.model_name = model_name
        self.max_length = int(os.getenv("MAX_LENGTH", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        self.top_k = int(os.getenv("TOP_K", "50"))
        self.repetition_penalty = float(os.getenv("REPETITION_PENALTY", "1.1"))
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"EleutherAI client will use device: {self.device}")
        
        # Model and tokenizer placeholders
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._is_loaded = False
        
        # Initialize if enough memory/resources available
        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(f"Failed to initialize EleutherAI model on startup: {e}")
            logger.info("Model will be loaded on first use")

    def _initialize_model(self):
        """Initialize the EleutherAI model and tokenizer."""
        if self._is_loaded:
            return
            
        try:
            logger.info(f"Loading EleutherAI model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'  # Important for generation
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if model_kwargs["device_map"] is None:
                self.model.to(self.device)
            
            # Create generation pipeline for easier use
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            self._is_loaded = True
            logger.info(f"EleutherAI model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EleutherAI model: {e}")
            self.model = None
            self.tokenizer = None
            self.generator = None
            self._is_loaded = False
            raise

    def is_available(self) -> bool:
        """Check if the EleutherAI model is available and loaded."""
        return self._is_loaded and self.model is not None and self.tokenizer is not None

    def ensure_loaded(self):
        """Ensure model is loaded, initialize if needed."""
        if not self._is_loaded:
            self._initialize_model()

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        session_id: str,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response using EleutherAI model.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            session_id: Session identifier for logging
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            self.ensure_loaded()
            
            if not self.is_available():
                return "EleutherAI model is not available. Please check the model configuration."

            # Format messages into a conversation prompt
            prompt = self._format_messages_to_prompt(messages)
            
            # Set generation parameters
            max_new_tokens = max_new_tokens or min(200, self.max_length // 2)
            
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            
            logger.info(f"Generating response for session {session_id} with EleutherAI")
            
            # Generate using pipeline
            if self.generator:
                outputs = self.generator(
                    prompt,
                    generation_config=generation_config,
                    return_full_text=False,  # Only return new text
                    clean_up_tokenization_spaces=True
                )
                
                if outputs and len(outputs) > 0:
                    response = outputs[0]["generated_text"].strip()
                else:
                    response = "I apologize, but I couldn't generate a response."
            else:
                # Fallback to direct model generation
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length - max_new_tokens,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # Decode only the new tokens
                new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            if not response or len(response.strip()) < 3:
                response = "I'm here to help with your medical questions. Could you please rephrase your question?"
            
            logger.info(f"EleutherAI generated response of {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"EleutherAI generation error for session {session_id}: {e}")
            return self._get_error_response(str(e))

    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation messages into a single prompt."""
        prompt_parts = []
        
        # Add system message if present
        system_msg = None
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
                break
        
        if system_msg:
            prompt_parts.append(f"System: {system_msg}")
        
        # Add conversation history (limit to last 10 exchanges)
        conversation_msgs = [msg for msg in messages if msg.get("role") in ["user", "assistant"]]
        recent_msgs = conversation_msgs[-20:]  # Last 20 messages (10 exchanges)
        
        for msg in recent_msgs:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            
            if role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add current prompt indicator
        prompt = "\n".join(prompt_parts)
        if not prompt.endswith("Assistant:"):
            prompt += "\nAssistant:"
        
        return prompt

    def _clean_response(self, response: str) -> str:
        """Clean up the generated response."""
        # Remove common artifacts
        response = response.strip()
        
        # Remove any role prefixes that might have leaked through
        prefixes_to_remove = [
            "Assistant:", "Human:", "User:", "System:",
            "AI:", "Bot:", "Response:", "Answer:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove trailing incomplete sentences (ends with incomplete punctuation)
        if response and response[-1] in ",-:;":
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        # Ensure proper sentence ending
        if response and not response[-1] in '.!?':
            response += '.'
        
        return response

    def _get_error_response(self, error_msg: str) -> str:
        """Get a user-friendly error response."""
        if "memory" in error_msg.lower() or "cuda" in error_msg.lower():
            return "I'm experiencing high demand right now. Please try again in a moment."
        elif "timeout" in error_msg.lower():
            return "The response is taking longer than expected. Please try a shorter question."
        else:
            return "I apologize, but I'm having trouble processing your request. Please try again."

    def test_connection(self) -> Dict[str, Any]:
        """Test the EleutherAI model connection and capability."""
        try:
            self.ensure_loaded()
            
            if not self.is_available():
                return {
                    "connected": False,
                    "error": "Model not loaded"
                }
            
            # Test with simple prompt
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            response = self.generate_response(test_messages, "test_session")
            
            return {
                "connected": True,
                "model": self.model_name,
                "device": str(self.device),
                "test_response": response[:100] + "..." if len(response) > 100 else response
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration and status information."""
        return {
            "available": self.is_available(),
            "model_name": self.model_name,
            "device": str(self.device),
            "loaded": self._is_loaded,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "memory_usage": self._get_memory_usage() if self._is_loaded else None
        }

    def _get_memory_usage(self) -> Dict[str, str]:
        """Get current memory usage information."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
            }
        return {"cpu_memory": "CPU mode"}

    def cleanup(self):
        """Clean up model resources."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                
            if hasattr(self, 'generator') and self.generator is not None:
                del self.generator
            
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._is_loaded = False
            logger.info("EleutherAI model resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during EleutherAI cleanup: {e}")

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()