# training/enhanced_local_model_trainer.py

import pickle
import logging
import os
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalModelTrainer:
    """
    Enhanced local model training & inference supporting multiple base models:
    - microsoft/DialoGPT-medium (default for chat)
    - EleutherAI/gpt-neo-2.7B (for general language tasks)
    - Custom fine-tuned models
    """

    def __init__(
        self, 
        training_id: Optional[str] = None,
        base_model_type: str = "dialogpt"  # "dialogpt" or "eleuther"
    ):
        # Model configuration
        self.base_model_type = base_model_type.lower()
        self.base_model_name = self._get_base_model_name()
        self.max_length = int(os.getenv("MAX_LENGTH", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Enhanced local trainer will use device: {self.device}")
        
        # Model components
        self.tokenizer = None
        self.model = None
        
        # Embeddings for similarity search
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Training state
        self.training_id = training_id
        self._is_loaded = False
        
        # Model-specific generation parameters
        self.generation_config = self._get_generation_config()
        
        # Initialize tokenizer and model
        try:
            self._load_model_and_tokenizer()
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer/model in __init__: {e}")
            raise

    def _get_base_model_name(self) -> str:
        """Get the base model name based on type."""
        model_mapping = {
            "dialogpt": "microsoft/DialoGPT-medium",
            "eleuther": "EleutherAI/gpt-neo-2.7B",
            "gpt-neo": "EleutherAI/gpt-neo-2.7B",
        }
        
        model_name = model_mapping.get(self.base_model_type, "microsoft/DialoGPT-medium")
        logger.info(f"Using base model: {model_name} (type: {self.base_model_type})")
        return model_name

    def _get_generation_config(self) -> Dict[str, Any]:
        """Get model-specific generation configuration."""
        if self.base_model_type in ["eleuther", "gpt-neo"]:
            return {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
            }
        else:  # DialoGPT
            return {
                "max_length": self.max_length,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": None,  # Set later with tokenizer
                "no_repeat_ngram_size": 2,
                "early_stopping": True,
            }

    def is_available(self) -> bool:
        """
        Return True if a fine-tuned checkpoint exists or base model is loadable.
        """
        if not self.training_id:
            # If no training_id, check if base model can be loaded
            return self._is_loaded or self._can_load_base_model()

        from config import MODELS_DIR
        fine_tuned_dir = MODELS_DIR / f"local_model_{self.training_id}"
        logger.info(f"DEBUG: Checking directory: {fine_tuned_dir}")
        
        if not fine_tuned_dir.is_dir():
            return self._can_load_base_model()

        # Check for model files in various locations
        locations_to_check = [
            fine_tuned_dir,
            fine_tuned_dir / "outputs" / f"local_model_{self.training_id}",
            fine_tuned_dir / "artifacts" / "outputs" / f"local_model_{self.training_id}"
        ]
        
        for location in locations_to_check:
            if location.is_dir():
                has_model_files = (
                    any(location.glob("*.bin")) or 
                    any(location.glob("*.safetensors")) or
                    any(location.glob("pytorch_model.bin")) or
                    any(location.glob("model.safetensors"))
                )
                if has_model_files:
                    logger.info(f"DEBUG: Model files found in {location}")
                    return True
        
        logger.info("DEBUG: No fine-tuned model files found, checking base model availability")
        return self._can_load_base_model()

    def _can_load_base_model(self) -> bool:
        """Check if the base model can be loaded."""
        try:
            # Quick check by trying to load tokenizer config
            AutoTokenizer.from_pretrained(self.base_model_name, cache_dir=None)
            return True
        except Exception as e:
            logger.warning(f"Cannot load base model {self.base_model_name}: {e}")
            return False

    def _load_model_and_tokenizer(self):
        """
        Load tokenizer + model:
        - If a fine-tuned checkpoint exists, load from there.
        - Otherwise, load the base model.
        """
        if self._is_loaded:
            return

        from config import MODELS_DIR

        model_path = None
        is_fine_tuned = False

        # Check for fine-tuned model if training_id is provided
        if self.training_id:
            fine_tuned_dir = MODELS_DIR / f"local_model_{self.training_id}"
            
            # Check different possible locations
            locations = [
                fine_tuned_dir,
                fine_tuned_dir / "outputs" / f"local_model_{self.training_id}",
                fine_tuned_dir / "artifacts" / "outputs" / f"local_model_{self.training_id}"
            ]
            
            for location in locations:
                if location.is_dir():
                    has_model_files = (
                        any(location.glob("*.bin")) or 
                        any(location.glob("*.safetensors")) or
                        any(location.glob("pytorch_model.bin")) or
                        any(location.glob("model.safetensors"))
                    )
                    if has_model_files:
                        model_path = str(location)
                        is_fine_tuned = True
                        logger.info(f"Loading fine-tuned model from: {model_path}")
                        break

        # Load model and tokenizer
        try:
            if is_fine_tuned and model_path:
                # Load fine-tuned model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                logger.info(f"Loaded fine-tuned {self.base_model_type} model from {model_path}")
            else:
                # Load base model
                logger.info(f"Loading base model: {self.base_model_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                
                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model with appropriate settings for the model type
                model_kwargs = {
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "low_cpu_mem_usage": True,
                }
                
                # For large models like EleutherAI, use device_map if CUDA is available
                if self.base_model_type in ["eleuther", "gpt-neo"] and self.device == "cuda":
                    model_kwargs["device_map"] = "auto"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    **model_kwargs
                )
                
                logger.info(f"Loaded base {self.base_model_type} model: {self.base_model_name}")

            # Move to device if not using device_map
            if not (self.base_model_type in ["eleuther", "gpt-neo"] and self.device == "cuda"):
                self.model.to(self.device)

            # Update generation config with tokenizer info
            if self.generation_config.get("pad_token_id") is None:
                self.generation_config["pad_token_id"] = self.tokenizer.eos_token_id

            self._is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            self.model = None
            self.tokenizer = None
            self._is_loaded = False
            raise

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize + filter the Hugging Face Dataset.
        Handles different model types appropriately.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized before preprocessing.")

        def tokenize_function(examples):
            # For EleutherAI models, we might want different tokenization
            if self.base_model_type in ["eleuther", "gpt-neo"]:
                # For causal LM, we want the labels to be the same as input_ids
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            else:
                # For DialoGPT (conversational model)
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

        def filter_short_sequences(example):
            min_length = 20 if self.base_model_type in ["eleuther", "gpt-neo"] else 10
            return len(example["input_ids"]) >= min_length

        filtered_dataset = tokenized_dataset.filter(filter_short_sequences)
        logger.info(f"Preprocessed dataset: {len(filtered_dataset)} examples (model type: {self.base_model_type})")
        return filtered_dataset

    def train_model(
        self,
        metadata_path: str,
        training_id: str,
        progress_callback: Optional[Callable[[bool, int, str], None]] = None
    ) -> str:
        """
        Fine-tune the model on the raw-text chunks at metadata_path.
        
        Args:
            metadata_path: Path to a pickle file containing a List[str] of chunks.
            training_id: Unique identifier; used to name the output folder.
            progress_callback: Optional function to report (success_flag, percent, message).

        Returns:
            Path (string) to the folder where the model was saved.
        """
        # Set training_id for model loading
        self.training_id = training_id

        try:
            # 1) Load raw-text chunks
            if progress_callback:
                progress_callback(True, 10, "Loading raw text chunks from pickle.")

            meta_file = Path(metadata_path)
            if not meta_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(meta_file, "rb") as f:
                chunk_text_list: List[str] = pickle.load(f)

            if not isinstance(chunk_text_list, list) or len(chunk_text_list) == 0:
                raise ValueError("Loaded metadata was not a nonempty list of strings.")

            logger.info(f"Loaded {len(chunk_text_list)} raw-text chunks from {metadata_path}")

            # 2) Build HF Dataset from strings
            if self.base_model_type in ["eleuther", "gpt-neo"]:
                # For EleutherAI models, format as general text completion
                hf_dicts = [{"text": txt} for txt in chunk_text_list]
            else:
                # For DialoGPT, format as conversational data
                hf_dicts = [{"text": txt} for txt in chunk_text_list]
            
            hf_dataset = Dataset.from_list(hf_dicts)
            logger.info(f"Created Hugging Face Dataset from raw-text chunks (model type: {self.base_model_type})")

            # 3) Preprocess (tokenize + filter)
            if progress_callback:
                progress_callback(True, 20, "Preprocessing dataset (tokenization).")
            
            processed_dataset = self._preprocess_dataset(hf_dataset)
            if len(processed_dataset) == 0:
                raise ValueError("No valid training examples after preprocessing.")

            # 4) Split into train/test
            if len(processed_dataset) > 100:
                split = processed_dataset.train_test_split(test_size=0.1)
                train_dataset = split["train"]
                eval_dataset = split["test"]
                logger.info(f"Dataset split: {len(train_dataset)} train / {len(eval_dataset)} eval")
            else:
                train_dataset = processed_dataset
                eval_dataset = None
                logger.info(f"Using full dataset ({len(train_dataset)}) for training; no eval split.")

            # 5) Load base model + tokenizer for training
            if progress_callback:
                progress_callback(True, 30, "Loading base model and tokenizer.")
            
            # Reset training_id temporarily to load base model
            temp_training_id = self.training_id
            self.training_id = None
            self._is_loaded = False
            self._load_model_and_tokenizer()
            self.training_id = temp_training_id

            # 6) Build output directory
            output_dir = Path("outputs") / f"local_model_{training_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 7) Configure training arguments based on model type
            if progress_callback:
                progress_callback(True, 40, "Setting up training arguments.")

            # Model-specific training configuration
            if self.base_model_type in ["eleuther", "gpt-neo"]:
                # EleutherAI models need different training parameters
                training_args = TrainingArguments(
                    output_dir=str(output_dir),
                    overwrite_output_dir=True,
                    
                    # Training parameters for large models
                    num_train_epochs=2,  # Fewer epochs for large models
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=16,  # Larger accumulation
                    
                    # Learning rate for large models
                    learning_rate=1e-5,  # Lower learning rate
                    warmup_steps=min(100, len(train_dataset) // 16),
                    
                    # Regularization
                    weight_decay=0.01,
                    max_grad_norm=1.0,
                    
                    # Memory and performance
                    fp16=True if self.device == "cuda" else False,
                    gradient_checkpointing=True,  # Save memory
                    dataloader_pin_memory=False,
                    
                    # Logging and saving
                    logging_steps=10,
                    save_steps=100,
                    evaluation_strategy="steps" if eval_dataset else "no",
                    eval_steps=100 if eval_dataset else None,
                    save_total_limit=2,
                    
                    # Other settings
                    prediction_loss_only=True,
                    remove_unused_columns=False,
                    report_to=None,
                    dataloader_num_workers=0,
                    logging_dir=str(output_dir / "logs"),
                )
            else:
                # DialoGPT training parameters (original)
                training_args = TrainingArguments(
                    output_dir=str(output_dir),
                    overwrite_output_dir=True,
                    
                    num_train_epochs=3,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=8,
                    
                    learning_rate=5e-6,
                    warmup_steps=min(50, len(train_dataset) // 8),
                    
                    weight_decay=0.01,
                    max_grad_norm=1.0,
                    
                    logging_steps=5,
                    save_steps=50,
                    evaluation_strategy="steps" if eval_dataset else "no",
                    eval_steps=50 if eval_dataset else None,
                    save_total_limit=3,
                    
                    prediction_loss_only=True,
                    fp16=False,
                    dataloader_pin_memory=False,
                    remove_unused_columns=False,
                    report_to=None,
                    dataloader_num_workers=0,
                    logging_dir=str(output_dir / "logs"),
                )

            # 8) Set up callbacks
            callbacks = []
            if eval_dataset:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
                training_args.load_best_model_at_end = True
                training_args.metric_for_best_model = "loss"
                training_args.greater_is_better = False

            # 9) Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                callbacks=callbacks
            )

            # 10) Add progress callback
            if progress_callback:
                total_steps = (len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs

                class ProgressCallbackTrainer(TrainerCallback):
                    def __init__(self, cb_fn, total_steps: int):
                        self.cb_fn = cb_fn
                        self.total_steps = total_steps

                    def on_train_begin(self, args, state, control, **kwargs):
                        if self.cb_fn:
                            self.cb_fn(True, 55, "Training begun")
                        return control

                    def on_step_end(self, args, state, control, **kwargs):
                        if self.cb_fn and state.global_step > 0:
                            prog = min(95, 55 + int((state.global_step / self.total_steps) * 40))
                            self.cb_fn(True, prog, f"Step {state.global_step}/{self.total_steps}")
                        return control

                trainer.add_callback(ProgressCallbackTrainer(progress_callback, total_steps))

            # 11) Train
            if progress_callback:
                progress_callback(True, 60, f"Starting training with {self.base_model_type} model.")
            
            trainer.train()

            # 12) Save the fine-tuned model & tokenizer
            if progress_callback:
                progress_callback(True, 95, "Saving fine-tuned model and tokenizer.")
            
            trainer.save_model(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))

            # 13) Reload the fine-tuned model
            self.training_id = training_id
            self._is_loaded = False
            
            if progress_callback:
                progress_callback(True, 100, f"Training completed successfully! ({self.base_model_type})")
            
            logger.info(f"Model training completed. Saved to {output_dir}")
            return str(output_dir)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if progress_callback:
                progress_callback(False, 0, f"Training failed: {str(e)}")
            raise

    def generate_with_local_model(self, messages: List[dict], session_id: str) -> str:
        """
        Generate a response with the fine-tuned model.
        Handles both DialoGPT and EleutherAI model types.
        """
        try:
            # Ensure model is loaded
            self._load_model_and_tokenizer()

            if not self._is_loaded:
                raise RuntimeError("Model not loaded")

            # Format prompt based on model type
            if self.base_model_type in ["eleuther", "gpt-neo"]:
                prompt = self._format_prompt_for_eleuther(messages)
                response = self._generate_eleuther_style(prompt, session_id)
            else:
                prompt = self._format_prompt_for_dialogpt(messages)
                response = self._generate_dialogpt_style(prompt, session_id)

            # Clean and validate response
            response = self._clean_response(response)
            
            if not response or len(response.strip()) < 3:
                response = "Hello! I'm here to help with your medical questions. How can I assist you today?"
                
            return response

        except Exception as e:
            logger.error(f"Failed to generate with local model: {e}")
            return "Hello! I'm here to help with your medical questions. How can I assist you today?"

    def _format_prompt_for_eleuther(self, messages: List[dict]) -> str:
        """Format messages for EleutherAI model."""
        prompt_parts = []
        
        # Add system message if present
        for msg in messages:
            if msg.get("role") == "system":
                prompt_parts.append(f"System: {msg.get('content', '')}")
                break
        
        # Add conversation (limit to recent messages)
        conversation_msgs = [msg for msg in messages if msg.get("role") in ["user", "assistant"]]
        recent_msgs = conversation_msgs[-10:]  # Last 10 messages
        
        for msg in recent_msgs:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            
            if role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts)
        if not prompt.endswith("Assistant:"):
            prompt += "\nAssistant:"
        
        return prompt

    def _format_prompt_for_dialogpt(self, messages: List[dict]) -> str:
        """Format messages for DialoGPT model."""
        prompt = ""
        for msg in messages[-10:]:  # Last 10 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "system":
                prompt += f"System: {content}\n"

        prompt += "Assistant:"
        return prompt

    def _generate_eleuther_style(self, prompt: str, session_id: str) -> str:
        """Generate response using EleutherAI-style generation."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length - 150  # Leave room for generation
        ).to(self.device)

        input_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=self.generation_config["temperature"],
                top_p=self.generation_config["top_p"],
                top_k=self.generation_config["top_k"],
                repetition_penalty=self.generation_config["repetition_penalty"],
                do_sample=self.generation_config["do_sample"],
                early_stopping=self.generation_config["early_stopping"],
                no_repeat_ngram_size=self.generation_config["no_repeat_ngram_size"],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only new tokens
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def _generate_dialogpt_style(self, prompt: str, session_id: str) -> str:
        """Generate response using DialoGPT-style generation."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        ).to(self.device)

        input_length = inputs['input_ids'].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Extract only new tokens
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def _clean_response(self, response: str) -> str:
        """Clean up the generated response."""
        response = response.strip()
        
        # Remove role prefixes
        prefixes = ["Assistant:", "Human:", "User:", "System:", "Bot:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Ensure proper sentence ending
        if response and not response[-1] in '.!?':
            response += '.'
        
        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration and status information."""
        return {
            "available": self.is_available(),
            "loaded": self._is_loaded,
            "base_model_type": self.base_model_type,
            "base_model_name": self.base_model_name,
            "training_id": self.training_id,
            "device": str(self.device),
            "max_length": self.max_length,
            "temperature": self.temperature,
            "generation_config": self.generation_config,
        }

    def cleanup(self):
        """Clean up model resources."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._is_loaded = False
            logger.info(f"Enhanced local model trainer resources cleaned up ({self.base_model_type})")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def switch_base_model(self, new_base_model_type: str):
        """Switch to a different base model type."""
        if new_base_model_type.lower() != self.base_model_type:
            logger.info(f"Switching base model from {self.base_model_type} to {new_base_model_type}")
            
            # Cleanup current model
            self.cleanup()
            
            # Set new model type
            self.base_model_type = new_base_model_type.lower()
            self.base_model_name = self._get_base_model_name()
            self.generation_config = self._get_generation_config()
            
            # Reload model
            try:
                self._load_model_and_tokenizer()
                logger.info(f"Successfully switched to {self.base_model_type}")
            except Exception as e:
                logger.error(f"Failed to switch to {new_base_model_type}: {e}")
                raise