"""
Enhanced Medical Chatbot Trainer - orchestrates all training and inference components,
with support for multiple model types including EleutherAI and hybrid approaches.
"""

import os
import logging
import pickle
import uuid
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Import from parent config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODEL_PREFERENCE, MODELS_DIR, OPENAI_API_KEY, LOCAL_BASE_MODEL_TYPE,
    ELEUTHER_MODEL_NAME, ENABLE_HYBRID_MODELS, get_model_config, get_system_prompt
)

# Import our enhanced components
from clients.azure_ml_client import AzureMLClient
from clients.openai_client import OpenAIClient
from clients.eleuther_model_client import EleutherModelClient
from .local_model_trainer import LocalModelTrainer
from .training_data_processor import TrainingDataProcessor
from .training_status_tracker import TrainingStatusTracker, AzureJobMonitor, TrainingProgressCallback
from .model_response_generator import ModelResponseGenerator

# Import database utility to fetch past conversation turns
from utils.db_utils import fetch_conversation_history, update_training_history_azure_job

logger = logging.getLogger(__name__)


class MedicalChatbotTrainer:
    """Enhanced orchestrator for medical chatbot training, indexing, and inference with multi-model support."""

    def __init__(self):
        self.model_preference = MODEL_PREFERENCE
        self._initialization_errors = []

        logger.info("Initializing Enhanced Medical Chatbot Trainer...")

        # Initialize all components with proper error handling
        self._init_components()
        self._setup_integrations()
        self._validate_setup()

        if self._initialization_errors:
            logger.warning(f"Trainer initialized with {len(self._initialization_errors)} warnings:")
            for error in self._initialization_errors:
                logger.warning(f"   - {error}")
        else:
            logger.info("Enhanced Medical Chatbot Trainer initialized successfully")

    def _init_components(self):
        """Initialize components with enhanced error handling and validation."""
        
        # Enhanced Azure ML client
        try:
            logger.info("Initializing Azure ML client...")
            self.azure_ml_client = AzureMLClient()
            if self.azure_ml_client.is_available():
                logger.info("Azure ML client initialized and connected")
            else:
                logger.warning("Azure ML client not available")
                self._initialization_errors.append("Azure ML client not available")
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML client: {e}")
            self.azure_ml_client = None
            self._initialization_errors.append(f"Azure ML initialization failed: {str(e)}")

        # Initialize traditional ML client reference for backward compatibility
        try:
            if self.azure_ml_client and self.azure_ml_client.is_available():
                self.ml_client = self.azure_ml_client.ml_client
                logger.info("Azure ML workspace client available")
            else:
                self.ml_client = None
                logger.info("Azure ML workspace client not available")
        except Exception as e:
            logger.error(f"Failed to get workspace client: {e}")
            self.ml_client = None

        # OpenAI client
        try:
            logger.info("Initializing OpenAI client...")
            self.openai_client = OpenAIClient()
            if self.openai_client.is_available():
                logger.info("OpenAI client initialized")
            else:
                logger.info("OpenAI client not available (API key not set)")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
            self._initialization_errors.append(f"OpenAI initialization failed: {str(e)}")

        # EleutherAI client
        try:
            logger.info("Initializing EleutherAI client...")
            self.eleuther_client = EleutherModelClient(model_name=ELEUTHER_MODEL_NAME)
            if self.eleuther_client.is_available():
                logger.info("EleutherAI client initialized")
            else:
                logger.warning("EleutherAI client not fully available")
        except Exception as e:
            logger.error(f"Failed to initialize EleutherAI client: {e}")
            self.eleuther_client = None
            self._initialization_errors.append(f"EleutherAI initialization failed: {str(e)}")

        # Enhanced local model trainer
        try:
            logger.info(f"Initializing enhanced local model trainer (base: {LOCAL_BASE_MODEL_TYPE})...")
            self.local_trainer = LocalModelTrainer(base_model_type=LOCAL_BASE_MODEL_TYPE)
            if self.local_trainer.is_available():
                logger.info("Enhanced local model trainer initialized")
            else:
                logger.warning("Enhanced local model trainer not fully available")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced local trainer: {e}")
            self.local_trainer = None
            self._initialization_errors.append(f"Enhanced local trainer initialization failed: {str(e)}")

        # Data processor: chunking + embedding + FAISS index
        try:
            logger.info("Initializing training data processor...")
            self.data_processor = TrainingDataProcessor()
            logger.info("Training data processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize data processor: {e}")
            self.data_processor = None
            self._initialization_errors.append(f"Data processor initialization failed: {str(e)}")

        # Status tracking for both local and Azure jobs
        try:
            logger.info("Initializing status tracker...")
            self.status_tracker = TrainingStatusTracker()

            # Enhanced Azure job monitor
            if self.azure_ml_client and self.azure_ml_client.is_available():
                self.azure_monitor = AzureJobMonitor(self.azure_ml_client, self.status_tracker)
                logger.info("Azure job monitor initialized")
            else:
                self.azure_monitor = None
                logger.info("Azure job monitor not available")

            logger.info("Status tracking initialized")
        except Exception as e:
            logger.error(f"Failed to initialize status tracker: {e}")
            self.status_tracker = None
            self.azure_monitor = None
            self._initialization_errors.append(f"Status tracker initialization failed: {str(e)}")

        # Enhanced response generator that uses all model types
        try:
            logger.info("Initializing enhanced response generator...")
            self.response_generator = ModelResponseGenerator(
                openai_client=self.openai_client,
                local_trainer=self.local_trainer,
                eleuther_client=self.eleuther_client
            )
            logger.info("Enhanced response generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced response generator: {e}")
            self.response_generator = None
            self._initialization_errors.append(f"Enhanced response generator initialization failed: {str(e)}")

    def _setup_integrations(self):
        """Set up inter-component links and integrations."""
        try:
            # Link status tracker to response generator if available
            if self.response_generator and self.status_tracker:
                if hasattr(self.response_generator, 'set_status_tracker'):
                    self.response_generator.set_status_tracker(self.status_tracker)

            # Ensure models directory exists
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Models directory: {MODELS_DIR}")

        except Exception as e:
            logger.error(f"Integration setup failed: {e}")
            self._initialization_errors.append(f"Integration setup failed: {str(e)}")

    def _validate_setup(self):
        """Validate the overall setup and log capabilities."""
        capabilities = {
            "azure_training": bool(self.azure_ml_client and self.azure_ml_client.is_available()),
            "local_training": bool(self.local_trainer and self.local_trainer.is_available()),
            "openai_inference": bool(self.openai_client and self.openai_client.is_available()),
            "eleuther_inference": bool(self.eleuther_client and self.eleuther_client.is_available()),
            "hybrid_models": bool(ENABLE_HYBRID_MODELS and self.response_generator),
            "data_processing": bool(self.data_processor),
            "status_tracking": bool(self.status_tracker),
            "azure_monitoring": bool(self.azure_monitor),
        }

        logger.info("Enhanced Trainer Capabilities:")
        for capability, available in capabilities.items():
            status = "AVAILABLE" if available else "NOT AVAILABLE"
            logger.info(f"   {capability.replace('_', ' ').title()}: {status}")

        # Determine if trainer is minimally functional
        minimal_requirements = ["data_processing", "status_tracking"]
        self.is_functional = all(capabilities.get(req, False) for req in minimal_requirements)

        # Check if at least one inference model is available
        inference_models = ["openai_inference", "eleuther_inference", "local_training"]
        has_inference_model = any(capabilities.get(model, False) for model in inference_models)
        
        if not has_inference_model:
            logger.warning("No inference models available")
            self._initialization_errors.append("No inference models available")

        if not self.is_functional:
            logger.error("Trainer is not functional - missing critical components")
        else:
            logger.info("Enhanced trainer is functional")

    def is_available(self) -> bool:
        """Check if the trainer is available and functional."""
        return self.is_functional

    def train_on_documents(
        self,
        texts: List[str],
        document_name: str,
        use_azure: bool = False,
        compute_target: Optional[str] = None,
        base_model_type: Optional[str] = None,
    ) -> str:
        """
        Enhanced training orchestrator with support for different base model types.
        
        Args:
            texts: List of text documents to train on
            document_name: Name for the training session
            use_azure: Whether to use Azure ML for training
            compute_target: Azure compute target (if using Azure)
            base_model_type: Base model type ("dialogpt" or "eleuther")
        
        Returns:
            training_id that's used for retrieving index later
        """
        if not self.is_available():
            raise RuntimeError("Trainer is not available - check initialization errors")

        training_id = str(uuid.uuid4())
        
        # Use provided base model type or default
        model_type = base_model_type or LOCAL_BASE_MODEL_TYPE

        # Validate inputs
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Invalid text inputs - must be non-empty strings")

        if not document_name or not document_name.strip():
            raise ValueError("Document name is required")

        try:
            logger.info(f"Starting enhanced training session: {training_id}")
            logger.info(f"   Documents: {document_name}")
            logger.info(f"   Base model type: {model_type}")
            platform_used = "Azure ML" if use_azure else "Local"
            logger.info(f"   Platform: {platform_used}")
            logger.info(f"   Text samples: {len(texts)}")
            total_chars = sum(len(text) for text in texts)
            logger.info(f"   Total characters: {total_chars:,}")

            # Create progress callback
            progress_callback = TrainingProgressCallback(self.status_tracker, training_id)
            self.status_tracker.start_training(training_id, document_name)

            # Enhanced Azure validation
            if use_azure:
                if not self.azure_ml_client or not self.azure_ml_client.is_available():
                    raise RuntimeError("Azure ML not available - check configuration and credentials")

                # Validate compute target and estimate costs
                try:
                    doc_ids = [1]  # Placeholder - in real implementation, would pass actual doc IDs
                    validation = self.azure_ml_client.validate_compute_training(doc_ids)
                    if not validation.get("valid"):
                        reason = validation.get("reason", "Unknown error")
                        raise RuntimeError(f"Azure validation failed: {reason}")

                    if compute_target and compute_target != validation.get("compute_target"):
                        logger.warning(
                            f"Requested compute {compute_target} differs from recommended {validation.get('compute_target')}"
                        )

                    if not compute_target:
                        compute_target = validation.get("compute_target")
                        logger.info(f"Using recommended compute target: {compute_target}")

                except Exception as e:
                    raise RuntimeError(f"Azure compute validation failed: {str(e)}")

            # Prepare data: chunk text, embed, build FAISS index, write to disk
            logger.info("Processing training data...")
            progress_callback(True, 10, "Processing documents...")

            try:
                prep_result = self.data_processor.prepare_training_data(texts, training_id)
                index_path = prep_result["index_path"]
                metadata_path = prep_result["metadata_path"]
                num_chunks = prep_result["num_chunks"]

                logger.info(f"Prepared {num_chunks} chunks for training")
                progress_callback(True, 30, f"Prepared {num_chunks} chunks")

            except Exception as e:
                self.status_tracker.fail_training(training_id, f"Data preparation failed: {str(e)}")
                raise RuntimeError(f"Data preparation failed: {str(e)}")

            # Choose training method: Azure vs. local
            if use_azure:
                return self._fine_tune_on_azure(
                    index_path, metadata_path, training_id, progress_callback, compute_target, model_type
                )
            else:
                return self._fine_tune_locally(
                    index_path, metadata_path, training_id, progress_callback, model_type
                )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if hasattr(self, 'status_tracker') and self.status_tracker:
                self.status_tracker.fail_training(training_id, str(e))
            raise

    def _fine_tune_locally(
        self, 
        index_path: str, 
        metadata_path: str, 
        training_id: str, 
        progress_cb: Any,
        base_model_type: str
    ) -> str:
        """Enhanced local fine-tuning with support for different base model types."""
        try:
            logger.info(f"Starting local fine-tuning for {training_id} with base model: {base_model_type}")

            # Create trainer with specified base model type
            if base_model_type != LOCAL_BASE_MODEL_TYPE:
                logger.info(f"Creating trainer with base model type: {base_model_type}")
                local_trainer = LocalModelTrainer(
                    training_id=training_id,
                    base_model_type=base_model_type
                )
            else:
                local_trainer = self.local_trainer
                local_trainer.training_id = training_id

            if not local_trainer or not local_trainer.is_available():
                raise RuntimeError("Enhanced local trainer not available")

            # Load chunk metadata
            try:
                with open(metadata_path, "rb") as f:
                    chunks: List[str] = pickle.load(f)
                logger.info(f"Loaded {len(chunks)} text chunks")
            except Exception as e:
                raise RuntimeError(f"Failed to load chunk metadata: {e}")

            # Begin local training
            progress_cb(True, 40, f"Starting local fine-tuning with {base_model_type}...")

            try:
                model_dir = local_trainer.train_model(metadata_path, training_id, progress_cb)
                logger.info(f"Local model training completed: {model_dir}")
            except Exception as e:
                raise RuntimeError(f"Local training failed: {e}")

            # Update the main trainer's local trainer if we used a different one
            if base_model_type != LOCAL_BASE_MODEL_TYPE:
                self.local_trainer = local_trainer

            # Mark training as complete
            self.status_tracker.complete_training(training_id)
            logger.info(f"Local training completed successfully: {training_id}")

            return training_id

        except Exception as e:
            logger.error(f"Local fine-tuning failed: {e}")
            raise

    def _fine_tune_on_azure(
        self, 
        index_path: str, 
        metadata_path: str, 
        training_id: str, 
        progress_cb: Any, 
        compute_target: str,
        base_model_type: str
    ) -> str:
        """Enhanced Azure ML training with support for different base model types."""
        if not (self.azure_ml_client and self.azure_ml_client.is_available()):
            raise RuntimeError("Azure ML not available or not configured")

        try:
            logger.info(f"Starting Azure ML training for {training_id}")
            logger.info(f"   Compute target: {compute_target}")
            logger.info(f"   Base model type: {base_model_type}")

            # Validate compute target exists and is ready
            try:
                if hasattr(self.azure_ml_client, 'ml_client'):
                    compute_info = self.azure_ml_client.ml_client.compute.get(compute_target)
                    if compute_info.provisioning_state != "Succeeded":
                        raise RuntimeError(f"Compute target {compute_target} not ready: {compute_info.provisioning_state}")
                    logger.info(f"Compute target validated: {compute_target}")
            except Exception as e:
                raise RuntimeError(f"Compute target validation failed: {str(e)}")

            # Prepare files for Azure ML
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            target_dir = MODELS_DIR / "indexes" / training_id
            target_dir.mkdir(parents=True, exist_ok=True)
            remote_index = target_dir / "faiss.index"
            remote_metadata = target_dir / "chunks_metadata.pkl"

            try:
                # Copy files to the models directory
                Path(remote_index).write_bytes(Path(index_path).read_bytes())
                Path(remote_metadata).write_bytes(Path(metadata_path).read_bytes())
                logger.info(f"Prepared files for Azure ML: {remote_index.name}, {remote_metadata.name}")
            except Exception as e:
                raise RuntimeError(f"Failed to prepare files for Azure: {e}")

            progress_cb(True, 50, "Registering data assets in Azure ML...")

            # Register data assets
            try:
                from azure.ai.ml.entities import Data
                
                # Register metadata file
                metadata_data = Data(
                    path=str(remote_metadata),
                    type="uri_file",
                    name=f"chunks_metadata_{training_id}",
                    version="1"
                )
                logger.info(f"Creating metadata Data asset for: {remote_metadata}")
                registered_metadata = self.azure_ml_client.ml_client.data.create_or_update(metadata_data)
                metadata_uri = registered_metadata.id
                logger.info(f"Registered metadata asset: {metadata_uri}")

                # Register FAISS index file
                index_data = Data(
                    path=str(remote_index),
                    type="uri_file",
                    name=f"faiss_index_{training_id}",
                    version="1"
                )
                logger.info(f"Creating FAISS index Data asset for: {remote_index}")
                registered_index = self.azure_ml_client.ml_client.data.create_or_update(index_data)
                index_uri = registered_index.id
                logger.info(f"Registered index asset: {index_uri}")
                
            except Exception as e:
                logger.error(f"Failed to register data assets: {e}")
                raise RuntimeError(f"Data registration failed: {e}")

            progress_cb(True, 70, "Preparing Azure training script...")

            # Create enhanced Azure training script
            script_content = f"""
import pickle
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    try:
        logger.info("Starting Azure ML enhanced training...")
        logger.info(f"Training ID: {training_id}")
        logger.info(f"Compute Target: {compute_target}")
        logger.info(f"Base Model Type: {base_model_type}")

        # Load chunks metadata
        metadata_file = "{remote_metadata.name}"
        logger.info(f"Loading metadata from: {{metadata_file}}")

        with open(metadata_file, "rb") as f:
            chunks = pickle.load(f)

        logger.info(f"Loaded {{len(chunks)}} chunks for training")

        # Begin model training with specified base model type
        try:
            from training.enhanced_local_model_trainer import LocalModelTrainer
            trainer = LocalModelTrainer(base_model_type="{base_model_type}")

            logger.info(f"Starting model training with {{trainer.base_model_type}} base model...")
            model_dir = trainer.train_model(
                metadata_file,
                "{training_id}",
                lambda training, progress, message: logger.info(f"Progress: {{progress}}% - {{message}}")
            )

            logger.info(f"Training completed! Model saved to: {{model_dir}}")
            logger.info("Azure ML enhanced training job finished successfully")

        except Exception as e:
            logger.error(f"Training error: {{e}}")
            raise

    except Exception as e:
        logger.error(f"Azure ML job failed: {{e}}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
"""

            script_path = MODELS_DIR / f"azure_train_{training_id}.py"
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script_content)
                logger.info(f"Created Azure training script: {script_path.name}")
            except Exception as e:
                raise RuntimeError(f"Failed to create training script: {e}")

            progress_cb(True, 90, "Submitting job to Azure ML...")

            # Submit via enhanced AzureMLClient
            try:
                job_name = self.azure_ml_client.submit_training_job(
                    script_path=str(script_path),
                    index_path=str(remote_index),
                    metadata_path=str(remote_metadata),
                    compute_target=compute_target,
                    training_id=training_id,
                )

                # Update database with Azure job info
                update_training_history_azure_job(
                    training_id,
                    job_name,
                    "medical-chatbot-training"
                )

                logger.info(f"Azure ML job submitted: {job_name}")

            except Exception as e:
                raise RuntimeError(f"Failed to submit Azure job: {e}")

            # Start monitoring the job
            if self.azure_monitor:
                try:
                    self.azure_monitor.start_monitoring(job_name, training_id)
                    logger.info(f"Started monitoring Azure job: {job_name}")
                except Exception as e:
                    logger.warning(f"Failed to start job monitoring: {e}")

            return training_id

        except Exception as e:
            logger.error(f"Azure training submission failed: {e}")
            raise

    def generate_response(
        self,
        message: str,
        session_id: str,
        preferred_model: str = None
    ) -> Dict[str, Any]:
        """
        Enhanced response generation with multi-model support.
        """
        if not self.is_available():
            return {
                "reply": "I'm sorry, the AI assistant is not properly initialized. Please check the system configuration.",
                "source_documents": [],
                "error": "Trainer not available"
            }

        try:
            # Validate inputs
            if not message or not message.strip():
                return {
                    "reply": "I didn't receive a valid message. Please try again.",
                    "source_documents": []
                }

            if not session_id:
                session_id = str(uuid.uuid4())
                logger.warning("No session ID provided, generated new one")

            user_query = message.strip()
            logger.info(f"Generating response for query: {user_query[:100]}... (model: {preferred_model or self.model_preference})")

            # Fetch conversation history from the database
            try:
                raw_history = fetch_conversation_history(session_id)
                history: List[Dict[str, str]] = []

                # Process raw history into a cleaner format
                for row in raw_history:
                    try:
                        if isinstance(row, dict):
                            role = row.get("role")
                            content = row.get("content")
                        elif isinstance(row, (list, tuple)) and len(row) >= 2:
                            role = row[0] if len(row) > 0 else None
                            content = row[1] if len(row) > 1 else None
                        else:
                            continue

                        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                            history.append({"role": role, "content": content.strip()})
                    except Exception as e:
                        logger.warning(f"Error processing history row: {e}")
                        continue

                logger.info(f"Retrieved {len(history)} conversation turns")

            except Exception as e:
                logger.warning(f"Error fetching conversation history: {e}")
                history = []

            # Get relevant context from FAISS index
            training_id = None
            relevant_chunks: List[str] = []
            context = None

            if self.status_tracker:
                try:
                    training_id = self.status_tracker.get_latest_training_name()
                    if training_id and self.data_processor and hasattr(self.data_processor, 'query_index'):
                        logger.info(f"Querying index for training_id: {training_id}")
                        relevant_chunks = self.data_processor.query_index(
                            user_query, training_id, top_k=5
                        )
                        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
                        
                        if relevant_chunks:
                            context = "\n\n".join(relevant_chunks)
                except Exception as e:
                    logger.warning(f"FAISS query failed: {e}")

            # Use enhanced response generator
            if self.response_generator:
                try:
                    response_result = self.response_generator.generate_response(
                        message=user_query,
                        session_id=session_id,
                        preferred_model=preferred_model,
                        context=context
                    )
                    
                    # Create source_documents list for frontend display
                    source_documents = []
                    if relevant_chunks:
                        for idx, chunk in enumerate(relevant_chunks):
                            chunk_id = f"{training_id}_chunk_{idx}" if training_id else f"chunk_{idx}"
                            chunk_name = f"{training_id} (chunk #{idx + 1})" if training_id else f"Document chunk #{idx + 1}"
                            source_documents.append({
                                "id": chunk_id,
                                "name": chunk_name,
                                "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
                            })
                    
                    # Enhance response with additional info
                    response_result.update({
                        "source_documents": source_documents,
                        "session_id": session_id,
                        "training_id": training_id,
                        "context_used": bool(context),
                        "chunks_retrieved": len(relevant_chunks)
                    })
                    
                    logger.info(f"Enhanced response generated successfully ({len(response_result.get('reply', ''))} chars, "
                              f"model: {response_result.get('model_used', 'unknown')}, "
                              f"confidence: {response_result.get('confidence', 0):.2f})")
                    
                    return response_result
                    
                except Exception as e:
                    logger.error(f"Enhanced response generator failed: {e}")
                    # Fall back to basic response
                    
            # Fallback response generation
            logger.warning("Using fallback response generation")
            return {
                "reply": "I'm here to help with your medical questions. Could you please rephrase your question?",
                "source_documents": [],
                "model_used": "fallback",
                "confidence": 0.5,
                "session_id": session_id,
                "error": "Enhanced response generator not available"
            }

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return {
                "reply": "I'm sorry, I encountered an unexpected error. Please try again.",
                "source_documents": [],
                "model_used": "error",
                "confidence": 0.0,
                "session_id": session_id,
                "error": str(e)
            }

    def switch_model_preference(self, new_preference: str):
        """Switch the preferred model dynamically."""
        try:
            if self.response_generator:
                self.response_generator.set_model_preference(new_preference)
                self.model_preference = new_preference
                logger.info(f"Model preference switched to: {new_preference}")
            else:
                raise RuntimeError("Response generator not available")
        except Exception as e:
            logger.error(f"Failed to switch model preference: {e}")
            raise

    def get_available_models(self) -> Dict[str, Any]:
        """Get information about all available models."""
        if self.response_generator:
            return self.response_generator.get_available_models()
        else:
            return {
                "error": "Response generator not available",
                "available_models": {}
            }

    def get_model_health_status(self) -> Dict[str, Any]:
        """Get health status of all models."""
        if self.response_generator:
            return self.response_generator.get_model_health_status()
        else:
            return {
                "error": "Response generator not available",
                "health_status": {}
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Return comprehensive system health overview."""
        try:
            status = {
                "azure_ml": {
                    "available": bool(self.azure_ml_client and self.azure_ml_client.is_available()),
                    "configured": bool(self.ml_client),
                },
                "openai": {
                    "available": bool(self.openai_client and self.openai_client.is_available()),
                    "configured": bool(OPENAI_API_KEY),
                },
                "eleuther": {
                    "available": bool(self.eleuther_client and self.eleuther_client.is_available()),
                    "model_name": ELEUTHER_MODEL_NAME,
                },
                "local_model": self.local_trainer.get_model_info() if (self.local_trainer and self.local_trainer.is_available()) else {"available": False},
                "hybrid_models": {
                    "enabled": ENABLE_HYBRID_MODELS,
                    "available": bool(self.response_generator and ENABLE_HYBRID_MODELS),
                },
                "training": self.get_training_status(),
                "data_processor": {"available": bool(self.data_processor)},
                "status_tracker": {"available": bool(self.status_tracker)},
                "model_preference": self.model_preference,
                "initialization_errors": self._initialization_errors,
                "is_functional": self.is_functional,
            }

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """Enhanced cleanup with proper resource management."""
        try:
            logger.info("Starting enhanced trainer cleanup...")

            # Stop Azure job monitoring
            if hasattr(self, 'azure_monitor') and self.azure_monitor:
                try:
                    if hasattr(self.azure_monitor, 'get_active_jobs'):
                        for job_name in self.azure_monitor.get_active_jobs():
                            self.azure_monitor.stop_monitoring(job_name)
                    logger.info("Azure monitoring stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Azure monitoring: {e}")

            # Clean up local trainer resources
            if hasattr(self, 'local_trainer') and self.local_trainer:
                try:
                    self.local_trainer.cleanup()
                    logger.info("Local trainer cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up local trainer: {e}")

            # Clean up EleutherAI client resources
            if hasattr(self, 'eleuther_client') and self.eleuther_client:
                try:
                    self.eleuther_client.cleanup()
                    logger.info("EleutherAI client cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up EleutherAI client: {e}")

            # Clean up data processor resources
            if hasattr(self, 'data_processor') and self.data_processor:
                try:
                    if hasattr(self.data_processor, 'cleanup'):
                        self.data_processor.cleanup()
                    logger.info("Data processor cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up data processor: {e}")

            logger.info("Enhanced trainer cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


# Convenience function for backward compatibility
def create_enhanced_trainer() -> MedicalChatbotTrainer:
    """Create and return an Enhanced Medical Chatbot Trainer instance."""
    return MedicalChatbotTrainer()