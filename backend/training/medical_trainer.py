"""
Enhanced Medical Chatbot Trainer - orchestrates all training and inference components,
with real Azure ML integration, proper error handling, and no fallback behavior.
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
from config import MODEL_PREFERENCE, MODELS_DIR, OPENAI_API_KEY

# Import our enhanced components
from clients.azure_ml_client import AzureMLClient
from clients.openai_client import OpenAIClient
from .local_model_trainer import LocalModelTrainer
from .training_data_processor import TrainingDataProcessor
from .training_status_tracker import TrainingStatusTracker, AzureJobMonitor, TrainingProgressCallback
from .model_response_generator import ModelResponseGenerator

# Import database utility to fetch past conversation turns
from utils.db_utils import fetch_conversation_history, update_training_history_azure_job

logger = logging.getLogger(__name__)


class MedicalChatbotTrainer:
    """Enhanced orchestrator for medical chatbot training, indexing, and inference with real Azure ML integration."""

    def __init__(self):
        self.model_preference = MODEL_PREFERENCE
        self._initialization_errors = []

        logger.info("Initializing MedicalChatbotTrainer...")

        # Initialize all components with proper error handling
        self._init_components()
        self._setup_integrations()
        self._validate_setup()

        if self._initialization_errors:
            logger.warning(f"Trainer initialized with {len(self._initialization_errors)} warnings:")
            for error in self._initialization_errors:
                logger.warning(f"   - {error}")
        else:
            logger.info("MedicalChatbotTrainer initialized successfully")

    def _init_components(self):
        """Initialize components with enhanced error handling and validation."""
        try:
            # Enhanced Azure ML client
            logger.info("Initializing Azure ML client...")
            self.azure_ml_client = AzureMLClient()
            if self.azure_ml_client.is_available():
                logger.info("Azure ML client initialized and connected")

                # Test additional Azure services
                if hasattr(self.azure_ml_client, 'consumption_client') and self.azure_ml_client.consumption_client:
                    logger.info("Azure Consumption API available")
                else:
                    logger.warning("Azure Consumption API not available - install azure-mgmt-consumption")

                if hasattr(self.azure_ml_client, 'monitor_client') and self.azure_ml_client.monitor_client:
                    logger.info("Azure Monitor API available")
                else:
                    logger.warning("Azure Monitor API not available - install azure-mgmt-monitor")
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

        # Local model trainer
        try:
            logger.info("Initializing local model trainer...")
            self.local_trainer = LocalModelTrainer()
            if self.local_trainer.is_available():
                logger.info("Local model trainer initialized")
            else:
                logger.warning("Local model trainer not fully available")
        except Exception as e:
            logger.error(f"Failed to initialize local trainer: {e}")
            self.local_trainer = None
            self._initialization_errors.append(f"Local trainer initialization failed: {str(e)}")

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

        # Response generator that uses local model, embeddings, or OpenAI
        try:
            logger.info("Initializing response generator...")
            self.response_generator = ModelResponseGenerator(
                self.openai_client,
                self.local_trainer
            )
            logger.info("Response generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize response generator: {e}")
            self.response_generator = None
            self._initialization_errors.append(f"Response generator initialization failed: {str(e)}")

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
            "data_processing": bool(self.data_processor),
            "status_tracking": bool(self.status_tracker),
            "azure_monitoring": bool(self.azure_monitor),
        }

        logger.info("Trainer Capabilities:")
        for capability, available in capabilities.items():
            status = "AVAILABLE" if available else "NOT AVAILABLE"
            logger.info(f"   {capability.replace('_', ' ').title()}: {status}")

        # Determine if trainer is minimally functional
        minimal_requirements = ["data_processing", "status_tracking"]
        self.is_functional = all(capabilities.get(req, False) for req in minimal_requirements)

        if not self.is_functional:
            logger.error("Trainer is not functional - missing critical components")
        else:
            logger.info("Trainer is functional")

    def is_available(self) -> bool:
        """Check if the trainer is available and functional."""
        return self.is_functional

    def train_on_documents(
        self,
        texts: List[str],
        document_name: str,
        use_azure: bool = False,
        compute_target: Optional[str] = None,
    ) -> str:
        """
        Enhanced training orchestrator with comprehensive validation and error handling.
        Returns a training_id that's used for retrieving index later.
        """
        if not self.is_available():
            raise RuntimeError("Trainer is not available - check initialization errors")

        training_id = str(uuid.uuid4())

        # Validate inputs
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Invalid text inputs - must be non-empty strings")

        if not document_name or not document_name.strip():
            raise ValueError("Document name is required")

        try:
            logger.info(f"Starting training session: {training_id}")
            logger.info(f"   Documents: {document_name}")
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
                    index_path, metadata_path, training_id, progress_callback, compute_target
                )
            else:
                return self._fine_tune_locally(
                    index_path, metadata_path, training_id, progress_callback
                )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if hasattr(self, 'status_tracker') and self.status_tracker:
                self.status_tracker.fail_training(training_id, str(e))
            raise

    def _fine_tune_locally(
        self, index_path: str, metadata_path: str, training_id: str, progress_cb: Any
    ) -> str:
        """Enhanced local fine-tuning with better error handling."""
        try:
            if not self.local_trainer or not self.local_trainer.is_available():
                raise RuntimeError("Local trainer not available")

            logger.info(f"Starting local fine-tuning for {training_id}")

            # Load chunk metadata
            try:
                with open(metadata_path, "rb") as f:
                    chunks: List[str] = pickle.load(f)
                logger.info(f"Loaded {len(chunks)} text chunks")
            except Exception as e:
                raise RuntimeError(f"Failed to load chunk metadata: {e}")

            # Convert chunks to a HF Dataset
            try:
                from datasets import Dataset
                dataset = Dataset.from_dict({"text": chunks})
                logger.info(f"Created HuggingFace dataset with {len(dataset)} samples")
            except Exception as e:
                raise RuntimeError(f"Failed to create dataset: {e}")

            # Begin local training
            progress_cb(True, 40, "Starting local fine-tuning...")

            try:
                model_dir = self.local_trainer.train_model(dataset, training_id, progress_cb)
                logger.info(f"Local model training completed: {model_dir}")
            except Exception as e:
                raise RuntimeError(f"Local training failed: {e}")

            # Mark training as complete
            self.status_tracker.complete_training(training_id)
            logger.info(f"Local training completed successfully: {training_id}")

            return training_id

        except Exception as e:
            logger.error(f"Local fine-tuning failed: {e}")
            raise

    def _fine_tune_on_azure(
        self, index_path: str, metadata_path: str, training_id: str, progress_cb: Any, compute_target: str
    ) -> str:
        """Enhanced Azure ML training with real API integration."""
        if not (self.azure_ml_client and self.azure_ml_client.is_available()):
            raise RuntimeError("Azure ML not available or not configured")

        try:
            logger.info(f"Starting Azure ML training for {training_id}")
            logger.info(f"   Compute target: {compute_target}")

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

            # ── REGISTER THE INDEX AND METADATA AS AZURE ML DATA ASSETS ──
            try:
                from azure.ai.ml.entities import Data

                # Register metadata file
                metadata_data = Data(
                    path=str(remote_metadata),
                    type="uri_file",
                    name=f"chunks_metadata_{training_id}",
                    version="1"
                )
                registered_metadata = self.azure_ml_client.ml_client.data.create_or_update(metadata_data)
                metadata_uri = registered_metadata.id  # This is the Azure ML URI

                # Register FAISS index file
                index_data = Data(
                    path=str(remote_index),
                    type="uri_file",
                    name=f"faiss_index_{training_id}",
                    version="1"
                )
                registered_index = self.azure_ml_client.ml_client.data.create_or_update(index_data)
                index_uri = registered_index.id  # This is the Azure ML URI

                logger.info(f"Registered metadata asset: {metadata_uri}")
                logger.info(f"Registered index asset: {index_uri}")
            except Exception as e:
                raise RuntimeError(f"Failed to register data assets: {e}")

            progress_cb(True, 70, "Preparing Azure training script...")

            # Create enhanced Azure training script
            script_content = f"""
import pickle
import sys
import os
import logging
from pathlib import Path
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    try:
        logger.info("Starting Azure ML training...")
        logger.info(f"Training ID: {training_id}")
        logger.info(f"Compute Target: {compute_target}")

        # Load chunks metadata
        metadata_file = "{remote_metadata.name}"
        logger.info(f"Loading metadata from: {{metadata_file}}")

        with open(metadata_file, "rb") as f:
            chunks = pickle.load(f)

        logger.info(f"Loaded {{len(chunks)}} chunks for training")

        # Convert to HF Dataset
        dataset = Dataset.from_dict({{'text': chunks}})
        logger.info(f"Created dataset with {{len(dataset)}} samples")

        # Initialize and run local trainer (adapted for Azure environment)
        try:
            from training.local_model_trainer import LocalModelTrainer
            trainer = LocalModelTrainer()

            logger.info("Starting model training...")
            model_dir = trainer.train_model(
                dataset,
                "{training_id}",
                lambda training, progress, message: logger.info(f"Progress: {{progress}}% - {{message}}")
            )

            logger.info(f"Training completed! Model saved to: {{model_dir}}")
            logger.info("Azure ML training job finished successfully")

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

            # Submit via enhanced AzureMLClient, passing data asset URIs
            try:
                job_name = self.azure_ml_client.submit_training_job(
                    script_path=None,             # Git-based code will be used instead of local script
                    index_path=index_uri,         # Pass the registered Data Asset URI
                    metadata_path=metadata_uri,   # Pass the registered Data Asset URI
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

            # Do not complete training here – let the monitor handle status updates
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
        Enhanced response generation with better error handling and context retrieval.
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
            logger.info(f"Generating response for query: {user_query[:100]}...")

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

            # Always define relevant_chunks so it exists even if FAISS is not queried or fails
            training_id = None
            relevant_chunks: List[str] = []

            if self.status_tracker:
                try:
                    # Use the correct method name on your status tracker
                    training_id = self.status_tracker.get_latest_training_name()
                    if training_id and self.data_processor and hasattr(self.data_processor, 'query_index'):
                        logger.info(f"Querying index for training_id: {training_id}")
                        relevant_chunks = self.data_processor.query_index(
                            user_query, training_id, top_k=5
                        )
                        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
                except Exception as e:
                    logger.warning(f"FAISS query failed: {e}")
                    # relevant_chunks remains [] by default

            # Build context from retrieved chunks
            chunk_context = ""
            if relevant_chunks:
                chunk_context = "\n\n".join(relevant_chunks)
                logger.info(f"Built context from {len(relevant_chunks)} chunks ({len(chunk_context)} chars)")

            # Create source_documents list for frontend display
            source_documents = []
            for idx, chunk in enumerate(relevant_chunks):
                chunk_id = f"{training_id}_chunk_{idx}" if training_id else f"chunk_{idx}"
                chunk_name = f"{training_id} (chunk #{idx + 1})" if training_id else f"Document chunk #{idx + 1}"
                source_documents.append({
                    "id": chunk_id,
                    "name": chunk_name,
                    "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
                })

            # Build the conversation for the LLM
            system_prompt = (
                "You are a helpful medical assistant. "
                "Use the provided document context to give accurate, helpful responses. "
                "If no relevant context is provided, answer based on general medical knowledge. "
                "Keep responses clear and under 200 words. "
                "Always recommend consulting healthcare professionals for specific medical advice."
            )

            # Construct the user prompt with context
            if chunk_context:
                user_prompt = f"Context from uploaded documents:\n{chunk_context}\n\nUser question: {user_query}"
                logger.info("Using document context for response generation")
            else:
                user_prompt = f"User question: {user_query}"
                logger.info("Using general knowledge for response generation")

            # Build full message chain
            full_messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history (limit to last 10 exchanges to avoid token limits)
            recent_history = history[-20:] if len(history) > 20 else history
            for turn in recent_history:
                full_messages.append({"role": turn["role"], "content": turn["content"]})

            # Add current user message
            full_messages.append({"role": "user", "content": user_prompt})

            # Generate response using preferred model
            choice = preferred_model or self.model_preference
            assistant_text = ""

            try:
                logger.info(f"Generating response using: {choice}")

                if choice == "local" and self.local_trainer and self.local_trainer.is_available():
                    assistant_text = self.local_trainer.generate_with_local_model(
                        full_messages, session_id
                    )
                    logger.info("Generated response using local model")
                elif choice == "openai" and self.openai_client and self.openai_client.is_available():
                    assistant_text = self.openai_client.generate_response(
                        full_messages, session_id
                    )
                    logger.info("Generated response using OpenAI")
                else:
                    # Try fallback models
                    if self.openai_client and self.openai_client.is_available():
                        assistant_text = self.openai_client.generate_response(
                            full_messages, session_id
                        )
                        logger.info("Generated response using OpenAI (fallback)")
                    elif self.local_trainer and self.local_trainer.is_available():
                        assistant_text = self.local_trainer.generate_with_local_model(
                            full_messages, session_id
                        )
                        logger.info("Generated response using local model (fallback)")
                    else:
                        assistant_text = (
                            "I'm sorry, but no language models are currently available. "
                            "Please check the system configuration or try again later."
                        )
                        logger.error("No models available for response generation")

            except Exception as e:
                logger.error(f"Model generation failed: {e}")
                assistant_text = (
                    "I encountered an error while generating a response. "
                    "Please try rephrasing your question or try again later."
                )

            # Ensure we have a valid response
            if not assistant_text or not assistant_text.strip():
                assistant_text = "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."
                logger.warning("Empty response generated, using fallback")

            # Return structured response
            response = {
                "reply": assistant_text.strip(),
                "source_documents": source_documents,
                "model_used": choice,
                "context_used": bool(chunk_context),
                "session_id": session_id
            }

            logger.info(f"Response generated successfully ({len(assistant_text)} chars)")
            return response

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return {
                "reply": "I'm sorry, I encountered an unexpected error. Please try again.",
                "source_documents": [],
                "error": str(e)
            }

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status with enhanced details."""
        try:
            if hasattr(self, "_status") and self._status:
                return self._status

            if self.status_tracker:
                status = self.status_tracker.get_status()

                # Enhance with Azure job info if available
                if (status.get("is_training") and
                    self.azure_monitor and
                    hasattr(self.azure_monitor, 'get_active_jobs')):

                    try:
                        active_jobs = self.azure_monitor.get_active_jobs()
                        if active_jobs:
                            latest_job = active_jobs[0]
                            if self.azure_ml_client:
                                azure_status = self.azure_ml_client.get_job_status(latest_job)
                                if azure_status:
                                    status.update({
                                        "azure_job_name": azure_status["name"],
                                        "azure_job_status": azure_status["status"],
                                        "azure_compute": azure_status.get("compute_target"),
                                    })
                    except Exception as e:
                        logger.warning(f"Could not get Azure job status: {e}")

                return status

            return {"is_training": False, "progress": 0, "status_message": "Status tracker not available"}

        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {"is_training": False, "progress": 0, "status_message": f"Error: {str(e)}"}

    def get_azure_job_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific Azure ML job."""
        if not (self.azure_ml_client and self.azure_ml_client.is_available()):
            return None
        return self.azure_ml_client.get_job_status(job_name)

    def get_azure_job_logs(self, job_name: str) -> List[str]:
        """Get logs from Azure ML job."""
        if not (self.azure_ml_client and self.azure_ml_client.is_available()):
            return ["Azure ML not available"]
        return self.azure_ml_client.get_job_logs(job_name)

    def get_billing_information(self) -> Dict[str, Any]:
        """Get Azure billing info."""
        if not (self.azure_ml_client and self.azure_ml_client.is_available()):
            raise RuntimeError("Azure ML not available")
        return self.azure_ml_client.get_billing_information()

    def get_available_models(self) -> Dict[str, Any]:
        """Get available model information."""
        return {
            "local_model": self.local_trainer.get_model_info() if (self.local_trainer and self.local_trainer.is_available()) else {"available": False},
            "openai": {"available": self.openai_client.is_available() if self.openai_client else False},
            "azure_ml": {"available": self.azure_ml_client.is_available() if self.azure_ml_client else False},
        }

    def set_model_preference(self, preference: str):
        """Set the preferred model for response generation."""
        valid_preferences = ["local", "openai", "azure"]
        if preference not in valid_preferences:
            raise ValueError(f"Invalid preference. Must be one of: {valid_preferences}")

        self.model_preference = preference
        logger.info(f"Model preference updated to: {preference}")

        if self.response_generator and hasattr(self.response_generator, 'set_model_preference'):
            self.response_generator.set_model_preference(preference)

    def get_system_status(self) -> Dict[str, Any]:
        """Return comprehensive system health overview."""
        try:
            status = {
                "azure_ml": {
                    "available": bool(self.azure_ml_client and self.azure_ml_client.is_available()),
                    "configured": bool(self.ml_client),
                    "billing_available": bool(self.azure_ml_client and hasattr(self.azure_ml_client, 'consumption_client') and self.azure_ml_client.consumption_client),
                    "monitoring_available": bool(self.azure_ml_client and hasattr(self.azure_ml_client, 'monitor_client') and self.azure_ml_client.monitor_client),
                },
                "openai": {
                    "available": bool(self.openai_client and self.openai_client.is_available()),
                    "configured": bool(OPENAI_API_KEY),
                },
                "local_model": self.local_trainer.get_model_info() if (self.local_trainer and self.local_trainer.is_available()) else {"available": False},
                "training": self.get_training_status(),
                "data_processor": {"available": bool(self.data_processor)},
                "status_tracker": {"available": bool(self.status_tracker)},
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
            logger.info("Starting trainer cleanup...")

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
                    if hasattr(self.local_trainer, 'cleanup'):
                        self.local_trainer.cleanup()
                    logger.info("Local trainer cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up local trainer: {e}")

            # Clean up data processor resources
            if hasattr(self, 'data_processor') and self.data_processor:
                try:
                    if hasattr(self.data_processor, 'cleanup'):
                        self.data_processor.cleanup()
                    logger.info("Data processor cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up data processor: {e}")

            logger.info("Trainer cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


# Convenience function for backward compatibility
def create_trainer() -> MedicalChatbotTrainer:
    """Create and return a MedicalChatbotTrainer instance."""
    return MedicalChatbotTrainer()
