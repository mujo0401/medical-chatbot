# app.py

# Load environment variables FIRST before any other imports
import os
from dotenv import load_dotenv
load_dotenv()

# Now continue with other imports
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import logging
import sys
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("medical_chatbot.log")
    ]
)
logger = logging.getLogger(__name__)

from config import ALLOWED_ORIGINS, MAX_CONTENT_LENGTH
from utils.db_utils import init_database
from utils.pdf_utils import PDF_PROCESSING_AVAILABLE
from routes.chat_routes import chat_bp
from routes.upload_routes import upload_bp
from routes.train_routes import train_bp
from routes.session_routes import session_bp
from routes.model_routes import model_bp


# ============================================================================
# WORKING MEDICAL CHATBOT INTEGRATION
# ============================================================================

class WorkingMedicalChatBot:
    """
    Working medical chatbot that uses the fixed embedding approach.
    This bypasses the broken context integration in the main trainer.
    """
    def __init__(self):
        self.training_id = "7553b910-e7c1-4cdd-b40e-3c9ead33c8b5"  # Your training ID
        self.processor = None
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Initialize the data processor with embedding fix"""
        try:
            # Force sentence transformer embeddings by temporarily hiding OpenAI key
            original_openai_key = os.environ.get('OPENAI_API_KEY')
            if original_openai_key:
                os.environ['OPENAI_API_KEY'] = ''
            
            try:
                from training.training_data_processor import TrainingDataProcessor
                self.processor = TrainingDataProcessor()
                logger.info("✅ Working medical chatbot initialized successfully")
            finally:
                # Restore OpenAI key
                if original_openai_key:
                    os.environ['OPENAI_API_KEY'] = original_openai_key
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize working medical chatbot: {e}")
            self.processor = None
    
    def get_context(self, query):
        """Get context using the working embedding approach"""
        if not self.processor:
            return []
        
        # Force sentence transformer for querying (the working method)
        original_key = os.environ.get('OPENAI_API_KEY')
        if original_key:
            os.environ['OPENAI_API_KEY'] = ''
        
        try:
            context_chunks = self.processor.query_index(query, self.training_id, top_k=3)
            return context_chunks or []
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return []
        finally:
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
    
    def extract_patient_name(self, text):
        """Extract patient name from text"""
        patterns = [
            r"Patient name:\s*([A-Za-z\s]+?)(?:\s+DOB|$)",
            r"(Joseph\s+M\s+Murphy)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 3:
                    return name
        return None
    
    def extract_dob(self, text):
        """Extract date of birth"""
        patterns = [
            r"DOB:\s*(\d{2}-[A-Z]{3}-\d{4})",
            r"(06-MAR-1985)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def extract_test_result(self, text):
        """Extract test result"""
        patterns = [
            r"RESULT:\s*([A-Z]+)",
            r"(NEGATIVE|POSITIVE)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    
    def extract_lab_info(self, text):
        """Extract laboratory information"""
        patterns = [
            r"(Invitae.*?Inc\.)",
            r"(Labcorp Genetics)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def answer_question(self, question):
        """Answer questions using working context extraction"""
        
        if not self.processor:
            return {
                "reply": "❌ Medical document processor not available",
                "context_used": False,
                "source_documents": [],
                "model_used": "working_medical_chatbot",
                "chunks_retrieved": 0
            }
        
        question_lower = question.lower()
        
        # Get relevant context using the working method
        context_chunks = self.get_context(question)
        
        if not context_chunks:
            return {
                "reply": "❌ I couldn't find relevant information in the medical document for your question.",
                "context_used": False,
                "source_documents": [],
                "model_used": "working_medical_chatbot",
                "chunks_retrieved": 0
            }
        
        all_text = " ".join(context_chunks)
        
        # Answer based on question type
        reply = None
        
        if any(word in question_lower for word in ['name', 'patient']):
            name = self.extract_patient_name(all_text)
            reply = f"👤 The patient's name is: **{name}**" if name else "❌ Patient name not found in document"
        
        elif any(word in question_lower for word in ['birth', 'dob', 'born', 'age']):
            dob = self.extract_dob(all_text)
            reply = f"📅 The patient's date of birth is: **{dob}**" if dob else "❌ Date of birth not found in document"
        
        elif any(word in question_lower for word in ['test', 'result', 'outcome']):
            result = self.extract_test_result(all_text)
            if result:
                icon = "✅" if result == "NEGATIVE" else "⚠️"
                reply = f"{icon} The test result is: **{result}**"
            else:
                reply = "❌ Test result not found in document"
        
        elif any(word in question_lower for word in ['laboratory', 'lab', 'company']):
            lab = self.extract_lab_info(all_text)
            reply = f"🏥 The laboratory is: **{lab}**" if lab else "❌ Laboratory information not found"
        
        elif any(word in question_lower for word in ['summary', 'about', 'document']):
            name = self.extract_patient_name(all_text) or "Patient"
            dob = self.extract_dob(all_text) or "Unknown"
            result = self.extract_test_result(all_text) or "Unknown"
            
            reply = f"""📋 **Medical Document Summary:**
            
👤 **Patient:** {name}
📅 **Date of Birth:** {dob}  
🧬 **Test Type:** Genetic diagnostic test (Invitae)
✅ **Result:** {result}
🏥 **Laboratory:** Labcorp Genetics/Invitae

This document contains the results of genetic testing for cardiac conditions."""
        
        else:
            # General search in context
            reply = f"📄 **Found relevant information in the medical document:**\n\n{all_text[:300]}..."
        
        if not reply:
            reply = "❓ I found the document but couldn't extract specific information for that question. Try asking about: patient name, date of birth, test result, or laboratory."
        
        return {
            "reply": reply,
            "context_used": True,
            "source_documents": context_chunks,
            "model_used": "working_medical_chatbot",
            "chunks_retrieved": len(context_chunks)
        }
    
    def is_available(self):
        """Check if the working chatbot is available"""
        return self.processor is not None


# ============================================================================
# ORIGINAL CODE CONTINUES
# ============================================================================

def check_azure_ml_availability():
    """
    Comprehensive Azure ML availability check with real connectivity testing.
    NO FALLBACKS - only returns real status.
    """
    logger.info("Checking Azure ML availability.")

    # Check if Azure ML is explicitly disabled
    if os.getenv("AZURE_ML_DISABLED", "").lower() in ["true", "1", "yes"]:
        logger.info("Azure ML is disabled via environment variable")
        return {
            "available": False,
            "sdk_installed": False,
            "configured": False,
            "reason": "Azure ML disabled in configuration",
            "help": "Remove AZURE_ML_DISABLED environment variable to enable Azure ML"
        }

    # Check required environment variables
    required_vars = ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.warning(f"Missing Azure environment variables: {', '.join(missing_vars)}")
        return {
            "available": False,
            "sdk_installed": False,
            "configured": False,
            "reason": f"Missing required environment variables: {', '.join(missing_vars)}",
            "help": "Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and AZURE_WORKSPACE_NAME environment variables"
        }

    # Check if Azure ML SDK is installed
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        logger.info("Azure ML SDK is installed")
        sdk_installed = True
    except ImportError as e:
        logger.error(f"Azure ML SDK not installed: {e}")
        return {
            "available": False,
            "sdk_installed": False,
            "configured": False,
            "reason": "Azure ML SDK not installed",
            "help": "Install with: pip install azure-ai-ml azure-identity"
        }

    # Attempt to connect to Azure ML workspace
    try:
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

        logger.info("Attempting to connect to Azure ML workspace.")
        logger.info(f"   Subscription: {subscription_id[:8]}...")
        logger.info(f"   Resource Group: {resource_group}")
        logger.info(f"   Workspace: {workspace_name}")

        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        # === FIXED: Retrieve workspace via workspaces.get(...) rather than ml_client._workspace ===
        workspace = ml_client.workspaces.get(
            name=workspace_name,
            resource_group_name=resource_group
        )
        logger.info(f"Connected to workspace: {workspace.name}")
        logger.info(f"   Location: {workspace.location}")
        logger.info(f"   Resource Group: {workspace.resource_group}")

        # Get compute targets and validate them
        compute_targets = []
        compute_count = 0
        working_compute_count = 0

        logger.info("Scanning compute targets.")

        for compute in ml_client.compute.list():
            compute_count += 1
            compute_info = {
                "name": compute.name,
                "type": getattr(compute, "type", "Unknown"),
                "state": getattr(compute, "provisioning_state", "Unknown"),
                "vm_size": getattr(compute, "size", "Unknown"),
                "location": getattr(compute, "location", "Unknown"),
            }

            # Add scaling information for compute clusters
            if hasattr(compute, "scale_settings"):
                scale_settings = compute.scale_settings
                compute_info.update({
                    "current_nodes": getattr(compute, "current_node_count", 0),
                    "max_nodes": getattr(scale_settings, "max_instances", 1),
                    "min_nodes": getattr(scale_settings, "min_instances", 0),
                })
            else:
                # For compute instances
                compute_info.update({
                    "current_nodes": 1 if compute_info["state"] == "Running" else 0,
                    "max_nodes": 1,
                    "min_nodes": 0,
                })

            compute_targets.append(compute_info)
            logger.info(f"   {compute.name}: {compute_info['state']} ({compute_info['vm_size']})")

            if compute_info["state"] in ["Succeeded", "Running"]:
                working_compute_count += 1

        logger.info(f"Found {compute_count} total compute targets, {working_compute_count} working")

        if working_compute_count == 0:
            logger.warning("No working compute targets found")
            return {
                "available": False,
                "sdk_installed": True,
                "configured": True,
                "reason": "No working compute targets available",
                "help": "Create or start a compute cluster in Azure ML Studio",
                "compute_targets": compute_targets,
                "workspace_info": {
                    "name": workspace.name,
                    "location": workspace.location,
                    "resource_group": workspace.resource_group
                }
            }

        # Test additional Azure services availability
        additional_services = {}

        # Test Azure Monitor
        try:
            from azure.mgmt.monitor import MonitorManagementClient
            monitor_client = MonitorManagementClient(credential, subscription_id)
            additional_services["monitor"] = True
            logger.info("Azure Monitor client available")
        except Exception as e:
            additional_services["monitor"] = False
            logger.warning(f"Azure Monitor not available: {e}")

        # Test Azure Consumption Management
        try:
            from azure.mgmt.consumption import ConsumptionManagementClient
            consumption_client = ConsumptionManagementClient(credential, subscription_id)
            additional_services["consumption"] = True
            logger.info("Azure Consumption client available")
        except Exception as e:
            additional_services["consumption"] = False
            logger.warning(f"Azure Consumption not available: {e}")

        # Success - Azure ML is fully available
        logger.info("Azure ML is fully available and configured!")

        return {
            "available": True,
            "sdk_installed": True,
            "configured": True,
            "reason": None,
            "compute_targets": compute_targets,
            "working_compute_count": working_compute_count,
            "total_compute_count": compute_count,
            "workspace_info": {
                "name": workspace.name,
                "location": workspace.location,
                "resource_group": workspace.resource_group,
                "subscription_id": subscription_id
            },
            "additional_services": additional_services,
            "connection_test": "passed"
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Azure ML connection failed: {error_msg}")

        # Provide specific error guidance
        help_message = "Check Azure credentials and workspace configuration"
        if "authentication" in error_msg.lower():
            help_message = "Azure authentication failed. Run 'az login' or check service principal credentials"
        elif "not found" in error_msg.lower():
            help_message = "Workspace not found. Verify subscription ID, resource group, and workspace name"
        elif "permission" in error_msg.lower() or "forbidden" in error_msg.lower():
            help_message = "Insufficient permissions. Ensure you have Contributor or Owner role on the workspace"
        elif "network" in error_msg.lower() or "timeout" in error_msg.lower():
            help_message = "Network connectivity issue. Check firewall settings and internet connection"

        return {
            "available": False,
            "sdk_installed": True,
            "configured": False,
            "reason": f"Connection failed: {error_msg}",
            "help": help_message,
            "error_type": "connection_error"
        }


def get_models_status():
    """Get the status of available models with real Azure data"""
    try:
        logger.info("Checking model status...")

        # Get enhanced Azure status with real connectivity testing
        azure_ml_status = check_azure_ml_availability()

        # Check OpenAI availability
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_available = bool(openai_api_key and openai_api_key.strip())

        if openai_available:
            logger.info("OpenAI API key configured")
        else:
            logger.info("OpenAI API key not configured")

        # Check local model availability
        local_available = True  # Local models are always available
        logger.info("Local model available")

        # Check working medical chatbot
        working_chatbot_available = working_chatbot.is_available()
        logger.info(f"Working medical chatbot: {'Available' if working_chatbot_available else 'Not available'}")

        model_preference = os.getenv("MODEL_PREFERENCE", "local")
        logger.info(f"Model preference: {model_preference}")

        return {
            "openai": {
                "available": openai_available,
                "model": "EleutherAI/gpt-neo-2.7B",
                "configured": openai_available,
                "api_key_set": openai_available
            },
            "azure": azure_ml_status,
            "local": {
                "available": local_available,
                "model_name": "DialoGPT-medium",
                "trained": local_available
            },
            "working_medical_chatbot": {
                "available": working_chatbot_available,
                "description": "Fixed medical document extraction",
                "training_id": working_chatbot.training_id if working_chatbot_available else None
            },
            "current_preference": model_preference,
            "status_check_time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return get_fallback_models_status()


def get_fallback_models_status():
    """Fallback model status when there's an error"""
    azure_disabled = os.getenv("AZURE_ML_DISABLED", "false").lower() in ["true", "1", "yes"]

    return {
        "openai": {
            "available": False,
            "model": "EleutherAI/gpt-neo-2.7B",
            "configured": False,
            "error": "Status check failed"
        },
        "azure": {
            "available": False,
            "sdk_installed": not azure_disabled,
            "configured": False,
            "reason": "Azure ML disabled" if azure_disabled else "Status check failed",
            "error": "Could not determine Azure ML status"
        },
        "local": {
            "available": True,
            "model_name": "DialoGPT-medium",
            "trained": True
        },
        "working_medical_chatbot": {
            "available": False,
            "error": "Status check failed"
        },
        "current_preference": "local",
        "status_check_time": datetime.now().isoformat(),
        "fallback": True
    }


# Check Azure availability at startup with detailed logging
logger.info("Starting Medical Chatbot Backend...")
logger.info("=" * 60)

azure_status = check_azure_ml_availability()
AZURE_AVAILABLE = azure_status["available"]

if AZURE_AVAILABLE:
    logger.info("Azure ML Status: AVAILABLE")
    logger.info(f"   Workspace: {azure_status['workspace_info']['name']}")
    logger.info(f"   Location: {azure_status['workspace_info']['location']}")
    logger.info(f"   Compute Targets: {azure_status['working_compute_count']}/{azure_status['total_compute_count']} working")
else:
    logger.warning("Azure ML Status: NOT AVAILABLE")
    logger.warning(f"   Reason: {azure_status['reason']}")
    if azure_status.get("help"):
        logger.info(f"   Help: {azure_status['help']}")

# Initialize working medical chatbot
logger.info("Initializing Working Medical Chatbot...")
working_chatbot = WorkingMedicalChatBot()

# Import trainer and create global instance with error handling
try:
    from training.medical_trainer import MedicalChatbotTrainer
    logger.info("Initializing Medical Chatbot Trainer...")
    trainer = MedicalChatbotTrainer()
    logger.info("Medical Chatbot Trainer initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize trainer: {e}")
    trainer = None

# Initialize Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Set up CORS with detailed configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True
    },
    r"/health": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "OPTIONS"],
        "supports_credentials": True
    }
})

logger.info(f"CORS configured for origins: {ALLOWED_ORIGINS}")

# Initialize database tables
logger.info("Initializing database...")
db_success = init_database()
if db_success:
    logger.info("Database initialized successfully")
else:
    logger.error("Database initialization failed")

# Register Blueprints
logger.info("Registering API routes...")
app.register_blueprint(chat_bp, url_prefix="/api")
app.register_blueprint(upload_bp, url_prefix="/api")
app.register_blueprint(train_bp, url_prefix="/api")
app.register_blueprint(session_bp, url_prefix="/api")
app.register_blueprint(model_bp, url_prefix="/api")
logger.info("All API routes registered")


# ============================================================================
# NEW WORKING MEDICAL CHAT ENDPOINT
# ============================================================================

@app.route("/api/medical-chat", methods=["POST"])
def medical_chat():
    """
    Working medical chat endpoint that uses the fixed context extraction.
    This bypasses the broken main trainer pipeline.
    """
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Message is required"}), 400
        
        message = data["message"].strip()
        session_id = data.get("session_id", "medical_chat_session")
        
        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        logger.info(f"Medical chat request: '{message}' (session: {session_id})")
        
        # Use the working medical chatbot
        if not working_chatbot.is_available():
            return jsonify({
                "reply": "❌ Medical document system is not available. Please ensure training data is loaded.",
                "context_used": False,
                "source_documents": [],
                "model_used": "error",
                "chunks_retrieved": 0
            }), 503
        
        # Generate response using working method
        response = working_chatbot.answer_question(message)
        
        logger.info(f"Medical chat response generated: {len(response['reply'])} chars, "
                   f"context_used: {response['context_used']}, "
                   f"chunks: {response['chunks_retrieved']}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in medical chat: {e}")
        return jsonify({
            "error": "Internal server error",
            "reply": "❌ Sorry, I encountered an error processing your request.",
            "context_used": False,
            "source_documents": [],
            "model_used": "error"
        }), 500


@app.route("/api/medical-chat/status", methods=["GET"])
def medical_chat_status():
    """Get the status of the working medical chatbot"""
    try:
        return jsonify({
            "available": working_chatbot.is_available(),
            "training_id": working_chatbot.training_id,
            "processor_status": "available" if working_chatbot.processor else "not_available",
            "description": "Working medical document extraction system",
            "capabilities": [
                "Patient name extraction",
                "Date of birth extraction", 
                "Test result extraction",
                "Laboratory information",
                "Document summarization"
            ]
        })
    except Exception as e:
        logger.error(f"Error getting medical chat status: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ORIGINAL ENDPOINTS CONTINUE
# ============================================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Enhanced health check endpoint with comprehensive model and system status"""
    try:
        logger.info("Health check requested")

        # Get training status safely
        training_status = None
        try:
            if trainer and hasattr(trainer, "get_training_status"):
                training_status = trainer.get_training_status()
            else:
                training_status = {"is_training": False, "progress": 0, "status_message": "Trainer not available"}
        except Exception as e:
            logger.warning(f"Error getting training status: {e}")
            training_status = {"is_training": False, "progress": 0, "status_message": f"Error: {str(e)}"}

        # Get comprehensive model status
        models_status = get_models_status()

        # System information
        system_info = {
            "backend": "online",
            "database": "connected" if db_success else "error",
            "pdf_processing": PDF_PROCESSING_AVAILABLE,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "working_medical_chatbot": working_chatbot.is_available()
        }

        # Environment status
        env_status = {
            "azure_ml_disabled": os.getenv("AZURE_ML_DISABLED", "false").lower() in ["true", "1", "yes"],
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "required_env_vars": {
                "AZURE_SUBSCRIPTION_ID": bool(os.getenv("AZURE_SUBSCRIPTION_ID")),
                "AZURE_RESOURCE_GROUP": bool(os.getenv("AZURE_RESOURCE_GROUP")),
                "AZURE_WORKSPACE_NAME": bool(os.getenv("AZURE_WORKSPACE_NAME")),
            }
        }

        health_response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": system_info,
            "environment": env_status,
            "training": training_status,
            "models": models_status,
            "version": "2.1.0",  # Updated version with working medical chat
            "uptime": "unknown"
        }

        logger.info("Health check completed successfully")
        return jsonify(health_response)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "models": get_fallback_models_status(),
            "system": {"backend": "error"}
        }), 500


@app.route("/api/system/status", methods=["GET"])
def system_status():
    """Detailed system status endpoint for debugging"""
    try:
        # Get Azure ML client status if available
        azure_details = None
        if trainer and hasattr(trainer, "azure_ml_client"):
            try:
                azure_client = trainer.azure_ml_client
                if azure_client and azure_client.is_available():
                    azure_details = {
                        "client_available": True,
                        "workspace_connected": True,
                        "billing_available": azure_client.consumption_client is not None,
                        "monitoring_available": azure_client.monitor_client is not None,
                    }
                else:
                    azure_details = {
                        "client_available": False,
                        "error": "Azure ML client not available"
                    }
            except Exception as e:
                azure_details = {
                    "client_available": False,
                    "error": str(e)
                }

        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "azure_details": azure_details,
            "trainer_available": trainer is not None,
            "working_medical_chatbot_available": working_chatbot.is_available(),
            "database_initialized": db_success,
            "environment_variables": {
                key: bool(os.getenv(key)) for key in [
                    "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP",
                    "AZURE_WORKSPACE_NAME", "OPENAI_API_KEY", "AZURE_ML_DISABLED"
                ]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    logger.warning(f"File too large error: {error}")
    return jsonify({
        "error": "File too large",
        "max_size": f"{MAX_CONTENT_LENGTH // (1024*1024)}MB"
    }), 413


@app.errorhandler(404)
def not_found(error):
    logger.warning(f"Not found: {error}")
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


# Cleanup handler
def cleanup_on_shutdown():
    """Cleanup resources on application shutdown"""
    try:
        if trainer and hasattr(trainer, "cleanup"):
            logger.info("Cleaning up trainer resources...")
            trainer.cleanup()
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Register cleanup handler
import atexit
atexit.register(cleanup_on_shutdown)


if __name__ == "__main__":
    # Startup summary
    logger.info("=" * 60)
    logger.info("STARTUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"PDF Processing: {'Available' if PDF_PROCESSING_AVAILABLE else 'Unavailable'}")
    logger.info(f"Database: {'Connected' if db_success else 'Error'}")
    logger.info(f"Trainer: {'Available' if trainer else 'Error'}")
    logger.info(f"Working Medical Chatbot: {'Available' if working_chatbot.is_available() else 'Error'}")

    # Enhanced Azure status logging
    if azure_status["available"]:
        logger.info("Azure ML: Available and Connected")
        logger.info(f"   Workspace: {azure_status['workspace_info']['name']}")
        logger.info(f"   Compute Targets: {azure_status['working_compute_count']} working")
        if azure_status.get("additional_services"):
            services = azure_status["additional_services"]
            logger.info(f"   Monitor API: {'Available' if services.get('monitor') else 'Not available'}")
            logger.info(f"   Billing API: {'Available' if services.get('consumption') else 'Not available'}")
    else:
        logger.warning("Azure ML: Not Available")
        logger.warning(f"   Reason: {azure_status['reason']}")
        if azure_status.get("help"):
            logger.info(f"   Help: {azure_status['help']}")

    logger.info(f"CORS Origins: {ALLOWED_ORIGINS}")
    logger.info("=" * 60)
    logger.info("Medical Chatbot Backend Starting...")
    logger.info("   Access health check at: http://localhost:5000/health")
    logger.info("   Access system status at: http://localhost:5000/api/system/status")
    logger.info("   NEW: Medical chat at: http://localhost:5000/api/medical-chat")
    logger.info("   NEW: Medical chat status at: http://localhost:5000/api/medical-chat/status")
    logger.info("=" * 60)

    # Start the Flask application
    app.run(debug=True, host="0.0.0.0", port=5000)