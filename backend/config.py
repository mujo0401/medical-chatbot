import os
from pathlib import Path

# Flask / CORS
ALLOWED_ORIGINS = ["http://localhost:3000"]
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Directories
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"

# ========== MODEL CONFIGURATION ==========

# Model preference: controls which model to use by default
# Options: "local_trained", "eleuther", "openai", "hybrid_local_eleuther", "hybrid_all"
MODEL_PREFERENCE = os.getenv("MODEL_PREFERENCE", "hybrid_local_eleuther")

# Base model type for local training
# Options: "dialogpt" (microsoft/DialoGPT-medium), "eleuther" (EleutherAI/gpt-neo-2.7B)
LOCAL_BASE_MODEL_TYPE = os.getenv("LOCAL_BASE_MODEL_TYPE", "dialogpt")

# EleutherAI Model Configuration
ELEUTHER_MODEL_NAME = os.getenv("ELEUTHER_MODEL_NAME", "EleutherAI/gpt-neo-2.7B")

# Model Generation Parameters
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
TOP_K = int(os.getenv("TOP_K", "50"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# Hybrid Model Weights (how much to weight each model in hybrid mode)
HYBRID_WEIGHTS = {
    "local_trained": float(os.getenv("HYBRID_WEIGHT_LOCAL", "0.6")),
    "eleuther": float(os.getenv("HYBRID_WEIGHT_ELEUTHER", "0.4")),
    "openai": float(os.getenv("HYBRID_WEIGHT_OPENAI", "0.5"))
}

# Model-specific settings
MODEL_CONFIGS = {
    "local_trained": {
        "base_model_type": LOCAL_BASE_MODEL_TYPE,
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
    },
    "eleuther": {
        "model_name": ELEUTHER_MODEL_NAME,
        "max_new_tokens": int(os.getenv("ELEUTHER_MAX_NEW_TOKENS", "150")),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
    },
    "openai": {
        "model": os.getenv("OPENAI_MODEL", "gpt-4"),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        "top_p": float(os.getenv("OPENAI_TOP_P", "0.9")),
        "frequency_penalty": float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0.5")),
        "presence_penalty": float(os.getenv("OPENAI_PRESENCE_PENALTY", "0.3")),
    }
}

# ========== AZURE CONFIGURATION ==========

# Azure / ML settings
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")

# Azure Service Principal Authentication
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

# Azure Compute Configuration
AZURE_COMPUTE_TARGET = os.getenv("AZURE_COMPUTE_TARGET", "gpu-cluster")

# ========== OPENAI CONFIGURATION ==========

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

# Sampling & Penalty Parameters for improved coherence
OPENAI_TOP_P = float(os.getenv("OPENAI_TOP_P", "0.9"))
FREQUENCY_PENALTY = float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0.5"))
PRESENCE_PENALTY = float(os.getenv("OPENAI_PRESENCE_PENALTY", "0.3"))

# ========== SYSTEM PROMPT CONFIGURATION ==========

# Base system prompt
BASE_SYSTEM_PROMPT = """You are a medical assistant whose responses must:
1. Directly address the user's question.
2. Use bullet points or numbered lists where appropriate.
3. Stay under 200 words.
4. Never provide speculative advice—only factual, evidence-based information."""

# Model-specific system prompts (if needed)
SYSTEM_PROMPTS = {
    "default": os.getenv("SYSTEM_PROMPT", BASE_SYSTEM_PROMPT),
    "local_trained": os.getenv("LOCAL_SYSTEM_PROMPT", BASE_SYSTEM_PROMPT),
    "eleuther": os.getenv("ELEUTHER_SYSTEM_PROMPT", BASE_SYSTEM_PROMPT),
    "openai": os.getenv("OPENAI_SYSTEM_PROMPT", BASE_SYSTEM_PROMPT),
    "hybrid": os.getenv("HYBRID_SYSTEM_PROMPT", BASE_SYSTEM_PROMPT),
}

# ========== TRAINING CONFIGURATION ==========

# Training parameters
TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-6"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "50"))

# Training configuration per model type
TRAINING_CONFIGS = {
    "dialogpt": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-6,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "fp16": False,
    },
    "eleuther": {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "fp16": True,  # Use fp16 for large models
        "gradient_checkpointing": True,
    }
}

# ========== DATABASE CONFIGURATION ==========

# Database
SQLITE_DB_PATH = BASE_DIR / "chatbot.db"
DATABASE_PATH = str(SQLITE_DB_PATH)

# ========== SERVER CONFIGURATION ==========

# Flask Configuration
FLASK_ENV = os.getenv("FLASK_ENV", "development")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

# ========== FILE UPLOAD CONFIGURATION ==========

# File upload settings
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "pdf,txt,md").split(",")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(16 * 1024 * 1024)))  # 16MB

# PDF Processing Configuration
PDF_EXTRACTION_METHOD = os.getenv("PDF_EXTRACTION_METHOD", "pdfplumber")

# ========== PERFORMANCE CONFIGURATION ==========

# Memory and performance settings
CUDA_MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "0.8"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))

# Response generation timeouts
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "30"))  # seconds
HYBRID_GENERATION_TIMEOUT = int(os.getenv("HYBRID_GENERATION_TIMEOUT", "45"))  # seconds

# ========== LOGGING CONFIGURATION ==========

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ========== FEATURE FLAGS ==========

# Feature toggles
ENABLE_HYBRID_MODELS = os.getenv("ENABLE_HYBRID_MODELS", "true").lower() == "true"
ENABLE_MODEL_SWITCHING = os.getenv("ENABLE_MODEL_SWITCHING", "true").lower() == "true"
ENABLE_AZURE_TRAINING = os.getenv("ENABLE_AZURE_TRAINING", "true").lower() == "true"
ENABLE_RESPONSE_CACHING = os.getenv("ENABLE_RESPONSE_CACHING", "false").lower() == "true"

# ========== UTILITY FUNCTIONS ==========

def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["local_trained"])

def get_system_prompt(model_name: str = None) -> str:
    """Get system prompt for a specific model or default."""
    if model_name and model_name in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[model_name]
    return SYSTEM_PROMPTS["default"]

def get_training_config(base_model_type: str) -> dict:
    """Get training configuration for a specific base model type."""
    return TRAINING_CONFIGS.get(base_model_type, TRAINING_CONFIGS["dialogpt"])

def is_model_enabled(model_name: str) -> bool:
    """Check if a specific model is enabled."""
    if model_name.startswith("hybrid") and not ENABLE_HYBRID_MODELS:
        return False
    return True

def validate_config():
    """Validate configuration settings."""
    errors = []
    warnings = []
    
    # Check model preference
    valid_preferences = ["local_trained", "eleuther", "openai", "hybrid_local_eleuther", "hybrid_all"]
    if MODEL_PREFERENCE not in valid_preferences:
        errors.append(f"Invalid MODEL_PREFERENCE: {MODEL_PREFERENCE}. Must be one of {valid_preferences}")
    
    # Check base model type
    valid_base_types = ["dialogpt", "eleuther"]
    if LOCAL_BASE_MODEL_TYPE not in valid_base_types:
        errors.append(f"Invalid LOCAL_BASE_MODEL_TYPE: {LOCAL_BASE_MODEL_TYPE}. Must be one of {valid_base_types}")
    
    # Check hybrid models are enabled if hybrid preference is set
    if MODEL_PREFERENCE.startswith("hybrid") and not ENABLE_HYBRID_MODELS:
        warnings.append("Hybrid model preference set but ENABLE_HYBRID_MODELS is False")
    
    # Check OpenAI configuration if needed
    if MODEL_PREFERENCE == "openai" or MODEL_PREFERENCE == "hybrid_all":
        if not OPENAI_API_KEY:
            warnings.append("OpenAI model selected but OPENAI_API_KEY not set")
    
    # Check Azure configuration if Azure training is enabled
    if ENABLE_AZURE_TRAINING:
        required_azure_vars = ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"]
        missing_vars = [var for var in required_azure_vars if not globals().get(var)]
        if missing_vars:
            warnings.append(f"Azure training enabled but missing variables: {missing_vars}")
    
    # Check directory permissions
    for directory in [UPLOADS_DIR, MODELS_DIR]:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {directory}: {e}")
    
    return {"errors": errors, "warnings": warnings}

# Validate configuration on import
if __name__ == "__main__":
    validation_result = validate_config()
    if validation_result["errors"]:
        print("Configuration Errors:")
        for error in validation_result["errors"]:
            print(f"  ERROR: {error}")
    
    if validation_result["warnings"]:
        print("Configuration Warnings:")
        for warning in validation_result["warnings"]:
            print(f"  WARNING: {warning}")
    
    if not validation_result["errors"] and not validation_result["warnings"]:
        print("Configuration validation passed!")

# Export current model preference for compatibility
SYSTEM_PROMPT = get_system_prompt()