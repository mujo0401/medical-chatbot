import os
from pathlib import Path

# Flask / CORS
ALLOWED_ORIGINS = ["http://localhost:3000"]
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Directories
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"

# Azure / OpenAI settings
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_MAX_TOKENS  = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

# Sampling & Penalty Parameters for improved coherence
OPENAI_TOP_P          = float(os.getenv("OPENAI_TOP_P", "0.9"))
FREQUENCY_PENALTY     = float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0.5"))
PRESENCE_PENALTY      = float(os.getenv("OPENAI_PRESENCE_PENALTY", "0.3"))

# Detailed system prompt guiding the model’s tone, structure, and length
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """You are a medical assistant whose responses must:
1. Directly address the user’s question.
2. Use bullet points or numbered lists where appropriate.
3. Stay under 200 words.
4. Never provide speculative advice—only factual, evidence-based information."""
)

MODEL_PREFERENCE = os.getenv("MODEL_PREFERENCE", "hybrid")

# Database
SQLITE_DB_PATH = BASE_DIR / "chatbot.db"
