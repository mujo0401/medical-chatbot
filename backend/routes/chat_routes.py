# routes/chat_routes.py - FIXED: Remove /api prefix since blueprint uses url_prefix="/api"

from flask import Blueprint, request, jsonify
import uuid
import logging
from datetime import datetime

from utils.db_utils import (
    fetch_conversation_history,
    save_message
)

# Get the global trainer instance from app.py
def get_trainer():
    """Get the global MedicalChatbotTrainer instance from app.py"""
    try:
        from __main__ import trainer
        return trainer
    except (ImportError, AttributeError):
        # Fallback: create a new trainer instance
        try:
            from training.medical_trainer import MedicalChatbotTrainer
            return MedicalChatbotTrainer()
        except Exception as e:
            logging.error(f"Failed to create trainer: {e}")
            return None

chat_bp = Blueprint("chat_bp", __name__)
logger = logging.getLogger(__name__)

# FIXED: Use "/chat" instead of "/api/chat" 
# because the blueprint is registered with url_prefix="/api"
@chat_bp.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    """
    Handle a chat request.
    Final URL will be: /api/chat (due to url_prefix="/api")
    Expects JSON:
      {
        "message": "User question",
        "session_id": "optional-session-id"
      }
    Returns:
      {
        "reply": "Chatbot reply",
        "source_documents": [{"id": "...", "name": "..."}],
        "session_id": "session-id-used",
        "timestamp": "2025-06-03T..."
      }
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        logger.info("OPTIONS request received for /chat")
        return '', 200
    
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    message = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not message:
        return jsonify({"error": "No message provided"}), 400

    trainer = get_trainer()
    if not trainer:
        return jsonify({
            "reply": "I'm sorry, the AI service is currently unavailable. Please try again later.",
            "source_documents": [],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }), 200

    # Persist the user's message for conversation history
    try:
        from utils.db_utils import ensure_session_exists
        ensure_session_exists(session_id)
        save_message(session_id, message, "")  # Save user message; bot_response to update later
    except Exception as e:
        logger.warning(f"Failed to save user message: {e}")

    # Generate a response using MedicalChatbotTrainer
    try:
        # Fixed: generate_response returns a dict, not a string
        response_data = trainer.generate_response(message, session_id)
        
        # Extract the reply text for database storage
        if isinstance(response_data, dict):
            reply_text = response_data.get("reply", "I'm sorry, I couldn't generate a response.")
            source_documents = response_data.get("source_documents", [])
        else:
            # Fallback if response_data is a string (shouldn't happen with fixed trainer)
            reply_text = str(response_data)
            source_documents = []
        
        # Persist the assistant's response
        try:
            save_message(session_id, "", reply_text)
        except Exception as e:
            logger.warning(f"Failed to save bot response: {e}")

        # Return the full response data including source documents
        return jsonify({
            "reply": reply_text,
            "source_documents": source_documents,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }), 200

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "error": "Internal server error",
            "reply": "I'm sorry, I encountered an error while processing your request. Please try again.",
            "source_documents": [],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }), 200  # Return 200 so frontend gets the error message