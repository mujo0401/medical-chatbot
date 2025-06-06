# routes/session_routes.py - FIXED: Remove /api prefix since blueprint uses url_prefix="/api"

from flask import Blueprint, request, jsonify
import uuid
import logging
import traceback

logger = logging.getLogger(__name__)

session_bp = Blueprint("session_bp", __name__)

# FIXED: Use "/chat/session" instead of "/api/chat/session" 
# because the blueprint is registered with url_prefix="/api"
@session_bp.route("/chat/session", methods=["POST", "OPTIONS"])
def create_session():
    """
    Create a new chat session row in the database and return its session_id.
    Final URL will be: /api/chat/session (due to url_prefix="/api")
    Returns: { "session_id": "<new-uuid>" }
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        logger.info("OPTIONS request received for /chat/session")
        return '', 200
    
    logger.info("Creating new chat session...")
    
    try:
        # Import here to avoid circular imports
        from utils.db_utils import create_chat_session
        
        new_session_id = create_chat_session()
        logger.info(f"Session created successfully: {new_session_id}")
        
        return jsonify({"session_id": new_session_id}), 201
        
    except ImportError as e:
        logger.error(f"Import error in create_session: {e}")
        logger.error(traceback.format_exc())
        
        # Fallback: create a UUID session ID
        fallback_session_id = str(uuid.uuid4())
        logger.info(f"Using fallback session ID: {fallback_session_id}")
        
        return jsonify({
            "session_id": fallback_session_id,
            "warning": "Session created with fallback ID due to database import error"
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        logger.error(traceback.format_exc())
        
        # Fallback: create a UUID session ID
        fallback_session_id = str(uuid.uuid4())
        logger.info(f"Using fallback session ID: {fallback_session_id}")
        
        return jsonify({
            "session_id": fallback_session_id,
            "warning": "Session created with fallback ID due to database error"
        }), 201

# FIXED: Remove /api prefix from other routes too
@session_bp.route("/sessions", methods=["GET", "OPTIONS"])
def get_sessions():
    """Get all chat sessions with their last messages."""
    if request.method == "OPTIONS":
        return '', 200
        
    try:
        from utils.db_utils import get_connection
        
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cs.session_id, cs.created_at,
                   (SELECT user_message FROM messages m
                    WHERE m.session_id = cs.session_id
                    ORDER BY m.timestamp DESC LIMIT 1) AS last_message
            FROM chat_sessions cs
            ORDER BY cs.created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for sid, created, last in rows:
            sessions.append({
                "session_id": sid,
                "created_at": created,
                "last_message": last or "New session"
            })
        return jsonify(sessions)
        
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return jsonify({"error": f"Failed to fetch sessions: {str(e)}"}), 500

@session_bp.route("/sessions/<session_id>/messages", methods=["GET", "OPTIONS"])
def get_session_messages(session_id):
    """Get all messages for a specific session."""
    if request.method == "OPTIONS":
        return '', 200
        
    try:
        from utils.db_utils import fetch_conversation_history
        
        history = fetch_conversation_history(session_id, limit=1000)
        return jsonify(history)
        
    except Exception as e:
        logger.error(f"Error fetching messages for session {session_id}: {e}")
        return jsonify({"error": f"Failed to fetch messages: {str(e)}"}), 500

# Test endpoint for debugging
@session_bp.route("/test-session", methods=["GET", "POST", "OPTIONS"])
def test_session_endpoint():
    """Simple test endpoint to verify the session blueprint is working"""
    if request.method == "OPTIONS":
        return '', 200
    
    logger.info(f"Session test endpoint called with method: {request.method}")
    return jsonify({
        "status": "success",
        "message": "Session blueprint is working correctly",
        "method": request.method,
        "final_url": "/api/test-session"
    }), 200