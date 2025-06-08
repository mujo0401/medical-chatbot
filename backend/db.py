# db.py
import sqlite3
import logging

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_message TEXT,
        bot_response TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_id TEXT UNIQUE,
        document_name TEXT,
        document_type TEXT,
        training_type TEXT,
        status TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        error_message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS uploaded_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        original_name TEXT,
        file_path TEXT,
        file_type TEXT,
        file_size INTEGER,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed BOOLEAN DEFAULT FALSE
    )
    """
]


def init_db(db_path: str = 'chatbot.db'):
    """
    Ensure that all required tables exist in the SQLite database.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for stmt in SCHEMA_STATEMENTS:
            cursor.execute(stmt)
        conn.commit()
        conn.close()
        logger.info("SQLite schema initialized/verified successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise


def get_db(db_path: str = 'chatbot.db'):
    """Get a database connection."""
    return sqlite3.connect(db_path)


def get_documents_by_ids(conn, doc_ids):
    """Get documents by their IDs."""
    if not doc_ids:
        return []
        
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(doc_ids))
    cursor.execute(
        f"SELECT id, filename, original_name, file_path, file_type, file_size, uploaded_at "
        f"FROM uploaded_documents WHERE id IN ({placeholders})",
        doc_ids
    )
    
    # Convert to list of dictionaries
    columns = ["id", "filename", "original_name", "file_path", "file_type", "file_size", "uploaded_at"]
    result = []
    for row in cursor.fetchall():
        result.append(dict(zip(columns, row)))
    
    return result


def get_training_history():
    """Get training history from the database."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT * FROM training_history ORDER BY started_at DESC"
        )
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        result = []
        for row in cursor.fetchall():
            result.append(dict(zip(columns, row)))
            
        return result
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        return []
    finally:
        conn.close()