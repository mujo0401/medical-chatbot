# utils/db_utils.py

import sqlite3
from config import SQLITE_DB_PATH
from typing import List, Dict, Optional
import config
import json
import logging
from datetime import datetime
import uuid
import platform
import os

logger = logging.getLogger(__name__)


def get_connection():
    return sqlite3.connect(str(SQLITE_DB_PATH))


def init_database():
    """Initialize the database and create all required tables, adding missing columns as needed."""
    try:
        print(f"🔄 Initializing database: {SQLITE_DB_PATH}")
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Create chat_sessions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                session_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Create messages table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
            )
            """
        )

        # Create uploaded_documents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT,
                file_path TEXT NOT NULL,
                file_type TEXT,
                file_size INTEGER,
                processed BOOLEAN DEFAULT FALSE,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Create training_history table with enhanced Azure support
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_id TEXT UNIQUE NOT NULL,
                document_name TEXT,
                document_type TEXT DEFAULT 'mixed',
                training_type TEXT NOT NULL,
                platform TEXT DEFAULT 'local',
                status TEXT NOT NULL,
                error_message TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                azure_job_name TEXT,
                azure_experiment TEXT,
                compute_target TEXT,
                estimated_cost REAL,
                actual_duration INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # New tables for additional features

        # Patients table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                firstName TEXT NOT NULL,
                lastName TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                dateOfBirth TEXT,
                gender TEXT,
                address TEXT,
                medicalConditions TEXT,
                allergies TEXT,
                medications TEXT,
                emergencyContact TEXT,
                insuranceProvider TEXT,
                notes TEXT,
                status TEXT DEFAULT 'active',
                createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                lastVisit TIMESTAMP
            )
            """
        )

        # Patient consultations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_consultations (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                type TEXT NOT NULL,
                summary TEXT NOT NULL,
                notes TEXT,
                diagnosis TEXT,
                treatment TEXT,
                follow_up TEXT,
                session_id TEXT,
                documents TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
            """
        )

        # Reports table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                format TEXT DEFAULT 'pdf',
                status TEXT DEFAULT 'completed',
                content TEXT,
                filters TEXT,
                dateRange TEXT,
                patientIds TEXT,
                createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                size TEXT,
                filename TEXT,
                downloadUrl TEXT
            )
            """
        )

        # Settings table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, key)
            )
            """
        )

        # Analytics data table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_range TEXT,
                type TEXT DEFAULT 'general'
            )
            """
        )

        # System backups table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_backups (
                id TEXT PRIMARY KEY,
                backup_type TEXT NOT NULL,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                size INTEGER,
                status TEXT DEFAULT 'completed'
            )
            """
        )

        # Check and add missing columns to training_history
        cursor.execute("PRAGMA table_info(training_history)")
        existing_columns = [column[1] for column in cursor.fetchall()]

        columns_to_add = [
            ("azure_job_name", "TEXT"),
            ("azure_experiment", "TEXT"),
            ("compute_target", "TEXT"),
            ("estimated_cost", "REAL"),
            ("actual_duration", "INTEGER"),
            ("platform", "TEXT DEFAULT 'local'"),
            ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            # Added for FAISS index and local model support:
            ("index_path", "TEXT"),
            ("metadata_path", "TEXT"),
            ("model_path", "TEXT"),
            ("total_documents", "INTEGER")
        ]

        for column_name, column_type in columns_to_add:
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE training_history ADD COLUMN {column_name} {column_type}")
                print(f"✅ Added column {column_name} to training_history table")

        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


# ===============================
# DOCUMENT FUNCTIONS
# ===============================

def insert_uploaded_document(
    filename: str, original_name: str, file_path: str, file_type: str, file_size: int
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO uploaded_documents
            (filename, original_name, file_path, file_type, file_size)
        VALUES (?, ?, ?, ?, ?)
        """,
        (filename, original_name, file_path, file_type, file_size),
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def mark_document_processed(doc_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE uploaded_documents SET processed = TRUE WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


def get_documents_info(doc_ids: List[int]):
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in doc_ids)
    cursor.execute(
        f"""
        SELECT id, original_name, file_path, file_type
        FROM uploaded_documents
        WHERE id IN ({placeholders}) AND processed = TRUE
        """,
        doc_ids,
    )
    results = cursor.fetchall()
    conn.close()

    documents = []
    for row in results:
        documents.append({
            "id": row[0],
            "original_name": row[1],
            "file_path": row[2],
            "file_type": row[3],
        })
    return documents


def delete_uploaded_document(doc_id: int) -> tuple:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM uploaded_documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, None
    file_path = row[0]
    cursor.execute("DELETE FROM uploaded_documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return True, file_path


def get_all_documents():
    """Get all uploaded documents with their metadata"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, filename, original_name, file_path, file_type, 
               file_size, uploaded_at, processed
        FROM uploaded_documents
        ORDER BY uploaded_at DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    documents = []
    for row in rows:
        documents.append({
            "id": row[0],
            "filename": row[1],
            "original_name": row[2],
            "file_path": row[3],
            "file_type": row[4],
            "file_size": row[5],
            "uploaded_at": row[6],
            "processed": bool(row[7]),
        })
    return documents


# ===============================
# SESSION AND MESSAGE FUNCTIONS
# ===============================

def ensure_session_exists(session_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO chat_sessions (session_id) VALUES (?)", (session_id,))
    conn.commit()
    conn.close()


def save_message(session_id: str, user_message: str, bot_response: str):
    conn = get_connection()
    cursor = conn.cursor()
    ensure_session_exists(session_id)
    cursor.execute(
        """
        INSERT INTO messages (session_id, user_message, bot_response)
        VALUES (?, ?, ?)
        """,
        (session_id, user_message, bot_response),
    )
    conn.commit()
    conn.close()

def create_chat_session() -> str:
    """
    Create a new chat session row in chat_sessions and return the generated session_id (UUID).
    """
    conn = get_connection()
    cursor = conn.cursor()
    new_id = str(uuid.uuid4())  # generate a random UUID string
    # Insert a new row with that session_id; 'created_at' will auto-fill
    cursor.execute(
        "INSERT INTO chat_sessions (session_id, created_at) VALUES (?, ?)",
        (new_id, datetime.utcnow())
    )
    conn.commit()
    conn.close()
    return new_id


def fetch_conversation_history(session_id: str, limit: int = 1000) -> List[Dict]:
    """
    Return up to 'limit' messages (both user and bot) for the given session_id.
    Each row is { id, session_id, user_message, bot_response, timestamp }.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, session_id, user_message, bot_response, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
        """,
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for (msg_id, sid, user_msg, bot_msg, ts) in rows:
        history.append({
            "id": msg_id,
            "session_id": sid,
            "user_message": user_msg,
            "bot_response": bot_msg,
            "timestamp": ts
        })
    return history


# ===============================
# TRAINING HISTORY FUNCTIONS (Enhanced)
# ===============================

def insert_training_history(
    training_id: str,
    doc_name: str,
    doc_type: str,
    training_type: str,
    status: str,
    error_msg: str = None,
    compute_target: str = None,
    estimated_cost: float = None,
    index_path: str = None,
    metadata_path: str = None,
    model_path: str = None,
    total_documents: int = None
):
    """
    Insert a new training history record or update if status is 'completed'.
    - doc_type: e.g., 'pdf' or 'txt' or 'mixed'
    - training_type: 'local' or 'azure'
    - status: 'started', 'completed', 'failed'
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Determine platform from training_type
    platform = "azure" if training_type.lower() == "azure" else "local"

    if status.lower() == "completed":
        cursor.execute(
            """
            UPDATE training_history
            SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?, 
                compute_target = ?, estimated_cost = ?, index_path = ?, metadata_path = ?, 
                model_path = ?, total_documents = ?, updated_at = CURRENT_TIMESTAMP
            WHERE training_id = ?
            """,
            (
                status,
                error_msg,
                compute_target,
                estimated_cost,
                index_path,
                metadata_path,
                model_path,
                total_documents,
                training_id,
            ),
        )
    else:
        cursor.execute(
            """
            INSERT OR IGNORE INTO training_history
            (training_id, document_name, document_type, training_type, platform, status, 
             error_message, compute_target, estimated_cost, index_path, metadata_path, model_path, total_documents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                training_id,
                doc_name,
                doc_type,
                training_type,
                platform,
                status,
                error_msg,
                compute_target,
                estimated_cost,
                index_path,
                metadata_path,
                model_path,
                total_documents,
            ),
        )

    conn.commit()
    conn.close()


def update_training_history_azure_job(training_id: str, azure_job_name: str, azure_experiment: str = "medical-chatbot"):
    """Update training history with Azure job information."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE training_history 
            SET azure_job_name = ?, azure_experiment = ?, updated_at = CURRENT_TIMESTAMP
            WHERE training_id = ?
            """,
            (azure_job_name, azure_experiment, training_id),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating Azure job info: {e}")
        return False


def update_training_history_status(
    training_id: str,
    status: str,
    error_message: str = None,
    actual_duration: int = None,
    compute_target: str = None,
    estimated_cost: float = None,
    index_path: str = None,
    metadata_path: str = None,
    model_path: str = None,
    total_documents: int = None
):
    """Update the status of a training history entry (used for mid-job updates and failures)."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if status.lower() == "completed":
            cursor.execute(
                """
                UPDATE training_history
                SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?, 
                    actual_duration = ?, compute_target = ?, estimated_cost = ?, 
                    index_path = ?, metadata_path = ?, model_path = ?, total_documents = ?, updated_at = CURRENT_TIMESTAMP
                WHERE training_id = ?
                """,
                (
                    status,
                    error_message,
                    actual_duration,
                    compute_target,
                    estimated_cost,
                    index_path,
                    metadata_path,
                    model_path,
                    total_documents,
                    training_id,
                ),
            )
        else:
            cursor.execute(
                """
                UPDATE training_history
                SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE training_id = ?
                """,
                (status, error_message, training_id),
            )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating training history status: {e}")
        return False


def get_training_history() -> List[Dict]:
    """Get all training history records with enhanced Azure and local model information."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT training_id, document_name, document_type, training_type, platform,
                   status, error_message, started_at, completed_at, azure_job_name,
                   azure_experiment, compute_target, estimated_cost, actual_duration,
                   index_path, metadata_path, model_path, total_documents
            FROM training_history
            ORDER BY started_at DESC
            """
        )
        rows = cursor.fetchall()

        history = []
        for row in rows:
            history_item = {
                "training_id": row[0],
                "document_name": row[1],
                "document_type": row[2],
                "training_type": row[3],
                "platform": row[4] or row[3],
                "status": row[5],
                "error_message": row[6],
                "started_at": row[7],
                "completed_at": row[8],
                "azure_job_name": row[9],
                "azure_experiment": row[10],
                "compute_target": row[11],
                "estimated_cost": row[12],
                "actual_duration": row[13],
                "index_path": row[14],
                "metadata_path": row[15],
                "model_path": row[16],
                "total_documents": row[17],
            }
            history.append(history_item)
        return history
    except Exception as e:
        print(f"Error fetching training history: {e}")
        return []
    finally:
        conn.close()


def get_training_history_by_id(training_id: str) -> Dict:
    """Get a specific training history entry by ID."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT training_id, document_name, document_type, training_type, platform,
                   status, error_message, started_at, completed_at, azure_job_name,
                   azure_experiment, compute_target, estimated_cost, actual_duration,
                   index_path, metadata_path, model_path, total_documents
            FROM training_history 
            WHERE training_id = ?
            """,
            (training_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "training_id": row[0],
                "document_name": row[1],
                "document_type": row[2],
                "training_type": row[3],
                "platform": row[4],
                "status": row[5],
                "error_message": row[6],
                "started_at": row[7],
                "completed_at": row[8],
                "azure_job_name": row[9],
                "azure_experiment": row[10],
                "compute_target": row[11],
                "estimated_cost": row[12],
                "actual_duration": row[13],
                "index_path": row[14],
                "metadata_path": row[15],
                "model_path": row[16],
                "total_documents": row[17],
            }
        return None
    except Exception as e:
        print(f"Error fetching training history: {e}")
        return None


def delete_training_history_by_id(training_id: str) -> bool:
    """Delete a specific training history entry by ID."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT training_id FROM training_history WHERE training_id = ?", (training_id,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return False

        cursor.execute("DELETE FROM training_history WHERE training_id = ?", (training_id,))
        conn.commit()
        conn.close()

        return True
    except Exception as e:
        print(f"Error deleting training history {training_id}: {e}")
        return False


def delete_all_training_history() -> int:
    """Delete all training history entries."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM training_history")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM training_history")
        conn.commit()
        conn.close()

        return count
    except Exception as e:
        print(f"Error deleting all training history: {e}")
        return 0


def get_azure_training_stats() -> Dict:
    """Get statistics about Azure training jobs."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_azure_jobs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_jobs,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_jobs,
                SUM(CASE WHEN status = 'started' THEN 1 ELSE 0 END) as running_jobs,
                AVG(estimated_cost) as avg_estimated_cost,
                SUM(estimated_cost) as total_estimated_cost,
                AVG(actual_duration) as avg_duration
            FROM training_history 
            WHERE platform = 'azure'
            """
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "total_azure_jobs": row[0] or 0,
                "completed_jobs": row[1] or 0,
                "failed_jobs": row[2] or 0,
                "running_jobs": row[3] or 0,
                "avg_estimated_cost": round(row[4] or 0, 2),
                "total_estimated_cost": round(row[5] or 0, 2),
                "avg_duration_minutes": round(row[6] or 0, 2),
            }
        return {}
    except Exception as e:
        print(f"Error fetching Azure training stats: {e}")
        return {}


def cleanup_old_training_records(days_old: int = 30) -> int:
    """Clean up training records older than the specified number of days."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM training_history 
            WHERE started_at < datetime('now', '-{} days')
            AND status IN ('completed', 'failed')
            """.format(days_old)
        )
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"Cleaned up {deleted_count} old training records")
        return deleted_count
    except Exception as e:
        print(f"Error cleaning up old training records: {e}")
        return 0


# ===============================
# NEW PATIENT FUNCTIONS
# ===============================

def create_patient(patient_data: Dict) -> Optional[Dict]:
    """Create a new patient record."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO patients 
            (id, firstName, lastName, email, phone, dateOfBirth, gender, address,
             medicalConditions, allergies, medications, emergencyContact, 
             insuranceProvider, notes, status, createdAt, updatedAt, lastVisit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_data["id"],
                patient_data["firstName"],
                patient_data["lastName"],
                patient_data.get("email"),
                patient_data.get("phone"),
                patient_data.get("dateOfBirth"),
                patient_data.get("gender"),
                patient_data.get("address"),
                patient_data.get("medicalConditions"),
                patient_data.get("allergies"),
                patient_data.get("medications"),
                patient_data.get("emergencyContact"),
                patient_data.get("insuranceProvider"),
                patient_data.get("notes"),
                patient_data["status"],
                patient_data["createdAt"],
                patient_data["updatedAt"],
                patient_data.get("lastVisit"),
            ),
        )

        conn.commit()
        return dict(patient_data)

    except Exception as e:
        logger.error(f"Error creating patient: {e}")
        conn.rollback()
        return None

    finally:
        conn.close()


def get_all_patients() -> List[Dict]:
    """Get all patient records."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM patients ORDER BY lastName, firstName")
        rows = cursor.fetchall()

        patients = []
        for row in rows:
            patients.append({
                "id": row[0],
                "firstName": row[1],
                "lastName": row[2],
                "email": row[3],
                "phone": row[4],
                "dateOfBirth": row[5],
                "gender": row[6],
                "address": row[7],
                "medicalConditions": row[8],
                "allergies": row[9],
                "medications": row[10],
                "emergencyContact": row[11],
                "insuranceProvider": row[12],
                "notes": row[13],
                "status": row[14],
                "createdAt": row[15],
                "updatedAt": row[16],
                "lastVisit": row[17],
            })
        return patients

    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        return []

    finally:
        conn.close()


def get_patient_by_id(patient_id: str) -> Optional[Dict]:
    """Get a patient record by its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        row = cursor.fetchone()

        if row:
            return {
                "id": row[0],
                "firstName": row[1],
                "lastName": row[2],
                "email": row[3],
                "phone": row[4],
                "dateOfBirth": row[5],
                "gender": row[6],
                "address": row[7],
                "medicalConditions": row[8],
                "allergies": row[9],
                "medications": row[10],
                "emergencyContact": row[11],
                "insuranceProvider": row[12],
                "notes": row[13],
                "status": row[14],
                "createdAt": row[15],
                "updatedAt": row[16],
                "lastVisit": row[17],
            }
        return None

    except Exception as e:
        logger.error(f"Error fetching patient: {e}")
        return None

    finally:
        conn.close()


def update_patient(patient_id: str, update_data: Dict) -> Optional[Dict]:
    """Update an existing patient’s information."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        set_clauses = []
        values = []

        for key, value in update_data.items():
            if key != "id":
                set_clauses.append(f"{key} = ?")
                values.append(value)

        if not set_clauses:
            return None

        values.append(patient_id)
        query = f"UPDATE patients SET {', '.join(set_clauses)} WHERE id = ?"

        cursor.execute(query, values)
        conn.commit()

        if cursor.rowcount > 0:
            return get_patient_by_id(patient_id)
        return None

    except Exception as e:
        logger.error(f"Error updating patient: {e}")
        conn.rollback()
        return None

    finally:
        conn.close()


def delete_patient(patient_id: str) -> bool:
    """Delete a patient (and their consultations)."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM patient_consultations WHERE patient_id = ?", (patient_id,))
        cursor.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        conn.commit()
        return cursor.rowcount > 0

    except Exception as e:
        logger.error(f"Error deleting patient: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


def search_patients(search_term: str, patients: List[Dict] = None) -> List[Dict]:
    """Search patients by name or email."""
    if patients is None:
        patients = get_all_patients()

    search_term = search_term.lower()
    return [
        p for p in patients
        if search_term in f"{p.get('firstName', '')} {p.get('lastName', '')}".lower()
        or search_term in (p.get('email') or "").lower()
    ]


def get_patient_consultations(patient_id: str) -> List[Dict]:
    """Get all consultations for a patient."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT * FROM patient_consultations 
            WHERE patient_id = ? 
            ORDER BY created_at DESC
            """,
            (patient_id,),
        )
        rows = cursor.fetchall()
        consultations = []

        for row in rows:
            consultation = {
                "id": row[0],
                "patient_id": row[1],
                "type": row[2],
                "summary": row[3],
                "notes": row[4],
                "diagnosis": row[5],
                "treatment": row[6],
                "follow_up": row[7],
                "session_id": row[8],
                "created_at": row[10],
                "updated_at": row[11],
            }

            # Parse JSON documents field
            if row[9]:
                try:
                    consultation["documents"] = json.loads(row[9])
                except:
                    consultation["documents"] = []
            else:
                consultation["documents"] = []

            consultations.append(consultation)

        return consultations

    except Exception as e:
        logger.error(f"Error fetching patient consultations: {e}")
        return []

    finally:
        conn.close()


def create_patient_consultation(consultation_data: Dict) -> Optional[Dict]:
    """Create a new patient consultation entry."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        documents_json = json.dumps(consultation_data.get("documents", []))

        cursor.execute(
            """
            INSERT INTO patient_consultations
            (id, patient_id, type, summary, notes, diagnosis, treatment, follow_up,
             session_id, documents, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                consultation_data["id"],
                consultation_data["patient_id"],
                consultation_data["type"],
                consultation_data["summary"],
                consultation_data.get("notes"),
                consultation_data.get("diagnosis"),
                consultation_data.get("treatment"),
                consultation_data.get("follow_up"),
                consultation_data.get("session_id"),
                documents_json,
                consultation_data["created_at"],
                consultation_data["updated_at"],
            ),
        )

        conn.commit()
        consultation_data["documents"] = consultation_data.get("documents", [])
        return consultation_data

    except Exception as e:
        logger.error(f"Error creating consultation: {e}")
        conn.rollback()
        return None

    finally:
        conn.close()


# ===============================
# REPORTS FUNCTIONS
# ===============================

def create_report(report_data: Dict) -> Optional[Dict]:
    """Create a new report entry."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO reports
            (id, title, type, description, format, status, content, filters,
             dateRange, patientIds, createdAt, size, filename, downloadUrl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_data["id"],
                report_data["title"],
                report_data["type"],
                report_data.get("description"),
                report_data.get("format", "pdf"),
                report_data.get("status", "completed"),
                json.dumps(report_data.get("content", {})),
                json.dumps(report_data.get("filters", {})),
                report_data.get("dateRange"),
                json.dumps(report_data.get("patientIds", [])),
                report_data["createdAt"],
                report_data.get("size"),
                report_data.get("filename"),
                report_data.get("downloadUrl"),
            ),
        )

        conn.commit()
        return dict(report_data)

    except Exception as e:
        logger.error(f"Error creating report: {e}")
        conn.rollback()
        return None

    finally:
        conn.close()


def get_all_reports() -> List[Dict]:
    """Get all report entries."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM reports ORDER BY createdAt DESC")
        rows = cursor.fetchall()

        reports = []
        for row in rows:
            report = {
                "id": row[0],
                "title": row[1],
                "type": row[2],
                "description": row[3],
                "format": row[4],
                "status": row[5],
                "dateRange": row[8],
                "createdAt": row[10],
                "size": row[11],
                "filename": row[12],
                "downloadUrl": row[13],
            }

            # Parse JSON fields
            try:
                report["content"] = json.loads(row[6]) if row[6] else {}
                report["filters"] = json.loads(row[7]) if row[7] else {}
                report["patientIds"] = json.loads(row[9]) if row[9] else []
            except:
                report["content"] = {}
                report["filters"] = {}
                report["patientIds"] = []

            reports.append(report)

        return reports

    except Exception as e:
        logger.error(f"Error fetching reports: {e}")
        return []

    finally:
        conn.close()


def get_report_by_id(report_id: str) -> Optional[Dict]:
    """Get a specific report by its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
        row = cursor.fetchone()

        if row:
            report = {
                "id": row[0],
                "title": row[1],
                "type": row[2],
                "description": row[3],
                "format": row[4],
                "status": row[5],
                "dateRange": row[8],
                "createdAt": row[10],
                "size": row[11],
                "filename": row[12],
                "downloadUrl": row[13],
            }

            # Parse JSON fields
            try:
                report["content"] = json.loads(row[6]) if row[6] else {}
                report["filters"] = json.loads(row[7]) if row[7] else {}
                report["patientIds"] = json.loads(row[9]) if row[9] else []
            except:
                report["content"] = {}
                report["filters"] = {}
                report["patientIds"] = []

            return report

        return None

    except Exception as e:
        logger.error(f"Error fetching report: {e}")
        return None

    finally:
        conn.close()


def delete_report(report_id: str) -> bool:
    """Delete a report entry."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()
        return cursor.rowcount > 0

    except Exception as e:
        logger.error(f"Error deleting report: {e}")
        return False

    finally:
        conn.close()


# ===============================
# SETTINGS FUNCTIONS
# ===============================

def get_system_settings() -> Optional[Dict]:
    """Get all system settings."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT category, key, value FROM settings")
        rows = cursor.fetchall()

        settings = {}
        for row in rows:
            category = row[0]
            key = row[1]
            value = json.loads(row[2])

            if category not in settings:
                settings[category] = {}

            settings[category][key] = value

        return settings

    except Exception as e:
        logger.error(f"Error fetching settings: {e}")
        return None

    finally:
        conn.close()


def save_system_settings(settings: Dict) -> bool:
    """Save system settings (overwrites existing)."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM settings")

        for category, category_settings in settings.items():
            if category.startswith("_"):
                continue
            for key, value in category_settings.items():
                cursor.execute(
                    """
                    INSERT INTO settings (category, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (category, key, json.dumps(value), datetime.now().isoformat()),
                )

        conn.commit()
        return True

    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


# ===============================
# ANALYTICS FUNCTIONS
# ===============================

def save_analytics_data(analytics_data: Dict) -> bool:
    """Save analytics data for later retrieval."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO analytics_data (data, time_range, type)
            VALUES (?, ?, ?)
            """,
            (
                json.dumps(analytics_data),
                analytics_data.get("timeRange", "30d"),
                "general",
            ),
        )

        conn.commit()
        return True

    except Exception as e:
        logger.error(f"Error saving analytics data: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


def get_analytics_data(time_range: str = None, limit: int = 10) -> List[Dict]:
    """Retrieve recent analytics data entries."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        if time_range:
            cursor.execute(
                """
                SELECT * FROM analytics_data 
                WHERE time_range = ? 
                ORDER BY generated_at DESC 
                LIMIT ?
                """,
                (time_range, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM analytics_data 
                ORDER BY generated_at DESC 
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        analytics = []

        for row in rows:
            try:
                data = json.loads(row[1])
                data["id"] = row[0]
                data["generated_at"] = row[2]
                analytics.append(data)
            except:
                continue

        return analytics

    except Exception as e:
        logger.error(f"Error fetching analytics data: {e}")
        return []

    finally:
        conn.close()


# ===============================
# SYSTEM FUNCTIONS
# ===============================

def create_system_backup(backup_type: str = "full") -> Optional[str]:
    """Create a system backup entry (metadata only)."""
    try:
        backup_id = str(uuid.uuid4())
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO system_backups (id, backup_type, created_at, status)
            VALUES (?, ?, ?, ?)
            """,
            (backup_id, backup_type, datetime.now().isoformat(), "completed"),
        )

        conn.commit()
        conn.close()

        logger.info(f"Backup {backup_id} created successfully")
        return backup_id

    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return None


def get_system_info() -> Dict:
    """Retrieve system information, including DB size and optionally resource usage."""
    try:
        system_info = {
            "system": {
                "os": platform.system(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture()[0],
            },
            "database": {
                "version": sqlite3.sqlite_version,
                "size": f"{os.path.getsize(SQLITE_DB_PATH) / (1024*1024):.1f} MB"
                if os.path.exists(SQLITE_DB_PATH)
                else "0 MB",
            },
            "azure_available": False,  # Placeholder: would check Azure ML SDK
            "storage": {
                "documents": len(get_all_documents()),
                "total_size": "Unknown",
                "available_space": "Unknown",
            },
            "performance": {
                "uptime": "Unknown",
                "memory_usage": "Unknown",
                "cpu_usage": "Unknown",
            },
        }

        try:
            import psutil

            memory = psutil.virtual_memory()
            system_info["performance"]["memory_usage"] = f"{memory.percent:.1f}%"
            system_info["performance"]["cpu_usage"] = f"{psutil.cpu_percent(interval=1):.1f}%"

            documents = get_all_documents()
            total_size = sum(doc.get("file_size", 0) for doc in documents)
            system_info["storage"]["total_size"] = f"{total_size / (1024*1024):.1f} MB"

        except ImportError:
            pass

        return system_info

    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {}


# Convenience functions to match API expectations
def get_user_settings(user_id: str = None) -> Optional[Dict]:
    """Get user-specific settings (stub; returns system settings)."""
    return get_system_settings()


def save_user_settings(settings: Dict, user_id: str = None) -> bool:
    """Save user-specific settings (stub; writes system settings)."""
    return save_system_settings(settings)

def create_chat_session() -> str:
    """
    Insert a new row into `chat_sessions` and return the newly generated session_id (UUID).
    """
    conn = get_connection()
    cursor = conn.cursor()
    new_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO chat_sessions (session_id, created_at) VALUES (?, ?)",
        (new_id, datetime.utcnow())
    )
    conn.commit()
    conn.close()
    return new_id