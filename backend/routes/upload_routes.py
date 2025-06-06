from flask import Blueprint, request, jsonify
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

from config import UPLOADS_DIR
# Remove the trainer import from here to avoid circular imports
from utils.pdf_utils import extract_text_from_pdf, read_text_file
from utils.db_utils import (
    insert_uploaded_document, 
    mark_document_processed, 
    delete_uploaded_document,
    get_all_documents
)

upload_bp = Blueprint("upload_bp", __name__)

# Global trainer variable for lazy loading
_trainer = None

def get_trainer():
    """Get or create trainer instance (lazy loading)"""
    global _trainer
    if _trainer is None:
        from training.medical_trainer import MedicalChatbotTrainer
        _trainer = MedicalChatbotTrainer()
    return _trainer

ALLOWED_EXTS = {".pdf", ".txt", ".md"}

def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTS

@upload_bp.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    original_name = secure_filename(file.filename)
    ext = os.path.splitext(original_name)[1].lower()
    if not allowed_file(original_name):
        return jsonify({"error": "Unsupported file type"}), 400

    unique_name = f"{uuid.uuid4().hex}{ext}"
    UPLOADS_DIR.mkdir(exist_ok=True)
    save_path = UPLOADS_DIR / unique_name
    file.save(str(save_path))

    size = os.path.getsize(save_path)
    file_type = "pdf" if ext == ".pdf" else "text"

    doc_id = insert_uploaded_document(
        filename=unique_name,
        original_name=original_name,
        file_path=str(save_path),
        file_type=file_type,
        file_size=size
    )

    try:
        if file_type == "pdf":
            text = extract_text_from_pdf(str(save_path))
        else:
            text = read_text_file(str(save_path))

        mark_document_processed(doc_id)
        preview = text[:500] + "..." if len(text) > 500 else text
        
        return jsonify({
            "message": "Uploaded successfully",
            "document_id": doc_id,
            "filename": original_name,
            "file_type": file_type,
            "file_size": size,
            "text_preview": preview,
            "ready_for_training": True
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {e}"}), 500

@upload_bp.route("/documents", methods=["GET"])
def get_documents():
    """Get all uploaded documents"""
    try:
        documents = get_all_documents()
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@upload_bp.route("/documents/<int:doc_id>", methods=["DELETE"])
def delete_document_route(doc_id):
    try:
        success, file_path = delete_uploaded_document(doc_id)
        if not success:
            return jsonify({"error": f"No document with id {doc_id}"}), 404
        
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        return jsonify({"message": f"Document {doc_id} deleted"}), 200
    except Exception as e:
        return jsonify({"error": f"Error deleting document: {e}"}), 500