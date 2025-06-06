from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import uuid
import logging
import json

from utils.db_utils import (
    create_patient,
    get_all_patients,
    get_patient_by_id,
    update_patient,
    delete_patient,
    search_patients,
    get_patient_consultations,
    create_patient_consultation
)

patients_bp = Blueprint("patients_bp", __name__)
logger = logging.getLogger(__name__)

@patients_bp.route("/patients", methods=["GET"])
def get_patients():
    """Get all patients with optional filtering and sorting"""
    try:
        # Get query parameters
        search = request.args.get('search', '')
        status = request.args.get('status', 'all')
        sort_by = request.args.get('sort', 'name')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        
        # Get patients from database
        patients = get_all_patients()
        
        # Apply search filter
        if search:
            patients = search_patients(search, patients)
        
        # Apply status filter
        if status != 'all':
            patients = [p for p in patients if p.get('status', 'active') == status]
        
        # Apply sorting
        if sort_by == 'name':
            patients.sort(key=lambda p: f"{p.get('firstName', '')} {p.get('lastName', '')}")
        elif sort_by == 'date':
            patients.sort(key=lambda p: p.get('lastVisit') or p.get('createdAt', ''), reverse=True)
        elif sort_by == 'status':
            patients.sort(key=lambda p: p.get('status', 'active'))
        
        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_patients = patients[start_idx:end_idx]
        
        return jsonify({
            "patients": paginated_patients,
            "total": len(patients),
            "page": page,
            "per_page": per_page,
            "total_pages": (len(patients) + per_page - 1) // per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        return jsonify({"error": f"Failed to fetch patients: {str(e)}"}), 500

@patients_bp.route("/patients", methods=["POST"])
def create_new_patient():
    """Create a new patient"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['firstName', 'lastName']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate patient ID
        patient_id = str(uuid.uuid4())
        
        # Prepare patient data
        patient_data = {
            "id": patient_id,
            "firstName": data.get('firstName', '').strip(),
            "lastName": data.get('lastName', '').strip(),
            "email": data.get('email', '').strip(),
            "phone": data.get('phone', '').strip(),
            "dateOfBirth": data.get('dateOfBirth'),
            "gender": data.get('gender'),
            "address": data.get('address', '').strip(),
            "medicalConditions": data.get('medicalConditions', '').strip(),
            "allergies": data.get('allergies', '').strip(),
            "medications": data.get('medications', '').strip(),
            "emergencyContact": data.get('emergencyContact', '').strip(),
            "insuranceProvider": data.get('insuranceProvider', '').strip(),
            "notes": data.get('notes', '').strip(),
            "status": data.get('status', 'active'),
            "createdAt": datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
            "lastVisit": None
        }
        
        # Create patient in database
        created_patient = create_patient(patient_data)
        
        if created_patient:
            return jsonify({
                "message": "Patient created successfully",
                "patient": created_patient
            }), 201
        else:
            return jsonify({"error": "Failed to create patient"}), 500
            
    except Exception as e:
        logger.error(f"Error creating patient: {e}")
        return jsonify({"error": f"Failed to create patient: {str(e)}"}), 500

@patients_bp.route("/patients/<patient_id>", methods=["GET"])
def get_patient(patient_id):
    """Get a specific patient by ID"""
    try:
        patient = get_patient_by_id(patient_id)
        
        if not patient:
            return jsonify({"error": "Patient not found"}), 404
        
        # Get patient's consultation history
        consultations = get_patient_consultations(patient_id)
        patient['consultations'] = consultations
        
        return jsonify(patient), 200
        
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {e}")
        return jsonify({"error": f"Failed to fetch patient: {str(e)}"}), 500

@patients_bp.route("/patients/<patient_id>", methods=["PUT"])
def update_patient_info(patient_id):
    """Update patient information"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Check if patient exists
        existing_patient = get_patient_by_id(patient_id)
        if not existing_patient:
            return jsonify({"error": "Patient not found"}), 404
        
        # Prepare update data
        update_data = {
            "firstName": data.get('firstName', existing_patient.get('firstName', '')).strip(),
            "lastName": data.get('lastName', existing_patient.get('lastName', '')).strip(),
            "email": data.get('email', existing_patient.get('email', '')).strip(),
            "phone": data.get('phone', existing_patient.get('phone', '')).strip(),
            "dateOfBirth": data.get('dateOfBirth', existing_patient.get('dateOfBirth')),
            "gender": data.get('gender', existing_patient.get('gender')),
            "address": data.get('address', existing_patient.get('address', '')).strip(),
            "medicalConditions": data.get('medicalConditions', existing_patient.get('medicalConditions', '')).strip(),
            "allergies": data.get('allergies', existing_patient.get('allergies', '')).strip(),
            "medications": data.get('medications', existing_patient.get('medications', '')).strip(),
            "emergencyContact": data.get('emergencyContact', existing_patient.get('emergencyContact', '')).strip(),
            "insuranceProvider": data.get('insuranceProvider', existing_patient.get('insuranceProvider', '')).strip(),
            "notes": data.get('notes', existing_patient.get('notes', '')).strip(),
            "status": data.get('status', existing_patient.get('status', 'active')),
            "updatedAt": datetime.now().isoformat()
        }
        
        # Update patient in database
        updated_patient = update_patient(patient_id, update_data)
        
        if updated_patient:
            return jsonify({
                "message": "Patient updated successfully",
                "patient": updated_patient
            }), 200
        else:
            return jsonify({"error": "Failed to update patient"}), 500
            
    except Exception as e:
        logger.error(f"Error updating patient {patient_id}: {e}")
        return jsonify({"error": f"Failed to update patient: {str(e)}"}), 500

@patients_bp.route("/patients/<patient_id>", methods=["DELETE"])
def delete_patient_record(patient_id):
    """Delete a patient record"""
    try:
        # Check if patient exists
        existing_patient = get_patient_by_id(patient_id)
        if not existing_patient:
            return jsonify({"error": "Patient not found"}), 404
        
        # Delete patient from database
        success = delete_patient(patient_id)
        
        if success:
            return jsonify({"message": "Patient deleted successfully"}), 200
        else:
            return jsonify({"error": "Failed to delete patient"}), 500
            
    except Exception as e:
        logger.error(f"Error deleting patient {patient_id}: {e}")
        return jsonify({"error": f"Failed to delete patient: {str(e)}"}), 500

@patients_bp.route("/patients/<patient_id>/consultations", methods=["GET"])
def get_patient_consultation_history(patient_id):
    """Get consultation history for a patient"""
    try:
        # Check if patient exists
        patient = get_patient_by_id(patient_id)
        if not patient:
            return jsonify({"error": "Patient not found"}), 404
        
        # Get consultations
        consultations = get_patient_consultations(patient_id)
        
        return jsonify({
            "patient_id": patient_id,
            "consultations": consultations,
            "total": len(consultations)
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching consultations for patient {patient_id}: {e}")
        return jsonify({"error": f"Failed to fetch consultations: {str(e)}"}), 500

@patients_bp.route("/patients/<patient_id>/consultations", methods=["POST"])
def create_consultation(patient_id):
    """Create a new consultation for a patient"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Check if patient exists
        patient = get_patient_by_id(patient_id)
        if not patient:
            return jsonify({"error": "Patient not found"}), 404
        
        # Validate required fields
        required_fields = ['type', 'summary']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate consultation ID
        consultation_id = str(uuid.uuid4())
        
        # Prepare consultation data
        consultation_data = {
            "id": consultation_id,
            "patient_id": patient_id,
            "type": data.get('type'),  # e.g., 'chat', 'document_analysis', 'follow_up'
            "summary": data.get('summary'),
            "notes": data.get('notes', ''),
            "diagnosis": data.get('diagnosis', ''),
            "treatment": data.get('treatment', ''),
            "follow_up": data.get('follow_up', ''),
            "session_id": data.get('session_id'),  # Link to chat session if applicable
            "documents": data.get('documents', []),  # Associated documents
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Create consultation in database
        created_consultation = create_patient_consultation(consultation_data)
        
        if created_consultation:
            # Update patient's last visit
            update_data = {"lastVisit": datetime.now().isoformat()}
            update_patient(patient_id, update_data)
            
            return jsonify({
                "message": "Consultation created successfully",
                "consultation": created_consultation
            }), 201
        else:
            return jsonify({"error": "Failed to create consultation"}), 500
            
    except Exception as e:
        logger.error(f"Error creating consultation for patient {patient_id}: {e}")
        return jsonify({"error": f"Failed to create consultation: {str(e)}"}), 500

@patients_bp.route("/patients/search", methods=["POST"])
def search_patients_advanced():
    """Advanced patient search with multiple criteria"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No search criteria provided"}), 400
        
        # Get all patients
        patients = get_all_patients()
        
        # Apply filters
        filtered_patients = patients
        
        # Name filter
        if data.get('name'):
            name_query = data['name'].lower()
            filtered_patients = [
                p for p in filtered_patients
                if name_query in f"{p.get('firstName', '')} {p.get('lastName', '')}".lower()
                or name_query in p.get('email', '').lower()
            ]
        
        # Age range filter
        if data.get('age_min') or data.get('age_max'):
            current_year = datetime.now().year
            filtered_patients = [
                p for p in filtered_patients
                if p.get('dateOfBirth') and (
                    (not data.get('age_min') or 
                     current_year - datetime.fromisoformat(p['dateOfBirth']).year >= data['age_min']) and
                    (not data.get('age_max') or 
                     current_year - datetime.fromisoformat(p['dateOfBirth']).year <= data['age_max'])
                )
            ]
        
        # Gender filter
        if data.get('gender'):
            filtered_patients = [
                p for p in filtered_patients
                if p.get('gender', '').lower() == data['gender'].lower()
            ]
        
        # Medical condition filter
        if data.get('condition'):
            condition_query = data['condition'].lower()
            filtered_patients = [
                p for p in filtered_patients
                if condition_query in p.get('medicalConditions', '').lower()
            ]
        
        # Status filter
        if data.get('status'):
            filtered_patients = [
                p for p in filtered_patients
                if p.get('status', 'active') == data['status']
            ]
        
        # Date range filter (last visit)
        if data.get('visit_from') or data.get('visit_to'):
            filtered_patients = [
                p for p in filtered_patients
                if p.get('lastVisit') and (
                    (not data.get('visit_from') or 
                     datetime.fromisoformat(p['lastVisit']) >= datetime.fromisoformat(data['visit_from'])) and
                    (not data.get('visit_to') or 
                     datetime.fromisoformat(p['lastVisit']) <= datetime.fromisoformat(data['visit_to']))
                )
            ]
        
        return jsonify({
            "patients": filtered_patients,
            "total": len(filtered_patients),
            "search_criteria": data
        }), 200
        
    except Exception as e:
        logger.error(f"Error in advanced patient search: {e}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@patients_bp.route("/patients/statistics", methods=["GET"])
def get_patient_statistics():
    """Get patient statistics and demographics"""
    try:
        patients = get_all_patients()
        
        if not patients:
            return jsonify({
                "total": 0,
                "demographics": {},
                "statistics": {}
            }), 200
        
        # Calculate demographics
        total_patients = len(patients)
        
        # Gender distribution
        gender_counts = {}
        for patient in patients:
            gender = patient.get('gender', 'Unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        # Age distribution
        age_ranges = {"0-18": 0, "19-35": 0, "36-50": 0, "51-65": 0, "65+": 0, "Unknown": 0}
        current_year = datetime.now().year
        
        for patient in patients:
            if patient.get('dateOfBirth'):
                try:
                    birth_year = datetime.fromisoformat(patient['dateOfBirth']).year
                    age = current_year - birth_year
                    
                    if age <= 18:
                        age_ranges["0-18"] += 1
                    elif age <= 35:
                        age_ranges["19-35"] += 1
                    elif age <= 50:
                        age_ranges["36-50"] += 1
                    elif age <= 65:
                        age_ranges["51-65"] += 1
                    else:
                        age_ranges["65+"] += 1
                except:
                    age_ranges["Unknown"] += 1
            else:
                age_ranges["Unknown"] += 1
        
        # Status distribution
        status_counts = {}
        for patient in patients:
            status = patient.get('status', 'active')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Recent activity
        recent_patients = len([
            p for p in patients
            if p.get('createdAt') and 
            datetime.fromisoformat(p['createdAt']) >= datetime.now() - timedelta(days=30)
        ])
        
        # Patients with recent visits
        recent_visits = len([
            p for p in patients
            if p.get('lastVisit') and 
            datetime.fromisoformat(p['lastVisit']) >= datetime.now() - timedelta(days=30)
        ])
        
        statistics = {
            "total": total_patients,
            "demographics": {
                "gender": gender_counts,
                "age_ranges": age_ranges,
                "status": status_counts
            },
            "activity": {
                "new_patients_30d": recent_patients,
                "recent_visits_30d": recent_visits,
                "active_patients": status_counts.get('active', 0),
                "inactive_patients": status_counts.get('inactive', 0)
            },
            "medical": {
                "patients_with_conditions": len([p for p in patients if p.get('medicalConditions')]),
                "patients_with_allergies": len([p for p in patients if p.get('allergies')]),
                "patients_with_medications": len([p for p in patients if p.get('medications')])
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return jsonify(statistics), 200
        
    except Exception as e:
        logger.error(f"Error generating patient statistics: {e}")
        return jsonify({"error": f"Failed to generate statistics: {str(e)}"}), 500

@patients_bp.route("/patients/export", methods=["POST"])
def export_patients():
    """Export patient data in various formats"""
    try:
        data = request.get_json() or {}
        export_format = data.get('format', 'json')
        include_fields = data.get('fields', 'all')
        patient_ids = data.get('patient_ids', [])
        
        # Get patients to export
        if patient_ids:
            patients = [get_patient_by_id(pid) for pid in patient_ids]
            patients = [p for p in patients if p]  # Remove None values
        else:
            patients = get_all_patients()
        
        # Filter fields if specified
        if include_fields != 'all' and isinstance(include_fields, list):
            filtered_patients = []
            for patient in patients:
                filtered_patient = {field: patient.get(field) for field in include_fields if field in patient}
                filtered_patients.append(filtered_patient)
            patients = filtered_patients
        
        # Generate export data
        export_data = {
            "patients": patients,
            "export_info": {
                "total_patients": len(patients),
                "exported_at": datetime.now().isoformat(),
                "format": export_format,
                "fields": include_fields
            }
        }
        
        filename = f"patients_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
        
        return jsonify({
            "data": export_data,
            "filename": filename,
            "format": export_format
        }), 200
        
    except Exception as e:
        logger.error(f"Error exporting patients: {e}")
        return jsonify({"error": f"Failed to export patients: {str(e)}"}), 500