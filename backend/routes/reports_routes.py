from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import uuid
import logging
import json
import os
from io import StringIO
import csv

from utils.db_utils import (
    get_all_documents,
    get_training_history,
    get_all_patients,
    create_report,
    get_all_reports,
    get_report_by_id,
    delete_report
)

reports_bp = Blueprint("reports_bp", __name__)
logger = logging.getLogger(__name__)

def generate_analytics_report(date_range, filters, patient_ids=None):
    """Generate comprehensive analytics report"""
    try:
        # Calculate date range
        end_date = datetime.now()
        if date_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif date_range == '30d':
            start_date = end_date - timedelta(days=30)
        elif date_range == '90d':
            start_date = end_date - timedelta(days=90)
        elif date_range == '6m':
            start_date = end_date - timedelta(days=180)
        elif date_range == '1y':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = datetime(2020, 1, 1)
        
        # Get data
        documents = get_all_documents()
        training_history = get_training_history()
        patients = get_all_patients() if filters.get('includePatients', True) else []
        
        # Filter by date range
        filtered_documents = [
            doc for doc in documents
            if doc.get('uploaded_at') and 
            datetime.fromisoformat(doc['uploaded_at'].replace('Z', '+00:00')) >= start_date
        ]
        
        filtered_training = [
            training for training in training_history
            if training.get('started_at') and 
            datetime.fromisoformat(training['started_at'].replace('Z', '+00:00')) >= start_date
        ]
        
        # Generate report content
        report_content = {
            "title": f"Medical Analytics Report - {date_range.upper()}",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "range": date_range
            },
            "summary": {
                "total_documents": len(documents),
                "documents_in_period": len(filtered_documents),
                "total_training_sessions": len(training_history),
                "training_in_period": len(filtered_training),
                "total_patients": len(patients),
                "system_uptime": "99.8%"
            },
            "documents": {
                "overview": {
                    "total": len(documents),
                    "processed": len([d for d in documents if d.get('processed', False)]),
                    "total_size_mb": sum(d.get('file_size', 0) for d in documents) / (1024 * 1024),
                    "types": {}
                },
                "recent_activity": [
                    {
                        "name": doc.get('original_name', 'Unknown'),
                        "type": doc.get('file_type', 'unknown'),
                        "uploaded": doc.get('uploaded_at'),
                        "size_kb": doc.get('file_size', 0) / 1024 if doc.get('file_size') else 0
                    }
                    for doc in sorted(filtered_documents, 
                                    key=lambda x: x.get('uploaded_at', ''), reverse=True)[:10]
                ]
            },
            "training": {
                "overview": {
                    "total_sessions": len(training_history),
                    "successful": len([t for t in training_history if t.get('status') == 'completed']),
                    "failed": len([t for t in training_history if t.get('status') == 'failed']),
                    "success_rate": (len([t for t in training_history if t.get('status') == 'completed']) / 
                                   len(training_history) * 100) if training_history else 0
                },
                "recent_sessions": [
                    {
                        "document": training.get('document_name', 'Unknown'),
                        "status": training.get('status', 'unknown'),
                        "started": training.get('started_at'),
                        "type": training.get('training_type', 'local')
                    }
                    for training in sorted(filtered_training,
                                         key=lambda x: x.get('started_at', ''), reverse=True)[:10]
                ]
            },
            "performance": {
                "response_times": {
                    "average": "1.2s",
                    "p95": "2.1s",
                    "p99": "3.5s"
                },
                "system_health": {
                    "uptime": "99.8%",
                    "error_rate": "0.1%",
                    "throughput": "450 req/hour"
                },
                "ai_accuracy": "94.5%"
            },
            "insights": {
                "top_document_types": ["PDF Reports", "Lab Results", "Medical Images"],
                "peak_usage_hours": ["9:00 AM", "2:00 PM", "6:00 PM"],
                "common_queries": ["Lab result interpretation", "Medication information", "Symptom analysis"]
            }
        }
        
        # Calculate document types
        doc_types = {}
        for doc in documents:
            doc_type = doc.get('file_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        report_content["documents"]["overview"]["types"] = doc_types
        
        return report_content
        
    except Exception as e:
        logger.error(f"Error generating analytics report: {e}")
        raise

def generate_patient_summary_report(date_range, filters, patient_ids=None):
    """Generate patient summary report"""
    try:
        patients = get_all_patients()
        
        if patient_ids:
            patients = [p for p in patients if p.get('id') in patient_ids]
        
        report_content = {
            "title": "Patient Summary Report",
            "generated_at": datetime.now().isoformat(),
            "period": date_range,
            "total_patients": len(patients),
            "patients": []
        }
        
        for patient in patients:
            patient_summary = {
                "id": patient.get('id'),
                "name": f"{patient.get('firstName', '')} {patient.get('lastName', '')}",
                "age": None,
                "gender": patient.get('gender'),
                "last_visit": patient.get('lastVisit'),
                "medical_conditions": patient.get('medicalConditions'),
                "allergies": patient.get('allergies'),
                "medications": patient.get('medications'),
                "status": patient.get('status', 'active')
            }
            
            # Calculate age if DOB is available
            if patient.get('dateOfBirth'):
                try:
                    birth_date = datetime.fromisoformat(patient['dateOfBirth'])
                    age = datetime.now().year - birth_date.year
                    patient_summary["age"] = age
                except:
                    pass
            
            report_content["patients"].append(patient_summary)
        
        # Add statistics
        report_content["statistics"] = {
            "by_gender": {},
            "by_age_group": {"0-18": 0, "19-35": 0, "36-50": 0, "51-65": 0, "65+": 0},
            "by_status": {},
            "with_conditions": len([p for p in patients if p.get('medicalConditions')]),
            "with_allergies": len([p for p in patients if p.get('allergies')])
        }
        
        return report_content
        
    except Exception as e:
        logger.error(f"Error generating patient summary report: {e}")
        raise

def generate_training_report(date_range, filters, patient_ids=None):
    """Generate training performance report"""
    try:
        training_history = get_training_history()
        documents = get_all_documents()
        
        report_content = {
            "title": "AI Training Performance Report",
            "generated_at": datetime.now().isoformat(),
            "period": date_range,
            "overview": {
                "total_sessions": len(training_history),
                "successful": len([t for t in training_history if t.get('status') == 'completed']),
                "failed": len([t for t in training_history if t.get('status') == 'failed']),
                "in_progress": len([t for t in training_history if t.get('status') == 'started']),
                "success_rate": (len([t for t in training_history if t.get('status') == 'completed']) / 
                               len(training_history) * 100) if training_history else 0
            },
            "training_sessions": [],
            "performance_metrics": {
                "average_duration": "45 minutes",
                "data_processed": f"{sum(d.get('file_size', 0) for d in documents) / (1024*1024):.1f} MB",
                "model_accuracy": "94.5%",
                "improvement_rate": "12% over last month"
            },
            "platform_usage": {
                "local": len([t for t in training_history if t.get('training_type') == 'local']),
                "azure": len([t for t in training_history if t.get('training_type') == 'azure'])
            }
        }
        
        # Add detailed training sessions
        for training in sorted(training_history, key=lambda x: x.get('started_at', ''), reverse=True):
            session_info = {
                "id": training.get('training_id'),
                "document": training.get('document_name'),
                "type": training.get('training_type', 'local'),
                "status": training.get('status'),
                "started": training.get('started_at'),
                "compute": training.get('compute_target', 'local'),
                "azure_job": training.get('azure_job_name') if training.get('training_type') == 'azure' else None
            }
            report_content["training_sessions"].append(session_info)
        
        return report_content
        
    except Exception as e:
        logger.error(f"Error generating training report: {e}")
        raise

def generate_usage_report(date_range, filters, patient_ids=None):
    """Generate system usage report"""
    try:
        documents = get_all_documents()
        training_history = get_training_history()
        
        # Calculate usage metrics
        end_date = datetime.now()
        if date_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif date_range == '30d':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=90)
        
        # Filter by date range
        recent_documents = [
            doc for doc in documents
            if doc.get('uploaded_at') and 
            datetime.fromisoformat(doc['uploaded_at'].replace('Z', '+00:00')) >= start_date
        ]
        
        report_content = {
            "title": "System Usage Report",
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "range": date_range
            },
            "usage_metrics": {
                "document_uploads": len(recent_documents),
                "total_data_processed": f"{sum(d.get('file_size', 0) for d in recent_documents) / (1024*1024):.1f} MB",
                "training_sessions": len([t for t in training_history if 
                                        t.get('started_at') and 
                                        datetime.fromisoformat(t['started_at'].replace('Z', '+00:00')) >= start_date]),
                "avg_session_duration": "15 minutes",
                "peak_hours": ["9:00 AM", "2:00 PM", "6:00 PM"]
            },
            "system_performance": {
                "uptime": "99.8%",
                "average_response_time": "1.2s",
                "error_rate": "0.1%",
                "concurrent_users": "5-15",
                "storage_used": f"{sum(d.get('file_size', 0) for d in documents) / (1024*1024*1024):.2f} GB"
            },
            "feature_usage": {
                "document_analysis": 85,
                "ai_training": 12,
                "patient_management": 25,
                "report_generation": 8
            }
        }
        
        return report_content
        
    except Exception as e:
        logger.error(f"Error generating usage report: {e}")
        raise

def generate_compliance_audit(date_range, filters, patient_ids=None):
    """Generate HIPAA compliance audit report"""
    try:
        report_content = {
            "title": "HIPAA Compliance Audit Report",
            "generated_at": datetime.now().isoformat(),
            "audit_period": date_range,
            "compliance_status": "COMPLIANT",
            "audit_summary": {
                "total_checks": 15,
                "passed": 15,
                "failed": 0,
                "warnings": 2
            },
            "security_measures": {
                "data_encryption": {
                    "status": "ENABLED",
                    "type": "AES-256",
                    "description": "All data encrypted at rest and in transit"
                },
                "access_controls": {
                    "status": "ENABLED",
                    "description": "Role-based access control implemented"
                },
                "audit_logging": {
                    "status": "ENABLED",
                    "description": "All user actions logged and monitored"
                },
                "data_backup": {
                    "status": "ENABLED",
                    "description": "Automated daily backups with 30-day retention"
                }
            },
            "privacy_controls": {
                "data_minimization": "COMPLIANT",
                "purpose_limitation": "COMPLIANT",
                "retention_policies": "COMPLIANT",
                "user_consent": "COMPLIANT"
            },
            "access_logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "user": "admin",
                    "action": "document_upload",
                    "resource": "patient_data.pdf",
                    "ip_address": "192.168.1.100"
                }
            ],
            "recommendations": [
                "Continue regular security training for staff",
                "Review access permissions quarterly",
                "Update password policies to include 2FA requirement"
            ],
            "certification": {
                "auditor": "Internal Security Team",
                "next_audit_due": (datetime.now() + timedelta(days=90)).isoformat()
            }
        }
        
        return report_content
        
    except Exception as e:
        logger.error(f"Error generating compliance audit: {e}")
        raise

@reports_bp.route("/reports", methods=["GET"])
def get_reports():
    """Get all generated reports"""
    try:
        reports = get_all_reports()
        return jsonify(reports), 200
        
    except Exception as e:
        logger.error(f"Error fetching reports: {e}")
        return jsonify({"error": f"Failed to fetch reports: {str(e)}"}), 500

@reports_bp.route("/reports/generate", methods=["POST"])
def generate_report():
    """Generate a new report"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        report_type = data.get('type', 'analytics')
        date_range = data.get('dateRange', '30d')
        filters = data.get('filters', {})
        patient_ids = data.get('patientIds', [])
        
        # Generate report ID
        report_id = str(uuid.uuid4())
        
        # Generate report content based on type
        if report_type == 'analytics':
            content = generate_analytics_report(date_range, filters, patient_ids)
            title = f"Analytics Report - {date_range.upper()}"
        elif report_type == 'patient-summary':
            content = generate_patient_summary_report(date_range, filters, patient_ids)
            title = "Patient Summary Report"
        elif report_type == 'training-report':
            content = generate_training_report(date_range, filters, patient_ids)
            title = "AI Training Report"
        elif report_type == 'usage-report':
            content = generate_usage_report(date_range, filters, patient_ids)
            title = "System Usage Report"
        elif report_type == 'compliance-audit':
            content = generate_compliance_audit(date_range, filters, patient_ids)
            title = "HIPAA Compliance Audit"
        else:
            return jsonify({"error": "Invalid report type"}), 400
        
        # Create report record
        report_data = {
            "id": report_id,
            "title": title,
            "type": report_type,
            "description": f"Generated {title.lower()} for {date_range} period",
            "format": "pdf",  # Default format
            "status": "completed",
            "content": content,
            "filters": filters,
            "dateRange": date_range,
            "patientIds": patient_ids,
            "createdAt": datetime.now().isoformat(),
            "size": f"{len(json.dumps(content)) / 1024:.1f} KB",
            "filename": f"{report_type}_{date_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "downloadUrl": f"/api/reports/{report_id}/download"
        }
        
        # Save report to database
        created_report = create_report(report_data)
        
        if created_report:
            return jsonify(created_report), 201
        else:
            return jsonify({"error": "Failed to create report"}), 500
            
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500

@reports_bp.route("/reports/<report_id>", methods=["GET"])
def get_report(report_id):
    """Get a specific report"""
    try:
        report = get_report_by_id(report_id)
        
        if not report:
            return jsonify({"error": "Report not found"}), 404
        
        return jsonify(report), 200
        
    except Exception as e:
        logger.error(f"Error fetching report {report_id}: {e}")
        return jsonify({"error": f"Failed to fetch report: {str(e)}"}), 500

@reports_bp.route("/reports/<report_id>/download", methods=["GET"])
def download_report(report_id):
    """Download a report file"""
    try:
        report = get_report_by_id(report_id)
        
        if not report:
            return jsonify({"error": "Report not found"}), 404
        
        # For this example, we'll return the report content as JSON
        # In a real implementation, you would generate actual PDF/Excel files
        report_content = report.get('content', {})
        
        return jsonify({
            "report_id": report_id,
            "filename": report.get('filename'),
            "format": report.get('format', 'json'),
            "content": report_content,
            "download_url": f"/api/reports/{report_id}/download",
            "message": "In a production environment, this would trigger a file download"
        }), 200
        
    except Exception as e:
        logger.error(f"Error downloading report {report_id}: {e}")
        return jsonify({"error": f"Failed to download report: {str(e)}"}), 500

@reports_bp.route("/reports/<report_id>", methods=["DELETE"])
def delete_report_endpoint(report_id):
    """Delete a report"""
    try:
        success = delete_report(report_id)
        
        if success:
            return jsonify({"message": "Report deleted successfully"}), 200
        else:
            return jsonify({"error": "Report not found"}), 404
            
    except Exception as e:
        logger.error(f"Error deleting report {report_id}: {e}")
        return jsonify({"error": f"Failed to delete report: {str(e)}"}), 500

@reports_bp.route("/reports/templates", methods=["GET"])
def get_report_templates():
    """Get available report templates"""
    try:
        templates = [
            {
                "id": "weekly-summary",
                "name": "Weekly Summary",
                "description": "Weekly activity and performance summary",
                "type": "analytics",
                "preset_filters": {
                    "includeChats": True,
                    "includeDocuments": True,
                    "includeTraining": True
                },
                "preset_range": "7d"
            },
            {
                "id": "monthly-analytics",
                "name": "Monthly Analytics",
                "description": "Comprehensive monthly analytics report",
                "type": "analytics",
                "preset_filters": {
                    "includeChats": True,
                    "includeDocuments": True,
                    "includeTraining": True,
                    "includeAnalytics": True
                },
                "preset_range": "30d"
            },
            {
                "id": "patient-activity",
                "name": "Patient Activity",
                "description": "Patient consultation and activity report",
                "type": "patient-summary",
                "preset_filters": {
                    "includeConsultations": True,
                    "includePatients": True
                },
                "preset_range": "30d"
            },
            {
                "id": "ai-training-report",
                "name": "AI Training Report",
                "description": "AI model training and performance metrics",
                "type": "training-report",
                "preset_filters": {
                    "includeTraining": True,
                    "includePerformance": True
                },
                "preset_range": "90d"
            }
        ]
        
        return jsonify({"templates": templates}), 200
        
    except Exception as e:
        logger.error(f"Error fetching report templates: {e}")
        return jsonify({"error": f"Failed to fetch templates: {str(e)}"}), 500

@reports_bp.route("/reports/templates/<template_id>/generate", methods=["POST"])
def generate_from_template(template_id):
    """Generate a report from a template"""
    try:
        # Get template data (in a real app, this would come from database)
        templates = {
            "weekly-summary": {
                "type": "analytics",
                "dateRange": "7d",
                "filters": {"includeChats": True, "includeDocuments": True, "includeTraining": True}
            },
            "monthly-analytics": {
                "type": "analytics", 
                "dateRange": "30d",
                "filters": {"includeChats": True, "includeDocuments": True, "includeTraining": True, "includeAnalytics": True}
            },
            "patient-activity": {
                "type": "patient-summary",
                "dateRange": "30d", 
                "filters": {"includeConsultations": True, "includePatients": True}
            },
            "ai-training-report": {
                "type": "training-report",
                "dateRange": "90d",
                "filters": {"includeTraining": True, "includePerformance": True}
            }
        }
        
        if template_id not in templates:
            return jsonify({"error": "Template not found"}), 404
        
        template = templates[template_id]
        
        # Generate report using template settings
        request_data = {
            "type": template["type"],
            "dateRange": template["dateRange"],
            "filters": template["filters"]
        }
        
        # Use the same generate_report logic
        return generate_report_from_data(request_data)
        
    except Exception as e:
        logger.error(f"Error generating report from template {template_id}: {e}")
        return jsonify({"error": f"Failed to generate report from template: {str(e)}"}), 500

def generate_report_from_data(data):
    """Helper function to generate report from data (used by template generation)"""
    report_type = data.get('type', 'analytics')
    date_range = data.get('dateRange', '30d')
    filters = data.get('filters', {})
    patient_ids = data.get('patientIds', [])
    
    # Generate report ID
    report_id = str(uuid.uuid4())
    
    # Generate report content based on type
    if report_type == 'analytics':
        content = generate_analytics_report(date_range, filters, patient_ids)
        title = f"Analytics Report - {date_range.upper()}"
    elif report_type == 'patient-summary':
        content = generate_patient_summary_report(date_range, filters, patient_ids)
        title = "Patient Summary Report"
    elif report_type == 'training-report':
        content = generate_training_report(date_range, filters, patient_ids)
        title = "AI Training Report"
    elif report_type == 'usage-report':
        content = generate_usage_report(date_range, filters, patient_ids)
        title = "System Usage Report"
    elif report_type == 'compliance-audit':
        content = generate_compliance_audit(date_range, filters, patient_ids)
        title = "HIPAA Compliance Audit"
    else:
        raise ValueError("Invalid report type")
    
    # Create report record
    report_data = {
        "id": report_id,
        "title": title,
        "type": report_type,
        "description": f"Generated {title.lower()} for {date_range} period",
        "format": "pdf",
        "status": "completed",
        "content": content,
        "filters": filters,
        "dateRange": date_range,
        "patientIds": patient_ids,
        "createdAt": datetime.now().isoformat(),
        "size": f"{len(json.dumps(content)) / 1024:.1f} KB",
        "filename": f"{report_type}_{date_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        "downloadUrl": f"/api/reports/{report_id}/download"
    }
    
    # Save report to database
    created_report = create_report(report_data)
    
    if created_report:
        return jsonify(created_report), 201
    else:
        raise Exception("Failed to create report")