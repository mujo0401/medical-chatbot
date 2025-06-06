// src/components/patients/PatientsTab.jsx

import React, { useState, useEffect } from 'react';
import {
  Users, Plus, Search, Filter, MoreVertical, User, Phone, Mail,
  Calendar, FileText, Heart, Activity, AlertTriangle, CheckCircle,
  Edit, Trash2, Eye, Clock, MapPin, Stethoscope, Pill, X, Save
} from 'lucide-react';

const PatientsTab = ({
  backendConnected = false,
  chatSessions = []
}) => {
  const [patients, setPatients] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [showAddPatient, setShowAddPatient] = useState(false);
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortBy, setSortBy] = useState('name');
  const [isLoading, setIsLoading] = useState(false);

  // New patient form state
  const [newPatient, setNewPatient] = useState({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    dateOfBirth: '',
    gender: '',
    address: '',
    medicalConditions: '',
    allergies: '',
    medications: '',
    emergencyContact: '',
    insuranceProvider: '',
    notes: ''
  });

// Fetch patients from backend
  const fetchPatients = async () => {
    if (!backendConnected) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/patients');
      if (response.ok) {
        const data = await response.json();
        setPatients(Array.isArray(data.patients) ? data.patients : []);
      }
    } catch (error) {
      console.error('Failed to fetch patients:', error);
      setPatients([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, [backendConnected]);

  // Filter and sort patients
  const filteredPatients = patients
    .filter(patient => {
      const matchesSearch = `${patient.firstName} ${patient.lastName}`
        .toLowerCase()
        .includes(searchTerm.toLowerCase()) ||
        patient.email?.toLowerCase().includes(searchTerm.toLowerCase());
      
      if (filterStatus === 'all') return matchesSearch;
      return matchesSearch && patient.status === filterStatus;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return `${a.firstName} ${a.lastName}`.localeCompare(`${b.firstName} ${b.lastName}`);
        case 'date':
          return new Date(b.lastVisit || b.createdAt) - new Date(a.lastVisit || a.createdAt);
        case 'status':
          return a.status?.localeCompare(b.status || '') || 0;
        default:
          return 0;
      }
    });

  // Add new patient
  const handleAddPatient = async () => {
    if (!newPatient.firstName || !newPatient.lastName) {
      alert('First name and last name are required');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/patients', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...newPatient,
          status: 'active',
          createdAt: new Date().toISOString()
        })
      });

      if (response.ok) {
        fetchPatients();
        setShowAddPatient(false);
        setNewPatient({
          firstName: '', lastName: '', email: '', phone: '', dateOfBirth: '',
          gender: '', address: '', medicalConditions: '', allergies: '',
          medications: '', emergencyContact: '', insuranceProvider: '', notes: ''
        });
      }
    } catch (error) {
      console.error('Failed to add patient:', error);
    }
  };

  const PatientCard = ({ patient }) => {
    const age = patient.dateOfBirth 
      ? Math.floor((new Date() - new Date(patient.dateOfBirth)) / (365.25 * 24 * 60 * 60 * 1000))
      : 'N/A';

    return (
      <div className="patient-card" onClick={() => setSelectedPatient(patient)}>
        <div className="patient-avatar">
          <User className="w-8 h-8" />
          <div className={`status-indicator ${patient.status || 'active'}`} />
        </div>
        <div className="patient-info">
          <div className="patient-name">
            {patient.firstName} {patient.lastName}
          </div>
          <div className="patient-details">
            <div className="detail-item">
              <Calendar className="w-3 h-3" />
              <span>Age: {age}</span>
            </div>
            <div className="detail-item">
              <Mail className="w-3 h-3" />
              <span>{patient.email || 'No email'}</span>
            </div>
            <div className="detail-item">
              <Phone className="w-3 h-3" />
              <span>{patient.phone || 'No phone'}</span>
            </div>
          </div>
          {patient.medicalConditions && (
            <div className="medical-conditions">
              <Stethoscope className="w-3 h-3" />
              <span>{patient.medicalConditions}</span>
            </div>
          )}
        </div>
        <div className="patient-actions">
          <div className="last-visit">
            <Clock className="w-3 h-3" />
            <span>{patient.lastVisit ? new Date(patient.lastVisit).toLocaleDateString() : 'No visits'}</span>
          </div>
          <button className="action-btn">
            <MoreVertical className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  };

  const PatientModal = ({ patient, onClose }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [editData, setEditData] = useState(patient);

    const patientSessions = chatSessions.filter(session => 
      session.patientId === patient.id
    );

    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="patient-modal" onClick={e => e.stopPropagation()}>
          <div className="modal-header">
            <div className="patient-header-info">
              <div className="patient-avatar large">
                <User className="w-12 h-12" />
              </div>
              <div>
                <h2>{patient.firstName} {patient.lastName}</h2>
                <div className="patient-id">ID: {patient.id}</div>
              </div>
            </div>
            <div className="modal-actions">
              <button 
                onClick={() => setIsEditing(!isEditing)}
                className="btn btn-secondary"
              >
                <Edit className="w-4 h-4" />
                {isEditing ? 'Cancel' : 'Edit'}
              </button>
              <button onClick={onClose} className="btn btn-secondary">
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="modal-content">
            <div className="patient-sections">
              {/* Personal Information */}
              <div className="info-section">
                <h3>Personal Information</h3>
                <div className="info-grid">
                  <div className="info-item">
                    <label>Email:</label>
                    {isEditing ? (
                      <input 
                        type="email" 
                        value={editData.email || ''} 
                        onChange={(e) => setEditData({...editData, email: e.target.value})}
                      />
                    ) : (
                      <span>{patient.email || 'Not provided'}</span>
                    )}
                  </div>
                  <div className="info-item">
                    <label>Phone:</label>
                    {isEditing ? (
                      <input 
                        type="tel" 
                        value={editData.phone || ''} 
                        onChange={(e) => setEditData({...editData, phone: e.target.value})}
                      />
                    ) : (
                      <span>{patient.phone || 'Not provided'}</span>
                    )}
                  </div>
                  <div className="info-item">
                    <label>Date of Birth:</label>
                    {isEditing ? (
                      <input 
                        type="date" 
                        value={editData.dateOfBirth || ''} 
                        onChange={(e) => setEditData({...editData, dateOfBirth: e.target.value})}
                      />
                    ) : (
                      <span>{patient.dateOfBirth ? new Date(patient.dateOfBirth).toLocaleDateString() : 'Not provided'}</span>
                    )}
                  </div>
                  <div className="info-item">
                    <label>Gender:</label>
                    {isEditing ? (
                      <select 
                        value={editData.gender || ''} 
                        onChange={(e) => setEditData({...editData, gender: e.target.value})}
                      >
                        <option value="">Select gender</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                      </select>
                    ) : (
                      <span>{patient.gender || 'Not specified'}</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Medical Information */}
              <div className="info-section">
                <h3>Medical Information</h3>
                <div className="info-grid">
                  <div className="info-item full-width">
                    <label>Medical Conditions:</label>
                    {isEditing ? (
                      <textarea 
                        value={editData.medicalConditions || ''} 
                        onChange={(e) => setEditData({...editData, medicalConditions: e.target.value})}
                        rows="3"
                      />
                    ) : (
                      <span>{patient.medicalConditions || 'None reported'}</span>
                    )}
                  </div>
                  <div className="info-item full-width">
                    <label>Allergies:</label>
                    {isEditing ? (
                      <textarea 
                        value={editData.allergies || ''} 
                        onChange={(e) => setEditData({...editData, allergies: e.target.value})}
                        rows="2"
                      />
                    ) : (
                      <span>{patient.allergies || 'None reported'}</span>
                    )}
                  </div>
                  <div className="info-item full-width">
                    <label>Current Medications:</label>
                    {isEditing ? (
                      <textarea 
                        value={editData.medications || ''} 
                        onChange={(e) => setEditData({...editData, medications: e.target.value})}
                        rows="3"
                      />
                    ) : (
                      <span>{patient.medications || 'None reported'}</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Consultation History */}
              <div className="info-section">
                <h3>Recent Consultations</h3>
                <div className="consultations-list">
                  {patientSessions.length > 0 ? (
                    patientSessions.slice(0, 5).map((session, index) => (
                      <div key={index} className="consultation-item">
                        <div className="consultation-icon">
                          <Stethoscope className="w-4 h-4" />
                        </div>
                        <div className="consultation-info">
                          <div className="consultation-title">{session.name}</div>
                          <div className="consultation-date">
                            {new Date(session.timestamp).toLocaleDateString()}
                          </div>
                          <div className="consultation-summary">{session.lastMessage}</div>
                        </div>
                        <button className="btn btn-sm">
                          <Eye className="w-3 h-3" />
                        </button>
                      </div>
                    ))
                  ) : (
                    <div className="no-consultations">
                      <FileText className="w-8 h-8" />
                      <p>No consultations yet</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {isEditing && (
            <div className="modal-footer">
              <button 
                onClick={async () => {
                  // Save changes
                  try {
                    const response = await fetch(`http://localhost:5000/api/patients/${patient.id}`, {
                      method: 'PUT',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(editData)
                    });
                    if (response.ok) {
                      fetchPatients();
                      setIsEditing(false);
                      onClose();
                    }
                  } catch (error) {
                    console.error('Failed to update patient:', error);
                  }
                }}
                className="btn btn-primary"
              >
                <Save className="w-4 h-4" />
                Save Changes
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="tab-content patients-tab">
      {/* Patients Header */}
      <div className="patients-header">
        <div className="patients-title">
          <Users className="w-6 h-6" />
          <h2>Patient Management</h2>
          <div className="patients-count">{patients.length} patients</div>
        </div>
        <div className="patients-actions">
          <button 
            onClick={() => setShowAddPatient(true)}
            className="btn btn-primary"
          >
            <Plus className="w-4 h-4" />
            Add Patient
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="patients-controls">
        <div className="search-bar">
          <Search className="w-4 h-4" />
          <input
            type="text"
            placeholder="Search patients by name or email..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="filter-controls">
          <select 
            value={filterStatus} 
            onChange={(e) => setFilterStatus(e.target.value)}
          >
            <option value="all">All Patients</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
          </select>
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
          >
            <option value="name">Sort by Name</option>
            <option value="date">Sort by Last Visit</option>
            <option value="status">Sort by Status</option>
          </select>
        </div>
      </div>

      {/* Patients List */}
      <div className="patients-list">
        {isLoading ? (
          <div className="loading-state">
            <Activity className="w-8 h-8 spinning" />
            <p>Loading patients...</p>
          </div>
        ) : filteredPatients.length > 0 ? (
          filteredPatients.map(patient => (
            <PatientCard key={patient.id} patient={patient} />
          ))
        ) : (
          <div className="empty-state">
            <Users className="w-12 h-12" />
            <h3>No patients found</h3>
            <p>Start by adding your first patient or adjust your search criteria</p>
            <button 
              onClick={() => setShowAddPatient(true)}
              className="btn btn-primary"
            >
              <Plus className="w-4 h-4" />
              Add First Patient
            </button>
          </div>
        )}
      </div>

      {/* Add Patient Modal */}
      {showAddPatient && (
        <div className="modal-overlay" onClick={() => setShowAddPatient(false)}>
          <div className="add-patient-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Add New Patient</h2>
              <button onClick={() => setShowAddPatient(false)}>
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="modal-content">
              <div className="form-grid">
                <div className="form-group">
                  <label>First Name *</label>
                  <input
                    type="text"
                    value={newPatient.firstName}
                    onChange={(e) => setNewPatient({...newPatient, firstName: e.target.value})}
                    placeholder="Enter first name"
                  />
                </div>
                <div className="form-group">
                  <label>Last Name *</label>
                  <input
                    type="text"
                    value={newPatient.lastName}
                    onChange={(e) => setNewPatient({...newPatient, lastName: e.target.value})}
                    placeholder="Enter last name"
                  />
                </div>
                <div className="form-group">
                  <label>Email</label>
                  <input
                    type="email"
                    value={newPatient.email}
                    onChange={(e) => setNewPatient({...newPatient, email: e.target.value})}
                    placeholder="Enter email address"
                  />
                </div>
                <div className="form-group">
                  <label>Phone</label>
                  <input
                    type="tel"
                    value={newPatient.phone}
                    onChange={(e) => setNewPatient({...newPatient, phone: e.target.value})}
                    placeholder="Enter phone number"
                  />
                </div>
                <div className="form-group">
                  <label>Date of Birth</label>
                  <input
                    type="date"
                    value={newPatient.dateOfBirth}
                    onChange={(e) => setNewPatient({...newPatient, dateOfBirth: e.target.value})}
                  />
                </div>
                <div className="form-group">
                  <label>Gender</label>
                  <select
                    value={newPatient.gender}
                    onChange={(e) => setNewPatient({...newPatient, gender: e.target.value})}
                  >
                    <option value="">Select gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
                <div className="form-group full-width">
                  <label>Medical Conditions</label>
                  <textarea
                    value={newPatient.medicalConditions}
                    onChange={(e) => setNewPatient({...newPatient, medicalConditions: e.target.value})}
                    placeholder="List any known medical conditions"
                    rows="3"
                  />
                </div>
                <div className="form-group full-width">
                  <label>Allergies</label>
                  <textarea
                    value={newPatient.allergies}
                    onChange={(e) => setNewPatient({...newPatient, allergies: e.target.value})}
                    placeholder="List any known allergies"
                    rows="2"
                  />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button 
                onClick={() => setShowAddPatient(false)}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button 
                onClick={handleAddPatient}
                className="btn btn-primary"
              >
                <Save className="w-4 h-4" />
                Add Patient
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Patient Detail Modal */}
      {selectedPatient && (
        <PatientModal 
          patient={selectedPatient} 
          onClose={() => setSelectedPatient(null)} 
        />
      )}
    </div>
  );
};

export default PatientsTab;