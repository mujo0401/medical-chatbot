// src/components/documents/DocumentsTab.jsx

import React, { useState, useRef, useEffect } from 'react';
import {
  Upload,
  RefreshCw,
  Trash2,
  FileText,
  FileCheck,
  CheckCircle,
  Clock,
  X,
  CheckSquare,
  Square,
  AlertTriangle,
  AlertCircle,
  Info,
  Loader,
} from 'lucide-react';

// Import training components and hook
import { useTraining, useAzureValidation } from '../../hooks';
import { TrainingMonitor, TrainingControls } from '../training';
import ConnectionStatus from '../training/ConnectionStatus';

// Helper function to format file size
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// File validation constants
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_TYPES = {
  'application/pdf': 'pdf',
  'text/plain': 'txt',
  'text/markdown': 'md',
  'text/x-markdown': 'md',
};

// Generate unique file identifier
const generateFileId = (file) => {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 15);
  return `${file.name}-${file.size}-${timestamp}-${random}`;
};

// Check if file already exists in documents
const isDuplicateFile = (file, existingDocuments) => {
  return existingDocuments.some(
    (doc) =>
      (doc.original_name === file.name || doc.name === file.name) &&
      doc.file_size === file.size
  );
};

const DocumentsTab = ({
  documents = [],
  onDocumentUploaded = () => {},
  onDocumentDeleted = () => {},
  backendConnected = false,
  modelStatus = null,
  backendUrl = 'http://localhost:5000',
}) => {
  // Document management state
  const [dragOver, setDragOver] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessingUpload, setIsProcessingUpload] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const [uploadErrors, setUploadErrors] = useState([]);
  const [activeUploads, setActiveUploads] = useState(new Set());

  const fileInputRef = useRef();

  // Use training hook
  const trainingData = useTraining(backendUrl, backendConnected, modelStatus);

  // Use stable Azure validation
  const azureValidation = useAzureValidation(selectedDocuments, backendConnected, trainingData);

  // Debug logging for selection changes (remove in production)
  useEffect(() => {
    console.log('Document selection changed:', {
      count: selectedDocuments.length,
      documents: selectedDocuments,
      backendConnected,
      azureAvailable: trainingData.getAzureStatus().available,
    });
  }, [selectedDocuments, backendConnected, trainingData.getAzureStatus]);

  // File validation function
  const validateFile = (file) => {
    const errors = [];
    if (file.size > MAX_FILE_SIZE) {
      errors.push(`File size exceeds ${formatFileSize(MAX_FILE_SIZE)} limit`);
    }
    if (!ALLOWED_TYPES[file.type]) {
      errors.push(`File type ${file.type || 'unknown'} is not supported`);
    }
    if (!file.name || file.name.trim() === '') {
      errors.push('File name is empty');
    }
    const fileName = file.name.toLowerCase();
    const allowedExtensions = ['pdf', 'txt', 'md'];
    const fileExtension = fileName.split('.').pop();
    if (!allowedExtensions.includes(fileExtension)) {
      errors.push(`File extension .${fileExtension} is not allowed`);
    }
    if (isDuplicateFile(file, documents)) {
      errors.push('A file with the same name and size already exists');
    }
    return errors;
  };

  // Upload a single file with retry logic
  const uploadSingleFile = async (file, retryAttempt = 0) => {
    const maxRetries = 2;
    const fileId = generateFileId(file);
    const uploadKey = `${file.name}-${file.size}`;

    if (activeUploads.has(uploadKey)) {
      throw new Error('Upload already in progress for this file');
    }

    try {
      setActiveUploads((prev) => new Set([...prev, uploadKey]));
      setUploadProgress((prev) => ({
        ...prev,
        [fileId]: { progress: 0, status: 'uploading', fileName: file.name },
      }));

      const formData = new FormData();
      formData.append('file', file);
      formData.append('upload_id', fileId);
      formData.append('client_timestamp', Date.now().toString());

      const response = await fetch(`${backendUrl}/api/upload`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(60000),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      setUploadProgress((prev) => ({
        ...prev,
        [fileId]: { progress: 100, status: 'completed', fileName: file.name },
      }));

      setActiveUploads((prev) => {
        const newSet = new Set(prev);
        newSet.delete(uploadKey);
        return newSet;
      });

      return { filename: file.name, success: true, result, fileId };
    } catch (error) {
      const shouldRetry =
        retryAttempt < maxRetries &&
        !error.message.includes('already exists') &&
        !error.message.includes('already in progress') &&
        !error.message.includes('timeout') &&
        !error.message.includes('400') &&
        !error.message.includes('413');

      if (shouldRetry) {
        await new Promise((resolve) => setTimeout(resolve, 1000 * Math.pow(2, retryAttempt)));
        return uploadSingleFile(file, retryAttempt + 1);
      } else {
        setUploadProgress((prev) => ({
          ...prev,
          [fileId]: { progress: 0, status: 'failed', fileName: file.name, error: error.message },
        }));
        setActiveUploads((prev) => {
          const newSet = new Set(prev);
          newSet.delete(uploadKey);
          return newSet;
        });
        throw error;
      }
    }
  };

  // Handle file selection / drag-and-drop → validate & upload
  const handleFileUpload = async (files) => {
    if (!files || files.length === 0 || isProcessingUpload) return;

    const fileArray = Array.from(files);
    setIsProcessingUpload(true);
    setIsUploading(true);
    setUploadErrors([]);
    setUploadProgress({});

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    const uploadResults = [];
    const validationErrors = [];

    try {
      // Validate all files first
      const validFiles = [];
      for (const file of fileArray) {
        const errors = validateFile(file);
        if (errors.length > 0) {
          validationErrors.push({ fileName: file.name, errors });
        } else {
          validFiles.push(file);
        }
      }

      if (validationErrors.length > 0) {
        setUploadErrors(validationErrors);
      }
      if (validFiles.length === 0) {
        throw new Error('No valid files to upload');
      }

      // Upload valid files sequentially
      for (const file of validFiles) {
        try {
          const result = await uploadSingleFile(file);
          uploadResults.push(result);
          // Tiny delay between uploads
          if (validFiles.indexOf(file) < validFiles.length - 1) {
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
        } catch (error) {
          uploadResults.push({ filename: file.name, success: false, error: error.message });
        }
      }

      const successCount = uploadResults.filter((r) => r.success).length;
      const failureCount = uploadResults.filter((r) => !r.success).length;

      if (successCount > 0) {
        onDocumentUploaded(); // Refresh document list after upload
      }

      let message = '';
      if (successCount > 0) {
        message += ` ${successCount} file${successCount > 1 ? 's' : ''} uploaded successfully!`;
      }
      if (failureCount > 0) {
        message += `${message ? '\n' : ''} ${failureCount} file${
          failureCount > 1 ? 's' : ''
        } failed to upload.`;
      }
      if (message) {
        alert(message);
      }
    } catch (error) {
      alert('Upload failed: ' + error.message);
    } finally {
      setIsUploading(false);
      setIsProcessingUpload(false);
      setActiveUploads(new Set());
      setTimeout(() => {
        setUploadProgress({});
      }, 3000);
    }
  };

  // Clear all documents
  const clearAllDocuments = async () => {
    if (documents.length === 0) return;
    if (!window.confirm('Are you sure you want to delete ALL uploaded documents?')) {
      return;
    }
    try {
      const deletePromises = documents.map((doc) =>
        fetch(`${backendUrl}/api/documents/${doc.id}`, { method: 'DELETE' })
      );
      await Promise.allSettled(deletePromises);
      onDocumentDeleted();
      setSelectedDocuments([]);
      alert('All documents deleted successfully.');
    } catch (error) {
      alert(`Error clearing documents: ${error.message}`);
    }
  };

  // Fetch documents
  const fetchDocuments = () => {
    onDocumentUploaded();
  };

  // Selection functions
  const clearSelection = () => {
    setSelectedDocuments([]);
  };

  const toggleSelectAll = () => {
    if (selectedDocuments.length === documents.length) {
      setSelectedDocuments([]);
    } else {
      setSelectedDocuments(documents.map((doc) => doc.id));
    }
  };

  const toggleDocumentSelection = (docId) => {
    if (selectedDocuments.includes(docId)) {
      setSelectedDocuments((prev) => prev.filter((id) => id !== docId));
    } else {
      setSelectedDocuments((prev) => [...prev, docId]);
    }
  };

  const allSelected = documents.length > 0 && selectedDocuments.length === documents.length;

  const handleDeleteDocument = async (e, docId) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this document?')) {
      return;
    }
    try {
      const response = await fetch(`${backendUrl}/api/documents/${docId}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const errorText = await response.text();
        alert('Failed to delete document: ' + errorText);
        return;
      }
      onDocumentDeleted();
      setSelectedDocuments((prev) => prev.filter((id) => id !== docId));
    } catch (error) {
      alert('Error deleting document: ' + error.message);
    }
  };

  return (
    <div className="tab-content">
      {/* Header with Upload / Refresh / Clear All */}
      <div className="documents-header">
        <h3>Medical Documents</h3>
        <div className="documents-actions">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="btn btn-primary"
            disabled={isUploading || !backendConnected}
            title={!backendConnected ? 'Backend not connected' : 'Upload medical documents'}
          >
            <Upload className="w-4 h-4" />
            {isUploading ? 'Uploading...' : 'Upload Files'}
          </button>
          <button
            onClick={fetchDocuments}
            className="btn btn-secondary"
            disabled={!backendConnected}
            title="Refresh document list"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
          <button
            onClick={clearAllDocuments}
            className="btn btn-secondary"
            style={{ backgroundColor: '#f87171', color: 'white' }}
            disabled={documents.length === 0 || !backendConnected}
            title="Delete all uploaded documents"
          >
            <Trash2 className="w-4 h-4" />
            Clear All
          </button>
        </div>
      </div>

      {/* Training Monitor - Only shows when training is active */}
      <TrainingMonitor
        realTimeStatus={trainingData.realTimeStatus}
        azureJobs={trainingData.azureJobs}
        loadingAzureJobs={trainingData.loadingAzureJobs}
        trainingMetrics={trainingData.trainingMetrics}
        notifications={trainingData.notifications}
      />

      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        multiple
        accept=".pdf,txt,md"
        onChange={(e) => handleFileUpload(e.target.files)}
      />

      {/* Upload validation errors */}
      {uploadErrors.length > 0 && (
        <div className="upload-error-container">
          <div className="upload-error-header">
            <AlertCircle className="w-4 h-4 text-red-600" />
            <span>Upload Errors</span>
          </div>
          {uploadErrors.map((err, idx) => (
            <div key={idx} className="upload-error-item">
              <strong>{err.fileName}:</strong> {err.errors.join('; ')}
            </div>
          ))}
          <button className="upload-error-dismiss" onClick={() => setUploadErrors([])}>
            Dismiss
          </button>
        </div>
      )}

      {/* Upload progress indicators */}
      {Object.keys(uploadProgress).length > 0 && (
        <div className="upload-progress-container">
          <div className="upload-progress-header">
            <Loader className="w-4 h-4 spinning text-blue-600" />
            <span>Upload Progress</span>
          </div>
          {Object.entries(uploadProgress).map(([fileId, info]) => (
            <div key={fileId} className="upload-progress-item">
              <div className="upload-progress-row">
                <span className="upload-progress-filename">{info.fileName}</span>
                <span className={`upload-progress-status ${info.status}`}>
                  {info.status === 'uploading'
                    ? `${info.progress}%`
                    : info.status === 'completed'
                    ? '✓'
                    : '✗'}
                </span>
              </div>
              {info.error && <div className="upload-progress-error">{info.error}</div>}
            </div>
          ))}
        </div>
      )}

      {/* Upload zone */}
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragOver(false);
          if (isProcessingUpload) return;
          const files = e.dataTransfer.files;
          if (files && files.length > 0) {
            handleFileUpload(files);
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragOver(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragOver(false);
        }}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (isProcessingUpload) return;
          fileInputRef.current?.click();
        }}
        style={{
          pointerEvents: isProcessingUpload ? 'none' : 'auto',
          opacity: isProcessingUpload ? 0.7 : 1,
          background: dragOver ? 'rgba(59, 130, 246, 0.1)' : 'rgba(243, 244, 246, 0.8)',
          border: dragOver ? '2px dashed #3b82f6' : '2px dashed #d1d5db',
          borderRadius: '12px',
          padding: '40px 20px',
          textAlign: 'center',
          cursor: isProcessingUpload ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
        }}
      >
        <Upload
          className="upload-icon"
          style={{
            width: '48px',
            height: '48px',
            color: dragOver ? '#3b82f6' : '#9ca3af',
            marginBottom: '16px',
          }}
        />
        <p
          style={{
            fontSize: '16px',
            color: dragOver ? '#3b82f6' : '#6b7280',
            fontWeight: '600',
            marginBottom: '8px',
          }}
        >
          {isProcessingUpload
            ? 'Processing upload...'
            : dragOver
            ? 'Drop files here!'
            : 'Drop PDF or text files here, or click to browse'}
        </p>
        <small style={{ color: '#9ca3af' }}>
          Supported formats: PDF, TXT, MD • Max size: {formatFileSize(MAX_FILE_SIZE)}
        </small>
      </div>

      {/* Selection controls */}
      {documents.length > 0 && (
        <div
          className="selection-controls"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '16px 0',
            borderBottom: '1px solid #e5e7eb',
          }}
        >
          <button
            onClick={toggleSelectAll}
            className="btn btn-secondary"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '14px',
              padding: '8px 16px',
            }}
          >
            {allSelected ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4" />}
            {allSelected ? 'Deselect All' : 'Select All'}
          </button>
          {selectedDocuments.length > 0 && (
            <span className="selection-info" style={{ fontSize: '14px', color: '#6b7280' }}>
              {selectedDocuments.length} of {documents.length} documents selected
            </span>
          )}
        </div>
      )}

      {/* List of documents */}
      {documents.length > 0 && (
        <div className="documents-list" style={{ marginTop: '16px' }}>
          {documents.map((doc) => {
            const isSelected = selectedDocuments.includes(doc.id);
            return (
              <div
                key={doc.id}
                className={`document-item ${isSelected ? 'selected' : ''}`}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '12px 16px',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  marginBottom: '8px',
                  backgroundColor: isSelected ? 'rgba(59, 130, 246, 0.1)' : '#fff',
                  cursor: 'pointer',
                }}
                onClick={() => toggleDocumentSelection(doc.id)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {isSelected ? (
                    <CheckSquare className="w-5 h-5 text-blue-600" />
                  ) : (
                    <Square className="w-5 h-5 text-gray-400" />
                  )}
                  <span>{doc.original_name}</span>
                </div>
                <button
                  onClick={(e) => handleDeleteDocument(e, doc.id)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    color: '#ef4444',
                  }}
                  title="Delete this document"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* Training Controls */}
      <TrainingControls
        selectedDocuments={selectedDocuments}
        onClearSelection={clearSelection}
        onStartTraining={trainingData.startTraining}
        trainingData={trainingData}
        backendConnected={backendConnected}
      />
    </div>
  );
};

export default DocumentsTab;