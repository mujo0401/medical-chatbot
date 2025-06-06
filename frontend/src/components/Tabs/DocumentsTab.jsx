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
  Server,
  Cloud,
  Activity,
  Shield,
  Settings,
  BarChart3,
  Timer,
  StopCircle,
  Minimize2,
  Maximize2,
} from 'lucide-react';

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
  trainingStatus = { is_training: false },
  backendUrl = 'http://localhost:5000',
}) => {
  // State management
  const [dragOver, setDragOver] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessingUpload, setIsProcessingUpload] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const [azureValidation, setAzureValidation] = useState(null);
  const [azureBillingInfo, setAzureBillingInfo] = useState(null);
  const [azureWorkspaceInfo, setAzureWorkspaceInfo] = useState(null);
  const [showBillingTooltip, setShowBillingTooltip] = useState(false);
  const [isValidatingAzure, setIsValidatingAzure] = useState(false);
  const [uploadErrors, setUploadErrors] = useState([]);
  const [activeUploads, setActiveUploads] = useState(new Set());
  const [azureError, setAzureError] = useState(null);
  const [validationPassed, setValidationPassed] = useState(false);
  const [isTrainingRequesting, setIsTrainingRequesting] = useState(false);

  // Training monitoring state
  const [realTimeStatus, setRealTimeStatus] = useState({
    is_training: false,
    progress: 0,
    status_message: 'Ready',
    current_document: null,
    start_time: null,
    estimated_completion: null,
    training_id: null,
  });
  const [azureJobs, setAzureJobs] = useState([]);
  const [loadingAzureJobs, setLoadingAzureJobs] = useState(false);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [isTrainingMinimized, setIsTrainingMinimized] = useState(false);
  const [notifications, setNotifications] = useState([]);

  const fileInputRef = useRef();

  // Training status polling
  useEffect(() => {
    if (!backendConnected) return;

    const fetchTrainingStatus = async () => {
      try {
        const res = await fetch(`${backendUrl}/api/training/status`);
        if (res.ok) {
          const data = await res.json();
          setRealTimeStatus((prev) => ({
            ...prev,
            ...data,
            start_time: data.start_time ? new Date(data.start_time) : prev.start_time,
            estimated_completion: data.estimated_completion
              ? new Date(data.estimated_completion)
              : prev.estimated_completion,
          }));
        }
      } catch (error) {
        console.error('Failed to fetch training status:', error);
      }
    };

    const fetchAzureJobs = async () => {
      if (!getAzureStatus().available) return;
      
      setLoadingAzureJobs(true);
      try {
        const res = await fetch(`${backendUrl}/api/training/azure/jobs`);
        if (res.ok) {
          const data = await res.json();
          setAzureJobs(
            (data.jobs || []).map((job) => ({
              ...job,
              creation_time: job.creation_time ? new Date(job.creation_time) : null,
              start_time: job.start_time ? new Date(job.start_time) : null,
              end_time: job.end_time ? new Date(job.end_time) : null,
            }))
          );
        }
      } catch (error) {
        console.error('Failed to fetch Azure jobs:', error);
      } finally {
        setLoadingAzureJobs(false);
      }
    };

    const fetchTrainingMetrics = async () => {
      try {
        const res = await fetch(`${backendUrl}/api/training/metrics`);
        if (res.ok) {
          const data = await res.json();
          setTrainingMetrics(data);
        }
      } catch (error) {
        console.error('Failed to fetch training metrics:', error);
      }
    };

    const fetchNotifications = async () => {
      try {
        const res = await fetch(`${backendUrl}/api/training/notifications`);
        if (res.ok) {
          const data = await res.json();
          setNotifications(data.notifications || []);
        }
      } catch (error) {
        console.error('Failed to fetch notifications:', error);
      }
    };

    // Initial fetch
    fetchTrainingStatus();
    fetchAzureJobs();
    fetchTrainingMetrics();
    fetchNotifications();

    // Set up polling intervals
    const statusInterval = setInterval(fetchTrainingStatus, 5000); // Every 5 seconds
    const jobsInterval = setInterval(fetchAzureJobs, 15000); // Every 15 seconds
    const metricsInterval = setInterval(fetchTrainingMetrics, 30000); // Every 30 seconds
    const notifInterval = setInterval(fetchNotifications, 20000); // Every 20 seconds

    return () => {
      clearInterval(statusInterval);
      clearInterval(jobsInterval);
      clearInterval(metricsInterval);
      clearInterval(notifInterval);
    };
  }, [backendConnected, backendUrl]);

  // ————————————————————————————————
  // Helper: Check Azure status based solely on backendConnected & modelStatus.azure
  // ————————————————————————————————
  const getAzureStatus = () => {
    if (!backendConnected) {
      return { available: false, reason: 'Backend not connected', type: 'connection' };
    }
    if (!modelStatus?.azure) {
      return { available: false, reason: 'Azure status unknown', type: 'unknown' };
    }
    const azure = modelStatus.azure;
    if (!azure.available) {
      return {
        available: false,
        reason: azure.reason || 'Azure ML not available',
        type: 'backend_unavailable',
        help: azure.help,
      };
    }
    if (!azure.sdk_installed) {
      return { available: false, reason: 'Azure ML SDK not installed', type: 'sdk' };
    }
    if (!azure.configured) {
      return { available: false, reason: 'Azure ML not configured', type: 'config' };
    }
    const hasWorkingComputeTargets =
      azure.compute_targets &&
      Array.isArray(azure.compute_targets) &&
      azure.compute_targets.length > 0 &&
      azure.compute_targets.some((target) => target.state === 'Succeeded');
    if (!hasWorkingComputeTargets) {
      return {
        available: false,
        reason: 'No working compute targets available',
        type: 'compute',
        help: 'Create or start a compute cluster in Azure ML Studio',
      };
    }
    return { available: true, reason: null, type: 'available' };
  };

  // ————————————————————————————————
  // File validation function
  // ————————————————————————————————
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

  // ————————————————————————————————
  // Upload a single file with retry logic
  // ————————————————————————————————
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

  // ————————————————————————————————
  // Handle file selection / drag‐and‐drop → validate & upload
  // ————————————————————————————————
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

  // ————————————————————————————————
  // Azure validation: run whenever selectedDocuments changes, but only clear once selection is empty
  // ————————————————————————————————
  useEffect(() => {
    // If no documents are selected, reset validation
    if (selectedDocuments.length === 0) {
      setAzureValidation(null);
      setAzureError(null);
      setValidationPassed(false);
      return;
    }

    // If we have documents AND Azure is available, debounce & validate
    if (backendConnected && getAzureStatus().available) {
      const debounceTimer = setTimeout(() => {
        validateAzureTraining();
      }, 500);
      return () => clearTimeout(debounceTimer);
    }
    // Otherwise, do NOT clear existing validation info
  }, [selectedDocuments, backendConnected, modelStatus]);

  // ————————————————————————————————
  // Fetch Azure billing/workspace info whenever backendConnected & Azure is available
  // ————————————————————————————————
  useEffect(() => {
    if (backendConnected && getAzureStatus().available) {
      fetchAzureBillingInfo();
      fetchAzureWorkspaceInfo();
    }
  }, [backendConnected, modelStatus]);

  // ————————————————————————————————
  // validateAzureTraining → POST /api/azure/validate-training
  // ————————————————————————————————
  const validateAzureTraining = async () => {
    if (selectedDocuments.length === 0 || !getAzureStatus().available) return;

    setIsValidatingAzure(true);
    setAzureError(null);

    try {
      const response = await fetch(`${backendUrl}/api/azure/validate-training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_ids: selectedDocuments }),
      });

      const data = await response.json();
      if (response.ok) {
        setAzureValidation(data);
        if (data.can_train) {
          setValidationPassed(true);
        }
        setAzureError(null);
      } else {
        setAzureValidation(null);
        setAzureError({
          message: data.error || 'Validation failed',
          help: data.help,
          recommendation: data.recommendation,
        });
        setValidationPassed(false);
      }
    } catch (error) {
      setAzureValidation(null);
      setAzureError({
        message: 'Failed to connect to Azure ML service',
        help: 'Check your network connection and backend status',
      });
      setValidationPassed(false);
    } finally {
      setIsValidatingAzure(false);
    }
  };

  // ————————————————————————————————
  // fetchAzureBillingInfo → GET /api/azure/billing-info
  // ————————————————————————————————
  const fetchAzureBillingInfo = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/azure/billing-info`);
      if (response.ok) {
        const billingInfo = await response.json();
        setAzureBillingInfo(billingInfo);
      }
    } catch (error) {
      console.error('Failed to fetch Azure billing info:', error);
    }
  };

  // ————————————————————————————————
  // fetchAzureWorkspaceInfo → GET /api/training/azure/workspace-info
  // ————————————————————————————————
  const fetchAzureWorkspaceInfo = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/training/azure/workspace-info`);
      if (response.ok) {
        const workspaceInfo = await response.json();
        setAzureWorkspaceInfo(workspaceInfo);
      }
    } catch (error) {
      console.error('Failed to fetch Azure workspace info:', error);
    }
  };

  // ————————————————————————————————
  // clearAllDocuments → DELETE each document → then call onDocumentDeleted()
  // ————————————————————————————————
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

  // ————————————————————————————————
  // fetchDocuments → call onDocumentUploaded() to refresh
  // ————————————————————————————————
  const fetchDocuments = () => {
    onDocumentUploaded();
  };

  // ————————————————————————————————
  // startTraining → POST /api/train with use_azure flag
  // ————————————————————————————————
  const startTraining = async (useAzure = false) => {
    if (selectedDocuments.length === 0) return;

    setIsTrainingRequesting(true);
    try {
      const requestBody = {
        document_ids: selectedDocuments,
        use_azure: useAzure,
      };
      if (useAzure && azureValidation?.compute_target) {
        requestBody.compute_target = azureValidation.compute_target;
      }

      const response = await fetch(`${backendUrl}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Training failed to start');
      }

      if (useAzure) {
        alert(
          `Azure ML training started successfully!\n\n` +
            `Training ID: ${data.training_id}\n` +
            `Compute Target: ${data.compute_target}\n` +
            `Documents: ${data.documents.join(', ')}\n` +
            `Estimated Cost: $${data.estimated_cost || 'N/A'}`
        );
      } else {
        alert(
          `Local training started successfully!\n\n` +
            `Training ID: ${data.training_id}\n` +
            `Documents: ${data.documents.join(', ')}`
        );
      }

      // Clear selection + validation after starting
      setSelectedDocuments([]);
      setValidationPassed(false);
    } catch (error) {
      // Show detailed error if available
      alert('Training failed: ' + error.message);
    } finally {
      setIsTrainingRequesting(false);
    }
  };

  // ————————————————————————————————
  // toggleSelectAll, clearSelection, toggleDocumentSelection
  // ————————————————————————————————
  const clearSelection = () => {
    setSelectedDocuments([]);
    setAzureValidation(null);
    setAzureError(null);
    setValidationPassed(false);
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

  const isAzureEnabled = () => {
    const azureStatus = getAzureStatus();
    return (
      selectedDocuments.length > 0 &&
      azureStatus.available &&
      azureValidation?.can_train &&
      !realTimeStatus?.is_training &&
      !isValidatingAzure &&
      !azureError
    );
  };

  const getAzureButtonTooltip = () => {
    const azureStatus = getAzureStatus();
    if (!azureStatus.available) return azureStatus.reason;
    if (selectedDocuments.length === 0) return 'No documents selected';
    if (isValidatingAzure) return 'Validating Azure configuration.';
    if (azureError) return azureError.message;
    if (azureValidation && !azureValidation.can_train) return azureValidation.reason;
    if (realTimeStatus?.is_training) return 'Training already in progress';
    return null;
  };

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

  // Training monitor component
  const renderTrainingMonitor = () => {
    if (!realTimeStatus.is_training && azureJobs.filter(job => ['Running', 'Starting', 'Preparing'].includes(job.status)).length === 0) {
      return null;
    }

    const runningAzureJobs = azureJobs.filter(job => ['Running', 'Starting', 'Preparing'].includes(job.status));

    return (
      <div className="training-monitor-container" style={{
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(16, 185, 129, 0.05))',
        border: '2px solid rgba(59, 130, 246, 0.2)',
        borderRadius: '16px',
        padding: isTrainingMinimized ? '12px 20px' : '20px',
        marginBottom: '24px',
        position: 'relative',
        transition: 'all 0.3s ease',
      }}>
        {/* Header with minimize/maximize */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: isTrainingMinimized ? 0 : '16px',
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '16px',
            fontWeight: '600',
            color: '#3b82f6',
          }}>
            <Activity className="w-5 h-5" />
            <span>Training Monitor</span>
            {realTimeStatus.is_training && (
              <span style={{
                fontSize: '12px',
                background: 'rgba(59, 130, 246, 0.1)',
                color: '#3b82f6',
                padding: '2px 8px',
                borderRadius: '8px',
                fontWeight: '500',
              }}>
                ACTIVE
              </span>
            )}
          </div>
          <button
            onClick={() => setIsTrainingMinimized(!isTrainingMinimized)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#6b7280',
            }}
          >
            {isTrainingMinimized ? <Maximize2 className="w-4 h-4" /> : <Minimize2 className="w-4 h-4" />}
          </button>
        </div>

        {!isTrainingMinimized && (
          <>
            {/* Local Training Status */}
            {realTimeStatus.is_training && (
              <div style={{
                background: 'rgba(255, 255, 255, 0.8)',
                padding: '16px',
                borderRadius: '12px',
                marginBottom: '16px',
                border: '1px solid rgba(59, 130, 246, 0.1)',
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '12px',
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#1f2937',
                }}>
                  <Server className="w-4 h-4" />
                  <span>Local Training Progress</span>
                </div>
                
                <div style={{
                  width: '100%',
                  height: '8px',
                  background: 'rgba(59, 130, 246, 0.1)',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  marginBottom: '8px',
                }}>
                  <div style={{
                    width: `${realTimeStatus.progress}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #3b82f6, #10b981)',
                    transition: 'width 0.3s ease',
                    borderRadius: '4px',
                  }} />
                </div>
                
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '12px',
                  color: '#6b7280',
                }}>
                  <span>{realTimeStatus.progress}% - {realTimeStatus.status_message}</span>
                  {realTimeStatus.current_document && (
                    <span style={{ 
                      background: 'rgba(59, 130, 246, 0.1)', 
                      padding: '2px 6px', 
                      borderRadius: '4px',
                      fontSize: '11px',
                    }}>
                      {realTimeStatus.current_document}
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Azure Jobs Status */}
            {runningAzureJobs.length > 0 && (
              <div style={{
                background: 'rgba(255, 255, 255, 0.8)',
                padding: '16px',
                borderRadius: '12px',
                marginBottom: '16px',
                border: '1px solid rgba(59, 130, 246, 0.1)',
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '12px',
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#1f2937',
                }}>
                  <Cloud className="w-4 h-4" />
                  <span>Azure ML Jobs</span>
                  {loadingAzureJobs && <Loader className="w-4 h-4 animate-spin" />}
                </div>
                
                {runningAzureJobs.map((job, index) => (
                  <div key={job.name} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '8px 12px',
                    background: 'rgba(59, 130, 246, 0.05)',
                    borderRadius: '8px',
                    marginBottom: index < runningAzureJobs.length - 1 ? '8px' : 0,
                    fontSize: '12px',
                  }}>
                    <div>
                      <div style={{ fontWeight: '600', color: '#1f2937' }}>
                        {job.display_name || job.name}
                      </div>
                      <div style={{ color: '#6b7280', fontSize: '11px' }}>
                        {job.compute_target} • {job.status}
                      </div>
                    </div>
                    <div style={{
                      background: job.status === 'Running' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                      color: job.status === 'Running' ? '#059669' : '#d97706',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      fontSize: '11px',
                      fontWeight: '600',
                    }}>
                      {job.status}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Quick Metrics */}
            {trainingMetrics && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                gap: '12px',
                marginBottom: '16px',
              }}>
                <div style={{
                  background: 'rgba(255, 255, 255, 0.8)',
                  padding: '12px',
                  borderRadius: '8px',
                  textAlign: 'center',
                  border: '1px solid rgba(59, 130, 246, 0.1)',
                }}>
                  <div style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>
                    {trainingMetrics.total_training_sessions}
                  </div>
                  <div style={{ fontSize: '11px', color: '#6b7280', fontWeight: '500' }}>
                    Total Sessions
                  </div>
                </div>
                <div style={{
                  background: 'rgba(255, 255, 255, 0.8)',
                  padding: '12px',
                  borderRadius: '8px',
                  textAlign: 'center',
                  border: '1px solid rgba(59, 130, 246, 0.1)',
                }}>
                  <div style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>
                    {trainingMetrics.successful_sessions}
                  </div>
                  <div style={{ fontSize: '11px', color: '#6b7280', fontWeight: '500' }}>
                    Successful
                  </div>
                </div>
                <div style={{
                  background: 'rgba(255, 255, 255, 0.8)',
                  padding: '12px',
                  borderRadius: '8px',
                  textAlign: 'center',
                  border: '1px solid rgba(59, 130, 246, 0.1)',
                }}>
                  <div style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>
                    ${trainingMetrics.total_cost || '0'}
                  </div>
                  <div style={{ fontSize: '11px', color: '#6b7280', fontWeight: '500' }}>
                    Total Cost
                  </div>
                </div>
              </div>
            )}

            {/* Notifications */}
            {notifications.length > 0 && (
              <div style={{
                background: 'rgba(255, 255, 255, 0.8)',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid rgba(59, 130, 246, 0.1)',
              }}>
                <div style={{
                  fontSize: '12px',
                  fontWeight: '600',
                  color: '#1f2937',
                  marginBottom: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}>
                  <Info className="w-3 h-3" />
                  Recent Updates
                </div>
                {notifications.slice(0, 2).map((notif, index) => (
                  <div key={notif.id || index} style={{
                    fontSize: '11px',
                    color: '#6b7280',
                    marginBottom: index < Math.min(notifications.length, 2) - 1 ? '4px' : 0,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}>
                    {notif.type === 'success' && <CheckCircle className="w-3 h-3 text-green-600" />}
                    {notif.type === 'warning' && <AlertTriangle className="w-3 h-3 text-yellow-600" />}
                    {notif.type === 'info' && <Info className="w-3 h-3 text-blue-600" />}
                    <span>{notif.message}</span>
                    <span style={{ marginLeft: 'auto', fontSize: '10px', opacity: 0.7 }}>
                      {notif.time}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* Minimized view */}
        {isTrainingMinimized && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontSize: '12px',
            color: '#6b7280',
          }}>
            {realTimeStatus.is_training && (
              <>
                <div style={{
                  width: '60px',
                  height: '4px',
                  background: 'rgba(59, 130, 246, 0.2)',
                  borderRadius: '2px',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    width: `${realTimeStatus.progress}%`,
                    height: '100%',
                    background: '#3b82f6',
                    borderRadius: '2px',
                    transition: 'width 0.3s ease',
                  }} />
                </div>
                <span>{realTimeStatus.progress}%</span>
              </>
            )}
            {runningAzureJobs.length > 0 && (
              <span>{runningAzureJobs.length} Azure job{runningAzureJobs.length > 1 ? 's' : ''} running</span>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="tab-content">
      {/* Header with Upload / Refresh / Clear All */}
      <div className="documents-header">
        <h3> Medical Documents</h3>
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
      {renderTrainingMonitor()}

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

      {/* Azure status display (with "validationPassed" taking top priority) */}
      {selectedDocuments.length > 0 &&
        (() => {
          if (validationPassed) {
            return (
              <div
                style={{
                  padding: '12px 16px',
                  marginBottom: '16px',
                  borderRadius: '8px',
                  border: '1px solid #10b981',
                  backgroundColor: 'rgba(16, 185, 129, 0.1)',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    marginBottom: '8px',
                  }}
                >
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  <span style={{ fontWeight: 'bold' }}>Azure ML Ready for Training</span>
                  {azureValidation?.estimated_cost && (
                    <span
                      style={{
                        marginLeft: '8px',
                        fontSize: '12px',
                        color: '#6b7280',
                        backgroundColor: '#f3f4f6',
                        padding: '2px 6px',
                        borderRadius: '4px',
                      }}
                    >
                      Est. cost: ${azureValidation.estimated_cost}
                    </span>
                  )}
                </div>
                <div
                  style={{
                    fontSize: '12px',
                    color: '#6b7280',
                    marginBottom: '8px',
                  }}
                >
                  Compute: {azureValidation.compute_target} | VM: {azureValidation.vm_size} | Duration:{' '}
                  {azureValidation.estimated_duration} | Documents: {azureValidation.total_documents}
                </div>
                {azureWorkspaceInfo && (
                  <div
                    style={{
                      fontSize: '11px',
                      color: '#9ca3af',
                      marginTop: '8px',
                    }}
                  >
                    Workspace: {azureWorkspaceInfo.name} ({azureWorkspaceInfo.location})
                  </div>
                )}
                {azureValidation.warnings?.length > 0 && (
                  <div style={{ marginTop: '8px' }}>
                    {azureValidation.warnings.map((warning, idx) => (
                      <div
                        key={idx}
                        style={{
                          fontSize: '11px',
                          color: '#f59e0b',
                        }}
                      >
                        ⚠️ {warning}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          } else if (azureError) {
            return (
              <div className="azure-validation-status error">
                <AlertTriangle className="w-4 h-4 text-red-600" />
                <span>{azureError.message}</span>
                {azureError.help && (
                  <a
                    href={azureError.help}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ marginLeft: '8px', fontSize: '11px', color: '#9ca3af' }}
                  >
                    Learn more
                  </a>
                )}
              </div>
            );
          } else if (isValidatingAzure) {
            return (
              <div className="azure-validation-status">
                <Loader className="w-4 h-4 spinning text-blue-600" />
                <span>Validating Azure ML configuration...</span>
              </div>
            );
          }
          return null;
        })()}

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

      {/* Training controls */}
      {selectedDocuments.length > 0 && (
        <div
          className="training-controls"
          style={{
            marginTop: '24px',
            padding: '20px',
            borderRadius: '12px',
            backgroundColor: 'rgba(243, 244, 246, 0.5)',
            border: '1px solid #e5e7eb',
          }}
        >
          <div
            className="training-info"
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '16px',
            }}
          >
            <div
              className="selected-info"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '16px',
                fontWeight: '600',
                color: '#1f2937',
              }}
            >
              <Activity className="w-5 h-5 text-blue-600" />
              {selectedDocuments.length} document
              {selectedDocuments.length > 1 ? 's' : ''} selected for training
            </div>
            <button
              onClick={clearSelection}
              className="btn btn-secondary"
              style={{
                fontSize: '12px',
                padding: '6px 12px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                backgroundColor: 'white',
                border: '1px solid #d1d5db',
              }}
            >
              <X className="w-3 h-3" />
              Clear Selection
            </button>
          </div>

          <div
            className="training-buttons"
            style={{
              display: 'flex',
              gap: '12px',
              alignItems: 'center',
            }}
          >
            {/* Local training */}
            <button
              onClick={() => startTraining(false)}
              className="btn btn-primary"
              disabled={realTimeStatus?.is_training || !backendConnected || isTrainingRequesting}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '12px 20px',
                fontSize: '14px',
                fontWeight: '600',
                backgroundColor: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor:
                  realTimeStatus?.is_training || !backendConnected || isTrainingRequesting
                    ? 'not-allowed'
                    : 'pointer',
                opacity:
                  realTimeStatus?.is_training || !backendConnected || isTrainingRequesting
                    ? 0.6
                    : 1,
              }}
              title={
                realTimeStatus?.is_training
                  ? 'Training already in progress'
                  : !backendConnected
                  ? 'Backend not connected'
                  : 'Train model locally'
              }
            >
              <Server className="w-5 h-5" />
              {isTrainingRequesting ? <>Launching...</> : 'Train Locally'}
            </button>

            {/* Azure ML training */}
            {getAzureStatus().available && (
              <div style={{ position: 'relative' }}>
                <button
                  onClick={() => startTraining(true)}
                  className="btn btn-azure"
                  disabled={!isAzureEnabled() || isTrainingRequesting}
                  onMouseEnter={() => setShowBillingTooltip(true)}
                  onMouseLeave={() => setShowBillingTooltip(false)}
                  title={getAzureButtonTooltip()}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '12px 20px',
                    fontSize: '14px',
                    fontWeight: '600',
                    background: isAzureEnabled()
                      ? 'linear-gradient(135deg, #0078d4, #106ebe)'
                      : '#9ca3af',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: isAzureEnabled() && !isTrainingRequesting ? 'pointer' : 'not-allowed',
                    transition: 'all 0.2s ease',
                    boxShadow: isAzureEnabled()
                      ? '0 4px 12px rgba(16, 110, 190, 0.3)'
                      : 'none',
                    opacity: isTrainingRequesting ? 0.6 : 1,
                  }}
                >
                  <Cloud className="w-5 h-5" />
                  {isTrainingRequesting ? 'Launching Azure...' : 'Train on Azure ML'}
                  {(isValidatingAzure || isTrainingRequesting) && (
                    <Loader className="w-4 h-4 ml-2 spinning" />
                  )}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentsTab;