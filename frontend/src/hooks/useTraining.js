// src/hooks/useTraining.js

import { useState, useEffect, useCallback } from 'react';
import { useAzure } from './useAzure';

/**
 * Custom hook for managing training functionality
 * @param {string} backendUrl - The backend URL
 * @param {boolean} backendConnected - Whether backend is connected
 * @param {object} modelStatus - Model status from backend
 * @returns {object} Training state and functions
 */
export const useTraining = (backendUrl, backendConnected, modelStatus) => {
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
  const [notifications, setNotifications] = useState([]);
  const [isTrainingRequesting, setIsTrainingRequesting] = useState(false);

  // Use Azure hook for Azure-specific functionality
  const azureData = useAzure(backendUrl, backendConnected, modelStatus);

  // Error tracking for backoff
  const [fetchErrors, setFetchErrors] = useState({
    training: 0,
    azure: 0,
    metrics: 0,
    notifications: 0,
  });

  // Fetch functions with error handling and backoff
  const fetchTrainingStatus = useCallback(async () => {
    if (!backendConnected || fetchErrors.training >= 3) return;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout
      
      const res = await fetch(`${backendUrl}/api/training/status`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
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
        
        // Reset error count on success
        setFetchErrors(prev => ({ ...prev, training: 0 }));
      } else if (res.status !== 404) {
        // Don't count 404s as errors (endpoint might not exist)
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.warn('Failed to fetch training status:', error.message);
        setFetchErrors(prev => ({ ...prev, training: prev.training + 1 }));
      }
    }
  }, [backendConnected, backendUrl, fetchErrors.training]);

  const fetchAzureJobs = useCallback(async () => {
    if (!azureData.getAzureStatus().available || fetchErrors.azure >= 3) return;
    
    setLoadingAzureJobs(true);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const res = await fetch(`${backendUrl}/api/training/azure/jobs`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
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
        
        setFetchErrors(prev => ({ ...prev, azure: 0 }));
      } else if (res.status !== 404) {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.warn('Failed to fetch Azure jobs:', error.message);
        setFetchErrors(prev => ({ ...prev, azure: prev.azure + 1 }));
      }
    } finally {
      setLoadingAzureJobs(false);
    }
  }, [backendUrl, azureData, fetchErrors.azure]);

  const fetchTrainingMetrics = useCallback(async () => {
    if (!backendConnected || fetchErrors.metrics >= 3) return;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const res = await fetch(`${backendUrl}/api/training/metrics`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (res.ok) {
        const data = await res.json();
        setTrainingMetrics(data);
        setFetchErrors(prev => ({ ...prev, metrics: 0 }));
      } else if (res.status !== 404) {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.warn('Failed to fetch training metrics:', error.message);
        setFetchErrors(prev => ({ ...prev, metrics: prev.metrics + 1 }));
      }
    }
  }, [backendConnected, backendUrl, fetchErrors.metrics]);

  const fetchNotifications = useCallback(async () => {
    if (!backendConnected || fetchErrors.notifications >= 3) return;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const res = await fetch(`${backendUrl}/api/training/notifications`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (res.ok) {
        const data = await res.json();
        setNotifications(data.notifications || []);
        setFetchErrors(prev => ({ ...prev, notifications: 0 }));
      } else if (res.status !== 404) {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.warn('Failed to fetch notifications:', error.message);
        setFetchErrors(prev => ({ ...prev, notifications: prev.notifications + 1 }));
      }
    }
  }, [backendConnected, backendUrl, fetchErrors.notifications]);

  // Start training function
  const startTraining = useCallback(async (selectedDocuments, useAzure = false) => {
    if (selectedDocuments.length === 0) return;

    setIsTrainingRequesting(true);
    try {
      const requestBody = {
        document_ids: selectedDocuments,
        use_azure: useAzure,
      };
      if (useAzure && azureData.azureValidation?.compute_target) {
        requestBody.compute_target = azureData.azureValidation.compute_target;
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
            `Estimated Cost: ${data.estimated_cost || 'N/A'}`
        );
      } else {
        alert(
          `Local training started successfully!\n\n` +
            `Training ID: ${data.training_id}\n` +
            `Documents: ${data.documents.join(', ')}`
        );
      }

      return { success: true, data };
    } catch (error) {
      alert('Training failed: ' + error.message);
      return { success: false, error: error.message };
    } finally {
      setIsTrainingRequesting(false);
    }
  }, [backendUrl, azureData]);

  // Reset error counts when backend reconnects
  useEffect(() => {
    if (backendConnected) {
      setFetchErrors({ training: 0, azure: 0, metrics: 0, notifications: 0 });
    }
  }, [backendConnected]);

  // Setup polling intervals with conservative timing
  useEffect(() => {
    if (!backendConnected) return;

    // Initial fetch with delay to avoid overwhelming backend
    const initialDelay = setTimeout(() => {
      fetchTrainingStatus();
      setTimeout(() => fetchAzureJobs(), 1000);
      setTimeout(() => fetchTrainingMetrics(), 2000);
      setTimeout(() => fetchNotifications(), 3000);
    }, 1000);

    // Set up polling intervals - more conservative timing
    const statusInterval = setInterval(fetchTrainingStatus, 10000); // Every 10 seconds
    const jobsInterval = setInterval(fetchAzureJobs, 30000); // Every 30 seconds
    const metricsInterval = setInterval(fetchTrainingMetrics, 60000); // Every 60 seconds
    const notifInterval = setInterval(fetchNotifications, 45000); // Every 45 seconds

    return () => {
      clearTimeout(initialDelay);
      clearInterval(statusInterval);
      clearInterval(jobsInterval);
      clearInterval(metricsInterval);
      clearInterval(notifInterval);
    };
  }, [
    backendConnected,
    fetchTrainingStatus,
    fetchAzureJobs,
    fetchTrainingMetrics,
    fetchNotifications,
  ]);

  return {
    // Training state
    realTimeStatus,
    azureJobs,
    loadingAzureJobs,
    trainingMetrics,
    notifications,
    isTrainingRequesting,
    fetchErrors, // Add error tracking for UI

    // Azure data (delegated to useAzure hook)
    ...azureData,

    // Training functions
    startTraining,
    fetchTrainingStatus,
    fetchAzureJobs,
    fetchTrainingMetrics,
    fetchNotifications,
  };
};