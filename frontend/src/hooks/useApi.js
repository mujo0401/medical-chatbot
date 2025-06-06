// src/hooks/useApi.js
import { useState, useCallback, useRef } from 'react';
import apiService from '../services/api';

export const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortController = useRef(null);
  const lastCallTime = useRef({});

  const execute = useCallback(async (apiCall, ...args) => {
    // Cancel any previous request
    if (abortController.current) {
      abortController.current.abort();
    }

    // Create new abort controller for this request
    abortController.current = new AbortController();

    // Simple rate limiting - prevent rapid successive calls to the same endpoint
    const callKey = apiCall.name;
    const now = Date.now();
    if (lastCallTime.current[callKey] && now - lastCallTime.current[callKey] < 1000) {
      // Less than 1 second since last call to this endpoint
      throw new Error('Rate limit: Too many requests');
    }
    lastCallTime.current[callKey] = now;

    setLoading(true);
    setError(null);
    
    try {
      const result = await apiCall(...args);
      return result;
    } catch (err) {
      // Don't set error for aborted requests
      if (err.name !== 'AbortError') {
        setError(err.message);
      }
      throw err;
    } finally {
      setLoading(false);
      abortController.current = null;
    }
  }, []);

  const sendMessage = useCallback((message, sessionId) => 
    execute(apiService.sendMessage, message, sessionId), [execute]);

  const uploadFile = useCallback((file) => 
    execute(apiService.uploadFile, file), [execute]);

  const startTraining = useCallback((documentIds, useAzure, computeTarget) => 
    execute(apiService.startTraining, documentIds, useAzure, computeTarget), [execute]);

  const getDocuments = useCallback(() => 
    execute(apiService.getDocuments), [execute]);

  const getTrainingStatus = useCallback(() => 
    execute(apiService.getTrainingStatus), [execute]);

  const getTrainingHistory = useCallback(() => 
    execute(apiService.getTrainingHistory), [execute]);

  const checkHealth = useCallback(() => 
    execute(apiService.checkHealth), [execute]);

  const deleteDocument = useCallback((docId) => 
    execute(apiService.deleteDocument, docId), [execute]);

  // Cleanup function to abort any pending requests
  const cleanup = useCallback(() => {
    if (abortController.current) {
      abortController.current.abort();
    }
  }, []);

  return {
    loading,
    error,
    sendMessage,
    uploadFile,
    startTraining,
    getDocuments,
    getTrainingStatus,
    getTrainingHistory,
    checkHealth,
    deleteDocument,
    cleanup,
  };
};