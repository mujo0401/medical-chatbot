// src/hooks/useAzure.js

import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for managing Azure ML functionality
 * @param {string} backendUrl - The backend URL
 * @param {boolean} backendConnected - Whether backend is connected
 * @param {object} modelStatus - Model status from backend
 * @returns {object} Azure state and functions
 */
export const useAzure = (backendUrl, backendConnected, modelStatus) => {
  // Azure validation state
  const [azureValidation, setAzureValidation] = useState(null);
  const [azureBillingInfo, setAzureBillingInfo] = useState(null);
  const [azureWorkspaceInfo, setAzureWorkspaceInfo] = useState(null);
  const [isValidatingAzure, setIsValidatingAzure] = useState(false);
  const [azureError, setAzureError] = useState(null);
  const [validationPassed, setValidationPassed] = useState(false);

  // Error tracking for backoff
  const [azureFetchErrors, setAzureFetchErrors] = useState({
    billing: 0,
    workspace: 0,
    validation: 0,
  });

  // Helper: Check Azure status based on backend connection and model status
  const getAzureStatus = useCallback(() => {
    if (!backendConnected) {
      return { 
        available: false, 
        reason: 'Backend not connected', 
        type: 'connection' 
      };
    }
    
    if (!modelStatus?.azure) {
      return { 
        available: false, 
        reason: 'Azure status unknown', 
        type: 'unknown' 
      };
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
      return { 
        available: false, 
        reason: 'Azure ML SDK not installed', 
        type: 'sdk' 
      };
    }
    
    if (!azure.configured) {
      return { 
        available: false, 
        reason: 'Azure ML not configured', 
        type: 'config' 
      };
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
    
    return { 
      available: true, 
      reason: null, 
      type: 'available' 
    };
  }, [backendConnected, modelStatus]);

  // Fetch Azure billing information
  const fetchAzureBillingInfo = useCallback(async () => {
    if (!getAzureStatus().available || azureFetchErrors.billing >= 3) return;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const response = await fetch(`${backendUrl}/api/azure/billing-info`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const billingInfo = await response.json();
        setAzureBillingInfo(billingInfo);
        setAzureFetchErrors(prev => ({ ...prev, billing: 0 }));
      } else if (response.status !== 404) {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.warn('Failed to fetch Azure billing info:', error.message);
        setAzureFetchErrors(prev => ({ ...prev, billing: prev.billing + 1 }));
      }
    }
  }, [backendUrl, getAzureStatus, azureFetchErrors.billing]);

  // Fetch Azure workspace information
  const fetchAzureWorkspaceInfo = useCallback(async () => {
    if (!getAzureStatus().available || azureFetchErrors.workspace >= 3) return;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const response = await fetch(`${backendUrl}/api/training/azure/workspace-info`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const workspaceInfo = await response.json();
        setAzureWorkspaceInfo(workspaceInfo);
        setAzureFetchErrors(prev => ({ ...prev, workspace: 0 }));
      } else if (response.status !== 404) {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.warn('Failed to fetch Azure workspace info:', error.message);
        setAzureFetchErrors(prev => ({ ...prev, workspace: prev.workspace + 1 }));
      }
    }
  }, [backendUrl, getAzureStatus, azureFetchErrors.workspace]);

  // Validate Azure training configuration
  const validateAzureTraining = useCallback(async (selectedDocuments) => {
    if (selectedDocuments.length === 0 || !getAzureStatus().available || azureFetchErrors.validation >= 3) return;

    // Prevent duplicate validation requests
    if (isValidatingAzure) return;

    setIsValidatingAzure(true);
    setAzureError(null);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // Longer timeout for validation

      const response = await fetch(`${backendUrl}/api/azure/validate-training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_ids: selectedDocuments }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      const data = await response.json();

      if (response.ok) {
        setAzureValidation(data);

        if (data.can_train) {
          setValidationPassed(true);
          setAzureError(null);
        } else {
          setValidationPassed(false);
          setAzureError({
            message: data.error || 'Azure validation failed',
            help: data.help,
            recommendation: data.recommendation,
          });
        }
        
        setAzureFetchErrors(prev => ({ ...prev, validation: 0 }));
      } else {
        setAzureValidation(null);
        setAzureError({
          message: `Azure validation failed (HTTP ${response.status})`,
          help: 'Check your Azure configuration and compute targets',
        });
        setValidationPassed(false);
        
        if (response.status !== 404) {
          setAzureFetchErrors(prev => ({ ...prev, validation: prev.validation + 1 }));
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        setAzureValidation(null);
        setAzureError({
          message: 'Failed to connect to Azure ML service',
          help: 'Check your network connection and backend status',
        });
        setValidationPassed(false);
        setAzureFetchErrors(prev => ({ ...prev, validation: prev.validation + 1 }));
      }
    } finally {
      setIsValidatingAzure(false);
    }
  }, [backendUrl, getAzureStatus, azureFetchErrors.validation, isValidatingAzure]);

  // Reset Azure validation state
  const resetAzureValidation = useCallback(() => {
    setAzureValidation(null);
    setAzureError(null);
    setValidationPassed(false);
  }, []);

  // Check if Azure training is enabled for given documents
  const isAzureTrainingEnabled = useCallback((selectedDocuments) => {
    const azureStatus = getAzureStatus();
    return (
      selectedDocuments.length > 0 &&
      azureStatus.available &&
      validationPassed &&
      !isValidatingAzure &&
      !azureError
    );
  }, [getAzureStatus, validationPassed, isValidatingAzure, azureError]);

  // Get tooltip message for Azure training button
  const getAzureButtonTooltip = useCallback((selectedDocuments) => {
    const azureStatus = getAzureStatus();
    if (!azureStatus.available) return azureStatus.reason;
    if (selectedDocuments.length === 0) return 'No documents selected';
    if (isValidatingAzure) return 'Validating Azure configuration.';
    if (azureError) return azureError.message;
    if (!validationPassed) return 'Azure ML validation failed';
    return null;
  }, [getAzureStatus, isValidatingAzure, azureError, validationPassed]);

  // Get Azure compute targets from model status
  const getAvailableComputeTargets = useCallback(() => {
    if (!modelStatus?.azure?.compute_targets) return [];
    return modelStatus.azure.compute_targets.filter(target => target.state === 'Succeeded');
  }, [modelStatus]);

  // Get Azure subscription info
  const getSubscriptionInfo = useCallback(() => {
    if (!modelStatus?.azure) return null;
    return {
      subscription_id: modelStatus.azure.subscription_id,
      resource_group: modelStatus.azure.resource_group,
      workspace_name: modelStatus.azure.workspace_name,
    };
  }, [modelStatus]);

  // Validate Azure configuration without documents (general validation)
  const validateAzureConfiguration = useCallback(async () => {
    if (!getAzureStatus().available) return;

    setIsValidatingAzure(true);
    setAzureError(null);

    try {
      const response = await fetch(`${backendUrl}/api/azure/validate-configuration`);
      const data = await response.json();

      if (response.ok) {
        // Configuration is valid
        setAzureError(null);
        return { valid: true, data };
      } else {
        setAzureError({
          message: data.error || 'Azure configuration validation failed',
          help: data.help,
          recommendation: data.recommendation,
        });
        return { valid: false, error: data.error };
      }
    } catch (error) {
      setAzureError({
        message: 'Failed to validate Azure configuration',
        help: 'Check your network connection and backend status',
      });
      return { valid: false, error: error.message };
    } finally {
      setIsValidatingAzure(false);
    }
  }, [backendUrl, getAzureStatus]);

  // Test Azure connection
  const testAzureConnection = useCallback(async () => {
    if (!backendConnected) return { connected: false, error: 'Backend not connected' };

    try {
      const response = await fetch(`${backendUrl}/api/azure/test-connection`);
      const data = await response.json();

      return {
        connected: response.ok,
        data: response.ok ? data : null,
        error: response.ok ? null : data.error || 'Connection test failed',
      };
    } catch (error) {
      return {
        connected: false,
        error: error.message,
      };
    }
  }, [backendUrl, backendConnected]);

  // Fetch Azure resource usage
  const fetchAzureResourceUsage = useCallback(async () => {
    if (!getAzureStatus().available) return null;

    try {
      const response = await fetch(`${backendUrl}/api/azure/resource-usage`);
      if (response.ok) {
        const data = await response.json();
        return data;
      }
    } catch (error) {
      console.error('Failed to fetch Azure resource usage:', error);
    }
    return null;
  }, [backendUrl, getAzureStatus]);

  // Reset error counts when backend reconnects
  useEffect(() => {
    if (backendConnected) {
      setAzureFetchErrors({ billing: 0, workspace: 0, validation: 0 });
    }
  }, [backendConnected]);

  // Effect to fetch Azure info when Azure becomes available
  useEffect(() => {
    if (backendConnected && getAzureStatus().available) {
      // Add delay to avoid overwhelming backend
      const timer = setTimeout(() => {
        fetchAzureBillingInfo();
        setTimeout(() => fetchAzureWorkspaceInfo(), 2000);
      }, 2000);
      
      return () => clearTimeout(timer);
    }
  }, [backendConnected, getAzureStatus, fetchAzureBillingInfo, fetchAzureWorkspaceInfo]);

  return {
    // State
    azureValidation,
    azureBillingInfo,
    azureWorkspaceInfo,
    isValidatingAzure,
    azureError,
    validationPassed,

    // Functions
    getAzureStatus,
    validateAzureTraining,
    resetAzureValidation,
    isAzureTrainingEnabled,
    getAzureButtonTooltip,
    getAvailableComputeTargets,
    getSubscriptionInfo,
    validateAzureConfiguration,
    testAzureConnection,
    fetchAzureBillingInfo,
    fetchAzureWorkspaceInfo,
    fetchAzureResourceUsage,
  };
};