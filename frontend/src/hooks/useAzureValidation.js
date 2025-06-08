// src/hooks/useAzureValidation.js

import { useEffect, useRef, useCallback } from 'react';

/**
 * Custom hook for managing Azure validation in a stable way
 * @param {array} selectedDocuments - Array of selected document IDs
 * @param {boolean} backendConnected - Whether backend is connected
 * @param {object} azureData - Azure data from useAzure hook
 * @returns {object} Validation control functions
 */
export const useAzureValidation = (selectedDocuments, backendConnected, azureData) => {
  const validationTimeoutRef = useRef(null);
  const lastValidatedDocumentsRef = useRef([]);
  const lastBackendStateRef = useRef(false);

  const {
    getAzureStatus,
    validateAzureTraining,
    resetAzureValidation,
    isValidatingAzure,
  } = azureData;

  // Helper to check if documents have actually changed
  const documentsChanged = useCallback((newDocs, oldDocs) => {
    if (newDocs.length !== oldDocs.length) return true;
    return newDocs.some((doc, index) => doc !== oldDocs[index]);
  }, []);

  // Stable validation trigger
  const triggerValidation = useCallback(() => {
    // Clear any existing timeout
    if (validationTimeoutRef.current) {
      clearTimeout(validationTimeoutRef.current);
    }

    if (selectedDocuments.length === 0) {
      resetAzureValidation();
      lastValidatedDocumentsRef.current = [];
      return;
    }

    // Check if we actually need to validate
    const documentsDidChange = documentsChanged(selectedDocuments, lastValidatedDocumentsRef.current);
    const backendStateChanged = backendConnected !== lastBackendStateRef.current;
    
    if (!documentsDidChange && !backendStateChanged && !isValidatingAzure) {
      return; // No need to re-validate
    }

    if (backendConnected && getAzureStatus().available) {
      // Debounce the validation
      validationTimeoutRef.current = setTimeout(() => {
        validateAzureTraining(selectedDocuments);
        lastValidatedDocumentsRef.current = [...selectedDocuments];
        lastBackendStateRef.current = backendConnected;
      }, 500);
    }
  }, [
    selectedDocuments,
    backendConnected,
    resetAzureValidation,
    getAzureStatus,
    validateAzureTraining,
    documentsChanged,
    isValidatingAzure,
  ]);

  // Effect that only runs when absolutely necessary
  useEffect(() => {
    triggerValidation();

    // Cleanup timeout on unmount
    return () => {
      if (validationTimeoutRef.current) {
        clearTimeout(validationTimeoutRef.current);
      }
    };
  }, [triggerValidation]);

  // Manual trigger for external use
  const manualValidate = useCallback(() => {
    if (validationTimeoutRef.current) {
      clearTimeout(validationTimeoutRef.current);
    }
    
    if (selectedDocuments.length > 0 && backendConnected && getAzureStatus().available) {
      validateAzureTraining(selectedDocuments);
      lastValidatedDocumentsRef.current = [...selectedDocuments];
      lastBackendStateRef.current = backendConnected;
    }
  }, [selectedDocuments, backendConnected, getAzureStatus, validateAzureTraining]);

  return {
    triggerValidation,
    manualValidate,
  };
};