// src/components/training/TrainingControls.jsx

import React, { useState, useEffect } from 'react';
import {
  Activity,
  Server,
  Cloud,
  Loader,
  X,
} from 'lucide-react';
import { AzureValidation } from '../azure';

/**
 * TrainingControls component - handles training controls and Azure validation
 * @param {array} selectedDocuments - Array of selected document IDs
 * @param {function} onClearSelection - Function to clear document selection
 * @param {function} onStartTraining - Function to start training
 * @param {object} trainingData - Training-related data and functions
 * @param {boolean} backendConnected - Whether backend is connected
 * @returns {JSX.Element|null} Training controls component
 */
const TrainingControls = ({
  selectedDocuments,
  onClearSelection,
  onStartTraining,
  trainingData,
  backendConnected,
}) => {
  const [showBillingTooltip, setShowBillingTooltip] = useState(false);

  const {
    realTimeStatus,
    isTrainingRequesting,
    getAzureStatus,
    isAzureTrainingEnabled,
    getAzureButtonTooltip,
    isValidatingAzure,
    validationPassed,
    azureError,
  } = trainingData;

  const azureStatus = getAzureStatus();
  const azureEnabled = isAzureTrainingEnabled(selectedDocuments);
  
  // Debug logging for button state (remove in production) - MUST BE BEFORE CONDITIONAL RETURN
  useEffect(() => {
    if (selectedDocuments.length > 0) {
      console.log('Azure button state:', {
        azureEnabled,
        selectedCount: selectedDocuments.length,
        azureAvailable: azureStatus.available,
        validationPassed,
        isValidating: isValidatingAzure,
        hasError: !!azureError,
        isTrainingRequesting,
      });
    }
  }, [azureEnabled, selectedDocuments.length, azureStatus.available, validationPassed, isValidatingAzure, azureError, isTrainingRequesting]);

  const handleStartTraining = async (useAzure = false) => {
    const result = await onStartTraining(selectedDocuments, useAzure);
    if (result.success) {
      onClearSelection(); // Clear selection after successful training start
    }
  };

  // Don't render if no documents are selected
  if (selectedDocuments.length === 0) {
    return null;
  }

  return (
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
          onClick={onClearSelection}
          className="btn btn-secondary"
          style={{
            fontSize: '12px',
            padding: '6px 12px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            backgroundColor: 'white',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            cursor: 'pointer',
          }}
        >
          <X className="w-3 h-3" />
          Clear Selection
        </button>
      </div>

      {/* Azure validation status using dedicated component */}
      <AzureValidation selectedDocuments={selectedDocuments} azureData={trainingData} />

      <div
        className="training-buttons"
        style={{
          display: 'flex',
          gap: '12px',
          alignItems: 'center',
        }}
      >

        {/* Azure ML training */}
        {azureStatus.available && (
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => handleStartTraining(true)}
              className="btn btn-azure"
              disabled={!azureEnabled || isTrainingRequesting}
              onMouseEnter={() => setShowBillingTooltip(true)}
              onMouseLeave={() => setShowBillingTooltip(false)}
              title={getAzureButtonTooltip(selectedDocuments)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '12px 20px',
                fontSize: '14px',
                fontWeight: '600',
                background: azureEnabled
                  ? 'linear-gradient(135deg, #0078d4, #106ebe)'
                  : '#9ca3af',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: azureEnabled && !isTrainingRequesting ? 'pointer' : 'not-allowed',
                transition: 'all 0.2s ease',
                boxShadow: azureEnabled
                  ? '0 4px 12px rgba(16, 110, 190, 0.3)'
                  : 'none',
                opacity: isTrainingRequesting ? 0.6 : 1,
              }}
            >
              <Cloud className="w-5 h-5" />
              {isTrainingRequesting ? 'Launching Azure...' : 'Train on Azure ML'}
              {(isValidatingAzure || isTrainingRequesting) && (
                <Loader className="w-4 h-4 ml-2 animate-spin" />
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingControls;