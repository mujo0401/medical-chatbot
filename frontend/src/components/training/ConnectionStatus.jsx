// src/components/ui/ConnectionStatus.jsx

import React from 'react';
import {
  WifiOff,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react';

/**
 * ConnectionStatus component - displays backend connection status
 * @param {boolean} backendConnected - Whether backend is connected
 * @param {object} trainingData - Training data for error tracking (optional)
 * @returns {JSX.Element} Connection status component
 */
const ConnectionStatus = ({ 
  backendConnected, 
  trainingData = null 
}) => {
  if (!backendConnected) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px 12px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          borderRadius: '8px',
          marginBottom: '16px',
          fontSize: '14px',
          color: '#dc2626',
        }}
      >
        <WifiOff className="w-4 h-4" />
        <span style={{ fontWeight: '600' }}>Backend Disconnected</span>
        <span style={{ fontSize: '12px', color: '#6b7280' }}>
          Training features unavailable
        </span>
      </div>
    );
  }

  // Show error status if training data is available and has errors
  if (trainingData?.fetchErrors) {
    const hasErrors = Object.values(trainingData.fetchErrors).some(count => count >= 3);
    
    if (hasErrors) {
      return (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 12px',
            background: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.2)',
            borderRadius: '8px',
            marginBottom: '16px',
            fontSize: '14px',
            color: '#d97706',
          }}
        >
          <AlertTriangle className="w-4 h-4" />
          <span style={{ fontWeight: '600' }}>Limited Functionality</span>
          <span style={{ fontSize: '12px', color: '#6b7280' }}>
            Some training endpoints unavailable
          </span>
        </div>
      );
    }
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '8px 12px',
        background: 'rgba(16, 185, 129, 0.1)',
        border: '1px solid rgba(16, 185, 129, 0.2)',
        borderRadius: '8px',
        marginBottom: '16px',
        fontSize: '14px',
        color: '#059669',
      }}
    >
      <CheckCircle className="w-4 h-4" />
      <span style={{ fontWeight: '600' }}>Backend Connected</span>
      <span style={{ fontSize: '12px', color: '#6b7280' }}>
        All features available
      </span>
    </div>
  );
};

export default ConnectionStatus;