// src/components/azure/AzureValidation.jsx

import {
  CheckCircle,
  AlertTriangle,
  Loader,
  ExternalLink,
  Server,
  Cloud,

  DollarSign,
} from 'lucide-react';

/**
 * AzureValidation component - displays Azure ML validation status
 * @param {array} selectedDocuments - Array of selected document IDs
 * @param {object} azureData - Azure-related data and functions from useAzure hook
 * @returns {JSX.Element|null} Azure validation component
 */
const AzureValidation = ({ selectedDocuments, azureData }) => {
  const {
    azureValidation,
    azureError,
    isValidatingAzure,
    validationPassed,
    azureWorkspaceInfo,
    azureBillingInfo,
    getAzureStatus,
  } = azureData;

  // Don't render if no documents are selected
  if (selectedDocuments.length === 0) {
    return null;
  }

  const azureStatus = getAzureStatus();

  // Render validation passed status
  if (validationPassed) {
    return (
      <div
        className="azure-validation-status success"
        style={{
          padding: '16px',
          background: 'rgba(16, 185, 129, 0.05)',
          border: '1px solid rgba(16, 185, 129, 0.2)',
          borderRadius: '12px',
          marginBottom: '16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
          <CheckCircle className="w-5 h-5 text-green-600 mt-1" />
          <div style={{ flex: 1 }}>
            <div style={{ marginBottom: '8px' }}>
              <span style={{ color: '#059669', fontWeight: '600', fontSize: '14px' }}>
                Azure ML is ready for training
              </span>
              {azureValidation?.compute_target && (
                <span
                  style={{
                    marginLeft: '8px',
                    fontSize: '12px',
                    color: '#059669',
                    background: 'rgba(16, 185, 129, 0.1)',
                    padding: '2px 6px',
                    borderRadius: '4px',
                  }}
                >
                  Compute: {azureValidation.compute_target}
                </span>
              )}
            </div>

            {/* Azure Workspace Info */}
            {azureWorkspaceInfo && (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '12px',
                  marginTop: '12px',
                }}
              >
                <div
                  style={{
                    background: 'rgba(255, 255, 255, 0.8)',
                    padding: '12px',
                    borderRadius: '8px',
                    border: '1px solid rgba(16, 185, 129, 0.1)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      marginBottom: '4px',
                      fontSize: '12px',
                      fontWeight: '600',
                      color: '#374151',
                    }}
                  >
                    <Cloud className="w-3 h-3" />
                    Workspace
                  </div>
                  <div style={{ fontSize: '11px', color: '#6b7280' }}>
                    <div>{azureWorkspaceInfo.name}</div>
                    <div>{azureWorkspaceInfo.location}</div>
                  </div>
                </div>

                <div
                  style={{
                    background: 'rgba(255, 255, 255, 0.8)',
                    padding: '12px',
                    borderRadius: '8px',
                    border: '1px solid rgba(16, 185, 129, 0.1)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      marginBottom: '4px',
                      fontSize: '12px',
                      fontWeight: '600',
                      color: '#374151',
                    }}
                  >
                    <Server className="w-3 h-3" />
                    Compute
                  </div>
                  <div style={{ fontSize: '11px', color: '#6b7280' }}>
                    {azureWorkspaceInfo.compute_targets?.length || 0} targets available
                  </div>
                </div>

                {/* Estimated Cost */}
                {azureValidation?.estimated_cost && (
                  <div
                    style={{
                      background: 'rgba(255, 255, 255, 0.8)',
                      padding: '12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(16, 185, 129, 0.1)',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        marginBottom: '4px',
                        fontSize: '12px',
                        fontWeight: '600',
                        color: '#374151',
                      }}
                    >
                      <DollarSign className="w-3 h-3" />
                      Est. Cost
                    </div>
                    <div style={{ fontSize: '11px', color: '#6b7280' }}>
                      ${azureValidation.estimated_cost}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Billing Information */}
            {azureBillingInfo && (
              <div
                style={{
                  marginTop: '12px',
                  padding: '8px 12px',
                  background: 'rgba(59, 130, 246, 0.05)',
                  borderRadius: '6px',
                  fontSize: '11px',
                  color: '#6b7280',
                }}
              >
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Render Azure not available status
  if (!azureStatus.available) {
    return (
      <div
        className="azure-validation-status error"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '16px',
          background: 'rgba(239, 68, 68, 0.05)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          borderRadius: '12px',
          marginBottom: '16px',
        }}
      >
        <AlertTriangle className="w-5 h-5 text-red-600" />
        <div style={{ flex: 1 }}>
          <div style={{ color: '#dc2626', fontWeight: '600', fontSize: '14px', marginBottom: '4px' }}>
            Azure ML Unavailable
          </div>
          <div style={{ color: '#6b7280', fontSize: '12px', marginBottom: '8px' }}>
            {azureStatus.reason}
          </div>
          
          {/* Azure Setup Help */}
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {azureStatus.help && (
              <a
                href={azureStatus.help}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  fontSize: '11px',
                  color: '#3b82f6',
                  textDecoration: 'none',
                  padding: '4px 8px',
                  background: 'rgba(59, 130, 246, 0.1)',
                  borderRadius: '4px',
                }}
              >
                <ExternalLink className="w-3 h-3" />
                Setup Guide
              </a>
            )}
            
            {azureStatus.type === 'config' && (
              <span
                style={{
                  fontSize: '11px',
                  color: '#6b7280',
                  padding: '4px 8px',
                  background: 'rgba(156, 163, 175, 0.1)',
                  borderRadius: '4px',
                }}
              >
                Check Azure credentials and workspace configuration
              </span>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Render Azure error status
  if (azureError) {
    return (
      <div
        className="azure-validation-status error"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '16px',
          background: 'rgba(239, 68, 68, 0.05)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          borderRadius: '12px',
          marginBottom: '16px',
        }}
      >
        <AlertTriangle className="w-5 h-5 text-red-600" />
        <div style={{ flex: 1 }}>
          <div style={{ color: '#dc2626', fontWeight: '600', fontSize: '14px', marginBottom: '4px' }}>
            Azure Validation Failed
          </div>
          <div style={{ color: '#6b7280', fontSize: '12px', marginBottom: '8px' }}>
            {azureError.message}
          </div>
          
          {/* Error Help and Recommendations */}
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {azureError.help && (
              <a
                href={azureError.help}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  fontSize: '11px',
                  color: '#3b82f6',
                  textDecoration: 'none',
                  padding: '4px 8px',
                  background: 'rgba(59, 130, 246, 0.1)',
                  borderRadius: '4px',
                }}
              >
                <ExternalLink className="w-3 h-3" />
                Help
              </a>
            )}
            
            {azureError.recommendation && (
              <span
                style={{
                  fontSize: '11px',
                  color: '#6b7280',
                  padding: '4px 8px',
                  background: 'rgba(156, 163, 175, 0.1)',
                  borderRadius: '4px',
                }}
              >
                {azureError.recommendation}
              </span>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Render validating status
  if (isValidatingAzure) {
    return (
      <div
        className="azure-validation-status validating"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '16px',
          background: 'rgba(59, 130, 246, 0.05)',
          border: '1px solid rgba(59, 130, 246, 0.2)',
          borderRadius: '12px',
          marginBottom: '16px',
        }}
      >
        <Loader className="w-5 h-5 animate-spin text-blue-600" />
        <div style={{ flex: 1 }}>
          <div style={{ color: '#3b82f6', fontWeight: '600', fontSize: '14px', marginBottom: '4px' }}>
            Validating Azure ML Configuration
          </div>
          <div style={{ color: '#6b7280', fontSize: '12px' }}>
            Checking workspace, compute targets, and training requirements...
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default AzureValidation;