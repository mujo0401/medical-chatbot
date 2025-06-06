// src/utils/azurehelper.js

/**
 * Format Azure ML job status for display
 */
export const formatJobStatus = (status) => {
  const statusMap = {
    Running:   { text: 'Running',   color: '#2563eb', bgColor: 'rgba(59,130,246,0.1)' },
    Completed: { text: 'Completed', color: '#059669', bgColor: 'rgba(16,185,129,0.1)' },
    Failed:    { text: 'Failed',    color: '#dc2626', bgColor: 'rgba(239,68,68,0.1)' },
    Canceled:  { text: 'Cancelled', color: '#6b7280', bgColor: 'rgba(107,114,128,0.1)' },
    Canceling: { text: 'Cancelling',color: '#f59e0b', bgColor: 'rgba(245,158,11,0.1)' },
    Starting:  { text: 'Starting',  color: '#3b82f6', bgColor: 'rgba(59,130,246,0.1)' },
    Preparing: { text: 'Preparing', color: '#8b5cf6', bgColor: 'rgba(139,92,246,0.1)' },
  };

  return statusMap[status] || { text: status || 'Unknown', color: '#6b7280', bgColor: 'rgba(107,114,128,0.1)' };
};

/**
 * Format currency amounts
 */
export const formatCurrency = (amount, currency = 'USD') => {
  if (typeof amount !== 'number') return 'N/A';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
};

/**
 * Format duration in minutes to human-readable form
 */
export const formatDuration = (minutes) => {
  if (!minutes || minutes < 0) return 'N/A';
  if (minutes < 60) {
    return `${Math.round(minutes)} min`;
  } else if (minutes < 1440) {
    const hrs = Math.floor(minutes / 60);
    const rem = Math.round(minutes % 60);
    return rem > 0 ? `${hrs}h ${rem}m` : `${hrs}h`;
  } else {
    const days = Math.floor(minutes / 1440);
    const remHrs = Math.floor((minutes % 1440) / 60);
    return remHrs > 0 ? `${days}d ${remHrs}h` : `${days}d`;
  }
};

/**
 * Format file size in bytes to human-readable form
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Format Azure compute target display name
 */
export const formatComputeTarget = (computeTarget) => {
  if (!computeTarget) return 'Unknown';
  return computeTarget
    .split('-')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
};

/**
 * Format Azure region to a friendly name
 */
export const getRegionDisplayName = (region) => {
  const regionMap = {
    eastus: 'East US',
    eastus2: 'East US 2',
    westus: 'West US',
    westus2: 'West US 2',
    westus3: 'West US 3',
    centralus: 'Central US',
    northcentralus: 'North Central US',
    southcentralus: 'South Central US',
    westcentralus: 'West Central US',
    canadacentral: 'Canada Central',
    canadaeast: 'Canada East',
    brazilsouth: 'Brazil South',
    northeurope: 'North Europe',
    westeurope: 'West Europe',
    uksouth: 'UK South',
    ukwest: 'UK West',
    francecentral: 'France Central',
    francesouth: 'France South',
    switzerlandnorth: 'Switzerland North',
    germanywestcentral: 'Germany West Central',
    norwayeast: 'Norway East',
    swedencentral: 'Sweden Central',
    eastasia: 'East Asia',
    southeastasia: 'Southeast Asia',
    japaneast: 'Japan East',
    japanwest: 'Japan West',
    australiaeast: 'Australia East',
    australiasoutheast: 'Australia Southeast',
    centralindia: 'Central India',
    southindia: 'South India',
    westindia: 'West India',
    koreacentral: 'Korea Central',
    koreasouth: 'Korea South',
  };
  return regionMap[region?.toLowerCase()] || region || 'Unknown Region';
};

/**
 * Estimate training cost for given VM size, duration, and region
 */
export const estimateTrainingCost = ({
  vmSize = 'Standard_DS3_v2',
  durationHours = 1,
  region = 'eastus',
} = {}) => {
  const baseCosts = {
    Standard_DS1_v2: 0.098,
    Standard_DS2_v2: 0.196,
    Standard_DS3_v2: 0.392,
    Standard_DS4_v2: 0.784,
    Standard_DS5_v2: 1.568,
    Standard_NC6: 0.90,
    Standard_NC12: 1.80,
    Standard_NC24: 3.60,
    Standard_NC6s_v3: 3.06,
    Standard_NC12s_v3: 6.12,
    Standard_NC24s_v3: 12.24,
  };

  const rate = baseCosts[vmSize] || 0.40;
  const regionMultiplier = region.toLowerCase().includes('west') ? 1.1 : 1.0;
  const totalCost = rate * regionMultiplier * durationHours;

  return { hourlyRate: rate * regionMultiplier, totalCost, currency: 'USD' };
};

/**
 * Validate if compute target suits a given workload size
 */
export const validateComputeTarget = (computeTarget, workloadSize = 'medium') => {
  const cpuTargets = ['cpu-cluster', 'standard-ds-cluster'];
  const gpuTargets = ['gpu-cluster', 'standard-nc-cluster'];

  const recommendations = {
    small: {
      preferred: cpuTargets,
      minimum: 'Standard_DS2_v2',
    },
    medium: {
      preferred: [...cpuTargets, ...gpuTargets],
      minimum: 'Standard_DS3_v2',
    },
    large: {
      preferred: gpuTargets,
      minimum: 'Standard_NC6',
    },
  };

  const workloadRec = recommendations[workloadSize] || recommendations.medium;
  const isPreferred = workloadRec.preferred.some((target) =>
    computeTarget?.toLowerCase().includes(target.toLowerCase())
  );

  return {
    isRecommended: isPreferred,
    workloadSize,
    recommendations: workloadRec,
  };
};

/**
 * Parse Azure job logs into events/errors/warnings
 */
export const parseJobLogs = (logs) => {
  if (!Array.isArray(logs)) return { events: [], errors: [], warnings: [] };

  const events = [];
  const errors = [];
  const warnings = [];

  logs.forEach((log) => {
    if (typeof log !== 'string') return;
    const timestamp = log.match(/\[(.*?)\]/)?.[1];
    const message = log.replace(/\[.*?\]\s*/, '');

    if (log.toLowerCase().includes('error') || log.toLowerCase().includes('failed')) {
      errors.push({ timestamp, message, original: log });
    } else if (log.toLowerCase().includes('warning') || log.toLowerCase().includes('warn')) {
      warnings.push({ timestamp, message, original: log });
    } else {
      events.push({ timestamp, message, original: log });
    }
  });

  return { events, errors, warnings };
};

/**
 * Format subscription ID by masking middle characters
 */
export const formatSubscriptionId = (subscriptionId) => {
  if (!subscriptionId || subscriptionId.length < 8) return 'Unknown';
  const start = subscriptionId.substring(0, 8);
  const end = subscriptionId.substring(subscriptionId.length - 4);
  return `${start}...${end}`;
};

/**
 * Generate training recommendations based on uploaded documents
 */
export const getTrainingRecommendations = (documents = []) => {
  const totalSize = documents.reduce((sum, doc) => sum + (doc.file_size || 0), 0);
  const totalDocs = documents.length;

  let workloadSize = 'small';
  let recommendedVm = 'Standard_DS2_v2';
  let estimatedDuration = 30; // minutes

  if (totalSize > 10 * 1024 * 1024 || totalDocs > 10) {
    workloadSize = 'medium';
    recommendedVm = 'Standard_DS3_v2';
    estimatedDuration = 60;
  }
  if (totalSize > 50 * 1024 * 1024 || totalDocs > 50) {
    workloadSize = 'large';
    recommendedVm = 'Standard_NC6';
    estimatedDuration = 120;
  }

  const { totalCost } = estimateTrainingCost({
    vmSize: recommendedVm,
    durationHours: estimatedDuration / 60,
  });

  return {
    workloadSize,
    recommendedVm,
    estimatedDuration,
    costEstimate: totalCost,
    recommendations: [
      `Recommended VM: ${recommendedVm}`,
      `Estimated duration: ${formatDuration(estimatedDuration)}`,
      `Estimated cost: ${formatCurrency(totalCost)}`,
      totalDocs > 20
        ? 'Consider GPU compute for faster processing'
        : 'CPU compute is sufficient',
    ],
  };
};

/**
 * Check availability of Azure features based on model status
 */
export const isAzureFeatureAvailable = (modelStatus, feature) => {
  if (!modelStatus?.azure?.available) return false;

  const featureRequirements = {
    training: modelStatus.azure.configured,
    billing: modelStatus.azure.configured,
    monitoring: modelStatus.azure.configured,
    compute: modelStatus.azure.configured && modelStatus.azure.compute_targets?.length > 0,
  };

  return featureRequirements[feature] !== false;
};

/**
 * Format which documents contributed to a given answer
 */
export const formatSourceDocuments = (sourceDocs) => {
  if (!Array.isArray(sourceDocs) || sourceDocs.length === 0) {
    return 'None';
  }
  return sourceDocs.map((d) => d.name).join(', ');
};

export default {
  formatJobStatus,
  formatCurrency,
  formatDuration,
  formatFileSize,
  formatComputeTarget,
  getRegionDisplayName,
  estimateTrainingCost,
  validateComputeTarget,
  parseJobLogs,
  formatSubscriptionId,
  getTrainingRecommendations,
  isAzureFeatureAvailable,
  formatSourceDocuments,
};
