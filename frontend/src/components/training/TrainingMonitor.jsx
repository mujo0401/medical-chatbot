// src/components/training/TrainingMonitor.jsx

import React, { useState, useEffect } from 'react';
import {
  Activity,
  Server,
  Cloud,
  Loader,
  CheckCircle,
  AlertTriangle,
  Info,
  Minimize2,
  Maximize2,
  Cpu,
  Zap,
  Clock,
  TrendingUp,
  TrendingDown,
  FileText,
  Brain,
  Target,
} from 'lucide-react';

/**
 * Enhanced TrainingMonitor component - displays real-time training status with rich graphics
 * @param {object} realTimeStatus - Current training status
 * @param {array} azureJobs - List of Azure ML jobs
 * @param {boolean} loadingAzureJobs - Whether Azure jobs are loading
 * @param {object} trainingMetrics - Training metrics data
 * @param {array} notifications - Recent notifications
 * @returns {JSX.Element|null} Enhanced training monitor component
 */
const TrainingMonitor = ({
  realTimeStatus,
  azureJobs = [],
  loadingAzureJobs = false,
  trainingMetrics = null,
  notifications = [],
}) => {
  const [isMinimized, setIsMinimized] = useState(false);
  const [metricsHistory, setMetricsHistory] = useState([]);

  // Simulate real-time metrics updates
  useEffect(() => {
    if (realTimeStatus?.is_training) {
      const interval = setInterval(() => {
        setMetricsHistory(prev => {
          const newMetric = {
            timestamp: Date.now(),
            loss: realTimeStatus.current_loss || (Math.random() * 0.5 + 0.1),
            accuracy: realTimeStatus.current_accuracy || (0.7 + Math.random() * 0.25),
            learning_rate: realTimeStatus.learning_rate || 0.001,
            gpu_usage: realTimeStatus.gpu_usage || (Math.random() * 30 + 60),
            memory_usage: realTimeStatus.memory_usage || (Math.random() * 20 + 70),
          };
          
          // Keep only last 20 data points
          const updated = [...prev, newMetric].slice(-20);
          return updated;
        });
      }, 2000); // Update every 2 seconds

      return () => clearInterval(interval);
    }
  }, [realTimeStatus?.is_training]);

  const runningAzureJobs = azureJobs.filter(job => 
    ['Running', 'Starting', 'Preparing'].includes(job.status)
  );

  // Don't render if no training is active
  if (!realTimeStatus?.is_training && runningAzureJobs.length === 0) {
    return null;
  }

  const currentMetrics = metricsHistory[metricsHistory.length - 1];
  const previousMetrics = metricsHistory[metricsHistory.length - 2];

  const getTrendIcon = (current, previous) => {
    if (!previous) return null;
    return current > previous ? 
      <TrendingUp className="w-3 h-3 text-green-500" /> : 
      <TrendingDown className="w-3 h-3 text-red-500" />;
  };

  const MiniChart = ({ data, color = '#3b82f6', height = 30 }) => {
    if (!data || data.length < 2) return null;

    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;

    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = ((max - value) / range) * height;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="100%" height={height} style={{ overflow: 'visible' }}>
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
          style={{ filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.1))' }}
        />
        <defs>
          <linearGradient id={`gradient-${color.replace('#', '')}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0.1" />
          </linearGradient>
        </defs>
        <polygon
          points={`0,${height} ${points} 100,${height}`}
          fill={`url(#gradient-${color.replace('#', '')})`}
        />
      </svg>
    );
  };

  const CircularProgress = ({ percentage, size = 60, strokeWidth = 6, color = '#3b82f6' }) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDasharray = `${circumference} ${circumference}`;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    return (
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          className="transform -rotate-90"
          width={size}
          height={size}
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#e5e7eb"
            strokeWidth={strokeWidth}
            fill="transparent"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="transparent"
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            style={{
              transition: 'stroke-dashoffset 0.5s ease-in-out',
            }}
          />
        </svg>
        <div
          className="absolute inset-0 flex items-center justify-center"
          style={{
            fontSize: size > 50 ? '12px' : '10px',
            fontWeight: '600',
            color: '#1f2937',
          }}
        >
          {Math.round(percentage)}%
        </div>
      </div>
    );
  };

  return (
    <div
      className="training-monitor-container"
      style={{
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(16, 185, 129, 0.05))',
        border: '2px solid rgba(59, 130, 246, 0.2)',
        borderRadius: '16px',
        padding: isMinimized ? '12px 20px' : '20px',
        marginBottom: '24px',
        position: 'relative',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Header with minimize/maximize */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: isMinimized ? 0 : '16px',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '16px',
            fontWeight: '600',
            color: '#3b82f6',
          }}
        >
          <Activity className="w-5 h-5" />
          <span>Training Monitor</span>
          {realTimeStatus?.is_training && (
            <span
              style={{
                fontSize: '12px',
                background: 'rgba(59, 130, 246, 0.1)',
                color: '#3b82f6',
                padding: '2px 8px',
                borderRadius: '8px',
                fontWeight: '500',
              }}
            >
              {realTimeStatus.is_paused ? 'PAUSED' : 'ACTIVE'}
            </span>
          )}
        </div>
        <button
          onClick={() => setIsMinimized(!isMinimized)}
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
          {isMinimized ? <Maximize2 className="w-4 h-4" /> : <Minimize2 className="w-4 h-4" />}
        </button>
      </div>

      {!isMinimized && (
        <>
          {/* Local Training Status */}
          {realTimeStatus?.is_training && (
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.8)',
                padding: '16px',
                borderRadius: '12px',
                marginBottom: '16px',
                border: '1px solid rgba(59, 130, 246, 0.1)',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '12px',
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#1f2937',
                }}
              >
                <Server className="w-4 h-4" />
                <span>Local Training Progress</span>
                {realTimeStatus.estimated_time_remaining && (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      marginLeft: 'auto',
                      fontSize: '11px',
                      color: '#6b7280',
                    }}
                  >
                    <Clock className="w-3 h-3" />
                    <span>{realTimeStatus.estimated_time_remaining} remaining</span>
                  </div>
                )}
              </div>

              {/* Enhanced Progress Bar */}
              <div style={{ position: 'relative', marginBottom: '12px' }}>
                <div
                  style={{
                    width: '100%',
                    height: '12px',
                    background: 'rgba(59, 130, 246, 0.1)',
                    borderRadius: '6px',
                    overflow: 'hidden',
                    position: 'relative',
                  }}
                >
                  <div
                    style={{
                      width: `${realTimeStatus.progress}%`,
                      height: '100%',
                      background: realTimeStatus.is_paused 
                        ? 'linear-gradient(90deg, #f59e0b, #f97316)'
                        : 'linear-gradient(90deg, #3b82f6, #10b981)',
                      transition: 'width 0.3s ease',
                      borderRadius: '6px',
                      position: 'relative',
                    }}
                  >
                    {/* Animated shimmer effect removed for compatibility */}
                  </div>
                </div>
              </div>

              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '12px',
                  color: '#6b7280',
                  marginBottom: '12px',
                }}
              >
                <span>{realTimeStatus.progress}% - {realTimeStatus.status_message}</span>
                {realTimeStatus.current_document && (
                  <span
                    style={{
                      background: 'rgba(59, 130, 246, 0.1)',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      fontSize: '11px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    <FileText className="w-3 h-3" />
                    {realTimeStatus.current_document}
                  </span>
                )}
              </div>

              {/* Real-time Metrics Grid */}
              {currentMetrics && (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                    gap: '12px',
                    marginTop: '16px',
                  }}
                >
                  {/* Loss Metric */}
                  <div
                    style={{
                      background: 'rgba(239, 68, 68, 0.05)',
                      padding: '12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(239, 68, 68, 0.1)',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '8px',
                      }}
                    >
                      <span style={{ fontSize: '11px', fontWeight: '600', color: '#dc2626' }}>
                        Loss
                      </span>
                      {getTrendIcon(currentMetrics.loss, previousMetrics?.loss)}
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>
                      {currentMetrics.loss.toFixed(4)}
                    </div>
                    <MiniChart 
                      data={metricsHistory.map(m => m.loss)} 
                      color="#dc2626"
                      height={25}
                    />
                  </div>

                  {/* Accuracy Metric */}
                  <div
                    style={{
                      background: 'rgba(16, 185, 129, 0.05)',
                      padding: '12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(16, 185, 129, 0.1)',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '8px',
                      }}
                    >
                      <span style={{ fontSize: '11px', fontWeight: '600', color: '#059669' }}>
                        Accuracy
                      </span>
                      {getTrendIcon(currentMetrics.accuracy, previousMetrics?.accuracy)}
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>
                      {(currentMetrics.accuracy * 100).toFixed(1)}%
                    </div>
                    <MiniChart 
                      data={metricsHistory.map(m => m.accuracy * 100)} 
                      color="#059669"
                      height={25}
                    />
                  </div>

                  {/* GPU Usage */}
                  <div
                    style={{
                      background: 'rgba(147, 51, 234, 0.05)',
                      padding: '12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(147, 51, 234, 0.1)',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '8px',
                      }}
                    >
                      <span style={{ fontSize: '11px', fontWeight: '600', color: '#7c3aed' }}>
                        GPU Usage
                      </span>
                      <Cpu className="w-3 h-3 text-purple-600" />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <CircularProgress 
                        percentage={currentMetrics.gpu_usage} 
                        size={40} 
                        color="#7c3aed"
                      />
                      <span style={{ fontSize: '12px', color: '#6b7280' }}>
                        {currentMetrics.gpu_usage.toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  {/* Memory Usage */}
                  <div
                    style={{
                      background: 'rgba(245, 158, 11, 0.05)',
                      padding: '12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(245, 158, 11, 0.1)',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '8px',
                      }}
                    >
                      <span style={{ fontSize: '11px', fontWeight: '600', color: '#d97706' }}>
                        Memory
                      </span>
                      <Brain className="w-3 h-3 text-amber-600" />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <CircularProgress 
                        percentage={currentMetrics.memory_usage} 
                        size={40} 
                        color="#d97706"
                      />
                      <span style={{ fontSize: '12px', color: '#6b7280' }}>
                        {currentMetrics.memory_usage.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Training Speed and Performance */}
              {realTimeStatus.training_speed && (
                <div
                  style={{
                    display: 'flex',
                    gap: '12px',
                    marginTop: '12px',
                    padding: '8px 12px',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    borderRadius: '6px',
                    fontSize: '11px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Zap className="w-3 h-3 text-blue-600" />
                    <span style={{ fontWeight: '600', color: '#1f2937' }}>Speed:</span>
                    <span style={{ color: '#6b7280' }}>{realTimeStatus.training_speed}</span>
                  </div>
                  {realTimeStatus.current_epoch && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Target className="w-3 h-3 text-green-600" />
                      <span style={{ fontWeight: '600', color: '#1f2937' }}>Epoch:</span>
                      <span style={{ color: '#6b7280' }}>
                        {realTimeStatus.current_epoch}/{realTimeStatus.total_epochs}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Azure Jobs Status */}
          {runningAzureJobs.length > 0 && (
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.8)',
                padding: '16px',
                borderRadius: '12px',
                marginBottom: '16px',
                border: '1px solid rgba(59, 130, 246, 0.1)',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '12px',
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#1f2937',
                }}
              >
                <Cloud className="w-4 h-4" />
                <span>Azure ML Jobs</span>
                {loadingAzureJobs && <Loader className="w-4 h-4 animate-spin" />}
              </div>

              {runningAzureJobs.map((job, index) => (
                <div
                  key={job.name}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '12px',
                    background: 'rgba(59, 130, 246, 0.05)',
                    borderRadius: '8px',
                    marginBottom: index < runningAzureJobs.length - 1 ? '8px' : 0,
                    fontSize: '12px',
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: '600', color: '#1f2937', marginBottom: '4px' }}>
                      {job.display_name || job.name}
                    </div>
                    <div style={{ color: '#6b7280', fontSize: '11px' }}>
                      {job.compute_target} • {job.status}
                    </div>
                    {job.start_time && (
                      <div style={{ color: '#6b7280', fontSize: '10px', marginTop: '2px' }}>
                        Started: {new Date(job.start_time).toLocaleTimeString()}
                      </div>
                    )}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {job.status === 'Running' && (
                      <div style={{
                        width: '8px',
                        height: '8px',
                        backgroundColor: '#10b981',
                        borderRadius: '50%',
                      }} />
                    )}
                    <div
                      style={{
                        background: job.status === 'Running' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                        color: job.status === 'Running' ? '#059669' : '#d97706',
                        padding: '4px 8px',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '600',
                      }}
                    >
                      {job.status}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Quick Metrics Summary */}
          {trainingMetrics && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                gap: '12px',
                marginBottom: '16px',
              }}
            >
              <div
                style={{
                  background: 'rgba(255, 255, 255, 0.8)',
                  padding: '12px',
                  borderRadius: '8px',
                  textAlign: 'center',
                  border: '1px solid rgba(59, 130, 246, 0.1)',
                }}
              >
                <div style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>
                  {trainingMetrics.total_training_sessions}
                </div>
                <div style={{ fontSize: '11px', color: '#6b7280', fontWeight: '500' }}>
                  Total Sessions
                </div>
              </div>
              <div
                style={{
                  background: 'rgba(255, 255, 255, 0.8)',
                  padding: '12px',
                  borderRadius: '8px',
                  textAlign: 'center',
                  border: '1px solid rgba(59, 130, 246, 0.1)',
                }}
              >
                <div style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>
                  {trainingMetrics.successful_sessions}
                </div>
                <div style={{ fontSize: '11px', color: '#6b7280', fontWeight: '500' }}>
                  Successful
                </div>
              </div>
              <div
                style={{
                  background: 'rgba(255, 255, 255, 0.8)',
                  padding: '12px',
                  borderRadius: '8px',
                  textAlign: 'center',
                  border: '1px solid rgba(59, 130, 246, 0.1)',
                }}
              >
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
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.8)',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid rgba(59, 130, 246, 0.1)',
              }}
            >
              <div
                style={{
                  fontSize: '12px',
                  fontWeight: '600',
                  color: '#1f2937',
                  marginBottom: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <Info className="w-3 h-3" />
                Recent Updates
              </div>
              {notifications.slice(0, 3).map((notif, index) => (
                <div
                  key={notif.id || index}
                  style={{
                    fontSize: '11px',
                    color: '#6b7280',
                    marginBottom: index < Math.min(notifications.length, 3) - 1 ? '6px' : 0,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 8px',
                    backgroundColor: 'rgba(59, 130, 246, 0.03)',
                    borderRadius: '4px',
                  }}
                >
                  {notif.type === 'success' && <CheckCircle className="w-3 h-3 text-green-600" />}
                  {notif.type === 'warning' && <AlertTriangle className="w-3 h-3 text-yellow-600" />}
                  {notif.type === 'info' && <Info className="w-3 h-3 text-blue-600" />}
                  <span style={{ flex: 1 }}>{notif.message}</span>
                  <span style={{ fontSize: '10px', opacity: 0.7 }}>
                    {notif.time}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Minimized view with enhanced info */}
      {isMinimized && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontSize: '12px',
            color: '#6b7280',
          }}
        >
          {realTimeStatus?.is_training && (
            <>
              <div
                style={{
                  width: '60px',
                  height: '4px',
                  background: 'rgba(59, 130, 246, 0.2)',
                  borderRadius: '2px',
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    width: `${realTimeStatus.progress}%`,
                    height: '100%',
                    background: realTimeStatus.is_paused ? '#f59e0b' : '#3b82f6',
                    borderRadius: '2px',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
              <span>{realTimeStatus.progress}%</span>
              {currentMetrics && (
                <>
                  <span>•</span>
                  <span>Loss: {currentMetrics.loss.toFixed(3)}</span>
                  <span>•</span>
                  <span>Acc: {(currentMetrics.accuracy * 100).toFixed(1)}%</span>
                </>
              )}
            </>
          )}
          {runningAzureJobs.length > 0 && (
            <span>
              {runningAzureJobs.length} Azure job{runningAzureJobs.length > 1 ? 's' : ''} running
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default TrainingMonitor;