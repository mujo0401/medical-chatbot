// src/components/training/TrainingTab.jsx

import React, { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  Loader,
  Database,
  History,
  CheckCircle,
  AlertCircle,
  Info,
  Clock,
  Trash2,
  X,
  CheckSquare,
  Timer,
  DollarSign,
  FileText,
  Server,
  Cloud,
  Target,
  TrendingUp,
  Bell,
  StopCircle,  
  BarChart3,  
} from 'lucide-react';
import {
  formatJobStatus,
  formatCurrency,
  formatDuration,
} from '../../utils/azurehelper';
import '../../css/training.css';

const TrainingTab = ({
  backendUrl = 'http://localhost:5000', // adjust if your backend runs elsewhere
}) => {
  // ——————————————
  // State Hooks
  // ——————————————
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [azureJobs, setAzureJobs] = useState([]);
  const [loadingAzureJobs, setLoadingAzureJobs] = useState(false);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobLogs, setJobLogs] = useState([]);
  const [showJobDetails, setShowJobDetails] = useState(false);

  // Real‐time status (via polling)
  const [realTimeStatus, setRealTimeStatus] = useState({
    is_training: false,
    progress: 0,
    status_message: 'Ready',
    current_document: null,
    start_time: null,
    estimated_completion: null,
  });

  // Aggregated metrics/analytics
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [trainingAnalytics, setTrainingAnalytics] = useState(null);
  const [resourceUsage, setResourceUsage] = useState(null);
  const [costTracking, setCostTracking] = useState(null);
  const [notifications, setNotifications] = useState([]);

  // Poll intervals
  const STATUS_POLL_INTERVAL = 5000; // ms
  const JOBS_POLL_INTERVAL = 10000;

  // ——————————————
  // Helper: Fetch training history (from /training/history)
  // ——————————————
  const fetchTrainingHistory = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/training/history`);
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      // expecting an array of { training_id, documents, platform, compute, status, timestamp, duration_minutes, total_documents, cost }
      setTrainingHistory(data);
    } catch (e) {
      console.error('Failed to fetch training history:', e);
    }
  }, [backendUrl]);

  // ——————————————
  // Helper: Fetch Azure ML jobs list (from /training/azure/jobs)
  // ——————————————
  const fetchAzureJobs = useCallback(async () => {
    setLoadingAzureJobs(true);
    try {
      const res = await fetch(`${backendUrl}/training/azure/jobs`);
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      // data should be { jobs: [{ name, display_name, status, creation_time, start_time, end_time, compute_target }] }
      setAzureJobs(
        (data.jobs || []).map((job) => ({
          ...job,
          creation_time: job.creation_time ? new Date(job.creation_time) : null,
          start_time: job.start_time ? new Date(job.start_time) : null,
          end_time: job.end_time ? new Date(job.end_time) : null,
        }))
      );
    } catch (e) {
      console.error('Failed to fetch Azure jobs:', e);
    } finally {
      setLoadingAzureJobs(false);
    }
  }, [backendUrl]);

  // ——————————————
  // Helper: Fetch details/logs for a specific job
  // ——————————————
  const fetchJobLogs = async (jobName) => {
    try {
      const res = await fetch(`${backendUrl}/training/azure/jobs/${jobName}/logs`);
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      setJobLogs(data.logs || []);
    } catch (e) {
      console.error(`Failed to fetch logs for job ${jobName}:`, e);
      setJobLogs([`Failed to load logs: ${e.message}`]);
    }
  };

  // ——————————————
  // Helper: Poll current training status (from /training/status)
  // ——————————————
  const fetchRealTimeStatus = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/training/status`);
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      // Expecting { is_training, progress, status_message, current_document, start_time, estimated_completion }
      setRealTimeStatus((prev) => ({
        ...prev,
        ...data,
        start_time: data.start_time ? new Date(data.start_time) : prev.start_time,
        estimated_completion: data.estimated_completion
          ? new Date(data.estimated_completion)
          : prev.estimated_completion,
      }));
    } catch (e) {
      console.error('Failed to fetch real‐time status:', e);
    }
  }, [backendUrl]);

  // ——————————————
  // Helper: Fetch real metrics & analytics from backend
  // ——————————————
  const fetchMetricsAndAnalytics = useCallback(async () => {
    try {
      // Fetch metrics
      const metricsRes = await fetch(`${backendUrl}/training/metrics`);
      if (!metricsRes.ok) throw new Error(`Metrics failed: ${metricsRes.status}`);
      const metricsData = await metricsRes.json();
      setTrainingMetrics(metricsData);

      // Fetch analytics
      const analyticsRes = await fetch(`${backendUrl}/training/analytics`);
      if (!analyticsRes.ok) throw new Error(`Analytics failed: ${analyticsRes.status}`);
      const analyticsData = await analyticsRes.json();
      setTrainingAnalytics(analyticsData);

      // Fetch resource usage
      const resourceRes = await fetch(`${backendUrl}/training/resource`);
      if (!resourceRes.ok) throw new Error(`Resource usage failed: ${resourceRes.status}`);
      const resourceData = await resourceRes.json();
      setResourceUsage(resourceData);

      // Fetch cost tracking
      const costRes = await fetch(`${backendUrl}/training/cost`);
      if (!costRes.ok) throw new Error(`Cost tracking failed: ${costRes.status}`);
      const costData = await costRes.json();
      setCostTracking(costData);

      // Fetch notifications
      const notifRes = await fetch(`${backendUrl}/training/notifications`);
      if (!notifRes.ok) throw new Error(`Notifications failed: ${notifRes.status}`);
      const notifData = await notifRes.json();
      setNotifications(notifData.notifications || []);
    } catch (e) {
      console.error('Failed to fetch metrics/analytics:', e);
    }
  }, [backendUrl]);

  // ——————————————
  // Effects
  // ——————————————
  // (1) On mount, load history & jobs & start polling
  useEffect(() => {
    fetchTrainingHistory();
    fetchAzureJobs();
    fetchRealTimeStatus();
    fetchMetricsAndAnalytics();

    const statusInterval = setInterval(fetchRealTimeStatus, STATUS_POLL_INTERVAL);
    const jobsInterval = setInterval(fetchAzureJobs, JOBS_POLL_INTERVAL);

    return () => {
      clearInterval(statusInterval);
      clearInterval(jobsInterval);
    };
  }, [
    fetchTrainingHistory,
    fetchAzureJobs,
    fetchRealTimeStatus,
    fetchMetricsAndAnalytics,
  ]);

  // (2) Whenever trainingHistory or realTimeStatus changes, re‐fetch metrics & analytics
  useEffect(() => {
    fetchMetricsAndAnalytics();
  }, [trainingHistory, realTimeStatus, fetchMetricsAndAnalytics]);

  // (3) Whenever selectedJob changes and we show details, fetch logs
  useEffect(() => {
    if (showJobDetails && selectedJob) {
      fetchJobLogs(selectedJob.name);
    }
  }, [showJobDetails, selectedJob]);

  // (4) Refresh history after any Azure job changes
  const refreshAll = () => {
    fetchTrainingHistory();
    fetchAzureJobs();
    fetchRealTimeStatus();
    fetchMetricsAndAnalytics();
  };

  // ——————————————
  // Render Helpers
  // ——————————————
  const renderTrainingStatus = () => {
    if (!realTimeStatus.is_training) {
      return (
        <div className="training-idle">
          <Database className="w-12 h-12 text-gray-400 mb-4" />
          <p>No active training</p>
          <small>Select documents and start training to improve responses</small>
        </div>
      );
    }

    // If training is active, show progress bar + details
    return (
      <div className="training-active">
        <div className="training-progress">
          <div className="progress-header">
            <Loader className="w-5 h-5 animate-spin text-blue-600" />
            <span>Training in Progress</span>
          </div>

          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${realTimeStatus.progress}%` }}
            />
          </div>

          <div className="progress-details">
            <span>{realTimeStatus.progress}% Complete</span>
            <span>{realTimeStatus.status_message}</span>
          </div>

          {realTimeStatus.current_document && (
            <div className="current-document">
              <FileText className="w-4 h-4 text-blue-600" />
              <span>{realTimeStatus.current_document}</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderAzureJobs = () => {
    if (loadingAzureJobs) {
      return (
        <div className="azure-jobs-section">
          <div className="azure-jobs-header">
            <h4>Azure ML Jobs</h4>
            <Loader className="w-5 h-5 animate-spin text-gray-500" />
          </div>
          <p>Loading jobs…</p>
        </div>
      );
    }

    if (!azureJobs.length) {
      return (
        <div className="azure-jobs-section">
          <div className="azure-jobs-header">
            <h4>Azure ML Jobs</h4>
            <RefreshCw
              className="w-4 h-4 text-gray-500 cursor-pointer"
              onClick={fetchAzureJobs}
            />
          </div>
          <p>No Azure jobs found</p>
        </div>
      );
    }

    return (
      <div className="azure-jobs-section">
        <div className="azure-jobs-header">
          <h4>Azure ML Jobs</h4>
          <div className="training-header-actions">
            <RefreshCw
              className="w-4 h-4 text-gray-500 cursor-pointer"
              onClick={fetchAzureJobs}
            />
            <Trash2
              className="w-4 h-4 text-red-500 cursor-pointer"
              onClick={() => {
                if (
                  window.confirm('Delete entire training history?')
                ) {
                  fetch(`${backendUrl}/training/history`, {
                    method: 'DELETE',
                  })
                    .then((res) => {
                      if (res.ok) fetchTrainingHistory();
                    })
                    .catch((e) =>
                      console.error('Failed to delete history:', e)
                    );
                }
              }}
            />
          </div>
        </div>

        {azureJobs.map((job) => {
          const statusInfo = formatJobStatus(job.status);
          const isRunning =
            job.status === 'Running' ||
            job.status === 'Preparing' ||
            job.status === 'Starting';
          const isCompleted = job.status === 'Completed';
          const isFailed =
            job.status === 'Failed' || job.status === 'Canceled';

          let duration = 'N/A';
          if (
            job.start_time &&
            (isRunning || job.end_time)
          ) {
            const end = job.end_time || new Date();
            const minutes = (end - job.start_time) / 1000 / 60;
            duration = formatDuration(minutes);
          }

          return (
            <div
              key={job.name}
              className={`azure-job-item ${
                isRunning
                  ? 'running'
                  : isCompleted
                  ? 'completed'
                  : isFailed
                  ? 'failed'
                  : ''
              }`}
              onClick={() => {
                setSelectedJob(job);
                setShowJobDetails(true);
              }}
            >
              <div className="azure-job-info">
                <div className="azure-job-name">
                  {job.display_name}
                </div>
                <div className="azure-job-meta">
                  <span style={{ fontSize: '12px', color: '#6b7280' }}>
                    {job.compute_target} &middot;{' '}
                    {job.creation_time?.toLocaleString() || '—'}
                  </span>
                </div>
              </div>
              <div
                className="azure-job-stats"
                style={{ textAlign: 'right' }}
              >
                <span
                  style={{
                    fontSize: '14px',
                    fontWeight: '600',
                    color: statusInfo.color,
                    backgroundColor: statusInfo.bgColor,
                    borderRadius: '8px',
                    padding: '4px 8px',
                  }}
                >
                  {statusInfo.text}
                </span>
                <div
                  style={{
                    fontSize: '12px',
                    color: '#6b7280',
                    marginTop: '4px',
                  }}
                >
                  {duration}
                </div>
              </div>
            </div>
          );
        })}

        {/* Job Details Modal/Pane */}
        {showJobDetails && selectedJob && (
          <div
            className="azure-job-details"
            style={{ marginTop: '16px' }}
          >
            <div
              className="azure-job-details-header"
              style={{
                display: 'flex',
                justifyContent: 'space-between',
              }}
            >
              <h5
                style={{
                  fontSize: '18px',
                  fontWeight: '700',
                }}
              >
                Logs: {selectedJob.display_name}
              </h5>
              <X
                className="w-5 h-5 text-gray-600 cursor-pointer"
                onClick={() => {
                  setSelectedJob(null);
                  setShowJobDetails(false);
                  setJobLogs([]);
                }}
              />
            </div>
            <div
              className="azure-job-logs"
              style={{
                backgroundColor: 'rgba(243, 244, 246, 0.8)',
                padding: '12px',
                borderRadius: '8px',
                maxHeight: '200px',
                overflowY: 'auto',
                marginTop: '8px',
                fontSize: '12px',
                fontFamily: 'monospace',
                whiteSpace: 'pre-wrap',
              }}
            >
              {jobLogs.length > 0 ? (
                jobLogs.map((line, idx) => (
                  <div key={idx}>{line}</div>
                ))
              ) : (
                <p>Loading logs…</p>
              )}
            </div>
            <button
              className="btn btn-secondary"
              style={{ marginTop: '12px' }}
              onClick={() => {
                fetch(
                  `${backendUrl}/training/azure/jobs/${selectedJob.name}/cancel`,
                  {
                    method: 'POST',
                  }
                )
                  .then((res) => {
                    if (res.ok) {
                      alert(
                        `Cancellation requested for ${selectedJob.display_name}`
                      );
                      fetchAzureJobs();
                      setShowJobDetails(false);
                    } else {
                      alert(
                        `Failed to cancel ${selectedJob.display_name}`
                      );
                    }
                  })
                  .catch((e) => {
                    console.error('Cancel error:', e);
                    alert(`Cancel error: ${e.message}`);
                  });
              }}
              disabled={
                selectedJob.status === 'Completed' ||
                selectedJob.status === 'Failed' ||
                selectedJob.status === 'Canceled'
              }
            >
              <StopCircle className="w-4 h-4 mr-1" />
              Cancel Job
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderHistoryTable = () => {
    if (!trainingHistory.length) {
      return <p>No training sessions found</p>;
    }

    return (
      <div style={{ overflowX: 'auto', marginTop: '16px' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
          }}
        >
          <thead>
            <tr
              style={{ backgroundColor: '#F3F4F6' }}
            >
              <th
                style={{
                  padding: '8px',
                  border: '1px solid #E5E7EB',
                }}
              >
                ID
              </th>
              <th
                style={{
                  padding: '8px',
                  border: '1px solid #E5E7EB',
                }}
              >
                Documents
              </th>
              <th
                style={{
                  padding: '8px',
                  border: '1px solid #E5E7EB',
                }}
              >
                Type
              </th>
              <th
                style={{
                  padding: '8px',
                  border: '1px solid #E5E7EB',
                }}
              >
                Compute
              </th>
              <th
                style={{
                  padding: '8px',
                  border: '1px solid #E5E7EB',
                }}
              >
                Status
              </th>
              <th
                style={{
                  padding: '8px',
                  border: '1px solid #E5E7EB',
                }}
              >
                Started At
              </th>
              <th
                style={{
                  padding: '8px',
                  border: '1px solid #E5E7EB',
                }}
              >
                Cost (USD)
              </th>
              <th
                style={{
                  padding: '8px',
                  border: '1px solid #E5E7EB',
                }}
              >
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {trainingHistory.map((h) => {
              const started = new Date(h.timestamp);
              return (
                <tr key={h.training_id}>
                  <td
                    style={{
                      padding: '8px',
                      border: '1px solid #E5E7EB',
                    }}
                  >
                    {h.training_id.slice(0, 8)}
                  </td>
                  <td
                    style={{
                      padding: '8px',
                      border: '1px solid #E5E7EB',
                    }}
                  >
                    {h.documents}
                  </td>
                  <td
                    style={{
                      padding: '8px',
                      border: '1px solid #E5E7EB',
                    }}
                  >
                    {h.platform}
                  </td>
                  <td
                    style={{
                      padding: '8px',
                      border: '1px solid #E5E7EB',
                    }}
                  >
                    {h.compute}
                  </td>
                  <td
                    style={{
                      padding: '8px',
                      border: '1px solid #E5E7EB',
                      textAlign: 'center',
                    }}
                  >
                    {h.status === 'completed' ? (
                      <CheckCircle className="w-5 h-5 text-green-600 mx-auto" />
                    ) : h.status === 'failed' ? (
                      <AlertCircle className="w-5 h-5 text-red-600 mx-auto" />
                    ) : (
                      <Clock className="w-5 h-5 text-yellow-600 mx-auto" />
                    )}
                  </td>
                  <td
                    style={{
                      padding: '8px',
                      border: '1px solid #E5E7EB',
                    }}
                  >
                    {started.toLocaleString()}
                  </td>
                  <td
                    style={{
                      padding: '8px',
                      border: '1px solid #E5E7EB',
                    }}
                  >
                    {h.cost != null
                      ? formatCurrency(h.cost)
                      : '—'}
                  </td>
                  <td
                    style={{
                      padding: '8px',
                      border: '1px solid #E5E7EB',
                      textAlign: 'center',
                    }}
                  >
                    <Trash2
                      className="w-4 h-4 text-red-600 cursor-pointer"
                      onClick={() => {
                        if (
                          window.confirm(
                            `Delete training history ${h.training_id.slice(
                              0,
                              8
                            )}?`
                          )
                        ) {
                          fetch(
                            `${backendUrl}/training/history/${h.training_id}`,
                            {
                              method: 'DELETE',
                            }
                          )
                            .then((res) => {
                              if (res.ok) {
                                fetchTrainingHistory();
                              } else {
                                alert('Failed to delete');
                              }
                            })
                            .catch((e) =>
                              console.error(
                                'Delete history error:',
                                e
                              )
                            );
                        }
                      }}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };

  const renderMetricsSection = () => {
    if (!trainingMetrics) return null;
    return (
      <div
        style={{
          display: 'grid',
          gridTemplateColumns:
            'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '16px',
          marginTop: '24px',
        }}
      >
        {/* Total sessions */}
        <div
          style={{
            background: 'rgba(59, 130, 246, 0.05)',
            padding: '16px',
            borderRadius: '12px',
            border: '1px solid rgba(59, 130, 246, 0.1)',
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
            <History className="w-4 h-4 text-blue-600" />
            <span
              style={{
                fontSize: '14px',
                fontWeight: '600',
                color: '#374151',
              }}
            >
              Total Sessions
            </span>
          </div>
          <div
            style={{
              fontSize: '18px',
              fontWeight: '700',
              color: '#1f2937',
            }}
          >
            {trainingMetrics.total_training_sessions}
          </div>
        </div>

        {/* Successful/Failed breakdown */}
        <div
          style={{
            background: 'rgba(16, 185, 129, 0.05)',
            padding: '16px',
            borderRadius: '12px',
            border: '1px solid rgba(16, 185, 129, 0.1)',
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
            <CheckSquare className="w-4 h-4 text-green-600" />
            <span
              style={{
                fontSize: '14px',
                fontWeight: '600',
                color: '#374151',
              }}
            >
              Successful
            </span>
          </div>
          <div
            style={{
              fontSize: '18px',
              fontWeight: '700',
              color: '#1f2937',
            }}
          >
            {trainingMetrics.successful_sessions}
          </div>
          <div
            style={{
              fontSize: '12px',
              color: '#6b7280',
              marginTop: '4px',
            }}
          >
            Failed: {trainingMetrics.failed_sessions}
          </div>
        </div>

        {/* Total time */}
        <div
          style={{
            background: 'rgba(168, 85, 247, 0.05)',
            padding: '16px',
            borderRadius: '12px',
            border: '1px solid rgba(168, 85, 247, 0.1)',
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
            <Timer className="w-4 h-4 text-purple-600" />
            <span
              style={{
                fontSize: '14px',
                fontWeight: '600',
                color: '#374151',
              }}
            >
              Total Time
            </span>
          </div>
          <div
            style={{
              fontSize: '18px',
              fontWeight: '700',
              color: '#1f2937',
            }}
          >
            {trainingMetrics.total_training_time}
          </div>
          <div
            style={{
              fontSize: '12px',
              color: '#6b7280',
              marginTop: '4px',
            }}
          >
            Avg: {trainingMetrics.average_training_time}
          </div>
        </div>

        {/* Documents processed */}
        <div
          style={{
            background: 'rgba(59, 130, 246, 0.05)',
            padding: '16px',
            borderRadius: '12px',
            border: '1px solid rgba(59, 130, 246, 0.1)',
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
            <FileText className="w-4 h-4 text-blue-600" />
            <span
              style={{
                fontSize: '14px',
                fontWeight: '600',
                color: '#374151',
              }}
            >
              Docs Processed
            </span>
          </div>
          <div
            style={{
              fontSize: '18px',
              fontWeight: '700',
              color: '#1f2937',
            }}
          >
            {trainingMetrics.total_documents_processed}
          </div>
        </div>

        {/* Cost tracking */}
        <div
          style={{
            background: 'rgba(16, 185, 129, 0.05)',
            padding: '16px',
            borderRadius: '12px',
            border: '1px solid rgba(16, 185, 129, 0.1)',
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
            <DollarSign className="w-4 h-4 text-green-600" />
            <span
              style={{
                fontSize: '14px',
                fontWeight: '600',
                color: '#374151',
              }}
            >
              Total Cost
            </span>
          </div>
          <div
            style={{
              fontSize: '18px',
              fontWeight: '700',
              color: '#1f2937',
            }}
          >
            {formatCurrency(parseFloat(trainingMetrics.total_cost))}
          </div>
          <div
            style={{
              fontSize: '12px',
              color: '#6b7280',
              marginTop: '4px',
            }}
          >
            This month:{' '}
            {formatCurrency(parseFloat(trainingMetrics.cost_this_month))}
          </div>
        </div>
      </div>
    );
  };

  const renderAnalyticsCharts = () => {
    if (!trainingAnalytics) return null;
    // For brevity, placeholder icons represent charts.
    return (
      <div style={{ marginTop: '32px' }}>
        <h5
          style={{
            fontSize: '18px',
            fontWeight: '700',
            marginBottom: '12px',
          }}
        >
          Training Analytics (Last 7 days)
        </h5>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
          <div
            style={{
              flex: '1 1 250px',
              background: '#F9FAFB',
              padding: '16px',
              borderRadius: '12px',
            }}
          >
            <div
              style={{
                fontSize: '14px',
                fontWeight: '600',
                marginBottom: '8px',
              }}
            >
              Sessions per Day
            </div>
            <BarChart3 className="w-full h-24 text-gray-400" />
          </div>
          <div
            style={{
              flex: '1 1 250px',
              background: '#F9FAFB',
              padding: '16px',
              borderRadius: '12px',
            }}
          >
            <div
              style={{
                fontSize: '14px',
                fontWeight: '600',
                marginBottom: '8px',
              }}
            >
              Accuracy Trend
            </div>
            <TrendingUp className="w-full h-24 text-gray-400" />
          </div>
          <div
            style={{
              flex: '1 1 250px',
              background: '#F9FAFB',
              padding: '16px',
              borderRadius: '12px',
            }}
          >
            <div
              style={{
                fontSize: '14px',
                fontWeight: '600',
                marginBottom: '8px',
              }}
            >
              Cost Trend
            </div>
            <DollarSign className="w-full h-24 text-gray-400" />
          </div>
          <div
            style={{
              flex: '1 1 250px',
              background: '#F9FAFB',
              padding: '16px',
              borderRadius: '12px',
            }}
          >
            <div
              style={{
                fontSize: '14px',
                fontWeight: '600',
                marginBottom: '8px',
              }}
            >
              Duration Trend
            </div>
            <Timer className="w-full h-24 text-gray-400" />
          </div>
        </div>
      </div>
    );
  };

  const renderResourceUsage = () => {
    if (!resourceUsage) return null;
    return (
      <div
        style={{
          marginTop: '32px',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        {[
          'cpu_usage',
          'memory_usage',
          'gpu_usage',
          'disk_usage',
          'network_io',
        ].map((key) => (
          <div
            key={key}
            style={{
              flex: '1 1 180px',
              background: 'rgba(59, 130, 246, 0.05)',
              padding: '16px',
              borderRadius: '12px',
              border: '1px solid rgba(59, 130, 246, 0.1)',
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
              {key === 'cpu_usage' && <Server className="w-4 h-4 text-blue-600" />}
              {key === 'memory_usage' && <Database className="w-4 h-4 text-blue-600" />}
              {key === 'gpu_usage' && <Cloud className="w-4 h-4 text-blue-600" />}
              {key === 'disk_usage' && <Target className="w-4 h-4 text-blue-600" />}
              {key === 'network_io' && <TrendingUp className="w-4 h-4 text-blue-600" />}
              <span
                style={{
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {key
                  .replace('_', ' ')
                  .replace(/\b\w/g, (c) => c.toUpperCase())}
              </span>
            </div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                color: '#1f2937',
              }}
            >
              {resourceUsage[key]}
              {key === 'network_io' ? ' Mbps' : '%'}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderCostTracking = () => {
    if (!costTracking) return null;
    return (
      <div style={{ marginTop: '32px' }}>
        <h5
          style={{
            fontSize: '18px',
            fontWeight: '700',
            marginBottom: '12px',
          }}
        >
          Cost Tracking
        </h5>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns:
              'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '16px',
          }}
        >
          <div
            style={{
              background: 'rgba(16, 185, 129, 0.05)',
              padding: '16px',
              borderRadius: '12px',
              border: '1px solid rgba(16, 185, 129, 0.1)',
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
              <DollarSign className="w-4 h-4 text-green-600" />
              <span
                style={{
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                Current Session
              </span>
            </div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                color: '#1f2937',
              }}
            >
              {formatCurrency(parseFloat(costTracking.current_session_cost))}
            </div>
          </div>

          <div
            style={{
              background: 'rgba(59, 130, 246, 0.05)',
              padding: '16px',
              borderRadius: '12px',
              border: '1px solid rgba(59, 130, 246, 0.1)',
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
              <DollarSign className="w-4 h-4 text-blue-600" />
              <span
                style={{
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                Estimated Total
              </span>
            </div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                color: '#1f2937',
              }}
            >
              {formatCurrency(
                parseFloat(costTracking.estimated_total_cost)
              )}
            </div>
            <div
              style={{
                fontSize: '12px',
                color: '#6b7280',
                marginTop: '4px',
              }}
            >
              Per Hour:{' '}
              {formatCurrency(parseFloat(costTracking.cost_per_hour))}
            </div>
          </div>

          <div
            style={{
              background: 'rgba(239, 68, 68, 0.05)',
              padding: '16px',
              borderRadius: '12px',
              border: '1px solid rgba(239, 68, 68, 0.1)',
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
              <Bell className="w-4 h-4 text-red-600" />
              <span
                style={{
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                Monthly Budget
              </span>
            </div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                color: '#1f2937',
              }}
            >
              {formatCurrency(parseFloat(costTracking.monthly_budget))}
            </div>
            <div
              style={{
                fontSize: '12px',
                color: '#6b7280',
                marginTop: '4px',
              }}
            >
              Spent:{' '}
              {formatCurrency(parseFloat(costTracking.monthly_spent))}
            </div>
          </div>
        </div>

        {/* Suggestions */}
        <div style={{ marginTop: '16px' }}>
          <h6
            style={{
              fontSize: '16px',
              fontWeight: '600',
              marginBottom: '8px',
            }}
          >
            Cost Optimization Suggestions
          </h6>
          <ul style={{ listStyle: 'disc', marginLeft: '20px' }}>
            {costTracking.cost_optimization_suggestions.map(
              (sug, idx) => (
                <li
                  key={idx}
                  style={{
                    fontSize: '14px',
                    color: '#374151',
                  }}
                >
                  {sug}
                </li>
              )
            )}
          </ul>
        </div>
      </div>
    );
  };

  const renderNotifications = () => {
    if (!notifications.length) return null;
    return (
      <div style={{ marginTop: '32px' }}>
        <h5
          style={{
            fontSize: '18px',
            fontWeight: '700',
            marginBottom: '12px',
          }}
        >
          Notifications
        </h5>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
          }}
        >
          {notifications.map((n) => (
            <div
              key={n.id}
              style={{
                background:
                  n.type === 'success'
                    ? 'rgba(16, 185, 129, 0.05)'
                    : n.type === 'warning'
                    ? 'rgba(245, 158, 11, 0.05)'
                    : 'rgba(59, 130, 246, 0.05)',
                padding: '12px 16px',
                borderRadius: '8px',
                border:
                  '1px solid ' +
                  (n.type === 'success'
                    ? 'rgba(16, 185, 129, 0.1)'
                    : n.type === 'warning'
                    ? 'rgba(245, 158, 11, 0.1)'
                    : 'rgba(59, 130, 246, 0.1)'),
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                {n.type === 'success' && (
                  <CheckCircle className="w-4 h-4 text-green-600" />
                )}
                {n.type === 'warning' && (
                  <AlertCircle className="w-4 h-4 text-yellow-600" />
                )}
                {n.type === 'info' && (
                  <Info className="w-4 h-4 text-blue-600" />
                )}
                <span
                  style={{
                    fontSize: '14px',
                    color: '#374151',
                  }}
                >
                  {n.message}
                </span>
              </div>
              <span
                style={{
                  fontSize: '12px',
                  color: '#6b7280',
                }}
              >
                {n.time}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ——————————————
  // Main Render
  // ——————————————
  return (
    <div className="tab-content">
      {/* Header */}
      <div className="training-header">
        <h3>Training Dashboard</h3>
        <div className="training-header-actions">
          <RefreshCw
            className="w-5 h-5 text-gray-600 cursor-pointer"
            onClick={refreshAll}
          />
        </div>
      </div>

      {/* Real‐time Status */}
      {renderTrainingStatus()}

      {/* Training Metrics */}
      {renderMetricsSection()}

      {/* Resource Usage */}
      {renderResourceUsage()}

      {/* Cost Tracking */}
      {renderCostTracking()}

      {/* Notifications */}
      {renderNotifications()}

      {/* Azure ML Jobs */}
      {renderAzureJobs()}

      {/* Training History */}
      <div style={{ marginTop: '32px' }}>
        <h5
          style={{
            fontSize: '18px',
            fontWeight: '700',
            marginBottom: '12px',
          }}
        >
          Training History
        </h5>
        {renderHistoryTable()}
      </div>

      {/* Analytics Charts */}
      {renderAnalyticsCharts()}
    </div>
  );
};

export default TrainingTab;
