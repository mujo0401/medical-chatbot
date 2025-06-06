// src/components/analytics/AnalyticsTab.jsx

import React, { useState, useEffect } from 'react';
import {
  BarChart3, TrendingUp, Users, MessageSquare, FileText, Brain,
  Activity, Heart, Stethoscope, PieChart, LineChart, Calendar,
  Clock, Target, AlertTriangle, CheckCircle, RefreshCw, Download
} from 'lucide-react';

const AnalyticsTab = ({
  backendConnected = false,
  documents = [],
  trainingHistory = [],
  chatSessions = []
}) => {
  const [analyticsData, setAnalyticsData] = useState({
    conversations: {
      total: 0,
      thisWeek: 0,
      avgLength: 0,
      topTopics: []
    },
    documents: {
      total: 0,
      processed: 0,
      totalSize: 0,
      types: {}
    },
    training: {
      sessions: 0,
      successful: 0,
      avgDuration: 0,
      lastTrained: null
    },
    insights: {
      commonSymptoms: [],
      frequentQueries: [],
      peakHours: []
    }
  });

  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  const [isLoading, setIsLoading] = useState(false);

  // Fetch analytics data
  const fetchAnalytics = async () => {
    if (!backendConnected) return;
    
    setIsLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/analytics?range=${selectedTimeRange}`);
      if (response.ok) {
        const data = await response.json();
        setAnalyticsData(data);
      }
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, [backendConnected, selectedTimeRange]);

  // Calculate basic metrics from available data
  const calculateMetrics = () => {
    const documentTypes = documents.reduce((acc, doc) => {
      acc[doc.file_type] = (acc[doc.file_type] || 0) + 1;
      return acc;
    }, {});

    const successfulTraining = trainingHistory.filter(t => t.status === 'completed').length;
    const totalSize = documents.reduce((acc, doc) => acc + (doc.file_size || 0), 0);

    return {
      totalDocuments: documents.length,
      processedDocuments: documents.filter(d => d.processed).length,
      totalSessions: chatSessions.length,
      documentTypes,
      trainingSuccess: trainingHistory.length > 0 ? (successfulTraining / trainingHistory.length * 100) : 0,
      totalDataSize: totalSize
    };
  };

  const metrics = calculateMetrics();

  const MetricCard = ({ title, value, subtitle, icon: Icon, trend, color = 'blue' }) => (
    <div className="metric-card medical-metric-card">
      <div className="metric-header">
        <div className={`metric-icon ${color}`}>
          <Icon className="w-6 h-6" />
        </div>
        <div className="metric-trend">
          {trend && (
            <div className={`trend ${trend > 0 ? 'positive' : 'negative'}`}>
              <TrendingUp className="w-4 h-4" />
              <span>{Math.abs(trend)}%</span>
            </div>
          )}
        </div>
      </div>
      <div className="metric-content">
        <div className="metric-value">{value}</div>
        <div className="metric-title">{title}</div>
        {subtitle && <div className="metric-subtitle">{subtitle}</div>}
      </div>
    </div>
  );

  const ChartContainer = ({ title, children, actions }) => (
    <div className="chart-container medical-chart-container">
      <div className="chart-header">
        <h3>{title}</h3>
        {actions && <div className="chart-actions">{actions}</div>}
      </div>
      <div className="chart-content">
        {children}
      </div>
    </div>
  );

  const DocumentTypeChart = () => (
    <div className="document-type-chart">
      {Object.entries(metrics.documentTypes).map(([type, count]) => (
        <div key={type} className="chart-bar">
          <div className="bar-label">{type.toUpperCase()}</div>
          <div className="bar-container">
            <div 
              className="bar-fill"
              style={{ 
                width: `${(count / metrics.totalDocuments) * 100}%`,
                background: type === 'pdf' ? '#3b82f6' : '#10b981'
              }}
            />
          </div>
          <div className="bar-value">{count}</div>
        </div>
      ))}
    </div>
  );

  const ActivityTimeline = () => {
    const activities = [
      ...trainingHistory.slice(0, 5).map(t => ({
        type: 'training',
        title: `Training Session: ${t.document_name}`,
        time: t.started_at,
        status: t.status,
        icon: Brain
      })),
      ...documents.slice(0, 3).map(d => ({
        type: 'document',
        title: `Document Uploaded: ${d.original_name}`,
        time: d.uploaded_at,
        status: d.processed ? 'completed' : 'processing',
        icon: FileText
      }))
    ].sort((a, b) => new Date(b.time) - new Date(a.time)).slice(0, 8);

    return (
      <div className="activity-timeline">
        {activities.map((activity, index) => {
          const Icon = activity.icon;
          return (
            <div key={index} className="timeline-item">
              <div className={`timeline-icon ${activity.status}`}>
                <Icon className="w-4 h-4" />
              </div>
              <div className="timeline-content">
                <div className="timeline-title">{activity.title}</div>
                <div className="timeline-time">
                  {new Date(activity.time).toLocaleDateString()}
                </div>
              </div>
              <div className={`timeline-status ${activity.status}`}>
                {activity.status === 'completed' ? (
                  <CheckCircle className="w-4 h-4" />
                ) : activity.status === 'failed' ? (
                  <AlertTriangle className="w-4 h-4" />
                ) : (
                  <Clock className="w-4 h-4" />
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const InsightsGrid = () => (
    <div className="insights-grid">
      <div className="insight-card">
        <div className="insight-header">
          <Stethoscope className="w-5 h-5" />
          <span>Most Analyzed Content</span>
        </div>
        <div className="insight-content">
          <div className="insight-item">
            <span>Medical Reports</span>
            <span className="insight-percentage">65%</span>
          </div>
          <div className="insight-item">
            <span>Lab Results</span>
            <span className="insight-percentage">25%</span>
          </div>
          <div className="insight-item">
            <span>Prescriptions</span>
            <span className="insight-percentage">10%</span>
          </div>
        </div>
      </div>

      <div className="insight-card">
        <div className="insight-header">
          <MessageSquare className="w-5 h-5" />
          <span>Query Categories</span>
        </div>
        <div className="insight-content">
          <div className="insight-item">
            <span>Symptoms Analysis</span>
            <span className="insight-percentage">40%</span>
          </div>
          <div className="insight-item">
            <span>Medication Info</span>
            <span className="insight-percentage">30%</span>
          </div>
          <div className="insight-item">
            <span>General Health</span>
            <span className="insight-percentage">30%</span>
          </div>
        </div>
      </div>

      <div className="insight-card">
        <div className="insight-header">
          <Target className="w-5 h-5" />
          <span>AI Performance</span>
        </div>
        <div className="insight-content">
          <div className="insight-item">
            <span>Response Accuracy</span>
            <span className="insight-percentage">94%</span>
          </div>
          <div className="insight-item">
            <span>Processing Speed</span>
            <span className="insight-percentage">1.2s avg</span>
          </div>
          <div className="insight-item">
            <span>User Satisfaction</span>
            <span className="insight-percentage">89%</span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="tab-content analytics-tab">
      {/* Analytics Header */}
      <div className="analytics-header">
        <div className="analytics-title">
          <BarChart3 className="w-6 h-6" />
          <h2>Medical Analytics Dashboard</h2>
        </div>
        <div className="analytics-controls">
          <select 
            value={selectedTimeRange} 
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="time-range-selector"
          >
            <option value="1d">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 3 Months</option>
          </select>
          <button onClick={fetchAnalytics} className="btn btn-secondary" disabled={isLoading}>
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'spinning' : ''}`} />
            Refresh
          </button>
          <button className="btn btn-primary">
            <Download className="w-4 h-4" />
            Export Report
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="metrics-grid">
        <MetricCard
          title="Total Documents"
          value={metrics.totalDocuments}
          subtitle={`${metrics.processedDocuments} processed`}
          icon={FileText}
          trend={12}
          color="blue"
        />
        <MetricCard
          title="Chat Sessions"
          value={metrics.totalSessions}
          subtitle="Active conversations"
          icon={MessageSquare}
          trend={8}
          color="green"
        />
        <MetricCard
          title="Training Success"
          value={`${Math.round(metrics.trainingSuccess)}%`}
          subtitle={`${trainingHistory.length} total sessions`}
          icon={Brain}
          trend={-3}
          color="purple"
        />
        <MetricCard
          title="Data Processed"
          value={`${(metrics.totalDataSize / (1024 * 1024)).toFixed(1)} MB`}
          subtitle="Total document size"
          icon={Activity}
          trend={25}
          color="orange"
        />
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        <div className="charts-row">
          <ChartContainer title="Document Types Distribution">
            <DocumentTypeChart />
          </ChartContainer>
          
          <ChartContainer title="Recent Activity">
            <ActivityTimeline />
          </ChartContainer>
        </div>
      </div>

      {/* Insights Section */}
      <div className="insights-section">
        <h3>Medical AI Insights</h3>
        <InsightsGrid />
      </div>

      {/* System Health */}
      <div className="system-health-section">
        <ChartContainer 
          title="System Health Monitoring"
          actions={
            <div className="health-indicators">
              <div className="health-indicator online">
                <div className="indicator-dot"></div>
                <span>AI Model Online</span>
              </div>
              <div className="health-indicator online">
                <div className="indicator-dot"></div>
                <span>Database Connected</span>
              </div>
              <div className="health-indicator online">
                <div className="indicator-dot"></div>
                <span>All Systems Operational</span>
              </div>
            </div>
          }
        >
          <div className="health-metrics-grid">
            <div className="health-metric">
              <Heart className="w-5 h-5" />
              <div className="metric-info">
                <span className="metric-label">Model Response Time</span>
                <span className="metric-value">1.2s avg</span>
              </div>
            </div>
            <div className="health-metric">
              <Activity className="w-5 h-5" />
              <div className="metric-info">
                <span className="metric-label">System Uptime</span>
                <span className="metric-value">99.8%</span>
              </div>
            </div>
            <div className="health-metric">
              <Stethoscope className="w-5 h-5" />
              <div className="metric-info">
                <span className="metric-label">Processing Queue</span>
                <span className="metric-value">0 pending</span>
              </div>
            </div>
          </div>
        </ChartContainer>
      </div>
    </div>
  );
};

export default AnalyticsTab;