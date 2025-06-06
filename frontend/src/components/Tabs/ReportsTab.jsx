// src/components/reports/ReportsTab.jsx

import React, { useState, useEffect } from 'react';
import {
  FileText, Download, Calendar, Filter, BarChart3, PieChart,
  TrendingUp, Users, MessageSquare, Brain, Activity, Clock,
  Printer, Share, Eye, Plus, RefreshCw, Search, CheckCircle,
  AlertCircle, FileSpreadsheet, File, Image, Settings
} from 'lucide-react';

const ReportsTab = ({
  backendConnected = false,
  documents = [],
  trainingHistory = [],
  chatSessions = []
}) => {
  const [reportType, setReportType] = useState('analytics');
  const [dateRange, setDateRange] = useState('30d');
  const [isGenerating, setIsGenerating] = useState(false);
  const [reports, setReports] = useState([]);
  const [selectedPatients, setSelectedPatients] = useState([]);
  const [reportFilters, setReportFilters] = useState({
    includeChats: true,
    includeDocuments: true,
    includeTraining: true,
    includeAnalytics: true
  });

  // Fetch existing reports
  const fetchReports = async () => {
    if (!backendConnected) return;
    
    try {
      const response = await fetch('http://localhost:5000/api/reports');
      if (response.ok) {
        const data = await response.json();
        setReports(data);
      }
    } catch (error) {
      console.error('Failed to fetch reports:', error);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [backendConnected]);

  // Generate report
  const generateReport = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch('http://localhost:5000/api/reports/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: reportType,
          dateRange,
          filters: reportFilters,
          patientIds: selectedPatients
        })
      });

      if (response.ok) {
        const report = await response.json();
        setReports(prev => [report, ...prev]);
        
        // Auto-download the report
        if (report.downloadUrl) {
          const link = document.createElement('a');
          link.href = report.downloadUrl;
          link.download = report.filename;
          link.click();
        }
      }
    } catch (error) {
      console.error('Failed to generate report:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const reportTypeOptions = [
    {
      value: 'analytics',
      label: 'Analytics Report',
      description: 'Comprehensive analytics and insights',
      icon: BarChart3
    },
    {
      value: 'patient-summary',
      label: 'Patient Summary',
      description: 'Individual patient consultation summary',
      icon: Users
    },
    {
      value: 'training-report',
      label: 'Training Report',
      description: 'AI model training performance and history',
      icon: Brain
    },
    {
      value: 'usage-report',
      label: 'Usage Report',
      description: 'System usage and activity metrics',
      icon: Activity
    },
    {
      value: 'compliance-audit',
      label: 'Compliance Audit',
      description: 'HIPAA compliance and audit trail',
      icon: CheckCircle
    },
    {
      value: 'custom',
      label: 'Custom Report',
      description: 'Build a custom report with specific parameters',
      icon: Settings
    }
  ];

  const ReportTypeCard = ({ option, selected, onSelect }) => {
    const Icon = option.icon;
    return (
      <div 
        className={`report-type-card ${selected ? 'selected' : ''}`}
        onClick={() => onSelect(option.value)}
      >
        <div className="report-type-icon">
          <Icon className="w-6 h-6" />
        </div>
        <div className="report-type-content">
          <h3>{option.label}</h3>
          <p>{option.description}</p>
        </div>
        <div className="report-type-indicator">
          {selected && <CheckCircle className="w-5 h-5" />}
        </div>
      </div>
    );
  };

  const ReportCard = ({ report }) => {
    const getStatusIcon = () => {
      switch (report.status) {
        case 'completed':
          return <CheckCircle className="w-4 h-4 text-green-500" />;
        case 'generating':
          return <RefreshCw className="w-4 h-4 text-blue-500 spinning" />;
        case 'failed':
          return <AlertCircle className="w-4 h-4 text-red-500" />;
        default:
          return <Clock className="w-4 h-4 text-yellow-500" />;
      }
    };

    const getFormatIcon = () => {
      switch (report.format) {
        case 'pdf':
          return <File className="w-4 h-4" />;
        case 'xlsx':
          return <FileSpreadsheet className="w-4 h-4" />;
        case 'csv':
          return <FileText className="w-4 h-4" />;
        default:
          return <FileText className="w-4 h-4" />;
      }
    };

    return (
      <div className="report-card">
        <div className="report-header">
          <div className="report-icon">
            {getFormatIcon()}
          </div>
          <div className="report-info">
            <h3>{report.title}</h3>
            <div className="report-meta">
              <span>{report.type}</span>
              <span>•</span>
              <span>{new Date(report.createdAt).toLocaleDateString()}</span>
              <span>•</span>
              <span>{report.size || 'N/A'}</span>
            </div>
          </div>
          <div className="report-status">
            {getStatusIcon()}
          </div>
        </div>
        
        <div className="report-description">
          {report.description}
        </div>

        <div className="report-actions">
          <button 
            className="btn btn-sm btn-secondary"
            disabled={report.status !== 'completed'}
          >
            <Eye className="w-3 h-3" />
            Preview
          </button>
          <button 
            className="btn btn-sm btn-secondary"
            disabled={report.status !== 'completed'}
            onClick={() => {
              if (report.downloadUrl) {
                const link = document.createElement('a');
                link.href = report.downloadUrl;
                link.download = report.filename;
                link.click();
              }
            }}
          >
            <Download className="w-3 h-3" />
            Download
          </button>
          <button className="btn btn-sm btn-secondary">
            <Share className="w-3 h-3" />
            Share
          </button>
        </div>
      </div>
    );
  };

  const QuickStats = () => {
    const stats = [
      {
        label: 'Total Documents',
        value: documents.length,
        icon: FileText,
        color: 'blue'
      },
      {
        label: 'Chat Sessions',
        value: chatSessions.length,
        icon: MessageSquare,
        color: 'green'
      },
      {
        label: 'Training Sessions',
        value: trainingHistory.length,
        icon: Brain,
        color: 'purple'
      },
      {
        label: 'Reports Generated',
        value: reports.filter(r => r.status === 'completed').length,
        icon: BarChart3,
        color: 'orange'
      }
    ];

    return (
      <div className="quick-stats">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className={`stat-card ${stat.color}`}>
              <div className="stat-icon">
                <Icon className="w-5 h-5" />
              </div>
              <div className="stat-content">
                <div className="stat-value">{stat.value}</div>
                <div className="stat-label">{stat.label}</div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="tab-content reports-tab">
      {/* Reports Header */}
      <div className="reports-header">
        <div className="reports-title">
          <FileText className="w-6 h-6" />
          <h2>Medical Reports</h2>
          <div className="reports-subtitle">Generate comprehensive medical and analytics reports</div>
        </div>
        <div className="reports-actions">
          <button onClick={fetchReports} className="btn btn-secondary">
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Quick Stats */}
      <QuickStats />

      {/* Report Generation Section */}
      <div className="report-generation-section">
        <h3>Generate New Report</h3>
        
        {/* Report Type Selection */}
        <div className="report-types-grid">
          {reportTypeOptions.map(option => (
            <ReportTypeCard
              key={option.value}
              option={option}
              selected={reportType === option.value}
              onSelect={setReportType}
            />
          ))}
        </div>

        {/* Report Configuration */}
        <div className="report-config">
          <div className="config-section">
            <h4>Report Configuration</h4>
            <div className="config-grid">
              <div className="config-item">
                <label>Date Range:</label>
                <select 
                  value={dateRange} 
                  onChange={(e) => setDateRange(e.target.value)}
                >
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                  <option value="90d">Last 3 Months</option>
                  <option value="6m">Last 6 Months</option>
                  <option value="1y">Last Year</option>
                  <option value="all">All Time</option>
                </select>
              </div>
              
              <div className="config-item">
                <label>Format:</label>
                <select defaultValue="pdf">
                  <option value="pdf">PDF Report</option>
                  <option value="xlsx">Excel Spreadsheet</option>
                  <option value="csv">CSV Data</option>
                  <option value="json">JSON Export</option>
                </select>
              </div>
            </div>
          </div>

          {/* Report Filters */}
          <div className="config-section">
            <h4>Include in Report</h4>
            <div className="filters-grid">
              <label className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={reportFilters.includeChats}
                  onChange={(e) => setReportFilters({
                    ...reportFilters,
                    includeChats: e.target.checked
                  })}
                />
                <MessageSquare className="w-4 h-4" />
                <span>Chat Sessions</span>
              </label>
              
              <label className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={reportFilters.includeDocuments}
                  onChange={(e) => setReportFilters({
                    ...reportFilters,
                    includeDocuments: e.target.checked
                  })}
                />
                <FileText className="w-4 h-4" />
                <span>Document Analysis</span>
              </label>
              
              <label className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={reportFilters.includeTraining}
                  onChange={(e) => setReportFilters({
                    ...reportFilters,
                    includeTraining: e.target.checked
                  })}
                />
                <Brain className="w-4 h-4" />
                <span>Training Data</span>
              </label>
              
              <label className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={reportFilters.includeAnalytics}
                  onChange={(e) => setReportFilters({
                    ...reportFilters,
                    includeAnalytics: e.target.checked
                  })}
                />
                <BarChart3 className="w-4 h-4" />
                <span>Analytics & Insights</span>
              </label>
            </div>
          </div>

          {/* Generate Button */}
          <div className="generate-section">
            <button 
              onClick={generateReport}
              disabled={isGenerating || !backendConnected}
              className="btn btn-primary btn-lg"
            >
              {isGenerating ? (
                <>
                  <RefreshCw className="w-5 h-5 spinning" />
                  Generating Report...
                </>
              ) : (
                <>
                  <Download className="w-5 h-5" />
                  Generate Report
                </>
              )}
            </button>
            <p className="generate-note">
              Report will be automatically downloaded when ready
            </p>
          </div>
        </div>
      </div>

      {/* Existing Reports */}
      <div className="existing-reports-section">
        <div className="section-header">
          <h3>Recent Reports</h3>
          <div className="section-actions">
            <div className="search-bar">
              <Search className="w-4 h-4" />
              <input 
                type="text" 
                placeholder="Search reports..." 
              />
            </div>
            <select defaultValue="all">
              <option value="all">All Types</option>
              <option value="analytics">Analytics</option>
              <option value="patient-summary">Patient Summary</option>
              <option value="training-report">Training</option>
              <option value="usage-report">Usage</option>
            </select>
          </div>
        </div>

        <div className="reports-grid">
          {reports.length > 0 ? (
            reports.map(report => (
              <ReportCard key={report.id} report={report} />
            ))
          ) : (
            <div className="empty-reports">
              <FileText className="w-12 h-12" />
              <h3>No reports generated yet</h3>
              <p>Generate your first report using the configuration above</p>
            </div>
          )}
        </div>
      </div>

      {/* Report Templates */}
      <div className="report-templates-section">
        <h3>Quick Report Templates</h3>
        <div className="templates-grid">
          <div className="template-card">
            <PieChart className="w-6 h-6" />
            <h4>Weekly Summary</h4>
            <p>Weekly activity and performance summary</p>
            <button className="btn btn-sm btn-secondary">
              <Download className="w-3 h-3" />
              Generate
            </button>
          </div>
          
          <div className="template-card">
            <TrendingUp className="w-6 h-6" />
            <h4>Monthly Analytics</h4>
            <p>Comprehensive monthly analytics report</p>
            <button className="btn btn-sm btn-secondary">
              <Download className="w-3 h-3" />
              Generate
            </button>
          </div>
          
          <div className="template-card">
            <Users className="w-6 h-6" />
            <h4>Patient Activity</h4>
            <p>Patient consultation and activity report</p>
            <button className="btn btn-sm btn-secondary">
              <Download className="w-3 h-3" />
              Generate
            </button>
          </div>
          
          <div className="template-card">
            <Brain className="w-6 h-6" />
            <h4>AI Training Report</h4>
            <p>AI model training and performance metrics</p>
            <button className="btn btn-sm btn-secondary">
              <Download className="w-3 h-3" />
              Generate
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportsTab;