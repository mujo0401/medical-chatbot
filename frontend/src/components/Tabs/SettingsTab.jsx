// src/components/settings/SettingsTab.jsx

import React, { useState, useEffect } from 'react';
import {
  Settings, User, Shield, Database, Palette, Bell, Globe,
  Key, Cloud, Monitor, Save, RefreshCw, AlertTriangle,
  CheckCircle, Eye, EyeOff, Download, Upload, Trash2,
  Moon, Sun, Volume2, VolumeX, Lock, Unlock, HelpCircle
} from 'lucide-react';

const SettingsTab = ({
  backendConnected = false,
  modelStatus = null
}) => {
  const [activeSection, setActiveSection] = useState('general');
  const [settings, setSettings] = useState({
    general: {
      theme: 'light',
      language: 'en',
      timezone: 'UTC',
      autoSave: true,
      notifications: true,
      soundEnabled: true
    },
    security: {
      sessionTimeout: 30,
      requireMFA: false,
      passwordExpiry: 90,
      auditLogging: true,
      dataEncryption: true,
      hipaaCompliance: true
    },
    ai: {
      defaultModel: 'local',
      maxTokens: 2048,
      temperature: 0.7,
      responseTimeout: 30,
      autoTraining: false,
      confidenceThreshold: 0.8
    },
    privacy: {
      dataRetention: 365,
      anonymizeData: true,
      shareAnalytics: false,
      exportData: true,
      rightToDelete: true
    },
    backup: {
      autoBackup: true,
      backupInterval: 24,
      retentionPeriod: 30,
      cloudBackup: false
    }
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  // Load settings from backend
  const loadSettings = async () => {
    if (!backendConnected) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/settings');
      if (response.ok) {
        const data = await response.json();
        setSettings(prev => ({ ...prev, ...data }));
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Save settings to backend
  const saveSettings = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      
      if (response.ok) {
        setHasChanges(false);
        alert('Settings saved successfully!');
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert('Failed to save settings. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
  }, [backendConnected]);

  const updateSetting = (section, key, value) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
    setHasChanges(true);
  };

  const settingsSections = [
    { id: 'general', label: 'General', icon: Settings },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'ai', label: 'AI & Models', icon: Monitor },
    { id: 'privacy', label: 'Privacy', icon: Lock },
    { id: 'backup', label: 'Backup & Export', icon: Database },
    { id: 'about', label: 'About', icon: HelpCircle }
  ];

  const SettingsSection = ({ title, children }) => (
    <div className="settings-section">
      <h3>{title}</h3>
      <div className="settings-grid">
        {children}
      </div>
    </div>
  );

  const SettingItem = ({ 
    label, 
    description, 
    type = 'toggle', 
    value, 
    onChange, 
    options = [],
    min,
    max,
    step,
    disabled = false,
    warning = false
  }) => (
    <div className={`setting-item ${warning ? 'warning' : ''}`}>
      <div className="setting-info">
        <label>{label}</label>
        {description && <p>{description}</p>}
      </div>
      <div className="setting-control">
        {type === 'toggle' && (
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={value}
              onChange={(e) => onChange(e.target.checked)}
              disabled={disabled}
            />
            <span className="toggle-slider"></span>
          </label>
        )}
        
        {type === 'select' && (
          <select 
            value={value} 
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
          >
            {options.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        )}
        
        {type === 'number' && (
          <input
            type="number"
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            min={min}
            max={max}
            step={step}
            disabled={disabled}
          />
        )}
        
        {type === 'range' && (
          <div className="range-control">
            <input
              type="range"
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
              min={min}
              max={max}
              step={step}
              disabled={disabled}
            />
            <span className="range-value">{value}</span>
          </div>
        )}
      </div>
    </div>
  );

  const renderGeneralSettings = () => (
    <SettingsSection title="General Settings">
      <SettingItem
        label="Theme"
        description="Choose your preferred color theme"
        type="select"
        value={settings.general.theme}
        onChange={(value) => updateSetting('general', 'theme', value)}
        options={[
          { value: 'light', label: 'Light' },
          { value: 'dark', label: 'Dark' },
          { value: 'auto', label: 'Auto (System)' }
        ]}
      />
      
      <SettingItem
        label="Language"
        description="Select your preferred language"
        type="select"
        value={settings.general.language}
        onChange={(value) => updateSetting('general', 'language', value)}
        options={[
          { value: 'en', label: 'English' },
          { value: 'es', label: 'Spanish' },
          { value: 'fr', label: 'French' },
          { value: 'de', label: 'German' }
        ]}
      />
      
      <SettingItem
        label="Auto-save"
        description="Automatically save changes as you work"
        type="toggle"
        value={settings.general.autoSave}
        onChange={(value) => updateSetting('general', 'autoSave', value)}
      />
      
      <SettingItem
        label="Notifications"
        description="Receive system notifications"
        type="toggle"
        value={settings.general.notifications}
        onChange={(value) => updateSetting('general', 'notifications', value)}
      />
      
      <SettingItem
        label="Sound Effects"
        description="Play sounds for alerts and notifications"
        type="toggle"
        value={settings.general.soundEnabled}
        onChange={(value) => updateSetting('general', 'soundEnabled', value)}
      />
    </SettingsSection>
  );

  const renderSecuritySettings = () => (
    <SettingsSection title="Security & Compliance">
      <SettingItem
        label="Session Timeout"
        description="Automatically log out after inactivity (minutes)"
        type="number"
        value={settings.security.sessionTimeout}
        onChange={(value) => updateSetting('security', 'sessionTimeout', value)}
        min={5}
        max={120}
      />
      
      <SettingItem
        label="Multi-Factor Authentication"
        description="Require additional verification for login"
        type="toggle"
        value={settings.security.requireMFA}
        onChange={(value) => updateSetting('security', 'requireMFA', value)}
      />
      
      <SettingItem
        label="Password Expiry"
        description="Require password changes every X days"
        type="number"
        value={settings.security.passwordExpiry}
        onChange={(value) => updateSetting('security', 'passwordExpiry', value)}
        min={30}
        max={365}
      />
      
      <SettingItem
        label="Audit Logging"
        description="Log all user actions for compliance"
        type="toggle"
        value={settings.security.auditLogging}
        onChange={(value) => updateSetting('security', 'auditLogging', value)}
      />
      
      <SettingItem
        label="Data Encryption"
        description="Encrypt all data at rest and in transit"
        type="toggle"
        value={settings.security.dataEncryption}
        onChange={(value) => updateSetting('security', 'dataEncryption', value)}
        disabled={true}
      />
      
      <SettingItem
        label="HIPAA Compliance Mode"
        description="Enable strict HIPAA compliance features"
        type="toggle"
        value={settings.security.hipaaCompliance}
        onChange={(value) => updateSetting('security', 'hipaaCompliance', value)}
        warning={!settings.security.hipaaCompliance}
      />
    </SettingsSection>
  );

  const renderAISettings = () => (
    <SettingsSection title="AI Model Configuration">
      <SettingItem
        label="Default AI Model"
        description="Choose the default AI model for responses"
        type="select"
        value={settings.ai.defaultModel}
        onChange={(value) => updateSetting('ai', 'defaultModel', value)}
        options={[
          { value: 'local', label: 'Local Medical Model' },
          { value: 'openai', label: 'OpenAI GPT' },
          { value: 'azure', label: 'Azure OpenAI' },
          { value: 'hybrid', label: 'Hybrid Mode' }
        ]}
      />
      
      <SettingItem
        label="Max Response Length"
        description="Maximum tokens for AI responses"
        type="range"
        value={settings.ai.maxTokens}
        onChange={(value) => updateSetting('ai', 'maxTokens', value)}
        min={512}
        max={4096}
        step={256}
      />
      
      <SettingItem
        label="Response Creativity"
        description="Higher values make responses more creative"
        type="range"
        value={settings.ai.temperature}
        onChange={(value) => updateSetting('ai', 'temperature', value)}
        min={0}
        max={1}
        step={0.1}
      />
      
      <SettingItem
        label="Response Timeout"
        description="Maximum wait time for AI responses (seconds)"
        type="number"
        value={settings.ai.responseTimeout}
        onChange={(value) => updateSetting('ai', 'responseTimeout', value)}
        min={10}
        max={60}
      />
      
      <SettingItem
        label="Auto-Training"
        description="Automatically retrain models with new data"
        type="toggle"
        value={settings.ai.autoTraining}
        onChange={(value) => updateSetting('ai', 'autoTraining', value)}
      />
      
      <SettingItem
        label="Confidence Threshold"
        description="Minimum confidence for AI responses"
        type="range"
        value={settings.ai.confidenceThreshold}
        onChange={(value) => updateSetting('ai', 'confidenceThreshold', value)}
        min={0.5}
        max={0.95}
        step={0.05}
      />
    </SettingsSection>
  );

  const renderPrivacySettings = () => (
    <SettingsSection title="Privacy & Data Management">
      <SettingItem
        label="Data Retention Period"
        description="How long to keep user data (days)"
        type="number"
        value={settings.privacy.dataRetention}
        onChange={(value) => updateSetting('privacy', 'dataRetention', value)}
        min={90}
        max={3650}
      />
      
      <SettingItem
        label="Anonymize Data"
        description="Remove personally identifiable information"
        type="toggle"
        value={settings.privacy.anonymizeData}
        onChange={(value) => updateSetting('privacy', 'anonymizeData', value)}
      />
      
      <SettingItem
        label="Share Analytics"
        description="Share anonymized usage analytics"
        type="toggle"
        value={settings.privacy.shareAnalytics}
        onChange={(value) => updateSetting('privacy', 'shareAnalytics', value)}
      />
      
      <SettingItem
        label="Data Export Rights"
        description="Allow users to export their data"
        type="toggle"
        value={settings.privacy.exportData}
        onChange={(value) => updateSetting('privacy', 'exportData', value)}
      />
      
      <SettingItem
        label="Right to Delete"
        description="Allow users to permanently delete their data"
        type="toggle"
        value={settings.privacy.rightToDelete}
        onChange={(value) => updateSetting('privacy', 'rightToDelete', value)}
      />
    </SettingsSection>
  );

  const renderBackupSettings = () => (
    <SettingsSection title="Backup & Export">
      <SettingItem
        label="Automatic Backup"
        description="Automatically backup data and settings"
        type="toggle"
        value={settings.backup.autoBackup}
        onChange={(value) => updateSetting('backup', 'autoBackup', value)}
      />
      
      <SettingItem
        label="Backup Interval"
        description="How often to create backups (hours)"
        type="select"
        value={settings.backup.backupInterval}
        onChange={(value) => updateSetting('backup', 'backupInterval', value)}
        options={[
          { value: 6, label: 'Every 6 hours' },
          { value: 12, label: 'Every 12 hours' },
          { value: 24, label: 'Daily' },
          { value: 168, label: 'Weekly' }
        ]}
        disabled={!settings.backup.autoBackup}
      />
      
      <SettingItem
        label="Retention Period"
        description="How long to keep backups (days)"
        type="number"
        value={settings.backup.retentionPeriod}
        onChange={(value) => updateSetting('backup', 'retentionPeriod', value)}
        min={7}
        max={365}
        disabled={!settings.backup.autoBackup}
      />
      
      <SettingItem
        label="Cloud Backup"
        description="Store backups in cloud storage"
        type="toggle"
        value={settings.backup.cloudBackup}
        onChange={(value) => updateSetting('backup', 'cloudBackup', value)}
      />
      
      <div className="backup-actions">
        <button className="btn btn-secondary">
          <Download className="w-4 h-4" />
          Export All Data
        </button>
        <button className="btn btn-secondary">
          <Upload className="w-4 h-4" />
          Import Settings
        </button>
        <button className="btn btn-primary">
          <Save className="w-4 h-4" />
          Create Backup Now
        </button>
      </div>
    </SettingsSection>
  );

  const renderAboutSection = () => (
    <SettingsSection title="About Medical AI Assistant">
      <div className="about-content">
        <div className="app-info">
          <h4>Medical AI Assistant v2.1.0</h4>
          <p>Advanced AI-powered medical consultation and document analysis platform</p>
        </div>
        
        <div className="system-info">
          <div className="info-grid">
            <div className="info-item">
              <label>Backend Status:</label>
              <span className={backendConnected ? 'status-online' : 'status-offline'}>
                {backendConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="info-item">
              <label>AI Model:</label>
              <span>{modelStatus?.current || 'Unknown'}</span>
            </div>
            <div className="info-item">
              <label>Database:</label>
              <span className="status-online">Connected</span>
            </div>
            <div className="info-item">
              <label>Last Updated:</label>
              <span>March 2024</span>
            </div>
          </div>
        </div>
        
        <div className="support-links">
          <h4>Support & Resources</h4>
          <div className="links-grid">
            <a href="#" className="support-link">
              <HelpCircle className="w-4 h-4" />
              Help Documentation
            </a>
            <a href="#" className="support-link">
              <Globe className="w-4 h-4" />
              Privacy Policy
            </a>
            <a href="#" className="support-link">
              <Shield className="w-4 h-4" />
              Security Guidelines
            </a>
            <a href="#" className="support-link">
              <User className="w-4 h-4" />
              Contact Support
            </a>
          </div>
        </div>
      </div>
    </SettingsSection>
  );

  const renderContent = () => {
    switch (activeSection) {
      case 'general':
        return renderGeneralSettings();
      case 'security':
        return renderSecuritySettings();
      case 'ai':
        return renderAISettings();
      case 'privacy':
        return renderPrivacySettings();
      case 'backup':
        return renderBackupSettings();
      case 'about':
        return renderAboutSection();
      default:
        return renderGeneralSettings();
    }
  };

  return (
    <div className="tab-content settings-tab">
      <div className="settings-layout">
        {/* Settings Navigation */}
        <div className="settings-sidebar">
          <div className="settings-nav">
            {settingsSections.map(section => {
              const Icon = section.icon;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`nav-item ${activeSection === section.id ? 'active' : ''}`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{section.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Settings Content */}
        <div className="settings-content">
          <div className="settings-header">
            <h2>Settings</h2>
            {hasChanges && (
              <div className="changes-indicator">
                <AlertTriangle className="w-4 h-4" />
                <span>You have unsaved changes</span>
              </div>
            )}
          </div>

          {renderContent()}
          
          {/* Save Button */}
          {hasChanges && (
            <div className="settings-actions">
              <button 
                onClick={saveSettings}
                disabled={isLoading}
                className="btn btn-primary btn-lg"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-5 h-5 spinning" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-5 h-5" />
                    Save Settings
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SettingsTab;