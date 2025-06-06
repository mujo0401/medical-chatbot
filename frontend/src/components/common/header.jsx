// src/components/header/Header.jsx
import React from 'react';
import { 
  Menu, MessageSquare, FileText,
} from 'lucide-react';

const MedicalHeader = ({
  sidebarOpen,
  setSidebarOpen,
  activeTab,
  setActiveTab,
  uploadedDocuments,
  trainingStatus,
  chatSessionsCount = 0,
}) => (
  <div className="header medical-header">
    <div className="header-info medical-header-info">
      {!sidebarOpen && (
        <button onClick={() => setSidebarOpen(true)} className="menu-btn medical-menu-btn">
          <Menu className="w-5 h-5" />
        </button>
      )}
    
    </div>
    
    <div className="tab-navigation medical-tab-navigation">
      <button 
        onClick={() => setActiveTab('chat')} 
        className={`tab-btn medical-tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
      >
        <MessageSquare className="w-4 h-4" />
        <span>Chat</span>
        {chatSessionsCount > 0 && (
          <div className="tab-badge">{chatSessionsCount}</div>
        )}
        {activeTab === 'chat' && <div className="tab-indicator"></div>}
      </button>
      
      <button 
        onClick={() => setActiveTab('documents')} 
        className={`tab-btn medical-tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
      >
        <FileText className="w-4 h-4" />
        <span>Documents</span>
          {trainingStatus?.is_training && (
          <div className="training-indicator medical-training-indicator">
            <div className="indicator-pulse"></div>
          </div>
        )}
        {uploadedDocuments.length > 0 && (
          <div className="tab-badge">{uploadedDocuments.length}</div>
        )}
        {activeTab === 'documents' && <div className="tab-indicator"></div>}
      </button>

    </div>
    
   
  </div>
);

export default MedicalHeader;