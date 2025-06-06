// src/App.jsx

import React, { useState, useEffect, useCallback, useRef } from 'react';

import Chat from './components/Tabs/ChatTab';
import DocumentsTab from './components/Tabs/DocumentsTab';
//import TrainingTab from './components/Tabs/TrainingTab';
//import AnalyticsTab from './components/Tabs/AnalyticsTab';
//import PatientsTab from './components/Tabs/PatientsTab';
//import ReportsTab from './components/Tabs/ReportsTab';
//import SettingsTab from './components/Tabs/SettingsTab';
import Sidebar from './components/common/sidebar';
import Header from './components/common/header';

import './css/base.css';
import './css/animations.css';
import './css/layout.css';
import './css/sidebar-content.css';
import './css/buttons.css';
import './css/model-selector.css';
import './css/sidebar-training.css';
import './css/chat.css';
import './css/documents.css';
import './css/training.css';
import './css/analytics.css';
import './css/patients.css';
import './css/reports.css';
import './css/settings.css';
import './css/messages.css';
import './css/input.css';
import './css/responsive.css';

import { useBackendConnection } from './hooks/useBackendConnection';
import { useApi } from './hooks/useApi';

function App() {
  // ─── Configuration ─────────────────────────────────────────────────────────────
  const BACKEND_URL = 'http://localhost:5000';

  // ─── Global Data States ───────────────────────────────────────────────────────
  const [documents, setDocuments] = useState([]);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState({ is_training: false });
  const [patients, setPatients] = useState([]);
  const [dataLoaded, setDataLoaded] = useState(false);

  // ─── Sidebar-related State ────────────────────────────────────────────────────
  // 1) Whether the sidebar is open or closed
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // 2) Which tab is currently active
  const [activeTab, setActiveTab] = useState('chat');

  // 3) Which AI-model/config is selected
  const [modelPreference, setModelPreference] = useState('openai');

  // 4) Chat‐sessions list & current session
  const [chatSessions, setChatSessions] = useState([
    // Example session objects; you can preload or fetch these from localStorage/DB
    { id: 's1', name: 'Consultation #1', lastMessage: 'Patient: Hello', timestamp: Date.now(), patientId: null },
    { id: 's2', name: 'Consultation #2', lastMessage: 'Doctor: Please clarify…', timestamp: Date.now(), patientId: null },
  ]);
  const [currentSession, setCurrentSession] = useState(
    chatSessions.length ? chatSessions[0].id : null
  );

  const [clearSignal, setClearSignal] = useState(0);

  // ─── Chat‐input State ──────────────────────────────────────────────────────────
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Refs for scrolling & focusing:
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // When pressing Enter (without Shift), send the message:
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const sendMessage = async () => {
    if (!chatInput.trim()) return;

    const userMsg = {
      id: Date.now().toString(),
      type: 'user',
      content: chatInput.trim(),
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setChatInput('');

    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 50);

    setIsLoading(true);
    try {
      const apiResult = await sendApiMessage(userMsg.content, currentSession);

      const botMsg = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: apiResult.response,
        timestamp: new Date(apiResult.timestamp),
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const errorMsg = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: 'Failed to fetch response from server.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
      console.error('sendMessage error:', err);
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 50);
    }
  };

  // ─── Backend / API Hooks ──────────────────────────────────────────────────────
  const { backendConnected, modelStatus, checkConnection } = useBackendConnection();
  const { getDocuments, getTrainingHistory, getTrainingStatus, loading, sendMessage: sendApiMessage } = useApi();

  // ─── Load Initial Data ────────────────────────────────────────────────────────
  const loadInitialData = useCallback(async () => {
    if (!backendConnected || dataLoaded || loading) return;

    try {
      // Fetch Documents
      try {
        const docs = await getDocuments();
        setDocuments(Array.isArray(docs) ? docs : []);
      } catch (error) {
        console.error('Failed to fetch documents:', error.message);
        setDocuments([]);
      }

      // Fetch Training History
      try {
        const history = await getTrainingHistory();
        setTrainingHistory(Array.isArray(history) ? history : []);
      } catch (error) {
        console.error('Failed to fetch training history:', error.message);
        setTrainingHistory([]);
      }

      // Fetch Training Status
      try {
        const status = await getTrainingStatus();
        setTrainingStatus(status || { is_training: false });
      } catch (error) {
        console.error('Failed to fetch training status:', error.message);
        setTrainingStatus({ is_training: false });
      }

      // Fetch Patients
      try {
        const response = await fetch(`${BACKEND_URL}/api/patients`);
        if (response.ok) {
          const data = await response.json();
          setPatients(Array.isArray(data.patients) ? data.patients : []);
        }
      } catch (error) {
        console.error('Failed to fetch patients:', error.message);
        setPatients([]);
      }

      setDataLoaded(true);
    } catch (error) {
      console.error('Failed to load initial data:', error);
      setDataLoaded(true);
    }
  }, [backendConnected, dataLoaded, loading, getDocuments, getTrainingHistory, getTrainingStatus]);

  useEffect(() => {
    if (backendConnected && !dataLoaded) {
      loadInitialData();
    }
  }, [backendConnected, loadInitialData, dataLoaded]);

  // ─── Refresh Helpers ──────────────────────────────────────────────────────────
  const refreshDocuments = useCallback(async () => {
    if (!backendConnected) return;
    try {
      const docs = await getDocuments();
      setDocuments(Array.isArray(docs) ? docs : []);
    } catch (error) {
      console.error('Failed to refresh documents:', error.message);
    }
  }, [backendConnected, getDocuments]);

  const refreshTrainingData = useCallback(async () => {
    if (!backendConnected) return;
    try {
      const [history, status] = await Promise.all([
        getTrainingHistory().catch(() => []),
        getTrainingStatus().catch(() => ({ is_training: false })),
      ]);
      setTrainingHistory(Array.isArray(history) ? history : []);
      setTrainingStatus(status || { is_training: false });
    } catch (error) {
      console.error('Failed to refresh training data:', error.message);
    }
  }, [backendConnected, getTrainingHistory, getTrainingStatus]);

  const refreshPatients = useCallback(async () => {
    if (!backendConnected) return;
    try {
      const response = await fetch(`${BACKEND_URL}/api/patients`);
      if (response.ok) {
        const data = await response.json();
        setPatients(Array.isArray(data.patients) ? data.patients : []);
      }
    } catch (error) {
      console.error('Failed to refresh patients:', error.message);
    }
  }, [backendConnected]);

  // ─── Auto-refresh training status when training is active ────────────────────
  useEffect(() => {
    if (!backendConnected || !trainingStatus?.is_training) return;

    const interval = setInterval(() => {
      refreshTrainingData();
    }, 5000);

    return () => clearInterval(interval);
  }, [backendConnected, trainingStatus?.is_training, refreshTrainingData]);

  // ─── Handle training completion notifications ─────────────────────────────────
  useEffect(() => {
    const handleTrainingComplete = () => {
      setTimeout(() => {
        refreshTrainingData();
      }, 1000);
    };

    if (typeof window !== 'undefined') {
      window.addEventListener('trainingComplete', handleTrainingComplete);
      return () => {
        window.removeEventListener('trainingComplete', handleTrainingComplete);
      };
    }
  }, [refreshTrainingData]);

  // ─── Handlers for Sidebar Actions ─────────────────────────────────────────────
  const newChatSession = () => {
    const newId = `s${Date.now()}`;
    const newSessionObj = {
      id: newId,
      name: `Consultation #${chatSessions.length + 1}`,
      lastMessage: '',
      timestamp: Date.now(),
      patientId: null,
    };
    setChatSessions((prev) => [newSessionObj, ...prev]);
    setCurrentSession(newId);
    setActiveTab('chat');
    setMessages([]);
    setChatInput('');
  };

  const clearChat = () => {
    setMessages([]);
    setChatInput('');
  };

  const setModel = (option) => {
    setModelPreference(option);
    console.log(`⚙️  Model switched to:`, option);
  };

  const removeSession = (sessionId) => {
    setChatSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== sessionId);
      return filtered;
    });
    if (currentSession === sessionId) {
      const remaining = chatSessions.filter((s) => s.id !== sessionId);
      if (remaining.length > 0) {
        setCurrentSession(remaining[0].id);
      } else {
        setCurrentSession(null);
      }
    }
  };

  const handleClearChat = () => {
    // bump the signal → ChatTab's useEffect([clearSignal]) will run
    setClearSignal(prev => prev + 1);
  };


  // ─── Manual "Retry Connection" ─────────────────────────────────────────────────
  const handleRefresh = useCallback(() => {
    setDataLoaded(false);
    checkConnection();
  }, [checkConnection]);

  // ─── Training event handlers ──────────────────────────────────────────────────
  const handleTrainingStarted = useCallback(
    (trainingInfo) => {
      console.log('Training started:', trainingInfo);
      refreshTrainingData();
      // Note: Don't auto-switch to training tab since DocumentsTab now has integrated monitoring
    },
    [refreshTrainingData]
  );

  const handleTrainingCompleted = useCallback(
    (trainingInfo) => {
      console.log('Training completed:', trainingInfo);
      Promise.all([refreshTrainingData(), refreshDocuments()]);
      if (trainingInfo?.success) {
        console.log('Training completed successfully!');
      } else {
        console.error('Training failed:', trainingInfo?.error);
      }
    },
    [refreshTrainingData, refreshDocuments]
  );

  // ─── Render the currently selected tab's component ────────────────────────────
  const renderActiveTab = () => {
    switch (activeTab) {
      case 'chat':
        return (
          <Chat
            key={currentSession}
            sessionId={currentSession}
            backendConnected={backendConnected}
            modelStatus={modelStatus}
            modelPreference={modelPreference}
            clearChat={handleClearChat}
            messages={messages}
            isLoading={isLoading}
            inputValue={chatInput}
            setInputValue={setChatInput}
            handleKeyPress={handleKeyPress}
            sendMessage={sendMessage}
            messagesEndRef={messagesEndRef}
            inputRef={inputRef}
          />
        );

      case 'documents':
        return (
          <DocumentsTab
            documents={documents}
            onDocumentUploaded={refreshDocuments}
            onDocumentDeleted={refreshDocuments}
            backendConnected={backendConnected}
            modelStatus={modelStatus}
            trainingStatus={trainingStatus}
            backendUrl={BACKEND_URL}
          />
        );

      default:
        return null;
    }
  };

  return (
    <div className="app-container">
      {/* ─── Sidebar ─────────────────────────────────────────────────────────────── */}
      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        newChatSession={newChatSession}
        clearChat={clearChat}
        modelPreference={modelPreference}
        setModel={setModel}
        backendConnected={backendConnected}
        trainingStatus={trainingStatus}
        chatSessions={chatSessions}
        currentSession={currentSession}
        setCurrentSession={(sessionId) => {
          setCurrentSession(sessionId);
          setActiveTab('chat');
        }}
          removeSession={() => {}}
        isLLMResponding={false}
      />

      {/* ─── Main Content (Header + Tab) ───────────────────────────────────────────── */}
      <div className="main-content">
        <Header
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          backendConnected={backendConnected}
          modelStatus={modelStatus}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          uploadedDocuments={documents}
          trainingStatus={trainingStatus}
          modelPreference={modelPreference}
          patientsCount={patients.length}
          chatSessionsCount={chatSessions.length}
        />

        {!backendConnected ? (
          <div className="tab-content">
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p>Connecting to Medical AI Backend...</p>
                <small>Please ensure the backend server is running</small>
                <div style={{ marginTop: '16px' }}>
                  <button
                    onClick={handleRefresh}
                    style={{
                      padding: '8px 16px',
                      background: '#3b82f6',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '14px',
                    }}
                  >
                    Retry Connection
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          renderActiveTab()
        )}
      </div>

      {/* Global notification area for training events */}
      {trainingStatus?.is_training && activeTab !== 'documents' && (
        <div
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            background: 'rgba(59, 130, 246, 0.95)',
            color: 'white',
            padding: '12px 16px',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            backdropFilter: 'blur(10px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '500',
            cursor: 'pointer',
          }}
          onClick={() => setActiveTab('documents')}
          title="Click to view training progress"
        >
          <div className="spinning">⚙️</div>
          <span>Training in progress... {trainingStatus.progress}%</span>
          <span style={{ fontSize: '12px', opacity: 0.8 }}>→</span>
        </div>
      )}

      {/* System Status Notification */}
      {!backendConnected && (
        <div
          style={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            background: 'rgba(239, 68, 68, 0.95)',
            color: 'white',
            padding: '12px 16px',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            backdropFilter: 'blur(10px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '500',
          }}
        >
          <div>⚠️</div>
          <span>Backend Disconnected</span>
        </div>
      )}

      {/* Success notification for completed training */}
      {trainingStatus?.progress === 100 && !trainingStatus?.is_training && (
        <div
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            background: 'rgba(16, 185, 129, 0.95)',
            color: 'white',
            padding: '12px 16px',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            backdropFilter: 'blur(10px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '500',
            animation: 'slideInRight 0.3s ease-out',
          }}
        >
          <div>✅</div>
          <span>Training completed successfully!</span>
        </div>
      )}
    </div>
  );
}

export default App;