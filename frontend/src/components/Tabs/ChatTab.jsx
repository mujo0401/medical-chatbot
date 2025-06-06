// FILE: frontend/src/components/chat/ChatTab.jsx

import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Sparkles,
  Activity,
  Heart,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react';
import LoadingDots from '../common/LoadingDots';
import '../../css/chat.css';

const ChatTab = ({
  backendUrl = 'http://localhost:5000',
  clearSignal = 0,        // <-- new prop to listen for “clear” events
}) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [backendConnected, setBackendConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [retryCount, setRetryCount] = useState(0);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const maxRetries = 3;

  // 1) Whenever `clearSignal` changes, wipe out the chat and reset session
  useEffect(() => {
    // Clear messages and drop the old session
    setMessages([]);
    setSessionId(null);

    // Optionally, immediately re‐initialize the session:
    // (If you want “Clear” to also spin up a new session automatically,
    // you can call your init logic again here. For now, we'll rely on the
    // existing init‐hook below, because sessionId=null triggers it.)
    setIsInitializing(true);
    setRetryCount(0);
  }, [clearSignal]);

  // Scroll to bottom when new messages appear
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize connection + create session (runs on mount, and also whenever sessionId is null & not already initializing)
  useEffect(() => {
    const init = async () => {
      setIsInitializing(true);

      try {
        setConnectionError(null);

        // 1) Health check
        const healthController = new AbortController();
        const healthTimeout = setTimeout(() => healthController.abort(), 10000);

        const healthRes = await fetch(`${backendUrl}/health`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          signal: healthController.signal,
        });
        clearTimeout(healthTimeout);

        if (!healthRes.ok) {
          throw new Error(`Health check failed: ${healthRes.status} ${healthRes.statusText}`);
        }

        setBackendConnected(true);

        // 2) Create a new chat session (only if sessionId is null)
        if (!sessionId) {
          const sessionController = new AbortController();
          const sessionTimeout = setTimeout(() => sessionController.abort(), 10000);

          const res = await fetch(`${backendUrl}/api/chat/session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: sessionController.signal,
          });
          clearTimeout(sessionTimeout);

          if (!res.ok) {
            const errorText = await res.text().catch(() => 'Unknown error');
            throw new Error(`Session creation failed: ${res.status} ${res.statusText} - ${errorText}`);
          }

          const data = await res.json();
          if (!data.session_id) {
            throw new Error('No session_id returned from server');
          }
          setSessionId(data.session_id);
          setRetryCount(0);
        }
      } catch (err) {
        setBackendConnected(false);

        let errorMessage = err.message;
        if (err.name === 'AbortError') {
          errorMessage = 'Connection timeout: Backend server is not responding.\nPlease check if the server is running on port 5000.';
        } else if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
          errorMessage = 'Cannot connect to backend server. Please check:\n' +
                         '• Flask server is running on port 5000\n' +
                         '• CORS is properly configured\n' +
                         '• No firewall is blocking the connection';
        } else if (err.message.includes('CORS')) {
          errorMessage = 'CORS error: Backend server needs CORS configuration.\nInstall flask-cors and configure it properly.';
        } else if (err.message.includes('500')) {
          errorMessage = 'Backend server error. Check server logs for details.';
        } else if (err.message.includes('404')) {
          errorMessage = 'API endpoint not found. Check route configuration.';
        }

        setConnectionError(errorMessage);

        if (retryCount < maxRetries) {
          setTimeout(() => {
            setRetryCount(prev => prev + 1);
          }, 2000 * (retryCount + 1));
        }
      } finally {
        setIsInitializing(false);
      }
    };

    // Only call init if we don’t already have a session running
    if (!sessionId && !isInitializing) {
      init();
    } else if (!sessionId && isInitializing) {
      // On first mount, sessionId is null and isInitializing starts as true → run init
      init();
    }
  }, [backendUrl, retryCount, sessionId, isInitializing]);

  // Manual retry if desired
  const retryConnection = () => {
    setRetryCount(0);
    setIsInitializing(true);
  };

  // Handle Enter (without Shift) to send message
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && inputValue.trim() && backendConnected && sessionId) {
        sendMessage();
      }
    }
  };

  // Send a user message → show loading → show bot response
  const sendMessage = async () => {
    const userText = inputValue.trim();
    if (!userText || !backendConnected || !sessionId || isLoading) return;

    // 1) Add user bubble
    const userMsg = {
      id: `user-${Date.now()}`,
      content: userText,
      type: 'user',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    // 2) Insert temporary loading bubble
    const loadingMsg = {
      id: `loading-${Date.now()}`,
      content: '',
      type: 'loading',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, loadingMsg]);

    try {
      const chatController = new AbortController();
      const chatTimeout = setTimeout(() => chatController.abort(), 30000);

      const res = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: userText,
        }),
        signal: chatController.signal,
      });
      clearTimeout(chatTimeout);

      // Remove loading bubble first
      setMessages(prev => prev.filter(m => m.id !== loadingMsg.id));

      if (!res.ok) {
        let errorMessage;
        try {
          const errorData = await res.json();
          errorMessage = errorData.error || `HTTP ${res.status} ${res.statusText}`;
        } catch {
          errorMessage = `HTTP ${res.status} ${res.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await res.json();
      const replyText = data.reply || data.response || 'No response received';
      const sourceDocuments = Array.isArray(data.source_documents) ? data.source_documents : [];

      // 4) Insert bot bubble
      const botMsg = {
        id: `bot-${Date.now()}`,
        content: replyText,
        type: 'bot',
        timestamp: new Date(),
        sourceDocuments: sourceDocuments,
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      // Remove loading bubble if still present
      setMessages(prev => prev.filter(m => m.id !== loadingMsg.id));

      let errorMessage = err.message;
      if (err.name === 'AbortError') {
        errorMessage = 'Request timed out. The server may be overloaded. Please try again.';
      } else if (err.message.includes('Failed to fetch')) {
        errorMessage = 'Connection lost. Please check your internet connection.';
      } else if (err.message.includes('CORS')) {
        errorMessage = 'Backend configuration error. Please contact support.';
      } else if (err.message.includes('500')) {
        errorMessage = 'Server error occurred. Please try again or contact support.';
      }

      // Insert error bubble
      const errorMsg = {
        id: `error-${Date.now()}`,
        content: `Error: ${errorMessage}`,
        type: 'bot',
        timestamp: new Date(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // Render connection‐status UI
  const renderConnectionStatus = () => {
    if (isInitializing) {
      return (
        <div className="connection-status initializing">
          <div className="status-indicator">
            <Clock className="w-4 h-4 animate-spin" />
            <span>Connecting to server... {retryCount > 0 && `(Attempt ${retryCount + 1})`}</span>
          </div>
        </div>
      );
    }

    if (!backendConnected) {
      return (
        <div className="connection-status error">
          <div className="status-indicator">
            <AlertCircle className="w-4 h-4" />
            <div className="status-content">
              <span className="status-text">Connection Failed</span>
              <div className="error-details">{connectionError}</div>
              {retryCount < maxRetries && (
                <button onClick={retryConnection} className="retry-button">
                  Retry Connection
                </button>
              )}
            </div>
          </div>
        </div>
      );
    }

    if (!sessionId) {
      return (
        <div className="connection-status warning">
          <div className="status-indicator">
            <Clock className="w-4 h-4 animate-pulse" />
            <span>Creating session...</span>
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="tab-content chat-tab enhanced-chat">
      {/* Connection Status */}
      {renderConnectionStatus()}

      {/* ─────────────────────── Messages Container ───────────────────────── */}
      <div className="messages-container enhanced-messages">
        {/* Animated medical background (unchanged) */}
        <div className="chat-background">
          <div className="medical-gradient" />
          <div className="pulse-background" />

          <div className="molecules-container">
            {[...Array(12)].map((_, i) => (
              <div
                key={i}
                className="molecule"
                style={{
                  left: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 20}s`,
                  animationDuration: `${25 + Math.random() * 15}s`,
                }}
              >
                <div className="molecule-atom">
                  <div
                    className="molecule-bond"
                    style={{ transform: `rotate(${Math.random() * 360}deg)` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="medical-crosses-container">
            {[...Array(8)].map((_, i) => (
              <div
                key={i}
                className="medical-cross"
                style={{
                  top: `${20 + Math.random() * 60}%`,
                  animationDelay: `${Math.random() * 15}s`,
                  animationDuration: `${20 + Math.random() * 10}s`,
                }}
              >
                <div className="cross-vertical" />
                <div className="cross-horizontal" />
              </div>
            ))}
          </div>

          <div className="dna-container">
            <div className="dna-strand dna-strand-1" />
            <div className="dna-strand dna-strand-2" />
            {[...Array(8)].map((_, i) => (
              <div
                key={i}
                className="dna-base"
                style={{
                  top: `${i * 25}px`,
                  animationDelay: `${i * 0.5}s`,
                }}
              />
            ))}
          </div>

          <div className="ecg-container">
            <div className="ecg-line" />
          </div>

          <div className="medical-icons-container">
            {[...Array(6)].map((_, i) => {
              const iconTypes = ['pill', 'heart'];
              const iconType = iconTypes[Math.floor(Math.random() * iconTypes.length)];
              return (
                <div
                  key={i}
                  className={`medical-icon ${iconType}`}
                  style={{
                    left: `${Math.random() * 100}%`,
                    animationDelay: `${Math.random() * 25}s`,
                    animationDuration: `${30 + Math.random() * 20}s`,
                  }}
                />
              );
            })}
          </div>

          <div className="medical-particles">
            {[...Array(15)].map((_, i) => (
              <div
                key={i}
                className="medical-particle"
                style={{
                  left: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 20}s`,
                  animationDuration: `${20 + Math.random() * 15}s`,
                }}
              />
            ))}
          </div>
        </div>

        {/* ─── Chat bubbles ─── */}
        <div className="messages-content enhanced-messages-content">
          {messages.length > 0 ? (
            messages.map((message, index) => (
              <div
                key={message.id}
                className={`message enhanced-message ${message.type} ${
                  message.isError ? 'error-message' : ''
                }`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="message-avatar">
                  {message.type === 'bot' ? (
                    <div
                      className={`bot-avatar enhanced-avatar medical-avatar ${
                        message.isError ? 'error-avatar' : ''
                      }`}
                    >
                      <Activity className="w-5 h-5" />
                      <div className="avatar-pulse medical-pulse" />
                    </div>
                  ) : message.type === 'user' ? (
                    <div className="user-avatar enhanced-avatar medical-user-avatar">
                      <div className="user-initial">U</div>
                      <div className="avatar-pulse medical-pulse" />
                    </div>
                  ) : (
                    <div className="bot-avatar enhanced-avatar loading-avatar medical-loading-avatar">
                      <Activity className="w-5 h-5" />
                      <div className="avatar-pulse loading-pulse medical-loading-pulse" />
                    </div>
                  )}
                </div>

                <div className="message-content enhanced-content">
                  <div
                    className={`message-bubble medical-bubble ${message.type} ${
                      message.isError ? 'error-bubble' : ''
                    }`}
                  >
                    {message.type === 'loading' ? (
                      <LoadingDots />
                    ) : (
                      <p style={{ whiteSpace: 'pre-wrap' }}>{message.content}</p>
                    )}

                    {message.type === 'bot' &&
                      message.sourceDocuments &&
                      message.sourceDocuments.length > 0 &&
                      !message.isError && (
                        <div className="message-source enhanced-source medical-source">
                          <p className="source-heading">Context from:</p>
                          <ul className="source-list">
                            {message.sourceDocuments.map((doc) => (
                              <li key={doc.id}>{doc.name}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                    <div className="message-decorations medical-decorations">
                      <div className="decoration-line medical-line" />
                    </div>
                  </div>

                  <p className="message-time enhanced-time medical-time">
                    {message.timestamp
                      ? new Date(message.timestamp).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })
                      : new Date().toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                  </p>
                </div>
              </div>
            ))
          ) : (
            <div className="empty-chat enhanced-empty medical-empty">
              <div className="empty-content">
                <div className="empty-icon-container medical-icon-container">
                  <div className="empty-icon-wrapper medical-icon-wrapper">
                    <Activity className="w-12 h-12 text-blue-500" />
                    <Heart
                      className="w-8 h-8 text-red-400 heartbeat"
                      style={{ position: 'absolute', top: '10px', right: '10px' }}
                    />
                  </div>
                </div>
                <div className="empty-text medical-empty-text">
                  <h3>Welcome to Your Medical AI Assistant</h3>
                  <p>Start a conversation by asking questions about your uploaded medical documents</p>
                  <div className="suggestion-pills medical-pills">
                    <span className="pill medical-pill">🔬 Analyze symptoms</span>
                    <span className="pill medical-pill">📋 Review reports</span>
                    <span className="pill medical-pill">💡 Medical insights</span>
                    <span className="pill medical-pill">🩺 Health guidance</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ─────────────────────── Input Area ───────────────────────── */}
      <div className="input-area enhanced-input medical-input">
        <div className="input-container enhanced-input-container medical-input-container">
          <div className="flex-1 relative">
            <div className="input-wrapper medical-input-wrapper">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  isInitializing
                    ? 'Connecting to server...'
                    : !backendConnected
                    ? 'Connection failed - check server'
                    : !sessionId
                    ? 'Setting up session...'
                    : 'Ask questions about your uploaded medical documents...'
                }
                className="message-input enhanced-message-input medical-message-input"
                rows={1}
                style={{ minHeight: '56px', maxHeight: '120px' }}
                disabled={isLoading || !backendConnected || !sessionId || isInitializing}
              />
              <div className="input-glow medical-glow" />
              <div className="input-border-animation medical-border-animation" />
            </div>

            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading || !backendConnected || !sessionId || isInitializing}
              className="send-btn enhanced-send-btn medical-send-btn"
              style={{
                position: 'absolute',
                right: '8px',
                top: '50%',
                transform: 'translateY(-50%)',
                minWidth: '48px',
                height: '48px',
                padding: '12px',
              }}
            >
              <div className="btn-content medical-btn-content">
                {isLoading ? <Sparkles className="w-5 h-5 spinning" /> : <Send className="w-5 h-5" />}
              </div>
              <div className="btn-ripple medical-ripple" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatTab;
