// components/tabs/ChatTab.jsx - Updated to integrate with your existing App structure

import React, { useState, useEffect } from 'react';
import {
  Send,
  Sparkles,
  Activity,
  Heart,
  AlertCircle,
  CheckCircle,
  Clock,
  Settings,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import ModelSelector from '../common/ModelSelector';

const Chat = ({
  sessionId,
  backendConnected,
  backendUrl = 'http://localhost:5000',
  modelStatus,
  modelPreference,
  onModelChange,
  clearSignal = 0,
  clearChat,
  messages,
  isLoading,
  isLLMResponding = false,
  inputValue,
  setInputValue,
  handleKeyPress,
  sendMessage,
  messagesEndRef,
  inputRef,
}) => {
  const [modelSelectorCollapsed, setModelSelectorCollapsed] = useState(true);

  // Handle clear signal from parent
  useEffect(() => {
    if (clearSignal > 0) {
      // This will clear messages in the parent component
      // since messages state is managed there
      console.log('Chat cleared via signal');
    }
  }, [clearSignal]);

  // Handle model changes
  const handleModelChange = (newModel) => {
    if (onModelChange) {
      onModelChange(newModel);
    }
    console.log(`Model switched to: ${newModel}`);
  };

  // Render connection status
  const renderConnectionStatus = () => {
    if (!backendConnected) {
      return (
        <div className="connection-status error" style={{
          padding: '12px 16px',
          background: '#fee2e2',
          border: '1px solid #fecaca',
          borderRadius: '8px',
          margin: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <AlertCircle className="w-4 h-4 text-red-500" />
          <span className="text-red-700">Backend Disconnected</span>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="tab-content chat-tab" style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      overflow: 'hidden',
    }}>
      {/* Connection Status */}
      {renderConnectionStatus()}

      {/* Model Selector Header */}
      <div className="model-selector-header" style={{
        background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
        borderBottom: '1px solid #e2e8f0',
        padding: '12px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 10,
        backdropFilter: 'blur(10px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Activity className="w-5 h-5" style={{ color: '#3b82f6' }} />
          <div>
            <div style={{ fontSize: '14px', fontWeight: '600', color: '#1f2937' }}>
              Medical AI Assistant
            </div>
            <div style={{ fontSize: '12px', color: '#6b7280' }}>
              Using {modelPreference.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} Model
            </div>
          </div>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ModelSelector
            backendUrl={backendUrl}
            onModelChange={handleModelChange}
            showAdvanced={false}
          />
          
          <button
            onClick={() => setModelSelectorCollapsed(!modelSelectorCollapsed)}
            style={{
              padding: '8px',
              background: 'transparent',
              border: 'none',
              borderRadius: '6px',
              color: '#6b7280',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease',
            }}
            onMouseOver={(e) => {
              e.target.style.background = '#f3f4f6';
              e.target.style.color = '#374151';
            }}
            onMouseOut={(e) => {
              e.target.style.background = 'transparent';
              e.target.style.color = '#6b7280';
            }}
            title={modelSelectorCollapsed ? 'Show Model Details' : 'Hide Model Details'}
          >
            {modelSelectorCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Expanded Model Information Panel */}
      {!modelSelectorCollapsed && (
        <div className="model-info-panel" style={{
          background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
          borderBottom: '1px solid #e2e8f0',
          padding: '16px',
          position: 'sticky',
          top: '73px',
          zIndex: 9,
        }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '12px',
          }}>
            <div style={{
              background: 'rgba(255, 255, 255, 0.6)',
              backdropFilter: 'blur(10px)',
              borderRadius: '8px',
              padding: '12px',
              border: '1px solid rgba(59, 130, 246, 0.1)',
            }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: '#3b82f6', marginBottom: '4px' }}>
                Current Model
              </div>
              <div style={{ fontSize: '13px', fontWeight: '600', color: '#1f2937' }}>
                {modelPreference.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </div>
            </div>
            
            <div style={{
              background: 'rgba(255, 255, 255, 0.6)',
              backdropFilter: 'blur(10px)',
              borderRadius: '8px',
              padding: '12px',
              border: '1px solid rgba(16, 185, 129, 0.1)',
            }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: '#10b981', marginBottom: '4px' }}>
                Connection Status
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: backendConnected ? '#10b981' : '#ef4444',
                }}></div>
                <div style={{ fontSize: '13px', fontWeight: '600', color: '#1f2937' }}>
                  {backendConnected ? 'Connected' : 'Disconnected'}
                </div>
              </div>
            </div>
            
            <div style={{
              background: 'rgba(255, 255, 255, 0.6)',
              backdropFilter: 'blur(10px)',
              borderRadius: '8px',
              padding: '12px',
              border: '1px solid rgba(139, 92, 246, 0.1)',
            }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: '#8b5cf6', marginBottom: '4px' }}>
                Session
              </div>
              <div style={{ fontSize: '13px', fontWeight: '600', color: '#1f2937' }}>
                {sessionId ? `Active (${sessionId.slice(-8)})` : 'No Session'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Messages Container */}
      <div className="messages-container" style={{
        flex: 1,
        overflow: 'hidden',
        position: 'relative',
      }}>
        {/* Your existing chat background animation */}
        <div className="chat-background">
          {/* Add your existing background animation elements here */}
        </div>

        {/* Messages Content */}
        <div className="messages-content" style={{
          height: '100%',
          overflowY: 'auto',
          padding: '20px',
        }}>
          {messages.length > 0 ? (
            messages.map((message, index) => (
              <div
                key={message.id}
                className={`message ${message.type} ${message.isError ? 'error-message' : ''}`}
                style={{
                  display: 'flex',
                  marginBottom: '16px',
                  animation: `fadeIn 0.3s ease-out ${index * 0.1}s both`,
                }}
              >
                <div className="message-avatar" style={{ marginRight: '12px' }}>
                  {message.type === 'bot' ? (
                    <div style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '50%',
                      background: message.isError 
                        ? 'linear-gradient(135deg, #ef4444, #dc2626)' 
                        : 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                    }}>
                      <Activity className="w-5 h-5" />
                    </div>
                  ) : (
                    <div style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '50%',
                      background: 'linear-gradient(135deg, #10b981, #059669)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '14px',
                      fontWeight: '600',
                    }}>
                      U
                    </div>
                  )}
                </div>

                <div className="message-content" style={{ flex: 1 }}>
                  <div
                    className={`message-bubble ${message.type}`}
                    style={{
                      padding: '12px 16px',
                      borderRadius: '16px',
                      background: message.type === 'user' 
                        ? 'linear-gradient(135deg, #3b82f6, #1d4ed8)'
                        : message.isError
                        ? '#fee2e2'
                        : '#f8fafc',
                      color: message.type === 'user' 
                        ? 'white'
                        : message.isError
                        ? '#dc2626'
                        : '#1f2937',
                      border: message.type === 'bot' && !message.isError ? '1px solid #e2e8f0' : 'none',
                      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
                    }}
                  >
                    {message.type === 'loading' ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                        <span>Thinking...</span>
                      </div>
                    ) : (
                      <p style={{ margin: 0, whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>
                        {message.content}
                      </p>
                    )}

                    {/* Enhanced metadata for bot messages */}
                    {message.type === 'bot' && !message.isError && (message.modelUsed || message.confidence) && (
                      <div style={{
                        marginTop: '8px',
                        padding: '6px 8px',
                        background: 'rgba(59, 130, 246, 0.05)',
                        borderRadius: '6px',
                        fontSize: '11px',
                        color: '#6b7280',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}>
                        {message.modelUsed && (
                          <span>Model: {message.modelUsed}</span>
                        )}
                        {message.confidence && (
                          <span>Confidence: {Math.round(message.confidence * 100)}%</span>
                        )}
                      </div>
                    )}

                    {/* Source documents */}
                    {message.type === 'bot' &&
                      message.sourceDocuments &&
                      message.sourceDocuments.length > 0 &&
                      !message.isError && (
                        <div style={{
                          marginTop: '12px',
                          padding: '8px 12px',
                          background: 'rgba(16, 185, 129, 0.05)',
                          borderRadius: '8px',
                          border: '1px solid rgba(16, 185, 129, 0.1)',
                        }}>
                          <p style={{
                            margin: '0 0 6px 0',
                            fontSize: '11px',
                            fontWeight: '600',
                            color: '#10b981',
                          }}>
                            Context from:
                          </p>
                          <ul style={{
                            margin: 0,
                            padding: '0 0 0 16px',
                            fontSize: '11px',
                            color: '#6b7280',
                          }}>
                            {message.sourceDocuments.map((doc) => (
                              <li key={doc.id}>{doc.name}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                  </div>

                  <p style={{
                    margin: '4px 0 0 0',
                    fontSize: '11px',
                    color: '#9ca3af',
                  }}>
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
            <div className="empty-chat" style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              textAlign: 'center',
            }}>
              <div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '16px',
                  position: 'relative',
                }}>
                  <Activity className="w-12 h-12" style={{ color: '#3b82f6' }} />
                  <Heart
                    className="w-8 h-8"
                    style={{
                      color: '#ef4444',
                      position: 'absolute',
                      top: '10px',
                      right: '10px',
                      animation: 'heartbeat 1.5s ease-in-out infinite',
                    }}
                  />
                </div>
                <h3 style={{ margin: '0 0 8px 0', color: '#1f2937' }}>
                  Welcome to Your Medical AI Assistant
                </h3>
                <p style={{ margin: '0 0 16px 0', color: '#6b7280' }}>
                  Start a conversation by asking questions about your uploaded medical documents
                </p>
                <div style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '8px',
                  justifyContent: 'center',
                  marginBottom: '16px',
                }}>
                  <span style={{
                    padding: '4px 8px',
                    background: '#f0f9ff',
                    color: '#0369a1',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}>
                    🔬 Analyze symptoms
                  </span>
                  <span style={{
                    padding: '4px 8px',
                    background: '#f0f9ff',
                    color: '#0369a1',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}>
                    📋 Review reports
                  </span>
                  <span style={{
                    padding: '4px 8px',
                    background: '#f0f9ff',
                    color: '#0369a1',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}>
                    💡 Medical insights
                  </span>
                  <span style={{
                    padding: '4px 8px',
                    background: '#f0f9ff',
                    color: '#0369a1',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}>
                    🩺 Health guidance
                  </span>
                </div>
                <div style={{ fontSize: '14px', color: '#6b7280' }}>
                  Current Model: <span style={{ fontWeight: '600', color: '#3b82f6' }}>
                    {modelPreference.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="input-area" style={{
        padding: '16px',
        borderTop: '1px solid #e5e7eb',
        background: '#ffffff',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: '12px',
          position: 'relative',
        }}>
          <div style={{ flex: 1, position: 'relative' }}>
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                !backendConnected
                  ? 'Connection failed - check server'
                  : `Ask questions about your medical documents... (${modelPreference.replace('_', ' ')})`
              }
              style={{
                width: '100%',
                minHeight: '56px',
                maxHeight: '120px',
                padding: '16px 60px 16px 16px',
                border: '2px solid #e5e7eb',
                borderRadius: '12px',
                fontSize: '14px',
                resize: 'none',
                outline: 'none',
                transition: 'border-color 0.2s ease',
                fontFamily: 'inherit',
              }}
              onFocus={(e) => {
                e.target.style.borderColor = '#3b82f6';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = '#e5e7eb';
              }}
              disabled={isLoading || !backendConnected}
            />

            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading || !backendConnected}
              style={{
                position: 'absolute',
                right: '8px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '40px',
                height: '40px',
                border: 'none',
                borderRadius: '8px',
                background: (!inputValue.trim() || isLoading || !backendConnected)
                  ? '#d1d5db'
                  : 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                color: 'white',
                cursor: (!inputValue.trim() || isLoading || !backendConnected) ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s ease',
              }}
              onMouseOver={(e) => {
                if (!e.target.disabled) {
                  e.target.style.transform = 'translateY(-50%) scale(1.05)';
                }
              }}
              onMouseOut={(e) => {
                e.target.style.transform = 'translateY(-50%) scale(1)';
              }}
            >
              {isLoading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* CSS Animations */}
      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes heartbeat {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
        
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        .w-4 { width: 16px; height: 16px; }
        .w-5 { width: 20px; height: 20px; }
        .w-8 { width: 32px; height: 32px; }
        .w-12 { width: 48px; height: 48px; }
      `}</style>
    </div>
  );
};

export default Chat;