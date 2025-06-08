// components/common/sidebar.jsx - Updated for Enhanced Model Support

import React, { useState, useEffect } from 'react';
import { 
  X,
  Trash2,
  Zap,
  Cpu,
  Plus,
  Activity,
  Brain,
  Settings,
  MessageCircle,
  Clock,
  Loader2,
  Cloud,
  GitMerge,
  Sparkles,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';

// DNA Animation Components (keeping your existing animation)
const DNAParticle = ({ id, type, initialX, initialY, delay = 0 }) => {
  const [position, setPosition] = useState({ x: initialX, y: initialY });
  const [scale, setScale] = useState(1);
  const [opacity, setOpacity] = useState(0.7);
  
  useEffect(() => {
    const animate = () => {
      const time = Date.now() * 0.001 + delay;
      
      const newX = initialX + Math.sin(time * 0.8) * 12 + Math.cos(time * 0.3) * 8;
      const newY = initialY + Math.cos(time * 0.6) * 10 + Math.sin(time * 0.4) * 6;
      const newScale = 0.8 + Math.sin(time * 1.2) * 0.3;
      const newOpacity = 0.6 + Math.cos(time * 0.9) * 0.25;
      
      setPosition({ x: newX, y: newY });
      setScale(newScale);
      setOpacity(newOpacity);
    };
    
    const interval = setInterval(animate, 60);
    return () => clearInterval(interval);
  }, [initialX, initialY, delay]);
  
  const getParticleStyle = () => {
    const baseStyle = {
      position: 'absolute',
      borderRadius: '50%',
      transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
      opacity: opacity,
      transition: 'transform 0.06s ease-out, opacity 0.06s ease-out',
      pointerEvents: 'none',
      zIndex: 2,
    };
    
    switch (type) {
      case 'blue':
        return {
          ...baseStyle,
          width: '6px',
          height: '6px',
          background: 'radial-gradient(circle, rgba(59, 130, 246, 0.9), rgba(59, 130, 246, 0.5))',
          boxShadow: '0 0 10px rgba(59, 130, 246, 0.5)',
        };
      case 'green':
        return {
          ...baseStyle,
          width: '5px',
          height: '5px',
          background: 'radial-gradient(circle, rgba(16, 185, 129, 0.9), rgba(16, 185, 129, 0.5))',
          boxShadow: '0 0 8px rgba(16, 185, 129, 0.5)',
        };
      case 'purple':
        return {
          ...baseStyle,
          width: '4px',
          height: '4px',
          background: 'radial-gradient(circle, rgba(139, 92, 246, 0.8), rgba(139, 92, 246, 0.4))',
          boxShadow: '0 0 6px rgba(139, 92, 246, 0.4)',
        };
      default:
        return baseStyle;
    }
  };
  
  return <div style={getParticleStyle()} />;
};

const DNAConnection = ({ startX, startY, endX, endY, delay = 0 }) => {
  const [opacity, setOpacity] = useState(0.3);
  const [thickness, setThickness] = useState(1);
  
  useEffect(() => {
    const animate = () => {
      const time = Date.now() * 0.001 + delay;
      const newOpacity = 0.2 + Math.sin(time * 1.5) * 0.3;
      const newThickness = 1 + Math.cos(time * 1.8) * 1;
      
      setOpacity(newOpacity);
      setThickness(newThickness);
    };
    
    const interval = setInterval(animate, 100);
    return () => clearInterval(interval);
  }, [delay]);
  
  const angle = Math.atan2(endY - startY, endX - startX) * 180 / Math.PI;
  const length = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);
  
  const style = {
    position: 'absolute',
    left: `${startX}px`,
    top: `${startY}px`,
    width: `${length}px`,
    height: `${thickness}px`,
    background: `linear-gradient(90deg, 
      rgba(59, 130, 246, ${opacity}), 
      rgba(16, 185, 129, ${opacity * 0.7}), 
      rgba(139, 92, 246, ${opacity}))`,
    transform: `rotate(${angle}deg)`,
    transformOrigin: '0 50%',
    borderRadius: '1px',
    pointerEvents: 'none',
    zIndex: 2,
    transition: 'opacity 0.1s ease-out',
  };
  
  return <div style={style} />;
};

const DNAAnimationOverlay = () => {
  const [backgroundOpacity, setBackgroundOpacity] = useState(1);
  
  useEffect(() => {
    const animateBackground = () => {
      const time = Date.now() * 0.001;
      const newOpacity = 0.7 + Math.sin(time * 0.5) * 0.2;
      setBackgroundOpacity(newOpacity);
    };
    
    const interval = setInterval(animateBackground, 200);
    return () => clearInterval(interval);
  }, []);
  
  const containerStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    width: '100%',
    height: '100%',
    overflow: 'hidden',
    pointerEvents: 'none',
    zIndex: 1,
  };
  
  const particles = [
    { type: 'blue', x: 15, y: 12, delay: 0 },
    { type: 'blue', x: 120, y: 35, delay: 0.5 },
    { type: 'blue', x: 200, y: 18, delay: 1 },
    { type: 'green', x: 45, y: 25, delay: 0.3 },
    { type: 'green', x: 90, y: 50, delay: 0.8 },
    { type: 'green', x: 160, y: 30, delay: 1.3 },
    { type: 'purple', x: 30, y: 40, delay: 0.7 },
    { type: 'purple', x: 75, y: 15, delay: 1.2 },
  ];
  
  const connections = [
    { startX: 15, startY: 12, endX: 45, endY: 25, delay: 0 },
    { startX: 90, startY: 50, endX: 120, endY: 35, delay: 0.5 },
    { startX: 160, startY: 30, endX: 200, endY: 18, delay: 1 },
  ];
  
  return (
    <div style={containerStyle}>
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
          radial-gradient(circle at 20% 30%, rgba(59, 130, 246, ${backgroundOpacity * 0.12}) 0%, transparent 50%),
          radial-gradient(circle at 80% 70%, rgba(16, 185, 129, ${backgroundOpacity * 0.12}) 0%, transparent 50%),
          radial-gradient(circle at 60% 20%, rgba(139, 92, 246, ${backgroundOpacity * 0.10}) 0%, transparent 45%)
        `,
        pointerEvents: 'none',
      }} />
      
      {particles.map((particle, index) => (
        <DNAParticle
          key={`particle-${index}`}
          id={index}
          type={particle.type}
          initialX={particle.x}
          initialY={particle.y}
          delay={particle.delay}
        />
      ))}
      
      {connections.map((connection, index) => (
        <DNAConnection
          key={`connection-${index}`}
          startX={connection.startX}
          startY={connection.startY}
          endX={connection.endX}
          endY={connection.endY}
          delay={connection.delay}
        />
      ))}
    </div>
  );
};

// Enhanced LLM Response Indicator
const LLMResponseIndicator = ({ isResponding, modelType }) => {
  const [dots, setDots] = useState('');
  
  useEffect(() => {
    if (!isResponding) return;
    
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    
    return () => clearInterval(interval);
  }, [isResponding]);
  
  if (!isResponding) return null;
  
  const getModelDisplayName = (type) => {
    switch (type) {
      case 'openai': return 'GPT-4';
      case 'local_trained': return 'Local AI';
      case 'eleuther': return 'EleutherAI';
      case 'hybrid_local_eleuther': return 'Hybrid AI';
      case 'hybrid_all': return 'Multi-AI';
      default: return 'AI';
    }
  };
  
  return (
    <div style={{
      position: 'fixed',
      top: '80px',
      right: '24px',
      background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.95), rgba(16, 185, 129, 0.95))',
      color: 'white',
      padding: '12px 16px',
      borderRadius: '16px',
      boxShadow: '0 8px 32px rgba(59, 130, 246, 0.3)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      zIndex: 1000,
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      animation: 'slideInFromRight 0.3s ease-out',
    }}>
      <Loader2 className="w-4 h-4 animate-spin" />
      <span style={{ fontSize: '14px', fontWeight: '600' }}>
        {getModelDisplayName(modelType)} thinking{dots}
      </span>
    </div>
  );
};

// Main Enhanced Sidebar Component
const Sidebar = ({
  sidebarOpen = true,
  setSidebarOpen = () => {},
  newChatSession = () => {},
  clearChat = () => {},
  modelPreference = 'hybrid_local_eleuther',
  setModel = () => {},
  backendConnected = false,
  trainingStatus = null,
  chatSessions = [], 
  currentSession = null,
  setCurrentSession = () => {},
  removeSession = () => {},
  isLLMResponding = false,
}) => {
  const [confirmClear, setConfirmClear] = useState(false);
  
  // Enhanced model configurations with new models
  const modelConfigs = {
    openai: {
      key: 'openai',
      icon: Cloud,
      label: 'OpenAI GPT-4',
      desc: 'Advanced reasoning & general knowledge',
      color: '#10b981',
      gradient: 'linear-gradient(135deg, #10b981, #059669)',
    },
    local_trained: {
      key: 'local_trained',
      icon: Cpu,
      label: 'Local Trained',
      desc: 'Fine-tuned on your medical data',
      color: '#3b82f6',
      gradient: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
    },
    eleuther: {
      key: 'eleuther',
      icon: Brain,
      label: 'EleutherAI',
      desc: 'Open-source large language model',
      color: '#8b5cf6',
      gradient: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
    },
    hybrid_local_eleuther: {
      key: 'hybrid_local_eleuther',
      icon: GitMerge,
      label: 'Hybrid Local+Eleuther',
      desc: 'Best of local training & EleutherAI',
      color: '#f59e0b',
      gradient: 'linear-gradient(135deg, #f59e0b, #d97706)',
    },
    hybrid_all: {
      key: 'hybrid_all',
      icon: Sparkles,
      label: 'Multi-Model Hybrid',
      desc: 'Combines all available models',
      color: '#ef4444',
      gradient: 'linear-gradient(135deg, #ef4444, #dc2626)',
    },
  };
  
  // Filter and format sessions to match current session
  const formattedSessions = chatSessions.map(session => ({
    ...session,
    isActive: session.id === currentSession,
    lastMessage: session.messages && session.messages.length > 0 
      ? session.messages[session.messages.length - 1].content 
      : session.lastMessage || 'New conversation',
    timestamp: session.timestamp || session.createdAt || new Date().toISOString()
  })).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  // Get current model config
  const currentModelConfig = modelConfigs[modelPreference] || modelConfigs.hybrid_local_eleuther;

  return (
    <>
      {/* Enhanced LLM Response Indicator */}
      <LLMResponseIndicator isResponding={isLLMResponding} modelType={modelPreference} />
      
      <div className={`sidebar ${sidebarOpen ? '' : 'closed'}`} style={{
        width: sidebarOpen ? '320px' : '0',
        background: '#ffffff',
        borderRight: '1px solid #f3f4f6',
        transition: 'all 0.3s ease',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        {/* Enhanced Header with DNA Animation */}
        <div style={{
          position: 'relative',
          padding: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: '#fefefe',
          borderBottom: '1px solid #f9fafb',
          minHeight: '100px',
          overflow: 'hidden',
          zIndex: 1
        }}>
          <DNAAnimationOverlay />
          
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            position: 'relative',
            zIndex: 10,
            flex: 1
          }}>
            <span style={{
              fontSize: '18px',
              fontWeight: '600',
              color: '#1f2937',
              textShadow: '0 0 20px rgba(255, 255, 255, 0.9)',
              letterSpacing: '-0.02em',
              position: 'relative',
              zIndex: 15,
              background: 'rgba(255, 255, 255, 0.9)',
              padding: '8px 16px',
              borderRadius: '12px',
              backdropFilter: 'blur(15px)',
              border: '1px solid rgba(255, 255, 255, 0.4)',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
            }}>
              Medical Assistant
            </span>
          </div>
          
          <button 
            onClick={() => setSidebarOpen(false)} 
            style={{
              background: 'rgba(255, 255, 255, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.4)',
              color: '#9ca3af',
              cursor: 'pointer',
              padding: '10px',
              borderRadius: '10px',
              transition: 'all 0.2s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              zIndex: 15,
              backdropFilter: 'blur(15px)',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
            }}
            onMouseOver={(e) => {
              e.target.style.background = 'rgba(255, 255, 255, 1)';
              e.target.style.color = '#374151';
              e.target.style.boxShadow = '0 6px 16px rgba(0, 0, 0, 0.15)';
              e.target.style.transform = 'translateY(-1px)';
            }}
            onMouseOut={(e) => {
              e.target.style.background = 'rgba(255, 255, 255, 0.9)';
              e.target.style.color = '#9ca3af';
              e.target.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';
              e.target.style.transform = 'translateY(0)';
            }}
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '24px 20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '24px',
          background: '#fafafa'
        }}>
          {/* Action Card */}
          <div style={{
            background: '#ffffff',
            borderRadius: '16px',
            padding: '20px',
            boxShadow: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
            border: '1px solid #f9fafb',
            transition: 'all 0.2s ease'
          }}>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '12px'
            }}>
              <button 
                onClick={newChatSession}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '14px 16px',
                  border: 'none',
                  borderRadius: '12px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  textAlign: 'left',
                  width: '100%',
                  background: 'linear-gradient(135deg, #6366f1, #06b6d4)',
                  color: 'white'
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-1px)';
                  e.target.style.boxShadow = '0 4px 6px -1px rgb(0 0 0 / 0.1)';
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = 'none';
                }}
              >
                <Plus className="w-4 h-4" />
                New Conversation
              </button>
              
              <button 
                onClick={() => {
                  console.log('Clear button clicked');
                  if (!confirmClear) {
                    console.log('Setting confirm to true');
                    setConfirmClear(true);
                    setTimeout(() => {
                      console.log('Resetting confirm to false');
                      setConfirmClear(false);
                    }, 3000);
                  } else {
                    console.log('Clearing chat');
                    clearChat();
                    setConfirmClear(false);
                  }
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '14px 16px',
                  border: '1px solid #f3f4f6',
                  borderRadius: '12px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  textAlign: 'left',
                  width: '100%',
                  background: confirmClear ? '#fee2e2' : '#fafafa',
                  color: confirmClear ? '#dc2626' : '#374151',
                  borderColor: confirmClear ? '#fecaca' : '#f3f4f6'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = confirmClear ? '#fecaca' : '#f9fafb';
                  e.currentTarget.style.borderColor = '#e5e7eb';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = confirmClear ? '#fee2e2' : '#fafafa';
                  e.currentTarget.style.borderColor = confirmClear ? '#fecaca' : '#f3f4f6';
                }}
              >
                <Trash2 className="w-4 h-4" />
                {confirmClear ? 'Click again to confirm' : 'Clear Current Chat'}
              </button>
            </div>
          </div>
          
          {/* Enhanced AI Model Selector */}
          <div style={{
            background: '#ffffff',
            borderRadius: '16px',
            padding: '18px',
            boxShadow: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
            border: '1px solid #f9fafb',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '4px'
            }}>
              <Settings className="w-4 h-4" style={{ color: '#6366f1' }} />
              <span style={{
                fontSize: '14px',
                fontWeight: '600',
                color: '#374151'
              }}>
                AI Model Selection
              </span>
            </div>
            
            {/* Current Model Display */}
            <div style={{
              padding: '12px 16px',
              background: currentModelConfig.gradient,
              borderRadius: '12px',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              marginBottom: '8px'
            }}>
              <currentModelConfig.icon className="w-5 h-5" />
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '13px', fontWeight: '600' }}>
                  {currentModelConfig.label}
                </div>
                <div style={{ fontSize: '11px', opacity: 0.9 }}>
                  {currentModelConfig.desc}
                </div>
              </div>
              <CheckCircle className="w-4 h-4" />
            </div>
            
            {/* Model Options */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '8px'
            }}>
              {Object.values(modelConfigs).map((model) => {
                const isSelected = modelPreference === model.key;
                const isAvailable = backendConnected; // You can enhance this with actual model availability
                
                return (
                  <div
                    key={model.key}
                    onClick={() => isAvailable && !isLLMResponding && setModel(model.key)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '10px 12px',
                      background: isSelected ? 'rgba(99, 102, 241, 0.1)' : '#fefefe',
                      borderRadius: '10px',
                      border: isSelected ? '1px solid rgba(99, 102, 241, 0.3)' : '1px solid #f3f4f6',
                      cursor: (isAvailable && !isLLMResponding) ? 'pointer' : 'not-allowed',
                      transition: 'all 0.2s ease',
                      opacity: (isAvailable && !isLLMResponding) ? 1 : 0.6
                    }}
                    onMouseOver={(e) => {
                      if (isAvailable && !isLLMResponding && !isSelected) {
                        e.currentTarget.style.background = '#f9fafb';
                        e.currentTarget.style.borderColor = '#e5e7eb';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.background = '#fefefe';
                        e.currentTarget.style.borderColor = '#f3f4f6';
                      }
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{
                        padding: '6px',
                        borderRadius: '8px',
                        background: isSelected ? model.gradient : '#f3f4f6',
                        color: isSelected ? 'white' : model.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <model.icon className="w-4 h-4" />
                      </div>
                      <div>
                        <div style={{ 
                          fontSize: '12px', 
                          fontWeight: '600', 
                          color: isSelected ? '#6366f1' : '#1f2937' 
                        }}>
                          {model.label}
                        </div>
                        <div style={{ fontSize: '10px', color: '#6b7280' }}>
                          {model.desc}
                        </div>
                      </div>
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {!isAvailable && <AlertTriangle className="w-3 h-3" style={{ color: '#f59e0b' }} />}
                      {isLLMResponding && isSelected && <Loader2 className="w-3 h-3 animate-spin" style={{ color: '#6366f1' }} />}
                      <div style={{
                        width: '24px',
                        height: '14px',
                        background: isSelected ? '#6366f1' : '#e5e7eb',
                        borderRadius: '14px',
                        position: 'relative',
                        transition: 'all 0.2s ease'
                      }}>
                        <div style={{
                          position: 'absolute',
                          top: '2px',
                          left: isSelected ? '12px' : '2px',
                          width: '10px',
                          height: '10px',
                          background: 'white',
                          borderRadius: '50%',
                          transition: 'all 0.2s ease',
                          boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)'
                        }} />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* Model Status */}
            <div style={{
              fontSize: '10px',
              color: '#6b7280',
              textAlign: 'center',
              paddingTop: '8px',
              borderTop: '1px solid #f3f4f6'
            }}>
              {backendConnected ? '✅ Models Available' : '⚠️ Backend Disconnected'}
            </div>
          </div>
          
          {/* Training Progress - Only show when training */}
          {trainingStatus?.is_training && (
            <div style={{
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.03), rgba(6, 182, 212, 0.03))',
              borderRadius: '16px',
              padding: '18px',
              border: '1px solid rgba(99, 102, 241, 0.1)'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '13px',
                fontWeight: '500',
                color: '#6366f1',
                marginBottom: '12px'
              }}>
                <Settings className="w-4 h-4 spinning" />
                Processing Data
              </div>
              
              <div style={{
                height: '6px',
                background: '#f3f4f6',
                borderRadius: '3px',
                overflow: 'hidden',
                marginBottom: '8px'
              }}>
                <div style={{
                  height: '100%',
                  background: 'linear-gradient(90deg, #6366f1, #06b6d4)',
                  transition: 'width 0.3s ease',
                  borderRadius: '3px',
                  width: `${trainingStatus.progress || 0}%`
                }} />
              </div>
              
              <div style={{
                fontSize: '11px',
                color: '#6b7280',
                textAlign: 'center'
              }}>
                {trainingStatus.progress || 0}% - Learning from data
              </div>
              
              {trainingStatus.current_document && (
                <div style={{
                  marginTop: '10px',
                  paddingTop: '10px',
                  borderTop: '1px solid #f9fafb'
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: '10px',
                    color: '#6b7280',
                    fontWeight: '500'
                  }}>
                    <Activity className="w-3 h-3" />
                    Processing: {trainingStatus.current_document}
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Enhanced Recent Chats */}
          <div style={{
            background: '#ffffff',
            borderRadius: '16px',
            padding: '18px',
            boxShadow: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
            border: '1px solid #f9fafb',
            marginTop: 'auto'
          }}>
            <div style={{
              fontSize: '14px',
              fontWeight: '600',
              color: '#374151',
              marginBottom: '12px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <MessageCircle className="w-4 h-4" />
              Recent Chats
            </div>
            
            <div style={{
              maxHeight: '200px',
              overflowY: 'auto'
            }}>
              {formattedSessions && formattedSessions.length > 0 ? (
                formattedSessions.slice(0, 6).map(session => (
                  <div
                    key={session.id}
                    onClick={() => setCurrentSession(session.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '10px 12px',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      marginBottom: '4px',
                      background: session.isActive ? 'rgba(99, 102, 241, 0.05)' : 'transparent',
                      border: session.isActive ? '1px solid rgba(99, 102, 241, 0.1)' : '1px solid transparent'
                    }}
                    onMouseOver={(e) => {
                      if (!session.isActive) {
                        e.currentTarget.style.background = '#fafafa';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (!session.isActive) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                  >
                    <div style={{
                      width: '28px',
                      height: '28px',
                      background: 'linear-gradient(135deg, #10b981, #06b6d4)',
                      borderRadius: '6px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '12px',
                      flexShrink: 0
                    }}>
                      {getSessionIcon(session.lastMessage)}
                    </div>
                    
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: '12px',
                        fontWeight: '500',
                        color: '#374151',
                        marginBottom: '2px',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {session.name || `Chat ${session.id.slice(-4)}`}
                      </div>
                      
                      <div style={{
                        fontSize: '10px',
                        color: '#6b7280',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {session.lastMessage ? 
                          (session.lastMessage.length > 30 ? 
                            session.lastMessage.substring(0, 30) + '...' : 
                            session.lastMessage
                          ) : 
                          'New conversation'
                        }
                      </div>
                      
                      <div style={{
                        fontSize: '9px',
                        color: '#d1d5db',
                        marginTop: '2px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}>
                        <Clock className="w-3 h-3" />
                        {formatTimeAgo(session.timestamp)}
                      </div>
                    </div>
                    
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeSession(session.id);
                        }}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          padding: '4px',
                          borderRadius: '4px',
                          transition: 'all 0.2s ease',
                          opacity: 0.5,
                          color: '#6b7280'
                        }}
                        onMouseOver={(e) => {
                          e.target.style.background = 'rgba(239, 68, 68, 0.1)';
                          e.target.style.opacity = '1';
                          e.target.style.color = '#ef4444';
                        }}
                        onMouseOut={(e) => {
                          e.target.style.background = 'transparent';
                          e.target.style.opacity = '0.5';
                          e.target.style.color = '#6b7280';
                        }}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div style={{
                  textAlign: 'center',
                  padding: '32px 16px',
                  color: '#6b7280'
                }}>
                  <div style={{
                    fontSize: '28px',
                    marginBottom: '12px',
                    opacity: 0.5
                  }}>
                    💬
                  </div>
                  <div style={{
                    fontSize: '13px',
                    fontWeight: '500',
                    marginBottom: '4px'
                  }}>
                    No recent chats
                  </div>
                  <div style={{
                    fontSize: '11px',
                    opacity: 0.8
                  }}>
                    Start a new conversation
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <style>{`
        @keyframes slideInFromRight {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        .spinning {
          animation: spin 2s linear infinite;
        }
        
        .w-3 { width: 12px; height: 12px; }
        .w-4 { width: 16px; height: 16px; }
        .w-5 { width: 20px; height: 20px; }
      `}</style>
    </>
  );
};

// Helper functions
const getSessionIcon = (text) => {
  if (!text) return '💬';
  
  const lowerText = text.toLowerCase();
  if (lowerText.includes('code') || lowerText.includes('program') || lowerText.includes('debug')) return '💻';
  if (lowerText.includes('design') || lowerText.includes('color') || lowerText.includes('ui')) return '🎨';
  if (lowerText.includes('plan') || lowerText.includes('project') || lowerText.includes('task')) return '📋';
  if (lowerText.includes('data') || lowerText.includes('analysis') || lowerText.includes('chart')) return '📊';
  if (lowerText.includes('write') || lowerText.includes('text') || lowerText.includes('content')) return '✍️';
  if (lowerText.includes('learn') || lowerText.includes('study') || lowerText.includes('research')) return '📚';
  if (lowerText.includes('idea') || lowerText.includes('creative') || lowerText.includes('brainstorm')) return '💡';
  if (lowerText.includes('help') || lowerText.includes('question') || lowerText.includes('how')) return '❓';
  if (lowerText.includes('medical') || lowerText.includes('health') || lowerText.includes('doctor')) return '🩺';
  
  return '💬';
};

const formatTimeAgo = (timestamp) => {
  if (!timestamp) return 'Just now';
  
  try {
    const now = new Date();
    const time = new Date(timestamp);
    const diffMs = now - time;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return time.toLocaleDateString();
  } catch (error) {
    return 'Recently';
  }
};

export default Sidebar;