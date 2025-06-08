// frontend/src/components/ModelSelector.jsx

import React, { useState, useEffect } from 'react';
import {
  Settings,
  Brain,
  Zap,
  Cloud,
  Cpu,
  Activity,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  ArrowRight,
  BarChart3
} from 'lucide-react';

const ModelSelector = ({ 
  backendUrl = 'http://localhost:5000',
  onModelChange = null,
  showAdvanced = false 
}) => {
  const [models, setModels] = useState({});
  const [currentModel, setCurrentModel] = useState('');
  const [modelHealth, setModelHealth] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [error, setError] = useState(null);
  const [testResults, setTestResults] = useState({});
  const [showComparison, setShowComparison] = useState(false);
  const [comparisonResults, setComparisonResults] = useState(null);

  // Model display information
  const modelInfo = {
    local_trained: {
      name: 'Local Trained',
      icon: <Cpu className="w-4 h-4" />,
      description: 'Fine-tuned model on your medical data',
      color: 'text-blue-500',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200'
    },
    eleuther: {
      name: 'EleutherAI GPT-Neo',
      icon: <Brain className="w-4 h-4" />,
      description: 'Large language model for general tasks',
      color: 'text-purple-500',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200'
    },
    openai: {
      name: 'OpenAI GPT',
      icon: <Cloud className="w-4 h-4" />,
      description: 'Advanced AI via OpenAI API',
      color: 'text-green-500',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200'
    },
    hybrid_local_eleuther: {
      name: 'Hybrid: Local + EleutherAI',
      icon: <Zap className="w-4 h-4" />,
      description: 'Combines local training with EleutherAI',
      color: 'text-orange-500',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200'
    },
    hybrid_all: {
      name: 'Hybrid: All Models',
      icon: <Activity className="w-4 h-4" />,
      description: 'Best of all available models',
      color: 'text-red-500',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200'
    }
  };

  // Fetch available models and current preference
  const fetchModels = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const [modelsRes, healthRes] = await Promise.all([
        fetch(`${backendUrl}/api/models/available`),
        fetch(`${backendUrl}/api/models/health`)
      ]);

      if (modelsRes.ok) {
        const modelsData = await modelsRes.json();
        setModels(modelsData.models || {});
        setCurrentModel(modelsData.current_preference || '');
      }

      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setModelHealth(healthData.health || {});
      }

    } catch (err) {
      setError(`Failed to fetch models: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Switch model preference
  const switchModel = async (modelName) => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${backendUrl}/api/models/preference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preference: modelName })
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentModel(modelName);
        if (onModelChange) {
          onModelChange(modelName);
        }
        
        // Show success message briefly
        setTimeout(() => {
          setIsOpen(false);
        }, 1000);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to switch model');
      }

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Test a specific model
  const testModel = async (modelName) => {
    try {
      setError(null);
      
      const response = await fetch(`${backendUrl}/api/models/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          model: modelName,
          message: 'Hello, this is a test message. Please respond briefly.'
        })
      });

      if (response.ok) {
        const data = await response.json();
        setTestResults(prev => ({
          ...prev,
          [modelName]: {
            success: true,
            response: data.response.reply,
            confidence: data.response.confidence || 'N/A',
            modelUsed: data.response.model_used
          }
        }));
      } else {
        const errorData = await response.json();
        setTestResults(prev => ({
          ...prev,
          [modelName]: {
            success: false,
            error: errorData.error
          }
        }));
      }

    } catch (err) {
      setTestResults(prev => ({
        ...prev,
        [modelName]: {
          success: false,
          error: err.message
        }
      }));
    }
  };

  // Compare models
  const compareModels = async () => {
    try {
      setError(null);
      
      const availableModels = Object.keys(models).filter(
        model => models[model]?.available && !model.startsWith('hybrid')
      );

      const response = await fetch(`${backendUrl}/api/models/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: 'What are the symptoms of diabetes?',
          models: availableModels
        })
      });

      if (response.ok) {
        const data = await response.json();
        setComparisonResults(data);
        setShowComparison(true);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to compare models');
      }

    } catch (err) {
      setError(err.message);
    }
  };

  // Get model status icon
  const getStatusIcon = (modelName) => {
    const health = modelHealth[modelName];
    const available = models[modelName]?.available;

    if (!available) {
      return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
    
    if (health?.healthy) {
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    }
    
    return <AlertCircle className="w-4 h-4 text-yellow-500" />;
  };

  // Initialize on mount
  useEffect(() => {
    fetchModels();
  }, []);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(fetchModels, 30000);
    return () => clearInterval(interval);
  }, []);

  const currentModelInfo = modelInfo[currentModel] || {
    name: 'Unknown Model',
    icon: <Settings className="w-4 h-4" />,
    description: 'Model information not available',
    color: 'text-gray-500'
  };

  return (
    <div className="relative">
      {/* Model Selector Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all
          ${currentModelInfo.bgColor} ${currentModelInfo.borderColor} ${currentModelInfo.color}
          hover:shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500
          ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        disabled={isLoading}
      >
        {currentModelInfo.icon}
        <span className="font-medium">{currentModelInfo.name}</span>
        <Settings className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-90' : ''}`} />
      </button>

      {/* Dropdown Panel */}
      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-96 bg-white rounded-lg shadow-xl border z-50 max-h-96 overflow-y-auto">
          {/* Header */}
          <div className="p-4 border-b bg-gray-50 rounded-t-lg">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-gray-800">Model Selection</h3>
              <div className="flex space-x-2">
                <button
                  onClick={compareModels}
                  className="p-1 text-gray-500 hover:text-blue-500 transition-colors"
                  title="Compare Models"
                >
                  <BarChart3 className="w-4 h-4" />
                </button>
                <button
                  onClick={fetchModels}
                  className={`p-1 text-gray-500 hover:text-blue-500 transition-colors ${isLoading ? 'animate-spin' : ''}`}
                  title="Refresh"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>
            {error && (
              <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
                {error}
              </div>
            )}
          </div>

          {/* Model List */}
          <div className="p-2">
            {Object.entries(models).map(([modelName, modelData]) => {
              const info = modelInfo[modelName] || modelInfo.local_trained;
              const isSelected = modelName === currentModel;
              const isAvailable = modelData?.available;
              const testResult = testResults[modelName];

              return (
                <div
                  key={modelName}
                  className={`
                    p-3 m-1 rounded-lg border transition-all cursor-pointer
                    ${isSelected 
                      ? `${info.bgColor} ${info.borderColor} border-2` 
                      : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                    }
                    ${!isAvailable ? 'opacity-50' : ''}
                  `}
                  onClick={() => isAvailable && !isLoading && switchModel(modelName)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded ${info.bgColor}`}>
                        {info.icon}
                      </div>
                      <div>
                        <div className="font-medium text-gray-800">{info.name}</div>
                        <div className="text-sm text-gray-600">{info.description}</div>
                        {!isAvailable && (
                          <div className="text-xs text-red-500 mt-1">
                            {modelData?.error || 'Not available'}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(modelName)}
                      {isSelected && (
                        <CheckCircle className="w-4 h-4 text-blue-500" />
                      )}
                      {isAvailable && showAdvanced && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            testModel(modelName);
                          }}
                          className="p-1 text-gray-400 hover:text-blue-500 transition-colors"
                          title="Test Model"
                        >
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Test Results */}
                  {testResult && (
                    <div className="mt-2 p-2 bg-white rounded border">
                      {testResult.success ? (
                        <div>
                          <div className="text-xs text-gray-600 mb-1">
                            Test Response (Confidence: {testResult.confidence}):
                          </div>
                          <div className="text-sm text-gray-800">
                            {testResult.response.slice(0, 100)}...
                          </div>
                        </div>
                      ) : (
                        <div className="text-xs text-red-600">
                          Test failed: {testResult.error}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Model Details */}
                  {showAdvanced && isAvailable && (
                    <div className="mt-2 text-xs text-gray-500">
                      <div>Health: {modelHealth[modelName]?.healthy ? 'Good' : 'Unknown'}</div>
                      {modelData.memory_usage && (
                        <div>Memory: {modelData.memory_usage}</div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Footer */}
          <div className="p-3 border-t bg-gray-50 rounded-b-lg">
            <div className="text-xs text-gray-600">
              Current: <span className="font-medium">{currentModelInfo.name}</span>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Modal */}
      {showComparison && comparisonResults && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-96 overflow-y-auto">
            <div className="p-4 border-b">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-800">Model Comparison</h3>
                <button
                  onClick={() => setShowComparison(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ×
                </button>
              </div>
              <div className="text-sm text-gray-600 mt-1">
                Question: {comparisonResults.message}
              </div>
            </div>
            
            <div className="p-4 space-y-4">
              {Object.entries(comparisonResults.responses || {}).map(([model, response]) => {
                const info = modelInfo[model] || modelInfo.local_trained;
                return (
                  <div key={model} className={`p-3 rounded-lg border ${info.borderColor} ${info.bgColor}`}>
                    <div className="flex items-center space-x-2 mb-2">
                      {info.icon}
                      <span className="font-medium">{info.name}</span>
                      {response.confidence && (
                        <span className="text-xs bg-white px-2 py-1 rounded">
                          Confidence: {response.confidence}
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-800">
                      {response.reply || response.error || 'No response'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;