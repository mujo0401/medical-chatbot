// src/hooks/useBackendConnection.js
import { useState, useEffect, useCallback } from 'react';
import { useApi } from './useApi';

export const useBackendConnection = () => {
  const [backendConnected, setBackendConnected] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);
  const { checkHealth } = useApi();

  const checkConnection = useCallback(async () => {
    try {
      console.log('🔄 Checking backend connection...');
      const data = await checkHealth();
      console.log('✅ Backend connected:', data);
      setBackendConnected(true);
      
      // Set default model status if not provided by backend
      if (!data.models) {
        setModelStatus({
          openai: { available: false, model: 'GPT-4 Turbo', configured: false },
          local: { available: true, model_name: 'DialoGPT-medium', trained: true },
          current_preference: 'local'
        });
      } else {
        setModelStatus(data.models);
      }
    } catch (error) {
      console.error('❌ Backend connection failed:', error.message);
      setBackendConnected(false);
      setModelStatus({
        openai: { available: false, model: 'GPT-4 Turbo', configured: false },
        local: { available: true, model_name: 'DialoGPT-medium', trained: true },
        current_preference: 'local'
      });
    }
  }, [checkHealth]);

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [checkConnection]);

  return { backendConnected, modelStatus, checkConnection };
};
