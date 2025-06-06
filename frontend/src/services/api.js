// src/services/api.js
import { API_BASE, API_ENDPOINTS } from '../config/api';

class ApiService {
  constructor() {
    // Bind all methods to preserve 'this' context
    this.request = this.request.bind(this);
    this.sendMessage = this.sendMessage.bind(this);
    this.getSessions = this.getSessions.bind(this);
    this.getSessionMessages = this.getSessionMessages.bind(this);
    this.startTraining = this.startTraining.bind(this);
    this.getTrainingStatus = this.getTrainingStatus.bind(this);
    this.getTrainingHistory = this.getTrainingHistory.bind(this);
    this.uploadFile = this.uploadFile.bind(this);
    this.getDocuments = this.getDocuments.bind(this);
    this.deleteDocument = this.deleteDocument.bind(this);
    this.checkHealth = this.checkHealth.bind(this);
    this.getModelStatus = this.getModelStatus.bind(this);
    this.setModelPreference = this.setModelPreference.bind(this);
  }

  async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server responded with ${response.status}: ${errorText}`);
      }

      // Handle different response types
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      return await response.text();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Health check (uses root URL since it's not under /api in Flask)
  async checkHealth() {
    const url = `${API_BASE.replace('/api', '')}${API_ENDPOINTS.HEALTH}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server responded with ${response.status}: ${errorText}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      return await response.text();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Chat API
  async sendMessage(message, sessionId) {
    return this.request(API_ENDPOINTS.CHAT, {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId }),
    });
  }

  // Session API
  async getSessions() {
    return this.request(API_ENDPOINTS.SESSIONS);
  }

  async getSessionMessages(sessionId) {
    return this.request(API_ENDPOINTS.SESSION_MESSAGES(sessionId));
  }

  // Training API
  async startTraining(documentIds, useAzure = false, computeTarget = 'Standard') {
    return this.request(API_ENDPOINTS.TRAIN, {
      method: 'POST',
      body: JSON.stringify({
        document_ids: documentIds,
        use_azure: useAzure,
        compute_target: computeTarget,
      }),
    });
  }

  async getTrainingStatus() {
    return this.request(API_ENDPOINTS.TRAINING_STATUS);
  }

  async getTrainingHistory() {
    return this.request(API_ENDPOINTS.TRAINING_HISTORY);
  }

  // Upload API
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.request(API_ENDPOINTS.UPLOAD, {
      method: 'POST',
      body: formData,
      headers: {}, // Remove Content-Type to let browser set it with boundary
    });
  }

  async getDocuments() {
    return this.request(API_ENDPOINTS.DOCUMENTS);
  }

  async deleteDocument(docId) {
    return this.request(API_ENDPOINTS.DELETE_DOCUMENT(docId), {
      method: 'DELETE',
    });
  }

  // Model API (if available)
  async getModelStatus() {
    return this.request(API_ENDPOINTS.MODELS_STATUS);
  }

  async setModelPreference(preference) {
    return this.request(API_ENDPOINTS.MODELS_PREFERENCE, {
      method: 'POST',
      body: JSON.stringify({ preference }),
    });
  }
}

export default new ApiService();