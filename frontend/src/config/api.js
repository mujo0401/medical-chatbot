// src/config/api.js
export const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000/api';  // Added /api here

export const API_ENDPOINTS = {
  // Chat routes
  CHAT: '/chat',
  
  // Session routes
  SESSIONS: '/sessions',
  SESSION_MESSAGES: (sessionId) => `/sessions/${sessionId}/messages`,
  
  // Training routes
  TRAIN: '/train',
  TRAINING_STATUS: '/training/status',
  TRAINING_HISTORY: '/training/history',
  
  // Upload routes
  UPLOAD: '/upload',
  DOCUMENTS: '/documents',
  DELETE_DOCUMENT: (docId) => `/documents/${docId}`,
  
  // Health check
  HEALTH: '/health',
  
  // Model routes (if they exist)
  MODELS_STATUS: '/models/status',
  MODELS_PREFERENCE: '/models/preference'
};