// src/utils/helpers.js
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const generateSessionId = () => `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

export const generateContextualFallback = (userInput, errorMessage, backendConnected) => {
  if (errorMessage.includes('fetch') || errorMessage.includes('NetworkError') || !backendConnected) {
    return `I'm sorry, I'm having trouble connecting to my medical knowledge base right now. Please check if the backend server is running on http://localhost:5000, or try again in a moment.

**Your message:** "${userInput}"

**Temporary advice:** If this is urgent, please contact a healthcare professional directly.`;
  }
  return `I apologize, but I'm experiencing technical difficulties and cannot process your request properly right now.

**Your message:** "${userInput}"

**Error:** ${errorMessage}

**Please try:**
1. Make sure the backend server is running
2. Check your network connection
3. Try your question again in a moment

If this is a medical emergency, please contact emergency services or your healthcare provider directly.`;
};