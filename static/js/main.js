/**
 * TTQuery GUI - Main JavaScript Module
 */

class TTQueryApp {
  constructor() {
    this.config = {};
    this.isLoading = false;
    this.currentSession = null;
    
    // Initialize the app
    this.init();
  }

  async init() {
    try {
      await this.loadConfig();
      await this.loadSessions();
      await this.loadHistory();
      this.setupEventListeners();
      this.showStatus('Ready', 'online');
    } catch (error) {
      this.showError('Failed to initialize app', error);
    }
  }

  setupEventListeners() {
    // Question input
    const questionInput = document.getElementById('question-input');
    questionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendQuestion();
      }
    });

    // Auto-resize textarea
    questionInput.addEventListener('input', (e) => {
      e.target.style.height = 'auto';
      e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
    });

    // Form submissions
    document.getElementById('send-btn').addEventListener('click', () => this.sendQuestion());
    document.getElementById('save-config-btn').addEventListener('click', () => this.saveConfig());
    document.getElementById('clear-chat-btn').addEventListener('click', () => this.clearChat());
    document.getElementById('export-chat-btn').addEventListener('click', () => this.exportChat());
    document.getElementById('refresh-sessions-btn').addEventListener('click', () => this.loadSessions());
    document.getElementById('new-session-btn').addEventListener('click', () => this.newSession());
    document.getElementById('load-session-btn').addEventListener('click', () => this.loadSelectedSession());

    // Config auto-save (debounced)
    this.setupConfigAutoSave();
  }

  setupConfigAutoSave() {
    const configInputs = ['system-prompt', 'topk', 'per-doc', 'lambda-mmr', 'timeout'];
    let saveTimeout;

    configInputs.forEach(id => {
      const element = document.getElementById(id);
      if (element) {
        element.addEventListener('input', () => {
          clearTimeout(saveTimeout);
          saveTimeout = setTimeout(() => this.saveConfig(false), 1000);
        });
      }
    });

    // Verbose checkbox
    const verboseCheckbox = document.getElementById('verbose');
    if (verboseCheckbox) {
      verboseCheckbox.addEventListener('change', () => this.saveConfig(false));
    }
  }

  async loadConfig() {
    try {
      const response = await fetch('/api/config');
      this.config = await response.json();
      this.updateConfigUI();
    } catch (error) {
      this.showError('Failed to load configuration', error);
    }
  }

  updateConfigUI() {
    document.getElementById('system-prompt').value = this.config.system_prompt || '';
    document.getElementById('topk').value = this.config.topk || 10;
    document.getElementById('per-doc').value = this.config.per_doc || 8;
    document.getElementById('lambda-mmr').value = this.config.lambda_mmr || 0.8;
    document.getElementById('timeout').value = this.config.timeout || 60;
    document.getElementById('verbose').checked = !!this.config.verbose;
  }

  async saveConfig(showNotification = true) {
    try {
      const config = {
        system_prompt: document.getElementById('system-prompt').value,
        topk: parseInt(document.getElementById('topk').value, 10),
        per_doc: parseInt(document.getElementById('per-doc').value, 10),
        lambda_mmr: parseFloat(document.getElementById('lambda-mmr').value),
        timeout: parseInt(document.getElementById('timeout').value, 10),
        verbose: document.getElementById('verbose').checked
      };

      const response = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (!response.ok) throw new Error('Failed to save config');
      
      this.config = { ...this.config, ...config };
      
      if (showNotification) {
        this.showSuccess('Configuration saved');
      }
    } catch (error) {
      this.showError('Failed to save configuration', error);
    }
  }

  async loadHistory() {
    try {
      const response = await fetch('/api/history');
      const data = await response.json();
      
      this.clearChatUI();
      
      for (const exchange of (data.history || [])) {
        this.addMessage('user', exchange.q || '');
        this.addMessage('assistant', '', exchange.a_html || '', exchange.sources_html || '');
      }

      this.currentSession = data.session_file;
      this.updateSessionInfo();
    } catch (error) {
      this.showError('Failed to load chat history', error);
    }
  }

  async loadSessions() {
    try {
      const response = await fetch('/api/sessions');
      const data = await response.json();
      
      const select = document.getElementById('session-select');
      select.innerHTML = '';
      
      for (const session of (data.sessions || [])) {
        const option = document.createElement('option');
        option.value = session.name;
        option.textContent = `${session.name} (${session.exchanges} exchanges)`;
        select.appendChild(option);
      }

      this.currentSession = data.current;
      this.updateSessionInfo();
    } catch (error) {
      this.showError('Failed to load sessions', error);
    }
  }

  updateSessionInfo() {
    const info = document.getElementById('session-info');
    if (this.currentSession) {
      const filename = this.currentSession.split('/').pop() || 'Unknown';
      info.textContent = `Current: ${filename}`;
    } else {
      info.textContent = 'No active session';
    }
  }

  async sendQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    
    if (!question || this.isLoading) return;

    this.setLoading(true);
    this.addMessage('user', question);
    input.value = '';
    input.style.height = 'auto';

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.ok === false) {
        throw new Error(data.error || 'Unknown error');
      }

      this.addMessage('assistant', data.answer || '', data.answer_html || '', data.sources_html || '');
      
      if (data.retrieval_info && this.config.verbose) {
        console.log('Retrieval info:', data.retrieval_info);
      }

    } catch (error) {
      this.showError('Failed to get response', error);
      this.addMessage('assistant', `Error: ${error.message}`);
    } finally {
      this.setLoading(false);
    }
  }

  addMessage(role, text, html = '', sourcesHtml = '') {
    const chat = document.getElementById('chat');
    const messageDiv = document.createElement('div');
    messageDiv.className = `msg ${role === 'user' ? 'me' : 'bot'}`;
    
    if (role === 'user') {
      messageDiv.textContent = text;
    } else {
      const content = html || this.escapeHtml(text);
      let fullContent = content;
      
      if (sourcesHtml) {
        fullContent += `<div class="sources">${sourcesHtml}</div>`;
      }
      
      messageDiv.innerHTML = fullContent;
    }
    
    chat.appendChild(messageDiv);
    chat.scrollTop = chat.scrollHeight;
  }

  clearChatUI() {
    document.getElementById('chat').innerHTML = '';
  }

  async clearChat() {
    if (!confirm('Clear the current chat session?')) return;

    try {
      const response = await fetch('/api/clear', { method: 'POST' });
      if (!response.ok) throw new Error('Failed to clear chat');
      
      this.clearChatUI();
      this.showSuccess('Chat cleared');
    } catch (error) {
      this.showError('Failed to clear chat', error);
    }
  }

  async exportChat() {
    try {
      const response = await fetch('/api/export');
      if (!response.ok) throw new Error('Failed to export chat');
      
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ttquery_session_export.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      this.showSuccess('Chat exported');
    } catch (error) {
      this.showError('Failed to export chat', error);
    }
  }

  async newSession() {
    try {
      const response = await fetch('/api/session/new', { method: 'POST' });
      if (!response.ok) throw new Error('Failed to create new session');
      
      await this.loadSessions();
      await this.loadHistory();
      this.showSuccess('New session created');
    } catch (error) {
      this.showError('Failed to create new session', error);
    }
  }

  async loadSelectedSession() {
    const select = document.getElementById('session-select');
    const filename = select.value;
    
    if (!filename) {
      this.showError('Please select a session to load');
      return;
    }

    try {
      const response = await fetch('/api/session/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });

      if (!response.ok) throw new Error('Failed to load session');
      
      await this.loadHistory();
      await this.loadSessions();
      this.showSuccess(`Session "${filename}" loaded`);
    } catch (error) {
      this.showError('Failed to load session', error);
    }
  }

  setLoading(loading) {
    this.isLoading = loading;
    const sendBtn = document.getElementById('send-btn');
    const questionInput = document.getElementById('question-input');
    
    if (loading) {
      sendBtn.disabled = true;
      sendBtn.innerHTML = '<div class="spinner"></div>';
      questionInput.disabled = true;
    } else {
      sendBtn.disabled = false;
      sendBtn.innerHTML = 'Send';
      questionInput.disabled = false;
      questionInput.focus();
    }
  }

  showStatus(message, type = 'online') {
    const statusElement = document.getElementById('status');
    if (statusElement) {
      statusElement.innerHTML = `<span class="status-dot ${type}"></span>${message}`;
    }
  }

  showSuccess(message) {
    this.showNotification(message, 'success');
  }

  showError(message, error = null) {
    console.error(message, error);
    const fullMessage = error ? `${message}: ${error.message}` : message;
    this.showNotification(fullMessage, 'error');
  }

  showNotification(message, type = 'info') {
    // Create a simple toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 12px 16px;
      border-radius: 8px;
      color: white;
      font-weight: 500;
      z-index: 1000;
      opacity: 0;
      transform: translateY(-20px);
      transition: all 0.3s ease;
      max-width: 400px;
      word-wrap: break-word;
    `;

    switch (type) {
      case 'success':
        toast.style.background = 'var(--accent-success)';
        break;
      case 'error':
        toast.style.background = 'var(--accent-danger)';
        break;
      default:
        toast.style.background = 'var(--accent-primary)';
    }

    document.body.appendChild(toast);

    // Animate in
    setTimeout(() => {
      toast.style.opacity = '1';
      toast.style.transform = 'translateY(0)';
    }, 10);

    // Remove after delay
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(-20px)';
      setTimeout(() => {
        if (toast.parentNode) {
          toast.parentNode.removeChild(toast);
        }
      }, 300);
    }, 3000);
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.ttqueryApp = new TTQueryApp();
});
