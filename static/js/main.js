/**
 * Synapse GUI - Main JavaScript Module
 */

class SynapseApp {
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
      this.setupResizer();
      this.setupTooltips();
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
    document.getElementById('save-as-default-btn').addEventListener('click', () => this.saveAsDefault());
    document.getElementById('reset-to-default-btn').addEventListener('click', () => this.resetToDefault());

    // Image controls synchronization
    this.setupImageControls();

    // Config auto-save (debounced)
    this.setupConfigAutoSave();
  }

  setupImageControls() {
    const quickToggle = document.getElementById('quick-images-toggle');
    const sidebarToggle = document.getElementById('images-enabled');
    const maxImagesInput = document.getElementById('max-images');
    const imageCountDisplay = document.getElementById('image-count-display');

    // Sync the two image toggle checkboxes
    const syncToggles = (source, target) => {
      if (source && target) {
        source.addEventListener('change', () => {
          target.checked = source.checked;
          this.updateImageCountDisplay();
          this.saveConfig(false); // Auto-save
        });
      }
    };

    syncToggles(quickToggle, sidebarToggle);
    syncToggles(sidebarToggle, quickToggle);

    // Update display when max images changes
    if (maxImagesInput) {
      maxImagesInput.addEventListener('input', () => {
        this.updateImageCountDisplay();
      });
    }

    // Initial display update
    this.updateImageCountDisplay();
  }

  updateImageCountDisplay() {
    const imageCountDisplay = document.getElementById('image-count-display');
    const maxImagesInput = document.getElementById('max-images');
    const imagesEnabled = document.getElementById('images-enabled');
    
    if (imageCountDisplay && maxImagesInput) {
      const maxImages = parseInt(maxImagesInput.value) || 0;
      const enabled = imagesEnabled?.checked !== false;
      
      if (!enabled || maxImages === 0) {
        imageCountDisplay.textContent = 'Disabled';
        imageCountDisplay.style.color = '#999';
      } else {
        imageCountDisplay.textContent = `Max: ${maxImages}`;
        imageCountDisplay.style.color = '#666';
      }
    }
  }

  setupConfigAutoSave() {
    const configInputs = ['system-prompt', 'topk', 'per-doc', 'lambda-mmr', 'timeout', 'max-images'];
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

    // Verbose and images checkboxes
    const verboseCheckbox = document.getElementById('verbose');
    const imagesCheckbox = document.getElementById('images-enabled');
    
    if (verboseCheckbox) {
      verboseCheckbox.addEventListener('change', () => this.saveConfig(false));
    }
    
    if (imagesCheckbox) {
      imagesCheckbox.addEventListener('change', () => {
        this.updateImageCountDisplay();
        this.saveConfig(false);
      });
    }
  }

  setupResizer() {
    const resizeHandle = document.querySelector('.resize-handle');
    const sidebar = document.querySelector('.sidebar');
    
    if (!resizeHandle || !sidebar) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    resizeHandle.addEventListener('mousedown', (e) => {
      isResizing = true;
      startX = e.clientX;
      startWidth = parseInt(window.getComputedStyle(sidebar).width, 10);
      
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      
      e.preventDefault();
    });

    const handleMouseMove = (e) => {
      if (!isResizing) return;
      
      const deltaX = e.clientX - startX;
      const newWidth = Math.max(280, Math.min(600, startWidth + deltaX));
      
      sidebar.style.width = newWidth + 'px';
    };

    const handleMouseUp = () => {
      isResizing = false;
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      
      // Save the width to localStorage
      const currentWidth = parseInt(window.getComputedStyle(sidebar).width, 10);
      localStorage.setItem('synapse-sidebar-width', currentWidth);
    };

    // Restore saved width on load
    const savedWidth = localStorage.getItem('synapse-sidebar-width');
    if (savedWidth) {
      const width = Math.max(280, Math.min(600, parseInt(savedWidth, 10)));
      sidebar.style.width = width + 'px';
    }
  }

  setupTooltips() {
    // Fix tooltip z-index issues by ensuring they're always positioned correctly
    const helpIcons = document.querySelectorAll('.help-icon');
    
    helpIcons.forEach(icon => {
      const tooltip = icon.querySelector('.tooltip');
      if (!tooltip) return;

      icon.addEventListener('mouseenter', () => {
        // Force tooltip to be visible above everything
        tooltip.style.zIndex = '999999';
        
        // Check if tooltip would go off screen and adjust position
        const iconRect = icon.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        // If tooltip would go off the top of the screen, position it below the icon instead
        if (iconRect.top - tooltipRect.height < 10) {
          tooltip.style.bottom = 'auto';
          tooltip.style.top = '100%';
          tooltip.style.marginTop = '8px';
          tooltip.style.marginBottom = '0';
        } else {
          tooltip.style.bottom = '100%';
          tooltip.style.top = 'auto';
          tooltip.style.marginTop = '0';
          tooltip.style.marginBottom = '8px';
        }
      });

      icon.addEventListener('mouseleave', () => {
        // Reset positioning
        tooltip.style.bottom = '100%';
        tooltip.style.top = 'auto';
        tooltip.style.marginTop = '0';
        tooltip.style.marginBottom = '8px';
      });
    });
  }

  async loadConfig() {
    try {
      const response = await fetch('/api/config');
      this.config = await response.json();
      this.updateConfigUI();
      this.updateConfigStatus();
    } catch (error) {
      this.showError('Failed to load configuration', error);
    }
  }

  updateConfigStatus() {
    const statusElement = document.getElementById('config-status');
    if (statusElement) {
      statusElement.textContent = 'Config linked to current session';
      statusElement.style.background = 'rgba(102, 126, 234, 0.1)';
      statusElement.style.color = '#495057';
    }
  }

  updateConfigUI() {
    document.getElementById('system-prompt').value = this.config.system_prompt || '';
    document.getElementById('topk').value = this.config.topk || 10;
    document.getElementById('per-doc').value = this.config.per_doc || 8;
    document.getElementById('lambda-mmr').value = this.config.lambda_mmr || 0.8;
    document.getElementById('timeout').value = this.config.timeout || 60;
    document.getElementById('verbose').checked = !!this.config.verbose;
    
    // Image settings
    document.getElementById('max-images').value = this.config.max_images || 2;
    document.getElementById('images-enabled').checked = this.config.images_enabled !== false;
    document.getElementById('quick-images-toggle').checked = this.config.images_enabled !== false;
    
    // Update image count display
    this.updateImageCountDisplay();
  }

  async saveConfig(showNotification = true) {
    try {
      const config = {
        system_prompt: document.getElementById('system-prompt').value,
        topk: parseInt(document.getElementById('topk').value, 10),
        per_doc: parseInt(document.getElementById('per-doc').value, 10),
        lambda_mmr: parseFloat(document.getElementById('lambda-mmr').value),
        timeout: parseInt(document.getElementById('timeout').value, 10),
        verbose: document.getElementById('verbose').checked,
        max_images: parseInt(document.getElementById('max-images').value, 10),
        images_enabled: document.getElementById('images-enabled').checked
      };

      const response = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (!response.ok) throw new Error('Failed to save config');
      
      this.config = { ...this.config, ...config };
      this.updateConfigStatus();
      
      if (showNotification) {
        this.showSuccess('Session configuration saved');
      }
    } catch (error) {
      this.showError('Failed to save configuration', error);
    }
  }

  async saveAsDefault() {
    if (!confirm('Save current configuration as default for new sessions?')) return;

    try {
      const response = await fetch('/api/config/save-as-default', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const result = await response.json();
      
      if (!response.ok || !result.ok) {
        throw new Error(result.error || 'Failed to save as default');
      }
      
      this.showSuccess('⭐ Configuration saved as default!');
      
      // Update config status to show it's now the default
      const statusElement = document.getElementById('config-status');
      if (statusElement) {
        statusElement.textContent = 'Current config saved as default ⭐';
        statusElement.style.background = 'rgba(40, 167, 69, 0.1)';
        statusElement.style.color = '#155724';
      }
      
    } catch (error) {
      this.showError('Failed to save as default', error);
    }
  }

  async resetToDefault() {
    if (!confirm('Reset current session configuration to default values?')) return;

    try {
      const response = await fetch('/api/config/reset-to-default', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const result = await response.json();
      
      if (!response.ok || !result.ok) {
        throw new Error(result.error || 'Failed to reset to default');
      }
      
      // Reload config to reflect changes
      await this.loadConfig();
      
      this.showSuccess('🔄 Configuration reset to default');
      
    } catch (error) {
      this.showError('Failed to reset to default', error);
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
      
      // Load session-specific config when switching sessions
      await this.loadConfig();
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
      // Get current image settings
      const imagesEnabled = document.getElementById('quick-images-toggle').checked;
      const maxImages = parseInt(document.getElementById('max-images').value, 10) || 2;
      
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question,
          images_enabled: imagesEnabled,
          max_images: maxImages
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.ok === false) {
        throw new Error(data.error || 'Unknown error');
      }

      this.addMessage('assistant', data.answer || '', data.answer_html || '', data.sources_html || '', data.images || []);
      
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

  addMessage(role, text, html = '', sourcesHtml = '', images = []) {
    const chat = document.getElementById('chat');
    const messageDiv = document.createElement('div');
    messageDiv.className = `msg ${role === 'user' ? 'me' : 'bot'}`;
    
    if (role === 'user') {
      messageDiv.textContent = text;
    } else {
      let content;
      if (html && html.trim()) {
        // Use provided HTML
        content = html;
      } else if (text && text.trim()) {
        // Convert plain text to HTML with basic formatting
        content = this.formatPlainText(text);
      } else {
        content = '';
      }
      
      let fullContent = content;
      
      // Add images if present
      if (role === 'assistant' && images && images.length > 0) {
        let imagesHtml = '<div class="message-images">';
        images.forEach((imagePath, index) => {
          const imageUrl = `/api/image/${encodeURIComponent(imagePath)}`;
          const fileName = imagePath.split('/').pop() || `Image ${index + 1}`;
          imagesHtml += `
            <div class="image-container">
              <img src="${imageUrl}" alt="${fileName}" class="message-image" onclick="window.open('${imageUrl}', '_blank')" />
              <div class="image-caption">${fileName}</div>
            </div>
          `;
        });
        imagesHtml += '</div>';
        fullContent += imagesHtml;
      }
      
      if (sourcesHtml && sourcesHtml.trim()) {
        fullContent += `<div class="sources">${sourcesHtml}</div>`;
      }
      
      messageDiv.innerHTML = fullContent;
      
      // Convert asterisk-based lists to proper HTML lists
      if (role === 'assistant') {
        this.convertAsterisksToBullets(messageDiv);
      }
    }
    
    chat.appendChild(messageDiv);
    chat.scrollTop = chat.scrollHeight;
  }

  formatPlainText(text) {
    // Convert plain text to HTML with basic formatting
    return text
      .replace(/\n\n/g, '</p><p>')  // Double newlines become paragraphs
      .replace(/\n/g, '<br>')       // Single newlines become breaks
      .replace(/^/, '<p>')          // Start with paragraph
      .replace(/$/, '</p>');        // End with paragraph
  }

  convertAsterisksToBullets(element) {
    // Convert asterisk-based lists to proper HTML lists
    const paragraphs = element.querySelectorAll('p');
    let currentList = null;
    let elementsToRemove = [];

    paragraphs.forEach(p => {
      const text = p.textContent.trim();
      
      // Check if paragraph starts with asterisk
      if (text.startsWith('* ')) {
        // Remove asterisk and create list item
        const listItemText = text.substring(2);
        
        // Create or continue list
        if (!currentList) {
          currentList = document.createElement('ul');
          p.parentNode.insertBefore(currentList, p);
        }
        
        const listItem = document.createElement('li');
        listItem.innerHTML = listItemText;
        currentList.appendChild(listItem);
        
        elementsToRemove.push(p);
      } else {
        // Reset list when we encounter non-asterisk paragraph
        currentList = null;
      }
    });

    // Remove original asterisk paragraphs
    elementsToRemove.forEach(el => el.remove());
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
      a.download = 'synapse_session_export.json';
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
      this.showSuccess('New session created with default config');
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
      
      await this.loadHistory(); // This will also load the session-specific config
      await this.loadSessions();
      this.showSuccess(`Session "${filename}" loaded with its config`);
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
      z-index: 99999;
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
  window.synapseApp = new SynapseApp();
});
