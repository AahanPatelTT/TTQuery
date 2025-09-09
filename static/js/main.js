/**
 * Synapse GUI - Main JavaScript Module
 */

class SynapseApp {
  constructor() {
    this.config = {};
    this.isLoading = false;
    this.currentSession = null;
    this.selectedFiles = [];
    
    // Initialize the app
    this.init();
  }

  async init() {
    try {
      await this.loadConfig();
      await this.loadSessions();
      await this.loadHistory();
      await this.loadKnowledgeBases();
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
    
    // Knowledge base management
    document.getElementById('refresh-kb-btn').addEventListener('click', () => this.loadKnowledgeBases());
    document.getElementById('select-all-kb-btn').addEventListener('click', () => this.selectAllKnowledgeBases());
    document.getElementById('deselect-all-kb-btn').addEventListener('click', () => this.deselectAllKnowledgeBases());
    document.getElementById('new-session-btn').addEventListener('click', () => this.newSession());
    document.getElementById('load-session-btn').addEventListener('click', () => this.loadSelectedSession());
    document.getElementById('save-as-default-btn').addEventListener('click', () => this.saveAsDefault());
    document.getElementById('reset-to-default-btn').addEventListener('click', () => this.resetToDefault());
    
    // Document upload functionality
    this.setupUploadHandlers();

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
      
      // Get selected knowledge bases
      const selectedKbs = this.getSelectedKnowledgeBases();
      
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question,
          images_enabled: imagesEnabled,
          max_images: maxImages,
          selected_knowledge_bases: selectedKbs
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
        const sourcesId = `sources-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        fullContent += `
          <div class="sources-container">
            <button class="sources-toggle" onclick="window.synapseApp.toggleSources('${sourcesId}')">
              <span class="sources-icon">📚</span>
              <span class="sources-text">Sources</span>
              <span class="sources-arrow">▼</span>
            </button>
            <div class="sources-content" id="${sourcesId}" style="display: none;">
              ${sourcesHtml}
            </div>
          </div>
        `;
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

  showWarning(message) {
    this.showNotification(message, 'warning');
  }

  showNotification(message, type = 'info') {
    // Create a sophisticated toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    // Create icon based on type
    let icon = '💬';
    switch (type) {
      case 'success':
        icon = '✅';
        break;
      case 'error':
        icon = '❌';
        break;
      case 'warning':
        icon = '⚠️';
        break;
    }
    
    toast.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 16px;">${icon}</span>
        <span style="flex: 1;">${message}</span>
      </div>
    `;

    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
      toast.style.opacity = '1';
      toast.style.transform = 'translateY(0)';
    });

    // Remove after delay
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(-20px)';
      setTimeout(() => {
        if (toast.parentNode) {
          toast.parentNode.removeChild(toast);
        }
      }, 300);
    }, 4000);
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Knowledge Base Management Methods
  async loadKnowledgeBases() {
    const loadingDiv = document.getElementById('kb-loading');
    const listDiv = document.getElementById('kb-list');
    const errorDiv = document.getElementById('kb-error');
    
    // Show loading state
    loadingDiv.style.display = 'block';
    listDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    
    try {
      const response = await fetch('/api/knowledge-bases');
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to load knowledge bases');
      }
      
      this.renderKnowledgeBases(data.knowledge_bases, data.current_kb_name);
      
      // Hide loading, show list
      loadingDiv.style.display = 'none';
      listDiv.style.display = 'block';
      
    } catch (error) {
      console.error('Failed to load knowledge bases:', error);
      loadingDiv.style.display = 'none';
      errorDiv.style.display = 'block';
      errorDiv.textContent = `Failed to load knowledge bases: ${error.message}`;
    }
  }
  
  renderKnowledgeBases(knowledgeBases, currentKbName) {
    const listDiv = document.getElementById('kb-list');
    
    if (!knowledgeBases || knowledgeBases.length === 0) {
      listDiv.innerHTML = `
        <div class="small" style="text-align: center; padding: 20px; color: #666;">
          No knowledge bases found. Run folder-based initialization first.
        </div>
      `;
      this.updateKbStatus();
      return;
    }
    
    listDiv.innerHTML = knowledgeBases.map(kb => {
      const isSelected = this.isKnowledgeBaseSelected(kb.name);
      const isCurrent = kb.name === currentKbName;
      const sizeInfo = kb.file_size > 0 ? `${(kb.file_size / (1024 * 1024)).toFixed(1)} MB` : 'unknown size';
      const chunkInfo = kb.chunk_count > 0 ? `${kb.chunk_count.toLocaleString()} chunks` : 'unknown chunks';
      
      return `
        <div class="kb-item" style="display: flex; align-items: flex-start; gap: 8px; padding: 8px; border: 1px solid #e1e5e9; border-radius: 4px; margin-bottom: 8px; ${isCurrent ? 'background-color: rgba(102, 126, 234, 0.1);' : ''}">
          <input 
            type="checkbox" 
            id="kb-${kb.name}" 
            data-kb-name="${kb.name}"
            ${isSelected ? 'checked' : ''}
            onchange="window.synapseApp.onKnowledgeBaseChange()"
            style="margin-top: 2px;"
          />
          <div style="flex: 1; min-width: 0;">
            <label for="kb-${kb.name}" style="font-weight: 500; cursor: pointer; display: block;">
              ${isCurrent ? '🔍 ' : ''}${this.escapeHtml(kb.display_name)}
              ${isCurrent ? '<span style="font-size: 11px; color: #666; font-weight: normal;"> (current)</span>' : ''}
            </label>
            <div style="font-size: 11px; color: #666; margin-top: 2px;">
              ${this.escapeHtml(kb.name)} • ${sizeInfo} • ${chunkInfo}
            </div>
            ${kb.description ? `<div style="font-size: 11px; color: #888; margin-top: 2px; font-style: italic;">${this.escapeHtml(kb.description)}</div>` : ''}
          </div>
        </div>
      `;
    }).join('');
    
    this.updateKbStatus();
  }
  
  isKnowledgeBaseSelected(kbName) {
    // Check if this knowledge base is currently selected
    const checkbox = document.getElementById(`kb-${kbName}`);
    if (checkbox) {
      return checkbox.checked;
    }
    
    // Default: select all knowledge bases on first load
    return true;
  }
  
  onKnowledgeBaseChange() {
    this.updateKbStatus();
  }
  
  updateKbStatus() {
    const statusDiv = document.getElementById('kb-status');
    const checkboxes = document.querySelectorAll('#kb-list input[type="checkbox"]');
    const selectedKbs = Array.from(checkboxes).filter(cb => cb.checked);
    
    if (selectedKbs.length === 0) {
      statusDiv.textContent = 'No knowledge bases selected - queries will use default';
      statusDiv.style.backgroundColor = 'rgba(220, 53, 69, 0.1)';
      statusDiv.style.color = '#721c24';
    } else if (selectedKbs.length === 1) {
      const kbName = selectedKbs[0].dataset.kbName;
      statusDiv.textContent = `Selected: ${kbName}`;
      statusDiv.style.backgroundColor = 'rgba(40, 167, 69, 0.1)';
      statusDiv.style.color = '#155724';
    } else {
      statusDiv.textContent = `Selected: ${selectedKbs.length} knowledge bases`;
      statusDiv.style.backgroundColor = 'rgba(102, 126, 234, 0.1)';
      statusDiv.style.color = '#1a1e21';
    }
  }
  
  selectAllKnowledgeBases() {
    const checkboxes = document.querySelectorAll('#kb-list input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = true);
    this.updateKbStatus();
    this.showToast('All knowledge bases selected', 'success');
  }
  
  deselectAllKnowledgeBases() {
    const checkboxes = document.querySelectorAll('#kb-list input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = false);
    this.updateKbStatus();
    this.showToast('All knowledge bases deselected', 'info');
  }
  
  getSelectedKnowledgeBases() {
    const checkboxes = document.querySelectorAll('#kb-list input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.dataset.kbName);
  }

  // Document Upload Functionality
  setupUploadHandlers() {
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('file-upload-area');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadFolder = document.getElementById('upload-folder');
    const customFolder = document.getElementById('custom-folder');

    // Initialize upload folder dropdown
    this.populateUploadFolders();

    // File input change
    fileInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));

    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
      this.handleFileSelection(e.dataTransfer.files);
    });

    // Upload button
    uploadBtn.addEventListener('click', () => this.uploadFiles());

    // Folder selection
    uploadFolder.addEventListener('change', (e) => {
      if (e.target.value === 'custom') {
        customFolder.style.display = 'block';
        customFolder.focus();
      } else {
        customFolder.style.display = 'none';
      }
      this.updateUploadButton();
    });

    // Custom folder input
    customFolder.addEventListener('input', () => this.updateUploadButton());
  }

  async populateUploadFolders() {
    try {
      // Get existing knowledge bases
      const response = await fetch('/api/knowledge-bases');
      const data = await response.json();
      
      const uploadFolder = document.getElementById('upload-folder');
      
      // Clear existing options except first
      while (uploadFolder.children.length > 1) {
        uploadFolder.removeChild(uploadFolder.lastChild);
      }

      // Add existing knowledge bases
      if (data.knowledge_bases && data.knowledge_bases.length > 0) {
        data.knowledge_bases.forEach(kb => {
          const option = document.createElement('option');
          option.value = kb.name;
          option.textContent = kb.display_name;
          uploadFolder.appendChild(option);
        });
      }

      // Add common folder options
      const commonFolders = ['documents', 'research', 'technical', 'uploads'];
      commonFolders.forEach(folder => {
        if (!Array.from(uploadFolder.options).some(opt => opt.value === folder)) {
          const option = document.createElement('option');
          option.value = folder;
          option.textContent = folder;
          uploadFolder.appendChild(option);
        }
      });

      // Add custom option
      const customOption = document.createElement('option');
      customOption.value = 'custom';
      customOption.textContent = '+ Create New Knowledge Base';
      uploadFolder.appendChild(customOption);

    } catch (error) {
      console.error('Failed to load upload folders:', error);
    }
  }

  handleFileSelection(files) {
    const fileArray = Array.from(files);
    const validFiles = fileArray.filter(file => this.isValidFileType(file));
    
    if (validFiles.length === 0) {
      this.showUploadMessage('No valid files selected. Please select PDF, PPTX, DOCX, TXT, MD, CSV, or image files.', 'error');
      return;
    }

    if (validFiles.length !== fileArray.length) {
      this.showUploadMessage(`${fileArray.length - validFiles.length} files were filtered out due to unsupported file types.`, 'info');
    }

    this.selectedFiles = validFiles;
    this.displaySelectedFiles();
    this.updateUploadButton();
  }

  isValidFileType(file) {
    const validExtensions = ['.pdf', '.pptx', '.docx', '.txt', '.md', '.csv', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'];
    const fileName = file.name.toLowerCase();
    return validExtensions.some(ext => fileName.endsWith(ext));
  }

  displaySelectedFiles() {
    const fileList = document.getElementById('file-list');
    const uploadPrompt = document.querySelector('.upload-prompt');
    
    if (this.selectedFiles.length === 0) {
      fileList.style.display = 'none';
      uploadPrompt.style.display = 'block';
      return;
    }

    uploadPrompt.style.display = 'none';
    fileList.style.display = 'block';
    fileList.innerHTML = '';

    this.selectedFiles.forEach((file, index) => {
      const fileItem = document.createElement('div');
      fileItem.className = 'file-item';
      
      const fileInfo = document.createElement('div');
      fileInfo.className = 'file-info';
      
      const fileIcon = document.createElement('span');
      fileIcon.className = 'file-icon';
      fileIcon.textContent = this.getFileIcon(file.name);
      
      const fileName = document.createElement('span');
      fileName.className = 'file-name';
      fileName.textContent = file.name;
      
      const fileSize = document.createElement('span');
      fileSize.className = 'file-size';
      fileSize.textContent = this.formatFileSize(file.size);
      
      const removeBtn = document.createElement('button');
      removeBtn.className = 'file-remove';
      removeBtn.textContent = '✕';
      removeBtn.title = 'Remove file';
      removeBtn.addEventListener('click', () => this.removeFile(index));
      
      fileInfo.appendChild(fileIcon);
      fileInfo.appendChild(fileName);
      fileInfo.appendChild(fileSize);
      fileItem.appendChild(fileInfo);
      fileItem.appendChild(removeBtn);
      fileList.appendChild(fileItem);
    });
  }

  getFileIcon(fileName) {
    const ext = fileName.toLowerCase().split('.').pop();
    const icons = {
      'pdf': '📄',
      'pptx': '📊',
      'docx': '📝',
      'txt': '📃',
      'md': '📋',
      'csv': '📈',
      'png': '🖼️',
      'jpg': '🖼️',
      'jpeg': '🖼️',
      'tiff': '🖼️',
      'bmp': '🖼️'
    };
    return icons[ext] || '📎';
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  removeFile(index) {
    this.selectedFiles.splice(index, 1);
    this.displaySelectedFiles();
    this.updateUploadButton();
  }

  updateUploadButton() {
    const uploadBtn = document.getElementById('upload-btn');
    const uploadFolder = document.getElementById('upload-folder');
    const customFolder = document.getElementById('custom-folder');
    
    const hasFiles = this.selectedFiles && this.selectedFiles.length > 0;
    const hasFolder = uploadFolder.value && uploadFolder.value !== '' && 
                     (uploadFolder.value !== 'custom' || customFolder.value.trim() !== '');
    
    uploadBtn.disabled = !hasFiles || !hasFolder;
    
    if (hasFiles && hasFolder) {
      uploadBtn.textContent = `📤 Upload ${this.selectedFiles.length} file${this.selectedFiles.length > 1 ? 's' : ''}`;
    } else {
      uploadBtn.textContent = '📤 Upload & Process';
    }
  }

  async uploadFiles() {
    if (!this.selectedFiles || this.selectedFiles.length === 0) {
      this.showUploadMessage('No files selected', 'error');
      return;
    }

    const uploadFolder = document.getElementById('upload-folder');
    const customFolder = document.getElementById('custom-folder');
    
    let folderName = uploadFolder.value;
    if (folderName === 'custom') {
      folderName = customFolder.value.trim();
      if (!folderName) {
        this.showUploadMessage('Please enter a knowledge base name', 'error');
        return;
      }
    }

    try {
      this.showUploadProgress(true);
      this.updateProgressBar(0, 'Preparing upload...');

      const formData = new FormData();
      this.selectedFiles.forEach(file => {
        formData.append('files', file);
      });
      formData.append('folder', folderName);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const result = await response.json();
      
      this.updateProgressBar(100, 'Upload completed!');
      
      // Show detailed progress steps if auto-processing started
      if (result.auto_processing) {
        this.showProgressDetails(true);
        this.updateProgressStep('upload-status', 'Complete ✅');
        this.updateProgressStep('parsing-status', 'Starting...');
      }
      
      // Show detailed results
      if (result.duplicate_files && result.duplicate_files.length > 0) {
        result.duplicate_files.forEach(dup => {
          this.showUploadMessage(`Skipped duplicate: ${dup.filename} (${dup.reason})`, 'info');
        });
      }
      
      if (result.failed_files && result.failed_files.length > 0) {
        result.failed_files.forEach(failed => {
          this.showUploadMessage(`Failed: ${failed.filename} - ${failed.error}`, 'error');
        });
      }
      
      setTimeout(() => {
        this.showUploadProgress(false);
        this.clearUploadForm();
        this.showUploadMessage(result.message, 'success');
        
        // If auto-processing started, show progress updates
        if (result.auto_processing) {
          this.monitorUploadProgress(result.folder);
        }
        
        // Refresh knowledge bases to show the new one
        this.loadKnowledgeBases();
        this.populateUploadFolders();
      }, 1000);

    } catch (error) {
      this.showUploadProgress(false);
      this.showUploadMessage(`Upload failed: ${error.message}`, 'error');
      console.error('Upload error:', error);
    }
  }

  showUploadProgress(show) {
    const progressDiv = document.getElementById('upload-progress');
    progressDiv.style.display = show ? 'block' : 'none';
    
    if (!show) {
      this.showProgressDetails(false);
    }
  }

  updateProgressBar(percent, text) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    progressFill.style.width = `${percent}%`;
    progressText.textContent = text;
  }

  showProgressDetails(show) {
    const progressDetails = document.getElementById('progress-details');
    progressDetails.style.display = show ? 'block' : 'none';
  }

  updateProgressStep(stepId, status) {
    const stepElement = document.getElementById(stepId);
    if (stepElement) {
      stepElement.textContent = status;
    }
  }

  clearUploadForm() {
    this.selectedFiles = [];
    document.getElementById('file-input').value = '';
    document.getElementById('upload-folder').value = '';
    document.getElementById('custom-folder').value = '';
    document.getElementById('custom-folder').style.display = 'none';
    this.displaySelectedFiles();
    this.updateUploadButton();
  }

  showUploadMessage(message, type) {
    const messagesContainer = document.getElementById('upload-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `upload-message ${type}`;
    messageDiv.textContent = message;
    
    messagesContainer.appendChild(messageDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (messageDiv.parentNode) {
        messageDiv.parentNode.removeChild(messageDiv);
      }
    }, 5000);
    
    // Scroll to show the message
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Sources toggle functionality
  toggleSources(sourcesId) {
    const sourcesContent = document.getElementById(sourcesId);
    const toggleButton = sourcesContent.previousElementSibling;
    const arrow = toggleButton.querySelector('.sources-arrow');
    
    if (sourcesContent.style.display === 'none') {
      sourcesContent.style.display = 'block';
      arrow.textContent = '▲';
      toggleButton.classList.add('expanded');
    } else {
      sourcesContent.style.display = 'none';
      arrow.textContent = '▼';
      toggleButton.classList.remove('expanded');
    }
  }

  // Monitor upload progress
  async monitorUploadProgress(folderKey) {
    const maxChecks = 30; // Maximum number of progress checks (5 minutes)
    let checkCount = 0;
    
    const checkProgress = async () => {
      try {
        const response = await fetch(`/api/upload/progress/${folderKey}`);
        if (!response.ok) return;
        
        const progress = await response.json();
        
        // Show progress message and update detailed steps
        if (progress.status === 'processing') {
          const overallProgress = Math.round(progress.overall_progress);
          
          // Update detailed progress steps
          if (progress.parsing.progress >= 100) {
            this.updateProgressStep('parsing-status', 'Complete ✅');
          } else {
            this.updateProgressStep('parsing-status', `${Math.round(progress.parsing.progress)}%`);
          }
          
          if (progress.embedding.progress >= 100) {
            this.updateProgressStep('embedding-status', 'Complete ✅');
          } else if (progress.embedding.progress > 0) {
            this.updateProgressStep('embedding-status', `${Math.round(progress.embedding.progress)}% (${progress.embedding.completed}/${progress.embedding.total})`);
          } else {
            this.updateProgressStep('embedding-status', 'Waiting...');
          }
          
          this.showUploadMessage(
            `Processing ${folderKey}: ${overallProgress}% complete`, 
            'info'
          );
          
          checkCount++;
          if (checkCount < maxChecks) {
            // Check again in 10 seconds
            setTimeout(checkProgress, 10000);
          }
        } else if (progress.status === 'complete') {
          // Update all steps to complete
          this.updateProgressStep('parsing-status', 'Complete ✅');
          this.updateProgressStep('embedding-status', 'Complete ✅');
          
          this.showUploadMessage(`✅ Processing completed for ${folderKey}! Knowledge base updated.`, 'success');
          
          // Hide progress details after a delay
          setTimeout(() => {
            this.showProgressDetails(false);
          }, 3000);
          
          // Refresh knowledge bases to reflect the updates
          this.loadKnowledgeBases();
        }
        
      } catch (error) {
        console.error('Progress monitoring error:', error);
      }
    };
    
    // Start monitoring after a short delay
    setTimeout(checkProgress, 5000);
  }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.synapseApp = new SynapseApp();
});
