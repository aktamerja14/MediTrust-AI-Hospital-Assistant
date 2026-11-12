/**
 * MediTrust AI Health Assistant Chat Application
 * Pure Vanilla JavaScript (Zero external dependencies)
 * Backend-integrated version using FastAPI + MongoDB
 */

const API_BASE = 'http://127.0.0.1:8000/api/v1';

const ChatApp = {
  currentConversationId: null,
  conversations: [],
  isLoading: false,

  init() {
    this.bindEvents();
    this.loadConversations();
    this.setupWelcomeSuggestions();
    this.updateUserDisplay();
  },

  bindEvents() {
    // New chat button
    const newChatBtn = document.getElementById('newChatBtn');
    if (newChatBtn) {
      newChatBtn.addEventListener('click', () => this.openNewChatModal());
    }

    // New chat modal buttons
    const cancelNewChatBtn = document.getElementById('cancelNewChatBtn');
    const confirmNewChatBtn = document.getElementById('confirmNewChatBtn');
    if (cancelNewChatBtn) {
      cancelNewChatBtn.addEventListener('click', () => this.closeNewChatModal());
    }
    if (confirmNewChatBtn) {
      confirmNewChatBtn.addEventListener('click', () => this.confirmNewChat());
    }

    // Chat form submission
    const chatForm = document.getElementById('chatForm');
    if (chatForm) {
      chatForm.addEventListener('submit', (e) => this.handleSendMessage(e));
    }

    // Chat input auto-resize
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
      chatInput.addEventListener('input', () => this.autoResizeTextarea(chatInput));
      chatInput.addEventListener('keydown', (e) => this.handleKeydown(e));
    }

    // Close modal on overlay click
    const newChatModal = document.getElementById('newChatModal');
    if (newChatModal) {
      newChatModal.addEventListener('click', (e) => {
        if (e.target === newChatModal) {
          this.closeNewChatModal();
        }
      });
    }

    // Escape key to close modals
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.closeNewChatModal();
      }
    });
  },

  setupWelcomeSuggestions() {
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const suggestion = chip.getAttribute('data-suggestion');
        if (suggestion) {
          this.startNewConversation(suggestion);
        }
      });
    });
  },

  async loadConversations() {
    this.showConversationsLoading();
    try {
      const response = await fetch(`${API_BASE}/chat/conversations`, {
        headers: this.getAuthHeader()
      });

      if (response.status === 401) {
        // Token expired or invalid
        window.Auth.logout();
        return;
      }

      if (!response.ok) {
        throw new Error('Failed to load conversations');
      }

      const data = await response.json();
      this.conversations = (data.conversations || []).map(conv => ({
        id: conv.id,
        title: conv.title,
        preview: conv.preview,
        updatedAt: conv.updated_at
      }));
      this.renderConversations();
    } catch (error) {
      console.error('Failed to load conversations:', error);
      this.showConversationsError();
    }
  },

  async loadConversation(conversationId) {
    this.showMessagesLoading();
    try {
      const response = await fetch(`${API_BASE}/chat/conversations/${conversationId}`, {
        headers: this.getAuthHeader()
      });

      if (response.status === 401) {
        window.Auth.logout();
        return;
      }

      if (response.status === 404) {
        this.showMessagesError('Conversation not found');
        return;
      }

      if (!response.ok) {
        throw new Error('Failed to load conversation');
      }

      const data = await response.json();
      this.currentConversationId = data.id;
      this.renderMessages(data.messages || []);
      this.showChatWindow();
      this.renderConversations(); // Update active state
    } catch (error) {
      console.error('Failed to load conversation:', error);
      this.showMessagesError('Failed to load conversation');
    }
  },

  async sendMessage(message, conversationId = null) {
    try {
      const response = await fetch(`${API_BASE}/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...this.getAuthHeader()
        },
        body: JSON.stringify({
          message,
          conversation_id: conversationId
        })
      });

      if (response.status === 401) {
        window.Auth.logout();
        return null;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to send message');
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  },

  async deleteConversation(conversationId) {
    try {
      const response = await fetch(`${API_BASE}/chat/conversations/${conversationId}`, {
        method: 'DELETE',
        headers: this.getAuthHeader()
      });

      if (response.status === 401) {
        window.Auth.logout();
        return false;
      }

      if (!response.ok) {
        throw new Error('Failed to delete conversation');
      }

      return true;
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      throw error;
    }
  },

  startNewConversation(initialMessage = null) {
    this.currentConversationId = null;
    this.showChatWindow();
    this.renderMessages([]);
    this.renderConversations(); // Clear active state

    if (initialMessage) {
      this.addMessage('user', initialMessage);
      this.sendMessageToBackend(initialMessage, null);
    }
  },

  async sendMessageToBackend(message, conversationId) {
    this.showTypingIndicator();
    this.setInputDisabled(true);

    try {
      const response = await this.sendMessage(message, conversationId);

      if (!response) return; // Auth error handled in sendMessage

      this.hideTypingIndicator();

      // Add assistant message
      this.addMessage('assistant', response.assistant_message.content);

      // Update conversation ID if new conversation was created
      if (!this.currentConversationId) {
        this.currentConversationId = response.conversation_id;
      }

      // Refresh conversations list to get updated preview/title
      await this.loadConversations();
    } catch (error) {
      this.hideTypingIndicator();
      this.addMessage('assistant', `Error: ${error.message}`, true);
    } finally {
      this.setInputDisabled(false);
    }
  },

  async handleSendMessage(e) {
    e.preventDefault();

    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');

    if (!input || !input.value.trim() || this.isLoading) return;

    const message = input.value.trim();
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    // Add user message immediately
    this.addMessage('user', message);

    // Send to backend
    await this.sendMessageToBackend(message, this.currentConversationId);

    sendBtn.disabled = false;
    input.focus();
  },

  addMessage(role, content, error = false) {
    const message = {
      id: 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
      role,
      content,
      timestamp: new Date().toISOString(),
      error
    };

    const container = document.getElementById('chatMessages');
    if (container) {
      container.insertAdjacentHTML('beforeend', this.createMessageHTML(message));
      this.scrollToBottom();
    }
  },

  renderMessages(messages) {
    const container = document.getElementById('chatMessages');
    if (!container) return;

    if (!messages || messages.length === 0) {
      container.innerHTML = '';
      return;
    }

    container.innerHTML = messages.map(msg => this.createMessageHTML(msg)).join('');
    this.scrollToBottom();
  },

  createMessageHTML(msg) {
    const isUser = msg.role === 'user';
    const avatar = isUser ? 'U' : 'AI';
    const time = this.formatTime(msg.timestamp);
    const errorClass = msg.error ? 'message-error' : '';

    return `
      <div class="message ${isUser ? 'message-user' : 'message-assistant'} ${errorClass}" data-message-id="${msg.id}">
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
          <div class="message-bubble ${errorClass}">${this.escapeHtml(msg.content)}</div>
          <span class="message-time">${time}</span>
        </div>
      </div>
    `;
  },

  renderConversations() {
    const list = document.getElementById('conversationList');
    const empty = document.getElementById('conversationEmpty');

    if (!list) return;

    if (this.conversations.length === 0) {
      if (empty) empty.classList.remove('hidden');
      list.innerHTML = '';
      return;
    }

    if (empty) empty.classList.add('hidden');

    list.innerHTML = this.conversations.map(conv => `
      <button type="button" class="conversation-item ${conv.id === this.currentConversationId ? 'active' : ''}"
              data-conversation-id="${conv.id}">
        <div class="conversation-avatar">${conv.title.charAt(0).toUpperCase()}</div>
        <div class="conversation-content">
          <div class="conversation-title">${this.escapeHtml(conv.title)}</div>
          <div class="conversation-preview">${this.escapeHtml(conv.preview || 'No messages yet')}</div>
        </div>
        <span class="conversation-time">${this.formatTime(conv.updatedAt)}</span>
      </button>
    `).join('');

    // Bind conversation clicks
    list.querySelectorAll('.conversation-item').forEach(item => {
      item.addEventListener('click', () => {
        const id = item.getAttribute('data-conversation-id');
        this.loadConversation(id);
      });
    });
  },

  showConversationsLoading() {
    const list = document.getElementById('conversationList');
    const empty = document.getElementById('conversationEmpty');
    if (list) list.innerHTML = '';
    if (empty) empty.classList.add('hidden');
  },

  showConversationsError() {
    const list = document.getElementById('conversationList');
    const empty = document.getElementById('conversationEmpty');
    if (list) list.innerHTML = '';
    if (empty) {
      empty.classList.remove('hidden');
      empty.innerHTML = `
        <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
        <p>Failed to load conversations</p>
        <span>Check your connection and try again</span>
      `;
    }
  },

  showMessagesLoading() {
    const container = document.getElementById('chatMessages');
    if (container) {
      container.innerHTML = `
        <div class="message message-assistant loading-placeholder">
          <div class="message-avatar">AI</div>
          <div class="message-content">
            <div class="message-bubble typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      `;
    }
  },

  showMessagesError(message) {
    const container = document.getElementById('chatMessages');
    if (container) {
      container.innerHTML = `
        <div class="message message-assistant message-error">
          <div class="message-avatar">AI</div>
          <div class="message-content">
            <div class="message-bubble">${this.escapeHtml(message)}</div>
          </div>
        </div>
      `;
    }
  },

  showChatWindow() {
    const welcome = document.getElementById('chatWelcome');
    const messages = document.getElementById('chatMessages');
    const input = document.getElementById('chatInput');

    if (welcome) welcome.classList.add('hidden');
    if (messages) messages.classList.remove('hidden');
    if (input) input.focus();
  },

  hideChatWindow() {
    const welcome = document.getElementById('chatWelcome');
    const messages = document.getElementById('chatMessages');

    if (welcome) welcome.classList.remove('hidden');
    if (messages) messages.classList.add('hidden');
  },

  showTypingIndicator() {
    const container = document.getElementById('chatMessages');
    if (!container) return;

    const typing = document.createElement('div');
    typing.className = 'message message-assistant typing-indicator-container';
    typing.id = 'typingIndicator';
    typing.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-content">
        <div class="message-bubble typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    `;

    container.appendChild(typing);
    this.scrollToBottom();
  },

  hideTypingIndicator() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
  },

  setInputDisabled(disabled) {
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');
    if (input) input.disabled = disabled;
    if (sendBtn) sendBtn.disabled = disabled;
    this.isLoading = disabled;
  },

  handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const form = document.getElementById('chatForm');
      if (form) {
        form.dispatchEvent(new Event('submit'));
      }
    }
  },

  autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  },

  openNewChatModal() {
    const modal = document.getElementById('newChatModal');
    if (modal) {
      modal.classList.add('active');
      modal.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    }
  },

  closeNewChatModal() {
    const modal = document.getElementById('newChatModal');
    if (modal) {
      modal.classList.remove('active');
      modal.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
    }
  },

  confirmNewChat() {
    this.closeNewChatModal();
    this.startNewConversation();
  },

  updateUserDisplay() {
    // Update user avatar and name in header from auth
    const user = window.getStoredUserAny ? window.getStoredUserAny() : null;
    if (user) {
      const avatar = document.getElementById('userAvatar');
      const name = document.getElementById('userName');
      const dropdownAvatar = document.getElementById('dropdownAvatar');
      const dropdownName = document.getElementById('dropdownUserName');
      const dropdownRole = document.getElementById('dropdownUserRole');

      const initials = user.email ? user.email.charAt(0).toUpperCase() : 'U';
      const roleLabel = user.role ? user.role.charAt(0).toUpperCase() + user.role.slice(1) : 'User';

      if (avatar) avatar.textContent = initials;
      if (name) name.textContent = user.email || 'User';
      if (dropdownAvatar) dropdownAvatar.textContent = initials;
      if (dropdownName) dropdownName.textContent = user.email || 'User';
      if (dropdownRole) dropdownRole.textContent = roleLabel;
    }
  },

  scrollToBottom() {
    const container = document.getElementById('chatMessages');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  },

  formatTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  },

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  getAuthHeader() {
    // Use the auth module's function to get the correct token
    if (typeof window.getValidAuthHeader === 'function') {
      return window.getValidAuthHeader();
    }
    // Fallback
    const token = localStorage.getItem('meditrust_patient_token') || localStorage.getItem('meditrust_admin_token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
  }
};

// Export for global access
window.ChatApp = ChatApp;