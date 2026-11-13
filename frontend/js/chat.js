/**
 * MediTrust AI Health Assistant Chat Application
 * Pure Vanilla JavaScript (Zero external dependencies)
 * Backend-integrated version using FastAPI + MongoDB
 */

// API_BASE is defined in auth.js (loaded first)
const ChatApp = {
  currentConversationId: null,
  conversations: [],
  isLoading: false,
  currentUser: null,

  init() {
    console.log('[ChatApp] init() called');
    this.bindEvents();
    this.initUserMenu();
    this.verifyUserAndLoad();
  },

  async verifyUserAndLoad() {
    console.log('[ChatApp] verifyUserAndLoad called');
    try {
      const userData = await this.verifyTokenAndGetUser();
      console.log('[ChatApp] verifyTokenAndGetUser result:', userData);
      if (userData && (userData.role === 'patient' || userData.role === 'admin')) {
        this.currentUser = userData;
        this.updateUserDisplay(userData);
        this.loadConversations();
        this.setupWelcomeSuggestions();
      } else {
        console.log('[ChatApp] No valid user data, logging out');
        window.Auth.logout();
      }
    } catch (error) {
      console.error('[ChatApp] User verification failed:', error);
      window.Auth.logout();
    }
  },

  async verifyTokenAndGetUser() {
    console.log('[ChatApp] verifyTokenAndGetUser called');
    const tokenKey = window.getValidTokenKey?.();
    console.log('[ChatApp] tokenKey:', tokenKey);
    if (!tokenKey) {
      console.log('[ChatApp] No token key found');
      return null;
    }
    try {
      const authHeader = window.getAuthHeader?.(tokenKey);
      console.log('[ChatApp] authHeader for /auth/me:', authHeader);
      const response = await fetch(`${API_BASE}/auth/me`, {
        headers: authHeader
      });
      console.log('[ChatApp] /auth/me response status:', response.status);
      if (response.ok) {
        const data = await response.json();
        console.log('[ChatApp] /auth/me data:', data);
        return data;
      }
      return null;
    } catch (error) {
      console.error('[ChatApp] verifyTokenAndGetUser error:', error);
      return null;
    }
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

    // Chat input auto-resize and send button state
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');
    if (chatInput) {
      chatInput.addEventListener('input', () => {
        this.autoResizeTextarea(chatInput);
        this.updateSendButtonState(chatInput, sendBtn);
      });
      chatInput.addEventListener('keydown', (e) => this.handleKeydown(e));
    }
    // Initial send button state
    this.updateSendButtonState(chatInput, sendBtn);

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

  initUserMenu() {
    const userMenuTrigger = document.getElementById('userMenuTrigger');
    const userDropdown = document.getElementById('userDropdown');
    const logoutBtn = document.getElementById('logoutBtn');

    if (userMenuTrigger && userDropdown) {
      function toggleDropdown() {
        const isOpen = userDropdown.getAttribute('aria-hidden') === 'false';
        userDropdown.setAttribute('aria-hidden', isOpen ? 'true' : 'false');
        userMenuTrigger.setAttribute('aria-expanded', (!isOpen).toString());
      }

      userMenuTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleDropdown();
      });

      document.addEventListener('click', (e) => {
        if (!userMenuTrigger.contains(e.target) && !userDropdown.contains(e.target)) {
          userDropdown.setAttribute('aria-hidden', 'true');
          userMenuTrigger.setAttribute('aria-expanded', 'false');
        }
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          userDropdown.setAttribute('aria-hidden', 'true');
          userMenuTrigger.setAttribute('aria-expanded', 'false');
        }
      });
    }

    if (logoutBtn) {
      logoutBtn.addEventListener('click', () => {
        window.Auth.logout();
      });
    }
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
    // Only show loading if we don't have conversations yet
    if (this.conversations.length === 0) {
      this.showConversationsLoading();
    }
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
      // Only show error if we have no conversations to display
      if (this.conversations.length === 0) {
        this.showConversationsError();
      }
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
    console.log('[ChatApp] sendMessage called, conversationId:', conversationId);
    const authHeader = this.getAuthHeader();
    console.log('[ChatApp] sendMessage authHeader:', authHeader);
    
    try {
      const response = await fetch(`${API_BASE}/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeader
        },
        body: JSON.stringify({
          message,
          conversation_id: conversationId
        })
      });

      console.log('[ChatApp] sendMessage response status:', response.status);

      if (response.status === 401) {
        console.log('[ChatApp] 401 Unauthorized - logging out');
        window.Auth.logout();
        return null;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('[ChatApp] sendMessage error:', errorData);
        throw new Error(errorData.detail || 'Failed to send message');
      }

      const result = await response.json();
      console.log('[ChatApp] sendMessage success:', result);
      return result;
    } catch (error) {
      console.error('[ChatApp] sendMessage failed:', error);
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

      if (!response) {
        this.hideTypingIndicator();
        return; // Auth error handled in sendMessage (logout)
      }

      this.hideTypingIndicator();

      // Add assistant message
      this.addMessage('assistant', response.assistant_message.content);

      // Update conversation ID if new conversation was created
      const isNewConversation = !this.currentConversationId;
      if (isNewConversation) {
        this.currentConversationId = response.conversation_id;
        // Optimistically add new conversation to sidebar immediately
        this.addConversationToSidebar(response.conversation_id, message);
      }

      // Refresh conversations list to get updated preview/title from backend
      await this.loadConversations();
    } catch (error) {
      this.hideTypingIndicator();
      this.addMessage('assistant', `Error: ${error.message}`, true);
    } finally {
      this.setInputDisabled(false);
    }
  },

  addConversationToSidebar(conversationId, firstMessage) {
    const title = firstMessage.length > 50 ? firstMessage.substring(0, 50) + '...' : firstMessage;
    const preview = firstMessage.length > 100 ? firstMessage.substring(0, 100) + '...' : firstMessage;
    const now = new Date().toISOString();

    // Check if conversation already exists in list
    const exists = this.conversations.some(c => c.id === conversationId);
    if (!exists) {
      this.conversations.unshift({
        id: conversationId,
        title: title,
        preview: preview,
        updatedAt: now
      });
      this.renderConversations();
    }
  },

  async handleSendMessage(e) {
    console.log('[ChatApp] handleSendMessage called');
    e.preventDefault();

    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');

    console.log('[ChatApp] input value:', input?.value);
    console.log('[ChatApp] isLoading:', this.isLoading);

    if (!input || !input.value.trim() || this.isLoading) {
      console.log('[ChatApp] Early return - input missing, empty, or loading');
      return;
    }

    const message = input.value.trim();
    input.value = '';
    input.style.height = 'auto';
    this.updateSendButtonState(input, sendBtn);

    // Show chat window if this is a new conversation (welcome state)
    if (!this.currentConversationId) {
      this.showChatWindow();
    }

    // Add user message immediately
    this.addMessage('user', message);

    // Send to backend (handles loading state and button disabling)
    console.log('[ChatApp] Calling sendMessageToBackend');
    await this.sendMessageToBackend(message, this.currentConversationId);

    // Sync button state with input content (empty after send)
    this.updateSendButtonState(input, sendBtn);
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

  updateSendButtonState(textarea, sendBtn) {
    if (!textarea || !sendBtn) return;
    const hasContent = textarea.value.trim().length > 0;
    sendBtn.disabled = hasContent ? false : true;
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

  updateUserDisplay(user) {
    if (!user) return;
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