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
  editingConversationId: null,
  deletingConversationId: null,

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

    // Rename modal buttons
    const cancelRenameBtn = document.getElementById('cancelRenameBtn');
    const confirmRenameBtn = document.getElementById('confirmRenameBtn');
    const renameInput = document.getElementById('renameInput');
    if (cancelRenameBtn) {
      cancelRenameBtn.addEventListener('click', () => this.closeRenameModal());
    }
    if (confirmRenameBtn) {
      confirmRenameBtn.addEventListener('click', () => this.confirmRename());
    }
    if (renameInput) {
      renameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          this.confirmRename();
        } else if (e.key === 'Escape') {
          this.closeRenameModal();
        }
      });
    }

    // Delete modal buttons
    const cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
    if (cancelDeleteBtn) {
      cancelDeleteBtn.addEventListener('click', () => this.closeDeleteModal());
    }
    if (confirmDeleteBtn) {
      confirmDeleteBtn.addEventListener('click', () => this.confirmDelete());
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

    const renameModal = document.getElementById('renameModal');
    if (renameModal) {
      renameModal.addEventListener('click', (e) => {
        if (e.target === renameModal) {
          this.closeRenameModal();
        }
      });
    }

    const deleteModal = document.getElementById('deleteModal');
    if (deleteModal) {
      deleteModal.addEventListener('click', (e) => {
        if (e.target === deleteModal) {
          this.closeDeleteModal();
        }
      });
    }

    // Escape key to close modals
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.closeNewChatModal();
        this.closeRenameModal();
        this.closeDeleteModal();
        this.closeAllConversationMenus();
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
        updatedAt: conv.last_message_at || conv.updated_at || conv.created_at || new Date().toISOString()
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

  async renameConversation(conversationId, newTitle) {
    try {
      const response = await fetch(`${API_BASE}/chat/conversations/${conversationId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          ...this.getAuthHeader()
        },
        body: JSON.stringify({ title: newTitle })
      });

      if (response.status === 401) {
        window.Auth.logout();
        return null;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to rename conversation');
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to rename conversation:', error);
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
        // Use assistant message timestamp from backend (UTC ISO string)
        this.addConversationToSidebar(response.conversation_id, message, response.assistant_message.timestamp);
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

  addConversationToSidebar(conversationId, firstMessage, timestamp) {
    const title = firstMessage.length > 50 ? firstMessage.substring(0, 50) + '...' : firstMessage;
    const preview = firstMessage.length > 100 ? firstMessage.substring(0, 100) + '...' : firstMessage;
    // Use backend timestamp if provided, otherwise fallback to current time
    const updatedAt = timestamp || new Date().toISOString();

    // Check if conversation already exists in list
    const exists = this.conversations.some(c => c.id === conversationId);
    if (!exists) {
      this.conversations.unshift({
        id: conversationId,
        title: title,
        preview: preview,
        updatedAt: updatedAt
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
      <div class="conversation-wrapper" data-conversation-id="${conv.id}">
        <button type="button" class="conversation-item ${conv.id === this.currentConversationId ? 'active' : ''}"
                data-conversation-id="${conv.id}"
                aria-label="Open conversation: ${this.escapeHtml(conv.title)}">
          <div class="conversation-avatar">${conv.title.charAt(0).toUpperCase()}</div>
          <div class="conversation-content">
            <div class="conversation-title">${this.escapeHtml(conv.title)}</div>
            <div class="conversation-preview">${this.escapeHtml(conv.preview || 'No messages yet')}</div>
          </div>
          <span class="conversation-time">${this.formatTime(conv.updatedAt)}</span>
        </button>
        <div class="conversation-menu">
          <button type="button" class="conversation-menu-trigger" aria-label="Conversation options" aria-expanded="false" aria-haspopup="true" data-conversation-id="${conv.id}">
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"></path></svg>
          </button>
          <div class="conversation-dropdown" role="menu" aria-hidden="true">
            <button type="button" class="dropdown-item" role="menuitem" data-action="rename" data-conversation-id="${conv.id}">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
              <span>Rename</span>
            </button>
            <button type="button" class="dropdown-item dropdown-item-danger" role="menuitem" data-action="delete" data-conversation-id="${conv.id}">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
              <span>Delete</span>
            </button>
          </div>
        </div>
      </div>
    `).join('');

    // Bind conversation clicks (only on the main button, not the menu)
    list.querySelectorAll('.conversation-item').forEach(item => {
      item.addEventListener('click', () => {
        const id = item.getAttribute('data-conversation-id');
        this.loadConversation(id);
      });
    });

    // Bind menu triggers
    list.querySelectorAll('.conversation-menu-trigger').forEach(trigger => {
      trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        this.toggleConversationMenu(trigger);
      });
      // Keyboard support: Enter/Space to open
      trigger.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          e.stopPropagation();
          this.toggleConversationMenu(trigger);
        }
      });
    });

    // Bind menu actions
    list.querySelectorAll('.conversation-dropdown .dropdown-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation();
        const action = item.getAttribute('data-action');
        const id = item.getAttribute('data-conversation-id');
        if (action === 'rename') {
          this.openRenameModal(id);
        } else if (action === 'delete') {
          this.openDeleteModal(id);
        }
        this.closeAllConversationMenus();
      });
    });

    // Close menus on outside click
    document.addEventListener('click', (e) => {
      const clickedMenu = e.target.closest('.conversation-menu');
      const clickedPortal = e.target.closest('.conversation-dropdown-portal');
      if (!clickedMenu && !clickedPortal) {
        this.closeAllConversationMenus();
      }
    });
  },

  toggleConversationMenu(trigger) {
    const dropdown = trigger.nextElementSibling;
    const isOpen = dropdown.getAttribute('aria-hidden') === 'false';

    if (isOpen) {
      this.closeConversationMenu(trigger, dropdown);
    } else {
      this.openConversationMenu(trigger, dropdown);
    }
  },

  openConversationMenu(trigger, dropdown) {
    // Close any other open menus first
    this.closeAllConversationMenus();

    // Clone dropdown for portal (to avoid sidebar overflow clipping)
    const portalDropdown = dropdown.cloneNode(true);
    portalDropdown.classList.add('conversation-dropdown-portal');
    portalDropdown.setAttribute('role', 'menu');
    portalDropdown.setAttribute('aria-hidden', 'false');

    // Set explicit width before appending to measure correctly
    portalDropdown.style.width = '165px';
    portalDropdown.style.minWidth = '160px';
    portalDropdown.style.maxWidth = '180px';

    // Append to body first to measure
    document.body.appendChild(portalDropdown);

    const triggerRect = trigger.getBoundingClientRect();
    const dropdownRect = portalDropdown.getBoundingClientRect();
    portalDropdown.style.position = 'fixed';
    portalDropdown.style.top = `${triggerRect.bottom + 4}px`;
    // Align right edge of dropdown with right edge of trigger
    let left = triggerRect.right - dropdownRect.width;
    // Ensure it doesn't go off-screen left
    if (left < 8) {
      left = 8;
    }
    // Ensure it doesn't go off-screen right
    const viewportWidth = window.innerWidth;
    if (left + dropdownRect.width > viewportWidth - 8) {
      left = viewportWidth - dropdownRect.width - 8;
    }
    portalDropdown.style.left = `${left}px`;
    portalDropdown.style.zIndex = '2100';

    // Re-bind action events for the portal dropdown
    portalDropdown.querySelectorAll('.dropdown-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation();
        const action = item.getAttribute('data-action');
        const id = item.getAttribute('data-conversation-id');
        if (action === 'rename') {
          this.openRenameModal(id);
        } else if (action === 'delete') {
          this.openDeleteModal(id);
        }
        this.closeAllConversationMenus();
      });
    });

    // Store reference to portal dropdown on trigger for cleanup
    trigger._portalDropdown = portalDropdown;

    // Update ARIA on original trigger
    dropdown.setAttribute('aria-hidden', 'true');
    trigger.setAttribute('aria-expanded', 'true');

    // Focus first menu item for keyboard accessibility
    const firstItem = portalDropdown.querySelector('.dropdown-item');
    if (firstItem) {
      firstItem.focus();
    }

    // Handle keyboard navigation within dropdown
    portalDropdown.addEventListener('keydown', (e) => this.handleDropdownKeydown(e, portalDropdown, trigger));
  },

  closeConversationMenu(trigger, dropdown) {
    // Remove portal dropdown if exists
    if (trigger._portalDropdown) {
      trigger._portalDropdown.remove();
      trigger._portalDropdown = null;
    }
    dropdown.setAttribute('aria-hidden', 'true');
    trigger.setAttribute('aria-expanded', 'false');
  },

  closeAllConversationMenus() {
    document.querySelectorAll('.conversation-menu-trigger').forEach(trigger => {
      const dropdown = trigger.nextElementSibling;
      if (trigger._portalDropdown) {
        trigger._portalDropdown.remove();
        trigger._portalDropdown = null;
      }
      if (dropdown) {
        dropdown.setAttribute('aria-hidden', 'true');
      }
      trigger.setAttribute('aria-expanded', 'false');
    });
    // Also clean up any orphaned portal dropdowns
    document.querySelectorAll('.conversation-dropdown-portal').forEach(el => el.remove());
  },

  handleDropdownKeydown(e, portalDropdown, trigger) {
    const items = portalDropdown.querySelectorAll('.dropdown-item');
    const currentIndex = Array.from(items).findIndex(item => item === document.activeElement);

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const nextIndex = (currentIndex + 1) % items.length;
      items[nextIndex].focus();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prevIndex = (currentIndex - 1 + items.length) % items.length;
      items[prevIndex].focus();
    } else if (e.key === 'Escape' || e.key === 'Tab') {
      e.preventDefault();
      this.closeConversationMenu(trigger, trigger.nextElementSibling);
      trigger.focus();
    } else if (e.key === 'Enter' || e.key === ' ') {
      // Let the click handler handle Enter/Space on menu items
      if (e.target.classList.contains('dropdown-item')) {
        e.target.click();
      }
    }
  },

  openRenameModal(conversationId) {
    const conv = this.conversations.find(c => c.id === conversationId);
    if (!conv) return;

    this.editingConversationId = conversationId;

    const modal = document.getElementById('renameModal');
    const input = document.getElementById('renameInput');
    const errorEl = document.getElementById('renameError');

    if (modal && input) {
      input.value = conv.title;
      input.focus();
      input.select();
      if (errorEl) errorEl.textContent = '';
      modal.classList.add('active');
      modal.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    }
  },

  closeRenameModal() {
    const modal = document.getElementById('renameModal');
    if (modal) {
      modal.classList.remove('active');
      modal.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
    }
    this.editingConversationId = null;
  },

  async confirmRename() {
    const input = document.getElementById('renameInput');
    const errorEl = document.getElementById('renameError');
    const newTitle = input?.value?.trim();

    if (!newTitle) {
      if (errorEl) errorEl.textContent = 'Title cannot be empty';
      input?.focus();
      return;
    }

    if (newTitle.length > 200) {
      if (errorEl) errorEl.textContent = 'Title must be 200 characters or less';
      input?.focus();
      return;
    }

    try {
      const result = await this.renameConversation(this.editingConversationId, newTitle);
      if (result) {
        // Update local conversation list
        const idx = this.conversations.findIndex(c => c.id === this.editingConversationId);
        if (idx !== -1) {
          this.conversations[idx].title = result.title;
          this.renderConversations();
        }
        this.closeRenameModal();
      }
    } catch (error) {
      if (errorEl) errorEl.textContent = error.message;
    }
  },

  openDeleteModal(conversationId) {
    this.deletingConversationId = conversationId;
    const modal = document.getElementById('deleteModal');
    if (modal) {
      modal.classList.add('active');
      modal.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    }
  },

  closeDeleteModal() {
    const modal = document.getElementById('deleteModal');
    if (modal) {
      modal.classList.remove('active');
      modal.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
    }
    this.deletingConversationId = null;
  },

  async confirmDelete() {
    try {
      const success = await this.deleteConversation(this.deletingConversationId);
      if (success) {
        // Remove from local list
        this.conversations = this.conversations.filter(c => c.id !== this.deletingConversationId);
        this.renderConversations();

        // If deleted conversation was active, clear chat window
        if (this.currentConversationId === this.deletingConversationId) {
          this.currentConversationId = null;
          this.renderMessages([]);
          this.hideChatWindow();
          // Optionally load the first available conversation
          if (this.conversations.length > 0) {
            this.loadConversation(this.conversations[0].id);
          }
        }

        this.closeDeleteModal();
      }
    } catch (error) {
      console.error('Delete failed:', error);
      // Could show error in modal, for now just close
      this.closeDeleteModal();
    }
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
    if (!isoString) return 'Unknown time';
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return 'Invalid time';

    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;

    // Check if it's yesterday (calendar day comparison)
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    }

    // Older: use local calendar date format (e.g., "13 Nov 2026")
    return date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
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