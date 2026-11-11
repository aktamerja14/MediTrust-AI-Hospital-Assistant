/**
 * MediTrust AI Health Assistant Chat Application
 * Pure Vanilla JavaScript (Zero external dependencies)
 * Frontend-only demo for Phase 2 - Backend integration in future phases
 */

const ChatApp = {
  currentConversationId: null,
  conversations: [],
  isLoading: false,

  init() {
    this.bindEvents();
    this.loadConversations();
    this.setupWelcomeSuggestions();
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

  loadConversations() {
    // Load from localStorage for demo purposes
    try {
      const stored = localStorage.getItem('meditrust_conversations');
      if (stored) {
        this.conversations = JSON.parse(stored);
        this.renderConversations();
      }
    } catch (e) {
      console.warn('Failed to load conversations:', e);
    }
  },

  saveConversations() {
    try {
      localStorage.setItem('meditrust_conversations', JSON.stringify(this.conversations));
    } catch (e) {
      console.warn('Failed to save conversations:', e);
    }
  },

  renderConversations() {
    const list = document.getElementById('conversationList');
    const empty = document.getElementById('conversationEmpty');

    if (!list) return;

    if (this.conversations.length === 0) {
      if (empty) empty.classList.remove('hidden');
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

  startNewConversation(initialMessage = null) {
    // Create new conversation
    const newConv = {
      id: 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
      title: initialMessage ? initialMessage.slice(0, 40) + (initialMessage.length > 40 ? '...' : '') : 'New Conversation',
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    this.conversations.unshift(newConv);
    this.currentConversationId = newConv.id;
    this.saveConversations();
    this.renderConversations();
    this.showChatWindow();

    if (initialMessage) {
      // Add user message
      this.addMessage('user', initialMessage);
      // Simulate AI response
      this.simulateAIResponse(initialMessage);
    }
  },

  loadConversation(id) {
    const conv = this.conversations.find(c => c.id === id);
    if (!conv) return;

    this.currentConversationId = id;
    this.showChatWindow();
    this.renderMessages(conv.messages);
    this.renderConversations(); // Update active state
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

    return `
      <div class="message ${isUser ? 'message-user' : 'message-assistant'}" data-message-id="${msg.id}">
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
          <div class="message-bubble ${msg.error ? 'message-error' : ''}">${this.escapeHtml(msg.content)}</div>
          <span class="message-time">${time}</span>
        </div>
      </div>
    `;
  },

  addMessage(role, content, error = false) {
    const conv = this.conversations.find(c => c.id === this.currentConversationId);
    if (!conv) return;

    const message = {
      id: 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
      role,
      content,
      timestamp: new Date().toISOString(),
      error
    };

    conv.messages.push(message);
    conv.updatedAt = message.timestamp;

    // Update preview
    if (role === 'user') {
      conv.preview = content.slice(0, 60) + (content.length > 60 ? '...' : '');
    }

    this.saveConversations();

    // Render the new message
    const container = document.getElementById('chatMessages');
    if (container) {
      container.insertAdjacentHTML('beforeend', this.createMessageHTML(message));
      this.scrollToBottom();
    }

    this.renderConversations();
  },

  scrollToBottom() {
    const container = document.getElementById('chatMessages');
    if (container) {
      container.scrollTop = container.scrollHeight;
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

    // Add user message
    this.addMessage('user', message);

    // Show typing indicator
    this.showTypingIndicator();

    // Simulate AI response (replace with actual API call in future)
    this.isLoading = true;
    try {
      await this.simulateAIResponse(message);
    } finally {
      this.isLoading = false;
      this.hideTypingIndicator();
      sendBtn.disabled = false;
      input.focus();
    }
  },

  simulateAIResponse(userMessage) {
    return new Promise(resolve => {
      // Simulate network delay
      setTimeout(() => {
        const responses = [
          "I understand you're asking about that. As an AI health assistant, I can provide general information, but please consult with a healthcare professional for personalized medical advice.",
          "That's a great question. For specific health concerns, I recommend scheduling an appointment with one of our specialists at MediTrust. Would you like me to help you find the right department?",
          "I can provide general health information on that topic. However, for accurate diagnosis and treatment planning, please consult with a qualified healthcare provider.",
          "Thank you for your question. Remember that this AI assistant provides informational support only and is not a substitute for professional medical consultation."
        ];

        const response = responses[Math.floor(Math.random() * responses.length)];
        this.addMessage('assistant', response);
        resolve();
      }, 1000 + Math.random() * 2000);
    });
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
  }
};

// Export for global access
window.ChatApp = ChatApp;