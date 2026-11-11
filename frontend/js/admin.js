/**
 * MediTrust Admin Panel JavaScript
 * Pure Vanilla JavaScript (Zero external dependencies)
 * Features:
 * - Admin Dashboard with real-time stats
 * - Document Management (Upload, List, Delete)
 * - Responsive Sidebar Navigation
 * - Toast Notifications
 * - Modal Management
 *
 * Depends on auth.js for shared utilities and authentication
 */

// API Configuration
const API_BASE = 'http://127.0.0.1:8000/api/v1';
const ADMIN_TOKEN_KEY = 'meditrust_admin_token';
const ADMIN_USER_KEY = 'meditrust_admin_user';

function getAdminAuthHeader() {
  const token = localStorage.getItem(ADMIN_TOKEN_KEY);
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

// ============================================================================
// Admin Authentication Module
// ============================================================================

const AdminAuth = {
  initLoginPage() {
    this.checkExistingSession();
    this.bindLoginForm();
    this.bindPasswordToggle();
  },

  checkExistingSession() {
    if (isAuthenticated(ADMIN_TOKEN_KEY)) {
      const user = getStoredUser(ADMIN_USER_KEY);
      // Verify token is not expired by making a test request
      this.verifyToken().then(valid => {
        if (valid && user?.role === 'admin') {
          window.location.href = 'admin-dashboard.html';
} else {
        clearAuthData(true);
      }
      }).catch(() => {
        clearAuthData(true);
      });
    }
  },

  async verifyToken() {
    try {
      const response = await fetch(`${API_BASE}/auth/me`, {
        headers: getAdminAuthHeader()
      });
      return response.ok;
    } catch {
      return false;
    }
  },

  bindLoginForm() {
    const form = document.getElementById('adminLoginForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const emailInput = document.getElementById('adminEmail');
      const passwordInput = document.getElementById('adminPassword');
      const loginBtn = document.getElementById('loginBtn');

      // Clear previous errors
      markValid(emailInput);
      markValid(passwordInput);

      // Validation
      let isValid = true;

      if (!emailInput.value.trim() || !validateEmail(emailInput.value.trim())) {
        markInvalid(emailInput, 'Please enter a valid email address');
        isValid = false;
      }

      if (!passwordInput.value.trim()) {
        markInvalid(passwordInput, 'Please enter your password');
        isValid = false;
      }

      if (!isValid) {
        showToast('Please correct the highlighted fields', 'warning');
        return;
      }

      setButtonLoading(loginBtn, true);

      try {
        const response = await fetch(`${API_BASE}/auth/login`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            email: emailInput.value.trim(),
            password: passwordInput.value
          })
        });

        const data = await response.json();

        if (!response.ok) {
          // Handle 401 Unauthorized specifically for invalid credentials
          if (response.status === 401) {
            // Show error inline in the form (red error), keep email, clear password
            markInvalid(emailInput, 'Invalid email or password');
            passwordInput.value = '';
            setButtonLoading(loginBtn, false);
            return;
          }
          throw new Error(data.detail || 'Login failed');
        }

        // Verify user is admin
        const meResponse = await fetch(`${API_BASE}/auth/me`, {
          headers: { 'Authorization': `Bearer ${data.access_token}` }
        });

        if (!meResponse.ok) {
          throw new Error('Failed to verify user role');
        }

        const userData = await meResponse.json();

        if (userData.role !== 'admin') {
          throw new Error('Access denied. Administrator privileges required.');
        }

        // Store auth data using shared auth.js function
        setAuthData(data.access_token, userData, true);

        showToast('Welcome back!', 'success');

        // Redirect to dashboard
        setTimeout(() => {
          window.location.href = 'admin-dashboard.html';
        }, 500);

      } catch (error) {
        // Network errors shown as toast
        if (error instanceof TypeError && error.message.includes('fetch')) {
          showToast('Unable to connect to the server. Please try again.', 'error');
        } else {
          showToast(error.message, 'error');
        }
      } finally {
        setButtonLoading(loginBtn, false);
      }
    });
  },

  bindPasswordToggle() {
    const toggleBtn = document.querySelector('.toggle-password');
    const passwordInput = document.getElementById('adminPassword');

    if (toggleBtn && passwordInput) {
      toggleBtn.addEventListener('click', () => {
        const isPassword = passwordInput.type === 'password';
        passwordInput.type = isPassword ? 'text' : 'password';
        toggleBtn.setAttribute('aria-expanded', isPassword.toString());
      });
    }
  },

  logout() {
    clearAuthData(true);
    showToast('You have been signed out', 'info');
    setTimeout(() => {
      window.location.href = 'admin-login.html';
    }, 500);
  },

  requireAuth() {
    if (!isAuthenticated(ADMIN_TOKEN_KEY)) {
      window.location.href = 'admin-login.html';
      return false;
    }

    const user = getStoredUser(ADMIN_USER_KEY);
    if (user?.role !== 'admin') {
      this.logout();
      return false;
    }

    return true;
  }
};

// ============================================================================
// Admin Dashboard Module
// ============================================================================

const AdminDashboard = {
  eventsBound: false,

  init() {
    if (!AdminAuth.requireAuth()) return;

    updateUserDisplay();
    initSidebar();
    initUserMenu();
    this.bindEvents();
    this.loadDashboard();
  },

  bindEvents() {
    if (this.eventsBound) return;
    this.eventsBound = true;

    const refreshBtn = document.getElementById('refreshDashboardBtn');
    const retryBtn = document.getElementById('retryDashboardBtn');
    const refreshAction = document.getElementById('refreshStatsAction');

    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => this.loadDashboard());
    }

    if (retryBtn) {
      retryBtn.addEventListener('click', () => this.loadDashboard());
    }

    if (refreshAction) {
      refreshAction.addEventListener('click', () => this.loadDashboard());
    }
  },

  async loadDashboard() {
    this.showLoading(true);
    this.showError(false);

    try {
      const response = await fetch(`${API_BASE}/admin/dashboard`, {
        headers: getAdminAuthHeader()
      });

      if (response.status === 401) {
        AdminAuth.logout();
        return;
      }

      if (response.status === 403) {
        AdminAuth.logout();
        return;
      }

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to load dashboard');
      }

      const data = await response.json();
      this.renderStats(data);
      this.showStats(true);

    } catch (error) {
      this.showError(true, error.message);
    } finally {
      this.showLoading(false);
    }
  },

  renderStats(data) {
    const elements = {
      totalUsers: document.getElementById('statTotalUsers'),
      totalPatients: document.getElementById('statTotalPatients'),
      totalAdmins: document.getElementById('statTotalAdmins'),
      activeUsers: document.getElementById('statActiveUsers'),
      inactiveUsers: document.getElementById('statInactiveUsers'),
      lastUpdated: document.getElementById('statLastUpdated')
    };

    if (elements.totalUsers) elements.totalUsers.textContent = data.total_users?.toLocaleString() || '—';
    if (elements.totalPatients) elements.totalPatients.textContent = data.total_patients?.toLocaleString() || '—';
    if (elements.totalAdmins) elements.totalAdmins.textContent = data.total_admins?.toLocaleString() || '—';
    if (elements.activeUsers) elements.activeUsers.textContent = data.active_users?.toLocaleString() || '—';
    if (elements.inactiveUsers) elements.inactiveUsers.textContent = data.inactive_users?.toLocaleString() || '—';
    if (elements.lastUpdated) elements.lastUpdated.textContent = formatRelativeTime(new Date().toISOString());
  },

  showLoading(show) {
    const loading = document.getElementById('dashboardLoading');
    const statsGrid = document.getElementById('dashboardStatsGrid');
    if (loading) loading.classList.toggle('hidden', !show);
    if (statsGrid) statsGrid.classList.toggle('hidden', show);
  },

  showStats(show) {
    const statsGrid = document.getElementById('dashboardStatsGrid');
    if (statsGrid) statsGrid.classList.toggle('hidden', !show);
  },

  showError(show, message = 'Failed to load dashboard statistics. Please try again.') {
    const errorEl = document.getElementById('dashboardError');
    const messageEl = document.getElementById('dashboardErrorMessage');
    const statsGrid = document.getElementById('dashboardStatsGrid');

    if (errorEl) errorEl.classList.toggle('hidden', !show);
    if (messageEl) messageEl.textContent = message;
    if (statsGrid) statsGrid.classList.toggle('hidden', show);
  }
};

// ============================================================================
// Admin Documents Module
// ============================================================================

const AdminDocuments = {
  currentPage: 1,
  pageSize: 20,
  totalPages: 0,
  totalDocuments: 0,
  selectedDocumentId: null,
  selectedFile: null,
  isUploading: false,
  eventsBound: false,

  init() {
    if (!AdminAuth.requireAuth()) return;

    updateUserDisplay();
    initSidebar();
    initUserMenu();
    this.bindEvents();
    this.loadDocuments();
  },

  bindEvents() {
    // Prevent duplicate event listener registration
    if (this.eventsBound) {
      return;
    }
    this.eventsBound = true;

    // Upload modal
    const openUploadBtn = document.getElementById('openUploadModalBtn');
    const closeUploadBtn = document.getElementById('closeUploadModal');
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    const uploadForm = document.getElementById('uploadForm');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const submitUploadBtn = document.getElementById('submitUploadBtn');

    if (openUploadBtn) {
      openUploadBtn.addEventListener('click', () => this.openUploadModal());
    }

    if (closeUploadBtn) {
      closeUploadBtn.addEventListener('click', () => this.closeUploadModal());
    }

    if (cancelUploadBtn) {
      cancelUploadBtn.addEventListener('click', () => this.closeUploadModal());
    }

    if (uploadForm) {
      uploadForm.addEventListener('submit', (e) => this.handleUpload(e));
    }

    // Drop zone events
    if (dropZone && fileInput) {
      dropZone.addEventListener('click', () => fileInput.click());
      dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          fileInput.click();
        }
      });

      dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
      });

      dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
      });

      dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
          this.handleFileSelect(files[0]);
        }
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.handleFileSelect(e.target.files[0]);
        }
      });
    }

    if (removeFileBtn) {
      removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.clearFileSelection();
      });
    }

    // Delete modal
    const closeDeleteBtn = document.getElementById('closeDeleteModal') || document.getElementById('cancelDeleteBtn');
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');

    if (closeDeleteBtn) {
      closeDeleteBtn.addEventListener('click', () => this.closeDeleteModal());
    }

    if (confirmDeleteBtn) {
      confirmDeleteBtn.addEventListener('click', () => this.confirmDelete());
    }

    // Pagination
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    const retryBtn = document.getElementById('retryDocumentsBtn');
    const emptyUploadBtn = document.getElementById('emptyUploadBtn');

    if (prevBtn) {
      prevBtn.addEventListener('click', () => this.goToPage(this.currentPage - 1));
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', () => this.goToPage(this.currentPage + 1));
    }

    if (retryBtn) {
      retryBtn.addEventListener('click', () => this.loadDocuments());
    }

    if (emptyUploadBtn) {
      emptyUploadBtn.addEventListener('click', () => this.openUploadModal());
    }

    // Close modals on overlay click
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
          closeAllModals();
        }
      });
    });

    // Escape key to close modals
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeAllModals();
      }
    });
  },

  // Upload Modal
  openUploadModal() {
    this.resetUploadForm();
    openModal('uploadModal');
  },

  closeUploadModal() {
    closeModal('uploadModal');
    this.resetUploadForm();
  },

  resetUploadForm() {
    const form = document.getElementById('uploadForm');
    const dropZone = document.getElementById('dropZone');
    const dropZoneContent = document.getElementById('dropZoneContent');
    const dropZonePreview = document.getElementById('dropZonePreview');
    const fileInput = document.getElementById('fileInput');
    const uploadError = document.getElementById('uploadError');
    const uploadProgress = document.getElementById('uploadProgress');
    const submitBtn = document.getElementById('submitUploadBtn');

    if (form) form.reset();
    if (dropZone) dropZone.classList.remove('drag-over');
    if (dropZoneContent) dropZoneContent.classList.remove('hidden');
    if (dropZonePreview) dropZonePreview.classList.add('hidden');
    if (fileInput) fileInput.value = '';
    if (uploadError) {
      uploadError.hidden = true;
      uploadError.textContent = '';
    }
    if (uploadProgress) uploadProgress.classList.add('hidden');
    if (submitBtn) {
      submitBtn.disabled = true;
      setButtonLoading(submitBtn, false);
    }
  },

  handleFileSelect(file) {
    const dropZoneContent = document.getElementById('dropZoneContent');
    const dropZonePreview = document.getElementById('dropZonePreview');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');
    const uploadError = document.getElementById('uploadError');
    const submitBtn = document.getElementById('submitUploadBtn');

    // Validate file
    const isPdf = file.type.includes('pdf') || file.name.toLowerCase().endsWith('.pdf');
    const isTxt = file.type === 'text/plain' || file.name.toLowerCase().endsWith('.txt');

    if (!isPdf && !isTxt) {
      this.showUploadError('Please select a PDF or TXT file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      this.showUploadError('File size exceeds 10 MB limit');
      return;
    }

    if (file.size === 0) {
      this.showUploadError('File is empty');
      return;
    }

    // Show preview
    if (dropZoneContent) dropZoneContent.classList.add('hidden');
    if (dropZonePreview) dropZonePreview.classList.remove('hidden');
    if (previewFileName) previewFileName.textContent = file.name;
    if (previewFileSize) previewFileSize.textContent = formatFileSize(file.size);
    if (uploadError) uploadError.hidden = true;
    if (submitBtn) submitBtn.disabled = false;

    // Store file for upload
    this.selectedFile = file;
  },

  clearFileSelection() {
    this.selectedFile = null;
    this.resetUploadForm();
  },

  showUploadError(message) {
    const uploadError = document.getElementById('uploadError');
    const submitBtn = document.getElementById('submitUploadBtn');

    if (uploadError) {
      uploadError.textContent = message;
      uploadError.hidden = false;
    }
    if (submitBtn) submitBtn.disabled = true;
  },

  async handleUpload(e) {
    e.preventDefault();

    // Prevent duplicate uploads
    if (this.isUploading) {
      return;
    }

    if (!this.selectedFile) {
      this.showUploadError('Please select a file first');
      return;
    }

    this.isUploading = true;

    const submitBtn = document.getElementById('submitUploadBtn');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const progressPercent = document.getElementById('progressPercent');
    const uploadError = document.getElementById('uploadError');

    setButtonLoading(submitBtn, true);
    if (uploadProgress) uploadProgress.classList.remove('hidden');
    if (uploadError) uploadError.hidden = true;

    // Simulate progress for better UX (since we can't track actual upload progress with fetch easily)
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress > 90) progress = 90;
      if (progressFill) progressFill.style.width = `${progress}%`;
      if (progressPercent) progressPercent.textContent = `${Math.round(progress)}%`;
    }, 200);

    try {
      const formData = new FormData();
      formData.append('file', this.selectedFile);

      const response = await fetch(`${API_BASE}/admin/documents/upload`, {
        method: 'POST',
        headers: getAdminAuthHeader(),
        body: formData
      });

      clearInterval(progressInterval);

      if (response.status === 401) {
        AdminAuth.logout();
        return;
      }

      if (response.status === 403) {
        AdminAuth.logout();
        return;
      }

      if (progressFill) progressFill.style.width = '100%';
      if (progressPercent) progressPercent.textContent = '100%';
      if (progressText) progressText.textContent = 'Processing...';

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      showToast(`Document "${data.original_filename}" uploaded successfully`, 'success');
      this.closeUploadModal();
      this.loadDocuments();

    } catch (error) {
      clearInterval(progressInterval);
      this.showUploadError(error.message);
      if (uploadProgress) uploadProgress.classList.add('hidden');
    } finally {
      this.isUploading = false;
      setButtonLoading(submitBtn, false);
    }
  },

  // Load Documents
  async loadDocuments(page = 1) {
    this.currentPage = page;
    this.showLoading(true);
    this.showError(false);

    try {
      const response = await fetch(`${API_BASE}/admin/documents?page=${page}&page_size=${this.pageSize}`, {
        headers: getAdminAuthHeader()
      });

      if (response.status === 401) {
        AdminAuth.logout();
        return;
      }

      if (response.status === 403) {
        AdminAuth.logout();
        return;
      }

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to load documents');
      }

      const data = await response.json();
      this.renderDocuments(data.documents);
      this.updatePagination(data);
      this.showTable(true);

    } catch (error) {
      this.showError(true, error.message);
    } finally {
      this.showLoading(false);
    }
  },

  renderDocuments(documents) {
    const tbody = document.getElementById('documentsTableBody');
    const emptyState = document.getElementById('documentsEmpty');

    if (!tbody) return;

    if (!documents || documents.length === 0) {
      tbody.innerHTML = '';
      this.showTable(false);
      if (emptyState) emptyState.classList.remove('hidden');
      return;
    }

    if (emptyState) emptyState.classList.add('hidden');

    tbody.innerHTML = documents.map(doc => `
      <tr data-document-id="${doc.document_id}">
        <td>
          <div class="doc-name">
            <span class="name" title="${this.escapeHtml(doc.original_filename)}">${this.escapeHtml(doc.original_filename)}</span>
            <span class="id">${this.escapeHtml(doc.document_id)}</span>
          </div>
        </td>
        <td class="file-size">${formatFileSize(doc.file_size)}</td>
        <td class="upload-date">${formatDate(doc.upload_timestamp)}</td>
        <td>
          <span class="status-badge status-${doc.processing_status || 'uploaded'}">
            ${this.escapeHtml(doc.processing_status || 'uploaded')}
          </span>
        </td>
        <td>
          <button type="button" class="action-btn action-delete" aria-label="Delete ${this.escapeHtml(doc.original_filename)}" data-document-id="${doc.document_id}" data-document-name="${this.escapeHtml(doc.original_filename)}">
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v12M10 3h4a1 1 0 011 1v3H9V4a1 1 0 011-1z"></path></svg>
          </button>
        </td>
      </tr>
    `).join('');

    // Bind delete buttons
    tbody.querySelectorAll('.action-delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const docId = btn.dataset.documentId;
        const docName = btn.dataset.documentName;
        this.openDeleteModal(docId, docName);
      });
    });
  },

  updatePagination(data) {
    this.totalPages = data.total_pages || 0;
    this.totalDocuments = data.total || 0;

    const pagination = document.getElementById('pagination');
    const pagesContainer = document.getElementById('paginationPages');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    const paginationInfo = document.getElementById('paginationInfo');

    if (this.totalPages <= 1) {
      if (pagination) pagination.hidden = true;
      return;
    }

    if (pagination) pagination.hidden = false;

    // Update buttons
    if (prevBtn) prevBtn.disabled = this.currentPage <= 1;
    if (nextBtn) nextBtn.disabled = this.currentPage >= this.totalPages;

    // Generate page numbers
    if (pagesContainer) {
      const pages = this.getPageNumbers(this.currentPage, this.totalPages);
      pagesContainer.innerHTML = pages.map(page => {
        if (page === '...') {
          return '<span class="pagination-ellipsis">…</span>';
        }
        const isActive = page === this.currentPage;
        return `<button type="button" class="pagination-page ${isActive ? 'active' : ''}" data-page="${page}" ${isActive ? 'aria-current="page"' : ''}>${page}</button>`;
      }).join('');

      // Bind page buttons
      pagesContainer.querySelectorAll('.pagination-page').forEach(btn => {
        btn.addEventListener('click', () => {
          const page = parseInt(btn.dataset.page, 10);
          this.goToPage(page);
        });
      });
    }

    // Update info
    const start = (this.currentPage - 1) * this.pageSize + 1;
    const end = Math.min(this.currentPage * this.pageSize, this.totalDocuments);
    if (paginationInfo) {
      paginationInfo.textContent = `${start}-${end} of ${this.totalDocuments}`;
    }
  },

  getPageNumbers(current, total) {
    const pages = [];
    const maxVisible = 5;

    if (total <= maxVisible) {
      for (let i = 1; i <= total; i++) pages.push(i);
      return pages;
    }

    let start = Math.max(1, current - Math.floor(maxVisible / 2));
    let end = start + maxVisible - 1;

    if (end > total) {
      end = total;
      start = total - maxVisible + 1;
    }

    if (start > 1) {
      pages.push(1);
      if (start > 2) pages.push('...');
    }

    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    if (end < total) {
      if (end < total - 1) pages.push('...');
      pages.push(total);
    }

    return pages;
  },

  goToPage(page) {
    if (page < 1 || page > this.totalPages || page === this.currentPage) return;
    this.loadDocuments(page);
  },

  // Delete Modal
  openDeleteModal(documentId, documentName) {
    this.selectedDocumentId = documentId;
    const deleteDocName = document.getElementById('deleteDocName');
    const confirmBtn = document.getElementById('confirmDeleteBtn');

    if (deleteDocName) deleteDocName.textContent = documentName;
    if (confirmBtn) {
      confirmBtn.disabled = false;
      setButtonLoading(confirmBtn, false);
    }

    openModal('deleteModal');
  },

  closeDeleteModal() {
    this.selectedDocumentId = null;
    closeModal('deleteModal');
  },

  async confirmDelete() {
    if (!this.selectedDocumentId) return;

    const confirmBtn = document.getElementById('confirmDeleteBtn');
    setButtonLoading(confirmBtn, true);

    try {
      const response = await fetch(`${API_BASE}/admin/documents/${this.selectedDocumentId}`, {
        method: 'DELETE',
        headers: getAdminAuthHeader()
      });

      if (response.status === 401) {
        AdminAuth.logout();
        return;
      }

      if (response.status === 403) {
        AdminAuth.logout();
        return;
      }

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to delete document');
      }

      showToast('Document deleted successfully', 'success');
      this.closeDeleteModal();
      this.loadDocuments(this.currentPage);

    } catch (error) {
      showToast(error.message, 'error');
    } finally {
      setButtonLoading(confirmBtn, false);
    }
  },

  // UI State Helpers
  showLoading(show) {
    const loading = document.getElementById('documentsLoading');
    const tableContainer = document.getElementById('documentsTableContainer');
    const emptyState = document.getElementById('documentsEmpty');

    if (loading) loading.classList.toggle('hidden', !show);
    if (tableContainer) tableContainer.classList.toggle('hidden', show);
    if (emptyState) emptyState.classList.toggle('hidden', show);
  },

  showTable(show) {
    const tableContainer = document.getElementById('documentsTableContainer');
    const emptyState = document.getElementById('documentsEmpty');

    if (tableContainer) tableContainer.classList.toggle('hidden', !show);
    if (emptyState) emptyState.classList.toggle('hidden', show);
  },

  showError(show, message = 'Failed to load documents. Please try again.') {
    const errorEl = document.getElementById('documentsError');
    const messageEl = document.getElementById('documentsErrorMessage');
    const tableContainer = document.getElementById('documentsTableContainer');
    const emptyState = document.getElementById('documentsEmpty');

    if (errorEl) errorEl.classList.toggle('hidden', !show);
    if (messageEl) messageEl.textContent = message;
    if (tableContainer) tableContainer.classList.toggle('hidden', show);
    if (emptyState) emptyState.classList.toggle('hidden', show);
  },

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
};

// ============================================================================
// Initialize on DOM Ready
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Check which page we're on and initialize accordingly
  // Note: admin-login-page is handled by auth.js (LoginPage.init)
  if (document.body.classList.contains('admin-dashboard-page')) {
    AdminDashboard.init();
  } else if (document.body.classList.contains('admin-documents-page')) {
    AdminDocuments.init();
  }
});

// ============================================================================
// Admin UI Helper Functions
// ============================================================================

function updateUserDisplay() {
  const user = getStoredUser(ADMIN_USER_KEY);
  if (!user) return;

  const userAvatar = document.getElementById('userAvatar');
  const userName = document.getElementById('userName');
  const dropdownAvatar = document.getElementById('dropdownAvatar');
  const dropdownUserName = document.getElementById('dropdownUserName');
  const dropdownUserRole = document.getElementById('dropdownUserRole');

  const initials = user.email ? user.email.charAt(0).toUpperCase() : 'A';

  if (userAvatar) userAvatar.textContent = initials;
  if (userName) userName.textContent = user.email || 'Admin';
  if (dropdownAvatar) dropdownAvatar.textContent = initials;
  if (dropdownUserName) dropdownUserName.textContent = user.email || 'Admin User';
  if (dropdownUserRole) dropdownUserRole.textContent = user.role ? user.role.charAt(0).toUpperCase() + user.role.slice(1) : 'Administrator';
}

function initSidebar() {
  const sidebarToggle = document.getElementById('sidebarToggle');
  const adminSidebar = document.getElementById('adminSidebar');
  const sidebarOverlay = document.getElementById('sidebarOverlay');

  if (!sidebarToggle || !adminSidebar || !sidebarOverlay) return;

  function toggleSidebar() {
    const isOpen = adminSidebar.classList.toggle('open');
    sidebarOverlay.classList.toggle('active', isOpen);
    sidebarToggle.setAttribute('aria-expanded', isOpen.toString());
    document.body.style.overflow = isOpen ? 'hidden' : '';
  }

  function closeSidebar() {
    adminSidebar.classList.remove('open');
    sidebarOverlay.classList.remove('active');
    sidebarToggle.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  }

  sidebarToggle.addEventListener('click', toggleSidebar);
  sidebarOverlay.addEventListener('click', closeSidebar);

  // Close sidebar on navigation link click (mobile)
  document.querySelectorAll('.admin-nav-link').forEach(link => {
    link.addEventListener('click', () => {
      if (window.innerWidth < 992) {
        closeSidebar();
      }
    });
  });

  // Handle escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && adminSidebar.classList.contains('open')) {
      closeSidebar();
    }
  });
}

function initUserMenu() {
  const userMenuTrigger = document.getElementById('userMenuTrigger');
  const userDropdown = document.getElementById('userDropdown');
  const logoutBtn = document.getElementById('logoutBtn');

  if (!userMenuTrigger || !userDropdown) return;

  function toggleDropdown() {
    const isOpen = userDropdown.getAttribute('aria-hidden') === 'false';
    userDropdown.setAttribute('aria-hidden', isOpen ? 'true' : 'false');
    userMenuTrigger.setAttribute('aria-expanded', (!isOpen).toString());
  }

  function closeDropdown() {
    userDropdown.setAttribute('aria-hidden', 'true');
    userMenuTrigger.setAttribute('aria-expanded', 'false');
  }

  userMenuTrigger.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleDropdown();
  });

  document.addEventListener('click', (e) => {
    if (!userMenuTrigger.contains(e.target) && !userDropdown.contains(e.target)) {
      closeDropdown();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeDropdown();
    }
  });

  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      AdminAuth.logout();
    });
  }
}

// ============================================================================
// Export for global access (if needed)
// ============================================================================

window.AdminAuth = AdminAuth;
window.AdminDashboard = AdminDashboard;
window.AdminDocuments = AdminDocuments;