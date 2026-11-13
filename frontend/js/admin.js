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

      markValid(emailInput);
      markValid(passwordInput);

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
          if (response.status === 401) {
            markInvalid(emailInput, 'Invalid email or password');
            passwordInput.value = '';
            setButtonLoading(loginBtn, false);
            return;
          }

          throw new Error(data.detail || 'Login failed');
        }

        const meResponse = await fetch(`${API_BASE}/auth/me`, {
          headers: {
            'Authorization': `Bearer ${data.access_token}`
          }
        });

        if (!meResponse.ok) {
          throw new Error('Failed to verify user role');
        }

        const userData = await meResponse.json();

        if (userData.role !== 'admin') {
          throw new Error('Access denied. Administrator privileges required.');
        }

        setAuthData(data.access_token, userData, true);

        showToast('Welcome back!', 'success');

        setTimeout(() => {
          window.location.href = 'admin-dashboard.html';
        }, 500);

      } catch (error) {
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

      if (response.status === 401 || response.status === 403) {
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

    if (elements.totalUsers) {
      elements.totalUsers.textContent = data.total_users?.toLocaleString() || '—';
    }

    if (elements.totalPatients) {
      elements.totalPatients.textContent = data.total_patients?.toLocaleString() || '—';
    }

    if (elements.totalAdmins) {
      elements.totalAdmins.textContent = data.total_admins?.toLocaleString() || '—';
    }

    if (elements.activeUsers) {
      elements.activeUsers.textContent = data.active_users?.toLocaleString() || '—';
    }

    if (elements.inactiveUsers) {
      elements.inactiveUsers.textContent = data.inactive_users?.toLocaleString() || '—';
    }

    if (elements.lastUpdated) {
      elements.lastUpdated.textContent = formatRelativeTime(new Date().toISOString());
    }
  },

  showLoading(show) {
    const loading = document.getElementById('documentsLoading');

    if (loading) {
      loading.classList.toggle('hidden', !show);
    }
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
    if (this.eventsBound) return;

    this.eventsBound = true;

    const openUploadBtn = document.getElementById('openUploadModalBtn');
    const closeUploadBtn = document.getElementById('closeUploadModal');
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    const uploadForm = document.getElementById('uploadForm');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const removeFileBtn = document.getElementById('removeFileBtn');

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

    const closeDeleteBtn =
      document.getElementById('closeDeleteModal') ||
      document.getElementById('cancelDeleteBtn');

    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');

    if (closeDeleteBtn) {
      closeDeleteBtn.addEventListener('click', () => this.closeDeleteModal());
    }

    if (confirmDeleteBtn) {
      confirmDeleteBtn.addEventListener('click', () => this.confirmDelete());
    }

    // Ingest modal handlers
    const closeIngestBtn = document.getElementById('closeIngestModal');
    const cancelIngestBtn = document.getElementById('cancelIngestBtn');
    const confirmIngestBtn = document.getElementById('confirmIngestBtn');

    if (closeIngestBtn) {
      closeIngestBtn.addEventListener('click', () => this.closeIngestModal());
    }

    if (cancelIngestBtn) {
      cancelIngestBtn.addEventListener('click', () => this.closeIngestModal());
    }

    if (confirmIngestBtn) {
      confirmIngestBtn.addEventListener('click', () => this.confirmIngest());
    }

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

    document.querySelectorAll('.modal-overlay').forEach(overlay => {
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
          closeAllModals();
        }
      });
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeAllModals();
      }
    });
  },

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

    this.selectedFile = null;
  },

  handleFileSelect(file) {
    const dropZoneContent = document.getElementById('dropZoneContent');
    const dropZonePreview = document.getElementById('dropZonePreview');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');
    const uploadError = document.getElementById('uploadError');
    const submitBtn = document.getElementById('submitUploadBtn');

    const isPdf =
      file.type.includes('pdf') ||
      file.name.toLowerCase().endsWith('.pdf');

    const isTxt =
      file.type === 'text/plain' ||
      file.name.toLowerCase().endsWith('.txt');

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

    if (dropZoneContent) dropZoneContent.classList.add('hidden');
    if (dropZonePreview) dropZonePreview.classList.remove('hidden');
    if (previewFileName) previewFileName.textContent = file.name;
    if (previewFileSize) previewFileSize.textContent = formatFileSize(file.size);
    if (uploadError) uploadError.hidden = true;
    if (submitBtn) submitBtn.disabled = false;

    this.selectedFile = file;
  },

  clearFileSelection() {
    const dropZoneContent = document.getElementById('dropZoneContent');
    const dropZonePreview = document.getElementById('dropZonePreview');
    const fileInput = document.getElementById('fileInput');
    const submitBtn = document.getElementById('submitUploadBtn');
    const uploadError = document.getElementById('uploadError');

    this.selectedFile = null;

    if (fileInput) fileInput.value = '';
    if (dropZoneContent) dropZoneContent.classList.remove('hidden');
    if (dropZonePreview) dropZonePreview.classList.add('hidden');
    if (submitBtn) submitBtn.disabled = true;

    if (uploadError) {
      uploadError.hidden = true;
      uploadError.textContent = '';
    }
  },

  showUploadError(message) {
    const uploadError = document.getElementById('uploadError');

    if (uploadError) {
      uploadError.textContent = message;
      uploadError.hidden = false;
    } else {
      showToast(message, 'error');
    }
  },

  async handleUpload(e) {
    e.preventDefault();

    if (this.isUploading) return;

    if (!this.selectedFile) {
      this.showUploadError('Please select a file first');
      return;
    }

    const title = this.selectedFile.name;
    const description = '';

    const formData = new FormData();

    formData.append('file', this.selectedFile);
    formData.append('title', title);
    formData.append('description', description);

    const submitBtn = document.getElementById('submitUploadBtn');

    this.isUploading = true;
    setButtonLoading(submitBtn, true);

    const uploadProgress = document.getElementById('uploadProgress');

    if (uploadProgress) {
      uploadProgress.classList.remove('hidden');
    }

    try {
      const response = await fetch(`${API_BASE}/admin/documents/upload`, {
        method: 'POST',
        headers: getAdminAuthHeader(),
        body: formData
      });

      const data = await response.json().catch(() => ({}));

      if (response.status === 401 || response.status === 403) {
        AdminAuth.logout();
        return;
      }

      if (!response.ok) {
        throw new Error(data.detail || 'Document upload failed');
      }

      showToast('Document uploaded successfully!', 'success');

      this.closeUploadModal();

      await this.loadDocuments();

    } catch (error) {
      this.showUploadError(error.message);
    } finally {
      this.isUploading = false;
      setButtonLoading(submitBtn, false);

      if (uploadProgress) {
        uploadProgress.classList.add('hidden');
      }
    }
  },

  async loadDocuments() {
  this.showLoading(true);
  this.showError(false);

  try {
    const response = await fetch(
      `${API_BASE}/admin/documents?page=${this.currentPage}&page_size=${this.pageSize}`,
      {
        headers: getAdminAuthHeader()
      }
    );

    if (response.status === 401 || response.status === 403) {
      AdminAuth.logout();
      return;
    }

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || 'Failed to load documents');
    }

    const data = await response.json();

    this.totalDocuments = data.total || 0;
    this.totalPages = data.total_pages || 0;

    this.renderDocuments(data.documents || []);
    this.renderPagination();
    this.showContent((data.documents || []).length > 0);

    } catch (error) {
      console.error('Documents loading error:', error);
      this.showError(true, error.message);
    } finally {
      this.showLoading(false);
    }
  },

  renderDocuments(documents) {
  const tbody = document.getElementById('documentsTableBody');
  const emptyState = document.getElementById('documentsEmpty');

  if (!tbody) {
    console.error('documentsTableBody element not found');
    return;
  }

  tbody.innerHTML = '';

  if (!documents || documents.length === 0) {
    if (emptyState) {
      emptyState.classList.remove('hidden');
    }
    return;
  }

  if (emptyState) {
    emptyState.classList.add('hidden');
  }

  documents.forEach(doc => {
    const row = document.createElement('tr');

    row.dataset.documentId = doc.document_id;

    row.innerHTML = `
      <td>
        <div class="doc-name">
          <span class="name"
            title="${escapeHTML(doc.original_filename)}">
            ${escapeHTML(doc.original_filename)}
          </span>

          <span class="id">
            ${escapeHTML(doc.document_id)}
          </span>
        </div>
      </td>

      <td class="file-size">
        ${formatFileSize(doc.file_size)}
      </td>

      <td class="upload-date">
        ${formatDate(doc.upload_timestamp)}
      </td>

      <td>
        <span class="status-badge status-${escapeHTML(doc.processing_status || 'uploaded')}">
          ${escapeHTML(doc.processing_status || 'uploaded')}
        </span>
      </td>

      <td>
        <div class="action-buttons">
          <button
            type="button"
            class="action-btn action-ingest ${doc.processing_status === 'completed' ? 'hidden' : ''}"
            data-document-id="${escapeHTML(doc.document_id)}"
            data-document-name="${escapeHTML(doc.original_filename)}"
            ${doc.processing_status === 'processing' ? 'disabled' : ''}>
            ${doc.processing_status === 'completed' ? 'Ingested' : (doc.processing_status === 'processing' ? 'Processing...' : 'Ingest')}
          </button>
          <button
            type="button"
            class="action-btn action-delete"
            data-document-id="${escapeHTML(doc.document_id)}"
            data-document-name="${escapeHTML(doc.original_filename)}">
            Delete
          </button>
        </div>
      </td>
    `;

    tbody.appendChild(row);
  });

  tbody.querySelectorAll('.action-delete').forEach(btn => {
    btn.addEventListener('click', () => {
      this.openDeleteModal(
        btn.dataset.documentId,
        btn.dataset.documentName
      );
    });
  });

  tbody.querySelectorAll('.action-ingest').forEach(btn => {
    btn.addEventListener('click', () => {
      this.openIngestModal(
        btn.dataset.documentId,
        btn.dataset.documentName
      );
    });
  });
},

  showLoading(show) {
    const loading = document.getElementById('documentsLoading');
    const tableContainer = document.getElementById('documentsTableContainer');

    if (loading) {
      loading.classList.toggle('hidden', !show);
    }

    if (tableContainer && show) {
      tableContainer.classList.add('hidden');
    }
  },

  showContent(hasDocuments) {
    const tableContainer = document.getElementById(
      'documentsTableContainer'
    );

    const emptyState = document.getElementById('documentsEmpty');

    if (tableContainer) {
      tableContainer.classList.toggle('hidden', !hasDocuments);
    }

    if (emptyState) {
      emptyState.classList.toggle('hidden', hasDocuments);
    }
  },

  showError(show, message = 'Unable to load documents') {
    const errorEl = document.getElementById('documentsError');
    const messageEl = document.getElementById('documentsErrorMessage');

    if (errorEl) errorEl.classList.toggle('hidden', !show);
    if (messageEl) messageEl.textContent = message;
  },
    renderPagination() {
    const currentPageEl = document.getElementById('currentPage');
    const totalPagesEl = document.getElementById('totalPages');
    const totalDocumentsEl = document.getElementById('totalDocuments');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    const pagination = document.getElementById('documentsPagination');

    if (currentPageEl) {
      currentPageEl.textContent = this.currentPage;
    }

    if (totalPagesEl) {
      totalPagesEl.textContent = this.totalPages;
    }

    if (totalDocumentsEl) {
      totalDocumentsEl.textContent = this.totalDocuments;
    }

    if (prevBtn) {
      prevBtn.disabled = this.currentPage <= 1;
    }

    if (nextBtn) {
      nextBtn.disabled =
        this.totalPages === 0 || this.currentPage >= this.totalPages;
    }

    if (pagination) {
      pagination.classList.toggle('hidden', this.totalDocuments === 0);
    }
  },

  goToPage(page) {
    if (page < 1 || page > this.totalPages) return;

    this.currentPage = page;
    this.loadDocuments();
  },

  openDeleteModal(documentId) {
    if (!documentId) {
      showToast('Invalid document ID', 'error');
      return;
    }

    this.selectedDocumentId = documentId;

    const deleteModal = document.getElementById('deleteModal');
    const deleteDocumentName = document.getElementById('deleteDocumentName');

    if (deleteDocumentName) {
      deleteDocumentName.textContent = 'this document';
    }

    if (deleteModal) {
      openModal('deleteModal');
    } else {
      const confirmed = window.confirm(
        'Are you sure you want to delete this document?'
      );

      if (confirmed) {
        this.confirmDelete();
      }
    }
  },

  closeDeleteModal() {
    closeModal('deleteModal');
    this.selectedDocumentId = null;
  },

  async confirmDelete() {
    if (!this.selectedDocumentId) return;

    const deleteBtn = document.getElementById('confirmDeleteBtn');

    setButtonLoading(deleteBtn, true);

    try {
      const response = await fetch(
        `${API_BASE}/admin/documents/${encodeURIComponent(this.selectedDocumentId)}`,
        {
          method: 'DELETE',
          headers: getAdminAuthHeader()
        }
      );

      const data = await response.json().catch(() => ({}));

      if (response.status === 401 || response.status === 403) {
        AdminAuth.logout();
        return;
      }

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to delete document');
      }

      showToast('Document deleted successfully', 'success');

      this.closeDeleteModal();

      if (this.currentPage > 1 && this.totalDocuments - 1 <=
          (this.currentPage - 1) * this.pageSize) {
        this.currentPage--;
      }

      await this.loadDocuments();

    } catch (error) {
      showToast(error.message, 'error');
    } finally {
      setButtonLoading(deleteBtn, false);
    }
  },

  openIngestModal(documentId, documentName) {
    this.selectedDocumentId = documentId;

    const ingestModal = document.getElementById('ingestModal');
    const ingestDocName = document.getElementById('ingestModalSubtitle');

    if (ingestDocName) {
      ingestDocName.textContent = `Processing: ${escapeHTML(documentName)}`;
    }

    // Reset progress steps
    document.querySelectorAll('.progress-step').forEach(step => {
      step.classList.remove('active', 'completed', 'error');
    });

    // Hide error and confirm button, show cancel
    const ingestError = document.getElementById('ingestError');
    const confirmIngestBtn = document.getElementById('confirmIngestBtn');
    const cancelIngestBtn = document.getElementById('cancelIngestBtn');
    const closeIngestBtn = document.getElementById('closeIngestModal');

    if (ingestError) ingestError.hidden = true;
    if (confirmIngestBtn) confirmIngestBtn.hidden = true;
    if (cancelIngestBtn) cancelIngestBtn.hidden = false;
    if (closeIngestBtn) closeIngestBtn.hidden = true;

    if (ingestModal) {
      openModal('ingestModal');
    }
  },

  closeIngestModal() {
    closeModal('ingestModal');
    this.selectedDocumentId = null;
  },

  async confirmIngest() {
    if (!this.selectedDocumentId) return;

    const confirmBtn = document.getElementById('confirmIngestBtn');
    const cancelBtn = document.getElementById('cancelIngestBtn');
    const closeBtn = document.getElementById('closeIngestModal');
    const ingestError = document.getElementById('ingestError');

    setButtonLoading(confirmBtn, true);
    if (cancelBtn) cancelBtn.disabled = true;
    if (ingestError) ingestError.hidden = true;

    try {
      // Step 1: Loading document
      this.updateIngestStep('stepLoad', 'active');

      // Call ingestion endpoint
      const response = await fetch(
        `${API_BASE}/admin/documents/${encodeURIComponent(this.selectedDocumentId)}/ingest`,
        {
          method: 'POST',
          headers: getAdminAuthHeader()
        }
      );

      const data = await response.json().catch(() => ({}));

      if (response.status === 401 || response.status === 403) {
        AdminAuth.logout();
        return;
      }

      if (!response.ok) {
        throw new Error(data.detail || 'Ingestion failed');
      }

      // Update progress steps to completed
      this.updateIngestStep('stepLoad', 'completed');
      this.updateIngestStep('stepSplit', 'completed');
      this.updateIngestStep('stepEmbed', 'completed');
      this.updateIngestStep('stepStore', 'completed');

      showToast('Document ingested successfully! Chunks created: ' + (data.chunks_created || 0), 'success');

      // Show close button, hide cancel
      if (confirmBtn) confirmBtn.hidden = true;
      if (cancelBtn) cancelBtn.hidden = true;
      if (closeBtn) closeBtn.hidden = false;

      // Reload documents to show updated status
      await this.loadDocuments();

    } catch (error) {
      // Mark current step as error
      document.querySelectorAll('.progress-step.active').forEach(step => {
        step.classList.remove('active');
        step.classList.add('error');
      });

      if (ingestError) {
        ingestError.textContent = error.message;
        ingestError.hidden = false;
      }

      // Show close button, hide cancel
      if (confirmBtn) confirmBtn.hidden = true;
      if (cancelBtn) cancelBtn.hidden = true;
      if (closeBtn) closeBtn.hidden = false;

      showToast(error.message, 'error');
    } finally {
      setButtonLoading(confirmBtn, false);
      if (cancelBtn) cancelBtn.disabled = false;
    }
  },

  updateIngestStep(stepId, state) {
    const step = document.getElementById(stepId);
    if (step) {
      step.classList.remove('active', 'completed', 'error');
      step.classList.add(state);
    }
  },
};

// ============================================================================
// Common UI Helpers
// ============================================================================

function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => {
    const entities = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    };

    return entities[char];
  });
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 Bytes';

  const units = ['Bytes', 'KB', 'MB', 'GB'];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  );

  return `${(bytes / Math.pow(1024, index)).toFixed(2)} ${units[index]}`;
}

function formatDate(dateValue) {
  if (!dateValue) return '—';

  const date = new Date(dateValue);

  if (Number.isNaN(date.getTime())) return '—';

  return date.toLocaleDateString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });
}

function formatRelativeTime(dateValue) {
  if (!dateValue) return '—';

  const date = new Date(dateValue);
  if (Number.isNaN(date.getTime())) return '—';

  const diff = Date.now() - date.getTime();
  const seconds = Math.floor(diff / 1000);

  if (seconds < 60) return 'Just now';

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minutes ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hours ago`;

  const days = Math.floor(hours / 24);
  return `${days} days ago`;
}

function showToast(message, type = 'info') {
  let container = document.getElementById('toastContainer');

  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.setAttribute('role', 'status');
  toast.textContent = message;

  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.add('show');
  });

  setTimeout(() => {
    toast.classList.remove('show');

    setTimeout(() => {
      toast.remove();
    }, 300);
  }, 4000);
}

function setButtonLoading(button, loading) {
  if (!button) return;

  button.disabled = loading;

  if (loading) {
    if (!button.dataset.originalText) {
      button.dataset.originalText = button.textContent.trim();
    }

    button.textContent = 'Please wait...';
  } else if (button.dataset.originalText) {
    button.textContent = button.dataset.originalText;
    delete button.dataset.originalText;
  }
}

function markInvalid(input, message) {
  if (!input) return;

  input.classList.add('invalid');
  input.setAttribute('aria-invalid', 'true');

  const error = input.parentElement?.querySelector('.field-error');

  if (error) {
    error.textContent = message;
    error.classList.remove('hidden');
  }
}

function markValid(input) {
  if (!input) return;

  input.classList.remove('invalid');
  input.removeAttribute('aria-invalid');

  const error = input.parentElement?.querySelector('.field-error');

  if (error) {
    error.textContent = '';
    error.classList.add('hidden');
  }
}

function validateEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}
// ============================================================================
// Authentication and User Display Helpers
// ============================================================================

function updateUserDisplay() {
  const user = getStoredUser(ADMIN_USER_KEY);

  if (!user) return;

  const name =
    user.full_name ||
    user.name ||
    user.username ||
    user.email ||
    'Admin';

  const email = user.email || '';

  const nameElements = document.querySelectorAll(
    '#adminName, #userName, .admin-name, .user-name'
  );

  const emailElements = document.querySelectorAll(
    '#adminEmailDisplay, #userEmail, .admin-email, .user-email'
  );

  const avatarElements = document.querySelectorAll(
    '#adminAvatar, #userAvatar, .admin-avatar, .user-avatar'
  );

  nameElements.forEach(el => {
    el.textContent = name;
  });

  emailElements.forEach(el => {
    el.textContent = email;
  });

  avatarElements.forEach(el => {
    el.textContent = name.charAt(0).toUpperCase();
  });
}

// ============================================================================
// Sidebar Navigation
// ============================================================================

function initSidebar() {
  const sidebarToggle = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('active');

      if (overlay) {
        overlay.classList.toggle('active');
      }
    });
  }

  if (overlay && sidebar) {
    overlay.addEventListener('click', () => {
      sidebar.classList.remove('active');
      overlay.classList.remove('active');
    });
  }

  const currentPage = window.location.pathname.split('/').pop();

  document.querySelectorAll('.sidebar a, .nav-link').forEach(link => {
    const href = link.getAttribute('href');

    if (href && href.split('/').pop() === currentPage) {
      link.classList.add('active');
    }
  });
}

// ============================================================================
// Admin Profile Dropdown
// ============================================================================

function initUserMenu() {
  const userMenuBtn = document.getElementById('userMenuTrigger');
  const userDropdown = document.getElementById('userDropdown');
  const logoutBtn = document.getElementById('logoutBtn');

  if (userMenuBtn && userDropdown) {
    userMenuBtn.addEventListener('click', (e) => {
      e.stopPropagation();

      const isOpen = userDropdown.classList.toggle('show');

      userMenuBtn.setAttribute('aria-expanded', String(isOpen));
      userDropdown.setAttribute('aria-hidden', String(!isOpen));
    });

    document.addEventListener('click', (e) => {
      if (
        !userMenuBtn.contains(e.target) &&
        !userDropdown.contains(e.target)
      ) {
        userDropdown.classList.remove('show');
        userMenuBtn.setAttribute('aria-expanded', 'false');
        userDropdown.setAttribute('aria-hidden', 'true');
      }
    });
  }

  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => AdminAuth.logout());
  }
}

// ============================================================================
// Modal Helpers
// ============================================================================

function openModal(modalId) {
  const modal = document.getElementById(modalId);

  if (!modal) return;

  modal.classList.remove('hidden');
  modal.classList.add('active', 'show');

  modal.setAttribute('aria-hidden', 'false');

  document.body.classList.add('modal-open');
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);

  if (!modal) return;

  modal.classList.add('hidden');
  modal.classList.remove('active', 'show');

  modal.setAttribute('aria-hidden', 'true');

  if (!document.querySelector('.modal-overlay.active, .modal-overlay.show')) {
    document.body.classList.remove('modal-open');
  }
}

function closeAllModals() {
  document.querySelectorAll('.modal-overlay').forEach(modal => {
    modal.classList.add('hidden');
    modal.classList.remove('active', 'show');
    modal.setAttribute('aria-hidden', 'true');
  });

  document.body.classList.remove('modal-open');
}

// ============================================================================
// Application Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  const currentPage = window.location.pathname.split('/').pop();

  switch (currentPage) {
    case 'admin-login.html':
      AdminAuth.initLoginPage();
      break;

    case 'admin-dashboard.html':
      AdminDashboard.init();
      break;

    case 'admin-documents.html':
      AdminDocuments.init();
      break;

    default:
      // Initialize common admin UI on other admin pages
      if (document.body.classList.contains('admin-page')) {
        if (AdminAuth.requireAuth()) {
          updateUserDisplay();
          initSidebar();
          initUserMenu();
        }
      }
      break;
  }
});