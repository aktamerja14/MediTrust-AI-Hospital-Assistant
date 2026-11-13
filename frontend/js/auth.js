/**
 * MediTrust Authentication Module
 * Pure Vanilla JavaScript (Zero external dependencies)
 * Handles patient & admin login, signup, token management, and protected page access
 * Real FastAPI backend integration
 */

// API Configuration
const API_BASE = 'http://127.0.0.1:8000/api/v1';
// Token keys for both patient and admin
const PATIENT_TOKEN_KEY = 'meditrust_patient_token';
const PATIENT_USER_KEY = 'meditrust_patient_user';
const ADMIN_TOKEN_KEY = 'meditrust_admin_token';
const ADMIN_USER_KEY = 'meditrust_admin_user';

// ============================================================================
// Utility Functions
// ============================================================================

function getAuthHeader(tokenKey = PATIENT_TOKEN_KEY) {
  const token = localStorage.getItem(tokenKey);
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

function setAuthData(token, user, isAdmin = false) {
  const tokenKey = isAdmin ? ADMIN_TOKEN_KEY : PATIENT_TOKEN_KEY;
  const userKey = isAdmin ? ADMIN_USER_KEY : PATIENT_USER_KEY;
  localStorage.setItem(tokenKey, token);
  localStorage.setItem(userKey, JSON.stringify(user));
}

function clearAuthData(isAdmin = false) {
  const tokenKey = isAdmin ? ADMIN_TOKEN_KEY : PATIENT_TOKEN_KEY;
  const userKey = isAdmin ? ADMIN_USER_KEY : PATIENT_USER_KEY;
  localStorage.removeItem(tokenKey);
  localStorage.removeItem(userKey);
}

function clearAllAuthData() {
  localStorage.removeItem(PATIENT_TOKEN_KEY);
  localStorage.removeItem(PATIENT_USER_KEY);
  localStorage.removeItem(ADMIN_TOKEN_KEY);
  localStorage.removeItem(ADMIN_USER_KEY);
}

function getStoredUser(userKey = PATIENT_USER_KEY) {
  try {
    return JSON.parse(localStorage.getItem(userKey) || 'null');
  } catch {
    return null;
  }
}

function isAuthenticated(tokenKey = PATIENT_TOKEN_KEY) {
  return !!localStorage.getItem(tokenKey);
}

function isAuthenticatedAny() {
  return isAuthenticated(PATIENT_TOKEN_KEY) || isAuthenticated(ADMIN_TOKEN_KEY);
}

function getStoredUserAny() {
  return getStoredUser(PATIENT_USER_KEY) || getStoredUser(ADMIN_USER_KEY);
}

function getAuthHeaderAny() {
  if (isAuthenticated(PATIENT_TOKEN_KEY)) {
    return getAuthHeader(PATIENT_TOKEN_KEY);
  }
  if (isAuthenticated(ADMIN_TOKEN_KEY)) {
    return getAuthHeader(ADMIN_TOKEN_KEY);
  }
  return {};
}

async function verifyToken(tokenKey = PATIENT_TOKEN_KEY) {
  try {
    const response = await fetch(`${API_BASE}/auth/me`, {
      headers: getAuthHeader(tokenKey)
    });
    if (response.ok) {
      return await response.json();
    }
    return null;
  } catch {
    return null;
  }
}

async function verifyTokenAny() {
  if (isAuthenticated(PATIENT_TOKEN_KEY)) {
    return verifyToken(PATIENT_TOKEN_KEY);
  }
  if (isAuthenticated(ADMIN_TOKEN_KEY)) {
    return verifyToken(ADMIN_TOKEN_KEY);
  }
  return null;
}

function getValidAuthHeader() {
  if (isAuthenticated(PATIENT_TOKEN_KEY)) {
    return getAuthHeader(PATIENT_TOKEN_KEY);
  }
  if (isAuthenticated(ADMIN_TOKEN_KEY)) {
    return getAuthHeader(ADMIN_TOKEN_KEY);
  }
  return {};
}

function getValidTokenKey() {
  if (isAuthenticated(PATIENT_TOKEN_KEY)) {
    return PATIENT_TOKEN_KEY;
  }
  if (isAuthenticated(ADMIN_TOKEN_KEY)) {
    return ADMIN_TOKEN_KEY;
  }
  return null;
}

function showToast(message, type = 'info') {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  let iconSvg = '';
  if (type === 'success') {
    iconSvg = '<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>';
  } else if (type === 'warning') {
    iconSvg = '<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>';
  } else if (type === 'error') {
    iconSvg = '<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>';
  } else {
    iconSvg = '<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
  }

  toast.innerHTML = `${iconSvg}<span>${message}</span>`;
  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.add('show');
  });

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => {
      if (toast.parentElement) {
        toast.remove();
      }
    }, 300);
  }, 4000);
}

function setButtonLoading(button, isLoading) {
  const btnText = button.querySelector('.btn-text');
  const btnLoading = button.querySelector('.btn-loading');

  if (isLoading) {
    button.disabled = true;
    if (btnText) btnText.classList.add('hidden');
    if (btnLoading) btnLoading.classList.remove('hidden');
  } else {
    button.disabled = false;
    if (btnText) btnText.classList.remove('hidden');
    if (btnLoading) btnLoading.classList.add('hidden');
  }
}

// ============================================================================
// Main Website Auth (index.html) - Profile Display & Dropdown
// ============================================================================

function initMainWebsiteAuth() {
  const authButtonsContainer = document.querySelector('.nav-actions');
  if (!authButtonsContainer) return;

  const user = getStoredUserAny();
  if (!user) {
    showAuthButtons(authButtonsContainer);
    bindModalAuthForms();
    return;
  }

  const tokenKey = getValidTokenKey();
  if (!tokenKey) {
    showAuthButtons(authButtonsContainer);
    bindModalAuthForms();
    return;
  }

  verifyToken(tokenKey).then(userData => {
    if (userData && (userData.role === 'patient' || userData.role === 'admin')) {
      showUserProfile(authButtonsContainer, userData);
    } else {
      clearAllAuthData();
      showAuthButtons(authButtonsContainer);
      bindModalAuthForms();
    }
  }).catch(() => {
    clearAllAuthData();
    showAuthButtons(authButtonsContainer);
    bindModalAuthForms();
  });
}

function showAuthButtons(container) {
  container.innerHTML = `
    <button type="button" class="btn btn-outline btn-sm btn-open-login" aria-label="Open Login Modal">Login</button>
    <button type="button" class="btn btn-primary btn-sm btn-open-signup" aria-label="Open Sign Up Modal">Sign Up</button>
    <button type="button" class="menu-toggle" id="mobileMenuToggle" aria-label="Toggle Navigation Menu" aria-expanded="false">
      <span></span>
      <span></span>
      <span></span>
    </button>
  `;
  initMobileMenu();
  initAuthModals();
  bindModalAuthForms();
}

function bindModalAuthForms() {
  // Bind login form in modal (index.html)
  const loginModalForm = document.getElementById('loginForm');
  if (loginModalForm && !loginModalForm.dataset.bound) {
    loginModalForm.dataset.bound = 'true';
    bindLoginFormHandler(loginModalForm, false);
  }

  // Bind signup form in modal (index.html)
  const signupModalForm = document.getElementById('signupForm');
  if (signupModalForm && !signupModalForm.dataset.bound) {
    signupModalForm.dataset.bound = 'true';
    bindSignupFormHandler(signupModalForm);
  }
}

function bindLoginFormHandler(form, isAdminPage = false) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const emailInput = form.querySelector('#loginEmail') || form.querySelector('#adminEmail');
    const passwordInput = form.querySelector('#loginPassword') || form.querySelector('#adminPassword');
    const loginBtn = form.querySelector('#loginBtn');

    if (!emailInput || !passwordInput || !loginBtn) return;

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
        if (response.status === 401) {
          throw new Error('Invalid email or password.');
        }
        throw new Error(data.detail || 'Login failed');
      }

      // Verify user role using /auth/me
      const userData = await Auth.verifyTokenAndGetUser(data.access_token);

      if (!userData) {
        throw new Error('Failed to verify user');
      }

      // Check role restrictions
      if (isAdminPage && userData.role !== 'admin') {
        throw new Error('Access denied. Administrator privileges required.');
      }

      // Both patient and admin can login from regular login page
      // Store auth data with correct key based on role
      const isAdmin = userData.role === 'admin';
      setAuthData(data.access_token, userData, isAdmin);

      showToast('Welcome back!', 'success');

      // If modal, close it and update UI
      const modal = form.closest('.modal-overlay');
      if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
      }

      // Redirect based on stored destination
      const redirect = Auth.getStoredRedirectDestination();
      setTimeout(() => {
        window.location.href = redirect;
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
}

function bindSignupFormHandler(form) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const nameInput = form.querySelector('#signupName');
    const emailInput = form.querySelector('#signupEmail');
    const passwordInput = form.querySelector('#signupPassword');
    const confirmInput = form.querySelector('#signupConfirmPassword');
    const termsInput = form.querySelector('#termsAgree');
    const signupBtn = form.querySelector('#signupBtn');

    if (!nameInput || !emailInput || !passwordInput || !confirmInput || !termsInput || !signupBtn) return;

    // Clear previous errors
    markValid(nameInput);
    markValid(emailInput);
    markValid(passwordInput);
    markValid(confirmInput);
    markValid(termsInput);

    // Validation
    let isValid = true;

    if (!nameInput.value.trim() || nameInput.value.trim().length < 2) {
      markInvalid(nameInput, 'Please enter your full name (at least 2 characters)');
      isValid = false;
    }

    if (!emailInput.value.trim() || !validateEmail(emailInput.value.trim())) {
      markInvalid(emailInput, 'Please enter a valid email address');
      isValid = false;
    }

    if (!passwordInput.value || passwordInput.value.length < 8) {
      markInvalid(passwordInput, 'Password must be at least 8 characters');
      isValid = false;
    }

    if (passwordInput.value !== confirmInput.value) {
      markInvalid(confirmInput, 'Passwords do not match');
      isValid = false;
    }

    if (!termsInput.checked) {
      markInvalid(termsInput, 'You must agree to the Terms of Service and Privacy Policy');
      isValid = false;
    }

    if (!isValid) {
      showToast('Please correct the highlighted fields', 'warning');
      return;
    }

    setButtonLoading(signupBtn, true);

    try {
      const response = await fetch(`${API_BASE}/auth/signup`, {
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
        throw new Error(data.detail || 'Signup failed');
      }

      showToast('Account created successfully! Redirecting to login...', 'success');

      // If modal, close it
      const modal = form.closest('.modal-overlay');
      if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
      }

      setTimeout(() => {
        window.location.href = 'login.html?signup=success';
      }, 1500);

    } catch (error) {
      showToast(error.message, 'error');
    } finally {
      setButtonLoading(signupBtn, false);
    }
  });
}

function showUserProfile(container, user) {
  const initials = user.email ? user.email.charAt(0).toUpperCase() : 'U';
  const roleLabel = user.role ? user.role.charAt(0).toUpperCase() + user.role.slice(1) : 'User';

  container.innerHTML = `
    <div class="user-menu" id="mainUserMenu">
      <button type="button" class="user-menu-trigger" id="mainUserMenuTrigger" aria-label="User Menu" aria-expanded="false" aria-haspopup="true">
        <div class="user-avatar">${initials}</div>
        <span class="user-name">${user.email || 'User'}</span>
        <svg class="chevron" width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
      </button>
      <div class="user-dropdown" id="mainUserDropdown" role="menu" aria-hidden="true">
        <div class="dropdown-header">
          <div class="dropdown-user-info">
            <div class="dropdown-avatar">${initials}</div>
            <div class="dropdown-user-details">
              <span class="dropdown-user-name">${user.email || 'User'}</span>
              <span class="dropdown-user-role">${roleLabel}</span>
            </div>
          </div>
        </div>
        <div class="dropdown-divider"></div>
        <a href="chat.html" class="dropdown-item" role="menuitem">
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
          <span>Open AI Chatbot</span>
        </a>
        <div class="dropdown-divider"></div>
        <button type="button" class="dropdown-item dropdown-item-danger" id="mainLogoutBtn" role="menuitem">
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path></svg>
          <span>Sign Out</span>
        </button>
      </div>
    </div>
    <button type="button" class="menu-toggle" id="mobileMenuToggle" aria-label="Toggle Navigation Menu" aria-expanded="false">
      <span></span>
      <span></span>
      <span></span>
    </button>
  `;
  initMobileMenu();
  initMainUserMenu();
}

function initMainUserMenu() {
  const userMenuTrigger = document.getElementById('mainUserMenuTrigger');
  const userDropdown = document.getElementById('mainUserDropdown');

  if (!userMenuTrigger || !userDropdown) return;

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

  const logoutBtn = document.getElementById('mainLogoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      Auth.logout();
    });
  }
}

function markInvalid(input, message) {
  input.classList.add('is-invalid');
  const parent = input.parentElement;
  let feedback = parent.querySelector('.error-feedback');
  if (!feedback) {
    feedback = document.createElement('div');
    feedback.className = 'error-feedback';
    parent.appendChild(feedback);
  }
  feedback.textContent = message;
  feedback.style.display = 'block';
}

function markValid(input) {
  input.classList.remove('is-invalid');
  const parent = input.parentElement;
  const feedback = parent.querySelector('.error-feedback');
  if (feedback) {
    feedback.style.display = 'none';
  }
}

function validateEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

// ============================================================================
// User Menu Dropdown & User Display
// ============================================================================
// NOTE: For admin pages (admin-dashboard.html, admin-documents.html), the
// implementations in admin.js are used (they handle cloning to avoid duplicate
// event listeners). This file provides fallback for non-admin pages.

// ============================================================================
// Shared Auth Module (used by both patient and admin pages)
// ============================================================================

const Auth = {
  // Determine redirect destination based on current page and referrer
  getRedirectDestination() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const urlParams = new URLSearchParams(window.location.search);
    const redirectParam = urlParams.get('redirect');

    // Explicit redirect parameter takes priority
    if (redirectParam) {
      return redirectParam;
    }

    // Determine from current page
    if (currentPage === 'chat.html' || currentPage.includes('chat')) {
      return 'chat.html';
    }
    if (currentPage === 'admin-login.html') {
      return 'admin-dashboard.html';
    }
    if (currentPage === 'admin-dashboard.html' || currentPage === 'admin-documents.html') {
      return currentPage;
    }

    // Default: main website
    return 'index.html';
  },

  // Store redirect destination before login
  setRedirectDestination(dest) {
    sessionStorage.setItem('auth_redirect', dest);
  },

  getStoredRedirectDestination() {
    const dest = sessionStorage.getItem('auth_redirect');
    sessionStorage.removeItem('auth_redirect');
    return dest || this.getRedirectDestination();
  },

  async verifyTokenAndGetUser(token) {
    try {
      const response = await fetch(`${API_BASE}/auth/me`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!response.ok) return null;
      return await response.json();
    } catch {
      return null;
    }
  },

  logout() {
    clearAllAuthData();
    showToast('You have been signed out', 'info');
    setTimeout(() => {
      window.location.href = 'index.html';
    }, 500);
  },

  // Check if current page requires auth and redirect if needed (synchronous)
  requireAuth(allowedRoles = ['patient', 'admin']) {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';

    // Pages that don't require auth
    const publicPages = ['index.html', 'login.html', 'signup.html', 'admin-login.html', ''];
    if (publicPages.includes(currentPage)) {
      return true;
    }

    if (!isAuthenticatedAny()) {
      // Store current page as redirect destination
      this.setRedirectDestination(currentPage);
      window.location.href = 'login.html';
      return false;
    }

    const user = getStoredUserAny();
    if (!user || !allowedRoles.includes(user.role)) {
      this.logout();
      return false;
    }

    // Token verification is handled by individual modules via API responses (401/403)
    // This avoids race conditions with async logout
    // updateUserDisplay() and initUserMenu() are called by page-specific init functions
    // (AdminDashboard.init, AdminDocuments.init, etc.)

    return true;
  }
};

// ============================================================================
// Login Page Module
// ============================================================================

const LoginPage = {
  init() {
    this.checkExistingSession();
    this.bindLoginForm();
    this.bindPasswordToggle();
    this.bindInputValidation();
    this.setRedirectFromReferrer();
  },

  setRedirectFromReferrer() {
    // Store redirect destination from referrer or current page
    const referrer = document.referrer;
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';

    // Don't override if already set
    if (sessionStorage.getItem('auth_redirect')) return;

    // Check if coming from chat page (chatbot login)
    if (referrer.includes('chat.html') || currentPage === 'chat.html') {
      Auth.setRedirectDestination('chat.html');
    } else if (referrer.includes('admin-dashboard.html') || referrer.includes('admin-documents.html')) {
      Auth.setRedirectDestination('admin-dashboard.html');
    } else if (referrer.includes('admin-login.html')) {
      Auth.setRedirectDestination('admin-dashboard.html');
    }
  },

  checkExistingSession() {
    if (isAuthenticatedAny()) {
      const tokenKey = getValidTokenKey();
      if (tokenKey) {
        verifyToken(tokenKey).then(userData => {
          if (userData && (userData.role === 'patient' || userData.role === 'admin')) {
            const redirect = Auth.getStoredRedirectDestination();
            window.location.href = redirect;
          } else {
            clearAllAuthData();
          }
        }).catch(() => {
          clearAllAuthData();
        });
      }
    }
  },

  bindLoginForm() {
    const form = document.getElementById('loginForm') || document.getElementById('adminLoginForm');
    if (!form) return;

    // Determine if this is admin login page
    const isAdminPage = document.body.classList.contains('admin-login-page');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const emailInput = document.getElementById('loginEmail') || document.getElementById('adminEmail');
      const passwordInput = document.getElementById('loginPassword') || document.getElementById('adminPassword');
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

        // Verify user role using /auth/me
        const userData = await Auth.verifyTokenAndGetUser(data.access_token);

        if (!userData) {
          throw new Error('Failed to verify user');
        }

        // Check role restrictions
        if (isAdminPage && userData.role !== 'admin') {
          throw new Error('Access denied. Administrator privileges required.');
        }

        // Both patient and admin can login from regular login page
        // Store auth data with correct key based on role
        const isAdmin = userData.role === 'admin';
        setAuthData(data.access_token, userData, isAdmin);

        showToast('Welcome back!', 'success');

        // Redirect based on stored destination
        const redirect = Auth.getStoredRedirectDestination();
        setTimeout(() => {
          window.location.href = redirect;
        }, 500);

      } catch (error) {
        // Handle network errors
        if (error instanceof TypeError && error.message.includes('fetch')) {
          showToast('Unable to connect to the server. Please try again.', 'error');
        } else {
          showToast(error.message, 'error');
        }
        // Ensure modal stays open on error - do not redirect
      } finally {
        setButtonLoading(loginBtn, false);
      }
    });
  },

  bindPasswordToggle() {
    document.querySelectorAll('.toggle-password').forEach(toggleBtn => {
      const passwordInput = toggleBtn.previousElementSibling;
      if (passwordInput && passwordInput.type === 'password') {
        toggleBtn.addEventListener('click', () => {
          const isPassword = passwordInput.type === 'password';
          passwordInput.type = isPassword ? 'text' : 'password';
          toggleBtn.setAttribute('aria-expanded', isPassword.toString());
        });
      }
    });
  },

  bindInputValidation() {
    // Clear validation errors when user starts typing
    document.querySelectorAll('.form-control').forEach(input => {
      input.addEventListener('input', () => {
        if (input.classList.contains('is-invalid')) {
          markValid(input);
        }
      });

      // Also clear on blur if valid
      input.addEventListener('blur', () => {
        if (input.type === 'email' && input.value.trim() && validateEmail(input.value.trim())) {
          markValid(input);
        } else if (input.type === 'password' && input.value.trim()) {
          markValid(input);
        }
      });
    });
  }
};

// ============================================================================
// Signup Page Module (Patient only)
// ============================================================================

const SignupPage = {
  init() {
    this.bindSignupForm();
    this.bindPasswordToggle();
    this.bindConfirmPasswordValidation();
    this.bindInputValidation();
  },

  bindSignupForm() {
    const form = document.getElementById('signupForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const nameInput = document.getElementById('signupName');
      const emailInput = document.getElementById('signupEmail');
      const passwordInput = document.getElementById('signupPassword');
      const confirmInput = document.getElementById('signupConfirmPassword');
      const termsInput = document.getElementById('termsAgree');
      const signupBtn = document.getElementById('signupBtn');

      // Clear previous errors
      markValid(nameInput);
      markValid(emailInput);
      markValid(passwordInput);
      markValid(confirmInput);
      markValid(termsInput);

      // Validation
      let isValid = true;

      if (!nameInput.value.trim() || nameInput.value.trim().length < 2) {
        markInvalid(nameInput, 'Please enter your full name (at least 2 characters)');
        isValid = false;
      }

      if (!emailInput.value.trim() || !validateEmail(emailInput.value.trim())) {
        markInvalid(emailInput, 'Please enter a valid email address');
        isValid = false;
      }

      if (!passwordInput.value || passwordInput.value.length < 8) {
        markInvalid(passwordInput, 'Password must be at least 8 characters');
        isValid = false;
      }

      if (passwordInput.value !== confirmInput.value) {
        markInvalid(confirmInput, 'Passwords do not match');
        isValid = false;
      }

      if (!termsInput.checked) {
        markInvalid(termsInput, 'You must agree to the Terms of Service and Privacy Policy');
        isValid = false;
      }

      if (!isValid) {
        showToast('Please correct the highlighted fields', 'warning');
        return;
      }

      setButtonLoading(signupBtn, true);

      try {
        const response = await fetch(`${API_BASE}/auth/signup`, {
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
          throw new Error(data.detail || 'Signup failed');
        }

        showToast('Account created successfully! Redirecting to login...', 'success');

        setTimeout(() => {
          window.location.href = 'login.html?signup=success';
        }, 1500);

      } catch (error) {
        showToast(error.message, 'error');
      } finally {
        setButtonLoading(signupBtn, false);
      }
    });
  },

  bindPasswordToggle() {
    document.querySelectorAll('.toggle-password').forEach(toggleBtn => {
      const passwordInput = toggleBtn.previousElementSibling;
      if (passwordInput && passwordInput.type === 'password') {
        toggleBtn.addEventListener('click', () => {
          const isPassword = passwordInput.type === 'password';
          passwordInput.type = isPassword ? 'text' : 'password';
          toggleBtn.setAttribute('aria-expanded', isPassword.toString());
        });
      }
    });
  },

  bindConfirmPasswordValidation() {
    const passwordInput = document.getElementById('signupPassword');
    const confirmInput = document.getElementById('signupConfirmPassword');

    if (passwordInput && confirmInput) {
      const validate = () => {
        if (confirmInput.value && passwordInput.value !== confirmInput.value) {
          markInvalid(confirmInput, 'Passwords do not match');
        } else {
          markValid(confirmInput);
        }
      };

      confirmInput.addEventListener('input', validate);
      passwordInput.addEventListener('input', validate);
    }
  },

  bindInputValidation() {
    document.querySelectorAll('.form-control').forEach(input => {
      input.addEventListener('input', () => {
        if (input.classList.contains('is-invalid')) {
          markValid(input);
        }
      });
    });
  }
};

// ============================================================================
// Initialize on DOM Ready
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Check for signup success message
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get('signup') === 'success') {
    showToast('Account created successfully! Please sign in.', 'success');
    window.history.replaceState({}, document.title, window.location.pathname);
  }

  // Initialize appropriate page module - ONLY on dedicated auth pages
  if (document.body.classList.contains('auth-login-page')) {
    LoginPage.init();
  } else if (document.body.classList.contains('auth-signup-page')) {
    SignupPage.init();
  }

  // Initialize main website auth (index.html) - shows profile or login buttons
  // This also handles modal forms on index.html
  initMainWebsiteAuth();
});

// Export for global access
window.Auth = Auth;
window.LoginPage = LoginPage;
window.SignupPage = SignupPage;
window.isAuthenticatedAny = isAuthenticatedAny;
window.getValidTokenKey = getValidTokenKey;
window.verifyToken = verifyToken;
window.getStoredUserAny = getStoredUserAny;
window.clearAllAuthData = clearAllAuthData;
window.getAuthHeader = getAuthHeader;
window.getAuthHeaderAny = getAuthHeaderAny;
window.getValidAuthHeader = getValidAuthHeader;