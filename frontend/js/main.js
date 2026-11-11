/**
 * MediTrust Hospital Website - Phase 1 JavaScript
 * Pure Vanilla JavaScript (Zero external dependencies)
 * Features:
 * - Responsive Navbar & Mobile Drawer
 * - Scrollspy & Sticky Header
 * - Doctor Specialty Filter & Pre-select Booking
 * - Form Validation (Appointment & Contact)
 * - Appointment Confirmation Modal & Demo Auth Modals
 * - Toast Notification System
 */

document.addEventListener('DOMContentLoaded', () => {
  initStickyHeader();
  initMobileMenu();
  initScrollSpy();
  initDoctorFilters();
  initDepartmentDoctorSync();
  initAppointmentBooking();
  initContactForm();
  initDateConstraints();
  initPlaceholderLinks();
  // Auth modals are now initialized by auth.js initMainWebsiteAuth()
});

/* ==========================================================================
   1. Sticky Header & Back-to-top / Shadow Effect
   ========================================================================== */
function initStickyHeader() {
  const header = document.querySelector('.site-header');
  if (!header) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  });
}

/* ==========================================================================
   2. Mobile Drawer Navigation
   ========================================================================== */
function initMobileMenu() {
  const toggleBtn = document.getElementById('mobileMenuToggle');
  const navMenu = document.getElementById('navMenu');
  const navLinks = document.querySelectorAll('.nav-link');

  if (!toggleBtn || !navMenu) return;

  function toggleMenu() {
    const isOpen = navMenu.classList.toggle('active');
    toggleBtn.classList.toggle('active', isOpen);
    toggleBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  }

  function closeMenu() {
    navMenu.classList.remove('active');
    toggleBtn.classList.remove('active');
    toggleBtn.setAttribute('aria-expanded', 'false');
  }

  toggleBtn.addEventListener('click', toggleMenu);

  navLinks.forEach(link => {
    link.addEventListener('click', closeMenu);
  });

  document.addEventListener('click', (e) => {
    if (!navMenu.contains(e.target) && !toggleBtn.contains(e.target)) {
      closeMenu();
    }
  });
}

/* ==========================================================================
   3. Scroll Spy (Active Navigation Indicator)
   ========================================================================== */
function initScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-menu .nav-link');

  if (!sections.length || !navLinks.length) return;

  window.addEventListener('scroll', () => {
    let current = '';
    const scrollPosition = window.scrollY + 120;

    sections.forEach(section => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      const href = link.getAttribute('href');
      if (href === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}

/* ==========================================================================
   4. Doctor Specialty Filter & Quick Booking Sync
   ========================================================================== */
function initDoctorFilters() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  const doctorCards = document.querySelectorAll('.doctor-card');

  if (!filterBtns.length || !doctorCards.length) return;

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Toggle active button style
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filterValue = btn.getAttribute('data-filter');

      doctorCards.forEach(card => {
        const specialty = card.getAttribute('data-specialty');
        if (filterValue === 'all' || specialty === filterValue) {
          card.classList.remove('hidden');
        } else {
          card.classList.add('hidden');
        }
      });
    });
  });

  // Handle "Book Consultation" clicks on doctor cards to pre-populate booking form
  const doctorBookingBtns = document.querySelectorAll('.btn-doctor-book');
  doctorBookingBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const doctorName = btn.getAttribute('data-doctor');
      const department = btn.getAttribute('data-department');

      const deptSelect = document.getElementById('appointmentDept');
      const docSelect = document.getElementById('appointmentDoctor');

      if (deptSelect && department) {
        deptSelect.value = department;
        populateDoctorsByDepartment(department, doctorName);
      }

      // Smooth scroll to appointment section
      const appointmentSection = document.getElementById('appointment');
      if (appointmentSection) {
        appointmentSection.scrollIntoView({ behavior: 'smooth' });
        showToast(`Selected Dr. ${doctorName} for appointment`, 'info');
      }
    });
  });
}

/* ==========================================================================
   5. Department & Doctor Dynamic Selection Sync
   ========================================================================== */
const doctorsByDepartment = {
  cardiology: [
    { name: 'Dr. Elena Rostova', title: 'MD, FACC - Senior Interventional Cardiologist' },
    { name: 'Dr. Marcus Vance', title: 'MD - Cardiac Electrophysiology Specialist' }
  ],
  neurology: [
    { name: 'Dr. Julian Thorne', title: 'MD, PhD - Senior Consultant Neurosurgeon' },
    { name: 'Dr. Priya Sharma', title: 'MD, DM - Comprehensive Neurologist' }
  ],
  orthopedics: [
    { name: 'Dr. David Chen', title: 'MS (Ortho), FRCS - Joint Replacement Surgeon' },
    { name: 'Dr. Rachel Adams', title: 'MS - Sports Medicine & Arthroscopy Specialist' }
  ],
  pediatrics: [
    { name: 'Dr. Sarah Jenkins', title: 'MD (Pediatrics) - Chief Pediatrician' },
    { name: 'Dr. Carlos Mendoza', title: 'MD - Neonatology & Adolescent Health' }
  ],
  oncology: [
    { name: 'Dr. Aaliyah Khan', title: 'MD, DM - Medical Oncologist' }
  ],
  medicine: [
    { name: 'Dr. Robert Sterling', title: 'MD - Internal Medicine & Preventive Health' }
  ]
};

function populateDoctorsByDepartment(departmentKey, preselectedDoctor = '') {
  const doctorSelect = document.getElementById('appointmentDoctor');
  if (!doctorSelect) return;

  doctorSelect.innerHTML = '<option value="">Select Doctor (Optional / Any Available)</option>';

  if (departmentKey && doctorsByDepartment[departmentKey]) {
    doctorsByDepartment[departmentKey].forEach(doc => {
      const option = document.createElement('option');
      option.value = doc.name;
      option.textContent = `${doc.name} (${doc.title.split('-')[0].trim()})`;
      if (preselectedDoctor && doc.name.toLowerCase().includes(preselectedDoctor.toLowerCase())) {
        option.selected = true;
      }
      doctorSelect.appendChild(option);
    });
  }
}

function initDepartmentDoctorSync() {
  const deptSelect = document.getElementById('appointmentDept');
  if (!deptSelect) return;

  deptSelect.addEventListener('change', (e) => {
    populateDoctorsByDepartment(e.target.value);
  });
}

/* ==========================================================================
   6. Date Picker Constraints (Prevent Past Dates)
   ========================================================================== */
function initDateConstraints() {
  const dateInput = document.getElementById('appointmentDate');
  if (!dateInput) return;

  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');

  // Set min date attribute to today
  dateInput.setAttribute('min', `${year}-${month}-${day}`);
}

/* ==========================================================================
   7. Appointment Booking Form Validation & Pass Modal
   ========================================================================== */
function initAppointmentBooking() {
  const form = document.getElementById('appointmentForm');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();

    let isValid = true;

    // Field references
    const nameInput = document.getElementById('appointmentName');
    const emailInput = document.getElementById('appointmentEmail');
    const phoneInput = document.getElementById('appointmentPhone');
    const deptInput = document.getElementById('appointmentDept');
    const doctorInput = document.getElementById('appointmentDoctor');
    const dateInput = document.getElementById('appointmentDate');
    const timeInput = document.getElementById('appointmentTime');
    const notesInput = document.getElementById('appointmentNotes');

    // Validation 1: Patient Name
    if (!nameInput.value.trim() || nameInput.value.trim().length < 3) {
      markInvalid(nameInput, 'Please enter a valid patient name (at least 3 characters)');
      isValid = false;
    } else {
      markValid(nameInput);
    }

    // Validation 2: Email
    if (!validateEmail(emailInput.value.trim())) {
      markInvalid(emailInput, 'Please enter a valid email address');
      isValid = false;
    } else {
      markValid(emailInput);
    }

    // Validation 3: Phone
    const cleanPhone = phoneInput.value.replace(/\D/g, '');
    if (cleanPhone.length < 10) {
      markInvalid(phoneInput, 'Please enter a valid phone number (at least 10 digits)');
      isValid = false;
    } else {
      markValid(phoneInput);
    }

    // Validation 4: Department
    if (!deptInput.value) {
      markInvalid(deptInput, 'Please select a medical department');
      isValid = false;
    } else {
      markValid(deptInput);
    }

    // Validation 5: Date
    if (!dateInput.value) {
      markInvalid(dateInput, 'Please select a preferred appointment date');
      isValid = false;
    } else {
      markValid(dateInput);
    }

    // Validation 6: Time slot
    if (!timeInput.value) {
      markInvalid(timeInput, 'Please select a preferred time slot');
      isValid = false;
    } else {
      markValid(timeInput);
    }

    if (!isValid) {
      showToast('Please correct the highlighted fields before booking.', 'warning');
      return;
    }

    // Create Appointment Summary Pass in Confirmation Modal
    const appointmentPassId = `MDT-${new Date().getFullYear()}-${Math.floor(1000 + Math.random() * 9000)}`;
    const doctorDisplay = doctorInput.value ? doctorInput.value : 'First Available Specialist';
    const deptDisplay = deptInput.options[deptInput.selectedIndex].text;

    document.getElementById('passToken').textContent = appointmentPassId;
    document.getElementById('passPatient').textContent = nameInput.value.trim();
    document.getElementById('passDepartment').textContent = deptDisplay;
    document.getElementById('passDoctor').textContent = doctorDisplay;
    document.getElementById('passDateTime').textContent = `${dateInput.value} at ${timeInput.value}`;

    // Open Confirmation Modal
    openModal('appointmentConfirmationModal');

    // Reset Form
    form.reset();
    populateDoctorsByDepartment('');
    showToast('Appointment request created (Demo Mode)', 'success');
  });

  // Real-time error clearance on input
  const inputs = form.querySelectorAll('.form-control');
  inputs.forEach(input => {
    input.addEventListener('input', () => {
      markValid(input);
    });
  });
}

/* ==========================================================================
   8. Contact Form Handling & Validation
   ========================================================================== */
function initContactForm() {
  const form = document.getElementById('contactInquiryForm');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();

    let isValid = true;
    const name = document.getElementById('contactName');
    const email = document.getElementById('contactEmail');
    const subject = document.getElementById('contactSubject');
    const message = document.getElementById('contactMessage');

    if (!name.value.trim() || name.value.trim().length < 2) {
      markInvalid(name, 'Please enter your full name');
      isValid = false;
    } else {
      markValid(name);
    }

    if (!validateEmail(email.value.trim())) {
      markInvalid(email, 'Please enter a valid email address');
      isValid = false;
    } else {
      markValid(email);
    }

    if (!subject.value.trim()) {
      markInvalid(subject, 'Please specify a subject');
      isValid = false;
    } else {
      markValid(subject);
    }

    if (!message.value.trim() || message.value.trim().length < 10) {
      markInvalid(message, 'Please enter a message of at least 10 characters');
      isValid = false;
    } else {
      markValid(message);
    }

    if (!isValid) {
      showToast('Please complete all contact inquiry fields.', 'warning');
      return;
    }

    form.reset();
    showToast('Thank you! Your inquiry has been sent (Demo mode).', 'success');
  });

  const inputs = form.querySelectorAll('.form-control');
  inputs.forEach(input => {
    input.addEventListener('input', () => {
      markValid(input);
    });
  });
}

/* ==========================================================================
   9. Authentication Modals (Login & Sign Up Demo)
   ========================================================================== */
function initAuthModals() {
  const loginBtns = document.querySelectorAll('.btn-open-login');
  const signupBtns = document.querySelectorAll('.btn-open-signup');
  const closeBtns = document.querySelectorAll('.modal-close-btn, .modal-close-btn-action');
  const overlays = document.querySelectorAll('.modal-overlay');

  // Open Login Modal
  loginBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      closeAllModals();
      openModal('loginModal');
    });
  });

  // Open Sign Up Modal
  signupBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      closeAllModals();
      openModal('signupModal');
    });
  });

  // Modal switcher links
  const switchToSignup = document.getElementById('switchToSignup');
  const switchToLogin = document.getElementById('switchToLogin');

  if (switchToSignup) {
    switchToSignup.addEventListener('click', (e) => {
      e.preventDefault();
      closeAllModals();
      openModal('signupModal');
    });
  }

  if (switchToLogin) {
    switchToLogin.addEventListener('click', (e) => {
      e.preventDefault();
      closeAllModals();
      openModal('loginModal');
    });
  }

  // Close modals
  closeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      closeAllModals();
    });
  });

  overlays.forEach(overlay => {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        closeAllModals();
      }
    });
  });

  // Close with Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeAllModals();
    }
  });

  // Note: Login/Signup form handlers are now in auth.js for real API integration.
  // The modal forms on index.html use the same IDs but are handled separately.
  // If modal forms exist on this page, they will be handled by auth.js when loaded.
}

/* ==========================================================================
   Helper Functions (Validation, Modal, Toast)
   ========================================================================== */
function validateEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

function markInvalid(inputElement, errorMessage) {
  inputElement.classList.add('is-invalid');
  const parent = inputElement.parentElement;
  let feedback = parent.querySelector('.error-feedback');
  if (!feedback) {
    feedback = document.createElement('div');
    feedback.className = 'error-feedback';
    parent.appendChild(feedback);
  }
  feedback.textContent = errorMessage;
  feedback.style.display = 'block';
}

function markValid(inputElement) {
  inputElement.classList.remove('is-invalid');
  const parent = inputElement.parentElement;
  const feedback = parent.querySelector('.error-feedback');
  if (feedback) {
    feedback.style.display = 'none';
  }
}

function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
}

function closeAllModals() {
  const activeModals = document.querySelectorAll('.modal-overlay.active');
  activeModals.forEach(m => m.classList.remove('active'));
  document.body.style.overflow = '';
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
  } else {
    iconSvg = '<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
  }

  toast.innerHTML = `${iconSvg}<span>${message}</span>`;
  container.appendChild(toast);

  // Trigger animation
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

/* ==========================================================================
   Handle Footer & Header Placeholder Links
   ========================================================================== */
function initPlaceholderLinks() {
  const placeholderLinks = document.querySelectorAll('a[href="#"]');
  placeholderLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      showToast('MediTrust Demo: Link will be enabled in production.', 'info');
    });
  });
}

// Export functions for auth.js to use
window.initAuthModals = initAuthModals;
window.initMobileMenu = initMobileMenu;
window.openModal = openModal;
window.closeAllModals = closeAllModals;
window.showToast = showToast;

