// Common JavaScript functionality for Web Scraper Dashboard

// Global configuration - can be overridden by unified_config.js
let CONFIG = {
    API_BASE_URL: '/api/v1',
    WS_URL: `ws://${window.location.host}/ws`,
    REFRESH_INTERVAL: 5000, // 5 seconds
    USE_REAL_DATA: true, // Always prefer real data over mock data
    CHART_COLORS: {
        primary: '#0d6efd',
        success: '#198754',
        warning: '#ffc107',
        danger: '#dc3545',
        info: '#0dcaf0',
        secondary: '#6c757d'
    }
};

// Override with unified configuration if available
if (typeof window !== 'undefined' && window.UNIFIED_CONFIG) {
    CONFIG = { ...CONFIG, ...window.UNIFIED_CONFIG };
}

// Global state
let globalState = {
    isConnected: false,
    activeJobs: 0,
    activeAgents: 0,
    systemStatus: 'unknown'
};

// Enhanced loading management with timeout and component-specific loading
let loadingTimeouts = new Map();
let componentLoadingStates = new Map();

function showLoading(componentId = null, timeout = 10000) {
    if (componentId) {
        // Component-specific loading
        const element = document.getElementById(componentId);
        if (element) {
            element.classList.add('loading-state');
            componentLoadingStates.set(componentId, true);
        }
    } else {
        // Global loading overlay
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('d-none');
        }
    }

    // Set timeout to automatically hide loading
    if (timeout > 0) {
        const timeoutId = setTimeout(() => {
            hideLoading(componentId);
            if (!componentId) {
                console.warn('Loading timeout reached - hiding loading overlay');
                showAlert('Request timed out. Please check your connection and try again.', 'warning', 5000);
            }
        }, timeout);

        loadingTimeouts.set(componentId || 'global', timeoutId);
    }
}

function hideLoading(componentId = null) {
    // Clear timeout
    const timeoutKey = componentId || 'global';
    if (loadingTimeouts.has(timeoutKey)) {
        clearTimeout(loadingTimeouts.get(timeoutKey));
        loadingTimeouts.delete(timeoutKey);
    }

    if (componentId) {
        // Component-specific loading
        const element = document.getElementById(componentId);
        if (element) {
            element.classList.remove('loading-state');
            componentLoadingStates.set(componentId, false);
        }
    } else {
        // Global loading overlay
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('d-none');
        }
    }
}

function isLoading(componentId = null) {
    if (componentId) {
        return componentLoadingStates.get(componentId) || false;
    }
    const overlay = document.getElementById('loading-overlay');
    return overlay && !overlay.classList.contains('d-none');
}

function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.getElementById('alert-container');
    const alertId = 'alert-' + Date.now();
    
    const alertHTML = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            <i class="bi bi-${getAlertIcon(type)}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    alertContainer.insertAdjacentHTML('beforeend', alertHTML);
    
    // Auto-dismiss after duration
    if (duration > 0) {
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, duration);
    }
}

function getAlertIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle',
        'primary': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Enhanced API helper functions with retry mechanism and better error handling
async function apiRequest(endpoint, options = {}, componentId = null, retries = 2) {
    const url = CONFIG.API_BASE_URL + endpoint;

    // Show loading indicator with timeout
    showLoading(componentId, 15000);

    let lastError;

    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const defaultOptions = {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                timeout: 10000 // 10 second timeout
            };

            const finalOptions = { ...defaultOptions, ...options };

            console.log(`Making API request to: ${url} (attempt ${attempt + 1}/${retries + 1})`);

            // Create abort controller for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), finalOptions.timeout);
            finalOptions.signal = controller.signal;

            const response = await fetch(url, finalOptions);
            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}`;

                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorData.message || errorMessage;
                } catch (e) {
                    errorMessage = response.statusText || errorMessage;
                }

                // Specific handling for common errors
                if (response.status === 404) {
                    errorMessage = 'API endpoint not found. The web interface may not be fully connected to the backend.';
                } else if (response.status === 500) {
                    errorMessage = 'Server error. Please check if the backend services are running.';
                } else if (response.status === 503) {
                    errorMessage = 'Service unavailable. Backend services may be starting up.';
                }

                throw new Error(errorMessage);
            }

            const data = await response.json();
            hideLoading(componentId);
            return data;

        } catch (error) {
            lastError = error;
            console.error(`API request failed (attempt ${attempt + 1}):`, error);

            // If this is the last attempt or a non-retryable error, break
            if (attempt === retries ||
                error.name === 'AbortError' ||
                error.message.includes('Failed to fetch') ||
                (error.message.includes('HTTP') && !error.message.includes('500'))) {
                break;
            }

            // Wait before retry (exponential backoff)
            if (attempt < retries) {
                const delay = Math.pow(2, attempt) * 1000; // 1s, 2s, 4s...
                console.log(`Retrying in ${delay}ms...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    // Handle final error
    hideLoading(componentId);

    if (lastError.name === 'AbortError') {
        showAlert('Request timed out. Please check your connection and try again.', 'warning', 8000);
    } else if (lastError.message.includes('Failed to fetch')) {
        showAlert('Backend API server is not running. Please start the backend server.', 'danger', 10000);
    } else {
        showAlert(`API Error: ${lastError.message}`, 'danger', 8000);
    }

    throw lastError;
}

async function getJobs(filters = {}) {
    const params = new URLSearchParams(filters);
    return await apiRequest(`/jobs?${params}`);
}

async function getAgents() {
    return await apiRequest('/agents');
}

async function getSystemMetrics() {
    return await apiRequest('/monitoring/system');
}

async function createJob(jobData) {
    return await apiRequest('/jobs', {
        method: 'POST',
        body: JSON.stringify(jobData)
    });
}

async function cancelJob(jobId) {
    return await apiRequest(`/jobs/${jobId}/cancel`, {
        method: 'POST'
    });
}

// Enhanced Navigation functions
function initializeNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link, .sidebar .nav-link');

    // Remove active class from all links
    navLinks.forEach(link => {
        link.classList.remove('active');
    });

    // Add active class to current page link
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (currentPath.startsWith(href) && href !== '/app')) {
            link.classList.add('active');

            // Add animation to active link
            link.style.animationDelay = '0.1s';
            link.classList.add('fade-in');
        }
    });

    // Add click handlers for smooth navigation
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');

            // Only handle internal navigation
            if (href && href.startsWith('/app')) {
                e.preventDefault();
                navigateToPage(href, this);
            }
        });
    });
}

function navigateToPage(url, clickedLink) {
    // Add loading state
    showPageTransition();

    // Update active states immediately for better UX
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });

    if (clickedLink) {
        clickedLink.classList.add('active');
    }

    // Navigate after a short delay for smooth transition
    setTimeout(() => {
        window.location.href = url;
    }, 150);
}

function showPageTransition() {
    // Add a subtle loading overlay for page transitions
    const overlay = document.createElement('div');
    overlay.className = 'page-transition-overlay';
    overlay.innerHTML = `
        <div class="transition-spinner">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);

    // Remove overlay after navigation
    setTimeout(() => {
        if (overlay.parentNode) {
            overlay.remove();
        }
    }, 1000);
}

// System status functions
async function updateSystemStatus() {
    try {
        const metrics = await getSystemMetrics();
        const jobs = await getJobs({ status: 'running' });
        const agents = await getAgents();
        
        // Update global state
        globalState.activeJobs = jobs.total_count || 0;
        globalState.activeAgents = agents.agents ? agents.agents.filter(a => a.status === 'active').length : 0;
        globalState.systemStatus = 'online';
        
        // Update UI elements
        updateStatusIndicators();
        
    } catch (error) {
        console.error('Failed to update system status:', error);
        globalState.systemStatus = 'offline';
        updateStatusIndicators();
    }
}

function updateStatusIndicators() {
    const statusElement = document.getElementById('system-status');
    const activeJobsElement = document.getElementById('active-jobs-count');
    const activeAgentsElement = document.getElementById('active-agents-count');
    
    if (statusElement) {
        statusElement.textContent = globalState.systemStatus;
        statusElement.className = `badge bg-${globalState.systemStatus === 'online' ? 'success' : 'danger'}`;
    }
    
    if (activeJobsElement) {
        activeJobsElement.textContent = globalState.activeJobs;
    }
    
    if (activeAgentsElement) {
        activeAgentsElement.textContent = globalState.activeAgents;
    }
}

// Quick action functions
function createNewJob() {
    window.location.href = '/app/jobs#create';
}

function viewSystemStatus() {
    window.location.href = '/app/monitoring';
}

function exportData() {
    window.location.href = '/app/data#export';
}

// Format helper functions
function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString();
}

function formatDuration(seconds) {
    if (!seconds) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatPercentage(value, decimals = 1) {
    return `${value.toFixed(decimals)}%`;
}

// Chart helper functions
function createChart(canvasId, type, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            }
        }
    };
    
    return new Chart(ctx, {
        type: type,
        data: data,
        options: { ...defaultOptions, ...options }
    });
}

// Table helper functions
function createDataTable(containerId, data, columns) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    let tableHTML = `
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        ${columns.map(col => `<th>${col.title}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
    `;
    
    data.forEach(row => {
        tableHTML += '<tr>';
        columns.forEach(col => {
            const value = col.render ? col.render(row[col.field], row) : row[col.field];
            tableHTML += `<td>${value}</td>`;
        });
        tableHTML += '</tr>';
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = tableHTML;
}

// Initialize periodic updates
function startPeriodicUpdates() {
    setInterval(updateSystemStatus, CONFIG.REFRESH_INTERVAL);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    startPeriodicUpdates();
});
