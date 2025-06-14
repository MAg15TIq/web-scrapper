/**
 * Authentication Manager for Web Scraper Dashboard
 * Handles JWT token management, user authentication, and session management
 */

class AuthManager {
    constructor() {
        this.token = null;
        this.user = null;
        this.refreshTimer = null;
        this.init();
    }

    init() {
        // Load token from storage
        this.loadToken();
        
        // Set up automatic token refresh
        if (this.token) {
            this.setupTokenRefresh();
        }
        
        // Set up API request interceptor
        this.setupApiInterceptor();
    }

    loadToken() {
        // Check both localStorage and sessionStorage
        this.token = localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
        this.tokenType = localStorage.getItem('token_type') || sessionStorage.getItem('token_type') || 'bearer';
    }

    saveToken(token, tokenType = 'bearer', remember = false) {
        this.token = token;
        this.tokenType = tokenType;
        
        if (remember) {
            localStorage.setItem('access_token', token);
            localStorage.setItem('token_type', tokenType);
            // Remove from session storage
            sessionStorage.removeItem('access_token');
            sessionStorage.removeItem('token_type');
        } else {
            sessionStorage.setItem('access_token', token);
            sessionStorage.setItem('token_type', tokenType);
            // Remove from local storage
            localStorage.removeItem('access_token');
            localStorage.removeItem('token_type');
        }
    }

    clearToken() {
        this.token = null;
        this.user = null;
        this.tokenType = null;
        
        // Clear from both storages
        localStorage.removeItem('access_token');
        localStorage.removeItem('token_type');
        sessionStorage.removeItem('access_token');
        sessionStorage.removeItem('token_type');
        
        // Clear refresh timer
        if (this.refreshTimer) {
            clearTimeout(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    isAuthenticated() {
        return !!this.token;
    }

    getAuthHeaders() {
        if (!this.token) {
            return {};
        }
        
        return {
            'Authorization': `${this.tokenType} ${this.token}`
        };
    }

    async login(username, password, remember = false) {
        try {
            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);
            
            const response = await fetch(`${CONFIG.API_BASE_URL}/auth/login`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Login failed');
            }
            
            const data = await response.json();
            
            // Save token
            this.saveToken(data.access_token, data.token_type, remember);
            
            // Load user profile
            await this.loadUserProfile();
            
            // Set up token refresh
            this.setupTokenRefresh();
            
            return data;
            
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    }

    async logout() {
        try {
            if (this.token) {
                // Call logout endpoint
                await fetch(`${CONFIG.API_BASE_URL}/auth/logout`, {
                    method: 'POST',
                    headers: this.getAuthHeaders()
                });
            }
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            // Clear token regardless of API call success
            this.clearToken();
        }
    }

    async loadUserProfile() {
        try {
            if (!this.token) {
                throw new Error('No authentication token');
            }
            
            const response = await fetch(`${CONFIG.API_BASE_URL}/auth/profile`, {
                headers: this.getAuthHeaders()
            });
            
            if (!response.ok) {
                throw new Error('Failed to load user profile');
            }
            
            this.user = await response.json();
            return this.user;
            
        } catch (error) {
            console.error('Error loading user profile:', error);
            // If profile loading fails, token might be invalid
            this.clearToken();
            throw error;
        }
    }

    setupTokenRefresh() {
        // Parse token to get expiration
        try {
            const tokenPayload = JSON.parse(atob(this.token.split('.')[1]));
            const expirationTime = tokenPayload.exp * 1000; // Convert to milliseconds
            const currentTime = Date.now();
            const timeUntilExpiry = expirationTime - currentTime;
            
            // Refresh token 5 minutes before expiry
            const refreshTime = Math.max(timeUntilExpiry - (5 * 60 * 1000), 60000); // At least 1 minute
            
            if (refreshTime > 0) {
                this.refreshTimer = setTimeout(() => {
                    this.refreshToken();
                }, refreshTime);
            }
        } catch (error) {
            console.error('Error setting up token refresh:', error);
        }
    }

    async refreshToken() {
        // Note: This would require implementing refresh token functionality
        // For now, we'll just redirect to login when token expires
        console.warn('Token expired, redirecting to login');
        this.clearToken();
        window.location.href = '/login';
    }

    setupApiInterceptor() {
        // Override the global apiRequest function to include auth headers
        const originalApiRequest = window.apiRequest;
        
        window.apiRequest = async (endpoint, options = {}, componentId = null, retries = 2) => {
            // Add auth headers if authenticated
            if (this.isAuthenticated()) {
                options.headers = {
                    ...options.headers,
                    ...this.getAuthHeaders()
                };
            }
            
            try {
                return await originalApiRequest(endpoint, options, componentId, retries);
            } catch (error) {
                // If we get a 401 error, token might be expired
                if (error.message.includes('401') || error.message.includes('Unauthorized')) {
                    this.clearToken();
                    // Redirect to login if not already there
                    if (!window.location.pathname.includes('/login')) {
                        window.location.href = '/login';
                    }
                }
                throw error;
            }
        };
    }

    requireAuth() {
        if (!this.isAuthenticated()) {
            window.location.href = '/login';
            return false;
        }
        return true;
    }

    hasRole(role) {
        return this.user && this.user.role === role;
    }

    isAdmin() {
        return this.user && (this.user.is_admin || this.user.role === 'admin');
    }
}

// Create global auth manager instance
const authManager = new AuthManager();

// Export for use in other scripts
window.authManager = authManager;

// Utility functions for backward compatibility
function isAuthenticated() {
    return authManager.isAuthenticated();
}

function requireAuth() {
    return authManager.requireAuth();
}

function getAuthHeaders() {
    return authManager.getAuthHeaders();
}

function logout() {
    return authManager.logout();
}

// Auto-redirect to login if not authenticated (except on login page)
document.addEventListener('DOMContentLoaded', function() {
    // Skip auth check on login page
    if (window.location.pathname.includes('/login')) {
        return;
    }
    
    // Check if authentication is required for this page
    const protectedPaths = ['/app'];
    const currentPath = window.location.pathname;
    
    const isProtectedPath = protectedPaths.some(path => currentPath.startsWith(path));
    
    if (isProtectedPath && !authManager.isAuthenticated()) {
        window.location.href = '/login';
    }
});

// Add user info to UI if authenticated
document.addEventListener('DOMContentLoaded', async function() {
    if (authManager.isAuthenticated() && !window.location.pathname.includes('/login')) {
        try {
            await authManager.loadUserProfile();
            updateUserUI();
        } catch (error) {
            console.error('Failed to load user profile:', error);
        }
    }
});

function updateUserUI() {
    if (!authManager.user) return;
    
    // Update user dropdown in navigation
    const userDropdown = document.querySelector('.user-dropdown');
    if (userDropdown) {
        userDropdown.innerHTML = `
            <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown">
                <i class="bi bi-person-circle me-2"></i>
                ${authManager.user.full_name || authManager.user.username}
            </a>
            <ul class="dropdown-menu">
                <li><a class="dropdown-item" href="#" onclick="showUserProfile()">
                    <i class="bi bi-person me-2"></i>Profile
                </a></li>
                <li><a class="dropdown-item" href="#" onclick="showSettings()">
                    <i class="bi bi-gear me-2"></i>Settings
                </a></li>
                <li><hr class="dropdown-divider"></li>
                <li><a class="dropdown-item" href="#" onclick="logout()">
                    <i class="bi bi-box-arrow-right me-2"></i>Logout
                </a></li>
            </ul>
        `;
    }
}

function showUserProfile() {
    // Implementation for showing user profile modal
    console.log('Show user profile');
}

function showSettings() {
    // Implementation for showing settings modal
    console.log('Show settings');
}
