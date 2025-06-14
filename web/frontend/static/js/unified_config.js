
// Auto-generated configuration for unified web interface
const CONFIG = {
    API_BASE_URL: 'http://localhost:8002/api/v1',
    WS_URL: 'ws://localhost:8002/ws',
    REFRESH_INTERVAL: 5000,
    USE_REAL_DATA: true,
    BACKEND_HOST: 'localhost',
    BACKEND_PORT: 8002,
    CHART_COLORS: {
        primary: '#0d6efd',
        success: '#198754',
        warning: '#ffc107',
        danger: '#dc3545',
        info: '#0dcaf0',
        secondary: '#6c757d'
    }
};

// Override the default CONFIG in common.js
if (typeof window !== 'undefined') {
    window.UNIFIED_CONFIG = CONFIG;
}
