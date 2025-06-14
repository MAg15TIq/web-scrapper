// Monitoring page JavaScript functionality

let monitoringCharts = {};
let monitoringData = {
    systemMetrics: null,
    applicationMetrics: null,
    alerts: [],
    logs: []
};
let currentTimeRange = '1h';
let logsUpdateInterval = null;
let logsPaused = false;

// Initialize monitoring page
document.addEventListener('DOMContentLoaded', function() {
    initializeMonitoringPage();
    setupEventListeners();
    loadMonitoringData();
    startRealTimeUpdates();
});

function initializeMonitoringPage() {
    console.log('Initializing monitoring page...');
    initializeCharts();
    setupWebSocketUpdates();
}

function setupEventListeners() {
    // Listen for WebSocket updates
    document.addEventListener('systemMetrics', handleSystemMetricsUpdate);
}

function setupWebSocketUpdates() {
    const wsManager = getWebSocketManager();
    if (wsManager) {
        wsManager.subscribe('system_metrics', handleSystemMetricsUpdate);
        wsManager.subscribe('alerts', handleAlertsUpdate);
        wsManager.subscribe('logs', handleLogsUpdate);
    }
}

async function loadMonitoringData() {
    try {
        // Load all monitoring data in parallel with component-specific loading
        const [systemMetrics, applicationMetrics] = await Promise.allSettled([
            apiRequest('/monitoring/system', {}, 'system-metrics-cards'),
            apiRequest('/monitoring/application', {}, 'application-metrics-cards')
        ]);

        // Handle successful responses
        if (systemMetrics.status === 'fulfilled') {
            monitoringData.systemMetrics = systemMetrics.value;
        } else {
            console.error('Failed to load system metrics:', systemMetrics.reason);
        }

        if (applicationMetrics.status === 'fulfilled') {
            monitoringData.applicationMetrics = applicationMetrics.value;
        } else {
            console.error('Failed to load application metrics:', applicationMetrics.reason);
        }

        updateMetricCards();
        updateCharts();
        updateMetricsTable();
        loadAlerts();
        loadLogs();

    } catch (error) {
        console.error('Failed to load monitoring data:', error);
        showAlert('Failed to load monitoring data. Please check if the backend is running.', 'danger', 8000);
    }
}

async function getApplicationMetrics() {
    return await apiRequest('/monitoring/application');
}

function updateMetricCards() {
    const systemMetrics = monitoringData.systemMetrics;
    const appMetrics = monitoringData.applicationMetrics;
    
    if (!systemMetrics) return;
    
    // System uptime
    const uptime = calculateUptime(systemMetrics.uptime || 0);
    document.getElementById('system-uptime').textContent = uptime;
    
    // CPU usage
    const cpuUsage = systemMetrics.cpu_usage || 0;
    document.getElementById('cpu-usage-metric').textContent = formatPercentage(cpuUsage, 1);
    updateMetricChange('cpu-change', cpuUsage, 80, 60);
    
    // Memory usage
    const memoryUsage = systemMetrics.memory_usage || 0;
    document.getElementById('memory-usage-metric').textContent = formatPercentage(memoryUsage, 1);
    updateMetricChange('memory-change', memoryUsage, 85, 70);
    
    // Requests per second
    const requestsPerSec = appMetrics?.active_connections || 0;
    document.getElementById('requests-per-second').textContent = requestsPerSec;
    updateMetricChange('requests-change', requestsPerSec, 100, 50);
}

function updateMetricChange(elementId, value, dangerThreshold, warningThreshold) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    let status = 'Normal';
    let icon = 'bi-check-circle';
    let className = 'metric-change positive';
    
    if (value >= dangerThreshold) {
        status = 'High';
        icon = 'bi-exclamation-triangle';
        className = 'metric-change negative';
    } else if (value >= warningThreshold) {
        status = 'Warning';
        icon = 'bi-exclamation-circle';
        className = 'metric-change';
    }
    
    element.innerHTML = `<i class="bi ${icon}"></i> ${status}`;
    element.className = className;
}

function calculateUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
        return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else {
        return `${minutes}m`;
    }
}

function initializeCharts() {
    // System Performance Chart (Multi-line)
    const timeLabels = generateTimeLabels(24);
    
    monitoringCharts.systemPerformance = createChart('systemPerformanceChart', 'line', {
        labels: timeLabels,
        datasets: [
            {
                label: 'CPU Usage (%)',
                data: generateMockMetrics(24, 10, 80),
                borderColor: CONFIG.CHART_COLORS.danger,
                backgroundColor: CONFIG.CHART_COLORS.danger + '20',
                tension: 0.4
            },
            {
                label: 'Memory Usage (%)',
                data: generateMockMetrics(24, 20, 70),
                borderColor: CONFIG.CHART_COLORS.warning,
                backgroundColor: CONFIG.CHART_COLORS.warning + '20',
                tension: 0.4
            },
            {
                label: 'Disk Usage (%)',
                data: generateMockMetrics(24, 30, 60),
                borderColor: CONFIG.CHART_COLORS.info,
                backgroundColor: CONFIG.CHART_COLORS.info + '20',
                tension: 0.4
            }
        ]
    }, {
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    });
    
    // Resource Usage Chart (Doughnut)
    monitoringCharts.resourceUsage = createChart('resourceUsageChart', 'doughnut', {
        labels: ['Used', 'Available'],
        datasets: [{
            data: [65, 35],
            backgroundColor: [CONFIG.CHART_COLORS.primary, CONFIG.CHART_COLORS.secondary + '40']
        }]
    }, {
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    });
    
    // Network Chart
    monitoringCharts.network = createChart('networkChart', 'line', {
        labels: timeLabels,
        datasets: [
            {
                label: 'Bytes Sent',
                data: generateMockMetrics(24, 1000, 50000),
                borderColor: CONFIG.CHART_COLORS.success,
                backgroundColor: CONFIG.CHART_COLORS.success + '20',
                tension: 0.4
            },
            {
                label: 'Bytes Received',
                data: generateMockMetrics(24, 2000, 80000),
                borderColor: CONFIG.CHART_COLORS.primary,
                backgroundColor: CONFIG.CHART_COLORS.primary + '20',
                tension: 0.4
            }
        ]
    });
    
    // Response Time Chart
    monitoringCharts.responseTime = createChart('responseTimeChart', 'bar', {
        labels: ['P50', 'P75', 'P90', 'P95', 'P99'],
        datasets: [{
            label: 'Response Time (ms)',
            data: [120, 180, 250, 400, 800],
            backgroundColor: [
                CONFIG.CHART_COLORS.success,
                CONFIG.CHART_COLORS.primary,
                CONFIG.CHART_COLORS.warning,
                CONFIG.CHART_COLORS.danger,
                CONFIG.CHART_COLORS.secondary
            ]
        }]
    }, {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    });
}

function updateCharts() {
    // Update charts with real data when available
    if (monitoringData.systemMetrics) {
        updateSystemPerformanceChart();
        updateResourceUsageChart();
    }
}

function updateSystemPerformanceChart() {
    const metrics = monitoringData.systemMetrics;
    if (!metrics || !monitoringCharts.systemPerformance) return;
    
    // In a real implementation, you would update with historical data
    // For now, we'll just update the current values
    const chart = monitoringCharts.systemPerformance;
    const datasets = chart.data.datasets;
    
    // Update last data point with current values
    const lastIndex = datasets[0].data.length - 1;
    datasets[0].data[lastIndex] = metrics.cpu_usage || 0;
    datasets[1].data[lastIndex] = metrics.memory_usage || 0;
    datasets[2].data[lastIndex] = metrics.disk_usage || 0;
    
    chart.update();
}

function updateResourceUsageChart() {
    const metrics = monitoringData.systemMetrics;
    if (!metrics || !monitoringCharts.resourceUsage) return;
    
    const memoryUsage = metrics.memory_usage || 0;
    const chart = monitoringCharts.resourceUsage;
    chart.data.datasets[0].data = [memoryUsage, 100 - memoryUsage];
    chart.update();
}

function updateMetricsTable() {
    const tbody = document.getElementById('metrics-table-body');
    if (!tbody) return;
    
    const metrics = [
        {
            name: 'CPU Usage',
            current: monitoringData.systemMetrics?.cpu_usage || 0,
            unit: '%',
            threshold: 80
        },
        {
            name: 'Memory Usage',
            current: monitoringData.systemMetrics?.memory_usage || 0,
            unit: '%',
            threshold: 85
        },
        {
            name: 'Disk Usage',
            current: monitoringData.systemMetrics?.disk_usage || 0,
            unit: '%',
            threshold: 90
        },
        {
            name: 'Active Connections',
            current: monitoringData.applicationMetrics?.active_connections || 0,
            unit: '',
            threshold: 100
        },
        {
            name: 'Response Time (avg)',
            current: monitoringData.applicationMetrics?.response_times?.avg || 0,
            unit: 'ms',
            threshold: 1000
        }
    ];
    
    let html = '';
    metrics.forEach(metric => {
        const status = metric.current > metric.threshold ? 'danger' : 
                      metric.current > metric.threshold * 0.8 ? 'warning' : 'success';
        const trend = Math.random() > 0.5 ? 'up' : 'down';
        const trendClass = trend === 'up' ? 'text-danger' : 'text-success';
        
        html += `
            <tr>
                <td><strong>${metric.name}</strong></td>
                <td>${metric.current}${metric.unit}</td>
                <td>${(metric.current * 0.9).toFixed(1)}${metric.unit}</td>
                <td>${(metric.current * 1.2).toFixed(1)}${metric.unit}</td>
                <td><span class="badge bg-${status}">${status.toUpperCase()}</span></td>
                <td><i class="bi bi-arrow-${trend} ${trendClass}"></i></td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

async function loadAlerts() {
    try {
        // Mock alerts data - in real implementation, fetch from API
        const alerts = [
            {
                id: 1,
                level: 'warning',
                message: 'High CPU usage detected',
                timestamp: new Date(Date.now() - 300000),
                resolved: false
            },
            {
                id: 2,
                level: 'info',
                message: 'Agent restarted successfully',
                timestamp: new Date(Date.now() - 600000),
                resolved: true
            }
        ];
        
        monitoringData.alerts = alerts;
        updateAlertsList();
        
    } catch (error) {
        console.error('Failed to load alerts:', error);
    }
}

function updateAlertsList() {
    const container = document.getElementById('alerts-list');
    const countElement = document.getElementById('alert-count');
    
    if (!container) return;
    
    const activeAlerts = monitoringData.alerts.filter(a => !a.resolved);
    countElement.textContent = activeAlerts.length;
    
    if (monitoringData.alerts.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-check-circle text-success display-6"></i>
                <p class="text-muted mt-2">No alerts</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    monitoringData.alerts.slice(0, 10).forEach(alert => {
        const levelClass = alert.level === 'danger' ? 'danger' :
                          alert.level === 'warning' ? 'warning' : 'info';
        const icon = alert.level === 'danger' ? 'exclamation-triangle' :
                    alert.level === 'warning' ? 'exclamation-circle' : 'info-circle';
        
        html += `
            <div class="alert alert-${levelClass} alert-sm mb-2 ${alert.resolved ? 'opacity-50' : ''}">
                <div class="d-flex align-items-center">
                    <i class="bi bi-${icon} me-2"></i>
                    <div class="flex-grow-1">
                        <div class="fw-bold">${alert.message}</div>
                        <small class="text-muted">${formatDateTime(alert.timestamp)}</small>
                    </div>
                    ${alert.resolved ? '<span class="badge bg-success">Resolved</span>' : ''}
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

async function loadLogs() {
    try {
        // Mock logs data - in real implementation, fetch from API or WebSocket
        const logs = [
            { timestamp: new Date(), level: 'INFO', message: 'System monitoring started' },
            { timestamp: new Date(Date.now() - 30000), level: 'DEBUG', message: 'Agent heartbeat received' },
            { timestamp: new Date(Date.now() - 60000), level: 'WARN', message: 'High memory usage detected' }
        ];
        
        monitoringData.logs = logs;
        updateLogsContainer();
        
        // Start periodic log updates
        startLogsUpdates();
        
    } catch (error) {
        console.error('Failed to load logs:', error);
    }
}

function updateLogsContainer() {
    const container = document.getElementById('logs-container');
    if (!container || logsPaused) return;
    
    let html = '';
    monitoringData.logs.slice(-50).forEach(log => {
        const levelClass = log.level === 'ERROR' ? 'text-danger' :
                          log.level === 'WARN' ? 'text-warning' :
                          log.level === 'DEBUG' ? 'text-muted' : 'text-info';
        
        html += `
            <div class="log-entry mb-1">
                <span class="text-muted">${log.timestamp.toLocaleTimeString()}</span>
                <span class="badge bg-secondary">${log.level}</span>
                <span class="${levelClass}">${log.message}</span>
            </div>
        `;
    });
    
    container.innerHTML = html;
    container.scrollTop = container.scrollHeight;
}

function startLogsUpdates() {
    if (logsUpdateInterval) {
        clearInterval(logsUpdateInterval);
    }
    
    logsUpdateInterval = setInterval(() => {
        if (!logsPaused) {
            // Simulate new log entries
            const newLog = {
                timestamp: new Date(),
                level: ['INFO', 'DEBUG', 'WARN'][Math.floor(Math.random() * 3)],
                message: `System event ${Math.floor(Math.random() * 1000)}`
            };
            
            monitoringData.logs.push(newLog);
            updateLogsContainer();
        }
    }, 5000);
}

function startRealTimeUpdates() {
    // Update monitoring data every 30 seconds
    setInterval(() => {
        if (!logsPaused) {
            loadMonitoringData();
        }
    }, 30000);
}

// Event handlers
function handleSystemMetricsUpdate(event) {
    const metricsData = event.detail;
    monitoringData.systemMetrics = metricsData;
    updateMetricCards();
    updateCharts();
    updateMetricsTable();
}

function handleAlertsUpdate(event) {
    const alertData = event.detail;
    monitoringData.alerts.unshift(alertData);
    updateAlertsList();
}

function handleLogsUpdate(event) {
    const logData = event.detail;
    monitoringData.logs.push(logData);
    updateLogsContainer();
}

// Utility functions
function generateMockMetrics(count, min, max) {
    return Array.from({ length: count }, () => 
        Math.floor(Math.random() * (max - min + 1)) + min
    );
}

function generateTimeLabels(hours) {
    const labels = [];
    const now = new Date();
    
    for (let i = hours - 1; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * 60 * 60 * 1000));
        labels.push(time.getHours() + ':00');
    }
    
    return labels;
}

// Public functions
function refreshMonitoring() {
    loadMonitoringData();
    showAlert('Monitoring data refreshed', 'success', 2000);
}

function changeTimeRange(range) {
    currentTimeRange = range;
    loadMonitoringData();
    showAlert(`Time range changed to ${range}`, 'info', 2000);
}

function pauseLogs() {
    logsPaused = !logsPaused;
    const btn = document.getElementById('pause-logs-btn');
    
    if (logsPaused) {
        btn.innerHTML = '<i class="bi bi-play"></i>';
        btn.classList.add('btn-warning');
    } else {
        btn.innerHTML = '<i class="bi bi-pause"></i>';
        btn.classList.remove('btn-warning');
    }
}

function clearLogs() {
    monitoringData.logs = [];
    updateLogsContainer();
    showAlert('Logs cleared', 'info', 2000);
}

function exportMetrics() {
    // Implementation for exporting metrics
    showAlert('Export functionality coming soon', 'info');
}
