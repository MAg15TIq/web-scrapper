// Agents page JavaScript functionality

let agentsData = [];
let selectedAgentId = null;
let performanceCharts = {};

// Initialize agents page
document.addEventListener('DOMContentLoaded', function() {
    initializeAgentsPage();
    setupEventListeners();
    loadAgents();
});

function initializeAgentsPage() {
    console.log('Initializing agents page...');
    setupRealTimeUpdates();
}

function setupEventListeners() {
    // Listen for WebSocket updates
    document.addEventListener('agentUpdate', handleAgentUpdate);
}

function setupRealTimeUpdates() {
    const wsManager = getWebSocketManager();
    if (wsManager) {
        wsManager.subscribe('agent_updates', handleAgentUpdate);
    }
}

async function loadAgents() {
    try {
        // Use component-specific loading for agents grid
        const response = await apiRequest('/agents', {}, 'agents-grid');
        agentsData = response.agents || [];

        updateAgentsGrid();
        updateAgentStatistics();

    } catch (error) {
        console.error('Failed to load agents:', error);
        showAlert('Failed to load agents. Please check if the backend is running.', 'danger', 8000);

        // Show empty state
        const container = document.getElementById('agents-grid');
        if (container) {
            container.innerHTML = `
                <div class="col-12 text-center py-5">
                    <i class="bi bi-exclamation-triangle display-4 text-warning"></i>
                    <h5 class="mt-3 text-muted">Unable to Load Agents</h5>
                    <p class="text-muted">Please check your connection and try again.</p>
                    <button class="btn btn-primary" onclick="loadAgents()">
                        <i class="bi bi-arrow-clockwise"></i> Retry
                    </button>
                </div>
            `;
        }
    }
}

function updateAgentsGrid() {
    const container = document.getElementById('agents-grid');
    if (!container) return;
    
    if (agentsData.length === 0) {
        container.innerHTML = `
            <div class="col-12 text-center py-4">
                <i class="bi bi-robot display-4 text-muted"></i>
                <p class="text-muted mt-2">No agents available</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    agentsData.forEach(agent => {
        const statusClass = getAgentStatusClass(agent.status);
        const statusIcon = getAgentStatusIcon(agent.status);
        
        html += `
            <div class="col-lg-4 col-md-6 mb-4">
                <div class="card h-100 agent-card" data-agent-id="${agent.agent_id}">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="card-title mb-0">
                            <i class="bi ${statusIcon} text-${statusClass}"></i>
                            ${agent.agent_type}
                        </h6>
                        <span class="badge bg-${statusClass}">${agent.status}</span>
                    </div>
                    <div class="card-body">
                        <div class="row text-center mb-3">
                            <div class="col-4">
                                <div class="metric-value">${agent.active_tasks || 0}</div>
                                <div class="metric-label">Active</div>
                            </div>
                            <div class="col-4">
                                <div class="metric-value">${agent.completed_tasks || 0}</div>
                                <div class="metric-label">Completed</div>
                            </div>
                            <div class="col-4">
                                <div class="metric-value">${agent.failed_tasks || 0}</div>
                                <div class="metric-label">Failed</div>
                            </div>
                        </div>
                        
                        <div class="mb-2">
                            <small class="text-muted">Last Activity:</small>
                            <br>
                            <small>${formatDateTime(agent.last_activity)}</small>
                        </div>
                        
                        <div class="mb-3">
                            <small class="text-muted">Capabilities:</small>
                            <br>
                            <div class="capabilities-list">
                                ${(agent.capabilities || []).slice(0, 3).map(cap => 
                                    `<span class="badge bg-light text-dark me-1">${cap}</span>`
                                ).join('')}
                                ${agent.capabilities && agent.capabilities.length > 3 ? 
                                    `<span class="badge bg-secondary">+${agent.capabilities.length - 3}</span>` : ''
                                }
                            </div>
                        </div>
                    </div>
                    <div class="card-footer">
                        <div class="btn-group w-100" role="group">
                            <button class="btn btn-outline-primary btn-sm" onclick="showAgentDetails('${agent.agent_id}')">
                                <i class="bi bi-eye"></i> Details
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" onclick="showPerformanceChart('${agent.agent_id}')">
                                <i class="bi bi-graph-up"></i> Metrics
                            </button>
                            <button class="btn btn-outline-warning btn-sm" onclick="restartAgent('${agent.agent_id}')">
                                <i class="bi bi-bootstrap-reboot"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function updateAgentStatistics() {
    const stats = {
        active: agentsData.filter(a => a.status === 'active').length,
        idle: agentsData.filter(a => a.status === 'idle').length,
        error: agentsData.filter(a => a.status === 'error').length,
        totalTasks: agentsData.reduce((sum, a) => sum + (a.active_tasks || 0), 0)
    };
    
    document.getElementById('active-agents-count').textContent = stats.active;
    document.getElementById('idle-agents-count').textContent = stats.idle;
    document.getElementById('error-agents-count').textContent = stats.error;
    document.getElementById('total-tasks-count').textContent = stats.totalTasks;
}

function getAgentStatusClass(status) {
    const statusMap = {
        'active': 'success',
        'idle': 'warning',
        'error': 'danger',
        'offline': 'secondary'
    };
    return statusMap[status] || 'secondary';
}

function getAgentStatusIcon(status) {
    const iconMap = {
        'active': 'bi-play-circle-fill',
        'idle': 'bi-pause-circle-fill',
        'error': 'bi-exclamation-circle-fill',
        'offline': 'bi-stop-circle-fill'
    };
    return iconMap[status] || 'bi-question-circle-fill';
}

async function showAgentDetails(agentId) {
    try {
        selectedAgentId = agentId;
        const agent = agentsData.find(a => a.agent_id === agentId);
        
        if (!agent) {
            showAlert('Agent not found', 'danger');
            return;
        }
        
        const detailsContainer = document.getElementById('agent-details-content');
        detailsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Basic Information</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Agent ID:</strong></td><td>${agent.agent_id}</td></tr>
                        <tr><td><strong>Type:</strong></td><td>${agent.agent_type}</td></tr>
                        <tr><td><strong>Status:</strong></td><td><span class="badge bg-${getAgentStatusClass(agent.status)}">${agent.status}</span></td></tr>
                        <tr><td><strong>Last Activity:</strong></td><td>${formatDateTime(agent.last_activity)}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Task Statistics</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Active Tasks:</strong></td><td>${agent.active_tasks || 0}</td></tr>
                        <tr><td><strong>Completed Tasks:</strong></td><td>${agent.completed_tasks || 0}</td></tr>
                        <tr><td><strong>Failed Tasks:</strong></td><td>${agent.failed_tasks || 0}</td></tr>
                        <tr><td><strong>Success Rate:</strong></td><td>${calculateSuccessRate(agent)}%</td></tr>
                    </table>
                </div>
            </div>
            
            <div class="mt-3">
                <h6>Capabilities</h6>
                <div class="capabilities-list">
                    ${(agent.capabilities || []).map(cap => 
                        `<span class="badge bg-primary me-1 mb-1">${cap}</span>`
                    ).join('')}
                </div>
            </div>
            
            <div class="mt-3">
                <h6>Configuration</h6>
                <pre class="bg-light p-3 rounded"><code>${JSON.stringify(agent.configuration || {}, null, 2)}</code></pre>
            </div>
        `;
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('agentDetailsModal'));
        modal.show();
        
    } catch (error) {
        console.error('Failed to show agent details:', error);
        showAlert('Failed to load agent details', 'danger');
    }
}

function calculateSuccessRate(agent) {
    const total = (agent.completed_tasks || 0) + (agent.failed_tasks || 0);
    if (total === 0) return 100;
    return Math.round(((agent.completed_tasks || 0) / total) * 100);
}

async function showPerformanceChart(agentId) {
    try {
        selectedAgentId = agentId;
        const agent = agentsData.find(a => a.agent_id === agentId);
        
        if (!agent) {
            showAlert('Agent not found', 'danger');
            return;
        }
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('performanceChartModal'));
        modal.show();
        
        // Initialize charts after modal is shown
        setTimeout(() => {
            initializePerformanceCharts(agent);
        }, 300);
        
    } catch (error) {
        console.error('Failed to show performance chart:', error);
        showAlert('Failed to load performance data', 'danger');
    }
}

function initializePerformanceCharts(agent) {
    // Generate mock performance data
    const timeLabels = generateTimeLabels(24);
    const cpuData = generateMockMetrics(24, 0, 100);
    const memoryData = generateMockMetrics(24, 0, 100);
    const tasksData = generateMockMetrics(24, 0, 10);
    const responseTimeData = generateMockMetrics(24, 100, 2000);
    
    // CPU Usage Chart
    performanceCharts.cpu = createChart('agentCpuChart', 'line', {
        labels: timeLabels,
        datasets: [{
            label: 'CPU Usage (%)',
            data: cpuData,
            borderColor: CONFIG.CHART_COLORS.danger,
            backgroundColor: CONFIG.CHART_COLORS.danger + '20',
            tension: 0.4
        }]
    }, {
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    });
    
    // Memory Usage Chart
    performanceCharts.memory = createChart('agentMemoryChart', 'line', {
        labels: timeLabels,
        datasets: [{
            label: 'Memory Usage (%)',
            data: memoryData,
            borderColor: CONFIG.CHART_COLORS.warning,
            backgroundColor: CONFIG.CHART_COLORS.warning + '20',
            tension: 0.4
        }]
    }, {
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    });
    
    // Tasks Chart
    performanceCharts.tasks = createChart('agentTasksChart', 'bar', {
        labels: timeLabels,
        datasets: [{
            label: 'Tasks Completed',
            data: tasksData,
            backgroundColor: CONFIG.CHART_COLORS.success
        }]
    }, {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    });
    
    // Response Time Chart
    performanceCharts.responseTime = createChart('agentResponseTimeChart', 'line', {
        labels: timeLabels,
        datasets: [{
            label: 'Response Time (ms)',
            data: responseTimeData,
            borderColor: CONFIG.CHART_COLORS.info,
            backgroundColor: CONFIG.CHART_COLORS.info + '20',
            tension: 0.4
        }]
    }, {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    });
}

function configureSelectedAgent() {
    if (!selectedAgentId) return;
    
    const agent = agentsData.find(a => a.agent_id === selectedAgentId);
    if (!agent) return;
    
    // Populate form with current configuration
    document.getElementById('agent-name').value = agent.agent_id;
    document.getElementById('agent-type').value = agent.agent_type;
    
    const config = agent.configuration || {};
    document.getElementById('max-concurrent-tasks').value = config.max_concurrent_tasks || 3;
    document.getElementById('timeout').value = config.timeout || 300;
    document.getElementById('retry-count').value = config.retry_count || 3;
    document.getElementById('auto-restart').checked = config.auto_restart || false;
    document.getElementById('debug-mode').checked = config.debug_mode || false;
    document.getElementById('custom-config').value = JSON.stringify(config.custom || {}, null, 2);
    
    // Hide details modal and show config modal
    const detailsModal = bootstrap.Modal.getInstance(document.getElementById('agentDetailsModal'));
    detailsModal.hide();
    
    const configModal = new bootstrap.Modal(document.getElementById('agentConfigModal'));
    configModal.show();
}

async function saveAgentConfiguration() {
    try {
        if (!selectedAgentId) return;
        
        let customConfig = {};
        try {
            const customConfigText = document.getElementById('custom-config').value.trim();
            if (customConfigText) {
                customConfig = JSON.parse(customConfigText);
            }
        } catch (e) {
            showAlert('Invalid JSON in custom configuration', 'danger');
            return;
        }
        
        const configuration = {
            max_concurrent_tasks: parseInt(document.getElementById('max-concurrent-tasks').value),
            timeout: parseInt(document.getElementById('timeout').value),
            retry_count: parseInt(document.getElementById('retry-count').value),
            auto_restart: document.getElementById('auto-restart').checked,
            debug_mode: document.getElementById('debug-mode').checked,
            custom: customConfig
        };
        
        const response = await apiRequest(`/agents/${selectedAgentId}/configure`, {
            method: 'POST',
            body: JSON.stringify({
                configuration: configuration,
                restart_agent: true
            })
        });
        
        if (response) {
            showAlert('Agent configuration saved successfully', 'success');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('agentConfigModal'));
            modal.hide();
            
            // Reload agents
            loadAgents();
        }
        
    } catch (error) {
        console.error('Failed to save agent configuration:', error);
        showAlert('Failed to save agent configuration', 'danger');
    }
}

// Event handlers and utility functions
function handleAgentUpdate(event) {
    const agentData = event.detail;
    
    // Update agent in local data
    const agentIndex = agentsData.findIndex(a => a.agent_id === agentData.agent_id);
    if (agentIndex !== -1) {
        agentsData[agentIndex] = { ...agentsData[agentIndex], ...agentData };
        updateAgentsGrid();
        updateAgentStatistics();
    }
}

function generateMockMetrics(count, min, max) {
    return Array.from({ length: count }, () => 
        Math.floor(Math.random() * (max - min + 1)) + min
    );
}

function refreshAgents() {
    loadAgents();
    showAlert('Agents refreshed', 'success', 2000);
}

async function restartAgent(agentId) {
    try {
        const response = await apiRequest(`/agents/${agentId}/restart`, {
            method: 'POST'
        });
        
        if (response) {
            showAlert('Agent restart initiated', 'success');
            loadAgents();
        }
        
    } catch (error) {
        console.error('Failed to restart agent:', error);
        showAlert('Failed to restart agent', 'danger');
    }
}

async function restartAllAgents() {
    if (!confirm('Are you sure you want to restart all agents?')) {
        return;
    }
    
    try {
        const response = await apiRequest('/agents/restart-all', {
            method: 'POST'
        });
        
        if (response) {
            showAlert('All agents restart initiated', 'success');
            loadAgents();
        }
        
    } catch (error) {
        console.error('Failed to restart all agents:', error);
        showAlert('Failed to restart all agents', 'danger');
    }
}

function restartSelectedAgent() {
    if (selectedAgentId) {
        restartAgent(selectedAgentId);
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('agentDetailsModal'));
        modal.hide();
    }
}
