// Dashboard-specific JavaScript functionality

let dashboardCharts = {};
let dashboardData = {
    jobs: [],
    agents: [],
    metrics: null
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    loadDashboardData();
});

function initializeDashboard() {
    console.log('Initializing dashboard...');
    
    // Initialize charts
    initializeCharts();
    
    // Set up real-time updates
    setupRealTimeUpdates();
}

function setupEventListeners() {
    // Listen for WebSocket updates
    document.addEventListener('jobUpdate', handleJobUpdate);
    document.addEventListener('agentUpdate', handleAgentUpdate);
    document.addEventListener('systemMetrics', handleSystemMetrics);
}

function setupRealTimeUpdates() {
    // Subscribe to real-time updates via WebSocket
    const wsManager = getWebSocketManager();
    if (wsManager) {
        wsManager.subscribe('dashboard_updates', handleDashboardUpdate);
    }
}

async function loadDashboardData() {
    try {
        // Load all dashboard data in parallel with component-specific loading
        const [jobsData, agentsData, metricsData, jobStatsData] = await Promise.allSettled([
            apiRequest('/jobs?limit=10&sort=created_at&order=desc', {}, 'recent-jobs-table'),
            apiRequest('/agents', {}, 'agent-status-list'),
            apiRequest('/monitoring/system', {}, 'system-performance'),
            apiRequest('/jobs/statistics', {}, 'job-statistics')
        ]);

        // Process results with fallbacks for failed requests
        dashboardData.jobs = jobsData.status === 'fulfilled' ? (jobsData.value.jobs || []) : [];
        dashboardData.agents = agentsData.status === 'fulfilled' ? (agentsData.value.agents || []) : [];
        dashboardData.metrics = metricsData.status === 'fulfilled' ? metricsData.value : getDefaultMetrics();
        dashboardData.jobStats = jobStatsData.status === 'fulfilled' ? jobStatsData.value : getDefaultJobStats();

        // Add animations to dashboard elements
        animateDashboardElements();

        // Update UI with real data
        updateMetricCards();
        updateRecentJobsTable();
        updateAgentStatusList();
        updateSystemPerformance();
        updateCharts();

        // Show success message if all data loaded
        if (jobsData.status === 'fulfilled' && agentsData.status === 'fulfilled') {
            console.log('Dashboard data loaded successfully');
        } else {
            showAlert('Some dashboard data could not be loaded', 'warning', 3000);
        }

    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        showAlert('Failed to load dashboard data. Using fallback data.', 'danger');

        // Use fallback data
        dashboardData.jobs = [];
        dashboardData.agents = [];
        dashboardData.metrics = getDefaultMetrics();
        dashboardData.jobStats = getDefaultJobStats();

        // Still update UI with fallback data
        updateMetricCards();
        updateRecentJobsTable();
        updateAgentStatusList();
        updateSystemPerformance();
        updateCharts();
    } finally {
        hideLoading();
    }
}

// Helper function to get job statistics
async function getJobStatistics() {
    return await apiRequest('/jobs/stats/summary?days=7');
}

// Fallback data functions
function getDefaultMetrics() {
    return {
        cpu_usage: 0,
        memory_usage: 0,
        disk_usage: 0,
        active_connections: 0,
        uptime: 0
    };
}

function getDefaultJobStats() {
    return {
        total_jobs: 0,
        pending_jobs: 0,
        running_jobs: 0,
        completed_jobs: 0,
        failed_jobs: 0,
        success_rate: 0,
        jobs_by_type: {},
        jobs_by_priority: {}
    };
}

// Animation function for dashboard elements
function animateDashboardElements() {
    // Animate metric cards with staggered delays
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach((card, index) => {
        card.classList.add('fade-in-up', `stagger-${index + 1}`);
    });

    // Animate charts
    const charts = document.querySelectorAll('.chart-container');
    charts.forEach((chart, index) => {
        chart.classList.add('scale-in');
        chart.style.animationDelay = `${(index + 1) * 0.2}s`;
    });

    // Animate tables
    const tables = document.querySelectorAll('.table-responsive');
    tables.forEach((table, index) => {
        table.classList.add('slide-in');
        table.style.animationDelay = `${(index + 1) * 0.3}s`;
    });
}

function updateMetricCards() {
    const jobs = dashboardData.jobs;
    const jobStats = dashboardData.jobStats;
    const agents = dashboardData.agents;

    // Use real job statistics when available
    const totalJobs = jobStats.total_jobs || jobs.length;
    const activeJobs = jobStats.running_jobs || jobs.filter(job => job.status === 'running').length;
    const successRate = jobStats.success_rate || (totalJobs > 0 ? (jobStats.completed_jobs / totalJobs * 100) : 0);
    const totalRecords = calculateTotalRecords();

    // Update metric values with animations
    animateCounterUpdate('total-jobs', totalJobs);
    animateCounterUpdate('active-jobs', activeJobs);
    animateCounterUpdate('success-rate', Math.round(successRate), '%');
    animateCounterUpdate('data-collected', totalRecords);

    // Calculate real change indicators
    const todayJobs = jobs.filter(job => {
        const jobDate = new Date(job.created_at);
        const today = new Date();
        return jobDate.toDateString() === today.toDateString();
    }).length;

    const activeAgents = agents.filter(agent => agent.status === 'active').length;

    // Update change indicators with real data
    updateChangeIndicator('jobs-change', `+${todayJobs} today`, todayJobs > 0 ? 'positive' : 'neutral');
    updateChangeIndicator('active-change', `${activeAgents} agents active`, activeAgents > 0 ? 'positive' : 'neutral');
    updateChangeIndicator('success-change', 'Last 7 days', successRate > 80 ? 'positive' : successRate > 60 ? 'neutral' : 'negative');
    updateChangeIndicator('data-change', 'All time', totalRecords > 0 ? 'positive' : 'neutral');
}

// Enhanced counter animation function
function animateCounterUpdate(elementId, targetValue, suffix = '') {
    const element = document.getElementById(elementId);
    if (!element) return;

    const currentValue = parseInt(element.textContent.replace(/[^\d]/g, '')) || 0;
    const increment = Math.ceil((targetValue - currentValue) / 20);
    let current = currentValue;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= targetValue) || (increment < 0 && current <= targetValue)) {
            current = targetValue;
            clearInterval(timer);
        }

        element.textContent = current.toLocaleString() + suffix;

        // Add a subtle scale animation
        element.style.transform = 'scale(1.05)';
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 100);
    }, 50);
}

function updateChangeIndicator(elementId, text, type) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = text;
        element.className = `metric-change ${type}`;
    }
}

function calculateTotalRecords() {
    // Calculate total records from job results
    let total = 0;
    dashboardData.jobs.forEach(job => {
        if (job.result && job.result.records_count) {
            total += job.result.records_count;
        }
    });
    return total.toLocaleString();
}

function updateRecentJobsTable() {
    const jobs = dashboardData.jobs.slice(0, 5); // Show only 5 most recent
    
    const columns = [
        {
            field: 'name',
            title: 'Job Name',
            render: (value, row) => `
                <div>
                    <strong>${value}</strong>
                    <br>
                    <small class="text-muted">${row.job_type}</small>
                </div>
            `
        },
        {
            field: 'status',
            title: 'Status',
            render: (value) => `<span class="job-status ${value}">${value}</span>`
        },
        {
            field: 'progress',
            title: 'Progress',
            render: (value) => `
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar" style="width: ${value}%">${value}%</div>
                </div>
            `
        },
        {
            field: 'created_at',
            title: 'Created',
            render: (value) => formatDateTime(value)
        }
    ];
    
    createDataTable('recent-jobs-table', jobs, columns);
}

function updateAgentStatusList() {
    const agents = dashboardData.agents;
    const container = document.getElementById('agent-status-list');
    
    if (!container) return;
    
    if (agents.length === 0) {
        container.innerHTML = '<p class="text-muted text-center">No agents available</p>';
        return;
    }
    
    let html = '';
    agents.forEach(agent => {
        const statusClass = agent.status === 'active' ? 'success' : 
                           agent.status === 'idle' ? 'warning' : 'danger';
        
        html += `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <div>
                    <h6 class="mb-1">${agent.agent_type}</h6>
                    <small class="text-muted">${agent.agent_id}</small>
                </div>
                <div class="text-end">
                    <span class="badge bg-${statusClass}">${agent.status}</span>
                    <br>
                    <small class="text-muted">${agent.active_tasks} tasks</small>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function updateSystemPerformance() {
    const metrics = dashboardData.metrics;
    if (!metrics) return;
    
    // Update CPU usage
    updateProgressBar('cpu-usage', 'cpu-progress', metrics.cpu_usage || 0);
    
    // Update Memory usage
    updateProgressBar('memory-usage', 'memory-progress', metrics.memory_usage || 0);
    
    // Update Disk usage
    updateProgressBar('disk-usage', 'disk-progress', metrics.disk_usage || 0);
}

function updateProgressBar(textId, progressId, value) {
    const textElement = document.getElementById(textId);
    const progressElement = document.getElementById(progressId);
    
    if (textElement) {
        textElement.textContent = formatPercentage(value, 1);
    }
    
    if (progressElement) {
        progressElement.style.width = `${value}%`;
        
        // Update color based on value
        progressElement.className = 'progress-bar';
        if (value > 80) {
            progressElement.classList.add('bg-danger');
        } else if (value > 60) {
            progressElement.classList.add('bg-warning');
        } else {
            progressElement.classList.add('bg-success');
        }
    }
}

function initializeCharts() {
    // Job Activity Chart (Line Chart) with enhanced styling
    const jobActivityData = {
        labels: generateTimeLabels(24), // Last 24 hours
        datasets: [{
            label: 'Jobs Created',
            data: generateJobActivityFromRealData(),
            borderColor: CONFIG.CHART_COLORS.primary,
            backgroundColor: createGradient('jobActivityChart', CONFIG.CHART_COLORS.primary),
            tension: 0.4,
            fill: true,
            pointBackgroundColor: CONFIG.CHART_COLORS.primary,
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6
        }]
    };

    dashboardCharts.jobActivity = createChart('jobActivityChart', 'line', jobActivityData, {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    usePointStyle: true,
                    padding: 20
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                titleColor: '#333',
                bodyColor: '#666',
                borderColor: CONFIG.CHART_COLORS.primary,
                borderWidth: 1
            }
        },
        scales: {
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    color: '#666'
                }
            },
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                },
                ticks: {
                    stepSize: 1,
                    color: '#666'
                }
            }
        },
        interaction: {
            intersect: false,
            mode: 'index'
        }
    });

    // Job Status Chart (Doughnut Chart) with enhanced styling
    const jobStatusData = {
        labels: ['Completed', 'Running', 'Failed', 'Pending'],
        datasets: [{
            data: [0, 0, 0, 0], // Will be updated with real data
            backgroundColor: [
                CONFIG.CHART_COLORS.success,
                CONFIG.CHART_COLORS.info,
                CONFIG.CHART_COLORS.danger,
                CONFIG.CHART_COLORS.warning
            ],
            borderWidth: 3,
            borderColor: '#ffffff',
            hoverBorderWidth: 5,
            hoverOffset: 10
        }]
    };

    dashboardCharts.jobStatus = createChart('jobStatusChart', 'doughnut', jobStatusData, {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    usePointStyle: true,
                    padding: 15,
                    generateLabels: function(chart) {
                        const data = chart.data;
                        if (data.labels.length && data.datasets.length) {
                            return data.labels.map((label, i) => {
                                const dataset = data.datasets[0];
                                const value = dataset.data[i];
                                return {
                                    text: `${label}: ${value}`,
                                    fillStyle: dataset.backgroundColor[i],
                                    strokeStyle: dataset.borderColor,
                                    lineWidth: dataset.borderWidth,
                                    pointStyle: 'circle',
                                    hidden: false,
                                    index: i
                                };
                            });
                        }
                        return [];
                    }
                }
            },
            tooltip: {
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                titleColor: '#333',
                bodyColor: '#666',
                borderColor: '#ddd',
                borderWidth: 1,
                callbacks: {
                    label: function(context) {
                        const label = context.label || '';
                        const value = context.parsed;
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                        return `${label}: ${value} (${percentage}%)`;
                    }
                }
            }
        },
        cutout: '60%'
    });
}

// Helper function to create gradient backgrounds
function createGradient(canvasId, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return color;

    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color + '40');
    gradient.addColorStop(1, color + '10');
    return gradient;
}

// Generate job activity data from real jobs
function generateJobActivityFromRealData() {
    const jobs = dashboardData.jobs || [];
    const hours = 24;
    const now = new Date();
    const activityData = new Array(hours).fill(0);

    jobs.forEach(job => {
        const jobDate = new Date(job.created_at);
        const hoursDiff = Math.floor((now - jobDate) / (1000 * 60 * 60));

        if (hoursDiff >= 0 && hoursDiff < hours) {
            const index = hours - 1 - hoursDiff;
            activityData[index]++;
        }
    });

    return activityData;
}

function updateCharts() {
    updateJobStatusChart();
    updateJobActivityChart();
}

function updateJobStatusChart() {
    const jobStats = dashboardData.jobStats;
    const jobs = dashboardData.jobs;

    // Use job statistics if available, otherwise calculate from jobs array
    const statusCounts = {
        completed: jobStats.completed_jobs || jobs.filter(j => j.status === 'completed').length,
        running: jobStats.running_jobs || jobs.filter(j => j.status === 'running').length,
        failed: jobStats.failed_jobs || jobs.filter(j => j.status === 'failed').length,
        pending: jobStats.pending_jobs || jobs.filter(j => j.status === 'pending').length
    };

    if (dashboardCharts.jobStatus) {
        const newData = [
            statusCounts.completed,
            statusCounts.running,
            statusCounts.failed,
            statusCounts.pending
        ];

        // Animate the chart update
        dashboardCharts.jobStatus.data.datasets[0].data = newData;
        dashboardCharts.jobStatus.update('active');

        // Add a subtle animation to the chart container
        const chartContainer = document.getElementById('jobStatusChart').parentElement;
        chartContainer.style.transform = 'scale(1.02)';
        setTimeout(() => {
            chartContainer.style.transform = 'scale(1)';
        }, 200);
    }
}

function updateJobActivityChart() {
    if (dashboardCharts.jobActivity) {
        const newActivityData = generateJobActivityFromRealData();
        dashboardCharts.jobActivity.data.datasets[0].data = newActivityData;
        dashboardCharts.jobActivity.update('active');

        // Add animation to the chart container
        const chartContainer = document.getElementById('jobActivityChart').parentElement;
        chartContainer.style.transform = 'scale(1.02)';
        setTimeout(() => {
            chartContainer.style.transform = 'scale(1)';
        }, 200);
    }
}

// Event handlers for real-time updates
function handleJobUpdate(event) {
    const jobData = event.detail;
    console.log('Job update received:', jobData);
    
    // Update job in local data
    const jobIndex = dashboardData.jobs.findIndex(j => j.id === jobData.job_id);
    if (jobIndex !== -1) {
        dashboardData.jobs[jobIndex] = { ...dashboardData.jobs[jobIndex], ...jobData };
    }
    
    // Refresh relevant UI components
    updateMetricCards();
    updateRecentJobsTable();
    updateCharts();
}

function handleAgentUpdate(event) {
    const agentData = event.detail;
    console.log('Agent update received:', agentData);
    
    // Update agent in local data
    const agentIndex = dashboardData.agents.findIndex(a => a.agent_id === agentData.agent_id);
    if (agentIndex !== -1) {
        dashboardData.agents[agentIndex] = { ...dashboardData.agents[agentIndex], ...agentData };
    }
    
    // Refresh agent status list
    updateAgentStatusList();
}

function handleSystemMetrics(event) {
    const metricsData = event.detail;
    console.log('System metrics received:', metricsData);
    
    dashboardData.metrics = metricsData;
    updateSystemPerformance();
}

function handleDashboardUpdate(data) {
    console.log('Dashboard update received:', data);
    // Handle specific dashboard updates
}

// Utility functions
function generateTimeLabels(hours) {
    const labels = [];
    const now = new Date();
    
    for (let i = hours - 1; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * 60 * 60 * 1000));
        labels.push(time.getHours() + ':00');
    }
    
    return labels;
}

function generateMockJobActivity() {
    // Generate mock data for job activity
    return Array.from({ length: 24 }, () => Math.floor(Math.random() * 10));
}

// Public functions
function refreshDashboard() {
    loadDashboardData();
    showAlert('Dashboard refreshed', 'success', 2000);
}
