// Phase 4: Enhanced Security & Compliance Web Interface

let securityData = {
    threats: [],
    auditLogs: [],
    complianceStatus: {},
    encryptionStatus: {},
    retentionPolicies: []
};

let securityCharts = {};

// Initialize security page
document.addEventListener('DOMContentLoaded', function() {
    if (window.location.pathname.includes('/security')) {
        initializeSecurityPage();
        setupSecurityEventListeners();
        loadSecurityData();
    }
});

function initializeSecurityPage() {
    console.log('Initializing security management page...');
    initializeSecurityCharts();
    setupRealTimeUpdates();
}

function setupSecurityEventListeners() {
    // Threat detection controls
    const threatScanBtn = document.getElementById('start-threat-scan');
    if (threatScanBtn) {
        threatScanBtn.addEventListener('click', startThreatScan);
    }
    
    // Audit log filters
    const auditFilterForm = document.getElementById('audit-filter-form');
    if (auditFilterForm) {
        auditFilterForm.addEventListener('submit', filterAuditLogs);
    }
    
    // Compliance report generation
    const generateReportBtn = document.getElementById('generate-compliance-report');
    if (generateReportBtn) {
        generateReportBtn.addEventListener('click', generateComplianceReport);
    }
    
    // Data retention controls
    const retentionPolicyForm = document.getElementById('retention-policy-form');
    if (retentionPolicyForm) {
        retentionPolicyForm.addEventListener('submit', createRetentionPolicy);
    }
}

async function loadSecurityData() {
    try {
        showLoading();
        
        // Load all security data in parallel
        await Promise.allSettled([
            loadThreatData(),
            loadAuditLogs(),
            loadComplianceStatus(),
            loadEncryptionStatus(),
            loadRetentionPolicies()
        ]);
        
        updateSecurityDashboard();
        updateThreatTable();
        updateAuditLogTable();
        updateComplianceCards();
        updateRetentionPolicyTable();
        updateSecurityCharts();
        
    } catch (error) {
        console.error('Failed to load security data:', error);
        showAlert('Failed to load security data. Please check if the backend is running.', 'danger', 8000);
    } finally {
        hideLoading();
    }
}

async function loadThreatData() {
    try {
        const response = await apiRequest('/security/threats', {}, 'threats-container');
        securityData.threats = response.threats || [];
    } catch (error) {
        console.error('Failed to load threat data:', error);
        securityData.threats = [];
    }
}

async function loadAuditLogs() {
    try {
        const response = await apiRequest('/security/audit-logs', {}, 'audit-logs-container');
        securityData.auditLogs = response.logs || [];
    } catch (error) {
        console.error('Failed to load audit logs:', error);
        securityData.auditLogs = [];
    }
}

async function loadComplianceStatus() {
    try {
        const response = await apiRequest('/security/compliance-status', {}, 'compliance-container');
        securityData.complianceStatus = response.status || {};
    } catch (error) {
        console.error('Failed to load compliance status:', error);
        securityData.complianceStatus = {};
    }
}

async function loadEncryptionStatus() {
    try {
        const response = await apiRequest('/security/encryption-status', {}, 'encryption-container');
        securityData.encryptionStatus = response.status || {};
    } catch (error) {
        console.error('Failed to load encryption status:', error);
        securityData.encryptionStatus = {};
    }
}

async function loadRetentionPolicies() {
    try {
        const response = await apiRequest('/security/retention-policies', {}, 'retention-container');
        securityData.retentionPolicies = response.policies || [];
    } catch (error) {
        console.error('Failed to load retention policies:', error);
        securityData.retentionPolicies = [];
    }
}

function updateSecurityDashboard() {
    // Update security metrics
    const totalThreats = securityData.threats.length;
    const activeThreats = securityData.threats.filter(t => t.status === 'active').length;
    const complianceScore = securityData.complianceStatus.overall_score || 0;
    const encryptionCoverage = securityData.encryptionStatus.coverage_percentage || 0;
    
    updateElement('total-threats', totalThreats);
    updateElement('active-threats', activeThreats);
    updateElement('compliance-score', `${(complianceScore * 100).toFixed(1)}%`);
    updateElement('encryption-coverage', `${encryptionCoverage.toFixed(1)}%`);
    
    // Update threat level indicator
    const threatLevel = calculateThreatLevel(activeThreats, totalThreats);
    updateThreatLevelIndicator(threatLevel);
}

function calculateThreatLevel(activeThreats, totalThreats) {
    if (activeThreats === 0) return 'low';
    if (activeThreats <= 2) return 'medium';
    if (activeThreats <= 5) return 'high';
    return 'critical';
}

function updateThreatLevelIndicator(level) {
    const indicator = document.getElementById('threat-level-indicator');
    if (!indicator) return;
    
    const colors = {
        low: 'success',
        medium: 'warning', 
        high: 'danger',
        critical: 'dark'
    };
    
    indicator.className = `badge bg-${colors[level]} fs-6`;
    indicator.textContent = level.toUpperCase();
}

function updateThreatTable() {
    const container = document.getElementById('threats-table-container');
    if (!container) return;
    
    if (securityData.threats.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-shield-check display-4 text-success"></i>
                <p class="text-muted mt-2">No active threats detected</p>
            </div>
        `;
        return;
    }
    
    const columns = [
        {
            field: 'threat_id',
            title: 'Threat ID',
            render: (value) => `<code>${value.substring(0, 8)}</code>`
        },
        {
            field: 'type',
            title: 'Type',
            render: (value) => `<span class="badge bg-warning">${value}</span>`
        },
        {
            field: 'severity',
            title: 'Severity',
            render: (value) => {
                const colors = { low: 'info', medium: 'warning', high: 'danger', critical: 'dark' };
                return `<span class="badge bg-${colors[value] || 'secondary'}">${value.toUpperCase()}</span>`;
            }
        },
        {
            field: 'description',
            title: 'Description',
            render: (value) => value.length > 50 ? value.substring(0, 50) + '...' : value
        },
        {
            field: 'detected_at',
            title: 'Detected',
            render: (value) => formatDateTime(value)
        },
        {
            field: 'status',
            title: 'Status',
            render: (value) => {
                const colors = { active: 'danger', investigating: 'warning', resolved: 'success' };
                return `<span class="badge bg-${colors[value] || 'secondary'}">${value.toUpperCase()}</span>`;
            }
        },
        {
            field: 'actions',
            title: 'Actions',
            render: (value, row) => `
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="viewThreatDetails('${row.threat_id}')">
                        <i class="bi bi-eye"></i>
                    </button>
                    <button class="btn btn-outline-success" onclick="resolveThreat('${row.threat_id}')">
                        <i class="bi bi-check"></i>
                    </button>
                </div>
            `
        }
    ];
    
    createDataTable('threats-table-container', securityData.threats, columns);
}

function updateAuditLogTable() {
    const container = document.getElementById('audit-logs-table-container');
    if (!container) return;
    
    if (securityData.auditLogs.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-journal-text display-4 text-muted"></i>
                <p class="text-muted mt-2">No audit logs available</p>
            </div>
        `;
        return;
    }
    
    const columns = [
        {
            field: 'timestamp',
            title: 'Timestamp',
            render: (value) => formatDateTime(value)
        },
        {
            field: 'event_type',
            title: 'Event Type',
            render: (value) => `<span class="badge bg-info">${value}</span>`
        },
        {
            field: 'user_id',
            title: 'User',
            render: (value) => value || 'System'
        },
        {
            field: 'action',
            title: 'Action',
            render: (value) => value
        },
        {
            field: 'resource',
            title: 'Resource',
            render: (value) => `<code>${value}</code>`
        },
        {
            field: 'result',
            title: 'Result',
            render: (value) => {
                const colors = { success: 'success', failure: 'danger', warning: 'warning' };
                return `<span class="badge bg-${colors[value] || 'secondary'}">${value.toUpperCase()}</span>`;
            }
        },
        {
            field: 'risk_score',
            title: 'Risk Score',
            render: (value) => {
                const color = value > 0.7 ? 'danger' : value > 0.4 ? 'warning' : 'success';
                return `<span class="badge bg-${color}">${(value * 100).toFixed(0)}%</span>`;
            }
        }
    ];
    
    createDataTable('audit-logs-table-container', securityData.auditLogs, columns);
}

function updateComplianceCards() {
    const status = securityData.complianceStatus;
    
    // GDPR Compliance
    updateComplianceCard('gdpr-compliance', status.gdpr || {});
    
    // Data Retention
    updateComplianceCard('data-retention', status.data_retention || {});
    
    // Encryption
    updateComplianceCard('encryption-compliance', status.encryption || {});
    
    // Audit Trail
    updateComplianceCard('audit-compliance', status.audit_trail || {});
}

function updateComplianceCard(cardId, data) {
    const card = document.getElementById(cardId);
    if (!card) return;
    
    const score = data.score || 0;
    const status = data.status || 'unknown';
    const issues = data.issues || [];
    
    const scoreElement = card.querySelector('.compliance-score');
    const statusElement = card.querySelector('.compliance-status');
    const issuesElement = card.querySelector('.compliance-issues');
    
    if (scoreElement) {
        scoreElement.textContent = `${(score * 100).toFixed(1)}%`;
        scoreElement.className = `compliance-score badge fs-6 bg-${score > 0.8 ? 'success' : score > 0.6 ? 'warning' : 'danger'}`;
    }
    
    if (statusElement) {
        statusElement.textContent = status.toUpperCase();
        statusElement.className = `compliance-status badge bg-${status === 'compliant' ? 'success' : 'warning'}`;
    }
    
    if (issuesElement) {
        issuesElement.textContent = `${issues.length} issues`;
    }
}

function updateRetentionPolicyTable() {
    const container = document.getElementById('retention-policies-table-container');
    if (!container) return;
    
    if (securityData.retentionPolicies.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-clock-history display-4 text-muted"></i>
                <p class="text-muted mt-2">No retention policies configured</p>
                <button class="btn btn-primary" onclick="showCreateRetentionPolicyModal()">
                    <i class="bi bi-plus"></i> Create Policy
                </button>
            </div>
        `;
        return;
    }
    
    const columns = [
        {
            field: 'name',
            title: 'Policy Name',
            render: (value) => `<strong>${value}</strong>`
        },
        {
            field: 'data_types',
            title: 'Data Types',
            render: (value) => value.join(', ')
        },
        {
            field: 'retention_period_days',
            title: 'Retention Period',
            render: (value) => `${value} days`
        },
        {
            field: 'deletion_method',
            title: 'Deletion Method',
            render: (value) => `<span class="badge bg-secondary">${value}</span>`
        },
        {
            field: 'is_active',
            title: 'Status',
            render: (value) => `<span class="badge bg-${value ? 'success' : 'secondary'}">${value ? 'Active' : 'Inactive'}</span>`
        },
        {
            field: 'actions',
            title: 'Actions',
            render: (value, row) => `
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="editRetentionPolicy('${row.policy_id}')">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button class="btn btn-outline-danger" onclick="deleteRetentionPolicy('${row.policy_id}')">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            `
        }
    ];
    
    createDataTable('retention-policies-table-container', securityData.retentionPolicies, columns);
}

function initializeSecurityCharts() {
    // Threat Trend Chart
    securityCharts.threatTrend = createChart('threatTrendChart', 'line', {
        labels: [],
        datasets: [{
            label: 'Threats Detected',
            data: [],
            borderColor: CONFIG.CHART_COLORS.danger,
            backgroundColor: CONFIG.CHART_COLORS.danger + '20',
            tension: 0.4
        }]
    });
    
    // Compliance Score Chart
    securityCharts.complianceScore = createChart('complianceScoreChart', 'radar', {
        labels: ['GDPR', 'Data Retention', 'Encryption', 'Audit Trail', 'Access Control'],
        datasets: [{
            label: 'Compliance Score',
            data: [0, 0, 0, 0, 0],
            borderColor: CONFIG.CHART_COLORS.primary,
            backgroundColor: CONFIG.CHART_COLORS.primary + '20'
        }]
    });
    
    // Risk Distribution Chart
    securityCharts.riskDistribution = createChart('riskDistributionChart', 'doughnut', {
        labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
        datasets: [{
            data: [0, 0, 0, 0],
            backgroundColor: [
                CONFIG.CHART_COLORS.success,
                CONFIG.CHART_COLORS.warning,
                CONFIG.CHART_COLORS.danger,
                '#dc3545'
            ]
        }]
    });
}

function updateSecurityCharts() {
    updateThreatTrendChart();
    updateComplianceScoreChart();
    updateRiskDistributionChart();
}

function updateThreatTrendChart() {
    if (!securityCharts.threatTrend) return;
    
    // Generate sample trend data (in real implementation, this would come from API)
    const last7Days = [];
    const threatCounts = [];
    
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        last7Days.push(date.toLocaleDateString());
        
        // Count threats for this day
        const dayStart = new Date(date);
        dayStart.setHours(0, 0, 0, 0);
        const dayEnd = new Date(date);
        dayEnd.setHours(23, 59, 59, 999);
        
        const dayThreats = securityData.threats.filter(threat => {
            const threatDate = new Date(threat.detected_at * 1000);
            return threatDate >= dayStart && threatDate <= dayEnd;
        }).length;
        
        threatCounts.push(dayThreats);
    }
    
    securityCharts.threatTrend.data.labels = last7Days;
    securityCharts.threatTrend.data.datasets[0].data = threatCounts;
    securityCharts.threatTrend.update();
}

function updateComplianceScoreChart() {
    if (!securityCharts.complianceScore) return;
    
    const status = securityData.complianceStatus;
    const scores = [
        (status.gdpr?.score || 0) * 100,
        (status.data_retention?.score || 0) * 100,
        (status.encryption?.score || 0) * 100,
        (status.audit_trail?.score || 0) * 100,
        (status.access_control?.score || 0) * 100
    ];
    
    securityCharts.complianceScore.data.datasets[0].data = scores;
    securityCharts.complianceScore.update();
}

function updateRiskDistributionChart() {
    if (!securityCharts.riskDistribution) return;
    
    const riskCounts = { low: 0, medium: 0, high: 0, critical: 0 };
    
    securityData.threats.forEach(threat => {
        if (threat.severity in riskCounts) {
            riskCounts[threat.severity]++;
        }
    });
    
    securityCharts.riskDistribution.data.datasets[0].data = [
        riskCounts.low,
        riskCounts.medium,
        riskCounts.high,
        riskCounts.critical
    ];
    securityCharts.riskDistribution.update();
}

// Security action functions
async function startThreatScan() {
    try {
        showLoading();
        
        const response = await apiRequest('/security/scan/start', {
            method: 'POST'
        });
        
        showAlert('Threat scan started successfully', 'success');
        
        // Refresh threat data after a delay
        setTimeout(() => {
            loadThreatData().then(() => updateThreatTable());
        }, 2000);
        
    } catch (error) {
        console.error('Failed to start threat scan:', error);
        showAlert('Failed to start threat scan', 'danger');
    } finally {
        hideLoading();
    }
}

async function generateComplianceReport() {
    try {
        showLoading();
        
        const response = await apiRequest('/security/compliance/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                start_date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
                end_date: new Date().toISOString()
            })
        });
        
        // Download the report
        const blob = new Blob([JSON.stringify(response, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `compliance_report_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        window.URL.revokeObjectURL(url);
        
        showAlert('Compliance report generated successfully', 'success');
        
    } catch (error) {
        console.error('Failed to generate compliance report:', error);
        showAlert('Failed to generate compliance report', 'danger');
    } finally {
        hideLoading();
    }
}

// Utility functions
function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function setupRealTimeUpdates() {
    // Set up periodic updates for security data
    setInterval(() => {
        loadSecurityData();
    }, 30000); // Update every 30 seconds
}
